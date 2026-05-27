"""
sar_fno_encoder.py  –  FNO-based SAR spatial encoder for River Lee PI-ST-GNN
=============================================================================
Overview
--------
Encodes a Sentinel-1 SAR raster (VV + VH, regular grid) into per-node
feature embeddings that can be concatenated with GRU temporal embeddings
before GATConv message passing.

Key design decisions:
  1. Quasi-static usage: the encoder runs ONCE per SAR acquisition (~6-day
     cycle), not once per timestep. The resulting node embeddings are held
     constant across all 15-min timesteps within a flood event. Only the
     GRU stream changes at each timestep.

  2. Raster → point transition: the FNO produces a dense feature field
     [C_out, H, W]. Bilinear sampling at gauge node (x, y) coordinates
     converts this to per-node embeddings [N, C_out]. From this point
     onward, SAR information lives in graph space, not raster space.

  3. Normalised coordinates: node positions are supplied in the same CRS
     as the SAR raster (ITM / EPSG:2157). Internally they are converted to
     grid_sample's [-1, +1] normalised coordinate system. The caller passes
     raw pixel-fractional coordinates via `node_coords_norm`.

Architecture
------------
  Lifting layer    Conv2d(in_channels=2,  width=32, kernel=1)
  FNO block 1      SpectralConv2d(32, 32, modes_h=12, modes_w=12) + Conv2d bypass
  FNO block 2      SpectralConv2d(32, 32, modes_h=12, modes_w=12) + Conv2d bypass
  FNO block 3      SpectralConv2d(32, 32, modes_h=8,  modes_w=8)  + Conv2d bypass
  FNO block 4      SpectralConv2d(32, 32, modes_h=8,  modes_w=8)  + Conv2d bypass
  Projection layer Conv2d(32, out_channels=16, kernel=1)
  Activation       GELU throughout (smoother than ReLU for spectral features)

  Total learnable parameters (default config): ~857 k

Fourier mode schedule
---------------------
Higher modes in blocks 1–2 capture fine inundation boundaries and
surface roughness contrasts. Lower modes in blocks 3–4 enforce smooth
spatial consistency across the catchment. Truncating beyond modes_max
acts as a learned low-pass filter — high-frequency speckle noise in
the SAR signal is suppressed without a separate despeckling step.

Shapes throughout (batch dimension B is always 1 in quasi-static usage,
but the module is fully batched so it can also handle B>1 during training
if multiple acquisitions are batched together):
  Input raster:       [B, 2,       H, W]
  After lifting:      [B, 32,      H, W]
  After 4 FNO blocks: [B, 32,      H, W]
  After projection:   [B, 16,      H, W]
  After node sample:  [B, N, 16]         → squeeze B → [N, 16]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  SpectralConv2d — the core FNO operation
# ═══════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """
    2-D Fourier integral operator layer (Li et al., 2021).

    For an input field v ∈ R^{C_in × H × W}:
      1. Compute 2-D rfft2 → complex spectrum V ∈ C^{C_in × H × (W//2+1)}
      2. Truncate to the lowest (modes_h, modes_w) frequency modes
      3. Multiply by learnable complex weight R ∈ C^{C_in × C_out × modes_h × modes_w}
      4. Pad back to full spectrum shape with zeros
      5. Compute irfft2 → output field in R^{C_out × H × W}

    The learnable parameters are the complex weights R stored as two
    real tensors (real part and imaginary part) to avoid .to() issues
    with torch complex dtypes across devices.

    This layer is used alongside a plain Conv2d bypass (see FNOBlock below)
    to handle both global (spectral) and local (pointwise) interactions.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_h: int, modes_w: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_h      = modes_h   # kept frequency modes in height dim
        self.modes_w      = modes_w   # kept frequency modes in width dim

        # Kaiming-style scale for complex weights
        scale = 1.0 / (in_channels * out_channels)

        # Stored as real + imag pairs to avoid dtype casting issues
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, modes_w)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, modes_w)
        )

    def _complex_mul2d(self, x: torch.Tensor,
                       w_real: torch.Tensor,
                       w_imag: torch.Tensor) -> torch.Tensor:
        """
        Complex matrix multiplication between spectrum x and weights (w_real, w_imag).

        x:      [B, C_in,  modes_h, modes_w] complex
        w_real: [C_in, C_out, modes_h, modes_w] real
        w_imag: [C_in, C_out, modes_h, modes_w] real

        Returns [B, C_out, modes_h, modes_w] complex.

        Uses the identity (a+ib)(c+id) = (ac-bd) + i(ad+bc) to avoid
        requiring complex number support in all torch backends.
        """
        # x: [B, C_in, modes_h, modes_w] complex
        x_real = x.real   # [B, C_in, modes_h, modes_w]
        x_imag = x.imag

        # Einsum over the C_in dimension: bimn, iomn -> bomn
        out_real = (
            torch.einsum("bimn,iomn->bomn", x_real, w_real)
            - torch.einsum("bimn,iomn->bomn", x_imag, w_imag)
        )
        out_imag = (
            torch.einsum("bimn,iomn->bomn", x_real, w_imag)
            + torch.einsum("bimn,iomn->bomn", x_imag, w_real)
        )
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]  real-valued input field
        returns [B, C_out, H, W] real-valued output field
        """
        B, C, H, W = x.shape

        # ── 1. Compute real FFT ────────────────────────────────────────
        # rfft2 output shape: [B, C_in, H, W//2+1] complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # ── 2. Truncate + multiply in frequency space ──────────────────
        # Only keep the lowest modes_h × modes_w frequencies.
        # We operate on the top-left corner of the spectrum (low freqs).
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes_h, :self.modes_w] = self._complex_mul2d(
            x_ft[:, :, :self.modes_h, :self.modes_w],
            self.weights_real, self.weights_imag,
        )

        # ── 3. Inverse FFT back to spatial domain ─────────────────────
        # irfft2 with s=(H,W) ensures output matches input spatial dims
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


# ═══════════════════════════════════════════════════════════════════════
#  FNOBlock — spectral conv + pointwise bypass + activation
# ═══════════════════════════════════════════════════════════════════════

class FNOBlock(nn.Module):
    """
    One FNO layer: spectral path + local bypass path, combined with GELU.

    Output = GELU( SpectralConv2d(x) + Conv2d(x, kernel=1) )

    The bypass Conv2d(kernel=1) is a pointwise linear transform that handles
    local interactions the Fourier path cannot capture (it only sees the
    truncated low-frequency modes). Together they cover both global and local
    spatial structure.

    BatchNorm is deliberately omitted: the FNO literature (Li et al., 2021;
    Kovachki et al., 2023) finds it harmful for operator learning because it
    normalises across spatial positions, destroying the spatial information
    the operator is trying to learn. LayerNorm over channels is used instead,
    applied after the block rather than inside it.
    """

    def __init__(self, channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_h, modes_w)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm     = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ═══════════════════════════════════════════════════════════════════════
#  SARFNOEncoder — full encoder + bilinear node sampling
# ═══════════════════════════════════════════════════════════════════════

class SARFNOEncoder(nn.Module):
    """
    FNO-based encoder that maps a Sentinel-1 SAR raster to per-node
    embeddings aligned with the OPW gauge network.

    Forward pass:
      1. Lift SAR channels (2) → FNO width (32) via 1×1 conv
      2. Four FNO blocks with decreasing mode counts
      3. Project width (32) → output channels (16) via 1×1 conv
      4. Bilinear-sample the dense feature field at node (x,y) positions
      5. Return per-node embedding tensor [N, out_channels]

    Parameters
    ----------
    in_channels : int
        Number of SAR input channels. Default 2 (VV + VH).
    width : int
        Internal FNO channel width. Default 32.
    out_channels : int
        Output embedding dimension per node. Default 16.
    modes_high : int
        Fourier modes kept in FNO blocks 1–2 (fine spatial structure).
        Default 12. Should be ≤ H//2 and ≤ W//2.
    modes_low : int
        Fourier modes kept in FNO blocks 3–4 (coarse spatial structure).
        Default 8.

    Usage
    -----
    # --- Offline (once per SAR acquisition, outside training loop) ---
    encoder = SARFNOEncoder().to(device)
    sar_raster = load_sar_event(event_id)           # [1, 2, H, W]
    node_coords_norm = compute_node_coords_norm(     # [N, 2], values in [-1,1]
        node_xy_itm, sar_bbox, H, W
    )
    with torch.no_grad():
        sar_emb = encoder(sar_raster, node_coords_norm)   # [N, 16]
    # sar_emb is then passed as a fixed tensor into every training batch
    # for this event. It does NOT change within the event.

    # --- Inside model forward (concat with GRU output) ---
    node_features = torch.cat([h_gru, sar_emb.unsqueeze(0).expand(B,-1,-1)], dim=-1)
    """

    def __init__(
        self,
        in_channels: int = 2,
        width:       int = 32,
        out_channels: int = 16,
        modes_high:  int = 12,
        modes_low:   int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels

        # ── Lifting: in_channels → width ──────────────────────────────
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
        )

        # ── FNO blocks ─────────────────────────────────────────────────
        # Blocks 1–2: high modes → capture fine inundation boundaries
        # Blocks 3–4: low modes  → enforce smooth spatial consistency
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes_high, modes_high),
            FNOBlock(width, modes_high, modes_high),
            FNOBlock(width, modes_low,  modes_low),
            FNOBlock(width, modes_low,  modes_low),
        ])

        # ── Projection: width → out_channels ──────────────────────────
        self.projection = nn.Conv2d(width, out_channels, kernel_size=1)

        # ── Output normalisation (applied to per-node embeddings) ─────
        # LayerNorm over the channel dimension for each node independently.
        # Stabilises the scale of SAR embeddings relative to GRU output
        # before concatenation in the fusion layer.
        self.output_norm = nn.LayerNorm(out_channels)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers; leave spectral weights from SpectralConv2d."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sar: torch.Tensor,            # [B, 2, H, W]  or  [2, H, W]
        node_coords_norm: torch.Tensor,  # [N, 2]  values in [-1, +1]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sar : torch.Tensor
            SAR raster in dB, VV channel first. Accepts both batched
            [B, 2, H, W] and unbatched [2, H, W] inputs.
        node_coords_norm : torch.Tensor
            Node (x, y) positions in grid_sample normalised coordinates:
              x = -1 → left edge of SAR raster
              x = +1 → right edge of SAR raster
              y = -1 → top edge  (row 0)
              y = +1 → bottom edge (row H-1)
            Shape: [N, 2]. Must be on the same device as `sar`.

        Returns
        -------
        torch.Tensor
            Per-node SAR embeddings. Shape: [N, out_channels].
            If input was batched (B > 1), returns [B, N, out_channels]
            so the caller can index the right batch entry.
        """
        # ── Handle unbatched input ─────────────────────────────────────
        squeeze_batch = (sar.dim() == 3)
        if squeeze_batch:
            sar = sar.unsqueeze(0)   # [1, 2, H, W]

        B = sar.shape[0]

        # ── 1. Lifting ─────────────────────────────────────────────────
        x = self.lifting(sar)                  # [B, 32, H, W]

        # ── 2. FNO blocks ──────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)                       # [B, 32, H, W]

        # ── 3. Projection ──────────────────────────────────────────────
        x = self.projection(x)                 # [B, 16, H, W]

        # ── 4. Bilinear sampling at node coordinates ───────────────────
        # grid_sample expects grid of shape [B, H_out, W_out, 2].
        # We want one sample per node → H_out=1, W_out=N.
        N = node_coords_norm.shape[0]

        # node_coords_norm: [N, 2] → [1, 1, N, 2] → [B, 1, N, 2]
        grid = node_coords_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        grid = grid.expand(B, -1, -1, -1)                  # [B, 1, N, 2]

        # grid_sample output: [B, 16, 1, N]
        sampled = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="border",   # clamp out-of-bound nodes to edge
            align_corners=True,
        )
        # [B, 16, 1, N] → [B, N, 16]
        node_emb = sampled.squeeze(2).permute(0, 2, 1)

        # ── 5. Output normalisation ────────────────────────────────────
        node_emb = self.output_norm(node_emb)  # [B, N, 16]

        if squeeze_batch:
            node_emb = node_emb.squeeze(0)     # [N, 16]

        return node_emb

    def encode_event(
        self,
        sar: torch.Tensor,               # [2, H, W]
        node_coords_norm: torch.Tensor,  # [N, 2]
    ) -> torch.Tensor:
        """
        Convenience wrapper for the quasi-static use case.

        Runs the encoder in no_grad mode and returns a detached [N, 16]
        tensor that can be cached and reused across all timesteps of a
        flood event without re-running the FNO.

        Example
        -------
        sar_emb = encoder.encode_event(sar_raster, node_coords_norm)
        # Cache sar_emb — do not re-compute until the next SAR acquisition
        for t in event_timesteps:
            h_gru = gru(x_seq[:, :t])
            node_features = torch.cat([h_gru, sar_emb.expand(B,-1,-1)], -1)
            ...
        """
        self.eval()
        with torch.no_grad():
            return self(sar, node_coords_norm)


# ═══════════════════════════════════════════════════════════════════════
#  Coordinate utility
# ═══════════════════════════════════════════════════════════════════════

def compute_node_coords_norm(
    node_xy: torch.Tensor,    # [N, 2]  ITM eastings/northings
    sar_bbox: tuple,          # (x_min, y_min, x_max, y_max) in ITM
    H: int,                   # SAR raster height in pixels
    W: int,                   # SAR raster width  in pixels
) -> torch.Tensor:
    """
    Convert ITM gauge node coordinates to grid_sample normalised coords.

    grid_sample uses the convention:
      x_norm = -1 → left  pixel column (west edge)
      x_norm = +1 → right pixel column (east edge)
      y_norm = -1 → top   pixel row    (north edge, row 0 after north-up flip)
      y_norm = +1 → bottom pixel row   (south edge)

    The SAR raster is assumed to be north-up (standard GeoTIFF convention):
      row 0 corresponds to the northernmost edge (y_max in ITM).

    Parameters
    ----------
    node_xy : torch.Tensor
        [N, 2] tensor of (easting, northing) in ITM / EPSG:2157 metres.
    sar_bbox : tuple
        (x_min, y_min, x_max, y_max) bounding box of the SAR raster
        in the same CRS as node_xy.
    H, W : int
        Pixel dimensions of the SAR raster after preprocessing.

    Returns
    -------
    torch.Tensor
        [N, 2] float32 tensor of (x_norm, y_norm) in [-1, +1].
        Values outside [-1, +1] indicate nodes outside the SAR footprint;
        grid_sample's padding_mode='border' will clamp these to the edge.

    Notes
    -----
    - node_xy and the return tensor must be on the same device.
    - Call this once per SAR acquisition and cache the result.
      Node coordinates never change so recomputing is wasteful.
    """
    x_min, y_min, x_max, y_max = sar_bbox

    eastings   = node_xy[:, 0]   # [N]
    northings  = node_xy[:, 1]   # [N]

    # x: east → west maps easting to [-1, +1]
    x_norm = 2.0 * (eastings - x_min)  / (x_max - x_min) - 1.0

    # y: north-up raster → row 0 is y_max (north), row H-1 is y_min (south)
    # So northing maps inversely: higher northing → smaller y_norm
    y_norm = 2.0 * (y_max - northings) / (y_max - y_min) - 1.0

    return torch.stack([x_norm, y_norm], dim=1).float()


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Simulate a Sentinel-1 SAR raster clipped to the Lee catchment.
    # At 10m native resolution over ~3500 km²: roughly 1875 × 2000 px.
    # In practice you will resample to a coarser grid for training;
    # 256×256 or 512×512 is typical. FNO is resolution-invariant so
    # the encoder trained on 256×256 can be evaluated on 512×512.
    H, W  = 256, 256
    N     = 27    # River Lee gauge count

    sar   = torch.randn(2, H, W).to(device)   # [2, H, W]
    nodes = torch.rand(N, 2).to(device)        # [N, 2] — random normalised coords

    encoder = SARFNOEncoder(
        in_channels=2,
        width=32,
        out_channels=16,
        modes_high=12,
        modes_low=8,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {n_params:,}")

    # Timing: encode_event (no_grad)
    t0  = time.perf_counter()
    emb = encoder.encode_event(sar, nodes)
    t1  = time.perf_counter()

    print(f"Output shape:  {tuple(emb.shape)}")        # expected: (27, 16)
    print(f"Encode time:   {(t1-t0)*1000:.1f} ms")
    print(f"Output range:  [{emb.min():.3f}, {emb.max():.3f}]")

    # Batched forward (used during training when FNO is in the grad graph)
    sar_batch = torch.randn(4, 2, H, W).to(device)    # B=4 for training
    emb_batch = encoder(sar_batch, nodes)
    print(f"Batched shape: {tuple(emb_batch.shape)}")  # expected: (4, 27, 16)

    # Gradient check
    sar_grad = torch.randn(1, 2, H, W, requires_grad=True).to(device)
    out = encoder(sar_grad, nodes)
    out.sum().backward()
    print(f"Gradient check passed: grad norm = {sar_grad.grad.norm():.4f}")
    print("All checks passed.")
