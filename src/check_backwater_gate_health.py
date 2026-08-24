"""
check_backwater_gate_health.py — BackwaterEdge analog of the SoilGate
threshold sweep + behavior checks, adapted to a DIFFERENT failure surface.

STGNNBackwaterEdge's gate:
    gate = sigmoid(gate_sharpness * (H_bridge - bw_gate_reference))

bw_gate_reference is a registered BUFFER (each bridge's p90_mAOD
threshold, fixed at construction from precompute_backwater_edges.py) --
NOT learnable, so it cannot drift out of its physically valid range the
way STGNNSoilGate's sat_threshold did. Only gate_sharpness (a single
scalar shared across all backwater edges) is trainable. Two distinct
failure modes are possible here, neither of which is "threshold escapes
range":

  1. gate_sharpness -> 0 (either sign): sigmoid(0 * anything) = 0.5
     regardless of input -- an uninformative, permanently half-open gate
     that carries zero discriminative signal about blockage risk.
  2. gate_sharpness < 0: SIGN INVERSION. The gate would open as H_bridge
     falls BELOW bw_gate_reference and close as it rises above --
     physically backwards (should open as bridge stage approaches its
     own capacity threshold, not recede from it).

Two parts:
  A. Sweep gate_sharpness across all seeds/horizons (same pattern as
     check_soilgate_threshold_sweep.py).
  B. For one checkpoint, evaluate the gate's actual trajectory against
     the real S6_ChannelBlockage injection at the actual blockage node,
     confirming it opens as the injected backwater rise crosses each
     bridge's own bw_gate_reference (same pattern as
     check_soilgate_s4_behavior.py, adapted to S6's meta fields).

Usage:
    python check_backwater_gate_health.py --sweep
    python check_backwater_gate_health.py --checkpoint checkpoints/st_gnn_backwater_edge/42/4/best_model.pt
"""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

SEEDS    = [42, 123, 456]
HORIZONS = [4, 12, 16, 24, 48]
HZ_LABEL = {4: "1hr", 12: "3hr", 16: "4hr", 24: "6hr", 48: "12hr"}

# Known-good reference values, verified earlier this session against
# precompute_backwater_edges.py's real output -- used as a sanity check
# that bw_gate_reference (the buffer) hasn't silently drifted between
# graph regeneration and this checkpoint's training run.
EXPECTED_BW_REFERENCE = {
    "Bawnafinny Bridge": 24.974, "Ovens Bridge": 21.628, "Macroom Town Bridge": 65.676,
}


def load_gate_state(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if "gate_sharpness" not in sd or "bw_gate_reference" not in sd:
        return None
    return {
        "gate_sharpness": sd["gate_sharpness"].item(),
        "bw_gate_reference": sd["bw_gate_reference"],
        "bw_gate_node": sd.get("bw_gate_node"),
        "epoch": ckpt.get("epoch"),
        "hparams": ckpt.get("hparams", {}),
    }


def classify_sharpness(s: float) -> str:
    if s < 0:
        return "SIGN-INVERTED (gate responds backwards to physical driver)"
    if abs(s) < 0.5:
        return "COLLAPSED TOWARD UNINFORMATIVE (gate ~0.5 regardless of input)"
    return "plausible, non-degenerate"


def sweep(ckpt_root: Path):
    print(f"{'Seed':<8}{'Horizon':<10}{'gate_sharpness':<18}{'epoch':<8}{'status'}")
    print("=" * 90)
    n_checked = n_bad = 0
    rows = []
    for seed in SEEDS:
        for hz in HORIZONS:
            path = ckpt_root / str(seed) / str(hz) / "best_model.pt"
            if not path.exists():
                print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{'--':<18}{'--':<8}checkpoint not found")
                continue
            state = load_gate_state(path)
            if state is None:
                print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{'--':<18}{'--':<8}"
                      f"gate_sharpness/bw_gate_reference not found -- wrong checkpoint type?")
                continue
            n_checked += 1
            s = state["gate_sharpness"]
            epoch = state.get("epoch")
            status = classify_sharpness(s)
            if not status.startswith("plausible"):
                n_bad += 1
            epoch_str = str(epoch) if epoch is not None else "?"
            print(f"{seed:<8}{HZ_LABEL.get(hz,hz):<10}{s:<18.4f}{epoch_str:<8}{status}")
            if epoch is not None:
                rows.append({"seed": seed, "horizon": hz, "gate_sharpness": s, "epoch": epoch})
    print("=" * 90)
    if n_checked:
        print(f"\n{n_bad}/{n_checked} checkpoints show a degenerate gate_sharpness")

    if len(rows) >= 4:
        import numpy as np
        sharpness_vals = np.array([r["gate_sharpness"] for r in rows])
        epoch_vals = np.array([r["epoch"] for r in rows])
        corr = np.corrcoef(sharpness_vals, epoch_vals)[0, 1]
        print(f"\nCorrelation between gate_sharpness and epoch count "
              f"(pooled across all seeds/horizons, n={len(rows)}): r={corr:.3f}")
        print("A strong positive correlation would suggest gate_sharpness simply "
              "tracks training duration (early-stopped sooner at longer horizons -> "
              "less time for sharpness to grow from its init value) rather than "
              "reflecting a genuine, horizon-specific optimum -- worth checking "
              "before attributing the pattern to anything more interesting.")


def check_real_injection(ckpt_path: Path,
                         scen_dir: Path = Path("dataset/scenarios/S6_ChannelBlockage")):
    state = load_gate_state(ckpt_path)
    if state is None:
        print("gate_sharpness/bw_gate_reference not found in checkpoint.")
        return

    s = state["gate_sharpness"]
    bw_ref = state["bw_gate_reference"]
    bw_node = state["bw_gate_node"]
    print(f"gate_sharpness: {s:.4f}  -> {classify_sharpness(s)}")
    print(f"bw_gate_reference: {bw_ref.tolist()}")
    print(f"bw_gate_node (bridge node indices): {bw_node.tolist() if bw_node is not None else '?'}")

    # Cross-check against known-good values from earlier this session --
    # catches silent staleness if the graph was regenerated with a
    # different threshold set after this checkpoint was trained.
    if bw_node is not None:
        import pandas as pd
        nd_path = Path("dataset/graph/nodes.csv")
        if nd_path.exists():
            nd = pd.read_csv(nd_path)
            for i, node_idx in enumerate(bw_node.tolist()):
                name = nd.loc[nd.node_idx == node_idx, "name"].values
                name = name[0] if len(name) else f"node_{node_idx}"
                expected = EXPECTED_BW_REFERENCE.get(name)
                actual = bw_ref[i].item()
                if expected is not None and abs(actual - expected) > 0.5:
                    print(f"  [warn] {name}: checkpoint bw_gate_reference={actual:.3f} "
                          f"differs from precompute_backwater_edges.py's known value "
                          f"({expected:.3f}) by more than 0.5m -- graph regenerated "
                          f"since this checkpoint was trained?")

    if not scen_dir.exists():
        print(f"\n{scen_dir} not found -- skipping real-injection trajectory check.")
        return

    meta = json.load(open(scen_dir / "scenario_meta.json"))
    X = np.load(scen_dir / "X_synthetic.npy")
    T_WINDOW = meta.get("T_per_window", 104)
    t_blk = meta["t_blockage_step"]
    bridge_idx = meta["blockage_node"]["idx"]
    bridge_name = meta["blockage_node"]["name"]

    gauge_datum_path = Path("dataset/graph/nodes.csv")
    import pandas as pd
    nd = pd.read_csv(gauge_datum_path)
    datum = float(nd.loc[nd.node_idx == bridge_idx, "gauge_datum_mOSGM15"].values[0])
    p90   = float(nd.loc[nd.node_idx == bridge_idx, "p90_mAOD"].values[0])
    stage_range = p90 - datum

    window0 = X[:T_WINDOW]
    norm_stage = window0[:, bridge_idx, 1]   # F_NORM index
    H_trajectory = datum + norm_stage * stage_range

    # Match this bridge to its row in bw_gate_reference/bw_gate_node
    if bw_node is not None and bridge_idx in bw_node.tolist():
        ref_idx = bw_node.tolist().index(bridge_idx)
        reference = bw_ref[ref_idx].item()
    else:
        print(f"\n[warn] {bridge_name} (node {bridge_idx}) not found in this "
              f"checkpoint's bw_gate_node -- cannot compute a matched gate trajectory.")
        return

    gate_trajectory = torch.sigmoid(
        torch.tensor(s) * (torch.from_numpy(H_trajectory).float() - reference)
    ).numpy()

    print(f"\nReal S6 injection, window 0, {bridge_name}, blockage at step {t_blk}:")
    print(f"  H at blockage step:              {H_trajectory[t_blk]:.3f} m OD  "
          f"(reference={reference:.3f})")
    print(f"  Gate activation at blockage:      {gate_trajectory[t_blk]:.4f}")
    print(f"  Gate activation 10 steps before:  {gate_trajectory[max(0,t_blk-10)]:.4f}")
    print(f"  Gate activation 10 steps after:   {gate_trajectory[min(T_WINDOW-1,t_blk+10)]:.4f}")

    if (gate_trajectory >= 0.5).any():
        open_step = int(np.argmax(gate_trajectory >= 0.5))
        print(f"  -> Gate crosses 0.5 activation at step {open_step} "
              f"({'before' if open_step < t_blk else 'at/after'} the blockage)")
    else:
        print(f"  -> Gate NEVER meaningfully opens (>=0.5) across this window")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--ckpt-root", type=str, default="checkpoints/st_gnn_backwater_edge")
    p.add_argument("--checkpoint", type=str)
    p.add_argument("--scenario-dir", type=str, default="dataset/scenarios/S6_ChannelBlockage")
    args = p.parse_args()

    if args.sweep:
        sweep(Path(args.ckpt_root))
    elif args.checkpoint:
        check_real_injection(Path(args.checkpoint), Path(args.scenario_dir))
    else:
        print("Specify --sweep or --checkpoint <path>")


if __name__ == "__main__":
    main()
