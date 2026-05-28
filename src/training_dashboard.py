"""
training_dashboard.py  –  Live training dashboard for PI-ST-GNN with SAR-FNO
=============================================================================
Polls live_metrics.json (written by the training script every 50 batches
and after every epoch) and renders a 6-panel matplotlib figure that
auto-refreshes every 2 seconds.

Run in a SECOND terminal alongside training:
    python src/training_dashboard.py

The dashboard shows:
  Panel 1  Epoch progress bar + current batch within epoch
  Panel 2  Training & validation loss curves over epochs
  Panel 3  Validation NSE vs persistence NSE
  Panel 4  Validation RMSE over epochs
  Panel 5  Learning rate schedule
  Panel 6  SAR integration status (coverage, cache size, event windows)

Usage
-----
  # Start dashboard BEFORE or DURING training
  python src/training_dashboard.py

  # Point to a specific metrics file (e.g. a previous run)
  python src/training_dashboard.py --metrics checkpoints/live_metrics.json

  # Increase/decrease refresh rate (seconds)
  python src/training_dashboard.py --interval 5

  # Save a static snapshot instead of live display
  python src/training_dashboard.py --save figures/training_snapshot.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Default metrics path (must match LIVE_METRICS_PATH in training script) ─
BASE_DIR     = Path(__file__).resolve().parent.parent
DEFAULT_PATH = BASE_DIR / "checkpoints" / "live_metrics.json"

# ── Colour palette ─────────────────────────────────────────────────────────
C_TRAIN   = "#2E86AB"   # steel blue   — training metrics
C_VAL     = "#F46036"   # coral        — validation metrics
C_PERSIST = "#7B7B7B"   # mid-grey     — persistence baseline
C_SAR     = "#44BBA4"   # teal         — SAR-specific metrics
C_WARN    = "#E94F37"   # red-orange   — warnings / LR events
C_BG      = "#0F1117"   # near-black   — figure background
C_PANEL   = "#1A1D27"   # dark panel   — axes background
C_TEXT    = "#E8E8E8"   # light grey   — all text
C_GRID    = "#2A2D3A"   # subtle grid


def load_metrics(path: Path) -> dict | None:
    """Read live_metrics.json. Returns None if file missing or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent dark-theme styling to an axes."""
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def draw_progress_bar(ax, batch_idx, n_batches, epoch, max_epochs,
                      phase, running_loss):
    """Panel 1: epoch/batch progress bar."""
    ax.clear()
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Epoch progress (top bar)
    epoch_frac = min((epoch or 0) / max(max_epochs or 1, 1), 1.0)
    ax.add_patch(plt.Rectangle((0.05, 0.72), 0.90, 0.12,
                                facecolor=C_GRID, zorder=1))
    ax.add_patch(plt.Rectangle((0.05, 0.72), 0.90 * epoch_frac, 0.12,
                                facecolor=C_TRAIN, zorder=2))
    ax.text(0.50, 0.78, f"Epoch {epoch or 0} / {max_epochs or '?'}",
            ha="center", va="center", fontsize=9, color=C_TEXT,
            fontweight="bold", zorder=3)

    # Batch progress (bottom bar)
    batch_frac = 0.0
    if batch_idx is not None and n_batches:
        batch_frac = min((batch_idx + 1) / n_batches, 1.0)
    ax.add_patch(plt.Rectangle((0.05, 0.50), 0.90, 0.10,
                                facecolor=C_GRID, zorder=1))
    ax.add_patch(plt.Rectangle((0.05, 0.50), 0.90 * batch_frac, 0.10,
                                facecolor=C_VAL, alpha=0.8, zorder=2))
    b_label = (f"Batch {(batch_idx or 0) + 1} / {n_batches or '?'}  "
               f"({batch_frac * 100:.1f}%)")
    ax.text(0.50, 0.55, b_label,
            ha="center", va="center", fontsize=8, color=C_TEXT, zorder=3)

    # Status line
    status_color = {"training": C_SAR, "complete": C_TRAIN,
                    None: C_PERSIST}.get(phase, C_TEXT)
    status_label = {"training": "● TRAINING", "complete": "✓ COMPLETE",
                    None: "◌ WAITING"}.get(phase, phase or "WAITING")
    ax.text(0.50, 0.34, status_label,
            ha="center", va="center", fontsize=10,
            color=status_color, fontweight="bold")

    if running_loss is not None:
        ax.text(0.50, 0.18,
                f"Batch loss: {running_loss:.5f}",
                ha="center", va="center", fontsize=8.5, color=C_TEXT)

    ax.set_title("Training Progress", fontsize=9, fontweight="bold",
                 color=C_TEXT, pad=6)


def draw_loss_curve(ax, history):
    """Panel 2: train/val loss over epochs."""
    ax.clear()
    style_ax(ax, "Loss Curves", "Epoch", "Horizon-weighted MSE")
    if not history:
        ax.text(0.5, 0.5, "Waiting for epoch 1…",
                ha="center", va="center", transform=ax.transAxes,
                color=C_TEXT, fontsize=9)
        return
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    ax.plot(epochs, train_loss, color=C_TRAIN, linewidth=1.8,
            label="Train", marker=".", markersize=4)
    ax.plot(epochs, val_loss,   color=C_VAL,   linewidth=1.8,
            label="Val",   marker=".", markersize=4)
    ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
              edgecolor=C_GRID)
    # Mark best val
    best_e = epochs[int(np.argmin(val_loss))]
    best_v = min(val_loss)
    ax.axvline(best_e, color=C_SAR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(best_e + 0.1, best_v, f"best {best_v:.4f}",
            fontsize=7, color=C_SAR, va="bottom")


def draw_nse_curve(ax, history):
    """Panel 3: NSE vs persistence NSE over epochs."""
    ax.clear()
    style_ax(ax, "Nash-Sutcliffe Efficiency", "Epoch", "NSE")
    if not history:
        ax.text(0.5, 0.5, "Waiting for epoch 1…",
                ha="center", va="center", transform=ax.transAxes,
                color=C_TEXT, fontsize=9)
        return
    epochs     = [h["epoch"]       for h in history]
    val_nse    = [h.get("val_nse",  float("nan")) for h in history]
    # Persist NSE is stable — take last value
    last = history[-1]
    persist_nse = last.get("persist_nse", None)

    ax.plot(epochs, val_nse, color=C_VAL, linewidth=1.8,
            label="Model NSE", marker=".", markersize=4)
    if persist_nse is not None:
        ax.axhline(persist_nse, color=C_PERSIST, linewidth=1.2,
                   linestyle="--", label=f"Persistence {persist_nse:.3f}")
    ax.axhline(1.0, color=C_GRID, linewidth=0.5, linestyle=":")
    valid_nse = [v for v in val_nse if not np.isnan(v)]
    bottom = (min(0.0, min(valid_nse)) if valid_nse else 0.0) - 0.05
    ax.set_ylim(bottom=bottom)
    ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
              edgecolor=C_GRID)


def draw_rmse_curve(ax, history):
    """Panel 4: validation RMSE over epochs."""
    ax.clear()
    style_ax(ax, "Validation RMSE", "Epoch", "RMSE (m)")
    if not history:
        ax.text(0.5, 0.5, "Waiting for epoch 1…",
                ha="center", va="center", transform=ax.transAxes,
                color=C_TEXT, fontsize=9)
        return
    epochs   = [h["epoch"] for h in history]
    val_rmse = [h.get("val_rmse", float("nan")) for h in history]
    ax.plot(epochs, val_rmse, color=C_TRAIN, linewidth=1.8,
            marker=".", markersize=4)
    best_rmse = min(v for v in val_rmse if not np.isnan(v))
    ax.axhline(best_rmse, color=C_SAR, linewidth=0.8,
               linestyle="--", alpha=0.7,
               label=f"Best {best_rmse:.4f} m")
    ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
              edgecolor=C_GRID)


def draw_lr_curve(ax, history):
    """Panel 5: learning rate schedule."""
    ax.clear()
    style_ax(ax, "Learning Rate", "Epoch", "LR")
    if not history:
        ax.text(0.5, 0.5, "Waiting for epoch 1…",
                ha="center", va="center", transform=ax.transAxes,
                color=C_TEXT, fontsize=9)
        return
    epochs = [h["epoch"]  for h in history]
    lrs    = [h.get("lr", float("nan")) for h in history]
    ax.semilogy(epochs, lrs, color=C_VAL, linewidth=1.8,
                marker=".", markersize=4)
    # Mark LR reductions
    for i in range(1, len(lrs)):
        if not (np.isnan(lrs[i]) or np.isnan(lrs[i-1])):
            if lrs[i] < lrs[i-1] * 0.6:
                ax.axvline(epochs[i], color=C_WARN,
                           linewidth=0.8, linestyle=":", alpha=0.8)


def draw_sar_status(ax, metrics):
    """Panel 6: SAR integration status card."""
    ax.clear()
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("SAR-FNO Integration", fontsize=9,
                 fontweight="bold", color=C_TEXT, pad=6)

    use_sar      = metrics.get("use_sar", False)
    n_events     = metrics.get("n_sar_events", 0)
    coverage_pct = metrics.get("sar_coverage_pct", 0.0)

    # SAR active indicator
    sar_color = C_SAR if use_sar else C_PERSIST
    sar_label = "SAR-FNO ACTIVE" if use_sar else "BASELINE MODE"
    ax.text(0.50, 0.88, sar_label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=sar_color)

    # Coverage bar
    ax.add_patch(plt.Rectangle((0.10, 0.70), 0.80, 0.08,
                                facecolor=C_GRID))
    ax.add_patch(plt.Rectangle((0.10, 0.70), 0.80 * coverage_pct / 100, 0.08,
                                facecolor=C_SAR, alpha=0.8))
    ax.text(0.50, 0.74, f"Timestep coverage: {coverage_pct:.1f}%",
            ha="center", va="center", fontsize=8, color=C_TEXT)

    rows = [
        ("SAR events cached",  f"{n_events}"),
        ("Encode resolution",  "256 × 256 px"),
        ("FNO channels",       "32  (4 blocks)"),
        ("SAR emb dim",        "16 per node"),
        ("Fusion",             "Linear(80→64) + LN"),
    ]
    y = 0.57
    for label, value in rows:
        ax.text(0.08, y, label, ha="left",  va="center",
                fontsize=7.5, color=C_PERSIST)
        ax.text(0.92, y, value, ha="right", va="center",
                fontsize=7.5, color=C_TEXT, fontweight="bold")
        y -= 0.09

    # Phase / test metrics if complete
    if metrics.get("phase") == "complete":
        ax.text(0.50, 0.08,
                f"TEST  NSE={metrics.get('test_nse','?')}  "
                f"RMSE={metrics.get('test_rmse','?')} m",
                ha="center", va="center", fontsize=8,
                color=C_SAR, fontweight="bold")


def build_figure():
    """Create and return the figure + all axes."""
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    fig.suptitle(
        "PI-ST-GNN + SAR-FNO  ·  Live Training Dashboard  ·  River Lee Catchment",
        fontsize=12, fontweight="bold", color=C_TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.97, top=0.93, bottom=0.07,
        wspace=0.32, hspace=0.45,
    )
    ax_prog  = fig.add_subplot(gs[0, 0])
    ax_loss  = fig.add_subplot(gs[0, 1])
    ax_sar   = fig.add_subplot(gs[0, 2])
    ax_nse   = fig.add_subplot(gs[1, 0])
    ax_rmse  = fig.add_subplot(gs[1, 1])
    ax_lr    = fig.add_subplot(gs[1, 2])
    return fig, ax_prog, ax_loss, ax_sar, ax_nse, ax_rmse, ax_lr


def update(fig, axes, metrics):
    """Redraw all panels from the latest metrics dict."""
    ax_prog, ax_loss, ax_sar, ax_nse, ax_rmse, ax_lr = axes
    history = metrics.get("history", [])

    draw_progress_bar(
        ax_prog,
        metrics.get("batch_idx"),
        metrics.get("n_batches"),
        metrics.get("epoch"),
        metrics.get("max_epochs"),
        metrics.get("phase"),
        metrics.get("running_loss"),
    )
    draw_loss_curve(ax_loss, history)
    draw_sar_status(ax_sar, metrics)
    draw_nse_curve(ax_nse, history)
    draw_rmse_curve(ax_rmse, history)
    draw_lr_curve(ax_lr, history)

    # Footer timestamp
    ts = metrics.get("ts")
    if ts:
        import datetime
        dt = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        fig.texts = [t for t in fig.texts
                     if "Last update" not in t.get_text()]
        fig.text(0.99, 0.005, f"Last update: {dt}",
                 ha="right", va="bottom", fontsize=7, color=C_PERSIST)


def run_live(metrics_path: Path, interval: float) -> None:
    """Poll metrics_path every interval seconds and redraw."""
    matplotlib.use("TkAgg" if sys.platform == "win32" else "Qt5Agg")
    plt.ion()

    fig, *axes = build_figure()
    plt.show(block=False)

    last_mtime = None
    print(f"Dashboard started. Watching: {metrics_path}")
    print(f"Refresh interval: {interval}s  |  Close window to exit")
    print()

    try:
        while plt.fignum_exists(fig.number):
            try:
                mtime = metrics_path.stat().st_mtime if metrics_path.exists() else None
            except OSError:
                mtime = None

            if mtime != last_mtime:
                metrics = load_metrics(metrics_path)
                if metrics:
                    update(fig, axes, metrics)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    epoch = metrics.get("epoch", "?")
                    phase = metrics.get("phase", "waiting")
                    batch = metrics.get("batch_idx")
                    n_bat = metrics.get("n_batches")
                    if batch is not None and n_bat:
                        print(f"\r  Epoch {epoch} | batch {batch+1}/{n_bat} "
                              f"| phase: {phase}        ", end="", flush=True)
                    else:
                        print(f"\r  Epoch {epoch} | phase: {phase}        ",
                              end="", flush=True)
                    last_mtime = mtime
                else:
                    print(f"\r  Waiting for {metrics_path.name}…", end="")
            else:
                # No change — just pump the event loop
                fig.canvas.flush_events()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nDashboard closed.")


def save_snapshot(metrics_path: Path, out_path: Path) -> None:
    """Save a static snapshot of the current metrics to a PNG file."""
    metrics = load_metrics(metrics_path)
    if not metrics:
        print(f"No metrics found at {metrics_path}")
        sys.exit(1)

    matplotlib.use("Agg")
    fig, *axes = build_figure()
    update(fig, axes, metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    print(f"Snapshot saved: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Live training dashboard for PI-ST-GNN + SAR-FNO"
    )
    p.add_argument(
        "--metrics", type=Path, default=DEFAULT_PATH,
        help=f"Path to live_metrics.json (default: {DEFAULT_PATH})"
    )
    p.add_argument(
        "--interval", type=float, default=2.0,
        help="Refresh interval in seconds (default: 2)"
    )
    p.add_argument(
        "--save", type=Path, default=None,
        help="Save a static snapshot to this path instead of live display"
    )
    args = p.parse_args()

    if args.save:
        save_snapshot(args.metrics, args.save)
    else:
        run_live(args.metrics, args.interval)


if __name__ == "__main__":
    main()
