"""
run_extended_experiments.py
══════════════════════════════════════════════════════════════════════
Extends the primary comparison matrix with three scientifically
motivated additional experiment tracks.

Track A — Extended forecast horizons (all 7 models)
────────────────────────────────────────────────────
Adds T_out = 24 (6 hr) and T_out = 48 (12 hr) to the existing
T_out = [4, 12, 16] suite.  These thresholds are operationally
motivated:
  • T_out = 24 (6 hr):  Irish Civil Defence activation threshold;
                         minimum lead time for flood response in Cork city
  • T_out = 48 (12 hr): Evacuation planning and road closure threshold;
                         directly comparable to OPW operational forecasts
Without these horizons, the study cannot be positioned as an
operational early warning contribution.

Track B — T_in sensitivity ablation (DFC-GNN vs GRU)
─────────────────────────────────────────────────────
Tests T_in ∈ {16, 64, 96} steps (4 hr, 16 hr, 24 hr) against the
baseline T_in = 32 (8 hr) with T_out = 4 (1 hr, most sensitive to
antecedent conditions).

Scientific motivation:
  (i)  T_in = 32 is 0.8× the Lee catchment's maximum wave travel time
       (9.4 h computed from edge_features.npz).  The model may not see
       a complete upstream pulse in 8 h of input.
  (ii) ERA5-Land soil moisture (swvl1/swvl2) is already in the feature
       set.  If it encodes the antecedent signal adequately, longer T_in
       adds no value — and that result is itself informative.
  (iii)Directly answers the examiner challenge:
       "Could you achieve the same antecedent wetness information by
       simply using a longer input window instead of SAR?"
  If DFC-GNN(T_in=32 + SAR) > DFC-GNN(T_in=96 − SAR), the SAR
  feature provides genuinely irreducible antecedent information.

Track C — 24-hour operational benchmark (DFC-GNN vs GRU)
──────────────────────────────────────────────────────────
Adds T_out = 96 (24 hr) for DFC-GNN and GRU only.  The OPW currently
operates a HEC-HMS + hydraulic routing pipeline at 24-hour lead time.
Without a 24-hour result, the thesis cannot claim operational
equivalence or superiority to existing practice.

Usage
──────
  # All three tracks
  python src/run_extended_experiments.py

  # Single track
  python src/run_extended_experiments.py --track A
  python src/run_extended_experiments.py --track B
  python src/run_extended_experiments.py --track C

  # Dry run — print what would be run without training
  python src/run_extended_experiments.py --dry-run

Checkpoint structure
─────────────────────
  Track A, C: checkpoints/{model}/{seed}/{t_out}/   (existing layout)
  Track B:    checkpoints/ablation_tin/{model}/{seed}/{t_in}/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR)       not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR/"src") not in sys.path: sys.path.insert(0, str(BASE_DIR/"src"))

from utils.common_utils import seed_everything
from utils.config       import load_config
from utils.logger       import get_logger

# ── Import all training functions ──────────────────────────────────────
from train_per_node_gru_model  import train as train_gru
from train_per_node_lstm_model import train as train_lstm
from train_st_gnn_flood_model       import train as train_st_gnn
from train_st_gnn_flood_model_sar   import train as train_st_gnn_sar
from train_st_gnn_dyn_edge          import train as train_st_gnn_dyn_edge
from train_st_gnn_hand_edge         import train as train_st_gnn_hand_edge
from train_dfc_gnn                  import train as train_dfc_gnn

# ── Experiment configuration ────────────────────────────────────────────
SEEDS      = [42, 123, 456]
MAX_EPOCHS = 300

# Track A: extended horizons — all 7 models × 2 new horizons × 3 seeds
# T_out = 24 (6 hr civil defence threshold) and 48 (12 hr evacuation threshold)
TRACK_A_T_IN  = 32                       # fixed — maintain comparability
TRACK_A_T_OUT = [24, 48]                 # new horizons only; [4,12,16] already trained

# Track B: T_in ablation — DFC-GNN and GRU only × 4 T_in values × 1 T_out × 3 seeds
# T_in=32 already exists in main checkpoints — skip to avoid redundant retraining
TRACK_B_T_IN_NEW = [16, 64, 96]          # new T_in values (32 = existing)
TRACK_B_T_IN_ALL = [16, 32, 64, 96]      # full set for analysis
TRACK_B_T_OUT    = 4                      # 1-hr horizon (most sensitive to antecedent)

# Track C: 24-hour operational benchmark — DFC-GNN and GRU × T_out=96 × 3 seeds
TRACK_C_T_IN  = 32
TRACK_C_T_OUT = 96                        # 24 hr

# ── Track A ─────────────────────────────────────────────────────────────

def run_track_a(base_dir, logger, dry_run: bool = False):
    """Extended horizons for all 7 models."""
    print("\n" + "═"*60)
    print("  TRACK A — Extended horizons: T_out = 24 (6hr) and 48 (12hr)")
    print("  Motivation: Civil Defence activation (6hr) and")
    print("  evacuation planning (12hr) thresholds for Cork flood response")
    print("═"*60)

    train_fns = {
        "gru":             train_gru,
        "lstm":            train_lstm,
        "st_gnn":          train_st_gnn,
        "st_gnn_sar":      train_st_gnn_sar,
        "st_gnn_dyn_edge": train_st_gnn_dyn_edge,
        "st_gnn_hand_edge":train_st_gnn_hand_edge,
        "dfc_gnn":         train_dfc_gnn,
    }

    total = len(SEEDS) * len(TRACK_A_T_OUT) * len(train_fns)
    done  = 0
    for seed in SEEDS:
        seed_everything(seed)
        for t_out in TRACK_A_T_OUT:
            for tag, fn in train_fns.items():
                done += 1
                print(f"\n[Track A  {done}/{total}]  {tag}  seed={seed}  "
                      f"t_in={TRACK_A_T_IN}  t_out={t_out}  ({t_out*15//60}hr)")
                if not dry_run:
                    fn(logger, seed, TRACK_A_T_IN, t_out, MAX_EPOCHS, base_dir)


# ── Track B ─────────────────────────────────────────────────────────────

class _GRUTrainAdaptor:
    """
    Thin wrapper so train_gru can be called with a custom t_in that is
    stored in the ablation checkpoint directory rather than the standard one.

    The training scripts all accept (logger, seed, t_in, t_out, max_epochs).
    For the ablation we additionally redirect the checkpoint root so ablation
    runs don't overwrite the primary comparison checkpoints.
    """
    pass


def _run_tin_ablation_model(base_dir, fn_train, tag: str, logger,
                             seed: int, t_in: int, dry_run: bool):
    """
    Run one T_in ablation training run.

    Checkpoint dir: checkpoints/ablation_tin/{tag}/{seed}/{t_in}/
    This is achieved by temporarily monkey-patching BASE_DIR inside the
    training script — the cleanest approach without modifying training scripts.
    """
    import importlib, types

    ckpt_root_ablation = base_dir / "checkpoints" / "ablation_tin"
    ckpt_root_ablation.mkdir(parents=True, exist_ok=True)

    # Each training script constructs its ckpt_dir from BASE_DIR / "checkpoints"
    # and the run_tag string.  We patch the module's CKPT_ROOT or equivalent.
    # Since training scripts use:
    #   ckpt_dir = BASE_DIR / "checkpoints" / run_tag / str(seed) / str(t_out)
    # we need to point them at the ablation directory.
    # Simplest approach: pass t_in and let the script use it via run_tag override.
    #
    # Because training scripts don't natively support ablation paths, we call
    # them with a custom run_tag via a temporary patch.

    # Determine the module for this training function
    module = sys.modules.get(fn_train.__module__)
    if module is None:
        print(f"  [skip — cannot find module for {tag}]")
        return

    orig_tag = getattr(module, "RUN_TAG", None) if hasattr(module, "RUN_TAG") else None

    # Temporarily override the checkpoint path by setting the ablation tag
    ablation_tag = f"ablation_tin/{tag}"
    if hasattr(module, "RUN_TAG"):
        module.RUN_TAG = ablation_tag

    if dry_run:
        print(f"  [DRY RUN] would train {tag}  seed={seed}  "
              f"t_in={t_in}  t_out={TRACK_B_T_OUT}")
    else:
        try:
            fn_train(logger, seed, t_in, TRACK_B_T_OUT, MAX_EPOCHS, base_dir)
        finally:
            # Restore original tag
            if hasattr(module, "RUN_TAG") and orig_tag is not None:
                module.RUN_TAG = orig_tag


def run_track_b(base_dir, logger, dry_run: bool = False):
    """T_in sensitivity ablation for DFC-GNN and GRU."""
    print("\n" + "═"*60)
    print("  TRACK B — T_in ablation: DFC-GNN vs GRU")
    print("  T_in ∈ {16, 32, 64, 96} steps = {4, 8, 16, 24} hours")
    print("  T_out = 4 (1 hr) — most sensitive to antecedent conditions")
    print("  Key question: is ERA5-Land + T_in=32 sufficient, or does")
    print("  longer context replace the SAR antecedent wetness signal?")
    print("═"*60)

    ablation_fns = {
        "gru":     train_gru,
        "dfc_gnn": train_dfc_gnn,
    }

    total = len(SEEDS) * len(TRACK_B_T_IN_NEW) * len(ablation_fns)
    done  = 0
    for seed in SEEDS:
        seed_everything(seed)
        for t_in in TRACK_B_T_IN_NEW:   # skip t_in=32 — already trained
            for tag, fn in ablation_fns.items():
                done += 1
                print(f"\n[Track B  {done}/{total}]  {tag}  seed={seed}  "
                      f"t_in={t_in}  t_out={TRACK_B_T_OUT}  "
                      f"({t_in*15//60}hr context)")
                _run_tin_ablation_model(base_dir, fn, tag, logger, seed, t_in, dry_run)


# ── Track C ─────────────────────────────────────────────────────────────

def run_track_c(base_dir, logger, dry_run: bool = False):
    """24-hour operational benchmark for GRU vs ST-GNN HAND vs DFC-GNN."""
    print("\n" + "═" * 60)
    print("  TRACK C — 24-hour benchmark: GRU vs ST-GNN HAND vs DFC-GNN")
    print("  T_out = 96 steps = 24 hr  |  3 models × 3 seeds = 9 runs")
    print("  Three-way: no-graph / best static-graph / proposed model")
    print("  Allows claim: DFC-GNN > best competing graph model at 24hr")
    print("  Benchmark: OPW HEC-HMS operational forecast (24hr lead)")
    print("═" * 60)

    benchmark_fns = {
        "gru": train_gru,  # no-graph baseline
        "st_gnn_hand_edge": train_st_gnn_hand_edge,  # best static-graph baseline
        "dfc_gnn": train_dfc_gnn,  # proposed model
    }

    total = len(SEEDS) * len(benchmark_fns)
    done  = 0
    for seed in SEEDS:
        seed_everything(seed)
        for tag, fn in benchmark_fns.items():
            done += 1
            print(f"\n[Track C  {done}/{total}]  {tag}  seed={seed}  "
                  f"t_in={TRACK_C_T_IN}  t_out={TRACK_C_T_OUT}  (24 hr)")
            if not dry_run:
                fn(logger, seed, TRACK_C_T_IN, TRACK_C_T_OUT, MAX_EPOCHS, base_dir)


# ── Summary printer ─────────────────────────────────────────────────────

def print_plan():
    """Print the full experiment plan before starting."""
    t_in = 32
    all_t_out = sorted(set([4, 12, 16] + TRACK_A_T_OUT + [TRACK_C_T_OUT]))
    all_t_in  = sorted(TRACK_B_T_IN_ALL)

    print("\n" + "═"*60)
    print("  EXTENDED EXPERIMENT PLAN")
    print("═"*60)

    a_runs = len(SEEDS) * len(TRACK_A_T_OUT) * 7
    b_runs = len(SEEDS) * len(TRACK_B_T_IN_NEW) * 2
    c_runs = len(SEEDS) * 2
    total  = a_runs + b_runs + c_runs

    print(f"""
  Track A  Extended horizons (all 7 models):
    T_in  = {t_in} steps ({t_in*15//60}hr input)
    T_out = {TRACK_A_T_OUT} steps ({[t*15//60 for t in TRACK_A_T_OUT]} hr)
    Seeds = {SEEDS}
    New runs = {len(SEEDS)} × {len(TRACK_A_T_OUT)} × 7 = {a_runs}

  Track B  T_in ablation (DFC-GNN + GRU):
    T_in  = {TRACK_B_T_IN_NEW} steps ({[t*15//60 for t in TRACK_B_T_IN_NEW]} hr)
    T_out = {TRACK_B_T_OUT} step (1 hr)
    Seeds = {SEEDS}
    New runs = {len(SEEDS)} × {len(TRACK_B_T_IN_NEW)} × 2 = {b_runs}
    (T_in=32 reuses existing checkpoints)

  Track C  24-hr OPW benchmark (DFC-GNN + GRU):
    T_in  = {TRACK_C_T_IN} steps ({TRACK_C_T_IN*15//60}hr input)
    T_out = {TRACK_C_T_OUT} steps ({TRACK_C_T_OUT*15//60} hr)
    Seeds = {SEEDS}
    New runs = {len(SEEDS)} × 2 = {c_runs}

  ─────────────────────────────────────
  Total new training runs: {total}
  (existing [4,12,16] with T_in=32 are NOT retrained)

  Forecast horizons after all tracks:
    Primary comparison: T_out = {[4,12,16,24,48]} ({[1,3,4,6,12]} hr)
    DFC-GNN/GRU only:  T_out = {all_t_out} ({[t*15//60 for t in all_t_out]} hr)

  Input window ablation (DFC-GNN + GRU):
    T_in  = {all_t_in} steps ({[t*15//60 for t in all_t_in]} hr)
""")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run extended experiment tracks A, B, C"
    )
    parser.add_argument("--track", choices=["A","B","C","all"], default="all",
                        help="Which track to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without training")
    args = parser.parse_args()

    config_path = BASE_DIR / "config" / "config.yaml"
    config = load_config(config_path)
    logger = get_logger(config["logging"]["train"])

    print_plan()

    if args.dry_run:
        print("  [DRY RUN — no training will occur]\n")

    if args.track in ("A", "all"):
        base_dir = BASE_DIR / "checkpoints" /"track_a"
        run_track_a(base_dir, logger, args.dry_run)

    if args.track in ("B", "all"):
        base_dir = BASE_DIR / "checkpoints" / "track_b"
        run_track_b(base_dir, logger, args.dry_run)

    if args.track in ("C", "all"):
        base_dir = BASE_DIR / "checkpoints" / "track_c"
        run_track_c(base_dir, logger, args.dry_run)

    print("\nAll requested tracks complete.")


if __name__ == "__main__":
    main()
