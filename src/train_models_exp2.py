"""
train_models_exp2.py  —  Experiment 2 Orchestrator  (Remote Sensing of Environment)
══════════════════════════════════════════════════════════════════════════════════════
Runs the targeted SAR contribution experiment and HANDDecoder calibration.

This script does NOT re-run the full 6-model comparison — those results come
from Experiment 1 (train_models_exp1.py) and are cited in the RSE paper as
[Author et al., 2025].  Experiment 2 answers two specific questions:

  Q1 — SAR ablation
       Does Sentinel-1 antecedent wetness encoding add measurable forecast
       skill above the best architecture from Experiment 1?
       Runs: best_model × {SAR on, SAR off} × 3 seeds × 2 horizons = 12 runs
       (typically ~6 hr on a single GPU)

  Q2 — HANDDecoder calibration
       Does calibrating per-node HAND depth scales (τ_k) from Sentinel-1
       flood extent masks improve CSI against withheld SAR observations?
       Runs: train_hand_decoder.py  (27 parameters, ~200 epochs, < 30 min)

Design rationale
────────────────
The SAR ablation uses only ONE architecture (the Experiment 1 winner).
Running all 6 architectures with and without SAR would cost 72 additional
training runs for a question whose answer only requires 12.  RSE reviewers
need the ablation to be clean (one variable changed, everything else fixed),
not comprehensive.

Horizons for the SAR ablation are T_out = 24 (6 hr) and T_out = 48 (12 hr)
— operationally relevant leads where antecedent soil moisture information is
most likely to improve performance.  The 1-hr and 3-hr results from Exp 1
already demonstrate short-lead accuracy.

Checkpoint structure
────────────────────
  Experiment 1 results (cited, not retrained):
    checkpoints/{model}/{seed}/{t_out}/

  Experiment 2 SAR-on runs (new):
    checkpoints/{model}_sar/{seed}/{t_out}/

  HANDDecoder calibration:
    checkpoints/hand_decoder/{seed}/
    results/hand_decoder/csi_report.csv

Configuration
─────────────
  BEST_MODEL_EXP1 : str
    Set this after reviewing Experiment 1 results.
    Options: "st_gnn_hand_edge" (predicted best) or "dfc_gnn".
    Default: "st_gnn_hand_edge"

Usage
──────
  # Full Experiment 2 (SAR ablation + HANDDecoder)
  python src/train_models_exp2.py

  # SAR ablation only
  python src/train_models_exp2.py --ablation-only

  # HANDDecoder only
  python src/train_models_exp2.py --decoder-only

  # Override best model
  python src/train_models_exp2.py --best-model dfc_gnn
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR)       not in sys.path: sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR/"src") not in sys.path: sys.path.insert(0, str(BASE_DIR/"src"))

from utils.common_utils import seed_everything
from utils.config       import load_config
from utils.logger       import get_logger

# ── Experiment 2 configuration ─────────────────────────────────────────
# Update BEST_MODEL_EXP1 after reviewing Experiment 1 results table
BEST_MODEL_EXP1 = "st_gnn_hand_edge"   # predicted winner from preliminary results

SEEDS            = [42, 123, 456]
T_OUTS_ABLATION  = [24, 48]            # 6hr and 12hr — operational leads
T_IN             = 32
MAX_EPOCHS       = 300

# Train functions for each candidate best model
_BEST_MODEL_TRAIN_MAP = {
    "st_gnn_hand_edge": None,   # populated below to avoid circular import
    "dfc_gnn":          None,
}


# ══════════════════════════════════════════════════════════════════════
# SAR ablation
# ══════════════════════════════════════════════════════════════════════

def run_sar_ablation(best_model: str, seeds: list[int],
                     t_outs: list[int], max_epochs: int,
                     logger, skip_existing: bool):
    """
    Run the SAR ablation: best Exp1 model with SAR on vs off.

    SAR-off results already exist in checkpoints/{model}/{seed}/{t_out}/
    from Experiment 1.  This function trains SAR-on runs only and saves
    them to checkpoints/{model}_sar/{seed}/{t_out}/ using the staging
    approach (train to a temporary directory, rename on completion).

    The final RSE comparison table will be:
      Model                         T_out=24hr  T_out=48hr
      ─────────────────────────────────────────────────────
      {best_model} (SAR off, Exp 1)  from Exp 1   from Exp 1
      {best_model} (SAR on,  Exp 2)  ← trained here
      ΔNSE                           Exp 2 − Exp 1
    """
    print("\n" + "═"*60)
    print(f"  EXPERIMENT 2 — SAR ablation")
    print(f"  Model: {best_model}  (SAR on vs off)")
    print(f"  Horizons: {t_outs}  steps = {[t*15//60 for t in t_outs]} hr")
    print("═"*60)

    # Select training function and module for patching
    if best_model == "st_gnn_hand_edge":
        import train_st_gnn_hand_edge as _mod
        fn = _mod.train
    elif best_model == "dfc_gnn":
        import train_dfc_gnn as _mod
        fn = _mod.train
    else:
        raise ValueError(f"Unknown best model: {best_model}. "
                         f"Options: 'st_gnn_hand_edge', 'dfc_gnn'")

    total = len(seeds) * len(t_outs)
    done  = 0

    for seed in seeds:
        seed_everything(seed)
        for t_out in t_outs:
            done += 1
            sar_ckpt_final = (BASE_DIR / "checkpoints"
                              / f"{best_model}_sar" / str(seed) / str(t_out))

            if skip_existing and (sar_ckpt_final / "best_model.pt").exists():
                print(f"\n[{done}/{total}]  SKIP  {best_model}_sar  "
                      f"seed={seed}  T_out={t_out}")
                continue

            print(f"\n[{done}/{total}]  {best_model}_sar  seed={seed}  "
                  f"T_in={T_IN}  T_out={t_out}  ({t_out*15//60}hr)  SAR=ON")

            # ── Staging: train to temp dir, then move ─────────────────
            staging = BASE_DIR / "_exp2_staging" / f"{best_model}_{seed}_{t_out}"
            staging.mkdir(parents=True, exist_ok=True)

            # Patch module: enable SAR, redirect BASE_DIR to staging
            orig_base = _mod.BASE_DIR
            _mod.BASE_DIR = staging

            if best_model == "st_gnn_hand_edge":
                orig_use_sar   = _mod.USE_SAR
                _mod.USE_SAR   = True
            elif best_model == "dfc_gnn":
                orig_use_sar   = _mod.USE_SAR_EDGE
                _mod.USE_SAR_EDGE = True

            try:
                fn(logger, seed, T_IN, t_out, max_epochs, None)

                # Locate the trained checkpoint inside staging
                # Train scripts write to staging / "checkpoints" / run_tag / seed / t_out
                if best_model == "dfc_gnn":
                    staged = staging / "checkpoints" / "dfc_gnn_sar" / str(seed) / str(t_out)
                else:
                    staged = staging / "checkpoints" / best_model / str(seed) / str(t_out)

                if staged.exists():
                    sar_ckpt_final.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged), str(sar_ckpt_final))
                    print(f"  → Moved to {sar_ckpt_final.relative_to(BASE_DIR)}")
                else:
                    print(f"  WARNING: expected staged checkpoint not found at {staged}")

            finally:
                _mod.BASE_DIR = orig_base
                if best_model == "st_gnn_hand_edge":
                    _mod.USE_SAR      = orig_use_sar
                elif best_model == "dfc_gnn":
                    _mod.USE_SAR_EDGE = orig_use_sar

                shutil.rmtree(staging, ignore_errors=True)

    print(f"\n  SAR ablation complete.")
    print(f"  SAR-off results:  checkpoints/{best_model}/")
    print(f"  SAR-on  results:  checkpoints/{best_model}_sar/")
    print()
    print("  To compare: run analyse_experiments.py including both tags.")


# ══════════════════════════════════════════════════════════════════════
# HANDDecoder calibration
# ══════════════════════════════════════════════════════════════════════

def run_hand_decoder(best_model: str, seeds: list[int], logger):
    """
    Call train_hand_decoder.py for each seed using the best model checkpoint.

    HANDDecoder trains only decoder.log_tau (27 parameters) — the main model
    is frozen.  The CSI result gates the RSE submission decision:
      CSI ≥ 0.10  → RSE submission viable
      CSI < 0.05  → target TGRS or HSJ with methodology framing
    """
    print("\n" + "═"*60)
    print("  EXPERIMENT 2 ")
    print(f"  Base model: {best_model}")
    print(f"  Seeds: {seeds}")
    print("  27 parameters (log_tau per node), ~200 epochs, < 30 min")
    print("  *** This result gates the RSE submission decision ***")
    print("═"*60)

    try:
        from train_hand_decoder import train_hand_decoder as _train_decoder
        for seed in seeds:
            seed_everything(seed)
            print(f"\n  Seed {seed} …")
            _train_decoder(
                logger      = logger,
                seed        = seed,
                model_tag   = best_model,
                n_epochs    = 200,
                base_dir    = None,
            )
    except ImportError:
        print()
        print("  train_hand_decoder.py not found in sys.path.")
        print("  Copy it to src/ and re-run:")
        print("    python src/train_models_exp2.py --decoder-only")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Experiment 2"
    )
    p.add_argument("--best-model", default=BEST_MODEL_EXP1,
                   choices=["st_gnn_hand_edge", "dfc_gnn"],
                   help=f"Best architecture from Experiment 1 (default: {BEST_MODEL_EXP1})")
    p.add_argument("--ablation-only", action="store_true",
                   help="Run SAR ablation only (skip HANDDecoder)")
    p.add_argument("--decoder-only",  action="store_true",
                   help="Run HANDDecoder calibration only (skip SAR ablation)")
    p.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    p.add_argument("--horizons", nargs="+", type=int, default=T_OUTS_ABLATION)
    p.add_argument("--epochs",   type=int,            default=MAX_EPOCHS)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    config = load_config(BASE_DIR / "config" / "config.yaml")
    logger = get_logger(config["logging"]["train"])

    print(f"\n  Best model from Experiment 1: {args.best_model}")
    print(f"  (change with --best-model after reviewing Exp 1 results)")

    if not args.decoder_only:
        run_sar_ablation(
            args.best_model, args.seeds, args.horizons,
            args.epochs, logger, args.skip_existing,
        )

    if not args.ablation_only:
        run_hand_decoder(args.best_model, args.seeds, logger)


if __name__ == "__main__":
    main()
