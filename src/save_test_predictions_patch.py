"""
save_test_predictions_patch.py
==============================
Instructions for adding test_predictions.npy saving to any training script.

The flood map generator (generate_flood_maps.py) looks for:
    checkpoints/<model>/test_predictions.npy   shape [T_test, N]

If this file does not exist the generator falls back to ground truth
(useful for testing map generation without re-training).

To save real predictions, add this block to the test evaluation section
of any training script, immediately after cat_pred is assembled:

    # ── Save test predictions for flood map generation ────────────────
    np.save(ckpt_dir / "test_predictions.npy", cat_pred.cpu().numpy())
    logger.info("Saved test_predictions.npy  shape=%s", cat_pred.shape)

cat_pred should be shape [T_test, N] — the absolute stage predictions
(i.e. after adding back last_obs, not the raw delta output).

In the existing training scripts cat_pred is constructed as:
    cat_pred = torch.cat(all_abs_pred, dim=0)   # [T_test, N]
so inserting the save right after that line works for all model variants.
"""

# Quick patch function — call this on any training script to add the save
import re
from pathlib import Path


def patch_training_script(script_path: str) -> bool:
    """
    Insert test_predictions.npy save into a training script.
    Returns True if patched, False if already patched or not applicable.
    """
    path = Path(script_path)
    text = path.read_text()

    if "test_predictions.npy" in text:
        print(f"{path.name}: already has test_predictions.npy save")
        return False

    # Find the point just after cat_pred is assembled in the test loop
    old = "    cat_pred    = torch.cat(all_abs_pred,  dim=0)"
    new = (
        "    cat_pred    = torch.cat(all_abs_pred,  dim=0)\n"
        "\n"
        "    # Save absolute stage predictions for flood map generation\n"
        "    np.save(ckpt_dir / \"test_predictions.npy\",\n"
        "            cat_pred.cpu().numpy())\n"
        "    logger.info(\"Saved test_predictions.npy  shape=%s\",\n"
        "                tuple(cat_pred.shape))"
    )

    if old not in text:
        print(f"{path.name}: target line not found — check script structure")
        return False

    text = text.replace(old, new, 1)

    import ast
    ast.parse(text)
    path.write_text(text)
    print(f"{path.name}: patched ✓")
    return True


if __name__ == "__main__":
    import sys
    scripts = sys.argv[1:] or [
        "src/train_st_gnn_flood_model.py",
        "src/train_st_gnn_flood_model_sar_v1.py",
        "src/train_st_gnn_dyn_edge.py",
        "src/train_st_gnn_hand_edge.py",
        "src/train_per_node_gru_model.py",
        "src/train_per_node_lstm_model.py",
    ]
    for s in scripts:
        try:
            patch_training_script(s)
        except Exception as e:
            print(f"  {s}: {e}")
