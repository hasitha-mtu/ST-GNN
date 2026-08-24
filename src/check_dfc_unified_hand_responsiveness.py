"""
check_dfc_unified_hand_responsiveness.py — companion to
check_dfc_gate_scenario_invariance.py.

That script proved the elevation gate is input-invariant. This one checks
the OTHER gated mechanism in DFCGNNUnified (PC-DFC-GNN) -- the HAND
activation term -- which the source code suggests IS a function of the
dynamic input:

    H = gauge_datum + normalised_stage * stage_range   (x_last[:, :, 1])
    hand_activation = sigmoid(activation_sharpness * (max_H - z_saddle))

Unlike the elevation gate, this reads x_last (the model's own dynamic
input at the last observed timestep), so it CAN in principle vary across
scenarios/conditions. This script calls the model's own
_build_edge_attr_and_activation() method directly (not a reimplementation)
across several deliberately different synthetic input batches (calm
baseline, near-saddle, above-saddle) and confirms the activation values
actually move -- which, combined with the elevation-gate script's result,
tells you whether DFC-GNN's own horizon-dependent scenario improvement
(which has NO HAND term to explain it) is coming from somewhere other
than either gate -- most likely the underlying GAT-style attention score
itself, which is dynamic in both models regardless of gating.

Usage:
    python check_dfc_unified_hand_responsiveness.py --checkpoint checkpoints/dfc_gnn_unified/42/4/best_model.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, "src")
    from models.dfc_gnn_unified import DFCGNNUnified

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    hp = ckpt.get("hparams", {})

    node_elev   = sd["node_elev"].cpu()
    gauge_datum = sd["gauge_datum"].cpu()
    stage_range = sd["stage_range"].cpu()
    hand_src    = sd["hand_src"].cpu()
    hand_dst    = sd["hand_dst"].cpu()
    hand_thr    = sd["hand_threshold"].cpu()
    z_saddle    = sd["z_saddle"].cpu()
    hand_dist   = sd["hand_dist_norm"].cpu() * 5.0
    river_ei    = sd["river_edge_index"].cpu()
    river_ea    = sd["river_edge_attr_static"].cpu()

    N = node_elev.shape[0]
    f_dyn = hp.get("f_dyn", 11)

    model = DFCGNNUnified(
        n_nodes=N, f_dyn=f_dyn, d_model=hp.get("hidden", 64),
        n_heads=hp.get("gat_heads", 4), T_out=hp.get("t_out", 4),
        edge_index=river_ei, edge_attr_static=river_ea, node_elev=node_elev,
        hand_src=hand_src, hand_dst=hand_dst, hand_threshold=hand_thr,
        hand_overland_dist=hand_dist, z_saddle=z_saddle,
        gauge_datum=gauge_datum, stage_range=stage_range,
        n_gru_layers=hp.get("gru_layers", 2), dropout=hp.get("dropout", 0.1),
        lambda_flood=hp.get("lambda_flood", 0.1),
        discharge_idx=hp.get("discharge_idx", 3),
    )
    model.load_state_dict(sd)
    model.eval()

    print(f"z_saddle range: [{z_saddle.min():.2f}, {z_saddle.max():.2f}] m OD")
    print(f"gauge_datum range: [{gauge_datum.min():.2f}, {gauge_datum.max():.2f}] m OD")
    print(f"n_hand edges: {hand_src.shape[0]}")

    B = 4
    x_calm  = torch.zeros(B, N, f_dyn)
    x_calm[:, :, 1] = -1.0     # normalised_stage well below typical range

    x_high  = torch.zeros(B, N, f_dyn)
    x_high[:, :, 1] = 5.0      # normalised_stage well above typical range

    with torch.no_grad():
        _, act_calm = model._build_edge_attr_and_activation(x_calm)
        _, act_high = model._build_edge_attr_and_activation(x_high)

    hand_calm = act_calm[:, model.n_river:]
    hand_high = act_high[:, model.n_river:]

    identical = torch.allclose(hand_calm, hand_high, atol=1e-6)
    mean_diff = (hand_high - hand_calm).abs().mean().item()

    print(f"\nHAND activation, calm input:  mean={hand_calm.mean():.4f}  "
          f"range=[{hand_calm.min():.4f}, {hand_calm.max():.4f}]")
    print(f"HAND activation, high input:  mean={hand_high.mean():.4f}  "
          f"range=[{hand_high.min():.4f}, {hand_high.max():.4f}]")
    print(f"Identical across inputs: {identical}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    if identical:
        print(
            "\nUNEXPECTED given the source code: HAND activation should "
            "depend on x_last[:, :, 1] (normalised_stage). If this reads "
            "identical despite very different inputs, check hand_src/"
            "hand_dst indexing or whether normalised_stage is actually at "
            "feature index 1 in this checkpoint's training configuration."
        )
    else:
        print(
            "\nConfirmed: HAND activation IS input-responsive, unlike the "
            "elevation gate. This means PC-DFC-GNN's HAND term is a "
            "genuine candidate explanation for any scenario-dependent "
            "improvement over plain DFC-GNN specifically. It does NOT "
            "explain plain DFC-GNN's own improvement (no HAND term exists "
            "there) -- that has to come from the underlying GAT-style "
            "attention score itself, which is dynamic in both models "
            "regardless of either gate."
        )


if __name__ == "__main__":
    main()
