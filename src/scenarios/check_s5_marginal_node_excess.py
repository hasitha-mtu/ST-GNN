"""
check_s5_marginal_node_excess.py — verifies, against REAL historical
rainfall/saturation data, whether the 4 marginal downstream nodes
excluded by the conservative g>1.05 fix would EVER actually trigger an
injection under the real two-stage gate
(g > 1.05 AND rain_excess * sat >= 5.0), or whether the second gate
already filters them out in practice, making the conservative exclusion
stricter than necessary.

Uses the EXACT same formula as generate_s5_spatial_gradient's injection
loop (copied verbatim, not reimplemented from memory) so this gives a
real answer, not an approximation:

    rain_excess = sum(X_w[T_IN:T_IN+16, n_idx, F_RAIN] * gradient[n_idx])
    sat         = mean(X_w[:T_IN, n_idx, F_SW2_SAT])
    excess      = rain_excess * sat
    triggers injection if excess >= 5.0

Checks across a large number of real historical windows (not just the
small number a scenario run would sample) and across the FULL sampled
gradient_max range (1.8-3.0), to give the most complete picture of
whether these 4 nodes are ever practically at risk of direct injection.

Usage:
    python check_s5_marginal_node_excess.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

T_IN = 32
F_RAIN = 4      # matches scenario_generator.py's F_RAIN index
F_SW2_SAT = 9
STEP_MIN = 15

MARGINAL_NODES = [22, 23, 24, 25]   # Waterworks Weir, Fitzgerald's Park, Pope's Quay, St. Patrick's Quay
GRADIENT_MAX_RANGE = (1.8, 3.0)


def main():
    proc_dir = Path("dataset/processed")
    graph_dir = Path("dataset/graph")

    X = np.load(proc_dir / "X.npy")
    nd = pd.read_csv(graph_dir / "nodes.csv")

    easting = nd["easting_itm"].values.astype(np.float64)
    e_min, e_max = easting.min(), easting.max()
    spatial_weight = 1.0 - (easting - e_min) / (e_max - e_min + 1)

    def gradient_for(gradient_max):
        return (1.0 + (gradient_max - 1.0) * spatial_weight).astype(np.float32)

    T = X.shape[0]
    T_WINDOW = 104
    n_windows_to_check = min(2000, (T - T_WINDOW) // T_IN)   # broad sample, real data
    starts = np.linspace(0, T - T_WINDOW - 1, n_windows_to_check, dtype=int)

    print(f"Checking {len(starts)} real historical windows across "
          f"gradient_max in [{GRADIENT_MAX_RANGE[0]}, {GRADIENT_MAX_RANGE[1]}]\n")

    for node in MARGINAL_NODES:
        name = nd.loc[nd.node_idx == node, "name"].values[0]
        max_excess_seen = -np.inf
        n_would_trigger = 0
        n_checked = 0

        for gradient_max in [GRADIENT_MAX_RANGE[0], 2.5, GRADIENT_MAX_RANGE[1]]:
            gradient = gradient_for(gradient_max)
            g = float(gradient[node])
            if g <= 1.05:
                continue   # not even eligible at this gradient_max, skip

            for t0 in starts:
                X_w = X[t0 : t0 + T_WINDOW]
                if X_w.shape[0] < T_WINDOW:
                    continue
                rain_slice = X_w[T_IN:T_IN+16, node, F_RAIN] * g
                rain_excess = float(np.sum(rain_slice))
                sat = float(np.mean(X_w[:T_IN, node, F_SW2_SAT]))
                excess = rain_excess * sat

                n_checked += 1
                max_excess_seen = max(max_excess_seen, excess)
                if excess >= 5.0:
                    n_would_trigger += 1

        frac = n_would_trigger / n_checked if n_checked else float("nan")
        print(f"Node {node} ({name}):")
        print(f"  windows checked (across eligible gradient_max draws): {n_checked}")
        print(f"  max excess ever observed: {max_excess_seen:.3f}  (threshold=5.0)")
        print(f"  fraction of checked (window, gradient_max) pairs that would trigger: {frac:.4%}")
        if n_would_trigger == 0:
            print(f"  -> NEVER triggers in {n_checked} real windows checked -- "
                  f"safe to re-include as a downstream node despite g>1.05.")
        else:
            print(f"  -> DOES trigger in real data ({n_would_trigger} cases) -- "
                  f"correctly excluded, the conservative fix was not overly strict here.")
        print()


if __name__ == "__main__":
    main()
