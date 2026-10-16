"""End-to-end: random-weight GRU and static ST-GNN checkpoints evaluated by
the real scenario_evaluator CLI on the generated mock scenarios."""
import json

import numpy as np
import pandas as pd
import torch


def _make_ckpt(root, tag, seed, t_out):
    import run_inference as ri
    cls = ri.import_class(ri.MODEL_REGISTRY[tag])
    if tag == "gru":
        m = cls(f_dyn=11, f_static=7, hidden=16, gru_layers=1, t_out=t_out, dropout=0.0)
        hp = {"hidden": 16, "gru_layers": 1}
    else:
        m = cls(f_dyn=11, f_static=7, hidden=16, t_out=t_out, dropout=0.0,
                f_edge=4, gat_heads=1, gru_layers=1, sar_emb_dim=0)
        hp = {"hidden": 16, "gru_layers": 1, "gat_heads": 1}
    d = root / "checkpoints" / tag / str(seed) / str(t_out)
    d.mkdir(parents=True)
    torch.save({"state_dict": m.state_dict(), "hparams": {**hp, "f_dyn": 11, "t_in": 32,
                                                          "dropout": 0.0}}, d / "best_model.pt")
    json.dump({"rmse": 0.05, "nse": 0.9}, open(d / "test_metrics.json", "w"))


def test_cli_end_to_end(generated, monkeypatch, tmp_path):
    out, data = generated
    root = out.parent.parent
    import run_inference as ri
    from scenarios import scenario_evaluator as se
    for tag in ("gru", "st_gnn"):
        _make_ckpt(root, tag, 42, 12)
    monkeypatch.setattr(ri, "CKPT_ROOT", root / "checkpoints")
    monkeypatch.setattr(se, "CKPT_ROOT", root / "checkpoints")
    monkeypatch.setattr(se, "PROC_DIR", root / "dataset/processed")
    monkeypatch.setattr(se, "GRAPH_DIR", root / "dataset/graph")
    res = tmp_path / "results"
    se.main(["--all-scenarios", "--scen-root", str(out), "--results-root", str(res),
             "--save-predictions"])
    df = pd.read_csv(res / "scenario_summary.csv")
    assert set(df.model) == {"gru", "st_gnn"} and df.scenario.nunique() == 6
    assert df["observable_nse_syn"].notna().any()
    assert (df["degradation_ratio"] == (df["rmse_node_mean"] / 0.05).round(3)).all()
    s2 = df[df.scenario == "S2_GaugeFailure"]
    assert s2["s2_rmse_failed_nodes"].notna().all()
    per = pd.read_csv(res / "scenario_summary_per_realization.csv")
    assert {"all", "observable"} <= set(per.scope)
    npz = np.load(next((res / "S3_InniscarraRelease/predictions").glob("*.npz")))
    assert npz["pred_last"].shape == npz["target_last"].shape
    # second call hits the cache (no re-evaluation, same rows)
    se.main(["--all-scenarios", "--scen-root", str(out), "--results-root", str(res)])
    assert len(pd.read_csv(res / "scenario_summary.csv")) == len(df)
