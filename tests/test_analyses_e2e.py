"""End-to-end runs of the new reviewer analyses on the mock project."""
import json

import numpy as np
import pandas as pd
import torch

from test_evaluator_e2e import _make_ckpt


def _soilgate_ckpt(root, seed, hz, theta_raw=3.0):
    d = root / "checkpoints/st_gnn_soil_gate" / str(seed) / str(hz)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"sat_threshold_raw": torch.tensor(theta_raw),
                               "sat_sharpness": torch.tensor(10.0),
                               "sat_bounds": torch.tensor([0.0, 1.0])}}, d / "_sg.pt")
    (d / "best_model.pt").write_bytes((d / "_sg.pt").read_bytes())


def test_sensitivity_mechanism_operational_panel(generated, monkeypatch, tmp_path):
    out, data = generated
    root = out.parent.parent
    import run_inference as ri
    from scenarios import scenario_evaluator as se, scenario_generator as sg
    from scenarios import generator_sensitivity as gs, mechanism_diagnostics as md
    from scenarios import plot_scenario_panel as pp
    import operational_metrics as om
    ck = root / "checkpoints"
    if not (ck / "gru/7/12").exists():
        _make_ckpt(root, "gru", 7, 12)
        _make_ckpt(root, "st_gnn", 7, 12)
    for mod, attrs in ((ri, {"CKPT_ROOT": ck}),
                       (se, {"CKPT_ROOT": ck, "PROC_DIR": root / "dataset/processed",
                             "GRAPH_DIR": root / "dataset/graph"}),
                       (sg, {"PROC_DIR": root / "dataset/processed",
                             "GRAPH_DIR": root / "dataset/graph"}),
                       (gs, {"SENS_SCEN": tmp_path / "sens_scen", "SENS_RES": tmp_path / "sens_res"})):
        for k, v in attrs.items():
            monkeypatch.setattr(mod, k, v)
    # --- generator sensitivity (S3 routing grid, 9 configs) -------------
    gs.main(["--scenarios", "S3", "--n-windows", "3"])
    agr = pd.read_csv(tmp_path / "sens_res/sensitivity_agreement.csv")
    assert len(agr) == 9
    base = agr[agr.config == "attenuation0.75_lag_scale1"]
    assert np.isclose(base.spearman_rho.iloc[0], 1.0)
    # configurations really differ in their targets
    y1 = np.load(tmp_path / "sens_scen/S3/attenuation0.65_lag_scale0.75/S3_InniscarraRelease/y_synthetic.npy")
    y2 = np.load(tmp_path / "sens_scen/S3/attenuation0.85_lag_scale1.25/S3_InniscarraRelease/y_synthetic.npy")
    assert not np.allclose(y1, y2)
    # --- mechanism diagnostics -----------------------------------------
    res = tmp_path / "results"
    se.main(["--all-scenarios", "--scen-root", str(out), "--results-root", str(res / "scenarios")])
    _soilgate_ckpt(root, 7, 12)
    monkeypatch.setattr(md, "CKPT_ROOT", ck); monkeypatch.setattr(md, "SCEN_DIR", out)
    monkeypatch.setattr(md, "RES", res); monkeypatch.setattr(md, "OUT", res / "mechanism")
    monkeypatch.setattr(md, "BASE_DIR", root)
    md.main()
    summ = pd.read_csv(res / "mechanism/soilgate_summary.csv")
    theta = 1 / (1 + np.exp(-3.0))
    assert np.isclose(summ.theta.iloc[0], theta, atol=1e-6)
    tl = pd.read_csv(res / "mechanism/soilgate_activation.csv")
    opened = tl[tl.gate >= 0.5]
    assert np.allclose(opened.sbar >= theta, True)               # explicit opening rule
    assert (res / "mechanism/mechanism_figure.png").exists()
    # --- operational table ----------------------------------------------
    monkeypatch.setattr(om, "CKPT", ck); monkeypatch.setattr(om, "OUT", res / "operational")
    monkeypatch.setattr(om, "BASE_DIR", root)
    for tag in ("gru", "st_gnn"):
        for hz in (4, 48):
            if not (ck / tag / "7" / str(hz)).exists():
                _make_ckpt(root, tag, 7, hz)
            json.dump({"pod": 0.5, "far": 0.2, "peak_timing_mean_hr": 1.0},
                      open(ck / tag / "7" / str(hz) / "extended_metrics.json", "w"))
    om.main()
    assert "| GRU | 1 h |" in (res / "operational/operational_table.md").read_text(encoding="utf-8")
    # --- six-panel figure -------------------------------------------------
    monkeypatch.setattr(pp, "SCEN_CSV", res / "scenarios/scenario_summary.csv")
    monkeypatch.setattr(pp, "OUT", res / "fig.png")
    monkeypatch.setattr("sys.argv", ["x"])
    pp.main()
    assert (res / "fig.png").exists()
