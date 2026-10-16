"""The runtime validator passes on a correctly built (mock) project and
detects the submitted pipeline's errors."""
import json
import numpy as np


def test_validator_on_mock(generated, monkeypatch):
    out, _ = generated
    root = out.parent.parent
    import validate_pipeline as vp
    monkeypatch.setattr(vp, "PROC", root / "dataset/processed")
    monkeypatch.setattr(vp, "GRAPH", root / "dataset/graph")
    vp.RESULTS.clear()
    for c in ("graph", "dataset", "uh", "lags"):
        vp.CHECKS[c]()
    vp.chk_scenarios(out)
    bad = [r for r in vp.RESULTS if r["status"] == "FAIL"]
    assert not bad, bad


def test_validator_detects_misaligned_mask(mock_project, monkeypatch, tmp_path):
    import shutil, validate_pipeline as vp
    p = tmp_path / "proc"; shutil.copytree(mock_project / "dataset/processed", p)
    iv = np.load(p / "input_valid_mask.npy"); np.save(p / "valid_mask.npy", iv)   # old convention
    monkeypatch.setattr(vp, "PROC", p); monkeypatch.setattr(vp, "GRAPH", mock_project / "dataset/graph")
    vp.RESULTS.clear(); vp.chk_dataset()
    assert any(r["check"] == "dataset.mask_alignment" and r["status"] == "FAIL" for r in vp.RESULTS)
