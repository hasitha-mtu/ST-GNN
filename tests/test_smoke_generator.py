def test_all_scenarios_generated(generated):
    out, _ = generated
    import json
    for s in ["S1_ConvectiveCell","S2_GaugeFailure","S3_InniscarraRelease",
              "S4_SatBreakthrough","S5_SpatialGradient","S6_ChannelBlockage"]:
        m = json.load(open(out / s / "scenario_meta.json"))
        assert m["n_windows"] > 0, s
