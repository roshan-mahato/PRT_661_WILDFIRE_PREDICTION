"""The scheduled `pipeline.py daily` job: order of steps and exit codes."""

import types

import pytest

from Scripts import pipeline


@pytest.fixture
def daily(monkeypatch, tmp_path):
    """Runs cmd_daily with the fetch, prune and server check replaced. Returns what happened."""
    calls = {"fetch": 0, "prune": [], "server": True, "fetch_rc": 0}
    monkeypatch.setattr(pipeline, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "ensure_server", lambda url, wait, log: calls["server"])

    def fake_fetch(cmd, log):
        calls["fetch"] += 1
        log("fake fetch output")
        return calls["fetch_rc"]

    monkeypatch.setattr(pipeline, "run_logged", fake_fetch)
    monkeypatch.setattr(pipeline, "prune_forecasts",
                        lambda keep_days, log: calls["prune"].append(keep_days))

    def run():
        with pytest.raises(SystemExit) as exc:
            pipeline.cmd_daily(types.SimpleNamespace(keep_days=30, wait=0))
        calls["exit"] = exc.value.code
        calls["log"] = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("daily_*.log"))
        return calls

    calls["run"] = run
    return calls


def test_fetch_then_prune(daily):
    out = daily["run"]()
    assert out["exit"] == 0 and out["fetch"] == 1 and out["prune"] == [30]
    assert "fake fetch output" in out["log"]


def test_failed_fetch_still_prunes_and_reports(daily):
    daily["fetch_rc"] = 1
    out = daily["run"]()
    assert out["exit"] == 1 and out["prune"] == [30]
    assert "not refreshed" in out["log"]


def test_no_server_stops_before_fetch(daily):
    daily["server"] = False
    out = daily["run"]()
    assert out["exit"] == 2 and out["fetch"] == 0 and out["prune"] == []
    assert "aborted" in out["log"]


@pytest.fixture
def setup_run(monkeypatch, tmp_path):
    """cmd_setup with init, history load, the archive check and daily replaced."""
    calls = {"init": 0, "load": [], "daily": 0, "history": {}}
    mask = tmp_path / "mask.csv"
    mask.write_text("lat_round,lon_round,is_land\n-33.0,151.0,1\n")
    csv = tmp_path / "history.csv"
    csv.write_text("lat_round,lon_round,acq_date\n")
    monkeypatch.setattr(pipeline, "LAND_MASK_CSV", str(mask))
    monkeypatch.setattr(pipeline, "init_db", lambda: calls.__setitem__("init", calls["init"] + 1))
    monkeypatch.setattr(pipeline, "cmd_load_history", lambda a: calls["load"].append(a.csv))
    monkeypatch.setattr(pipeline.weather, "archive_last_dates", lambda: calls["history"])
    monkeypatch.setattr(pipeline, "cmd_daily", lambda a: calls.__setitem__("daily", calls["daily"] + 1))
    calls["run"] = lambda: pipeline.cmd_setup(types.SimpleNamespace(csv=str(csv), keep_days=90, wait=0))
    calls["csv"] = csv
    return calls


def test_setup_loads_history_only_into_an_empty_database(setup_run):
    setup_run["run"]()
    assert setup_run["load"] == [str(setup_run["csv"])] and setup_run["daily"] == 1

    import pandas as pd
    setup_run["history"] = {(-33.0, 151.0): pd.Timestamp("2026-10-17")}
    setup_run["run"]()
    assert len(setup_run["load"]) == 1 and setup_run["daily"] == 2 and setup_run["init"] == 2


def test_setup_stops_on_git_lfs_pointer(setup_run):
    setup_run["csv"].write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(SystemExit, match="git lfs pull"):
        setup_run["run"]()
    assert setup_run["init"] == 0 and setup_run["daily"] == 0
