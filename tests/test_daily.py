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
