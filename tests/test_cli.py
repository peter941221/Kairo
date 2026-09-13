import json

from kairo_lab import cli


def test_environment_has_reproducibility_fields():
    data = cli.environment()
    assert data["timestamp_utc"]
    assert "python" in data


def test_init_run_writes_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = cli.init_run("decode")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["lane"] == "decode"
    assert data["result_contract"]["correctness_status"] == "required_before_performance_claim"
