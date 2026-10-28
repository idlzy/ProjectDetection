import json
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_test_viz_and_eval.sh"


@pytest.mark.parametrize("split_args,env_split,expected", [
    ([], None, "test"),
    (["--split", "val"], None, "val"),
    (["--split=train"], "val", "train"),
])
@pytest.mark.parametrize("backend", ["pytorch", "onnx", "hat-onnx"])
def test_wrapper_forwards_split_and_sample_limit(tmp_path, split_args, env_split, expected, backend):
    python_stub = tmp_path / "python_stub"
    python_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
        "    sys.exit(0)\n"
        "with open(os.environ['CALL_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "best.pth"
    config.touch()
    checkpoint.touch()
    log = tmp_path / "calls.jsonl"
    pgd = tmp_path / "model.pgd.json"
    pgd.write_text("{}")
    env = os.environ.copy()
    env.update({
        "TEST_PYTHON": str(python_stub),
        "CALL_LOG": str(log),
        "CONFIG": str(config),
        "CHECKPOINT": str(checkpoint),
        "RUN_DIR": str(tmp_path / "run"),
        "EVAL_MAX_SAMPLES": "8",
        "BACKEND": backend,
        "PGD_JSON": str(pgd) if backend == "hat-onnx" else "",
    })
    if env_split is None:
        env.pop("SPLIT", None)
    else:
        env["SPLIT"] = env_split
    result = subprocess.run(
        ["bash", str(SCRIPT)] + split_args,
        env=env, capture_output=True, text=True, check=True,
    )
    assert "split=%s" % expected in result.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0][0] == "tools/visualize_test.py"
    assert calls[1][0] == "tools/test.py"
    for call in calls:
        assert call[call.index("--split") + 1] == expected
        assert call[call.index("--backend") + 1] == backend
        if backend == "hat-onnx":
            assert call[call.index("--pgd-json") + 1] == str(pgd)
    if backend != "pytorch":
        assert "--cop-stats" not in calls[1]
        assert "--cop-ablation" not in calls[1]
    assert "runtime.max_%s_samples=8" % expected in calls[1]


def test_wrapper_rejects_invalid_split_before_writing_outputs(tmp_path):
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--split", "invalid"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "split must be train, val or test" in result.stderr
    assert not run_dir.exists()
