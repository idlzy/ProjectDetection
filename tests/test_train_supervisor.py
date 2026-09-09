import os
import subprocess
from pathlib import Path


def test_supervisor_records_signal_and_restarts_with_auto_resume(tmp_path):
    fake_python = tmp_path / "fake_python.sh"
    marker = tmp_path / "first_attempt_finished"
    auto_resume_seen = tmp_path / "auto_resume_seen"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ ! -e \"$FAKE_MARKER\" ]; then\n"
        "  touch \"$FAKE_MARKER\"\n"
        "  exit 139\n"
        "fi\n"
        "for argument in \"$@\"; do\n"
        "  [ \"$argument\" = --auto-resume ] && touch \"$AUTO_RESUME_SEEN\"\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        TRAIN_RUNTIME_DIR=str(tmp_path / "runtime"),
        TRAIN_MAX_RESTARTS="1",
        TRAIN_RESTART_DELAY="0",
        FAKE_MARKER=str(marker),
        AUTO_RESUME_SEEN=str(auto_resume_seen),
    )
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [str(project_root / "scripts/train_supervisor.sh"), str(fake_python)],
        cwd=str(project_root),
        env=environment,
        check=True,
    )

    status = (tmp_path / "runtime/DetectionTrain.status").read_text(encoding="utf-8")
    events = (tmp_path / "runtime/DetectionTrain.events.log").read_text(encoding="utf-8")
    assert "state=completed" in status
    assert "restart_count=1" in status
    assert "state=crashed exit_code=139 signal=SEGV" in events
    assert auto_resume_seen.is_file()
