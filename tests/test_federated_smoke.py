import json
import subprocess
import sys
from pathlib import Path


def test_federated_smoke_dry_run(tmp_path):
    script = Path("tools/federated_smoke.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--fed-root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert Path(payload["env_overrides"]["FED_ROOT"]) == tmp_path
    assert "run.py" in payload["command"]
