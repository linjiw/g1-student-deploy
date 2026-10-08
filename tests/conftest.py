import os
import sys
from pathlib import Path

LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", Path.home() / "lucid-sonic/lucid-g1-deploy"))
if str(LUCID / "sdk") not in sys.path:
    sys.path.append(str(LUCID / "sdk"))  # unitree_sdk2py (vendored by lucid-g1-deploy)
