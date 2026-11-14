"""pytest bootstrap -- ensure the repo root is importable as `backend.app...`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
