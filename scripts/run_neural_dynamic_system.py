from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_dynamic_system.app import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
