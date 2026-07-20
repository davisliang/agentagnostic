"""Where the repo keeps its non-code assets.

`config/`, `prompts/` and `skills/` are edited far more often than the code, so
they live at the repo root rather than inside the package. That means the
package has to find the root: it is two levels up from this file, which holds
for a source checkout and for an editable install (`uv sync`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
SKILLS_DIR = ROOT / "skills"
