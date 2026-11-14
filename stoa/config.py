"""All paths / model names / prices come from env vars with defaults here. No hardcoding elsewhere."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get(
    "STOA_DATASET_DIR", "~/Downloads/stoa-junction/preview/junction-dataset-preview")).expanduser()
# inside the HJ repo: everything generated goes to out/ (gitignored), weights to runs/ (gitignored)
CACHE_DIR = Path(os.environ.get("STOA_CACHE_DIR", ROOT / "out" / "stoa_cache"))
REPORTS_DIR = Path(os.environ.get("STOA_REPORTS_DIR", ROOT / "out" / "stoa_reports"))

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "-1"))  # -1 = model default
GEMINI_MEDIA_RESOLUTION = os.environ.get("GEMINI_MEDIA_RESOLUTION", "")  # "", "low", "medium", "high"

# USD per 1M tokens (input, output incl. thinking). Verify against current price page before reporting.
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3-pro-preview": (2.00, 12.0),
}

CUBICASA_REPO = Path(os.environ.get("CUBICASA_REPO", ROOT / "third_party" / "CubiCasa5k"))
CUBICASA_WEIGHTS = Path(os.environ.get("CUBICASA_WEIGHTS", ROOT / "runs" / "cubicasa" / "model_best_val_loss_var.pkl"))
CUBICASA_TEST_LIST = Path(os.environ.get("CUBICASA_TEST_LIST", ROOT / "eval" / "cubicasa_test.txt"))  # from CubiCasa5k zip; for honest (non-leaky) accuracy

# --- Claude (VLM) ---
def _load_env_file():
    for name in (".env", "env.md"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env_file()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("STOA_CLAUDE_EFFORT", "low")  # own name: Claude Code sets CLAUDE_EFFORT in its shell
CLAUDE_BOXES = os.environ.get("CLAUDE_BOXES", "1") == "1"  # ask for evidence boxes (costs output tokens)
CLAUDE_PRICES = {  # USD per 1M tokens (input, output)
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
