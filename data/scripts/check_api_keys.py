"""Pre-flight check: verify API keys are loadable before fetch/ingest commands."""
import sys

sys.path.insert(0, "src")
from solarpipe_data.config import get_settings

s = get_settings()
missing = []
if s.nasa_api_key == "DEMO_KEY":
    missing.append("NASA_API_KEY")
if not s.nasa_api_email:
    missing.append("NASA_API_EMAIL")
if not s.jsoc_email:
    missing.append("JSOC_EMAIL")

if missing:
    print(f"BLOCKED: Missing API keys: {missing}. Check data/.env")
    sys.exit(1)
