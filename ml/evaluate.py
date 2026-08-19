import json
from pathlib import Path

from app.core.config import BASE_DIR

def main():
    path = BASE_DIR / "models" / "metadata.json"
    if not path.exists():
        raise SystemExit("No metadata found. Train the model first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
