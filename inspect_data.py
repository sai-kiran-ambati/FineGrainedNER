from pathlib import Path

DATA_DIR = Path("anonymous_telner")

for path in DATA_DIR.rglob("*"):
    if path.is_file():
        print(path)