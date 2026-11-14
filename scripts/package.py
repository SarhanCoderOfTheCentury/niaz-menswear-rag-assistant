"""Create an LMS ZIP under 5 MB from tracked source files only."""
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
dest = root / "dist" / "Niaz-Knowledge-Assistant.zip"
dest.parent.mkdir(exist_ok=True)
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name in tracked:
        if name and (root / name).is_file():
            archive.write(root / name, "Niaz-Knowledge-Assistant/" + name)
if dest.stat().st_size >= 5_000_000:
    raise SystemExit("ZIP exceeds the assignment's 5 MB limit.")
print(f"{dest.name}: {dest.stat().st_size:,} bytes")
