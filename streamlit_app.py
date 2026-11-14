"""Optional conventional Streamlit Cloud entrypoint; primary frontend is streamlet.py."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("streamlet.py")), run_name="__main__")
