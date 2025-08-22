
import os
import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
CODES = BASE / "Codes"

CONVERT = CODES / "Convert_c3d_to_json.py"
REPORT  = CODES / "Make_report.py"
INTERP  = CODES / "Auto_interpret_report.py" 

def must_exist(p: Path):
    if not p.exists():
        raise SystemExit(f" Missing: {p}")

def run(cmd, *, cwd: Path):
    print(f"\n→ (cwd={cwd}) {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))

if __name__ == "__main__":
    must_exist(CODES)
    must_exist(CONVERT)
    must_exist(REPORT)
    must_exist(INTERP)

    # 1) C3D → JSON  (run inside Codes)
    print("Step 1: Converting C3D to JSON …")
    run([sys.executable, CONVERT.name], cwd=CODES)

    # 2) Build Word report  (inside Codes)
    print("\nStep 2: Creating the Word report …")
    run([sys.executable, REPORT.name], cwd=CODES)

    # 3) Append AI interpretation → PDF  (inside Codes)
    print("\nStep 3 (optional): Append interpretation & export PDF.")
    api_key = input("Enter your OPENAI_API_KEY (leave empty to skip this step): ").strip()
    if api_key:
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key
        print(f"\n→ (cwd={CODES}) {sys.executable} {INTERP.name} *****")
        subprocess.run(
            [sys.executable, INTERP.name, api_key],
            check=True,
            cwd=str(CODES),
            env=env,
        )
        print("\n All steps finished. Final PDF should be in the Codes folder.")
    else:
        print("Skipped AI interpretation step (no API key provided). Steps 1–2 completed.")
