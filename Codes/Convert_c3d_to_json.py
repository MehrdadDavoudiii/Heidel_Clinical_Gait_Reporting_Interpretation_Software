
import json
from pathlib import Path
import numpy as np
import ezc3d

# -------------------- Paths --------------------
HERE = Path(__file__).parent
C3D_FILE = HERE / "Heidel_file.c3d"
if not C3D_FILE.exists():
    raise SystemExit(f"File not found: {C3D_FILE}")

# -------------------- Step1 --------------------
def safe_number(v):
    """Convert value to float or None if invalid/NaN."""
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except Exception:
        return None

def detect_unit(label: str) -> str:
    """Decide unit based on label type."""
    ulabel = label.upper()
    if "ANGLE" in ulabel:
        return "deg"
    if "MOMENT" in ulabel:
        return "Nm/kg"
    if "POWER" in ulabel:
        return "W/kg"
    return ""

def read_sampling_rate(c3d) -> float:
    """Try to extract sampling rate, fallback to 120 Hz."""
    try:
        r = float(c3d["header"]["points"]["frame_rate"])
        if r > 0: return r
    except Exception:
        pass
    try:
        r = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
        if r > 0: return r
    except Exception:
        pass
    return 120.0

def read_labels(c3d) -> list[str]:
    """Get point labels as strings, ensure length matches data."""
    return [str(x) for x in c3d["parameters"]["POINT"]["LABELS"]["value"]]

def build_time_vector(n_frames: int, freq: float):
    """Return vector of time stamps starting at 1/freq."""
    step = 1.0 / freq
    return [(i + 1) * step for i in range(n_frames)]

# -------------------- Core Conversions --------------------
def make_entry(label: str, values: np.ndarray, times: list[float]):
    """
    Build one entry for JSON export.
    values: (frames, 3) = sagittal, frontal, transverse
    """
    series = []
    for t, (sag, fr, tr) in zip(times, values):
        series.append({
            "sagittal":   safe_number(sag),
            "frontal":    safe_number(fr),
            "transverse": safe_number(tr),
            "time": float(t),
        })
    return {"label": label, "values": series, "unit": detect_unit(label)}

def load_c3d_signals(c3d_path: Path):
    """
    Load C3D and return:
      - freq: sampling rate
      - labels: channel names
      - signal: (frames, labels, 3) sagittal/frontal/transverse
    """
    c3d = ezc3d.c3d(str(c3d_path))
    pts = c3d["data"]["points"]  # (4, labels, frames)
    if pts.shape[0] < 3:
        raise RuntimeError(f"Unexpected points shape {pts.shape}")

    # reorder to (frames, labels, planes)
    signal = np.transpose(pts[0:3, :, :], (2, 1, 0))
    labels = read_labels(c3d)
    # fix mismatch between labels and signal
    if len(labels) < signal.shape[1]:
        labels += [f"Label_{i}" for i in range(len(labels), signal.shape[1])]
    labels = labels[:signal.shape[1]]

    return read_sampling_rate(c3d), labels, signal

def export_jsons(freq: float, labels: list[str], signal: np.ndarray, out_dir: Path):
    """
    Export JSONs for angles, moments, powers.
    signal: (frames, labels, 3)
    """
    times = build_time_vector(signal.shape[0], freq)
    angles, moments, powers = {"data": []}, {"data": []}, {"data": []}

    for lbl, data in zip(labels, signal.transpose(1,0,2)):
        ulabel = lbl.upper()
        entry = make_entry(lbl, data, times)

        if "ANGLE" in ulabel:
            angles["data"].append(entry)
        elif "MOMENT" in ulabel and "GROUNDREACTION" not in ulabel:
            moments["data"].append(entry)
        elif "POWER" in ulabel:
            powers["data"].append(entry)
        # ignore other labels

    out_dir.mkdir(exist_ok=True)
    (out_dir / "angles.json").write_text(json.dumps(angles, indent=2), encoding="utf-8")
    (out_dir / "moments.json").write_text(json.dumps(moments, indent=2), encoding="utf-8")
    (out_dir / "powers.json").write_text(json.dumps(powers, indent=2), encoding="utf-8")

# -------------------- Main --------------------
def main():
    freq, labels, signal = load_c3d_signals(C3D_FILE)
    export_jsons(freq, labels, signal, HERE)
    print(f"Exported JSONs: {len(labels)} labels, {signal.shape[0]} frames, {freq:.1f} Hz")

if __name__ == "__main__":
    main()
