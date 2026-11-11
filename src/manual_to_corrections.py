"""
Step 6 helper - Convert the manual review file into corrections.csv + reviewed_ok.csv.

Input  labels/manual_review.csv: split,cls,file,verdict,x1p,y1p,x2p,y2p,note
   verdict = ok   -> auto box accepted as-is
             fix  -> replace with x1p..y2p (percent of width/height, 0-100)
             drop -> no reliable box (tumor not visible to the reviewer)
Output labels/corrections.csv (pixels) and labels/reviewed_ok.csv, read by apply_corrections.py
"""
import csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labels"
rows = list(csv.DictReader(open(LAB / "manual_review.csv")))
corr, ok = [], []
for r in rows:
    key = [r["split"], r["cls"], r["file"]]
    v = r["verdict"].strip()
    if v == "ok":
        ok.append(key)
    elif v == "fix":
        W, H = Image.open(ROOT / "data" / r["split"] / r["cls"] / r["file"]).size
        x1, y1, x2, y2 = (float(r[k]) / 100 for k in ("x1p", "y1p", "x2p", "y2p"))
        corr.append(key + [round(x1 * W), round(y1 * H), round(x2 * W), round(y2 * H), r.get("note", "")])
    elif v == "drop":
        corr.append(key + ["", "", "", "", r.get("note", "")])
    else:
        raise ValueError(f"bad verdict {v!r} for {key}")
with open(LAB / "corrections.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "cls", "file", "x1", "y1", "x2", "y2", "note"]); w.writerows(corr)
with open(LAB / "reviewed_ok.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "cls", "file"]); w.writerows(ok)
print(f"ok={len(ok)} fix={sum(1 for c in corr if c[3] != '')} drop={sum(1 for c in corr if c[3] == '')}")
