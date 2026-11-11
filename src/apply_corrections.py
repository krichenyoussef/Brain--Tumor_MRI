"""
Step 6 - Merge manual corrections into the auto boxes.

labels/corrections.csv has one row per corrected image:
    split,cls,file,x1,y1,x2,y2,note
  - x1..y2 in pixels of the original image (replace the auto box)
  - leave x1..y2 empty to mark "no usable box" (image is dropped from the reviewed set)
Images that were reviewed and found correct go in labels/reviewed_ok.csv (split,cls,file).

Produces:
  labels/final_boxes.csv        auto boxes with corrections applied + a 'status' column:
                                  auto / corrected / verified / dropped
  labels/final/<split>/<cls>/   YOLO label files for the final boxes
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labels"
YOLO_ID = {"glioma": 0, "meningioma": 1, "pituitary": 2}

SRC = LAB / "auto_boxes_v2.csv" if (LAB / "auto_boxes_v2.csv").exists() else LAB / "auto_boxes.csv"
print("auto boxes from", SRC.name)
auto = list(csv.DictReader(open(SRC)))
corr = {}
if (LAB / "corrections.csv").exists():
    for r in csv.DictReader(open(LAB / "corrections.csv")):
        corr[(r["split"], r["cls"], r["file"])] = r
ok = set()
if (LAB / "reviewed_ok.csv").exists():
    for r in csv.DictReader(open(LAB / "reviewed_ok.csv")):
        ok.add((r["split"], r["cls"], r["file"]))

out_rows = []
counts = {"auto": 0, "corrected": 0, "verified": 0, "dropped": 0, "background": 0}
for r in auto:
    key = (r["split"], r["cls"], r["file"])
    r = dict(r)
    if r["cls"] not in YOLO_ID:
        r["status"] = "background"
    elif key in corr:
        c = corr[key]
        if c["x1"].strip():
            r.update(x1=c["x1"], y1=c["y1"], x2=c["x2"], y2=c["y2"]); r["status"] = "corrected"
        else:
            r.update(x1="", y1="", x2="", y2=""); r["status"] = "dropped"
    elif key in ok:
        r["status"] = "verified"
    else:
        r["status"] = "auto"
    counts[r["status"]] += 1
    out_rows.append(r)

    d = LAB / "final" / r["split"] / r["cls"]; d.mkdir(parents=True, exist_ok=True)
    p = d / (Path(r["file"]).stem + ".txt")
    if r["status"] in ("background", "dropped") or not r["x1"]:
        p.write_text("")
    else:
        W, H = int(r["W"]), int(r["H"]); x1, y1, x2, y2 = map(float, (r["x1"], r["y1"], r["x2"], r["y2"]))
        p.write_text(f"{YOLO_ID[r['cls']]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")

with open(LAB / "final_boxes.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
print(counts)
