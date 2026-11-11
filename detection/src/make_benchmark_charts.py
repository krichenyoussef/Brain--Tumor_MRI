"""
Benchmark - charts + markdown tables from the benchmark result files.

Reads  outputs/bench_detection.json, outputs/cls_*/metrics.json, outputs/yolo/*/results.csv
Writes outputs/bench_detection.png, outputs/bench_classifiers.png, outputs/bench_yolo_curves.png,
       outputs/bench_tables.md
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (reference palette, light)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


det = json.load(open(OUT / "bench_detection.json"))
LABEL = {"cam_v1": "Grad-CAM v1", "cam_v2": "Grad-CAM v2 (refined)",
         "yolo_gold": "YOLO11n, 199 reviewed imgs", "yolo_full": "YOLO11n, 5,559 auto imgs"}

# ---- detection: mAP50 with 95% bootstrap CI (single series) ----
names = list(det)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ys = range(len(names))[::-1]
for y, k in zip(ys, names):
    r = det[k]; lo, hi = r["mAP50_CI95"]
    ax.barh(y, r["mAP50"], height=0.5, color=S1)
    ax.plot([lo, hi], [y, y], color=INK2, lw=1.5); ax.plot([lo, lo], [y - .12, y + .12], color=INK2, lw=1.5)
    ax.plot([hi, hi], [y - .12, y + .12], color=INK2, lw=1.5)
    ax.text(hi + 0.015, y, f"{r['mAP50']:.2f}", va="center", color=INK, fontsize=10)
ax.set_yticks(list(ys)); ax.set_yticklabels([LABEL[k] for k in names])
ax.set_xlim(0, 1.05); ax.set_xlabel("mAP@0.5 on the reviewed test set; line = 95% bootstrap interval")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); style(ax)
ax.set_title("Detection: how well each method boxes the tumor", loc="left", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "bench_detection.png", dpi=150); plt.close(fig)

# ---- classifiers: test accuracy vs glioma recall (2 series, legend + direct labels) ----
cls = []
for d in sorted(OUT.glob("cls_*")):
    if (d / "metrics.json").exists():
        cls.append(json.load(open(d / "metrics.json")))
cls.sort(key=lambda m: -m["test_acc"])
if cls:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.36; xs = range(len(cls))
    for off, key, col, lab in ((-w / 2 - 0.01, "test_acc", S1, "Test accuracy"), (w / 2 + 0.01, "glioma", S2, "Glioma recall")):
        vals = [m["test_acc"] if key == "test_acc" else m["recall"]["glioma"] for m in cls]
        ax.bar([x + off for x in xs], vals, width=w, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.006, f"{v:.1%}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{m['arch']}\n{m['params_m']}M params" for m in cls])
    lo = min(min(m["test_acc"], m["recall"]["glioma"]) for m in cls)
    ax.set_ylim(max(0, lo - 0.08), 1.02); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); style(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    ax.set_title("Classification on the 1,600 test images (y-axis does not start at 0)", loc="left", color=INK,
                 fontsize=11, pad=26)
    fig.tight_layout(); fig.savefig(OUT / "bench_classifiers.png", dpi=150); plt.close(fig)

# ---- YOLO training curves: val mAP50 per epoch (2 series) ----
fig, ax = plt.subplots(figsize=(7.2, 3.0)); any_curve = False
for mode, col, lab in (("gold", S1, "199 reviewed imgs"), ("full", S2, "5,559 auto imgs")):
    p = OUT / "yolo" / mode / "results.csv"
    if not p.exists():
        continue
    rows = list(csv.DictReader(open(p))); rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    ep = [int(float(r["epoch"])) for r in rows]; m = [float(r["metrics/mAP50(B)"]) for r in rows]
    ax.plot(ep, m, color=col, lw=2, label=lab)
    ax.text(ep[-1] + 1, m[-1] + 0.07, lab, color=INK2, va="center", fontsize=8.5)   # direct label above the line end
    any_curve = True
if any_curve:
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP@0.5"); ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID, lw=0.8); style(ax); ax.legend(frameon=False, loc="lower right")
    ax.set_title("YOLO11n validation mAP@0.5 during training", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "bench_yolo_curves.png", dpi=150)
plt.close(fig)

# ---- markdown tables ----
L = ["| Method | mAP@0.5 | 95% interval | mAP@0.5:0.95 | Box right place, any class (AP@0.5) | Tumor images boxed correctly | False alarms on no-tumor | ms / image |",
     "|---|---|---|---|---|---|---|---|"]
for k, r in det.items():
    L.append(f"| {LABEL[k]} | {r['mAP50']:.3f} | {r['mAP50_CI95'][0]:.2f} to {r['mAP50_CI95'][1]:.2f} | {r['mAP50_95']:.3f} | "
             f"{r['agnostic_AP50']:.3f} | {r['tumor_images_correct']:.0%} | {r['false_alarm_rate']:.0%} | {r['ms_per_image']:.0f} |")
L += ["", "| Method | Glioma AP@0.5 | Meningioma AP@0.5 | Pituitary AP@0.5 |", "|---|---|---|---|"]
for k, r in det.items():
    a = r["AP50_per_class"]; L.append(f"| {LABEL[k]} | {a['glioma']:.3f} | {a['meningioma']:.3f} | {a['pituitary']:.3f} |")
if cls:
    L += ["", "| Backbone | Params | Test acc | Macro F1 | Glioma | Meningioma | No tumor | Pituitary | Train time | Latency bs1 | Throughput bs32 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in cls:
        r = m["recall"]
        L.append(f"| {m['arch']} | {m['params_m']}M | {m['test_acc']:.1%} | {m['macro_f1']:.3f} | {r['glioma']:.1%} | {r['meningioma']:.1%} | "
                 f"{r['notumor']:.1%} | {r['pituitary']:.1%} | {m['train_time_s'] / 60:.0f} min | {m['latency_bs1_ms']:.1f} ms | {m['throughput_bs32_img_s']:.0f} img/s |")
(OUT / "bench_tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
