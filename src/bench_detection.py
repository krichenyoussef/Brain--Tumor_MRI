"""
Benchmark - score every box method on the same human-reviewed test set.

Test set: yolo_full/images/val = 48 reviewed Testing tumor images + 20 no-tumor Testing images.
Ground truth: their YOLO label files (reviewed boxes).

Methods (all must predict the class themselves, like a real detector):
  cam_v1     ResNet50 classifier -> if predicted tumor, Grad-CAM box (step 3 logic) for the predicted class
  cam_v2     same, with the brightness refinement (step 4 logic)
  yolo_gold  YOLO11n trained on the 199 reviewed training images   (last.pt)
  yolo_full  YOLO11n trained on the 5,559 mostly automatic images  (last.pt)
Score = classifier probability (CAM methods) or YOLO confidence.

Metrics:
  mAP50, mAP50-95 (COCO-style 101-point AP, averaged over the 3 classes)
  class-agnostic AP50 (is the box in the right place, whatever the class)
  image-level: share of tumor images with a correct box (IoU>=0.5, right class) at conf>=0.25,
               and false-alarm rate on no-tumor images at conf>=0.25
  95% bootstrap CI for mAP50 (1,000 resamples of images)
  speed: ms per image (end to end, batch 1)
Writes outputs/bench_detection.json and outputs/bench_detection.txt
"""
import json, sys, time
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gradcam_boxes as g1      # noqa: E402  (box logic of step 3)
import refine_boxes as g2       # noqa: E402  (box logic of step 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL = ROOT / "yolo_full" / "images" / "val"
LBL = ROOT / "yolo_full" / "labels" / "val"
NAMES = ["glioma", "meningioma", "pituitary"]
CLS2Y = {"glioma": 0, "meningioma": 1, "pituitary": 2}
images = sorted(VAL.glob("*.jpg"))


def load_gt():
    gt = {}
    for im in images:
        W, H = Image.open(im).size; boxes = []
        for line in (LBL / f"{im.stem}.txt").read_text().splitlines():
            if line.strip():
                c, cx, cy, w, h = line.split(); cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
                boxes.append((int(c), (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        gt[im.name] = boxes
    return gt


# ---------------- CAM-based detectors ----------------
ck = torch.load(ROOT / "outputs" / "model.pt", map_location=device)
classes = ck["classes"]
clf = models.resnet50(weights=None); clf.fc = nn.Linear(clf.fc.in_features, len(classes))
clf.load_state_dict(ck["state_dict"]); clf.to(device).eval()
acts, grads = {}, {}
clf.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
clf.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cam_predict(version):
    preds, times = {}, []
    for im in images:
        t = time.perf_counter()
        pil = Image.open(im).convert("RGB"); W, H = pil.size
        out = clf(tf(pil)[None].to(device)); p = F.softmax(out.detach(), 1)[0]
        ci = int(p.argmax()); cname = classes[ci]
        if cname not in CLS2Y:                       # predicted no tumor -> no box
            preds[im.name] = []; times.append(time.perf_counter() - t); continue
        clf.zero_grad(set_to_none=True); out[0, ci].backward()
        a, g = acts["a"], grads["g"]
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))[0].detach().float().cpu().numpy()
        cam = np.clip(cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC), 0, None)
        gray = np.array(pil.convert("L"))
        if version == 1:
            box = g1.cam_to_box(cam, g1.brain_mask(gray), g1.CAM_THRESH)
        else:
            m = g2.brain_mask(gray); box = g2.refine(gray, cam, m) or g2.cam_box(cam, m)
        preds[im.name] = [(CLS2Y[cname], float(p[ci]), *map(float, box))] if box else []
        times.append(time.perf_counter() - t)
    return preds, 1000 * float(np.median(times))


def yolo_predict(mode):
    from ultralytics import YOLO
    w = ROOT / "outputs" / "yolo" / mode / "weights" / "last.pt"
    model = YOLO(str(w)); model.predict(str(images[0]), imgsz=512, verbose=False)   # warm-up
    preds, times = {}, []
    for im in images:
        t = time.perf_counter()
        r = model.predict(str(im), imgsz=512, conf=0.001, max_det=10, verbose=False)[0]
        times.append(time.perf_counter() - t)
        b = r.boxes
        preds[im.name] = [(int(c), float(s), *map(float, xyxy)) for c, s, xyxy in
                          zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    return preds, 1000 * float(np.median(times))


# ---------------- metrics ----------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def ap_one(preds, gt, names, cls, thr, agnostic=False):
    dets, npos = [], 0
    for n in names:
        gb = [g[1:] for g in gt[n] if agnostic or g[0] == cls]; npos += len(gb)
        for p in preds[n]:
            if agnostic or p[0] == cls:
                dets.append((p[1], n, p[2:]))
    if npos == 0:
        return None
    dets.sort(key=lambda d: -d[0]); used = {n: [False] * len([g for g in gt[n] if agnostic or g[0] == cls]) for n in names}
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for k, (s, n, b) in enumerate(dets):
        gb = [g[1:] for g in gt[n] if agnostic or g[0] == cls]
        ious = [iou(b, x) for x in gb]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= thr and not used[n][j]:
            tp[k] = 1; used[n][j] = True
        else:
            fp[k] = 1
    if len(dets) == 0:
        return 0.0
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / npos; prec = tpc / np.maximum(tpc + fpc, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in rs]))


def maps(preds, gt, names):
    def m(thr):
        v = [ap_one(preds, gt, names, c, thr) for c in range(3)]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else 0.0
    per_cls = {NAMES[c]: ap_one(preds, gt, names, c, 0.5) for c in range(3)}
    return dict(mAP50=m(0.5), mAP50_95=float(np.mean([m(t) for t in np.arange(0.5, 0.96, 0.05)])),
                agnostic_AP50=ap_one(preds, gt, names, 0, 0.5, agnostic=True), AP50_per_class=per_cls)


def image_level(preds, gt, conf=0.25):
    hit, tum, fa, bg = 0, 0, 0, 0
    for n, gb in gt.items():
        ps = [p for p in preds[n] if p[1] >= conf]
        if gb:
            tum += 1
            hit += any(p[0] == g[0] and iou(p[2:], g[1:]) >= 0.5 for p in ps for g in gb)
        else:
            bg += 1; fa += bool(ps)
    return dict(tumor_images_correct=hit / tum, false_alarm_rate=fa / bg)


def main():
    gt = load_gt(); names = list(gt)
    rng = np.random.default_rng(0)
    methods = {"cam_v1": lambda: cam_predict(1), "cam_v2": lambda: cam_predict(2),
               "yolo_gold": lambda: yolo_predict("gold"), "yolo_full": lambda: yolo_predict("full")}
    res = {}
    for name, fn in methods.items():
        preds, ms = fn()
        r = maps(preds, gt, names); r.update(image_level(preds, gt)); r["ms_per_image"] = ms
        boots = [maps(preds, gt, list(rng.choice(names, len(names))))["mAP50"] for _ in range(1000)]
        r["mAP50_CI95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        res[name] = r; print(name, json.dumps(r), flush=True)
    json.dump(res, open(ROOT / "outputs" / "bench_detection.json", "w"), indent=2)
    lines = [f"test set: {len(names)} images, {sum(len(v) for v in gt.values())} reviewed boxes",
             f"{'method':<10} {'mAP50':>6} {'95% CI':>13} {'mAP50-95':>8} {'agn.AP50':>8} {'img ok':>6} {'false al.':>9} {'ms/img':>7}"]
    for k, r in res.items():
        lines.append(f"{k:<10} {r['mAP50']:>6.3f} [{r['mAP50_CI95'][0]:.2f}, {r['mAP50_CI95'][1]:.2f}] {r['mAP50_95']:>8.3f} "
                     f"{r['agnostic_AP50']:>8.3f} {r['tumor_images_correct']:>6.1%} {r['false_alarm_rate']:>9.1%} {r['ms_per_image']:>7.1f}")
    txt = "\n".join(lines); print(txt); (ROOT / "outputs" / "bench_detection.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
