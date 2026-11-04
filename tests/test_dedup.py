import csv
import json
from collections import defaultdict

from src import dedup


def test_dedup_removes_leaks_and_conflicts(raw_dataset, tmp_path):
    out, report_path = tmp_path / "splits.csv", tmp_path / "report.json"
    report = dedup.main(["--raw", str(raw_dataset), "--out", str(out), "--report", str(report_path)])

    assert report["images_found"] == 4 * 14 + 3
    assert report["duplicate_groups"] == 3
    assert report["groups_spanning_original_train_and_test"] == 2
    assert report["groups_with_conflicting_labels_dropped"] == 1
    # 3 planted copies removed, plus the conflicting scan kept nowhere
    assert report["images_kept"] == 4 * 14 - 1
    assert json.loads(report_path.read_text()) == report

    rows = list(csv.DictReader(out.open()))
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths))
    assert not any("conflict" in p or "Tr_meningioma_2" in p for p in paths)

    # every class appears in every split
    seen = defaultdict(set)
    for r in rows:
        seen[r["split"]].add(r["label"])
    assert all(len(v) == 4 for v in seen.values())
