import numpy as np
import cv2

from mkuav import qa


def _write_image(path, w=20, h=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_run_visdrone_rules(tmp_path):
    ann_dir = tmp_path / "annotations"
    img_dir = tmp_path / "images"
    ann_dir.mkdir()
    img_dir.mkdir()

    # 20x20 image for all stems
    for stem in ["a", "b", "c", "d", "e"]:
        _write_image(img_dir / f"{stem}.jpg")

    # a: zero_area box
    (ann_dir / "a.txt").write_text("1,1,0,5,1,4,0,0\n")
    # b: out_of_bounds box (20x20 image, box extends past)
    (ann_dir / "b.txt").write_text("15,15,10,10,1,4,0,0\n")
    # c: duplicate_box (identical bbox+category twice)
    (ann_dir / "c.txt").write_text("1,1,5,5,1,4,0,0\n1,1,5,5,1,4,0,0\n")
    # d: score_zero (category 1-10, score 0) and ignored_region (category 0)
    (ann_dir / "d.txt").write_text("1,1,5,5,0,4,0,0\n1,1,5,5,1,0,0,0\n")
    # e: malformed_row (not 8 fields)
    (ann_dir / "e.txt").write_text("1,1,5,5,1\n")

    result = qa.run_visdrone(ann_dir, img_dir)
    counts = result["counts"]

    assert counts["zero_area"] == 1
    assert counts["out_of_bounds"] == 1
    assert counts["duplicate_box"] == 1
    assert counts["score_zero"] == 1
    assert counts["ignored_region"] == 1
    assert counts["malformed_row"] == 1


def test_run_icg_invalid_class_id(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    csv_path = tmp_path / "class_dict_seg.csv"
    csv_path.write_text("name, r, g, b\n" + "\n".join(f"c{i}, 0, 0, 0" for i in range(24)))

    _write_image(img_dir / "x.jpg", w=4, h=4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 30
    cv2.imwrite(str(mask_dir / "x.png"), mask)

    result = qa.run_icg(img_dir, mask_dir, csv_path)
    assert result["counts"]["invalid_class_id"] == 1
