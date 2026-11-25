import argparse

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import os
    from pathlib import Path
    from PIL import Image

    # Eski ana klasör
    ROOT = args.data

    # Yeni crop'lanmış ana klasör
    OUT_ROOT = args.out

    # Sadece train ve val var
    SPLITS = ["train", "val"]

    CLASS_CONFIG = {
        "normal": {
            "img_dir_name": "normal",
            "label_dir_name": "normal labels"
        },
        "pothole": {
            "img_dir_name": "pothole",
            "label_dir_name": "pothole labels"
        }
    }

    # bbox etrafına biraz pay bırak
    PADDING_RATIO = 0.15

    # label yoksa resmi kopyala mı?
    COPY_IF_NO_LABEL = False


    def find_image_file(img_dir: Path, stem: str):
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for ext in exts:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None


    def load_yolo_boxes(label_path: Path):
        boxes = []
        if not label_path.exists():
            return boxes

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            cls_id, xc, yc, w, h = parts
            boxes.append((int(float(cls_id)), float(xc), float(yc), float(w), float(h)))

        return boxes


    def yolo_to_xyxy(box, img_w, img_h, padding_ratio=0.15):
        _, xc, yc, bw, bh = box

        xc *= img_w
        yc *= img_h
        bw *= img_w
        bh *= img_h

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio

        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img_w, int(round(x2)))
        y2 = min(img_h, int(round(y2)))

        return x1, y1, x2, y2


    def crop_with_union_of_boxes(img: Image.Image, boxes, padding_ratio=0.15):
        img_w, img_h = img.size

        xyxy_list = [yolo_to_xyxy(b, img_w, img_h, padding_ratio) for b in boxes]

        x1 = min(b[0] for b in xyxy_list)
        y1 = min(b[1] for b in xyxy_list)
        x2 = max(b[2] for b in xyxy_list)
        y2 = max(b[3] for b in xyxy_list)

        if x2 <= x1 or y2 <= y1:
            return None

        return img.crop((x1, y1, x2, y2))


    def process_split(split_name):
        print(f"\n=== Processing {split_name} ===")

        split_root = ROOT / split_name
        out_split_root = OUT_ROOT / split_name

        for class_name, cfg in CLASS_CONFIG.items():
            img_dir = split_root / cfg["img_dir_name"]
            label_dir = split_root / cfg["label_dir_name"]
            out_img_dir = out_split_root / cfg["img_dir_name"]

            out_img_dir.mkdir(parents=True, exist_ok=True)

            if not img_dir.exists():
                print(f"[WARN] Image dir not found: {img_dir}")
                continue

            if not label_dir.exists():
                print(f"[WARN] Label dir not found: {label_dir}")
                continue

            label_files = list(label_dir.glob("*.txt"))
            print(f"\nClass: {class_name}")
            print(f"Images dir : {img_dir}")
            print(f"Labels dir : {label_dir}")
            print(f"Found label files: {len(label_files)}")

            saved = 0
            skipped = 0
            copied = 0

            for label_path in label_files:
                stem = label_path.stem
                img_path = find_image_file(img_dir, stem)

                if img_path is None:
                    skipped += 1
                    continue

                boxes = load_yolo_boxes(label_path)

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    skipped += 1
                    continue

                if len(boxes) == 0:
                    if COPY_IF_NO_LABEL:
                        out_path = out_img_dir / img_path.name
                        img.save(out_path)
                        copied += 1
                    else:
                        skipped += 1
                    continue

                cropped = crop_with_union_of_boxes(img, boxes, padding_ratio=PADDING_RATIO)

                if cropped is None:
                    skipped += 1
                    continue

                out_path = out_img_dir / img_path.name
                cropped.save(out_path)
                saved += 1

            print(f"Saved cropped : {saved}")
            print(f"Copied original: {copied}")
            print(f"Skipped       : {skipped}")


    for split in SPLITS:
        process_split(split)

    print("\nDone.")
    print(f"New cropped dataset root: {OUT_ROOT}")

if __name__ == "__main__":
    main()
