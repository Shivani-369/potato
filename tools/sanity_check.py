import os
from collections import Counter
from PIL import Image

def sanity_check(data_dir="data/split/train"):
    """
    Checks dataset consistency:
    - Prints number of classes
    - Number of images per class
    - Tries opening some images to catch corrupt files
    """
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return

    print(f"🔍 Running sanity check on {data_dir}")

    class_counts = Counter()
    corrupt_files = []

    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = os.listdir(cls_dir)
        class_counts[cls] += len(images)

        for img in images[:5]:  # check first 5 images
            try:
                Image.open(os.path.join(cls_dir, img)).verify()
            except Exception:
                corrupt_files.append(os.path.join(cls_dir, img))

    print("\n📊 Class distribution:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} images")

    if corrupt_files:
        print("\n⚠️ Corrupt images found:")
        for f in corrupt_files:
            print("   ", f)
    else:
        print("\n✅ No corrupt images detected.")

if __name__ == "__main__":
    sanity_check()
