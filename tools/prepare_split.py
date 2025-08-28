import os
import shutil
import random

def prepare_split(data_dir="data/tubers", output_dir="data/split", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits dataset into train, val, test folders.
    Assumes folder structure: data/tubers/<class_name>/*.jpg
    """

    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Remove old split if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Get class folders
    classes = os.listdir(data_dir)
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = os.listdir(cls_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        # Copy to new split folders
        for split_name, split_imgs in splits.items():
            split_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_cls_dir, img)
                shutil.copy(src, dst)

    print(f"✅ Dataset split created at {output_dir}")

if __name__ == "__main__":
    prepare_split()
