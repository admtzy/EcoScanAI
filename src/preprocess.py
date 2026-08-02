"""
==========================================================
EcoScan AI
Dataset Preprocessing
==========================================================
"""

import shutil
import random

from pathlib import Path

from sklearn.model_selection import train_test_split

from config import (
    DATASET_DIR,
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    SEED
)

from classes import CLASS_NAMES

random.seed(SEED)


# =====================================================
# CREATE DIRECTORY
# =====================================================

def create_directory():

    for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

        folder.mkdir(parents=True, exist_ok=True)

        for class_name in CLASS_NAMES:

            (folder / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


# =====================================================
# CLEAR OLD DATA
# =====================================================

def clear_processed_dataset():

    for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

        for class_name in CLASS_NAMES:

            target = folder / class_name

            if target.exists():

                shutil.rmtree(target)

            target.mkdir(parents=True)


# =====================================================
# COPY IMAGE
# =====================================================

def copy_files(files, destination):

    for file in files:

        shutil.copy(file, destination / file.name)


# =====================================================
# SPLIT DATASET
# =====================================================

def split_dataset():

    print("=" * 60)
    print("DATASET PREPROCESSING")
    print("=" * 60)

    total_dataset = 0

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        images = []

        for ext in ("*.jpg", "*.jpeg", "*.png"):

            images.extend(class_dir.glob(ext))

        images = sorted(images)

        total_dataset += len(images)

        print(f"\nClass : {class_name}")
        print(f"Total : {len(images)} images")

        train_images, temp_images = train_test_split(

            images,

            train_size=TRAIN_RATIO,

            random_state=SEED,

            shuffle=True

        )

        valid_size = VALID_RATIO / (VALID_RATIO + TEST_RATIO)

        valid_images, test_images = train_test_split(

            temp_images,

            train_size=valid_size,

            random_state=SEED,

            shuffle=True

        )

        copy_files(train_images, TRAIN_DIR / class_name)

        copy_files(valid_images, VALID_DIR / class_name)

        copy_files(test_images, TEST_DIR / class_name)

        print(f" Train      : {len(train_images)}")

        print(f" Validation : {len(valid_images)}")

        print(f" Test       : {len(test_images)}")

    print("\n" + "=" * 60)

    print(f"TOTAL DATASET : {total_dataset}")

    print("=" * 60)


# =====================================================
# SUMMARY
# =====================================================

def dataset_summary():

    print("\n")

    print("=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    grand_total = 0

    for folder_name, folder in {

        "TRAIN": TRAIN_DIR,

        "VALIDATION": VALID_DIR,

        "TEST": TEST_DIR

    }.items():

        print(f"\n{folder_name}")

        subtotal = 0

        for class_name in CLASS_NAMES:

            total = len(list((folder / class_name).glob("*")))

            subtotal += total

            print(f"{class_name:<12} : {total}")

        grand_total += subtotal

        print(f"TOTAL {folder_name:<10}: {subtotal}")

    print("\n")

    print("=" * 60)

    print(f"TOTAL IMAGE : {grand_total}")

    print("=" * 60)


# =====================================================
# MAIN
# =====================================================

def main():

    create_directory()

    clear_processed_dataset()

    split_dataset()

    dataset_summary()

    print("\nDataset preprocessing finished successfully.")

    print("Processed dataset saved to:")

    print(TRAIN_DIR.parent)


if __name__ == "__main__":

    main()