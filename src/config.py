from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
PROCESSED_DATASET_DIR = ROOT_DIR / "processed_dataset"
TRAIN_DIR = PROCESSED_DATASET_DIR / "train"
VALID_DIR = PROCESSED_DATASET_DIR / "validation"
TEST_DIR = PROCESSED_DATASET_DIR / "test"

MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL = MODEL_DIR / "best_model.keras"
FINAL_MODEL = MODEL_DIR / "final_model.keras"

REPORT_DIR = ROOT_DIR / "reports"

IMAGE_SIZE = (224, 224)

CHANNEL = 3
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
SEED = 42


TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
NUM_CLASSES = 6
CLASS_MODE = "categorical"

HOST = "127.0.0.1"
PORT = 8001

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
PROCESSED_DATASET_DIR.mkdir(exist_ok=True)