"""Shared constants. Edit paths here, not in the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"                            # 8 concern folders (not tracked by git)
PRODUCTS_CSV = ROOT / "Ingrediants" / "product_info.csv"  # Sephora product_info.csv (not tracked)
KB_PATH = ROOT / "data" / "ingredient_kb.json"
CKPT_PATH = ROOT / "checkpoints" / "best_model.pth"
METRICS_PATH = ROOT / "checkpoints" / "metrics.json"

HEADS = ["acne", "pigmentation", "redness", "wrinkles"]
SKIN_TYPES = ["Oily", "Dry", "Combination", "Sensitive", "Normal"]

# dataset folder name -> concern head
FOLDER_TO_HEAD = {
    "Acne": "acne",
    "inflammatory acne": "acne",
    "non inflammatory acne black heads": "acne",
    "non inflammatory acne white heads": "acne",
    "pigmentation": "pigmentation",
    "dark spots": "pigmentation",
    "Redness": "redness",
    "wrinkles": "wrinkles",
}
# Rule-based skin-type labels (no ground truth exists in the dataset; see README limitations)
HEAD_TO_SKIN_TYPE = {"acne": "Oily", "redness": "Sensitive", "wrinkles": "Dry", "pigmentation": "Combination"}

IMG_SIZE = 224
SEED = 42
