"""Dataset, face crop + CLAHE preprocessing, and stratified split."""
import cv2
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import (DATASET_DIR, FOLDER_TO_HEAD, HEAD_TO_SKIN_TYPE, HEADS,
                    IMG_SIZE, SEED, SKIN_TYPES)

IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(img_bgr, pad=0.2):
    """Haar-cascade face crop with padding. Returns the original image if no face is found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    dx, dy = int(w * pad), int(h * pad)
    H, W = img_bgr.shape[:2]
    return img_bgr[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]


def clahe(img_bgr, clip=2.0, grid=8):
    """CLAHE on the L channel of LAB."""
    l, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_image(path, face=True):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if face:
        img = crop_face(img)
    return Image.fromarray(cv2.cvtColor(clahe(img), cv2.COLOR_BGR2RGB))


def train_tf():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.1, 0.02), T.ToTensor(), T.Normalize(**IMAGENET)])


def eval_tf():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(**IMAGENET)])


def index_dataset(root=DATASET_DIR):
    """Return (paths, folder_names) for every image under the concern folders."""
    paths, folders = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in FOLDER_TO_HEAD:
            print(f"[skip] unknown folder: {folder.name}")
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(f)
                folders.append(folder.name)
    return paths, folders


def split(paths, folders):
    """Stratified 70/15/15 split by folder, seed 42."""
    p_tr, p_rest, f_tr, f_rest = train_test_split(paths, folders, test_size=0.30, stratify=folders, random_state=SEED)
    p_va, p_te, f_va, f_te = train_test_split(p_rest, f_rest, test_size=0.50, stratify=f_rest, random_state=SEED)
    return (p_tr, f_tr), (p_va, f_va), (p_te, f_te)


class SkinDataset(Dataset):
    def __init__(self, paths, folders, train=False):
        self.paths, self.folders = paths, folders
        self.tf = train_tf() if train else eval_tf()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        head = FOLDER_TO_HEAD[self.folders[i]]
        y = torch.tensor([float(h == head) for h in HEADS])             # multi-hot over concern heads
        skin = torch.tensor(SKIN_TYPES.index(HEAD_TO_SKIN_TYPE[head]))  # heuristic label
        return self.tf(load_image(self.paths[i])), y, skin
