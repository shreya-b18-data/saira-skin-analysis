"""Train SAIRANet.  Usage:  python src/train.py --epochs 100 --batch 32"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CKPT_PATH, HEADS, METRICS_PATH, SEED
from dataset import SkinDataset, index_dataset, split
from model import SAIRANet


def focal_bce(logits, y, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def step_loss(out, y, skin, ce):
    reg = sum(focal_bce(out[h], y[:, i]) for i, h in enumerate(HEADS))
    return reg + 1.5 * ce(out["skin_type"], skin)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    P, Y, S, SP = [], [], [], []
    for x, y, s in loader:
        o = model(x.to(dev))
        P.append(torch.stack([torch.sigmoid(o[h]) for h in HEADS], 1).cpu())
        Y.append(y); S.append(s); SP.append(o["skin_type"].argmax(-1).cpu())
    P, Y, S, SP = map(torch.cat, (P, Y, S, SP))
    err = P - Y
    m = {h: {"mae": err[:, i].abs().mean().item(), "rmse": err[:, i].pow(2).mean().sqrt().item()}
         for i, h in enumerate(HEADS)}
    m["skin_type"] = {"accuracy": (SP == S).float().mean().item(),
                      "f1_macro": float(f1_score(S, SP, average="macro"))}
    m["mean_mae"] = float(np.mean([m[h]["mae"] for h in HEADS]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths, folders = index_dataset()
    (ptr, ftr), (pva, fva), (pte, fte) = split(paths, folders)
    print(f"train/val/test = {len(ptr)}/{len(pva)}/{len(pte)}  device={dev}")

    def mk(p, f, train):
        return DataLoader(SkinDataset(p, f, train), batch_size=a.batch, shuffle=train, num_workers=2)
    tr, va, te = mk(ptr, ftr, True), mk(pva, fva, False), mk(pte, fte, False)

    model = SAIRANet().to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    best, bad = float("inf"), 0

    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y, s in tqdm(tr, desc=f"epoch {ep}", leave=False):
            x, y, s = x.to(dev), y.to(dev), s.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = step_loss(model(x), y, s, ce)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y, s in va:
                x, y, s = x.to(dev), y.to(dev), s.to(dev)
                vl += step_loss(model(x), y, s, ce).item() * len(x)
        vl /= len(va.dataset)
        print(f"epoch {ep}: val_loss={vl:.4f}")
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(CKPT_PATH, map_location=dev))
    metrics = evaluate(model, te, dev)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
