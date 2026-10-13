"""End-to-end SAIRA pipeline.  Usage:  python src/pipeline.py path/to/selfie.jpg [--skin-type Oily]"""
import argparse
import json

import pandas as pd
import torch

from agents import ProductIndex, build_routine, explain, recommend, retrieve
from config import CKPT_PATH, HEADS, PRODUCTS_CSV, SKIN_TYPES
from dataset import eval_tf, load_image
from knowledge import concern_to_ingredients, load_kb


class SAIRAPipeline:
    def __init__(self, ckpt=CKPT_PATH, products_csv=PRODUCTS_CSV):
        from model import SAIRANet  # imported here so the agent layer works without torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SAIRANet(pretrained=False).to(self.dev)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))
        self.kb = load_kb()
        self.index = ProductIndex(pd.read_csv(products_csv), self.kb)

    def run(self, image_path, skin_type=None):
        x = eval_tf()(load_image(image_path)).unsqueeze(0).to(self.dev)
        pred = self.model.predict(x)
        scores = {h: float(pred[h][0]) for h in HEADS}
        # Self-reported skin type is preferred: it is hard to infer from one photo.
        st = skin_type or SKIN_TYPES[int(pred["skin_type"][0].argmax())]
        needed = concern_to_ingredients(scores)
        all_needed = {i for v in needed.values() for i in v}
        recs = recommend(retrieve(self.index, all_needed), st, self.kb)
        return {"scores": scores, "skin_type": st, "needed_ingredients": needed,
                "routine": {k: [r["product_name"] for r in v] for k, v in build_routine(recs).items()},
                "explanations": {r["product_name"]: explain(r, needed, self.kb) for r in recs}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--skin-type", choices=SKIN_TYPES)
    a = ap.parse_args()
    print(json.dumps(SAIRAPipeline().run(a.image, a.skin_type), indent=2))
