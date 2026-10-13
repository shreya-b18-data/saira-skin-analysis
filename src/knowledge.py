"""Ingredient knowledge base + concern -> ingredient mapping."""
import json

from config import KB_PATH

# primary = first-line ingredients, secondary = supporting
CONCERN_INGREDIENTS = {
    "acne":         {"primary": ["salicylic acid", "benzoyl peroxide", "adapalene"], "secondary": ["niacinamide", "azelaic acid", "mandelic acid"]},
    "pigmentation": {"primary": ["ascorbic acid", "arbutin", "tranexamic acid"],      "secondary": ["niacinamide", "azelaic acid", "kojic acid", "lactic acid"]},
    "redness":      {"primary": ["centella asiatica", "azelaic acid", "allantoin"],   "secondary": ["niacinamide", "panthenol", "ceramide np", "zinc oxide"]},
    "wrinkles":     {"primary": ["retinol", "peptides", "hyaluronic acid"],           "secondary": ["ascorbic acid", "squalane", "glycerin", "ceramide np"]},
}


def load_kb(path=KB_PATH):
    kb = json.loads(path.read_text(encoding="utf-8"))
    kb.pop("_note", None)
    return kb


def build_alias_index(kb):
    """alias / canonical name -> canonical name."""
    idx = {}
    for name, e in kb.items():
        idx[name] = name
        for a in e.get("aliases", []):
            idx[a.lower()] = name
    return idx


def concern_to_ingredients(scores, high=0.6, mid=0.3):
    """Severity score >= high: primary+secondary; mid..high: primary only; < mid: none."""
    need = {}
    for concern, s in scores.items():
        tiers = CONCERN_INGREDIENTS.get(concern)
        if not tiers or s < mid:
            continue
        need[concern] = tiers["primary"] + (tiers["secondary"] if s >= high else [])
    return need
