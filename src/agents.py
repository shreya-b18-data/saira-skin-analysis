"""Rule-based agent layer: Retriever, Safety, Recommendation, Routine, Explainability.
No LLM calls: every decision is deterministic and traceable to the knowledge base."""
import ast
import math

import pandas as pd

from knowledge import build_alias_index

# Hard blacklist of ingredients per skin type (extend with care)
BLACKLIST = {
    "Sensitive": {"fragrance", "alcohol denat.", "retinol", "glycolic acid", "benzoyl peroxide"},
    "Dry": {"alcohol denat.", "benzoyl peroxide"},
    "Oily": set(),
    "Combination": {"alcohol denat."},
    "Normal": set(),
}


def parse_ingredients(raw):
    """Sephora 'ingredients' column is a stringified list; fall back to comma-splitting."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, (list, tuple)):
            return [str(i).strip().lower() for i in items]
    except (ValueError, SyntaxError):
        pass
    return [i.strip().lower() for i in raw.split(",") if i.strip()]


def canonicalize(items, alias_idx):
    """Map raw INCI strings to KB canonical names (substring match on aliases)."""
    found = set()
    for it in items:
        for alias, canon in alias_idx.items():
            if alias in it:
                found.add(canon)
    return found


class ProductIndex:
    """Skincare products with KB-canonical ingredient sets."""

    def __init__(self, df, kb):
        self.kb, self.alias_idx = kb, build_alias_index(kb)
        df = df[df["primary_category"].str.lower() == "skincare"].copy() if "primary_category" in df else df.copy()
        df["ing_list"] = df["ingredients"].map(parse_ingredients)
        df = df[df["ing_list"].map(len) > 0]
        df["ing_set"] = df["ing_list"].map(lambda x: canonicalize(x, self.alias_idx))
        self.df = df.reset_index(drop=True)
        self.inverted = {}
        for i, s in enumerate(self.df["ing_set"]):
            for ing in s:
                self.inverted.setdefault(ing, set()).add(i)


def retrieve(index, needed, top_k=50):
    """Rank products by overlap coefficient |A∩B| / min(|A|,|B|) with the needed-ingredient set."""
    needed = set(needed)
    cand = set().union(*(index.inverted.get(i, set()) for i in needed)) if needed else set()
    rows = []
    for i in cand:
        s = index.df.at[i, "ing_set"]
        rows.append((i, len(needed & s) / max(1, min(len(needed), len(s)))))
    rows.sort(key=lambda r: -r[1])
    out = index.df.loc[[r[0] for r in rows[:top_k]]].copy()
    out["ingredient_match"] = [r[1] for r in rows[:top_k]]
    return out


def safety_check(ing_set, skin_type, kb):
    """Return (score 0-1, reasons). Hard-blacklist hit => score 0."""
    reasons = []
    bad = ing_set & BLACKLIST.get(skin_type, set())
    if bad:
        return 0.0, [f"contains {', '.join(sorted(bad))}, which is on the {skin_type} skin blacklist"]
    score = 1.0
    for a in ing_set:
        for b in kb.get(a, {}).get("avoid_with", []):
            if b in ing_set and a < b:  # count each conflicting pair once
                score -= 0.2
                reasons.append(f"{a} conflicts with {b}")
    return max(0.0, score), reasons


def recommend(cands, skin_type, kb, top_n=10, max_per_category=2):
    """score = 0.40 match + 0.30 safety + 0.20 rating + 0.10 log(reviews); diversity cap per category."""
    rows = []
    for _, r in cands.iterrows():
        safe, why = safety_check(r["ing_set"], skin_type, kb)
        if safe == 0.0:
            continue  # rejected by the Safety agent
        rating = float(r.get("rating", 0) or 0) / 5.0
        reviews = math.log1p(float(r.get("reviews", 0) or 0)) / math.log1p(10000)
        score = 0.40 * r["ingredient_match"] + 0.30 * safe + 0.20 * rating + 0.10 * min(1.0, reviews)
        rows.append({**r.to_dict(), "safety": safe, "safety_notes": why, "score": score})
    rows.sort(key=lambda x: -x["score"])
    per_cat, out = {}, []
    for r in rows:
        cat = r.get("secondary_category", "other")
        if per_cat.get(cat, 0) >= max_per_category:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        out.append(r)
        if len(out) == top_n:
            break
    return out


def build_routine(recs):
    """Assign products to AM / PM by evidence-based rules (retinoids -> PM, SPF/vitamin C -> AM)."""
    am, pm = [], []
    for r in recs:
        s, name = r["ing_set"], str(r.get("product_name", "")).lower()
        if "retinol" in s or "adapalene" in s or "glycolic acid" in s:
            pm.append(r)
        elif "spf" in name or "sunscreen" in name or "ascorbic acid" in s:
            am.append(r)
        else:
            am.append(r); pm.append(r)
    return {"AM": am, "PM": pm}


def explain(rec, needed_by_concern, kb):
    hits = sorted(rec["ing_set"] & {i for v in needed_by_concern.values() for i in v})
    parts = [f"{rec.get('product_name', 'This product')} contains {', '.join(hits)}." if hits else "General-purpose match."]
    for concern, ings in needed_by_concern.items():
        h = [i for i in hits if i in ings]
        if h:
            parts.append(f"{', '.join(h)} target {concern}.")
    parts += [f"Note: {n}." for n in rec.get("safety_notes", [])]
    return " ".join(parts)
