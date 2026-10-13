# SAIRA: Skin Analysis & Intelligent Recommendation Agent

Multi-task deep learning + a rule-based agent layer that turns a face photo into skin-concern scores, ingredient needs, and a safety-checked AM/PM skincare routine.

## Pipeline
```
face photo -> face crop + CLAHE -> SAIRANet (EfficientNet-B3)
           -> 4 concern scores (acne, pigmentation, redness, wrinkles) + skin type
           -> concern -> ingredient mapping (knowledge base)
           -> product retrieval (inverted index, overlap coefficient)
           -> safety check -> ranking -> AM/PM routine -> explanations
```

## Structure
```
src/config.py      paths, labels, constants
src/dataset.py     face crop, CLAHE, stratified 70/15/15 split, PyTorch Dataset
src/model.py       SAIRANet: EfficientNet-B3 + 4 concern heads + skin-type head
src/train.py       focal loss + CE, AdamW + cosine LR, early stopping, test metrics
src/knowledge.py   ingredient KB loader, concern -> ingredient tiers
src/agents.py      retrieval, safety, recommendation, routine, explainability
src/pipeline.py    end-to-end inference
data/ingredient_kb.json   starter KB (24 entries, to be expanded)
api/main.py        FastAPI server: POST /analyze (photo -> concern scores)
docs/              project summary, saira_demo.html (upload page)
```

## Data (not included)
- Skin images: 8 concern folders (Acne, inflammatory acne, non inflammatory acne black heads / white heads, pigmentation, dark spots, Redness, wrinkles) placed in `dataset/`.
- Products: Sephora `product_info.csv` (Kaggle) placed in `Ingrediants/`.

## Run
```bash
pip install -r requirements.txt
python src/train.py --epochs 100 --batch 32     # writes checkpoints/best_model.pth + metrics.json
python src/pipeline.py selfie.jpg --skin-type Oily
uvicorn api.main:app --port 8000                # then open docs/saira_demo.html, enter http://localhost:8000
```

## Results
Run `src/train.py` to produce `checkpoints/metrics.json` and report your own numbers here.

## Limitations
- The dataset has no skin-type labels. Skin-type labels are rule-based (acne -> Oily, redness -> Sensitive, ...), so skin-type accuracy measures agreement with that rule, not real skin type. Prefer user-reported skin type.
- Concern "severity" scores come from folder-level binary labels, so they behave like probabilities, not clinical severity.
- The knowledge base is a small starter set and is not medical advice.
- No fairness validation across skin tones.
- Research prototype, not a medical device.
