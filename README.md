# Offline Satellite Tile Classifier

A small offline service that classifies satellite tiles into 7 land-use classes (AnnualCrop, Forest, Highway, Industrial, Residential, River, SeaLake), stores each result in SQLite, and lets an analyst query the results. Everything runs on CPU with no internet after a one-time weight download.

**Approach:** frozen pretrained ResNet18 (ImageNet) as a feature extractor, plus logistic regression trained on the provided tiles. Predictions below a confidence threshold are stored but flagged `needs_review`.

**Results on the held-out eval set (210 tiles):** 92.9% accuracy. At the default threshold of 0.9, 87.6% of tiles are auto-accepted at 97.3% accuracy, and 12.4% go to review.

## Project structure

```
.
├── classifier.py         # shared: preprocessing, ResNet18 feature extractor, batched feature extraction
├── download_weights.py   # one-time: saves ResNet18 weights locally
├── train.py              # trains logistic regression on cached features, saves the model
├── evaluate.py           # accuracy, confusion matrix, confidence analysis, threshold table
├── app.py                # FastAPI service: POST /classify + query endpoints
├── requirements.txt
├── design_note.docx
└── PART3_ANSWERS.txt
```

Created when you run the scripts (not in the repo):

| Path | Created by | Contents |
|---|---|---|
| `models/resnet18_imagenet.pth` | `download_weights.py` | Pretrained ResNet18 weights |
| `X.npy`, `y.npy`, `classes.json` | `classifier.py` | Cached features, labels, class order |
| `models/classifier.joblib` | `train.py` | Trained classifier + class order + model version |
| `eval_predictions.csv` | `evaluate.py` | Per-tile eval results |
| `results.db`, `stored_tiles/` | `app.py` | SQLite results and raw uploaded tiles |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Unzip the dataset. It should look like this:

```
<dataset folder>/
├── candidate_tiles/<ClassName>/*.png
├── eval_set/*.png
└── eval_labels.csv
```

**Then set `BASE_DIR` at the top of `classifier.py` to that folder.**

## How to run

Run these in order from the project folder.

### 1. Download the model weights (once, needs internet)

```bash
python download_weights.py
```

This saves `models/resnet18_imagenet.pth` (about 45 MB). After this step, nothing needs the internet.

### 2. Extract features

```bash
python classifier.py
```

Passes every candidate tile through ResNet18 and saves `X.npy` (N × 512 features), `y.npy` (labels) and `classes.json` (class order). This is the slow step and only needs to run once. Delete these files if you change the tiles or the preprocessing.

### 3. Train

```bash
python train.py
```

Trains on an 80/20 stratified split and prints validation metrics, then refits on all tiles and saves `models/classifier.joblib` (classifier, class order, model version).

### 4. Evaluate

```bash
python evaluate.py
```

Runs the saved model on `eval_set/` against `eval_labels.csv`. Prints accuracy, a per-class report, the confusion matrix, mean confidence for correct vs wrong predictions, and the threshold table. Per-tile results are saved to `eval_predictions.csv`.

### 5. Start the API

```bash
uvicorn app:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

To change the review threshold without touching code:

```bash
CONFIDENCE_THRESHOLD=0.95 uvicorn app:app
```

## API

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/classify` | Upload a tile; validates, classifies, flags and stores it |
| GET | `/results` | List results. Filters: `label`, `status`, `min_conf`, `max_conf`, `limit` |
| GET | `/results/{tile_hash}` | All stored results for one tile |
| GET | `/stats` | Counts and average confidence per class, counts by status |
| GET | `/health` | Service status and loaded model version |

### Examples

```bash
# classify one tile
curl -F "file=@/path/to/eval_set/tile_001.png" http://127.0.0.1:8000/classify

# tiles needing review
curl "http://127.0.0.1:8000/results?status=needs_review"

# confident River predictions
curl "http://127.0.0.1:8000/results?label=River&min_conf=0.9"

# summary
curl http://127.0.0.1:8000/stats
```

### Example response from `POST /classify`

```json
{
  "duplicate": false,
  "tile_hash": "3f9a...",
  "model_version": "resnet18-logreg-v1",
  "filename": "tile_001.png",
  "predicted_label": "Forest",
  "confidence": 0.991,
  "probabilities": {"AnnualCrop": 0.001, "Forest": 0.991, "...": "..."},
  "status": "auto_accepted",
  "threshold": 0.9,
  "latency_ms": 41.2,
  "created_at": "2026-09-27T10:15:00+00:00"
}
```

### Quick checks

- Upload the same tile twice: the second response has `"duplicate": true` (dedupe on tile hash + model version).
- Upload a non-image file: `400 Not a valid image`.
- Upload all 210 eval tiles, then call `/stats`: about 26 should be `needs_review`, matching the 0.9 row of the threshold table.

## Built vs stubbed

| Part | Status |
|---|---|
| Feature extraction, training, evaluation scripts | Built |
| `POST /classify` with validation, dedupe, confidence flag, storage | Built |
| `GET /results`, `/results/{hash}`, `/stats`, `/health` | Built (basic) |
| Review workflow (mark a tile as reviewed or corrected) | Not built |
| Authentication | Not built |
| Batch upload and async processing | Not built |
| Drift monitoring and scheduled canary checks | Not built (described in Part 3) |
| Model rollout and re-classification of old tiles | Not built |
| Dataset path as a command-line argument | Not built (edit `BASE_DIR`) |


## Use of AI tools

I used Claude (Anthropic) code building and documentation improvements (README, design note and Part 3). I reviewed, ran and modified all of the code myself, verified the results on the eval set, and can explain and change any part of it.