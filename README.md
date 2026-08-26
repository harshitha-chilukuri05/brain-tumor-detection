# 🧠 Brain Tumor Detection from MRI Scans

An end-to-end deep learning project that classifies brain MRI scans into
four categories — **glioma**, **meningioma**, **pituitary tumor**, or
**no tumor** — using transfer learning on a ResNet-18 backbone, with
Grad-CAM explainability and a Streamlit demo app.

This is a **portfolio project**: everything from data pipeline to
evaluation to a deployable demo is included. It is **not** a validated
medical device and should never be used for real diagnosis.

---

## Why this project

Medical imaging is a strong choice because it forces you to
handle real challenges beyond "train a model, report accuracy":
- Class imbalance and small dataset sizes (transfer learning, not training from scratch)
- High-stakes predictions, where **interpretability matters** (Grad-CAM)
- Honest evaluation (confusion matrix + per-class metrics, not just overall accuracy)
- A usable interface, not just a notebook (Streamlit app)

---

## Project structure
brain-tumor-detection/
├── README.md
├── requirements.txt
├── data/ # populated after you download the dataset (see below)
│ ├── train/
│ │ ├── glioma/
│ │ ├── meningioma/
│ │ ├── notumor/
│ │ └── pituitary/
│ └── test/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
├── src/
│ ├── dataset.py # data loading + transforms
│ ├── model.py # ResNet-18 model definition
│ ├── train.py # training loop
│ ├── evaluate.py # test-set evaluation + confusion matrix
│ └── predict.py # single-image inference + Grad-CAM
├── app/
│ └── app.py # Streamlit demo app
├── saved_models/ # trained checkpoints saved here
└── outputs/ # plots, logs, reports saved here


---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Get the dataset

I couldn't bundle the actual MRI images in this zip (no internet access on
my end to fetch them), but the project is built around a well-known, free,
public dataset: **"Brain Tumor MRI Dataset"** by Masoud Nickparvar on
Kaggle (~7,000 images across the 4 classes, already split into train/test).

**Option A — Kaggle CLI (recommended):**
```bash
# 1. Get a Kaggle API token: https://www.kaggle.com/settings -> "Create New Token"
#    This downloads kaggle.json — place it at ~/.kaggle/kaggle.json

pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

**Option B — Manual download:**
1. Go to https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Download and unzip it
3. Arrange it so you end up with this structure (rename folders if needed):

data/
├── train/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
└── test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/


**Using your own dataset instead:** any dataset organized as
`data/train/<class_name>/*.jpg` and `data/test/<class_name>/*.jpg` will
work with this codebase unmodified — `dataset.py` uses `ImageFolder`,
which infers classes from folder names automatically.

---

## 3. Train the model

```bash
python src/train.py --data_dir data --epochs 15 --batch_size 32 --lr 1e-4
```

This will:
- Fine-tune a ResNet-18 (pretrained on ImageNet) on your data
- Save the best checkpoint (by test accuracy) to `saved_models/best_model.pt`
- Save loss/accuracy curves to `outputs/training_curves.png`
- Save a per-epoch metrics log to `outputs/training_log.csv`

Useful flags:
- `--freeze_backbone` — only train the final classifier layer (faster, good baseline)
- `--epochs`, `--batch_size`, `--lr` — standard hyperparameters

On a modern GPU this takes ~10-20 minutes for 15 epochs; on CPU expect much
longer (consider using `--freeze_backbone` and fewer epochs to iterate faster).

---

## 4. Evaluate

```bash
python src/evaluate.py --data_dir data --model_path saved_models/best_model.pt
```

Produces:
- `outputs/classification_report.txt` — precision/recall/F1 per class
- `outputs/confusion_matrix.png` — visual confusion matrix

**Results (test set, n=1289):**

| Metric | Score |
|---|---|
| Test Accuracy | 0.9860 |
| Macro F1 | 0.9691 |
| Weighted F1 | 0.9855 |
| Weakest class | glioma (0.90 F1 — see below) |

**Per-class breakdown:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| glioma | 1.0000 | 0.8202 | 0.9012 | 89 |
| meningioma | 0.9731 | 0.9950 | 0.9839 | 400 |
| notumor | 0.9877 | 1.0000 | 0.9938 | 400 |
| pituitary | 0.9950 | 1.0000 | 0.9975 | 400 |

**What the numbers actually show:** three of the four classes are
near-perfect (F1 > 0.98). Glioma is the outlier — recall of only 0.82
means the model misses about 18% of real glioma cases, most likely
misclassifying them as meningioma, since those two tumor types are the
ones radiologists themselves most commonly confuse on certain MRI slices.
Notably, glioma *precision* is a perfect 1.0000: when the model predicts
glioma, it's always right — so the model is conservative on this class
rather than over-predicting it. Two likely contributing factors: (1) the
glioma test set is much smaller (89 samples vs. 400 for the other
classes), making recall more sensitive to a handful of misclassifications,
and (2) visual similarity between glioma and meningioma on 2D MRI slices.

In a clinical framing, this is a meaningful limitation worth stating
plainly: missing a real tumor (false negative) is generally more costly
than a false alarm, so glioma recall — not overall accuracy — is the
number that matters most here, and it's the model's clearest weak point.

Being specific about *where* the model struggles (which classes get
confused, and a plausible reason why) is exactly the kind of detail that
makes a portfolio project credible in an interview.

---

## 5. Run inference with Grad-CAM on a single image

```bash
python src/predict.py --image path/to/scan.jpg --model_path saved_models/best_model.pt
```

Saves a heatmap overlay to `outputs/gradcam_result.png` showing which
regions of the scan drove the prediction.

---

## 6. Run the demo app

```bash
streamlit run app/app.py
```

Upload an MRI image in the browser and get a prediction, confidence score,
and Grad-CAM heatmap — this is the piece to record a short demo video/GIF
of for your resume or LinkedIn post.

---

## Notes on framing this for a resume / interview

- **Talk about the transfer learning decision.** Why ResNet-18 and not
  training from scratch? (Small dataset, limited compute, ImageNet features
  transfer well to texture/edge-heavy medical images.)
- **Talk about the confusion matrix, not just accuracy.** This model hits
  98.6% overall accuracy, but glioma recall is only 0.82 — it's the one
  class the model meaningfully struggles with, most likely confused with
  meningioma. Leading with that nuance instead of just "98.6% accuracy"
  is a much stronger interview answer, because it shows you looked past
  the headline number.
- **Talk about Grad-CAM.** It shows you think about trust and
  interpretability in high-stakes domains — a genuinely differentiating
  detail most portfolio projects skip.
- **Mention limitations honestly.** Not clinically validated, dataset size/
  diversity limitations, 2D slice classification vs. full 3D volumetric
  analysis, no external test set from a different scanner/hospital
  distribution. Showing you know the limits of your own model is a strong
  signal of maturity.

---

## Possible extensions (good "future work" talking points)

- **Close the glioma recall gap**: try class-weighted loss (upweighting
  glioma) or oversampling during training, since the current gap is
  likely driven partly by glioma having ~4.5x fewer training samples than
  the other three classes
- Swap ResNet-18 for a larger backbone (ResNet-50, EfficientNet) and compare
- Add k-fold cross-validation for more robust metrics
- Try a segmentation model (e.g. U-Net) to localize tumor boundaries, not
  just classify
- Deploy the Streamlit app publicly (Streamlit Community Cloud, Hugging
  Face Spaces) and link it directly from your resume
- Add test-time augmentation or ensembling for a small accuracy boost

---

## License / disclaimer

For educational and portfolio purposes only. Not a substitute for
professional medical diagnosis. Dataset license and usage terms are
governed by the original Kaggle dataset page.
