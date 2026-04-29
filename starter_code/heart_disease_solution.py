"""
╔══════════════════════════════════════════════════════════════╗
║   Heart Disease Prediction Challenge 2026 — Complete Solution ║
║   Run this on Google Colab (colab.research.google.com)        ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:
  1. Open Google Colab: https://colab.research.google.com
  2. Click File → New Notebook
  3. Paste this entire file into the first cell
  4. Change GROUP = "group1" to your actual group number (e.g. "group5")
  5. Click the ▶ Run button
  6. Download the generated CSV file from the Files panel (left sidebar)
"""

# ── STEP 0: Change this to your group number! ──────────────────────────────
GROUP = "C4"   # ← e.g. "group3", "group7", etc.
# ───────────────────────────────────────────────────────────────────────────


# ── 1. Import libraries ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries loaded")


# ── 2. Load data directly from the GitHub repo ──────────────────────────────


train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"✅ Data loaded:  train={train.shape}  test={test.shape}")
print(f"\nColumns: {list(train.columns)}")
print(f"\nTarget distribution:\n{train['target'].value_counts()}\n")


# ── 3. Feature Engineering ──────────────────────────────────────────────────
def engineer_features(df):
    """Add new features that help the model learn better."""
    df = df.copy()

    # Age group buckets
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 40, 55, 70, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    # Heart rate features
    if 'thalach' in df.columns:
        df['max_hr_reserve'] = 220 - df['age'] - df['thalach']
        df['hr_pct_max']     = df['thalach'] / (220 - df['age'])

    # Chest pain × blood pressure
    if 'cp' in df.columns and 'trestbps' in df.columns:
        df['cp_bp_product'] = df['cp'] * df['trestbps']

    # ST depression index
    if 'oldpeak' in df.columns and 'slope' in df.columns:
        df['st_index'] = df['oldpeak'] * (df['slope'] + 1)

    # Cholesterol per age
    if 'chol' in df.columns:
        df['chol_per_age'] = df['chol'] / df['age']

    return df


train_fe = engineer_features(train)
test_fe  = engineer_features(test)

FEATURES = [c for c in train_fe.columns if c != 'target']
X_train  = train_fe[FEATURES]
y_train  = train_fe['target']
X_test   = test_fe[FEATURES]

print(f"✅ Features used ({len(FEATURES)}): {FEATURES}")


# ── 4. Build & compare models ───────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=4,
        class_weight='balanced', random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        random_state=42
    ),
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced',
                    probability=True, random_state=42))
    ]),
}

print("\n📊 Cross-validation F1 scores (higher = better):")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring='f1', n_jobs=-1)
    print(f"  {name:25s}: {scores.mean():.4f}  ±{scores.std():.4f}")


# ── 5. Ensemble: combines all 4 models for the best F1 ─────────────────────
ensemble = VotingClassifier(
    estimators=list(models.items()),
    voting='soft'      # uses predicted probabilities — usually more accurate
)

ens_scores = cross_val_score(ensemble, X_train, y_train, cv=cv,
                              scoring='f1', n_jobs=-1)
print(f"  {'Ensemble (FINAL)':25s}: {ens_scores.mean():.4f}  ±{ens_scores.std():.4f}")


# ── 6. Train on all data & predict ─────────────────────────────────────────
print("\n🔧 Training on full training set...")
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

vals, counts = np.unique(predictions, return_counts=True)
print(f"Prediction counts: { dict(zip(vals.tolist(), counts.tolist())) }")
print(f"Total predictions : {len(predictions)}  (must be 61)")


# ── 7. Save submission file ─────────────────────────────────────────────────
filename   = f"{GROUP}_submission.csv"
submission = pd.DataFrame({'prediction': predictions})
submission.to_csv(filename, index=False)

print(f"\n✅ Saved: {filename}")
print(submission.head(10).to_string())
print(f"\n--- Submission ready! Download '{filename}' from the Files panel ---")


# ── 8. Verify format ────────────────────────────────────────────────────────
assert len(submission) == 61,        f"❌ Expected 61 rows, got {len(submission)}"
assert list(submission.columns) == ['prediction'], "❌ Column must be named 'prediction'"
assert set(submission['prediction'].unique()).issubset({0, 1}), "❌ Values must be 0 or 1"
print("\n✅ All format checks passed — you're good to submit!")
