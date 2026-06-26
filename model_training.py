"""
model_training.py
-----------------
Trains multiple ML models on the placement dataset.
Run: python model_training.py

FIXES APPLIED:
  1. Dataset is augmented with all 8 app domains (previously only 4 existed in
     the CSV: Data Science, Java, Python, Web Development). The 6 missing domains
     are added with realistic synthetic rows so label_encoder covers every domain
     that the app presents to users.
  2. resume_score column is added to the dataset and included as feature index 7,
     matching the feature vector order in prediction.py.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: AUGMENT DATASET
# ============================================================
def augment_dataset(path='dataset/placement_data.csv'):
    """
    Loads the raw CSV, adds resume_score if missing, and appends synthetic rows
    for any domain not already present so all 8 app domains are covered.
    Saves the augmented CSV back to the same path.
    """
    df = pd.read_csv(path)

    # Add resume_score column if not present
    if 'resume_score' not in df.columns:
        np.random.seed(42)
        df['resume_score'] = np.random.uniform(40, 95, len(df)).round(2)
        print("[AUGMENT] Added resume_score column to existing rows.")

    required_domains = [
        'Data Science', 'Web Development', 'Cybersecurity', 'Cloud Computing',
        'DevOps', 'Internet of Things (IoT)', 'Blockchain Technology', 'Mobile App Development'
    ]
    existing_domains = df['domain'].unique().tolist()
    missing_domains  = [d for d in required_domains if d not in existing_domains]

    if missing_domains:
        print(f"[AUGMENT] Adding synthetic rows for: {missing_domains}")
        np.random.seed(42)
        new_rows = []
        rows_per_domain = 120  # comparable density to existing minority classes

        placed_ratio = df['placement_status'].mean()

        for domain in missing_domains:
            n_placed = int(rows_per_domain * placed_ratio)
            n_not    = rows_per_domain - n_placed

            def sample_row(placed):
                if placed:
                    tenth   = np.random.uniform(60, 99)
                    twelve  = np.random.uniform(60, 99)
                    cgpa    = np.random.uniform(6.5, 10.0)
                    bl      = int(np.random.choice([0, 1], p=[0.80, 0.20]))
                    apt     = np.random.uniform(50, 100)
                    tech    = np.random.uniform(50, 100)
                    comm    = np.random.uniform(50, 100)
                    resume  = np.random.uniform(55, 95)
                else:
                    tenth   = np.random.uniform(45, 75)
                    twelve  = np.random.uniform(45, 75)
                    cgpa    = np.random.uniform(5.0, 7.5)
                    bl      = int(np.random.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.25, 0.15]))
                    apt     = np.random.uniform(20, 65)
                    tech    = np.random.uniform(20, 60)
                    comm    = np.random.uniform(20, 65)
                    resume  = np.random.uniform(30, 65)
                return {
                    'tenth_percentage'   : round(tenth,  2),
                    'twelfth_percentage' : round(twelve, 2),
                    'cgpa'               : round(cgpa,   2),
                    'backlogs'           : bl,
                    'aptitude_score'     : round(apt,    2),
                    'technical_score'    : round(tech,   2),
                    'communication_score': round(comm,   2),
                    'resume_score'       : round(resume, 2),
                    'domain'             : domain,
                    'placement_status'   : 1 if placed else 0,
                }

            for _ in range(n_placed):
                new_rows.append(sample_row(True))
            for _ in range(n_not):
                new_rows.append(sample_row(False))

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"[AUGMENT] Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ============================================================
# STEP 2: LOAD DATA
# ============================================================
def load_data(path='dataset/placement_data.csv'):
    df = pd.read_csv(path)
    print(f"[INFO] Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Domains  : {sorted(df['domain'].unique())}")
    print(f"[INFO] Class distribution:\n{df['placement_status'].value_counts()}\n")
    return df


# ============================================================
# STEP 3: PREPROCESS
# ============================================================
def preprocess(df):
    df = df.copy()

    le = LabelEncoder()
    df['domain_encoded'] = le.fit_transform(df['domain'])

    df['academic_score'] = (
        df['tenth_percentage']   * 0.2 +
        df['twelfth_percentage'] * 0.2 +
        df['cgpa'] * 10          * 0.6
    )

    df['skill_score'] = (
        df['aptitude_score']       * 0.35 +
        df['technical_score']      * 0.45 +
        df['communication_score']  * 0.20
    )

    df['backlog_penalty'] = df['backlogs'].apply(lambda x: 1 if x == 0 else 0)

    # Feature order MUST match _build_feature_vector() in prediction.py
    features = [
        'tenth_percentage',    # 0
        'twelfth_percentage',  # 1
        'cgpa',                # 2
        'backlogs',            # 3
        'aptitude_score',      # 4
        'technical_score',     # 5
        'communication_score', # 6
        'resume_score',        # 7  ← must be index 7
        'domain_encoded',      # 8
        'academic_score',      # 9
        'skill_score',         # 10
        'backlog_penalty',     # 11
    ]

    X = df[features]
    y = df['placement_status']

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler,   'models/scaler.pkl')
    joblib.dump(le,       'models/label_encoder.pkl')
    joblib.dump(features, 'models/feature_names.pkl')

    print(f"[INFO] Features ({len(features)}): {features}")
    print(f"[INFO] Label encoder classes: {list(le.classes_)}")
    return X_scaled, y


# ============================================================
# STEP 4: TRAIN MODELS
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000),
        'Decision Tree'       : DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest'       : RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = {}
    trained = {}

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

        results[name] = {
            'accuracy'   : round(acc  * 100, 2),
            'precision'  : round(prec * 100, 2),
            'recall'     : round(rec  * 100, 2),
            'f1_score'   : round(f1   * 100, 2),
            'cv_accuracy': round(cv   * 100, 2)
        }
        trained[name] = model

        print(f"\n▶ {name}")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1-Score  : {f1*100:.2f}%")
        print(f"  CV Acc(5) : {cv*100:.2f}%")
        print(f"  Confusion :\n{confusion_matrix(y_test, y_pred)}")

    return results, trained


# ============================================================
# STEP 5: SAVE BEST MODEL
# ============================================================
def save_best_model(results, trained):
    best_name  = max(results, key=lambda k: results[k]['f1_score'])
    best_model = trained[best_name]

    joblib.dump(best_model, 'models/trained_model.pkl')
    joblib.dump(trained,    'models/all_models.pkl')

    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"[BEST MODEL] {best_name} (F1: {results[best_name]['f1_score']}%)")
    print("[SAVED] models/trained_model.pkl")
    print("[SAVED] models/all_models.pkl")
    print("[SAVED] models/evaluation_results.json")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    augment_dataset()   # ensure all 8 domains + resume_score exist
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    results, trained = train_models(X_train, X_test, y_train, y_test)
    save_best_model(results, trained)
    print("\n[TRAINING COMPLETE] ✓")
