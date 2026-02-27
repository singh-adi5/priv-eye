import os
import sys
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from recon_engine import LinuxReconEngine  # noqa: E402
from ml_pipeline import MatrixTransformer  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def host_id() -> str:
    try:
        return os.uname().nodename
    except Exception:
        return "unknown-host"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


class PrivEyeOracle:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # 3-class classifier: 0=low, 1=medium, 2=high
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=7,
            class_weight="balanced"
        )

        self.model_path = os.path.join(data_dir, "priveye_model.pkl")
        self.synthetic_data_path = os.path.join(data_dir, "training_dataset_3class.csv")

        self.dataset_path = os.path.join(data_dir, "dataset.csv")
        self.raw_path = os.path.join(data_dir, "raw_state.json")
        self.feature_path = os.path.join(data_dir, "feature_vector.csv")

        self.trained_features = []
        self.label_map = {"low": 0, "medium": 1, "high": 2}
        self.inv_label_map = {0: "low", 1: "medium", 2: "high"}

    # ---------------------------
    # Phase 0: Build real dataset row (optional but recommended)
    # ---------------------------
    def build_dataset_row(self, label: str = "unknown", timeout: int = 10) -> pd.DataFrame:
        logging.info("Phase 0: Building real telemetry row (recon -> features)...")

        engine = LinuxReconEngine(timeout=timeout)
        system_matrix = engine.execute_parallel_recon()

        ensure_parent_dir(self.raw_path)
        with open(self.raw_path, "w") as f:
            json.dump(system_matrix, f, indent=4)
        logging.info(f"Saved raw state to: {self.raw_path}")

        transformer = MatrixTransformer(self.raw_path)
        feat_df = transformer.generate_feature_vector()

        meta = pd.DataFrame([{
            "timestamp_utc": utc_now_iso(),
            "host_id": host_id(),
            "label": label
        }])

        row_df = pd.concat([meta, feat_df], axis=1)

        ensure_parent_dir(self.feature_path)
        row_df.to_csv(self.feature_path, index=False)
        logging.info(f"Saved latest feature snapshot to: {self.feature_path}")

        # Append to dataset.csv (schema-safe)
        ensure_parent_dir(self.dataset_path)
        if os.path.exists(self.dataset_path) and os.path.getsize(self.dataset_path) > 0:
            existing = pd.read_csv(self.dataset_path)
            all_cols = list(existing.columns)
            for c in row_df.columns:
                if c not in all_cols:
                    all_cols.append(c)

            existing = existing.reindex(columns=all_cols, fill_value=0)
            row_df = row_df.reindex(columns=all_cols, fill_value=0)
            updated = pd.concat([existing, row_df], ignore_index=True)
        else:
            updated = row_df

        updated.to_csv(self.dataset_path, index=False)
        logging.info(f"Dataset updated: {self.dataset_path} (shape={updated.shape})")

        return row_df

    # ---------------------------
    # Phase 1: Synthetic 3-class dataset (fallback)
    # ---------------------------
    def synthesize_data_3class(self, num_samples: int = 1200) -> pd.DataFrame:
        """
        Generates a 3-class synthetic dataset:
          0 = low risk (locked down servers)
          1 = medium risk (developer / mixed posture)
          2 = high risk (permissive / pentest posture)
        """
        logging.info(f"Synthesizing {num_samples} simulated baselines (3-class)...")
        np.random.seed(42)

        n_each = num_samples // 3
        n_low = n_each
        n_med = n_each
        n_high = num_samples - (n_low + n_med)

        low = pd.DataFrame({
            "kernel_major": np.random.choice([5, 6], n_low),
            "kernel_minor": np.random.randint(12, 24, n_low),
            "kernel_patch": np.random.randint(0, 12, n_low),
            "kernel_flavor": np.random.choice(["ubuntu", "rhel", "standard"], n_low),

            "suid_pkexec": np.random.choice([0, 1], n_low, p=[0.98, 0.02]).astype(int),
            "suid_su": np.ones(n_low, dtype=int),
            "suid_sudo": np.ones(n_low, dtype=int),
            "suid_mount": np.random.choice([0, 1], n_low, p=[0.2, 0.8]).astype(int),
            "suid_passwd": np.ones(n_low, dtype=int),
            "suid_chsh": np.random.choice([0, 1], n_low, p=[0.8, 0.2]).astype(int),
            "suid_total_count": np.random.randint(10, 22, n_low),

            "sudo_has_all": np.random.choice([0, 1], n_low, p=[0.97, 0.03]).astype(int),
            "sudo_has_nopasswd": np.random.choice([0, 1], n_low, p=[0.99, 0.01]).astype(int),
            "euid_is_root": np.zeros(n_low, dtype=int),

            "risk_label": np.zeros(n_low, dtype=int)
        })

        medium = pd.DataFrame({
            "kernel_major": np.random.choice([5, 6], n_med),
            "kernel_minor": np.random.randint(8, 22, n_med),
            "kernel_patch": np.random.randint(0, 12, n_med),
            "kernel_flavor": np.random.choice(["ubuntu", "debian", "standard"], n_med),

            "suid_pkexec": np.random.choice([0, 1], n_med, p=[0.85, 0.15]).astype(int),
            "suid_su": np.ones(n_med, dtype=int),
            "suid_sudo": np.ones(n_med, dtype=int),
            "suid_mount": np.ones(n_med, dtype=int),
            "suid_passwd": np.ones(n_med, dtype=int),
            "suid_chsh": np.random.choice([0, 1], n_med, p=[0.5, 0.5]).astype(int),
            "suid_total_count": np.random.randint(18, 35, n_med),

            "sudo_has_all": np.random.choice([0, 1], n_med, p=[0.7, 0.3]).astype(int),
            "sudo_has_nopasswd": np.random.choice([0, 1], n_med, p=[0.9, 0.1]).astype(int),
            "euid_is_root": np.zeros(n_med, dtype=int),

            "risk_label": np.ones(n_med, dtype=int)
        })

        high = pd.DataFrame({
            "kernel_major": np.random.choice([4, 5, 6], n_high),
            "kernel_minor": np.random.randint(0, 18, n_high),
            "kernel_patch": np.random.randint(0, 10, n_high),
            "kernel_flavor": np.random.choice(["kali-amd64", "debian", "standard"], n_high),

            "suid_pkexec": np.random.choice([0, 1], n_high, p=[0.25, 0.75]).astype(int),
            "suid_su": np.ones(n_high, dtype=int),
            "suid_sudo": np.ones(n_high, dtype=int),
            "suid_mount": np.ones(n_high, dtype=int),
            "suid_passwd": np.ones(n_high, dtype=int),
            "suid_chsh": np.ones(n_high, dtype=int),
            "suid_total_count": np.random.randint(28, 60, n_high),

            "sudo_has_all": np.random.choice([0, 1], n_high, p=[0.25, 0.75]).astype(int),
            "sudo_has_nopasswd": np.random.choice([0, 1], n_high, p=[0.8, 0.2]).astype(int),
            "euid_is_root": np.zeros(n_high, dtype=int),

            "risk_label": np.full(n_high, 2, dtype=int)
        })

        df = pd.concat([low, medium, high]).sample(frac=1, random_state=42).reset_index(drop=True)

        # One-hot encode kernel_flavor
        df = pd.get_dummies(df, columns=["kernel_flavor"], dtype=int)

        ensure_parent_dir(self.synthetic_data_path)
        df.to_csv(self.synthetic_data_path, index=False)
        logging.info(f"Synthetic 3-class dataset saved: {self.synthetic_data_path} (shape={df.shape})")
        return df

    # ---------------------------
    # Phase 2: Train
    # ---------------------------
    def train(self, df: pd.DataFrame) -> None:
        logging.info("Initiating Random Forest training sequence (3-class)...")

        X = df.drop("risk_label", axis=1)
        y = df["risk_label"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        self.trained_features = X.columns.tolist()
        joblib.dump({"model": self.model, "features": self.trained_features}, self.model_path)

        accuracy = self.model.score(X_test, y_test)
        logging.info(f"Model trained. Validation Accuracy: {accuracy * 100:.2f}%")

        y_pred = self.model.predict(X_test)

        print("\n[+] Classification Report:\n")
        print(classification_report(
            y_test,
            y_pred,
            digits=4,
            target_names=[self.inv_label_map[0], self.inv_label_map[1], self.inv_label_map[2]]
        ))

        print("\n[+] Confusion Matrix (rows=true, cols=pred):\n")
        print(confusion_matrix(y_test, y_pred))

        importances = pd.Series(self.model.feature_importances_, index=self.trained_features)
        print("\n[+] Top Feature Importances:\n")
        print(importances.sort_values(ascending=False).head(12).to_string())

    def load(self) -> None:
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.trained_features = payload["features"]

    # ---------------------------
    # Phase 3: Train on real dataset.csv (preferred)
    # ---------------------------
    def try_train_on_real_dataset(self, min_rows: int = 30, min_per_class: int = 8) -> bool:
        """
        Prefer real dataset.csv if it has enough labelled rows for all 3 classes.
        Expects dataset.csv to include:
          - label column with values: low / medium / high
          - feature columns (including kernel_flavor string)
        """
        if not os.path.exists(self.dataset_path) or os.path.getsize(self.dataset_path) == 0:
            return False

        df = pd.read_csv(self.dataset_path)
        if "label" not in df.columns:
            return False

        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df = df[df["label"].isin(self.label_map.keys())].copy()

        if df.shape[0] < min_rows:
            logging.info(f"Real dataset exists but not enough labelled rows yet: {df.shape[0]}/{min_rows}")
            return False

        df["risk_label"] = df["label"].map(self.label_map).astype(int)

        counts = df["risk_label"].value_counts().to_dict()
        for cls in [0, 1, 2]:
            if counts.get(cls, 0) < min_per_class:
                logging.info(f"Not enough samples for class {self.inv_label_map[cls]}: {counts.get(cls, 0)}/{min_per_class}")
                return False

        drop_cols = [c for c in ["timestamp_utc", "host_id", "label"] if c in df.columns]
        X_df = df.drop(columns=drop_cols)

        if "kernel_flavor" in X_df.columns:
            X_df = pd.get_dummies(X_df, columns=["kernel_flavor"], dtype=int)

        train_df = X_df.copy()
        train_df["risk_label"] = df["risk_label"].values

        logging.info(f"Training on REAL dataset.csv (3-class): rows={train_df.shape[0]}")
        self.train(train_df)
        return True

    # ---------------------------
    # Phase 4: Inference
    # ---------------------------
    def infer(self, target_csv: str) -> None:
        logging.info("Executing inference on target vector...")
        target_df = pd.read_csv(target_csv)

        # Strip metadata if present
        metadata_cols = ["timestamp_utc", "host_id", "label"]
        target_df = target_df.drop(columns=[c for c in metadata_cols if c in target_df.columns], errors="ignore")

        if "kernel_flavor" in target_df.columns:
            target_df = pd.get_dummies(target_df, columns=["kernel_flavor"], dtype=int)

        target_df = target_df.reindex(columns=self.trained_features, fill_value=0)

        probs = self.model.predict_proba(target_df)[0]
        pred_cls = int(np.argmax(probs))
        pred_label = self.inv_label_map[pred_cls]

        print("\n" + "=" * 62)
        print("              PRIV-EYE POSTURE RISK ASSESSMENT")
        print("=" * 62)

        print(f"Prediction: {pred_label.upper()} RISK")
        print(f"Probabilities: low={probs[0]*100:.1f}% | medium={probs[1]*100:.1f}% | high={probs[2]*100:.1f}%")

        if pred_label == "high":
            print("[!] Host posture aligns with permissive privilege baselines (wide sudo / high SUID surface).")
        elif pred_label == "medium":
            print("[-] Host exhibits moderate divergence from enterprise security baselines.")
        else:
            print("[+] Host aligns with locked-down enterprise baselines.")

        print("=" * 62 + "\n")


if __name__ == "__main__":
    oracle = PrivEyeOracle(DATA_DIR)

    # Optional: build a real row each time you run (keeps dataset growing)
    # Change label to low/medium/high when you have a ground-truth view of the host.
    oracle.build_dataset_row(label="unknown", timeout=10)

    # Prefer real training if enough labelled rows exist; otherwise fallback to synthetic
    trained_real = oracle.try_train_on_real_dataset(min_rows=30, min_per_class=8)
    if not trained_real:
        dataset = oracle.synthesize_data_3class(num_samples=1200)
        oracle.train(dataset)

    oracle.infer(oracle.feature_path)
