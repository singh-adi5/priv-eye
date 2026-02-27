import json
import os
import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MatrixTransformer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Defensive nomenclature: high-sensitivity privilege boundary binaries
        self.hvt_basenames = ["pkexec", "su", "sudo", "mount", "passwd", "chsh"]
        
        # Strict column enforcement for reproducible ML training
        self.expected_columns = [
            "kernel_major", "kernel_minor", "kernel_patch", "kernel_flavor",
            "suid_pkexec", "suid_su", "suid_sudo", "suid_mount", "suid_passwd", "suid_chsh",
            "suid_total_count", "sudo_has_all", "sudo_has_nopasswd", "euid_is_root"
        ]

    def load_state(self) -> dict:
        with open(self.data_path, "r") as f:
            return json.load(f)

    def vectorize_kernel(self, kernel_str: str) -> dict:
        features = {"kernel_major": 0, "kernel_minor": 0, "kernel_patch": 0, "kernel_flavor": "unknown"}
        match = re.search(r"(\d+)\.(\d+)\.(\d+)(.*)", kernel_str)
        if match:
            features["kernel_major"] = int(match.group(1))
            features["kernel_minor"] = int(match.group(2))
            features["kernel_patch"] = int(match.group(3))
            features["kernel_flavor"] = match.group(4).strip("-+") if match.group(4) else "standard"
        return features

    def encode_suids(self, suid_list: list) -> dict:
        encoded = {}
        found_basenames = [path.split('/')[-1] for path in suid_list]
        for target in self.hvt_basenames:
            encoded[f"suid_{target}"] = 1 if target in found_basenames else 0
        encoded["suid_total_count"] = len(suid_list)
        return encoded

    def parse_sudo(self, sudo_str: str, euid: int) -> dict:
        features = {
            "sudo_has_all": 1 if "(ALL : ALL) ALL" in sudo_str else 0,
            "sudo_has_nopasswd": 1 if "NOPASSWD" in sudo_str else 0,
            "euid_is_root": 1 if euid == 0 else 0
        }
        return features

    def generate_feature_vector(self) -> pd.DataFrame:
        logging.info("Initializing dimensional transformation...")
        raw_data = self.load_state()

        kernel_features = self.vectorize_kernel(raw_data.get("kernel_version", ""))
        suid_features = self.encode_suids(raw_data.get("suid_binaries", []))
        sudo_features = self.parse_sudo(raw_data.get("sudo_privileges", ""), raw_data.get("euid", 1000))

        feature_dict = {**kernel_features, **suid_features, **sudo_features}
        
        df = pd.DataFrame([feature_dict])
        df = df.reindex(columns=self.expected_columns, fill_value=0)
        return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "data", "raw_state.json")
    output_path = os.path.join(project_root, "data", "feature_vector.csv")

    transformer = MatrixTransformer(input_path)
    feature_matrix = transformer.generate_feature_vector()
    
    feature_matrix.to_csv(output_path, index=False)
    
    logging.info(f"Transformation complete. Feature vector dimensionality: {feature_matrix.shape}")
    print("\n[+] Enterprise Feature Matrix generated:\n")
    print(feature_matrix.to_string(index=False))
