# dataset/fabric_dataset.py
import os
import torch
from torch.utils.data import Dataset
from utils.label_utils import LabelEncoder, MultiHotEncoder

class FabricDataset(Dataset):
    def __init__(self, label_file, feature_dir, weave_enc, material_enc, usage_enc, feature_enc):
        self.entries = []
        self.feature_dir = feature_dir
        self.weave_enc = weave_enc
        self.material_enc = material_enc
        self.usage_enc = usage_enc
        self.feature_enc = feature_enc

        # label{id: (weave, material, usage, features)}
        self.label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                fabric_id = parts[0]
                self.label_dict[fabric_id] = (parts[1], parts[2], parts[3], parts[4:])

        for filename in os.listdir(feature_dir):
            if filename.endswith(".pt"):
                fabric_id = filename.split("(")[0].strip()
                if fabric_id in self.label_dict:
                    self.entries.append((os.path.join(feature_dir, filename), fabric_id))
        print("Total label entries:", len(self.label_dict))
        print("Total feature files matched:", len(self.entries))
        print("First 5 matched entries:")
        for i in range(min(5, len(self.entries))):
            print(self.entries[i])
        if fabric_id not in self.label_dict:
            print(f"Unmatched feature: {filename} → fabric_id='{fabric_id}'")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        feature_path, fabric_id = self.entries[idx]
        x = torch.load(feature_path).float()
        x = torch.load(feature_path).float().squeeze(0)  # shape: [256]

        weave, material, usage, features = self.label_dict[fabric_id]

        y_weave = self.weave_enc.encode(weave)
        y_material = self.material_enc.encode(material)
        y_usage = self.usage_enc.encode(usage)
        y_features = self.feature_enc.encode(features)

        return x, y_weave, y_material, y_usage, y_features