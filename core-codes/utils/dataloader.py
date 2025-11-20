import torch
from torch.utils.data import Dataset
import os

class FabricDataset(Dataset):
    def __init__(self, feature_dir, label_path, is_train=True, split_ratio=0.8):
        self.feature_dir = feature_dir
        self.label_map = {}
        self.data = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                fid = parts[0]
                labels = parts[1:]
                self.label_map[fid] = labels

        all_feats = []
        for file in os.listdir(feature_dir):
            if file.endswith(".pt"):
                fid = file.split('(')[0].strip()
                if fid in self.label_map:
                    all_feats.append((file, fid, self.label_map[fid]))

        # dividing dataset
        split_idx = int(len(all_feats) * split_ratio)
        if is_train:
            self.data = all_feats[:split_idx]
        else:
            self.data = all_feats[split_idx:]

        self.label_encoders = self.build_encoders()

    def build_encoders(self):
        from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
        encoders = []
        label_lists = list(zip(*[entry[2] for entry in self.data]))
        for i in range(4):
            if i < 3:
                le = LabelEncoder()
                le.fit(label_lists[i])
                encoders.append(le)
            else:
                mlb = MultiLabelBinarizer()
                mlb.fit([labels[i:] for labels in self.data])
                encoders.append(mlb)
        return encoders

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, fid, labels = self.data[idx]
        feat = torch.load(os.path.join(self.feature_dir, fname)).squeeze()
        encoded = []
        for i in range(4):
            if i < 3:
                encoded.append(self.label_encoders[i].transform([labels[i]])[0])
            else:
                encoded.append(
                    torch.tensor(
                        self.label_encoders[i].transform([labels[i:]])[0], dtype=torch.float
                    )
                )
        return feat, *encoded