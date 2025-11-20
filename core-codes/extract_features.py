import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorchvideo.models.hub import x3d_m

VIDEO_FOLDER = 'TACs'
FEATURE_FOLDER = 'TACspt_x3d'
NUM_FRAMES = 16
FRAME_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(FEATURE_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),  # [H, W, C] → [C, H, W]
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def video_to_tensor(path, num_frames=16, size=224):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        print(f"warning {os.path.basename(path)} clips not enough，skipped")
        return None

    frame_idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(transform(frame))  # [C, H, W]

    cap.release()

    if len(frames) < num_frames:
        print(f"warning {os.path.basename(path)} exist clip loading failed, skipped")
        return None

    video = torch.stack(frames, dim=1)  # [C, T, H, W]
    video = video.unsqueeze(0)          # [1, C, T, H, W]
    return video

class X3D_FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = x3d_m(pretrained=True)
        self.model.blocks[-1].proj = torch.nn.Identity() 
        self.model.blocks[-1].output_pool = torch.nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x).flatten(1)  # [B, 2048]

def extract_and_save_features():
    model = X3D_FeatureExtractor().to(DEVICE)
    model.eval()

    for file in os.listdir(VIDEO_FOLDER):
        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(VIDEO_FOLDER, file)
        feature_path = os.path.join(FEATURE_FOLDER, file + '.pt')

        if os.path.exists(feature_path):
            print(f"already exist {file}, skipped")
            continue

        print(f"processing{file}")
        video_tensor = video_to_tensor(video_path, num_frames=NUM_FRAMES, size=FRAME_SIZE)
        if video_tensor is None:
            continue

        video_tensor = video_tensor.to(DEVICE)
        feature = model(video_tensor)  # [1, 2048]
        torch.save(feature.cpu(), feature_path)

    print("completed")

if __name__ == '__main__':
    extract_and_save_features()