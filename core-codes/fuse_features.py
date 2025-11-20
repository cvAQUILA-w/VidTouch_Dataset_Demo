import os
import torch
from mlp_fusion import MLPFusion
from tqdm import tqdm

IMG_FEAT_DIR = 'screenshotpt_tf'
VID_FEAT_DIR = 'TACspt_x3d'
FUSED_FEAT_DIR = 'new_fused_pic'

os.makedirs(FUSED_FEAT_DIR, exist_ok=True)

IMG_DIM = 552
VID_DIM = 2048
FUSED_DIM = 2600

fusion_model = MLPFusion(in_dim1=IMG_DIM, in_dim2=VID_DIM, hidden_dim=2600, out_dim=FUSED_DIM)
fusion_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model.to(device)

for filename in tqdm(os.listdir(IMG_FEAT_DIR)):
    if not filename.endswith('.pt'):
        continue

    img_path = os.path.join(IMG_FEAT_DIR, filename)
    vid_path = os.path.join(VID_FEAT_DIR, filename)
    fused_path = os.path.join(FUSED_FEAT_DIR, filename)

    if not os.path.exists(vid_path):
        print(f"Warning: missing video feature for {filename}")
        continue

    img_feat = torch.load(img_path).to(device)  # shape: [1, IMG_DIM]
    vid_feat = torch.load(vid_path).to(device)  # shape: [1, VID_DIM]

    print(f"Processing {filename}")
    print(f"  Image feat shape: {img_feat.shape}")
    print(f"  Video feat shape: {vid_feat.shape}")

    concat_dim = img_feat.shape[1] + vid_feat.shape[1]
    if concat_dim != fusion_model.fc1.in_features:
        print(f"Skipped: feature dim mismatch. Got {concat_dim}, expected {fusion_model.fc1.in_features}")
        continue

    with torch.no_grad():
        fused_feat = fusion_model(img_feat, vid_feat)  # shape: [1, FUSED_DIM]

    torch.save(fused_feat.cpu(), fused_path)