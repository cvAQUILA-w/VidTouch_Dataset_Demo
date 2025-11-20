from models.multitask_classifier import MultiTaskClassifier
import copy
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt

from dataset.fabric_dataset import FabricDataset
from models.mlp_classifier import MultiLabelMLP
from utils.label_utils import LabelEncoder, MultiHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

label_file = "label.txt"
feature_dir = "new_fused_pic"
input_dim = 2600
hidden_dim = 2600
batch_size = 16
epochs = 160
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

entries = open(label_file).readlines()
weave_set, material_set, usage_set, feature_set = set(), set(), set(), set()

for line in entries:
    parts = line.strip().split()
    weave_set.add(parts[1])
    material_set.add(parts[2])
    usage_set.add(parts[3])
    feature_set.update(parts[4:])

weave_enc = LabelEncoder(sorted(weave_set))
material_enc = LabelEncoder(sorted(material_set))
usage_enc = LabelEncoder(sorted(usage_set))
feature_enc = MultiHotEncoder(sorted(feature_set))

val_label_file = "val_label.txt"
val_feature_dir = "val_new_fused_pic"

val_dataset = FabricDataset(val_label_file, val_feature_dir,weave_enc, material_enc, usage_enc, feature_enc)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
print("Number of validation samples loaded:", len(val_dataset))

dataset = FabricDataset(label_file, feature_dir, weave_enc, material_enc, usage_enc, feature_enc)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print("Number of samples loaded:", len(dataset))

weave_train_classes = set()
material_train_classes = set()
usage_train_classes = set()

for i in range(len(dataset)):
    _, y_w, y_m, y_u, _ = dataset[i]
    weave_train_classes.add(y_w)
    material_train_classes.add(y_m)
    usage_train_classes.add(y_u)

model = MultiTaskClassifier(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_weave=len(weave_enc.class2idx),
    num_material=len(material_enc.class2idx),
    num_usage=len(usage_enc.class2idx),
    num_features=len(feature_enc.vocab),
    drop_path_rate=0.1
).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, dataloader, loss_ce, loss_bce, device,
             weave_train_classes, material_train_classes, usage_train_classes):
    model.eval()

    total_loss = 0
    count = 0

    y_true_w, y_pred_w = [], []
    y_true_m, y_pred_m = [], []
    y_true_u, y_pred_u = [], []

    y_true_f, y_pred_f = [], []

    for x, y_w, y_m, y_u, y_f in dataloader:
        x = x.to(device)
        y_w, y_m, y_u, y_f = y_w.to(device), y_m.to(device), y_u.to(device), y_f.to(device)

        pred_w, pred_m, pred_u, pred_f = model(x)

        loss = (
            loss_ce(pred_w, y_w) +
            loss_ce(pred_m, y_m) +
            loss_ce(pred_u, y_u) +
            loss_bce(pred_f, y_f)
        )
        total_loss += loss.item()
        count += 1

        y_true_w.extend(y_w.cpu().numpy())
        y_pred_w.extend(torch.argmax(pred_w, dim=1).cpu().numpy())

        y_true_m.extend(y_m.cpu().numpy())
        y_pred_m.extend(torch.argmax(pred_m, dim=1).cpu().numpy())

        y_true_u.extend(y_u.cpu().numpy())
        y_pred_u.extend(torch.argmax(pred_u, dim=1).cpu().numpy())

        y_true_f.extend(y_f.cpu().numpy())
        y_pred_f.extend((torch.sigmoid(pred_f) > 0.5).int().cpu().numpy())

    avg_loss = total_loss / count

    def filter_valid(y_true, y_pred, valid_set):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = np.isin(y_true, list(valid_set))
        return y_true[mask], y_pred[mask]

    filtered_y_true_w, filtered_y_pred_w = filter_valid(y_true_w, y_pred_w, weave_train_classes)
    filtered_y_true_m, filtered_y_pred_m = filter_valid(y_true_m, y_pred_m, material_train_classes)
    filtered_y_true_u, filtered_y_pred_u = filter_valid(y_true_u, y_pred_u, usage_train_classes)

    metrics = {
        'avg_loss': avg_loss,

        'weave_acc': accuracy_score(filtered_y_true_w, filtered_y_pred_w),
        'weave_f1': f1_score(filtered_y_true_w, filtered_y_pred_w, average='macro'),

        'material_acc': accuracy_score(filtered_y_true_m, filtered_y_pred_m),
        'material_f1': f1_score(filtered_y_true_m, filtered_y_pred_m, average='macro'),

        'usage_acc': accuracy_score(filtered_y_true_u, filtered_y_pred_u),
        'usage_f1': f1_score(filtered_y_true_u, filtered_y_pred_u, average='macro'),

        'feature_precision': precision_score(y_true_f, y_pred_f, average='macro', zero_division=0),
        'feature_recall': recall_score(y_true_f, y_pred_f, average='macro', zero_division=0),
        'feature_f1': f1_score(y_true_f, y_pred_f, average='macro', zero_division=0)
    }

    def plot_cm(cm, labels, title, filename):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    all_weave_labels = list(range(len(weave_enc.class2idx)))
    all_material_labels = list(range(len(material_enc.class2idx)))
    all_usage_labels = list(range(len(usage_enc.class2idx)))

    cm_weave = confusion_matrix(y_true_w, y_pred_w, labels=all_weave_labels)
    cm_material = confusion_matrix(y_true_m, y_pred_m, labels=all_material_labels)
    cm_usage = confusion_matrix(y_true_u, y_pred_u, labels=all_usage_labels)

    plot_cm(cm_weave, weave_enc.idx2class.values(), "Weave Confusion Matrix", "cm_weave.png")
    plot_cm(cm_material, material_enc.idx2class.values(), "Material Confusion Matrix", "cm_material.png")
    plot_cm(cm_usage, usage_enc.idx2class.values(), "Usage Confusion Matrix", "cm_usage.png")

    pd.DataFrame(cm_weave, index=weave_enc.idx2class.values(), columns=weave_enc.idx2class.values()).to_csv(
        "cm_weave.csv")
    pd.DataFrame(cm_material, index=material_enc.idx2class.values(), columns=material_enc.idx2class.values()).to_csv(
        "cm_material.csv")
    pd.DataFrame(cm_usage, index=usage_enc.idx2class.values(), columns=usage_enc.idx2class.values()).to_csv(
        "cm_usage.csv")

    return metrics

train_loss_list = []
val_loss_list = []

val_weave_f1_list = []
val_material_f1_list = []
val_usage_f1_list = []
val_feature_f1_list = []

val_weave_acc_list = []
val_material_acc_list = []
val_usage_acc_list = []

#EarlyStopping
patience = 160
best_val_loss = float('inf')
best_model_state = None
epoch_since_improvement = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    for x, y_w, y_m, y_u, y_f in dataloader:
        x = x.to(device)
        y_w, y_m, y_u, y_f = y_w.to(device), y_m.to(device), y_u.to(device), y_f.to(device)

        optimizer.zero_grad()
        pred_w, pred_m, pred_u, pred_f = model(x)

        loss = (
            loss_ce(pred_w, y_w) +
            loss_ce(pred_m, y_m) +
            loss_ce(pred_u, y_u) +
            loss_bce(pred_f, y_f)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_train_loss = total_loss / len(dataloader)
    train_loss_list.append(avg_train_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

    val_metrics = evaluate(model, val_loader, loss_ce, loss_bce, device,
                           weave_train_classes, material_train_classes, usage_train_classes)
    scheduler.step(val_metrics['avg_loss'])
    val_loss = val_metrics['avg_loss']
    val_loss_list.append(val_metrics['avg_loss'])
    val_weave_f1_list.append(val_metrics['weave_f1'])
    val_material_f1_list.append(val_metrics['material_f1'])
    val_usage_f1_list.append(val_metrics['usage_f1'])
    val_feature_f1_list.append(val_metrics['feature_f1'])
    val_weave_acc_list.append(val_metrics['weave_acc'])
    val_material_acc_list.append(val_metrics['material_acc'])
    val_usage_acc_list.append(val_metrics['usage_acc'])

    print(f"Validation Loss: {val_metrics['avg_loss']:.4f} | "
          f"Weave F1: {val_metrics['weave_f1']:.4f} | "
          f"Material F1: {val_metrics['material_f1']:.4f} | "
          f"Usage F1: {val_metrics['usage_f1']:.4f} | "
          f"Feature F1: {val_metrics['feature_f1']:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epoch_since_improvement = 0
        print("New best model saved.")
    else:
        epoch_since_improvement += 1
        print(f"No improvement for {epoch_since_improvement} epochs.")

    if epoch_since_improvement >= patience:
        print("Early stopping triggered.")
        break

torch.save(best_model_state, "best_model.pt")
print("Best model saved to best_model.pt")

torch.save(model.state_dict(), "mlp_model_v0.1.pt")

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_loss_list, label="Train Loss")
plt.plot(epochs_range, val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.savefig("loss_curve_v0.1.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, val_weave_f1_list, label="Weave F1")
plt.plot(epochs_range, val_material_f1_list, label="Material F1")
plt.plot(epochs_range, val_usage_f1_list, label="Usage F1")
plt.plot(epochs_range, val_feature_f1_list, label="Feature F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Scores")
plt.legend()
plt.savefig("val_f1_curve_v0.1.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, val_weave_acc_list, label="Weave Acc")
plt.plot(epochs_range, val_material_acc_list, label="Material Acc")
plt.plot(epochs_range, val_usage_acc_list, label="Usage Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Top-1 Accuracies")
plt.legend()
plt.savefig("val_acc_curve_v0.1.png")
plt.close()

with open("log_v0.1.txt", "w") as f:
    f.write("Epoch\tTrainLoss\tValLoss\tWeaveF1\tMaterialF1\tUsageF1\tFeatureF1\tWeaveAcc\tMaterialAcc\tUsageAcc\n")
    for i in range(epochs):
        f.write(f"{i+1}\t{train_loss_list[i]:.4f}\t{val_loss_list[i]:.4f}\t"
                f"{val_weave_f1_list[i]:.4f}\t{val_material_f1_list[i]:.4f}\t"
                f"{val_usage_f1_list[i]:.4f}\t{val_feature_f1_list[i]:.4f}\t"
                f"{val_weave_acc_list[i]:.4f}\t{val_material_acc_list[i]:.4f}\t{val_usage_acc_list[i]:.4f}\n")