
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# 1. Tạo mã băm Wavelet cho ảnh
def wavelet_hash(img_path, wavelet="db4", level=3):
    img = Image.open(img_path).convert("L").resize((64, 64))
    cA = pywt.wavedec2(np.array(img), wavelet, level=level)[0]
    return (cA.flatten() > np.median(cA)).astype(int)


# 2. Tính khoảng cách Hamming
def hamming_distance(h1, h2):
    return np.mean(h1 != h2)


# 3. Danh sách ảnh và nhãn thực tế
data = [
    ("cat_1.png", "cat"), ("cat_2.png", "cat"),
    ("dog_1.png", "dog"), ("dog_2.png", "horse"),
    ("car_1.png", "car"), ("car_2.png", "car"),
    ("dog_3.png", "dog"), ("dog_4.png", "dog"),
]

hashes = [wavelet_hash(path) for path, _ in data]
labels = [label for _, label in data]


# 4. So sánh từng cặp ảnh
y_true, distances = [], []

for i in range(len(data)):
    for j in range(i + 1, len(data)):
        distances.append(hamming_distance(hashes[i], hashes[j]))
        y_true.append(1 if labels[i] == labels[j] else 0)

y_true = np.array(y_true)
distances = np.array(distances)


# 5. Phân loại và tính các chỉ số
threshold = 0.35
y_pred = (distances <= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_true, y_pred)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

scores = 1 - distances
fpr, tpr, _ = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)


# 6. Hiển thị biểu đồ và thông số chung một hình
fig, ax = plt.subplots(figsize=(9, 6))

# Vẽ ROC
ax.plot(
    fpr, tpr,
    color="blue",
    linewidth=2,
    label=f"Wavelet Hash (AUC = {roc_auc:.3f})"
)

ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")

ax.set_xlabel("FPR (1 - Specificity)", fontsize=11)
ax.set_ylabel("TPR (Sensitivity)", fontsize=11)
ax.set_title("ROC Curve - Wavelet Hash", fontsize=14, fontweight="bold")

ax.legend(loc="lower right")
ax.grid(alpha=0.3)


# Khung hiển thị thông số
metrics_text = (
    "EVALUATION METRICS\n"
    "---------------------------\n"
    f"Accuracy       : {accuracy:.2%}\n"
    f"Sensitivity    : {sensitivity:.2%}\n"
    f"Specificity    : {specificity:.2%}\n"
    f"AUC            : {roc_auc:.3f}\n"
    "---------------------------\n"
    f"Threshold      : {threshold}\n"
    f"TP             : {tp}\n"
    f"TN             : {tn}\n"
    f"FP             : {fp}\n"
    f"FN             : {fn}"
)

ax.text(
    0.05, 0.95,
    metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    ),
    family="monospace"
)

plt.tight_layout()

# Lưu ảnh kết quả
plt.savefig("ket_qua.png", dpi=300, bbox_inches="tight")

plt.show()

print("Đã lưu kết quả tại: ket_qua.png")
