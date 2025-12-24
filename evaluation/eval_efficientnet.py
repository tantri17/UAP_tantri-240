import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/test"
MODEL_PATH = "models/efficientnetb0.h5"

# =========================
# LOAD DATASET TEST
# =========================
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

# =========================
# PREPROCESS INPUT (WAJIB!)
# =========================
test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTION
# =========================
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

# =========================
# CLASSIFICATION REPORT
# =========================
print("=== Classification Report (EfficientNetB0) ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - EfficientNetB0")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
