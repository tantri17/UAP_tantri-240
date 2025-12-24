import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATH
# =========================
MODEL_PATH = "models/mobilenetv2.h5"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# DATA GENERATOR (TEST)
# =========================
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False   # ⚠️ PENTING
)

# =========================
# PREDICTION
# =========================
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

y_pred_prob = model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== Classification Report (MobileNetV2) ===\n")
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
plt.title("Confusion Matrix - MobileNetV2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
