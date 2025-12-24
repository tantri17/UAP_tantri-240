import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/test"
MODEL_PATH = "models/cnn_scratch.h5"

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

print("=== Classification Report (CNN Scratch) ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix - CNN Scratch")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
