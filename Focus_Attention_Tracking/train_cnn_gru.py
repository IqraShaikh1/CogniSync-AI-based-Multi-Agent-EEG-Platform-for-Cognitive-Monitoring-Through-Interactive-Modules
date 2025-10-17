import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---------------------------
# Load preprocessed data
# ---------------------------
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------
# Build CNN + GRU Model
# ---------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),

    # --- CNN Layers ---
    layers.Conv1D(64, 5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(256, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # --- GRU Layers ---
    layers.GRU(128, return_sequences=True),
    layers.GRU(64),
    layers.Dropout(0.4),

    # --- Dense Layers ---
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(4, activation="softmax")  # 4 activities
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Callbacks
# ---------------------------
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ---------------------------
# Evaluate
# ---------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ---------------------------
# Confusion Matrix + Report
# ---------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
labels = ["baseline", "distraction", "focus1", "focus2"]  # adjust if different

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# ---------------------------
# Save Model + Training Curve
# ---------------------------
model.save("cnn_gru_focus_model_v2.h5")
print("Model saved as cnn_gru_focus_model_v2.h5")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_v2.png")
plt.close()
