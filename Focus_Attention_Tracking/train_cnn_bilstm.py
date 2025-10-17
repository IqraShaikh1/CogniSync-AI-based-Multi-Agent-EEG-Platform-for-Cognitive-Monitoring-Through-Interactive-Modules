# train_cnn_bilstm.py
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks

# SETTINGS - adjust paths if needed
OUT_DIR = r"C:\Users\major\Desktop\eeg data train"
X_train = np.load(OUT_DIR + "/X_train_long.npy")
X_test  = np.load(OUT_DIR + "/X_test_long.npy")

y_train = np.load(OUT_DIR + "/y_train_long.npy")
y_test  = np.load(OUT_DIR + "/y_test_long.npy")

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# compute num classes automatically
num_classes = len(np.unique(y_train))
print("num_classes =", num_classes)

# Compute class weights (float -> dictionary)
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, cw))
print("Class weights:", class_weight)

# Build model (CNN + Bidirectional LSTM)
input_shape = (X_train.shape[1], X_train.shape[2])  # (1000, 6)
model = models.Sequential([
    layers.Input(shape=input_shape),

    # CNN stack
    layers.Conv1D(64, 5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(256, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # BiLSTM stack
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.4),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[es, rl],
    class_weight=class_weight,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Predict + Confusion Matrix
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Labels: load encoder to map indices to names
le = joblib.load(OUT_DIR + "/label_encoder.joblib")
labels = list(le.classes_)
print("Labels:", labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_bilstm.png")
plt.close()

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# Save model and history
model.save("cnn_bilstm_focus_model_v1.keras")   # Keras native format
joblib.dump(history.history, "training_history_bilstm.joblib")
print("Saved model and history.")

# Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.plot(history.history["accuracy"], label="train"); plt.plot(history.history["val_accuracy"], label="val"); plt.legend(); plt.title("Accuracy")
plt.subplot(1,2,2); plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="val"); plt.legend(); plt.title("Loss")
plt.tight_layout(); plt.savefig("training_curves_bilstm.png"); plt.close()
