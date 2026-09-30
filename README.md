[README.md](https://github.com/user-attachments/files/32872671/README.md)
[requirements.txt](https://github.com/user-attachments/files/32872687/requirements.txt)[oral_cancer_classification.py](https://github.com/user-attachments/files/32872779/oral_cancer_classification.py)
[oral_cancer_classification.ipynb](https://github.com/user-attachments/files/32872690/oral_cancer_classification.ipynb)
[oral_cancer_classification.py](https://github.com/user-attachments/files/32872809/oral_cancer_classification.py)
 Oral Cancer Image Classification
 EfficientNetV2-S | Binary classification | Reproducible research workflow

 Dataset: muhammadatef/oral-cancer-images-for-classification
 Split: 64% train / 16% validation / 20% test (stratified)
 Seed: 42
 Important:
 - The test set remains untouched during model development.
 - The classification threshold is optimized on the validation set only.
 - The final test metrics are computed once using the fixed validation-derived threshold.


 ==============================================================================
 ORAL CANCER CLASSIFICATION - REGULARIZED & LEAKAGE-SAFE VERSION
 Dataset: muhammadatef/oral-cancer-images-for-classification

 Backbone: EfficientNetV2-S
 Input: 224x224
 Max Epochs: 30
 Stratified split: 64% train / 16% validation / 20% test

 Includes:
 - Stratified splitting
 - Controlled augmentation
 - L2 regularization
 - Dropout
 - Class weighting
 - Early stopping
 - ReduceLROnPlateau
 - Best-model checkpoint
 - Backup/restore
 - Threshold optimization using VALIDATION ONLY
 - Accuracy / Precision / Sensitivity / Specificity / F1 / AUC
 - Confusion Matrix
 - ROC curve
 - Precision-Recall curve
 - Training curves
 - Overfitting analysis
 ==============================================================================


 ============================================================
 1. INSTALL / IMPORT
 ============================================================

!pip install -q kagglehub

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)

 Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


 ============================================================
 2. DOWNLOAD DATASET
 ============================================================

print("\nDownloading dataset...")

dataset_root = kagglehub.dataset_download(
    "muhammadatef/oral-cancer-images-for-classification"
)

print("Dataset downloaded to:")
print(dataset_root)


 ============================================================
 3. FIND NORMAL / CANCER DIRECTORIES
 ============================================================

all_dirs = []

for root, dirs, files in os.walk(dataset_root):
    for d in dirs:
        all_dirs.append(os.path.join(root, d))

print("\nSearching for class folders...")

normal_dir = None
cancer_dir = None

for d in all_dirs:

    name = os.path.basename(d).lower().replace("_", " ").replace("-", " ")

    if "normal" in name:
        normal_dir = d

    if "cancer" in name:
        cancer_dir = d

if normal_dir is None or cancer_dir is None:
    raise ValueError(
        "Could not automatically find Normal and Cancer folders."
    )

print("Normal folder:", normal_dir)
print("Cancer folder:", cancer_dir)


 ============================================================
 4. COLLECT IMAGE PATHS
 ============================================================

def get_images(folder):

    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:

        images.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    return sorted(images)


normal_images = get_images(normal_dir)
cancer_images = get_images(cancer_dir)

print("\nImage counts:")
print("Normal:", len(normal_images))
print("Cancer:", len(cancer_images))
print("Total :", len(normal_images) + len(cancer_images))


if len(normal_images) == 0 or len(cancer_images) == 0:
    raise ValueError("One of the classes contains zero images!")


 ============================================================
 5. CREATE LABELS
 ============================================================
 Normal = 0
 Oral Cancer = 1

image_paths = np.array(
    normal_images + cancer_images
)

labels = np.array(
    [0] * len(normal_images) +
    [1] * len(cancer_images)
)


 ============================================================
 6. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
 ============================================================

 First:
 80% development
 20% untouched test

X_dev, X_test, y_dev, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

 Then:
 80% train
 20% validation

 Final:
 Train      = 64%
 Validation = 16%
 Test       = 20%

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=SEED,
    stratify=y_dev
)


 ============================================================
 7. VERIFY SPLITS
 ============================================================

print("\n================ DATA SPLIT ================")

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nTrain class distribution:")
print("Normal:", np.sum(y_train == 0))
print("Cancer:", np.sum(y_train == 1))

print("\nValidation class distribution:")
print("Normal:", np.sum(y_val == 0))
print("Cancer:", np.sum(y_val == 1))

print("\nTest class distribution:")
print("Normal:", np.sum(y_test == 0))
print("Cancer:", np.sum(y_test == 1))


 Safety check
assert len(set(X_train) & set(X_val)) == 0
assert len(set(X_train) & set(X_test)) == 0
assert len(set(X_val) & set(X_test)) == 0

print("\n✓ No image overlap between train / validation / test.")


 ============================================================
 8. PARAMETERS
 ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 30

 Runtime-only directories. These are not part of the GitHub repository.
PROJECT_DIR = "/content/oral_cancer_project"
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


 ============================================================
 9. IMAGE LOADING
 ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


def create_dataset(
    paths,
    labels,
    shuffle=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


train_ds = create_dataset(
    X_train,
    y_train,
    shuffle=True
)

val_ds = create_dataset(
    X_val,
    y_val,
    shuffle=False
)

test_ds = create_dataset(
    X_test,
    y_test,
    shuffle=False
)


 ============================================================
 10. CLASS WEIGHTS
 ============================================================

train_counts = np.bincount(y_train)

class_weights = {

    0: len(y_train) / (
        2 * train_counts[0]
    ),

    1: len(y_train) / (
        2 * train_counts[1]
    )
}

print("\nClass weights:")
print(class_weights)


 ============================================================
 11. DATA AUGMENTATION
 ============================================================

data_augmentation = tf.keras.Sequential(

    [

         Horizontal flip is usually reasonable
         for oral photographs.
        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.06
        ),

        tf.keras.layers.RandomZoom(
            height_factor=(-0.08, 0.08),
            width_factor=(-0.08, 0.08)
        ),

        tf.keras.layers.RandomContrast(
            0.08
        ),

    ],

    name="controlled_augmentation"
)


 ============================================================
 12. BUILD MODEL
 ============================================================

def build_model():

    base_model = tf.keras.applications.EfficientNetV2S(

        input_shape=(224, 224, 3),

        include_top=False,

        weights="imagenet",

         EfficientNetV2 already includes
         its preprocessing layer by default.
        include_preprocessing=True
    )

     Freeze pretrained backbone initially
    base_model.trainable = False


    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="image"
    )


     Augmentation only during training
    x = data_augmentation(inputs)


     EfficientNetV2
    x = base_model(
        x,
        training=False
    )


    x = tf.keras.layers.GlobalAveragePooling2D()(x)


     Batch normalization
    x = tf.keras.layers.BatchNormalization()(x)


     Regularization
    x = tf.keras.layers.Dropout(
        0.40
    )(x)


     L2 regularized dense layer
    x = tf.keras.layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=tf.keras.regularizers.l2(
            1e-4
        )

    )(x)


    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Dropout(
        0.30
    )(x)


    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="cancer_probability"
    )(x)


    model = tf.keras.Model(
        inputs,
        outputs,
        name="OralCancer_EfficientNetV2S_Regularized"
    )


    return model


model = build_model()

model.summary()


 ============================================================
 13. COMPILE
 ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        )

    ]
)


 ============================================================
 14. CALLBACKS
 ============================================================

checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_oral_cancer_model.keras")


callbacks = [

    # Save only the best validation-AUC model
    tf.keras.callbacks.ModelCheckpoint(

        checkpoint_path,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1

    ),


    # Reduce LR when validation AUC plateaus
    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_auc",

        mode="max",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1

    ),


    # Stop when validation AUC stops improving
    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=5,

        min_delta=0.002,

        restore_best_weights=True,

        verbose=1

    ),

]


 ============================================================
 15. TRAIN
 ============================================================

print("\n================ TRAINING ================\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=MAX_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


 ============================================================
 16. LOAD BEST SAVED MODEL
 ============================================================

print("\nLoading best validation-AUC model...")

best_model = tf.keras.models.load_model(
    checkpoint_path
)

print("✓ Best model loaded.")


 ============================================================
 17. FIND BEST TRAINING EPOCH
 ============================================================

best_epoch = np.argmax(
    history.history["val_auc"]
) + 1

best_val_auc = max(
    history.history["val_auc"]
)

print("\n================ BEST EPOCH ================")

print(
    f"Best epoch based on Validation AUC: "
    f"{best_epoch}"
)

print(
    f"Best Validation AUC: "
    f"{best_val_auc:.4f}"
)


 ============================================================
 18. TRAINING CURVES
 ============================================================

epochs_range = range(
    1,
    len(history.history["loss"]) + 1
)


 ---------------- LOSS ----------------

plt.figure(figsize=(8,5))

plt.plot(
    epochs_range,
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history.history["val_loss"],
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ---------------- ACCURACY ----------------

plt.figure(figsize=(8,5))

plt.plot(
    epochs_range,
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ---------------- AUC ----------------

plt.figure(figsize=(8,5))

plt.plot(
    epochs_range,
    history.history["auc"],
    label="Training AUC"
)

plt.plot(
    epochs_range,
    history.history["val_auc"],
    label="Validation AUC"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")

plt.title(
    "Training vs Validation ROC-AUC"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ---------------- PRECISION ----------------

plt.figure(figsize=(8,5))

plt.plot(
    epochs_range,
    history.history["precision"],
    label="Training Precision"
)

plt.plot(
    epochs_range,
    history.history["val_precision"],
    label="Validation Precision"
)

plt.xlabel("Epoch")
plt.ylabel("Precision")

plt.title(
    "Training vs Validation Precision"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ---------------- RECALL ----------------

plt.figure(figsize=(8,5))

plt.plot(
    epochs_range,
    history.history["recall"],
    label="Training Recall"
)

plt.plot(
    epochs_range,
    history.history["val_recall"],
    label="Validation Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Recall / Sensitivity")

plt.title(
    "Training vs Validation Recall"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ============================================================
 19. PREDICTIONS ON VALIDATION
 ============================================================

val_probs = best_model.predict(
    val_ds,
    verbose=0
).flatten()


 ============================================================
 20. FIND OPTIMAL THRESHOLD USING VALIDATION ONLY
 ============================================================

thresholds = np.arange(
    0.20,
    0.81,
    0.01
)

val_accuracies = []

for threshold in thresholds:

    val_preds_temp = (
        val_probs >= threshold
    ).astype(int)

    acc_temp = accuracy_score(
        y_val,
        val_preds_temp
    )

    val_accuracies.append(
        acc_temp
    )


best_threshold = thresholds[
    np.argmax(val_accuracies)
]

best_val_accuracy = max(
    val_accuracies
)


print("\n================ THRESHOLD ================")

print(
    f"Optimal threshold from VALIDATION set: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation accuracy at threshold: "
    f"{best_val_accuracy*100:.2f}%"
)


 ============================================================
 21. THRESHOLD CURVE
 ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    thresholds,
    np.array(val_accuracies) * 100
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")

plt.ylabel("Validation Accuracy (%)")

plt.title(
    "Validation Accuracy vs Classification Threshold"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ============================================================
22. FINAL TEST PREDICTIONS
 ============================================================

test_probs = best_model.predict(
    test_ds,
    verbose=0
).flatten()


 IMPORTANT:
 Threshold was chosen using validation only.
test_preds = (
    test_probs >= best_threshold
).astype(int)


 ============================================================
 23. FINAL METRICS
 ============================================================

accuracy = accuracy_score(
    y_test,
    test_preds
)

precision = precision_score(
    y_test,
    test_preds,
    zero_division=0
)

sensitivity = recall_score(
    y_test,
    test_preds,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_preds,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probs
)


 ============================================================
 24. CONFUSION MATRIX
 ============================================================

cm = confusion_matrix(
    y_test,
    test_preds
)

TN, FP, FN, TP = cm.ravel()


specificity = TN / (
    TN + FP
)


=============================================================
 25. FINAL TEST EVALUATION
============================================================

print("\n")
print("=" * 70)
print("FINAL ORAL CANCER TEST EVALUATION")
print("=" * 70)

print(
    f"\nClassification threshold: {best_threshold:.2f}"
)

print(
    f"Accuracy:              {accuracy*100:.2f}%"
)

print(
    f"Precision:             {precision*100:.2f}%"
)

print(
    f"Sensitivity / Recall:  {sensitivity*100:.2f}%"
)

print(
    f"Specificity:            {specificity*100:.2f}%"
)

print(
    f"F1-Score:               {f1*100:.2f}%"
)

print(
    f"ROC-AUC:                {auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)


 ============================================================
 26. CLASSIFICATION REPORT
 ============================================================

print("\n================ CLASSIFICATION REPORT ================\n")

print(
    classification_report(

        y_test,

        test_preds,

        target_names=[
            "Normal",
            "Oral Cancer"
        ],

        digits=4,

        zero_division=0

    )
)


 ============================================================
 27. CONFUSION MATRIX VISUALIZATION
 ============================================================

plt.figure(figsize=(7,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=[
        "Normal",
        "Oral Cancer"
    ],

    yticklabels=[
        "Normal",
        "Oral Cancer"
    ]

)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title(
    "Confusion Matrix - Test Set"
)

plt.show()


 ============================================================
 28. ROC CURVE
 ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    test_probs
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Test Set"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ============================================================
 29. PRECISION-RECALL CURVE
 ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        test_probs
    )
)

average_precision = average_precision_score(
    y_test,
    test_probs
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"AP = {average_precision:.4f}"
)

plt.xlabel("Recall / Sensitivity")

plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Test Set"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ============================================================
 30. PROBABILITY DISTRIBUTION
 ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    test_probs[y_test == 0],
    bins=20,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    test_probs[y_test == 1],
    bins=20,
    alpha=0.6,
    label="Oral Cancer"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Threshold = {best_threshold:.2f}"
)

plt.xlabel(
    "Predicted Probability of Oral Cancer"
)

plt.ylabel("Number of Images")

plt.title(
    "Prediction Probability Distribution"
)

plt.legend()

plt.grid(alpha=0.25)

plt.show()


 ============================================================
 31. OVERFITTING DIAGNOSTIC
 ============================================================

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]

final_train_auc = history.history["auc"][-1]
final_val_auc = history.history["val_auc"][-1]

accuracy_gap = final_train_acc - final_val_acc
auc_gap = final_train_auc - final_val_auc


print("\n")
print("=" * 70)
print("OVERFITTING DIAGNOSTIC")
print("=" * 70)

print(
    f"\nFinal Train Accuracy: "
    f"{final_train_acc*100:.2f}%"
)

print(
    f"Final Validation Accuracy: "
    f"{final_val_acc*100:.2f}%"
)

print(
    f"Accuracy gap: "
    f"{accuracy_gap*100:.2f} percentage points"
)

print(
    f"\nFinal Train AUC: "
    f"{final_train_auc:.4f}"
)

print(
    f"Final Validation AUC: "
    f"{final_val_auc:.4f}"
)

print(
    f"AUC gap: "
    f"{auc_gap:.4f}"
)


if accuracy_gap > 0.10 and auc_gap > 0.10:

    print(
        "\n⚠️ Strong overfitting signal."
    )

elif accuracy_gap > 0.07 or auc_gap > 0.07:

    print(
        "\n⚠️ Moderate overfitting signal."
    )

else:

    print(
        "\n✓ No strong overfitting signal."
    )


============================================================
32. FINAL SUMMARY
============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Test Accuracy:      {accuracy*100:.2f}%"
)

print(
    f"Test Sensitivity:   {sensitivity*100:.2f}%"
)

print(
    f"Test Specificity:   {specificity*100:.2f}%"
)

print(
    f"Test F1:             {f1*100:.2f}%"
)

print(
    f"Test ROC-AUC:        {auc:.4f}"
)

print(
    f"Threshold:           {best_threshold:.2f}"
)

print(
    f"Best Epoch:          {best_epoch}"
)

print("\nTraining completed successfully.")

__pycache__/
.ipynb_checkpoints/
*.pyc
.env
/content/
*.keras
*.h5
checkpoints/
backup/
## Results

The final model achieved the following performance on the test set:

| Metric      | Test Set |
| ----------- | -------: |
| Accuracy    |   87.85% |
| Sensitivity |   89.05% |
| Specificity |   86.36% |
| F1-score    |   89.05% |
| ROC-AUC     |     0.90 |

These results are reported on the held-out test set after model development and validation-based threshold selection.

