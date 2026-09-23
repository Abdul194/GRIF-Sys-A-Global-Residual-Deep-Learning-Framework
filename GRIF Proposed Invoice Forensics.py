"""
Global-Residual Invoice Forensics Network (GRIF-Sys)

Standalone .py conversion of the supplied notebook.

Important:
- The model architecture is preserved.
- ResNet50 remains fully frozen.
- The forensic residual branch and fixed high-pass filters are unchanged.
- AdamW, learning rate, weight decay, loss, metric, callbacks, batch size,
  split strategy, seed, epochs, and regularization values are unchanged.
- No data augmentation has been introduced because it was not part of the
  supplied proposed-model notebook.

Expected dataset structure:
dataset_path/
    Real-Invoices/
    Fake-Invoices/
    Temp-Invoices/

Example:
python grif_proposed_invoice_forensics.py \
    --dataset-path "/path/to/Local_Dataset" \
    --result-path "/path/to/Results"
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# ORIGINAL CONFIGURATION
# ============================================================

SEED = 42

CLASS_NAMES = [
    "Real-Invoices",
    "Fake-Invoices",
    "Temp-Invoices",
]

NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 16

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

L2_WEIGHT = 1e-4
GLOBAL_DROPOUT = 0.50
FUSION_DROPOUT = 0.50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

def load_dataset_information(dataset_dir):
    file_paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_path = Path(dataset_dir) / class_name

        if not class_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {class_path}"
            )

        for file_path in class_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                file_paths.append(str(file_path))
                labels.append(class_index)

    dataset_df = pd.DataFrame({
        "filepath": file_paths,
        "label": labels,
    })

    if dataset_df.empty:
        raise ValueError("No supported images were found in the dataset.")

    print("\nTotal Images:", len(dataset_df))
    print("\nClass Distribution:")

    for class_index, class_name in enumerate(CLASS_NAMES):
        count = np.sum(dataset_df["label"] == class_index)
        print(f"{class_name}: {count}")

    return dataset_df


# ============================================================
# ORIGINAL STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

def split_dataset(dataset_df):
    train_df, temp_df = train_test_split(
        dataset_df,
        test_size=0.30,
        stratify=dataset_df["label"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nDataset Split")
    print("-" * 40)
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    print("\nTraining Distribution")
    for i, name in enumerate(CLASS_NAMES):
        print(name, np.sum(train_df["label"] == i))

    print("\nValidation Distribution")
    for i, name in enumerate(CLASS_NAMES):
        print(name, np.sum(val_df["label"] == i))

    print("\nTesting Distribution")
    for i, name in enumerate(CLASS_NAMES):
        print(name, np.sum(test_df["label"] == i))

    return train_df, val_df, test_df


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(filepath, label):
    image = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        method="bilinear",
    )

    image = tf.cast(image, tf.float32)

    return image, label


# ============================================================
# ORIGINAL DATA PIPELINE
# ============================================================

def create_dataset(dataframe, training=False):
    filepaths = dataframe["filepath"].values
    labels_array = dataframe["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels_array)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# HIGH-FREQUENCY FORENSIC RESIDUAL LAYER
# ============================================================

@keras.utils.register_keras_serializable(package="GRIF")
class ForensicResidualLayer(layers.Layer):
    """
    Fixed high-pass filtering layer from the original notebook.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        laplacian = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ], dtype=np.float32)

        horizontal = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ], dtype=np.float32)

        vertical = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ], dtype=np.float32)

        kernels = np.stack(
            [laplacian, horizontal, vertical],
            axis=-1,
        )

        kernels = kernels[:, :, np.newaxis, :]

        self.kernel = tf.constant(
            kernels,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(self, inputs):
        gray = tf.image.rgb_to_grayscale(
            inputs / 255.0
        )

        residual = tf.nn.conv2d(
            gray,
            self.kernel,
            strides=1,
            padding="SAME",
        )

        residual = tf.abs(residual)

        return residual

    def get_config(self):
        return super().get_config()


# ============================================================
# ORIGINAL LIGHTWEIGHT FORENSIC BRANCH
# ============================================================

def build_forensic_branch(forensic_input):
    # Block 1
    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
    )(forensic_input)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 3
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
        name="forensic_feature",
    )(x)

    x = layers.Dropout(0.30)(x)

    return x


# ============================================================
# ORIGINAL PROPOSED MODEL ARCHITECTURE
# ============================================================

def build_proposed_model():
    image_input = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        name="invoice_image",
    )

    # Global RGB branch
    resnet_backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    )

    # Original notebook: freeze complete ResNet50 backbone.
    resnet_backbone.trainable = False

    global_input = (
        tf.keras.applications.resnet50.preprocess_input(
            image_input
        )
    )

    global_feature_map = resnet_backbone(
        global_input,
        training=False,
    )

    global_feature = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(global_feature_map)

    global_feature = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
        name="global_feature",
    )(global_feature)

    global_feature = layers.Dropout(
        GLOBAL_DROPOUT,
        name="global_dropout",
    )(global_feature)

    # Forensic residual branch
    forensic_residual = ForensicResidualLayer(
        name="forensic_residual_layer"
    )(image_input)

    forensic_feature = build_forensic_branch(
        forensic_residual
    )

    # Feature fusion
    fused_feature = layers.Concatenate(
        name="global_forensic_fusion"
    )([
        global_feature,
        forensic_feature,
    ])

    fused_feature = layers.BatchNormalization(
        name="fusion_batch_norm"
    )(fused_feature)

    fused_feature = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
        name="fusion_dense",
    )(fused_feature)

    fused_feature = layers.Dropout(
        FUSION_DROPOUT,
        name="fusion_dropout",
    )(fused_feature)

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification",
    )(fused_feature)

    model = keras.Model(
        inputs=image_input,
        outputs=output,
        name="Global_Residual_Invoice_Forensics_Network",
    )

    return model, resnet_backbone


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

def save_training_curves(history, result_path):
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "Epoch"
    history_df.to_csv(
        result_path / "Training_History.csv"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        result_path / "Training_Validation_Accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        result_path / "Training_Validation_Loss.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm, result_path):
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        result_path / "Proposed_Model_Confusion_Matrix.csv"
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Proposed Model Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=45,
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        result_path / "Proposed_Model_Confusion_Matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main(dataset_path, result_path):
    set_reproducibility()

    dataset_path = Path(dataset_path).expanduser().resolve()
    result_path = Path(result_path).expanduser().resolve()
    result_path.mkdir(parents=True, exist_ok=True)

    model_path = (
        result_path
        / "Proposed_Invoice_Forensics_Model.keras"
    )

    print("TensorFlow Version:", tf.__version__)
    print("Dataset Path:", dataset_path)
    print("Result Path:", result_path)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset_df = load_dataset_information(dataset_path)

    train_df, val_df, test_df = split_dataset(
        dataset_df
    )

    # Save the exact split used.
    split_path = result_path / "Dataset_Splits"
    split_path.mkdir(exist_ok=True)

    train_df.to_csv(
        split_path / "Train.csv",
        index=False,
    )
    val_df.to_csv(
        split_path / "Validation.csv",
        index=False,
    )
    test_df.to_csv(
        split_path / "Test.csv",
        index=False,
    )

    train_ds = create_dataset(
        train_df,
        training=True,
    )

    val_ds = create_dataset(
        val_df,
        training=False,
    )

    test_ds = create_dataset(
        test_df,
        training=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model, resnet_backbone = build_proposed_model()

    model.summary()

    trainable_params = np.sum([
        np.prod(v.shape)
        for v in model.trainable_weights
    ])

    non_trainable_params = np.sum([
        np.prod(v.shape)
        for v in model.non_trainable_weights
    ])

    print("\nTrainable Parameters:")
    print(f"{trainable_params:,}")

    print("\nNon-Trainable Parameters:")
    print(f"{non_trainable_params:,}")

    print("\nResNet50 Trainable:")
    print(resnet_backbone.trainable)

    # --------------------------------------------------------
    # ORIGINAL OPTIMIZER AND COMPILE CONFIGURATION
    # --------------------------------------------------------
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    # --------------------------------------------------------
    # ORIGINAL CALLBACK CONFIGURATION
    # --------------------------------------------------------
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # --------------------------------------------------------
    # ORIGINAL TRAINING METHOD
    # --------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("TRAINING PROPOSED GLOBAL-RESIDUAL FRAMEWORK")
    print("=" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_curves(
        history,
        result_path,
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------
    custom_objects = {
        "ForensicResidualLayer":
            ForensicResidualLayer
    }

    best_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
    )

    print("\nBest model successfully loaded.")

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------
    true_labels = []
    predicted_labels = []
    prediction_probabilities = []

    for images, labels_batch in test_ds:
        probabilities = best_model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        true_labels.extend(
            labels_batch.numpy()
        )

        predicted_labels.extend(
            predictions
        )

        prediction_probabilities.extend(
            probabilities
        )

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    prediction_probabilities = np.array(
        prediction_probabilities
    )

    # --------------------------------------------------------
    # ORIGINAL TEST METRICS
    # --------------------------------------------------------
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    print("\n")
    print("=" * 70)
    print("PROPOSED MODEL TEST PERFORMANCE")
    print("=" * 70)

    print(f"Accuracy        : {accuracy:.6f}")
    print(f"Accuracy (%)    : {accuracy * 100:.2f}%")
    print(f"Macro Precision : {precision:.6f}")
    print(f"Macro Recall    : {recall:.6f}")
    print(f"Macro F1-Score  : {f1:.6f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------
    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification Report\n")
    print(report_text)

    (
        result_path
        / "Proposed_Model_Classification_Report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        result_path
        / "Proposed_Model_Classification_Report.csv"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print("\nConfusion Matrix")
    print(cm)

    save_confusion_matrix(
        cm,
        result_path,
    )

    # --------------------------------------------------------
    # INDIVIDUAL TEST PREDICTIONS
    # --------------------------------------------------------
    results_df = pd.DataFrame({
        "True_Label": [
            CLASS_NAMES[i]
            for i in true_labels
        ],
        "Predicted_Label": [
            CLASS_NAMES[i]
            for i in predicted_labels
        ],
        "Probability_Real":
            prediction_probabilities[:, 0],
        "Probability_Fake":
            prediction_probabilities[:, 1],
        "Probability_Tampered":
            prediction_probabilities[:, 2],
    })

    prediction_file = (
        result_path
        / "Proposed_Model_Test_Predictions.csv"
    )

    results_df.to_csv(
        prediction_file,
        index=False,
    )

    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------
    performance_df = pd.DataFrame({
        "Model": [
            "Global-Residual Proposed Framework"
        ],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-Score": [f1],
    })

    performance_file = (
        result_path
        / "Proposed_Model_Performance.csv"
    )

    performance_df.to_csv(
        performance_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE CONFIGURATION / ARCHITECTURE INFORMATION
    # --------------------------------------------------------
    configuration = {
        "dataset_path": str(dataset_path),
        "result_path": str(result_path),
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "image_size": IMG_SIZE,
        "channels": CHANNELS,
        "batch_size": BATCH_SIZE,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "l2_weight": L2_WEIGHT,
        "global_dropout": GLOBAL_DROPOUT,
        "fusion_dropout": FUSION_DROPOUT,
        "forensic_branch_dropout": 0.30,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "SparseCategoricalCrossentropy",
        "metric": "SparseCategoricalAccuracy",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_monitor": "val_accuracy",
        "checkpoint_mode": "max",
        "early_stopping_monitor": "val_loss",
        "reduce_lr_monitor": "val_loss",
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 2,
        "reduce_lr_min_lr": 1e-7,
        "resnet50_weights": "imagenet",
        "resnet50_trainable": False,
        "augmentation": False,
        "tensorflow_version": tf.__version__,
    }

    with open(
        result_path / "Experiment_Configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )

    with open(
        result_path / "Model_Summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        model.summary(
            print_fn=lambda line: file.write(
                line + "\n"
            )
        )

    print("\nPredictions saved to:", prediction_file)
    print("Performance summary saved to:", performance_file)

    print("\nFinal Result\n")
    print(performance_df)

    print(
        "\nAll model and result files saved to:",
        result_path,
    )


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Global-Residual Invoice Forensics Network "
            "using the architecture and training configuration "
            "of the supplied notebook."
        )
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Dataset root containing Real-Invoices, "
            "Fake-Invoices, and Temp-Invoices."
        ),
    )

    parser.add_argument(
        "--result-path",
        required=True,
        help=(
            "Directory in which the trained model and "
            "all experimental results will be saved."
        ),
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    main(
        dataset_path=args.dataset_path,
        result_path=args.result_path,
    )
