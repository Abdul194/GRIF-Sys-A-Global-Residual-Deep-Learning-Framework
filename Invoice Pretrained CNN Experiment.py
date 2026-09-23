"""
Pretrained CNN comparison for invoice image classification.

Expected dataset structure:
dataset_path/
    Class_A/
        image1.png
        ...
    Class_B/
        ...
    Class_C/
        ...

The script performs a stratified train/validation/test split, trains selected
ImageNet-pretrained CNNs with frozen backbones, optionally applies data
augmentation to the training stream, evaluates each model, and saves all
experiment artifacts under result_path.

Example:
    python invoice_pretrained_cnn_experiment.py \
        --dataset-path "/path/to/dataset" \
        --result-path "/path/to/results" \
        --augmentation yes
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_MODELS = (
    "MobileNetV3Small",
    "MobileNetV2",
    "EfficientNetB0",
    "DenseNet121",
    "ResNet50",
    "VGG16",
)


def parse_yes_no(value):
    value = str(value).strip().lower()
    if value in {"yes", "y", "true", "1"}:
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Use yes/no, y/n, true/false, or 1/0.")


def set_reproducibility(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    class_names = sorted(p.name for p in dataset_path.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise ValueError("Dataset must contain at least two class subdirectories.")

    rows = []
    for class_name in class_names:
        for file_path in sorted((dataset_path / class_name).iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                rows.append((str(file_path), class_name))

    if not rows:
        raise ValueError("No supported image files were found in the dataset.")

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    df["label_id"] = df["label"].map(class_to_index)

    return df, class_names, class_to_index


def split_dataset(df, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0; received {total:.4f}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label_id"],
        random_state=seed,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label_id"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def load_image(filepath, label, img_size):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    return image, label


def create_dataset(dataframe, img_size, batch_size, seed, training=False):
    paths = dataframe["filepath"].astype(str).values
    labels = dataframe["label_id"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda path, label: load_image(path, label, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        ds = ds.shuffle(
            buffer_size=max(len(dataframe), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augmentation_layers():
    """Training-only augmentation; validation and test images remain unchanged."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def build_model(model_name, img_size, num_classes, learning_rate, use_augmentation):
    tf.keras.backend.clear_session()
    input_shape = (img_size, img_size, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = inputs
    if use_augmentation:
        x = augmentation_layers()(x)

    if model_name == "MobileNetV3Small":
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV3Small(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif model_name == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif model_name == "EfficientNetB0":
        # EfficientNetB0 includes its input rescaling internally.
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif model_name == "DenseNet121":
        x = tf.keras.applications.densenet.preprocess_input(x)
        base_model = tf.keras.applications.DenseNet121(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif model_name == "ResNet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif model_name == "VGG16":
        x = tf.keras.applications.vgg16.preprocess_input(x)
        base_model = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history, model_name, model_dir):
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "epoch"
    history_df.to_csv(model_dir / f"{model_name}_training_history.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / f"{model_name}_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / f"{model_name}_loss.png", dpi=300)
    plt.close()


def train_model(
    model_name,
    train_ds,
    val_ds,
    model_dir,
    img_size,
    num_classes,
    learning_rate,
    epochs,
    patience,
    lr_patience,
    use_augmentation,
):
    print("\n" + "=" * 70)
    print(f"Training: {model_name} | Augmentation: {'Yes' if use_augmentation else 'No'}")
    print("=" * 70)

    model = build_model(
        model_name, img_size, num_classes, learning_rate, use_augmentation
    )
    best_model_path = model_dir / f"{model_name}_best.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history, best_model_path


def evaluate_model(model, model_name, test_ds, class_names, model_dir):
    y_true, y_pred = [], []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Accuracy        : {metrics['Accuracy']:.4f}")
    print(f"Macro Precision : {metrics['Precision']:.4f}")
    print(f"Macro Recall    : {metrics['Recall']:.4f}")
    print(f"Macro F1-score  : {metrics['F1_Score']:.4f}")
    print("\nClassification Report:\n")
    print(report_text)

    (model_dir / f"{model_name}_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        model_dir / f"{model_name}_classification_report.csv"
    )

    pd.DataFrame(
        {
            "true_label_id": y_true,
            "true_label": [class_names[i] for i in y_true],
            "predicted_label_id": y_pred,
            "predicted_label": [class_names[i] for i in y_pred],
        }
    ).to_csv(model_dir / f"{model_name}_test_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        model_dir / f"{model_name}_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    return metrics


def calculate_complexity(
    model_name, img_size, num_classes, learning_rate, use_augmentation
):
    model = build_model(
        model_name, img_size, num_classes, learning_rate, use_augmentation
    )
    total_params = model.count_params()
    trainable_params = int(
        sum(np.prod(v.shape) for v in model.trainable_weights)
    )
    result = {
        "Model": model_name,
        "Total Parameters": int(total_params),
        "Trainable Parameters": trainable_params,
        "Non-Trainable Parameters": int(total_params - trainable_params),
    }
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def save_experiment_configuration(args, result_path, class_names, class_to_index):
    config = vars(args).copy()
    config["dataset_path"] = str(Path(args.dataset_path).expanduser().resolve())
    config["result_path"] = str(result_path.resolve())
    config["augmentation"] = bool(args.augmentation)
    config["class_names"] = class_names
    config["class_to_index"] = class_to_index
    config["tensorflow_version"] = tf.__version__
    config["gpu_devices"] = [d.name for d in tf.config.list_physical_devices("GPU")]

    with open(result_path / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def main(args):
    set_reproducibility(args.seed)

    result_path = Path(args.result_path).expanduser()
    result_path.mkdir(parents=True, exist_ok=True)

    print("TensorFlow Version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices("GPU"))
    print("Dataset path:", args.dataset_path)
    print("Result path:", result_path)
    print("Augmentation:", "Yes" if args.augmentation else "No")

    df, class_names, class_to_index = read_dataset(args.dataset_path)
    num_classes = len(class_names)

    print("\nClasses:", class_names)
    print("Total images:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_dataset(
        df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    # Preserve the exact split used in the experiment.
    split_dir = result_path / "dataset_splits"
    split_dir.mkdir(exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(
        f"\nTraining Images: {len(train_df)} | "
        f"Validation Images: {len(val_df)} | Testing Images: {len(test_df)}"
    )

    train_ds = create_dataset(
        train_df, args.img_size, args.batch_size, args.seed, training=True
    )
    val_ds = create_dataset(
        val_df, args.img_size, args.batch_size, args.seed, training=False
    )
    test_ds = create_dataset(
        test_df, args.img_size, args.batch_size, args.seed, training=False
    )

    save_experiment_configuration(
        args, result_path, class_names, class_to_index
    )

    all_results = []
    complexity_results = []

    for model_name in args.models:
        tf.keras.backend.clear_session()
        gc.collect()

        model_dir = result_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        complexity_results.append(
            calculate_complexity(
                model_name,
                args.img_size,
                num_classes,
                args.learning_rate,
                args.augmentation,
            )
        )

        model, history, best_model_path = train_model(
            model_name=model_name,
            train_ds=train_ds,
            val_ds=val_ds,
            model_dir=model_dir,
            img_size=args.img_size,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.early_stopping_patience,
            lr_patience=args.lr_patience,
            use_augmentation=args.augmentation,
        )

        save_training_history(history, model_name, model_dir)
        result = evaluate_model(
            model, model_name, test_ds, class_names, model_dir
        )
        result["Augmentation"] = "Yes" if args.augmentation else "No"
        result["Best_Model_Path"] = str(best_model_path)
        all_results.append(result)

        del model, history
        tf.keras.backend.clear_session()
        gc.collect()

    results_df = pd.DataFrame(all_results).sort_values(
        by="F1_Score", ascending=False
    )
    results_df.to_csv(result_path / "Pretrained_Model_Comparison.csv", index=False)

    complexity_df = pd.DataFrame(complexity_results)
    complexity_df.to_csv(result_path / "Model_Complexity.csv", index=False)

    print("\nFinal Model Comparison:\n")
    print(results_df.to_string(index=False))
    print("\nModel Complexity:\n")
    print(complexity_df.to_string(index=False))
    print(f"\nAll results saved to: {result_path.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and compare pretrained CNNs for folder-based image classification."
    )
    parser.add_argument("--dataset-path", required=True, help="Dataset root directory.")
    parser.add_argument("--result-path", required=True, help="Directory for all outputs.")
    parser.add_argument(
        "--augmentation",
        type=parse_yes_no,
        default=False,
        help="Apply augmentation layers to training images: yes/no (default: no).",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="One or more models to train. Default: all supported models.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
