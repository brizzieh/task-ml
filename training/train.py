from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "tasks.csv"
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"

OUTPUT_DIR = BASE_DIR / "models"

CATEGORY_MODEL_DIR = OUTPUT_DIR / "category_model"
PRIORITY_MODEL_DIR = OUTPUT_DIR / "priority_model"

CATEGORY_LABELS_FILE = OUTPUT_DIR / "category_labels.json"
PRIORITY_LABELS_FILE = OUTPUT_DIR / "priority_labels.json"


# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

MAX_LENGTH = 128
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("MINILM TASK CLASSIFIER TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_PATH}")


# ============================================================
# LOAD DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

required_columns = {
    "title",
    "description",
    "category",
    "priority",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


print(f"\nTotal examples: {len(df)}")


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

for column in ["title", "description", "category", "priority"]:
    df[column] = df[column].astype(str).str.strip()

df = df.dropna(
    subset=["title", "description", "category", "priority"]
)

df = df.drop_duplicates()

# Combine title and description.
#
# Example:
#
# Fix authentication bug.
# Users receive a 403 error when trying to log in.
#
df["text"] = (
    df["title"]
    + ". "
    + df["description"]
)


print(f"Examples after cleaning: {len(df)}")


# ============================================================
# LABEL ENCODING
# ============================================================

category_labels = sorted(df["category"].unique())
priority_labels = sorted(df["priority"].unique())

category_to_id = {
    label: index
    for index, label in enumerate(category_labels)
}

priority_to_id = {
    label: index
    for index, label in enumerate(priority_labels)
}

df["category_id"] = df["category"].map(category_to_id)
df["priority_id"] = df["priority"].map(priority_to_id)


print("\nCategory labels:")

for index, label in enumerate(category_labels):
    print(f"  {index}: {label}")


print("\nPriority labels:")

for index, label in enumerate(priority_labels):
    print(f"  {index}: {label}")


# ============================================================
# CREATE STRATIFICATION LABEL
# ============================================================
#
# We want both category AND priority represented across
# train/validation/test as evenly as possible.
#
# Example:
#
# SOFTWARE_HIGH
# STUDY_LOW
# FINANCE_MEDIUM
#
# This is especially useful because the dataset is small.
#

df["stratify_label"] = (
    df["category"]
    + "_"
    + df["priority"]
)

# Some category/priority combinations can be small.
# If every combination has enough examples, use it.
# Otherwise fall back to category stratification.

stratify_counts = df["stratify_label"].value_counts()

if stratify_counts.min() >= 2:
    stratify_column = df["stratify_label"]
    print("\nUsing category + priority stratification.")
else:
    stratify_column = df["category"]
    print("\nUsing category stratification.")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First:
# 90% temporary training data
# 10% test data

train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_column,
)


# For the remaining 90%, take approximately 11.11%
# to create a final 10% validation set.
#
# Result:
#
# 80% train
# 10% validation
# 10% test

train_val_stratify = train_val_df["category"]

validation_relative_size = (
    VALIDATION_SIZE / (1.0 - TEST_SIZE)
)

train_df, validation_df = train_test_split(
    train_val_df,
    test_size=validation_relative_size,
    random_state=RANDOM_STATE,
    stratify=train_val_stratify,
)


print("\nDataset split:")
print(f"  Training:   {len(train_df)}")
print(f"  Validation: {len(validation_df)}")
print(f"  Test:       {len(test_df)}")


# ============================================================
# SAVE SPLITS
# ============================================================

splits_dir = BASE_DIR / "data" / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    splits_dir / "train.csv",
    index=False,
)

validation_df.to_csv(
    splits_dir / "validation.csv",
    index=False,
)

test_df.to_csv(
    splits_dir / "test.csv",
    index=False,
)

print(f"\nDataset splits saved to: {splits_dir}")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET PREPARATION
# ============================================================

def create_huggingface_dataset(dataframe, label_column):
    """
    Convert a pandas DataFrame into a Hugging Face Dataset.

    label_column:
        category_id OR priority_id
    """

    dataset_df = dataframe[
        ["text", label_column]
    ].copy()

    dataset_df = dataset_df.rename(
        columns={
            label_column: "labels"
        }
    )

    dataset = Dataset.from_pandas(
        dataset_df,
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    dataset = dataset.remove_columns(
        ["text"]
    )

    return dataset


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):
    """
    Calculate accuracy and weighted F1 score.
    """

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_classifier(
    label_column,
    labels,
    output_dir,
    classifier_name,
):
    print("\n")
    print("=" * 70)
    print(f"TRAINING {classifier_name.upper()} CLASSIFIER")
    print("=" * 70)

    train_dataset = create_huggingface_dataset(
        train_df,
        label_column,
    )

    validation_dataset = create_huggingface_dataset(
        validation_df,
        label_column,
    )

    test_dataset = create_huggingface_dataset(
        test_df,
        label_column,
    )

    print("\nLoading MiniLM model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={
            index: label
            for index, label in enumerate(labels)
        },
        label2id={
            label: index
            for index, label in enumerate(labels)
        },
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",
        greater_is_better=True,

        logging_strategy="epoch",

        report_to="none",

        fp16=torch.cuda.is_available(),

        save_total_limit=2,

        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")

    trainer.train()

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print("\nValidation results:")

    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )

    print(
        f"  Accuracy: "
        f"{validation_results.get('eval_accuracy', 0):.4f}"
    )

    print(
        f"  F1: "
        f"{validation_results.get('eval_f1', 0):.4f}"
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\nTest results:")

    test_results = trainer.evaluate(
        eval_dataset=test_dataset
    )

    print(
        f"  Accuracy: "
        f"{test_results.get('eval_accuracy', 0):.4f}"
    )

    print(
        f"  F1: "
        f"{test_results.get('eval_f1', 0):.4f}"
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print(
        f"\nModel saved to:\n{output_dir}"
    )

    return {
        "validation": validation_results,
        "test": test_results,
    }


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE LABEL MAPS
# ============================================================

with open(
    CATEGORY_LABELS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "label_to_id": category_to_id,
            "id_to_label": {
                str(index): label
                for index, label in enumerate(category_labels)
            },
        },
        file,
        indent=2,
    )


with open(
    PRIORITY_LABELS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "label_to_id": priority_to_id,
            "id_to_label": {
                str(index): label
                for index, label in enumerate(priority_labels)
            },
        },
        file,
        indent=2,
    )


# ============================================================
# TRAIN CATEGORY MODEL
# ============================================================

category_results = train_classifier(
    label_column="category_id",
    labels=category_labels,
    output_dir=CATEGORY_MODEL_DIR,
    classifier_name="category",
)


# ============================================================
# TRAIN PRIORITY MODEL
# ============================================================

priority_results = train_classifier(
    label_column="priority_id",
    labels=priority_labels,
    output_dir=PRIORITY_MODEL_DIR,
    classifier_name="priority",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nModels:")

print(f"  Category:")
print(f"    {CATEGORY_MODEL_DIR}")

print(f"\n  Priority:")
print(f"    {PRIORITY_MODEL_DIR}")

print("\nLabel files:")

print(f"  {CATEGORY_LABELS_FILE}")
print(f"  {PRIORITY_LABELS_FILE}")

print("\nDone.")