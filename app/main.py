from pathlib import Path
import json

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ============================================================
# FastAPI Configuration
# ============================================================

app = FastAPI(
    title="Task Management ML Service",
    version="1.0.0"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

CATEGORY_MODEL_DIR = MODELS_DIR / "category_model"
PRIORITY_MODEL_DIR = MODELS_DIR / "priority_model"

CATEGORY_LABELS_FILE = MODELS_DIR / "category_labels.json"
PRIORITY_LABELS_FILE = MODELS_DIR / "priority_labels.json"


# ============================================================
# Load Label Mappings
# ============================================================

def load_labels(file_path: Path):
    """
    Load labels from the JSON files generated during training.

    Supports the structure:

    {
        "label_to_id": {
            "ERRANDS": 0,
            "FINANCE": 1
        },
        "id_to_label": {
            "0": "ERRANDS",
            "1": "FINANCE"
        }
    }
    """

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Our training script stores id_to_label
    if "id_to_label" in data:
        return {
            int(key): value
            for key, value in data["id_to_label"].items()
        }

    # Fallback if labels are stored directly as:
    # {"0": "ERRANDS", "1": "FINANCE"}
    return {
        int(key): value
        for key, value in data.items()
    }


category_labels = load_labels(
    CATEGORY_LABELS_FILE
)

priority_labels = load_labels(
    PRIORITY_LABELS_FILE
)


# ============================================================
# Load Trained Category Model
# ============================================================

category_tokenizer = AutoTokenizer.from_pretrained(
    CATEGORY_MODEL_DIR
)

category_model = AutoModelForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_DIR
)

category_model.eval()


# ============================================================
# Load Trained Priority Model
# ============================================================

priority_tokenizer = AutoTokenizer.from_pretrained(
    PRIORITY_MODEL_DIR
)

priority_model = AutoModelForSequenceClassification.from_pretrained(
    PRIORITY_MODEL_DIR
)

priority_model.eval()


# ============================================================
# Request Model
# ============================================================

class TaskRequest(BaseModel):
    title: str
    description: str | None = None


# ============================================================
# Prepare Task Text
# ============================================================

def prepare_text(task: TaskRequest) -> str:
    """
    Combine the task title and description into one input.
    """

    title = task.title.strip()

    if task.description:
        description = task.description.strip()

        if description:
            return f"{title} {description}"

    return title


# ============================================================
# Prediction Function
# ============================================================

def predict_with_model(
    text: str,
    tokenizer,
    model,
    labels
):
    """
    Run a classification model and return:

    - predicted label
    - confidence
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=-1
    )

    class_id = predicted_class.item()

    confidence_value = confidence.item()

    label = labels.get(
        class_id,
        f"UNKNOWN_{class_id}"
    )

    return label, confidence_value


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Task Management ML Service",
        "model": "microsoft/MiniLM-L12-H384-uncased",
        "tasks": [
            "category classification",
            "priority classification"
        ],
        "status": "running"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "category_model": "loaded",
        "priority_model": "loaded"
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(task: TaskRequest):

    text = prepare_text(task)

    # -----------------------------
    # Category prediction
    # -----------------------------

    category, category_confidence = predict_with_model(
        text=text,
        tokenizer=category_tokenizer,
        model=category_model,
        labels=category_labels
    )

    # -----------------------------
    # Priority prediction
    # -----------------------------

    priority, priority_confidence = predict_with_model(
        text=text,
        tokenizer=priority_tokenizer,
        model=priority_model,
        labels=priority_labels
    )

    # -----------------------------
    # Response
    # -----------------------------

    return {
        "task": {
            "title": task.title,
            "description": task.description
        },
        "prediction": {
            "category": category,
            "priority": priority
        },
        "confidence": {
            "category": round(category_confidence, 4),
            "priority": round(priority_confidence, 4)
        },
        "status": "processed"
    }