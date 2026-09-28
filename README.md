# Task Management ML Service

<p align="center">
  <strong>AI-powered task classification service built with FastAPI and Transformer models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/PyTorch-ML%20Runtime-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Docker%20Hub-Image-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/AWS%20EC2-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS">
</p>

<p align="center">
  <strong>Created by Brighton Brown</strong>
</p>

---

# 📌 Overview

The **Task Management ML Service** is an independent machine learning microservice responsible for automatically classifying tasks created through the Task Management API.

The service receives a task title and description and predicts:

* 📂 Task category
* 🚨 Task priority
* 📊 Prediction confidence

The ML service is built with **FastAPI**, **Python**, **PyTorch**, and **Hugging Face Transformers**.

It runs independently from the main Spring Boot API and communicates with the API through HTTP/JSON over a private network in production.

---

# 🤖 What the ML Service Does

When a user creates a task, the Spring Boot API sends the task information to this service.

Example request:

```json
{
  "title": "Pay electricity bill",
  "description": "Pay the electricity bill today"
}
```

The ML service processes the text and returns:

```json
{
  "task": {
    "title": "Pay electricity bill",
    "description": "Pay the electricity bill today"
  },
  "prediction": {
    "category": "FINANCE",
    "priority": "URGENT"
  },
  "confidence": {
    "category": 0.9276,
    "priority": 0.9827
  },
  "status": "processed"
}
```

The Spring Boot API then stores the predicted category and priority with the task.

---

# 🏗️ System Architecture

The ML service is deployed separately from the main Task Management API.

```text
                         ┌─────────────────────┐
                         │   Flutter / Web     │
                         │    Mobile App       │
                         └──────────┬──────────┘
                                    │
                                    │ HTTP / JSON
                                    ▼
                         ┌─────────────────────┐
                         │   Spring Boot API   │
                         │                     │
                         │      AWS EC2        │
                         │       :8080         │
                         └──────────┬──────────┘
                                    │
                                    │ Private VPC Network
                                    │ HTTP / JSON
                                    ▼
                         ┌─────────────────────┐
                         │     ML Service      │
                         │      FastAPI        │
                         │                     │
                         │      AWS EC2        │
                         │   Host :8001        │
                         │ Container :8000     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                  Category Model       Priority Model
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                               Prediction
```

The ML service is intentionally separated from the Spring Boot API so that the machine learning workload can be:

* Deployed independently
* Updated independently
* Scaled independently
* Maintained separately from the main API

---

# 🔄 Prediction Flow

```text
Task Created
     │
     ▼
Spring Boot API
     │
     │ POST /predict
     ▼
FastAPI ML Service
     │
     ├───────────────┐
     │               │
     ▼               ▼
Category Model   Priority Model
     │               │
     ▼               ▼
Category         Priority
     │               │
     └───────┬───────┘
             ▼
        Confidence
             │
             ▼
       JSON Response
             │
             ▼
       Spring Boot API
             │
             ▼
         PostgreSQL
```

---

# 🧠 Machine Learning

The service uses Transformer-based text classification models.

The underlying Transformer architecture is based on:

```text
microsoft/MiniLM-L12-H384-uncased
```

Two independently trained classification models are used:

```text
Category Model
     │
     └── Predicts task category


Priority Model
     │
     └── Predicts task priority
```

Each model contains its own:

* Model configuration
* Transformer weights
* Tokenizer
* Label mappings
* Training configuration

---

# 📂 Task Categories

The category classifier supports:

```text
SHOPPING
HOUSEHOLD
WORK
STUDY
SOFTWARE
FINANCE
PERSONAL
ERRANDS
HEALTH_FITNESS
GENERAL
```

Examples:

```text
"Buy groceries"
        │
        ▼
    SHOPPING
```

```text
"Fix production server"
        │
        ▼
    SOFTWARE
```

```text
"Prepare machine learning assignment"
        │
        ▼
      STUDY
```

---

# 🚨 Task Priorities

The priority classifier predicts task urgency.

Supported priorities:

```text
LOW
MEDIUM
URGENT
```

Example:

```text
"Pay electricity bill today"
        │
        ▼
      URGENT
```

The priority model evaluates the task's text and determines the most likely priority.

---

# 📊 Confidence Scores

The service returns confidence values for both predictions.

Example:

```json
{
  "confidence": {
    "category": 0.9276,
    "priority": 0.9827
  }
}
```

These values represent the model's confidence in the corresponding classification.

---

# 🌐 API Endpoints

## Health Check

```text
GET /health
```

Used to verify that the ML service is running and both models are loaded.

### Local Example

```bash
curl http://localhost:8000/health
```

### Response

```json
{
  "status": "healthy",
  "category_model": "loaded",
  "priority_model": "loaded"
}
```

---

## 🔮 Prediction

```text
POST /predict
```

Accepts a task title and description.

### Request

```json
{
  "title": "Pay electricity bill",
  "description": "Pay the electricity bill today"
}
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Pay electricity bill","description":"Pay the electricity bill today"}'
```

### Response

```json
{
  "task": {
    "title": "Pay electricity bill",
    "description": "Pay the electricity bill today"
  },
  "prediction": {
    "category": "FINANCE",
    "priority": "URGENT"
  },
  "confidence": {
    "category": 0.9276,
    "priority": 0.9827
  },
  "status": "processed"
}
```

---

# 📁 Project Structure

```text
task-ml/
│
├── app/
│   └── main.py
│
├── data/
│   ├── tasks.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── models/
│   ├── category_labels.json
│   │
│   ├── category_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── training_args.bin
│   │
│   ├── priority_labels.json
│   │
│   └── priority_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── training/
│   ├── check_dataset.py
│   └── train.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── requirements-training.txt
```

---

# 🧩 Application Structure

## `app/main.py`

The FastAPI application provides:

```text
FastAPI
  │
  ├── /health
  │
  └── /predict
         │
         ├── Category Model
         │
         └── Priority Model
```

The models are loaded when the service starts.

---

# 🧠 Model Files

The trained model weights are stored using Git LFS.

```text
models/category_model/model.safetensors
models/priority_model/model.safetensors
```

Approximate model sizes:

```text
Category model  ≈ 133 MB
Priority model  ≈ 133 MB
```

Git LFS is used so that the large model binaries can be versioned separately from normal Git objects.

---

# 🏋️ Training

The repository also contains the training components.

Training data is stored under:

```text
data/
```

Check the dataset:

```bash
python training/check_dataset.py
```

Train the models:

```bash
python training/train.py
```

The training process produces the model files used by the FastAPI service.

---

# 🐍 Python Environment

The ML service uses **Python 3.12**.

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For training:

```bash
pip install -r requirements-training.txt
```

---

# ▶️ Running Locally

Start the FastAPI service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# 📖 FastAPI Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://localhost:8000/docs
```

The documentation allows developers to:

* View available endpoints
* Inspect request schemas
* Send prediction requests
* Test the health endpoint
* Inspect API responses

---

# 🐳 Docker

The ML service is containerized using Docker.

The Docker image contains:

```text
Python runtime
     │
     ├── FastAPI
     ├── PyTorch
     ├── Transformers
     ├── Tokenizers
     │
     └── Trained ML models
```

## Build

```bash
docker build -t brizzieh/task-ml:latest .
```

## Run Locally

The FastAPI container listens on port `8000`.

```bash
docker run -d \
  --name task-ml \
  -p 8000:8000 \
  brizzieh/task-ml:latest
```

Check the container:

```bash
docker ps
```

View logs:

```bash
docker logs task-ml
```

Test health:

```bash
curl http://localhost:8000/health
```

---

# 🐳 Docker Hub

The production Docker image is published as:

```text
brizzieh/task-ml:latest
```

Pull the image:

```bash
docker pull brizzieh/task-ml:latest
```

Run locally:

```bash
docker run -d \
  --name task-ml \
  -p 8000:8000 \
  brizzieh/task-ml:latest
```

---

# ☁️ AWS EC2 Deployment

The ML service runs on a dedicated AWS EC2 instance.

Production architecture:

```text
┌─────────────────────────────────────────┐
│                 AWS VPC                 │
│                                         │
│  ┌───────────────────┐                  │
│  │ API EC2           │                  │
│  │                   │                  │
│  │ Spring Boot       │                  │
│  │ Host :8080        │                  │
│  └─────────┬─────────┘                  │
│            │                             │
│            │ Private VPC Network         │
│            │                             │
│            ▼                             │
│  ┌───────────────────┐                  │
│  │ ML EC2            │                  │
│  │                   │                  │
│  │ FastAPI           │                  │
│  │ Docker            │                  │
│  │ Host :8001        │                  │
│  │ Container :8000   │                  │
│  └───────────────────┘                  │
│                                         │
└─────────────────────────────────────────┘
```

The ML service is separated onto its own EC2 instance so that its machine-learning workload does not run inside the API server.

---

# 🔒 Network Security

The ML service is designed to communicate with the Spring Boot API through the AWS private network.

```text
Internet
   │
   ▼
API EC2 :8080
   │
   │ Private VPC Network
   ▼
ML EC2 :8001
   │
   ▼
FastAPI :8000
```

The ML service is **not intended to be directly accessible from the public internet**.

The AWS Security Group should restrict inbound access to port `8001` so that only the API EC2 instance or the appropriate private network resources can communicate with the ML service.

The Spring Boot API uses a private ML URL similar to:

```yaml
ml:
  service:
    url: http://ML_PRIVATE_IP:8001
```

---

# 🔗 Spring Boot Integration

The Spring Boot API communicates with the ML service using HTTP.

The API sends:

```text
POST /predict
```

to the private ML service.

Example configuration:

```yaml
ml:
  service:
    url: http://ML_PRIVATE_IP:8001
```

The integration flow is:

```text
Spring Boot API
       │
       │ POST /predict
       ▼
FastAPI ML Service
       │
       ▼
Prediction
       │
       ▼
Spring Boot API
       │
       ▼
PostgreSQL
```

The predicted values are stored with the task:

```text
Task
├── title
├── description
├── completed
├── category
├── priority
├── createdAt
└── updatedAt
```

The client application receives the resulting task through the main API and does not need to communicate directly with the ML service.

---

# 🚀 CI/CD Pipeline

The ML service has its own GitHub Actions deployment pipeline.

Every push to the `main` branch triggers the deployment workflow.

```text
Developer
    │
    │ git push origin main
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ├── Checkout repository
    │
    ├── Checkout Git LFS models
    │
    ├── Verify model files
    │
    ├── Login to Docker Hub
    │
    ├── Build ML Docker image
    │
    ├── Push ML image
    │
    └── SSH to ML EC2
             │
             ▼
       Pull latest image
             │
             ▼
       Remove old container
             │
             ▼
       Start new container
             │
             ▼
       Verify deployment
```

---

# 🔄 GitHub Actions Deployment

The deployment workflow performs:

1. Checkout source code.
2. Download Git LFS model files.
3. Verify model binaries.
4. Authenticate with Docker Hub.
5. Build the Docker image.
6. Push the image to Docker Hub.
7. Connect to the ML EC2 instance using SSH.
8. Pull the latest Docker image.
9. Remove the previous ML container.
10. Start the new ML container.
11. Verify the running container.

The production image is:

```text
brizzieh/task-ml:latest
```

The production container maps:

```text
EC2 host :8001
        │
        ▼
Docker container :8000
```

---

# 🗂️ Git LFS

Large model files are tracked using Git LFS.

Check tracked model files:

```bash
git lfs ls-files
```

The repository tracks:

```text
models/category_model/model.safetensors
models/priority_model/model.safetensors
```

When GitHub Actions checks out the repository, Git LFS must be enabled so that the Docker image receives the actual model binaries rather than Git LFS pointer files.

---

# 🧪 Production Verification

After deployment, verify that the container is running:

```bash
sudo docker ps
```

Expected:

```text
task-ml
```

Check the model sizes:

```bash
sudo docker exec task-ml \
python -c "import os; print(os.path.getsize('/app/models/category_model/model.safetensors')); print(os.path.getsize('/app/models/priority_model/model.safetensors))"
```

Expected approximately:

```text
133478696
133469456
```

---

# ❤️ Production Health Check

On the ML EC2 instance:

```bash
curl http://localhost:8001/health
```

Expected:

```json
{
  "status": "healthy",
  "category_model": "loaded",
  "priority_model": "loaded"
}
```

The ML service is private-only, so the production endpoint should be tested from the ML EC2 instance or from another authorized private resource rather than from the public internet.

---

# 🧪 Production Prediction Test

On the ML EC2 instance:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Pay electricity bill","description":"Pay the electricity bill today"}'
```

Expected prediction structure:

```json
{
  "task": {
    "title": "Pay electricity bill",
    "description": "Pay the electricity bill today"
  },
  "prediction": {
    "category": "FINANCE",
    "priority": "URGENT"
  },
  "confidence": {
    "category": 0.9276,
    "priority": 0.9827
  },
  "status": "processed"
}
```

---

# 🔌 API Integration Example

The Spring Boot API sends:

```json
{
  "title": "Complete machine learning assignment",
  "description": "Finish the ML assignment before tomorrow"
}
```

The ML service returns:

```json
{
  "prediction": {
    "category": "STUDY",
    "priority": "URGENT"
  }
}
```

The API then stores:

```text
category = STUDY
priority = URGENT
```

This allows the client application to receive AI-generated task metadata without directly communicating with the ML service.

---

# 🛠️ Useful Commands

## Git

Check status:

```bash
git status
```

Pull latest changes:

```bash
git pull origin main
```

Push changes:

```bash
git add .
git commit -m "Update ML service"
git push origin main
```

## Git LFS

Check installation:

```bash
git lfs version
```

List tracked files:

```bash
git lfs ls-files
```

Pull LFS files:

```bash
git lfs pull
```

---

# 🐳 Useful Docker Commands

Build:

```bash
docker build -t brizzieh/task-ml:latest .
```

Run locally:

```bash
docker run -d \
  --name task-ml \
  -p 8000:8000 \
  brizzieh/task-ml:latest
```

Stop:

```bash
docker stop task-ml
```

Remove:

```bash
docker rm task-ml
```

View logs:

```bash
docker logs -f task-ml
```

View running containers:

```bash
docker ps
```

Remove unused images:

```bash
docker image prune -f
```

---

# 📈 Scalability

Separating the ML service from the Spring Boot API makes it possible to scale the machine-learning workload independently.

A future architecture could look like:

```text
                         Load Balancer
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Spring API     Spring API     Spring API
               │              │              │
               └──────────────┼──────────────┘
                              │
                         ML Requests
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
            ML EC2         ML EC2         ML EC2
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       Transformer Models
```

This architecture allows the API and ML workloads to be scaled independently.

---

# 🔮 Future Improvements

Possible future improvements include:

* Model versioning
* Automated model retraining
* Larger training datasets
* Model performance evaluation
* Precision / recall reporting
* Confusion matrices
* Automated ML tests
* Model monitoring
* Prediction logging
* Model confidence thresholds
* Batch prediction
* GPU-enabled inference
* Model optimization
* ONNX Runtime
* Quantized models
* Automatic model deployment
* ML-specific monitoring
* Horizontal ML service scaling
* HTTPS through a private service gateway

---

# 🧱 Technology Stack

| Technology                | Purpose                                           |
| ------------------------- | ------------------------------------------------- |
| Python 3.12               | Programming language                              |
| FastAPI                   | ML REST API                                       |
| Uvicorn                   | ASGI server                                       |
| PyTorch                   | Machine learning runtime                          |
| Hugging Face Transformers | Transformer models                                |
| MiniLM                    | Text representation / classification architecture |
| Git LFS                   | Large model file versioning                       |
| Docker                    | Containerization                                  |
| Docker Hub                | Container image registry                          |
| GitHub Actions            | CI/CD automation                                  |
| AWS EC2                   | Production deployment                             |

---

# 🔗 Relationship With Task Management API

This repository is the machine-learning component of the larger Task Management system.

```text
┌──────────────────────────────────────┐
│          Task Management API         │
│                                      │
│       Spring Boot / Java 21          │
│                                      │
│          brizzieh/task-api           │
└──────────────────┬───────────────────┘
                   │
                   │ Private HTTP
                   ▼
┌──────────────────────────────────────┐
│          Task ML Service             │
│                                      │
│       FastAPI / Python 3.12          │
│                                      │
│          brizzieh/task-ml            │
└──────────────────────────────────────┘
```

The Spring Boot API remains responsible for:

* Authentication
* Authorization
* Users
* Tasks
* PostgreSQL
* Business logic

The ML service is responsible for:

* Text processing
* Category prediction
* Priority prediction
* Confidence scores
* Machine-learning inference

---

# 👨‍💻 Author

**Brighton Brown**

Software Developer / Backend Engineer

Built with:

* Python
* FastAPI
* PyTorch
* Hugging Face Transformers
* MiniLM
* Docker
* Docker Hub
* GitHub Actions
* AWS EC2

---

# ⭐ Project Philosophy

> Separate responsibilities. Train intelligently. Deploy independently. Automate everything.

The Task ML Service is designed as an independent machine-learning microservice that can integrate with the Task Management API or any other application capable of sending HTTP/JSON requests.

---

<p align="center">
  <strong>© 2026 Brighton Brown</strong>
</p>

<p align="center">
  Built with ❤️ using FastAPI and Transformers
</p>
