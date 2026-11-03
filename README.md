# SLM Domain Assistant: Fine-Tuned Llama 3.2 1B

A production-grade Small Language Model (SLM) backend built with **FastAPI**, **PyTorch**, **Transformers**, and **PEFT (QLoRA)** using **Llama-3.2-1B-Instruct**.

The application enables domain-specific natural language generation and summarization on resource-constrained hardware (8GB RAM, 4GB VRAM) via 4-bit `nf4` quantization and LoRA adapter fine-tuning.

---

## 🏗 Architecture & Key Features

- **Resource-Constrained Optimization**: 4-bit NormalFloat (`nf4`) quantization via `bitsandbytes`, maintaining a minimal VRAM footprint (~1GB peak usage).
- **QLoRA Fine-Tuning**: Efficient domain adaptation utilizing Hugging Face `peft` and `trl` (`SFTConfig`), saving lightweight adapter checkpoints under `experiments/final_adapter`.
- **Flexible REST API**: Multiple specialized FastAPI endpoints for inference, dynamic parameter control, summarization, and model diagnostics.
- **Automated Benchmarking**: Custom evaluation scripts (`evaluate.py`) tracking generation latency, peak GPU VRAM allocation, and response validation.
- **Interactive OpenAPI/Swagger UI**: Auto-generated interactive API documentation with dynamic parameter testing enabled.

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI running on Uvicorn |
| **Base Model** | `unsloth/Llama-3.2-1B-Instruct` |
| **Fine-Tuning & PEFT** | PEFT (QLoRA), `bitsandbytes` (4-bit `nf4`), `trl` |
| **Data & Validation** | PyTorch, Hugging Face `transformers`, Pydantic |
| **Evaluation & Benchmarks** | Custom PyTorch evaluation suite |

---

## 📁 Repository Structure

```text
slm-project/
│
├── configs/               # Hyperparameter and environment settings
│   └── model_config.yaml
│
├── data/                  # Dataset directory
│   ├── raw/               # Raw domain files
│   └── processed/         # Cleaned JSONL train/validation splits
│       └── train.jsonl
│
├── src/                   # Source application code
│   ├── model/             # Model loading, training, and inference logic
│   │   ├── train.py
│   │   └── inference.py
│   ├── evaluation/        # Benchmark and latency evaluation scripts
│   │   └── evaluate.py
│   ├── optimization/      # Quantization settings and VRAM management
│   │   └── quantization.py
│   └── api/               # FastAPI endpoints and Pydantic schemas
│       └── main.py
│
├── tests/                 # Unit and integration test scripts
│   └── test_api.py
│
├── experiments/           # Training output artifacts and PEFT adapters
│   └── final_adapter/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── reports/               # Evaluation results and benchmark summaries
│   └── benchmark_report.md
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (Minimum 4GB VRAM)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/slm-domain-assistant.git
   cd slm-domain-assistant
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Fine-Tuning & Evaluation Pipeline

To train the QLoRA adapter on your domain dataset:

```bash
python src/model/train.py
```

To run benchmarks on latency, memory allocation, and generation quality:

```bash
python src/evaluation/evaluate.py
```

---

## 🏃 Running the Application

Start the FastAPI deployment server:

```bash
uvicorn src.api.main:app --reload
```

The server will launch at `http://127.0.0.1:8000`.

---

## 🧪 Testing with Swagger UI

1. Open your browser and navigate to **`http://127.0.0.1:8000/docs`**.
2. **Model Info**: Call `GET /model-info` to verify adapter path and model loading status.
3. **Generate Response**: Execute `POST /generate` with a basic query:
   ```json
   {
     "prompt": "Where is the main library?",
     "max_tokens": 60
   }
   ```
4. **Advanced Generation**: Execute `POST /generate-advanced` to test custom temperature settings:
   ```json
   {
     "prompt": "List the library hours for the weekend.",
     "max_tokens": 80,
     "temperature": 0.3
   }
   抓```
5. **Summarization**: Call `POST /summarize-prompt` to compress long input text.
6. **Statistics**: Call `GET /model-stats` to view total inference counts and device status.

---

## 🔐 System Performance & Highlights

- **Low Footprint**: Bounded VRAM usage at ~1,004 MB (~1GB) during active inference.
- **Hardware Compatibility**: Designed for zero-out-of-memory execution on local consumer GPUs.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.