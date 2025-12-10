# Graph-Enhanced Knowledge Tracing

A deep learning system for predicting student performance using knowledge tracing.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python local_assets/download_data.py
```

3. Build the knowledge graph:
```bash
python local_assets/build_graph.py
```

## Training

Train the models using the provided scripts:

```bash
# Deep Knowledge Tracing (DKT)
python scripts/train.py --model dkt --config configs/dkt_config.yaml

# Graph-Enhanced DKT
python scripts/train.py --model graph_dkt --config configs/graph_config.yaml

# Logistic Regression Baseline
python scripts/train_logistic_regression.py --config configs/baseline_config.yaml
```

## Evaluation

Evaluate model performance:
```bash
python scripts/evaluate.py --model <model_name>
```

## Pre-trained Models

This repository includes pre-trained models in the `models/` directory that you can use directly for evaluation without training from scratch.

## Project Structure

```

├── configs/                    # Model configuration files
│   ├── baseline_config.yaml   # Logistic regression settings
│   ├── dkt_config.yaml        # DKT model hyperparameters
│   └── graph_config.yaml      # Graph-Enhanced DKT settings
├── models/                     # Pre-trained model weights
│   ├── baseline/
│   │   └── logistic_regression.joblib
│   ├── dkt/
│   │   └── best_model.pth
│   └── graph_enhanced/
│       └── best_model.pth
├── scripts/                    # Training and evaluation scripts
│   ├── train.py              # Main training script
│   ├── train_logistic_regression.py
│   └── evaluate.py           # Model evaluation
├── src/                        # Core source code
│   ├── data/
│   │   ├── assistments_loader.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── graph/
│   │   ├── gnn_layers.py
│   │   └── knowledge_graph.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── dkt.py
│   │   └── graph_dkt.py
│   └── utils/
│       ├── metrics.py
│       ├── trainer.py
│       └── build_graph.py
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore                 # Git ignore rules
```


## Configuration

See `configs/` for model hyperparameters:
- `baseline_config.yaml` — Data paths and splits
- `dkt_config.yaml` — LSTM architecture settings
- `graph_config.yaml` — GCN layers, edge threshold, fusion method

## Dataset

**ASSISTments 2015:** 104,804 interactions, 100 skills, 70/15/15 split

