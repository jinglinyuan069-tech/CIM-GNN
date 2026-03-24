# 🧠 CIM-GNN: Causality-Informed Multimodal Graph Neural Network

> An interpretable graph neural network framework for brain network analysis and disease diagnosis with causal representation learning.

---

## 📌 Overview

**CIM-GNN** is a causality-aware graph neural network designed for brain network analysis. Unlike traditional GNN models that rely on correlation, CIM-GNN explicitly models **causal relationships** within graph-structured data, improving both **prediction performance** and **interpretability**.

This framework is particularly suitable for:

- Brain connectivity analysis (fMRI / DTI / sMRI)
- Disease diagnosis (e.g., MDD, AD)
- Interpretable AI in medical applications

---

## 🚀 Key Features

- 🔍 **Built-in Interpretability**  
  Learns explanations during training (no post-hoc explainer needed)

- 🧩 **Causal Subgraph Discovery**  
  Identifies task-relevant brain regions automatically

- 🧠 **Multimodal Brain Data Support**  
  Supports functional and structural connectivity

- ⚖️ **Disentangled Representation Learning**  
  Separates causal and non-causal information

- 📊 **Medical AI Ready**  
  Supports classification and regression tasks

---

## 🏗️ Method Overview

CIM-GNN consists of three core modules:

### 1. Graph Encoder
Encodes brain connectivity graphs into latent representations.

### 2. Causal Disentanglement Module
Learns:
- **Causal representation (α)**
- **Non-causal representation (β)**

### 3. Subgraph Selection Module
Identifies important subgraphs responsible for prediction.

---

## 📐 Pipeline

Input Brain Network (Graph)
↓
Graph Encoder (GNN)
↓
Disentangled Representations
├── Causal Component
└── Non-Causal Component
↓
Subgraph Selection
↓
Prediction Head
↓
Interpretation (Important Brain Regions)

---

## 📂 Project Structure

CIM-GNN/
│
├── data/ # Dataset files
├── models/ # Model definitions
├── utils/ # Utility functions
├── train.py # Training script
├── test.py # Evaluation script
├── config.py # Hyperparameters
├── requirements.txt # Dependencies
└── README.md

---

## ⚙️ Installation

```bash
git clone https://github.com/jinglinyuan069-tech/CIM-GNN.git
cd CIM-GNN

conda create -n cim-gnn python=3.8
conda activate cim-gnn

pip install -r requirements.txt

📊 Dataset

The model expects:

Graph adjacency matrix (connectivity)
Node features (brain region signals)
Labels (diagnosis or scores)

Supported data types:

Functional Connectivity (FC)
Structural Connectivity (SC)
Multimodal brain data


🧪 Training
python train.py \
    --dataset your_dataset \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 64
📈 Evaluation
python test.py \
    --model_path checkpoints/model.pth

Evaluation metrics include:

Accuracy
AUC
F1-score
Interpretability metrics
🔬 Outputs
Prediction results (classification/regression)
Important brain subgraphs
Causal contribution scores






















