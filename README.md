# 🧠 CIM-GNN: Causality-Informed Multimodal Graph Neural Network

> An interpretable graph neural network framework for brain network analysis and disease diagnosis with causal representation learning.

---

## 📌 Overview

**CIM-GNN** is a causality-aware graph neural network designed for brain network analysis. Unlike traditional GNN models that rely on correlation, CIM-GNN explicitly models **causal relationships** within graph-structured data, improving both **prediction performance** and **interpretability**.

---

## 🚀 Key Features

- 🔍 Built-in Interpretability  
- 🧩 Causal Subgraph Discovery  
- 🧠 Multimodal Brain Data Support  
- ⚖️ Disentangled Representation Learning  
- 📊 Medical AI Ready  

---

## 📂 Project Structure

```
CIM-GNN/
├── data/
├── models/
├── utils/
├── train.py
├── test.py
├── config.py
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

```bash
git clone https://github.com/jinglinyuan069-tech/CIM-GNN.git
cd CIM-GNN

conda create -n cim-gnn python=3.8
conda activate cim-gnn

pip install -r requirements.txt
````

---

## 🧪 Training

```bash
python train.py --dataset your_dataset --epochs 100 --lr 0.001
```

---

## 📈 Evaluation

```bash
python test.py --model_path checkpoints/model.pth
```


