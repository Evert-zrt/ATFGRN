<div align="center">

# ATFGRN
**Revealing Hidden Regulatory Dependencies: Multi-Perspective Graph Learning for Single-Cell Gene Regulatory Network Inference**

[![Python](https://img.shields.io/badge/Python-3.10.16-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-orange.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6.1-green.svg?style=flat-square&logo=pyg&logoColor=white)](https://www.pyg.org/)

[Abstract](#abstract) • [Methodology](#methodology) • [Installation](#installation) • [Quick Start](#quick-start) • [Contact](#contact)

</div>

---

## 🧬 Abstract

Gene regulatory networks (GRNs) play a crucial role in revealing cellular state transitions, understanding regulatory mechanisms, and exploring the mechanisms of disease occurrence. With the development of single-cell sequencing technologies, accurately inferring GRNs from complex and high-dimensional single-cell transcriptomic data has become one of the core challenges in current research.

**ATFGRN** is a multi-perspective fusion deep learning model designed to address this challenge. It integrates diverse expression and structural information to improve the accuracy and robustness of GRN prediction. By constructing a **structure–expression–similarity tri-perspective framework**, ATFGRN fully integrates local topological structures, expression-contextual regulatory information, and the potential regulation between genes with similar expression patterns. Evaluations on single-cell transcriptomic datasets demonstrate that ATFGRN outperforms several mainstream methods in terms of AUROC and AUPRC metrics.

---

## 🧠 Methodology

ATFGRN proposes a **Structure–Expression–Similarity Tri-Perspective Framework** to extract features of candidate TF–gene pairs from three complementary perspectives.

<div align="center">
  <br>
  <img src="https://github.com/user-attachments/assets/4e61af41-48ed-4849-9836-6226f9155f6a" alt="ATFGRN Framework" width="95%">
  <br>
  <em>Figure 1: The overall architecture of the ATFGRN framework.</em>
  <br><br>
</div>

The framework consists of three distinct modules fused by a **Shared-Query Attention Mechanism**:

*   **Module 1: Local Topology-Aware Subgraph Encoding Module**
*   **Module 2: Attention-based Expression-Guided Module**
*   **Module 3: Similarity-Based Embedding Module**


---

## ⚙️ Installation

### System Requirements
*   **Python**: `3.10.16`
*   **PyTorch**: `2.5.0` (CUDA 11.8+ recommended)
*   **PyG (PyTorch Geometric)**: `2.6.1`

### Setup Environment

```bash
# 1. Create a virtual environment
conda create -n atfgrn python=3.10
conda activate atfgrn

# 2. Install PyTorch
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install Dependencies
pip install torch-geometric==2.6.1
pip install numpy==2.0.1 pandas==2.2.3 scikit-learn==1.6.1 scipy==1.15.2

```
---
## 🚀 Quick Start
### Data Preparation
```bash
.
├── data/
│   └── Specific/
│       └── hESC 500/
│           ├── Train_set.csv          # Training edges
│           ├── Validation_set.csv     # Validation edges
│           └── Test_set.csv           # Testing edges
│
├── Benchmark Dataset/
│   └── Specific Dataset/
│       └── hESC/
│           └── TFs+500/
│               └── BL--ExpressionData.csv   # Expression matrix
│
├── train.py
└── ...
```
### Training
To train the model on the hESC dataset with default settings, run:
```bash
python train.py --netType Specific --dataset hESC --num 500
```
### Key Arguments
You can configure the training process using the following arguments:
| Argument   | Default  | Type  | Description                                      |
|------------|----------|--------|--------------------------------------------------|
| `--dataset`| hESC     | str    | Which dataset to use (e.g., hESC, mDC).         |
| `--netType`| Specific | str    | Which network type to use.                     |
| `--num`    | 500      | str    | Scale of the dataset (e.g., 500, 1000).         |
| `--runs`   | 10       | int    | Number of independent runs.                     |
| `--epochs` | 401      | int    | Number of training epochs.                      |
| `--bs`     | 32       | int    | Batch size.                                     |
| `--lr`     | 0.001    | float  | Learning rate.                                  |


### 📧 Contact
For any questions or suggestions, please open an issue or leave a comment on this repository.
