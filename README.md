# ECL-Net
This repository contains the official PyTorch implementation of **ECL-Net**, a deep learning framework designed to predict protein thermal stability.

## ⚙️ Installation

### 1. Environment Setup
Requires **Python 3.11** and **PyTorch 2.7.1 (CUDA 11.8)**. We recommend using Conda to set up your environment and install dependencies via `requirements.txt`.

### 2. External Dependencies
* **Rosetta**: Apply for a license and download Rosetta 3.12 from the [Rosetta Commons](https://www.rosettacommons.org/software/license-and-download).
* **ESM Weights**: Download the pretrained `ESM2_t36_3B_UR50D` model weights from the [Official ESM GitHub](https://github.com/facebookresearch/esm) and place them in your model cache directory.

---

## 🧬 Data Processing

### 1. Protein Relaxation and Mutation
Use Rosetta scripts to perform energy optimization and construct mutants. Related scripts are located in `Data_processing/`:
* **Wild-type Relaxation**: Use `Rosetta_relax_wt.py`.
* **Mutant Generation**: Use `Rosetta_relax_mut.py`.

### 2. Feature Extraction Pipeline
Execute the following scripts in sequence to build the graph datasets:

**Step 1: Sequence Extraction**  
Extract FASTA sequences from PDB files.
```bash
python Fasta.py
```

**Step 2: ESM Feature Extraction**  
Generate feature embeddings and calculate self-attention scores.
```bash
python ESM2.py
```

**Step 3: Graph Construction**  
Integrate spatial structures with ESM-2 features into graph data formats.
```bash
python Graph.py
```

---

## 🚀 Train Model

The training pipeline consists of two stages: **Pre-training** and **Fine-tuning**.

### 1. Data Preparation
Ensure the graph data serialization is complete and stored in `data/graphs/`.

### 2. Start Training
```bash
python Train.py
```

### 3. Custom Training
Hyperparameters and experiment settings are managed in the `CONFIG` dictionary at the top of `Train.py`. Modify this dictionary to adapt to your own hardware or datasets.

---

## 🔮 Mutation Effect Prediction

To predict the thermal stability of new mutations, we provide a robust ensemble prediction script.

### 1. Configure Models
Open `Predict.py` and modify `CONFIG['model_paths']` to include your best-trained checkpoint paths (`.pth` files).

### 2. Run Inference
```bash
python Predict.py
```
**Expected Output:**
The script generates `.csv`. It ranks mutations by average predicted probability (`Avg_Prob`) and provides standard deviation (`Std_Dev`) and consensus vote counts (`Agree_Count`) for robust candidate prioritization.
