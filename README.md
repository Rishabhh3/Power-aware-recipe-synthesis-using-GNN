# Power-Aware Logic Synthesis Optimization using GNNs

A machine learning-driven Electronic Design Automation (EDA) framework that replaces slow physical VLSI simulation with a multi-objective Graph Neural Network (GNN). The model guides a Simulated Annealing search engine to navigate logic synthesis transformations, discovering sequences that reduce circuit power under strict timing constraints.

---

##  Overview

Logic synthesis transforms high-level digital designs into optimized gate-level representations. Finding the optimal sequence of transformations is challenging:
- Selecting a **20-step recipe** from **7 core logic operations** produces an optimization space of $7^{20} \approx 7.98 \times 10^{16}$ possible sequences.
- Traditional industry heuristics (such as Berkeley ABC's `resyn2`) rely on greedy rules and frequently get trapped in suboptimal local minima.
- Physical simulation through standard cell libraries is computationally expensive, making exhaustive exploration impractical.

This pipeline utilizes a 3-lobe GNN surrogate oracle to predict Power, Performance, and Area (PPA) in milliseconds, allowing Simulated Annealing to evaluate thousands of candidate sequences in seconds.

---

##  Architecture & Pipeline

### 1. Data Generation & Ingestion
- **C++ Multi-threaded Engine (`dataset_generator.cpp`):** Spawns worker threads across CPU cores using a task queue (`std::mutex`, `std::condition_variable`) to execute Berkeley ABC system calls concurrently. Generated 200,000 modified graphs across 10 EPFL arithmetic benchmarks.
- **Log Parsing (`data_labels.py`):** Uses regex to parse execution outputs and map transformation sequences to verified physical metrics (`Power`, `Area`, `Delay`) into `labels.csv`.
- **DAG Tensor Conversion (`bench_to_tensor.py`):** Translates `.bench` netlists into PyTorch Geometric (PyG) Directed Acyclic Graphs:
  - **Node Features ($X$):** One-hot encodings for Primary Inputs (PI), Primary Outputs (PO), and internal logic gates.
  - **Edge Indices (`edge_index`):** Directional connectivity matrices ($2 \times E$).
  - **Edge Attributes (`edge_attr`):** Binary weights distinguishing standard wires (`0.0`) from logic inversions (`1.0`).

### 2. Multi-Objective 3-Lobe GNN (`gnn_model.py`)
- **Lobe 1 (Circuit Topology):** 2 Graph Convolutional layers (`GCNConv`) with Batch Normalization and ReLU, followed by a combined pooling layer (Global Mean Pooling + Global Max Pooling) to produce a 128D context vector.
- **Lobe 2 (Recipe Sequence):** An `nn.Embedding` layer and 1D Convolution (`nn.Conv1d`) to capture sequential dependencies across the 20-step transformation string, outputting a 64D vector.
- **Lobe 3 (Multi-Objective Regression):** Concatenates the 128D and 64D representations into a 192D feature vector, fed into a Multi-Layer Perceptron (MLP) outputting simultaneous predictions for `[Power, Area, Delay]`.

### 3. Normalization & Active Few-Shot Fine-Tuning
- **Per-Node Target Normalization:** Mitigates scale collapse across circuits of differing sizes by training on Power-per-node and Area-per-node, while predicting Delay globally at the graph level.
- **K-Means Smart Anchors (`step1_anchors.py`):** Partitions 1,500 random recipes into 5 clusters based on command frequency and extracts the centroid recipes for physical ground-truth simulation.
- **Graph Shielding & Gradient Balancing (`step2_micro_tune.py`):** Freezes Lobe 1 (`requires_grad = False`) to prevent catastrophic forgetting while fine-tuning Lobes 2 & 3. Balances loss gradients dynamically ($\text{target\_delay} / \text{target\_power}$) so delay metrics do not dominate microwatt power values.

### 4. Heuristic Search & Physical Verification
- **Model Validation (`step3_evaluate.py`):** Requires $\ge 85\%$ accuracy on unseen test sequences before permitting deployment.
- **Simulated Annealing Engine (`step4_search.py`):** Explores 5,000 candidate sequences in seconds using surrogate inference.
- **Timing Shield:** Discards any recipe that exceeds a 2% timing slack limit by assigning infinite penalty energy.
- **Top-K Memory Bank:** Retains the top 10 candidate sequences and verifies them through physical Berkeley ABC compilation, eliminating out-of-distribution neural hallucinations (such as negative power values).

---

##  Experimental Results

### Multi-Objective Prediction Accuracy vs. True Physical Simulation
| Circuit | Real Power ($\mu\text{W}$) | Predicted Power ($\mu\text{W}$) | Power Error | Real Delay ($\text{ps}$) | Predicted Delay ($\text{ps}$) | Delay Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bar.v` | 2695.67 | 2649.42 | **1.72%** | 236.04 | 236.23 | **0.08%** |
| `adder.v` | 722.07 | 733.21 | **1.54%** | 3867.57 | 3799.12 | **1.77%** |
| `c17.v` | 7.91 | 7.34 | **7.26%** | 62.93 | 59.48 | **5.48%** |
| `c1355.v` | 466.24 | 481.56 | **3.29%** | 354.90 | 330.06 | **7.00%** |
| `dec.v` | 180.10 | 173.23 | **3.82%** | 79.95 | 79.06 | **1.12%** |
| `sin.v` | 3802.69 | 4025.71 | **5.86%** | 2711.21 | 2645.80 | **2.41%** |

### Power Optimization vs. Industry Baseline (`resyn2`)
- **`adder.v`:** **+14.77% power reduction** under strict timing constraints.
- **`c1355.v`:** **+12.77% power reduction**.
- **`max.v`:** **+8.09% power reduction**.
- **`bar.v`:** **+3.66% power reduction**.

---

##  Repository Layout

```text
├── abc/                     # Compiled Berkeley ABC binary and scripts
├── data/
│   ├── epfl_circuits/       # Baseline golden circuits (.v / .bench)
│   ├── generated_bench/     # Generated mutant netlists and log files
│   └── tensors/             # Serialized PyG .pt graph tensors
├── dataset_generator.cpp    # Multi-threaded C++ synthesis factory
├── data_labels.py           # Regex log parser aggregating labels.csv
├── bench_to_tensor.py       # Netlist-to-PyG DAG converter
├── power_dataset.py         # Custom streaming Dataset class
├── gnn_model.py             # 3-lobe multi-objective GNN model architecture
├── train.py                 # Base model training routine
├── step1_anchors.py         # K-Means clustering for diverse recipe anchors
├── step2_micro_tune.py      # Graph-shielded few-shot fine-tuning
├── step3_evaluate.py        # Blind accuracy evaluation & validation shield
├── step4_search.py          # Simulated Annealing search engine with Top-K check
└── voter.v                  # Sample verification benchmark
```

---
##  Setup & Execution

### 1. Build Berkeley ABC
```bash
git clone [https://github.com/berkeley-abc/abc.git](https://github.com/berkeley-abc/abc.git) abc/
cd abc && make -j$(nproc) && cd ..
```

### 2. Generate Dataset
```bash
g++ -O3 -std=c++17 -pthread dataset_generator.cpp -o dataset_generator
./dataset_generator
python3 data_labels.py
python3 bench_to_tensor.py
```

### 3. Train Base Model
```bash
python3 train.py
```

### 4. Optimize Target Circuit
```bash
# 1. Generate diverse K-Means anchor sequences
python3 step1_anchors.py

# 2. Run graph-shielded fine-tuning on the anchors
python3 step2_micro_tune.py

# 3. Validate prediction accuracy against ground truth (requires >= 85%)
python3 step3_evaluate.py

# 4. Run Simulated Annealing heuristic search with Top-K physical check
python3 step4_search.py
```
