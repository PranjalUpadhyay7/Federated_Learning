# Federated Learning for Mental Wellbeing Prediction 🧠📱

> **How can we predict human mental wellbeing without invading users' privacy?**
> This project builds a privacy-first machine learning system that tracks mood and behavioral shifts directly on a mobile phone without sending personal data to the cloud.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FFB000)

## 📖 Table of Contents
- [Architecture & Flow](#-architecture--flow)
- [Project Overview](#-project-overview)
- [The DiversityOne Dataset](#-the-diversityone-dataset)
- [Deep Learning Models](#-deep-learning-models)
- [Installation & Setup](#-installation--setup)
- [Usage: Centralized vs Federated](#-usage-centralized-vs-federated)
- [Key Findings & Generalization](#-key-findings--generalization)

---

## 🏗 Architecture & Flow

Instead of pooling sensitive smartphone and survey data into a centralized database, we use **Federated Learning**. 
1. **Local Training:** The AI model lives on the user's phone, continuously learning from their specific daily behavior.
2. **Privacy Preserved:** Their steps, screen time, location data, and mental health metrics NEVER leave their device.
3. **Global Collaboration:** The single thing the phone shares with the central server is a numerical "understanding" (a weight update). The server averages all these anonymous updates together to build a robust, global master model.

```mermaid
graph TD
    %% Styling variables
    classDef edge fill:#1A202C,stroke:#38B2AC,stroke-width:3px,color:#E2E8F0;
    classDef cloud fill:#2D3748,stroke:#ED64A6,stroke-width:3px,color:#F7FAFC,shape:hexagon;
    classDef model fill:#3182CE,stroke:#E2E8F0,stroke-width:2px,color:#FFFFFF,rx:5,ry:5;
    classDef data fill:#48BB78,stroke:#E2E8F0,stroke-width:2px,color:#FFFFFF,shape:cylinder;
    classDef mainBg fill:#edf2f7,stroke:#A0AEC0,stroke-width:2px;

    S{{"☁️ Federated Aggregator Layer<br>(Server)"}}:::cloud

    subgraph "🔐 Edge Environment (Privacy Maintained)"
        direction LR
        
        subgraph C1 ["📍 Client 1 (China)"]
            D1[("Sensor Data")]:::data --> M1["Local Training"]:::model
        end
        
        subgraph C2 ["📍 Client 2 (UK)"]
            D2[("Sensor Data")]:::data --> M2["Local Training"]:::model
        end
        
        subgraph CN ["📍 Client N (Denmark)"]
            DN[("Sensor Data")]:::data --> MN["Local Training"]:::model
        end
    end

    %% Communication Flow
    M1 -. "1. Gradients" .-> S
    M2 -. "1. Gradients" .-> S
    MN -. "1. Gradients" .-> S

    S -- "2. FedAvg Aggregation" --> S

    S == "3. Global Weights" ==> M1
    S == "3. Global Weights" ==> M2
    S == "3. Global Weights" ==> MN

    class C1,C2,CN edge;
    %% Using a generic ID for the subgraph to apply styles safely
```

### Module Codebase Dependency Graph
```mermaid
graph LR
    classDef common fill:#FFF2CC,stroke:#ED8936,stroke-width:2px,color:#2D3748;
    classDef model fill:#EBF8FF,stroke:#4299E1,stroke-width:2px,color:#2D3748;
    classDef main fill:#E6FFFA,stroke:#38B2AC,stroke-width:2px,color:#2D3748;
    classDef fl fill:#C3DAFE,stroke:#3182CE,stroke-width:2px,color:#2D3748;
    
    C[("common/<br>data_preprocessing/dataset.py")]:::common
    M[("centralized/<br>model.py")]:::model
    
    TC["centralized/<br>train_central.py"]:::main
    
    FL["federated/<br>fl.py"]:::fl
    FLS["federated/<br>server.py"]:::fl
    FLC["federated/<br>client.py"]:::fl
    
    MAIN{"main.py<br>(Evaluator)"}:::main
    
    C --> TC
    M --> TC
    
    C --> FL
    M --> FL
    
    TC --> MAIN
    FL --> MAIN
    
    FL --> FLS
    FLS --> FLC
```


---

## 🌍 Project Overview
Mental health tracking usually relies on invasive tracking or globally biased models. This project presents a dual study on **Mood Prediction** and **Cross-Country Generalization** relying purely on Edge AI tools.

We analyze two paradigms:
1. **Multimodal Mood Classification:** Using deep architectures on passive sensors and self-reports natively on the client.
2. **Cross-Country Regression (Domain Shift):** Analyzing how an FL model trains over diverse global datasets compared to a centralized approach, simulating phenomena like **Client Drift**, **Mode Collapse**, and **Cold States (Cascading FL)**.

---

## 📊 The DiversityOne Dataset
We use the **DiversityOne** dataset collected via the `iLog` app.
- **Scope:** 666 active participants across 8 countries (Global North and Global South).
- **Data Modalities:**
  - **Sensors:** Accelerometer, Bluetooth, Screen Status, Battery, Location variance, etc. (high frequency).
  - **Static Profiles:** Demographics, Big Five Inventory personality traits.
- **Benefit:** Highly resistant to geographic model bias.

---

## 🧠 Deep Learning Models
The solution incorporates distinct architectures tailored for dynamic tabular/time-series data:
- **`TimeSeriesLSTM`**: Ingests sequential, fine-grained temporal sensor data over rolling windows to capture fatigue, movement habits, and interaction consistency.
- **`StaticNet` (MLP)**: Processes single-frame localized inputs and static questionnaires (e.g., Demographics, Jungian scale).
- **`TabM_Single` (Tabular Foundation)**: Optimized specifically for categorical and mixed tabular behavioral records across participants.

### Model Performance (Mood Classification Accuracy)
| Model | Accuracy | Verdict |
|-------|----------|---------|
| **TabM (Tabular Ensemble)** | **69.95%** | **Best Performer** |
| **LSTM (Sequential)** | 68.28% | Competitive |
| **StaticNet (Deep MLP)** | 56.11% | Baseline |

### Model Stability & Convergence (Loss Metrics)
| Model | Loss (Lower is Better) | Analysis |
|-------|------------------------|----------|
| **TabM** | **1.5** | **Most Robust** |
| **LSTM** | 1.9 | Moderate Stability |
| **StaticNet**| 2.2 | Poor Convergence |

---

## ⚙️ Installation & Setup

1. **Clone and Install Dependencies:**
```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements via pip
pip install -r requirements.txt

# Or, if using Hatch/UV (based on pyproject.toml):
pip install .
```

---

## 🚀 Usage: Centralized vs Federated

### 1. Centralized Baseline (The "Global Oracle")
To run the traditional, non-privacy centralized training loop on the aggregated dataset:
```bash
python centralized/train_central.py
```

### 2. Federated Distributed Training (`flwr`)
To start the multi-node, zero-trust training, open separate terminal windows.

**Start the FL Server:**
```bash
flower-server-app federated.server:app --insecure
# Or directly run: flwr run .
```

**Start Local Clients (Edge nodes):**
```bash
flower-client-app federated.client:app --insecure
```

### 3. Model Back-to-Back Evaluation
To evaluate the Centralized Weights against the Federated Weights on downstream tasks, run the unified entry point:
```bash
python main.py
```
This generates an end-to-end `classification_report` testing how well the global federated edge models stood off against a pure centralized database.

---

## 📉 Key Findings & Generalization
- **Privacy vs Performance:** FL achieves parity against heavily centralized oracles when handling mild Non-IID distributions but highlights significant **client drift** when training strictly on heavily skewed individual user states.
- **Cross-Cultural Shift:** Generalization from Global South (Jilin, China) to Global North (London/Copenhagen) demonstrates how Federated Learning inherently handles generalized cross-demographic variances better over prolonged multi-round server updates. 
- **Cold-Start Validation:** Integrating "Cascading FL" simulates new user app installations perfectly, preventing system halts on minimal initial data drops.
