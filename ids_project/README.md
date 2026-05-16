# 🛡️ LLM-Powered Intrusion Detection System (IDS)

> **InfoSec Semester Project 2026 — Tier S**  
> Category A: AI & LLM-Powered Security Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)](https://openai.com)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Detailed Setup](#detailed-setup)
7. [How to Use](#how-to-use)
8. [API Key Setup](#api-key-setup)
9. [Dataset](#dataset)
10. [ML Model](#ml-model)
11. [Tech Stack](#tech-stack)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This system builds an **AI-powered Intrusion Detection System** that:

- **Captures and classifies** network traffic flows using a **Random Forest ML model** (99%+ accuracy)
- **Explains threats** in natural language using **OpenAI GPT-4o-mini** with **RAG-based threat intelligence**
- **Processes big data** using **PySpark** (with automatic pandas fallback)
- **Summarizes alerts** using an **NLP summarizer**
- **Interacts** via a **conversational AI security assistant**

All wrapped in a professional **Streamlit dashboard** with dark-mode UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Classification | Random Forest with 99%+ accuracy on 6 threat classes |
| 🧠 LLM Threat Reports | GPT-4o-mini powered analysis with RAG context |
| 📚 RAG Knowledge Base | TF-IDF retrieval over threat intelligence entries |
| ⚡ PySpark Processing | Batch analysis of large traffic datasets |
| 📋 NLP Summarizer | Executive summaries of alert batches |
| 💬 Security Assistant | Multi-turn RAG-powered chat |
| 🔄 Graceful Fallback | Works fully without OpenAI key |
| 🌑 Dark Mode UI | Professional GitHub-style dark dashboard |

---

## 🏗️ Architecture

```
Network Traffic
      │
      ▼
┌─────────────────┐     ┌──────────────────────┐
│  Feature        │────▶│  Random Forest       │
│  Engineering    │     │  Classifier          │──▶ Prediction + Confidence
│  (9 features)   │     │  (scikit-learn)      │
└─────────────────┘     └──────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  RAG Knowledge Base  │
                        │  (TF-IDF retrieval)  │──▶ Relevant Threat Intel
                        └──────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  LLM Analyzer        │
                        │  (GPT-4o-mini /      │──▶ Human-Readable Report
                        │   Rule-based fallback│
                        └──────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  Streamlit Dashboard │
                        │  (6 pages)           │
                        └──────────────────────┘
```

---

## 📁 Project Structure

```
ids_project/
├── app.py                      ← Main Streamlit application
├── setup_and_run.py            ← One-click setup script
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment variable template
│
├── data/
│   ├── generate_dataset.py     ← Synthetic traffic generator (5,000 records)
│   └── network_traffic.csv     ← Pre-generated dataset (included)
│
├── models/
│   ├── ml_model.py             ← Random Forest training & prediction
│   ├── ids_model.pkl           ← Pre-trained model (included)
│   ├── scaler.pkl              ← Feature scaler
│   └── label_encoder.pkl       ← Class label encoder
│
├── rag/
│   ├── knowledge_base.py       ← TF-IDF RAG retriever
│   └── knowledge_base.json     ← Saved threat intel entries
│
└── utils/
    ├── llm_analyzer.py         ← OpenAI integration + rule-based fallback
    └── spark_processor.py      ← PySpark/Pandas batch processor
```

---

## 🚀 Quick Start

### Option 1: One-Click Setup (Recommended)

```bash
# 1. Unzip and enter project
unzip ids_project.zip
cd ids_project

# 2. Run setup (installs deps, generates data, trains model, launches app)
python setup_and_run.py
```

### Option 2: Manual Setup

```bash
cd ids_project

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_dataset.py

# Train model
python models/ml_model.py

# Launch dashboard
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🔧 Detailed Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required |
| pip | latest | `pip install --upgrade pip` |
| Java 8/11/17 | Optional | Only needed for PySpark (falls back to pandas) |

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `streamlit>=1.32.0` — Dashboard UI
- `scikit-learn>=1.4.0` — ML classification
- `pandas>=2.1.0` — Data processing
- `numpy>=1.26.0` — Numerical computing
- `plotly>=5.18.0` — Interactive charts
- `openai>=1.30.0` — GPT-4o integration
- `python-dotenv>=1.0.0` — Environment variables

**Optional (for PySpark):**
```bash
pip install pyspark>=3.5.0
# Requires Java 8/11/17 installed
```

---

## 📖 How to Use

### Dashboard (🏠)
Overview of all alerts with metrics, pie chart distribution, and timeline.

### Analyze Traffic (🔍)
Two modes:
- **Manual Input** — Enter packet details manually and get instant classification + AI report
- **Simulate Attack** — Generate realistic attack/benign samples and analyze them

### Batch Analysis (📊)
- Run PySpark/Pandas analysis on the full 5,000-record dataset
- View threat distribution, protocol breakdown, top attacker IPs, attacked ports
- Generate AI executive summary

### ML Model (🧠)
- View accuracy metrics (99%+), confusion matrix, feature importance
- Architecture details and hyperparameters

### Threat KB (📚)
- Search the RAG knowledge base for threat intelligence
- Browse all 7 threat entries with severity levels and mitigations

### Security Assistant (💬)
- Chat with the RAG-powered security assistant
- Ask about specific attacks, mitigations, IDS concepts
- Use NLP summarizer to summarize all logged alerts

---

## 🔑 API Key Setup

The system works **fully without an OpenAI API key** using rule-based fallback.  
To enable GPT-4o-powered analysis:

### Method 1: Sidebar Input
Paste your key directly in the sidebar API key field when the app is running.

### Method 2: Environment File
```bash
cp .env.example .env
# Edit .env:
OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY_HERE
```

### Method 3: Environment Variable
```bash
export OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
streamlit run app.py
```

> ⚠️ **Security:** Never share your API key in chat messages, code files, or screenshots. Rotate immediately if exposed at https://platform.openai.com/api-keys

---

## 📊 Dataset

The synthetic dataset simulates realistic network traffic with 6 classes:

| Label | % of Dataset | Key Characteristics |
|---|---|---|
| Benign | 50% | Normal HTTP/DNS/SSH traffic |
| DoS | 15% | High packet rate, SYN flag, ports 80/443 |
| PortScan | 12% | Varying dst_port, low byte count |
| BruteForce | 10% | Ports 22/21/3389, high repeat rate |
| SQLInjection | 8% | HTTP ports, anomalous payload size |
| Backdoor | 5% | Non-standard high ports, long duration |

**Features:**

| Feature | Type | Description |
|---|---|---|
| src_port / dst_port | int | Source and destination ports |
| packet_count | int | Total packets in flow |
| byte_count | int | Total bytes transferred |
| duration_ms | int | Connection duration |
| packets_per_sec | float | Derived rate feature |
| avg_packet_size | float | Derived size feature |
| protocol_enc | int | TCP=0, UDP=1, ICMP=2 |
| flag_enc | int | SYN=0, ACK=1, etc. |

---

## 🤖 ML Model

**Algorithm:** Random Forest Classifier

| Metric | Value |
|---|---|
| Accuracy | 99%+ |
| F1 Score (weighted) | 0.99+ |
| Training records | 4,000 |
| Test records | 1,000 |
| n_estimators | 100 |
| max_depth | 15 |

**Top features by importance:**
1. `packet_count` (~37%)
2. `byte_count` (~18%)
3. `packets_per_sec` (~14%)
4. `duration_ms` (~13%)
5. `avg_packet_size` (~7%)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Interactive dashboard |
| scikit-learn | Random Forest ML model |
| OpenAI GPT-4o-mini | LLM-powered threat reports |
| PySpark / Pandas | Big data batch processing |
| TF-IDF (sklearn) | RAG retrieval engine |
| Plotly | Interactive visualizations |
| python-dotenv | Environment management |

---

## 🐛 Troubleshooting

### "Model not trained" warning
→ Click **Train ML Model** in the sidebar, or run `python models/ml_model.py`

### OpenAI Error 429 (quota exceeded) / 401 (invalid key)
→ The system automatically falls back to rule-based reports. No action needed.  
→ To use GPT-4o: get a new key at https://platform.openai.com/api-keys and paste it in the sidebar.

### PySpark not available
→ The system automatically uses pandas. To enable PySpark: install Java 8/11/17, then `pip install pyspark`

### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Dataset not found
→ Click **Generate Dataset** in the sidebar or run `python data/generate_dataset.py`

---

## 📄 License

Academic project — InfoSec Semester 2026. Not for production use.

---

*Built with ❤️ using Python · Streamlit · scikit-learn · OpenAI · PySpark*
