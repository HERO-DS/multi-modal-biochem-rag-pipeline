# Multi-Modal Biochemical RAG Pipeline 🧬🧠

An enterprise-grade, retrieval-augmented generation (RAG) pipeline designed for medicinal chemists and computational toxicologists. This system extracts deep structural representations of small molecules via specialized chemical embeddings, predicts Blood-Brain Barrier (BBB) permeability metrics, and retrieves matched spatial vector space analogs to deliver automated, context-aware expert clinical dossiers.

---

## 🏗️ Architectural Overview

The system bridges local high-dimensional vector embeddings with remote large language models via a decoupled, microservice-inspired blueprint:

1. **Frontend Presentation (`src/app.py`)**: A Streamlit dashboard collecting structural SMILES strings and visualizing vector spaces, inference results, and dynamic reports.
2. **Backend Engine (`src/serving_app.py`)**: A high-performance FastAPI service computing model inferences and coordinating database traffic.
3. **Retrieval Router (`src/rag_engine.py`)**: The system core. It extracts mathematical coordinate maps, interfaces with a local vector database, and formats context structures for cloud synthesis.
4. **Vector Storage (`src/vector_store.py`)**: A local instances instance of ChromaDB indexing a curated structural chemical dataset.

---

## 🛡️ Core Case Study: Catching the "Caffeine Contradiction"

Standard machine learning models often fail silently or produce contradictory metrics on out-of-distribution chemical structures. 

During internal testing with **Caffeine** (`CN1C=NC2=C1C(=O)N(C(=O)N2C)C`), the pipeline demonstrated the immense value of its **Retrieval-Augmented Architecture**:
* **The ML Conflict**: The system's regression layer predicted high brain penetration ($LogBB = 1.4006$), while its classification layer erroneously asserted that barrier passage was blocked ($BBB-$) with a weak confidence threshold ($54.27\%$).
* **The RAG Intervention**: Recognizing a lack of nearby historical vector space analogs, the pipeline fed both conflicting internal data arrays and empirical properties directly to the generative agent. 
* **The Result**: Instead of trusting a broken inference score, the pipeline overrode its own faulty classification layer, alerting the researcher that the compound was a canonical CNS-active stimulant.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Frameworks**: FastAPI, Streamlit, Uvicorn
* **AI & Embeddings**: Google Gemini 2.5 Production Flash Engine, Numpy
* **Vector DB**: ChromaDB (Serverless, Locally Persistent Store)

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
Clone the repository and install requirements inside a Python virtual environment:
```bash
git clone [https://github.com/HERO-DS/multi-modal-biochem-rag-pipeline.git](https://github.com/HERO-DS/multi-modal-biochem-rag-pipeline.git)
cd multi-modal-biochem-rag-pipeline