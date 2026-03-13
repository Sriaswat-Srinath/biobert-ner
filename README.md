# Biomedical Named Entity Recognition (BioBERT-NER) Pipeline

A comprehensive machine learning and big data engineering project that leverages **BioBERT** for extracting critical medical entities (Diseases, Drugs, Symptoms, Anatomy, etc.) from clinical abstracts. This repository demonstrates an end-to-end MLOps and Big Data architecture, ranging from PyTorch-based model fine-tuning with LoRA, to real-time inference streaming via Apache Kafka, Hadoop, and Spark.

## 🌟 Project Architecture

This repository is divided into four major logical components across the `src/` directory:

### 1. Model Development & Training (`src/ner_model/`)
A scalable Python pipeline to train and optimize a custom BioBERT Token Classifier.
* **`01_preprocess_data.py`** - Advanced linguistic boundary refinement using SpaCy, rule-based conflict resolution, and UMLS Semantic Network type mappings (parsing PubTator format).
* **`02_EDA.py`** & **`03_ClassLabelsTrainTestSplit.py`** - Exploratory Data Analysis and class-balanced dataset splitting.
* **`04_Trainer.py`** - High-performance model training using HuggingFace `Trainer`. Features **Parameter-Efficient Fine-Tuning (PEFT)** with **QLoRA** (4-bit quantization via `bitsandbytes`) to fine-tune the `dmis-lab/biobert-base-cased-v1.2` model efficiently. Includes custom weighted cross-entropy loss to handle class imbalance.

### 2. Interactive Inference App (`src/demo/`)
* **`demo_app.py`** - A visually appealing, interactive **Streamlit** Python web application allowing users to paste clinical text and view parsed entity extractions bounding boxes and confidence metrics in real-time. Features dynamic fallback configuration to run with either the base or locally fine-tuned adapter weights.

### 3. Distributed Data Streaming (`src/big_data_pipeline/`)
* **`docker-compose.yml`** - A multi-container orchestration file spinning up an enterprise Big Data environment, including **Apache Kafka**, **Zookeeper**, and a distributed **Hadoop/HDFS** (NameNode & DataNode) cluster for high-throughput messaging and storage.
* **`producer.py`** - A Python-based Kafka Producer to push large volumes of biomedical text into the data streams.

### 4. Real-time Big Data Processing (`src/spark_poc/`)
A **Scala** and **Apache Spark** project built with SBT (`build.sbt`) designed for distributed stream processing.
* Built using `spark-sql`, `spark-mllib`, and **Spark NLP**.
* Leverages **Akka Streams** and the `kafka-clients` library to consume, distribute, and process inference workloads from the Kafka topics in real-time.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Docker Desktop (for the Big Data Pipeline)
* SBT and Java 8/11 (for the Spark POC)

### Local Development Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sriaswat-Srinath/biobert-ner.git
   cd biobert-ner
   ```
2. **Install Python dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run the Interactive Demo:**
   Ensure the Streamlit application operates effectively.
   ```bash
   streamlit run src/demo/demo_app.py
   ```

### Booting the Big Data Pipeline
To initialize the Kafka Message Brokers and HDFS Cluster:
```bash
cd src/big_data_pipeline
docker-compose up -d
```

---

## 🧬 Supported Entity Classes
The fine-tuned BioBERT model is capable of classifying the following distinct biomedical entity categories:
* `ANATOMY`
* `CELL_LINE`
* `CHEMICAL`
* `DIAGNOSTIC_TEST`
* `DISEASE`
* `DRUG`
* `GENE_PROTEIN`
* `QUANTITY`
* `SYMPTOM`
* `TREATMENT`
