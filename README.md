# DSA4213_Final_Project

This repository contains code for:
- **4-bit QLoRA supervised fine-tuning** of Qwen3-1.7B on Chinese medical QA
- **Hybrid retrieval (BM25 + BGE embeddings)** for knowledge retrieval
- **RAG (Retrieval-Augmented Generation)** with medical chain-of-thought prompting

The project evaluates:
- **Baseline Qwen3-1.7B**
- **Fine-tuned QLoRA model**
- **RAG-enhanced model**

---

## Project Structure
```text
DSA4213_Final_Project/
│
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    │
    ├── config/
    │   ├── __init__.py
    │   └── config.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── preprocess.py            # preprocess_example()
    │   ├── tokenization.py          # tokenize_function()
    │   ├── collators.py             # pad_sequence, left-padding
    │   └── data_utils.py            # load_sft_test()
    │
    ├── models/
    │   ├── __init__.py
    │   ├── models.py                # base model loader + 4bit config
    │   └── plotting.py              # training loss plotting
    │
    ├── training/
    │   ├── __init__.py
    │   └── train_sft.py             # QLoRA supervised finetuning
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py               # BERTScore / BLEURT / final score
    │   ├── eval_sft_finetune.py     # evaluate QLoRA fine-tuned model
    │   ├── eval_sft_baseline.py     # evaluate original baseline model
    │   └── rag_eval.py              # RAG batch evaluation
    │
    ├── retriever/
    │   ├── __init__.py
    │   ├── build_retriever.py       # BM25 + BGE hybrid retriever + metrics
    │   ├── rag_prompt.py            # Medical CoT prompt template
    │   └── rag_chain.py             # hybrid retriever + model → RetrievalQA chain
    │
    └── __init__.py

```
## Overview

| Item | Description |
|------|--------------|
| **Dataset (SFT)** | [Huatuo-Encyclopedia-QA](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa) |
| **Dataset (RAG)** | [Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite) |
| **Model** | Qwen/Qwen3-1.7B |
| **Finetuning** | QLoRA (4-bit NF4) |
| **Retrieval** | BM25 + BGE-large-zh-v1.5 |
| **Evaluation** | BERTScore, BLEURT, Hit Rate, MRR |

---
## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/wqretre/DSA4213_Final_Project.git
cd DSA4213_Final_Project
```
### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## Run Supervised Fine-Tuning (QLoRA)
```bash
python -m src.training.train_sft
```
This trains **Qwen3-1.7B** with LoRA adapters in 4-bit quantization and saves the model to:
```bash
output/qwen3-1.7b-sft-huatuo/
```

## Evaluate Finetuned Model
### Baseline (Original model)
```bash
python -m src.evaluation.eval_sft_baseline
```
### Fine-tuned QLoRA model
```bash
python -m src.evaluation.eval_sft_finetune
```
This project evaluates model performance using:

- **BERTScore-F1** — semantic similarity between predicted and reference answers  
- **BLEURT-20** — learned metric for factual correctness and coherence  
- **Final Combined Score** — weighted fusion of BERTScore and BLEURT  

Finetuning (QLoRA) consistently improves **fluency, completeness, and factual accuracy** compared to the baseline model.

## Build and Evaluate Hybrid Retriever
```bash
python -m src.retriever.build_retriever
```

This project evaluates model performance using:

- **Hit Rate**  
- **MRR (Mean Reciprocal Rank)**

This helps quantify how well the retriever can find the correct medical passage.

## Run RAG Evaluation (Hybrid Retrieval + Medical CoT Prompt)
```bash
python -m src.evaluation.rag_eval
```
he RAG evaluation pipeline performs the following steps:

1. **Loads the hybrid retriever** (BM25 + BGE embeddings) to retrieve relevant medical knowledge.
2. **Loads the fine-tuned QLoRA model** for generation under low-memory 4-bit quantization.
3. **Applies a Medical Chain-of-Thought prompt**, guiding the model to reason before answering.
4. **Generates answers in batch**, enabling efficient large-scale evaluation.
5. **Evaluates predictions** using BERTScore and BLEURT.

This RAG setup typically provides improvements in accuracy, factual correctness, and response consistency by grounding model outputs in retrieved evidence.
