import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"

SFT_OUTPUT_DIR = "./output/qwen3-1.7b-sft-huatuo"
PEFT_PATH = SFT_OUTPUT_DIR  

# fine-tuning dataset
TRAIN_DATASET_NAME = "FreedomIntelligence/huatuo_encyclopedia_qa"
TEST_DATASET_NAME = "FreedomIntelligence/huatuo_encyclopedia_qa"

# RAG corpus
RAG_CORPUS_NAME = "FreedomIntelligence/Huatuo26M-Lite"

# sequence length
MAX_LEN = 1024
