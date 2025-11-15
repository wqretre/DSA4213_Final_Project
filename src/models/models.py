import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.config.config import BASE_MODEL_NAME, DEVICE

# load the model with 4-bit quantization
def get_4bit_bnb_config():
  return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",              
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.float16 
  )

def load_base_model_4bit(model_name=BASE_MODEL_NAME):
  bnb_config = get_4bit_bnb_config()
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      trust_remote_code=True
  )
  model.to(DEVICE)
  model.gradient_checkpointing_enable()
  model.config.use_cache = False
  model = prepare_model_for_kbit_training(model)
  return model

def get_tokenizer(model_name=BASE_MODEL_NAME):
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer

def apply_lora_for_training(model):
  lora_cfg = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
  )
  model = get_peft_model(model, lora_cfg)
  model.print_trainable_parameters()
  return model
