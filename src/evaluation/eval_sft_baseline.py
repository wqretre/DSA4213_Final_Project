import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.data.data_utils import load_sft_test
from src.data.collators import collate_fn_leftpad
from src.evaluation.metrics import eval_generation
from src.config.config import BASE_MODEL_NAME, DEVICE

# load origninal model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
      
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.to(DEVICE)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

model.eval()
predictions_text = []
model_original.config.pad_token_id = tokenizer.pad_token_id
model_original.base_model.config.pad_token_id = tokenizer.pad_token_id

ds_test, questions, questions_text, references_text = load_sft_test(tokenizer)
dataloader = DataLoader(questions, batch_size=16, collate_fn=lambda batch: collate_fn_leftpad(batch, tokenizer.pad_token_id))

predictions_text = []

for batch, attention_mask in tqdm(dataloader, desc="Generating (original model)"):
    batch = batch.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model_original.generate(
            input_ids=batch,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            use_cache=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions_text_original.extend(decoded)

torch.cuda.empty_cache()

# original model metrics
results_original = eval_generation(predictions_text, references_text, lang="zh", use_bleurt=True)
print("baseline")
print(f"BERTScore_F1: {results_original['BERTScore_F1']:.4f}")
print(f"BLEURT: {results_original['BLEURT']:.4f}")
print(f"Final Score: {results_original['Final_Score']:.4f}")
