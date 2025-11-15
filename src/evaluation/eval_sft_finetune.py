import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

from src.data.data_utils import load_sft_test
from src.data.collators import collate_fn_leftpad
from src.evaluation.metrics import eval_generation
from src.config.config import PEFT_PATH, DEVICE, BASE_MODEL_NAME

config = PeftConfig.from_pretrained(PEFT_PATH)

# quantization setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,             
    bnb_4bit_quant_type="nf4",       
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16  
)


tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,  
    device_map="auto",               
    trust_remote_code=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base_model, PEFT_PATH)

# generate answers
model.eval()
model.config.use_cache = True
model.config.pad_token_id = tokenizer.pad_token_id
model.base_model.config.pad_token_id = tokenizer.pad_token_id

# load test dataset
ds_test, questions, questions_text, references_text = load_sft_test(tokenizer)

dataloader = DataLoader(questions, batch_size=16, collate_fn=lambda batch: collate_fn_leftpad(batch, tokenizer.pad_token_id))

predictions_text = []
for batch, attention_mask in tqdm(dataloader, desc="Generating"):
    batch = batch.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            use_cache=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions_text.extend(decoded)

torch.cuda.empty_cache()

# fine-tune metrics
results = eval_generation(predictions_text, references_text, lang="zh", use_bleurt=True)
print("fine-tune")
print(f"BERTScore_F1: {results['BERTScore_F1']:.4f}")
print(f"BLEURT: {results['BLEURT']:.4f}")
print(f"Final Score: {results['Final_Score']:.4f}")
