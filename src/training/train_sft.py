import torch
from bitsandbytes.optim import Adam8bit
from transformers import TrainingArguments, Trainer

from src.models.models import (
    get_tokenizer,
    load_base_model_4bit,
    apply_lora_for_training,
)
from src.data.data_utils import load_sft_train_val
from src.data.collators import collate_fn_sft
from src.models.plotting import plot_loss_curve
from src.config.config import SFT_OUTPUT_DIR, DEVICE

tokenizer = get_tokenizer()
ds_train, ds_val = load_sft_train_val(tokenizer)
model = load_base_model_4bit()
model = apply_lora_for_training(model)

# train the model 
training_args = TrainingArguments(
    output_dir=SFT_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=5,
    eval_strategy="no", 
    save_strategy="steps",
    save_steps=600,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none"
)
optimizer = Adam8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    data_collator=lambda batch: collate_fn_sft(batch, tokenizer.pad_token_id),
    optimizers=(optimizer, None)
)
  
trainer.train()
trainer.save_model(SFT_OUTPUT_DIR)
plot_loss_curve(trainer)

