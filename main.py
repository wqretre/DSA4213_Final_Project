import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset,Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from bitsandbytes.optim import Adam8bit
import matplotlib.pyplot as plt
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
import torch
import jieba
import gc
import evaluate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 1024

# process dataset
def preprocess_example(ex):
    questions = ex.get("questions", [])
    if isinstance(questions, list) and len(questions) > 0:
        if isinstance(questions[0], list):
            question = " ".join([q[0].strip() if len(q) > 0 else "" for q in questions]).strip()
        else:
            question = questions[0].strip()
    else:
        question = str(questions).strip()

    answers = ex.get("answers", [])
    if isinstance(answers, list) and len(answers) > 0:
        if isinstance(answers[0], list):
            answers = " ".join([a[0].strip() if len(a) > 0 else "" for a in answers]).strip()
        else:
            answers = str(answers[0]).strip()
    else:
        answers = str(answers).strip()

    if not answers:
        return None

    return {"input": question, "output": answers}


def tokenize_function(ex, tokenizer, max_len=MAX_LEN):
    inp = ex["input"].strip()
    out = ex["output"].strip()
    eos = tokenizer.eos_token
    text = inp + eos + out + eos
    tok = tokenizer(text, truncation=True, max_length=max_len)

    inp_tok = tokenizer(inp + eos, truncation=True, max_length=max_len)
    labels = [-100] * len(tok["input_ids"])
    labels[len(inp_tok["input_ids"]):] = tok["input_ids"][len(inp_tok["input_ids"]):]
    tok["labels"] = labels
    return tok

# load the model with 4-bit quantization
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.to(device)
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
model.print_trainable_parameters()

# load train and validation set
ds_train = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="train")
ds_val = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="validation")

ds_train = ds_train.map(preprocess_example, remove_columns=ds_train.column_names)
ds_val = ds_val.map(preprocess_example, remove_columns=ds_val.column_names)

ds_train = ds_train.filter(lambda x: x is not None)
ds_val = ds_val.filter(lambda x: x is not None)

ds_train = ds_train.shuffle(seed=42).select(range(10000))
ds_val = ds_val.shuffle(seed=42).select(range(1000))

ds_train = ds_train.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])
ds_val = ds_val.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])

# pad inputs_id and lebels to the same length
def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

# train the model 
training_args = TrainingArguments(
    output_dir="./output/qwen3-1.7b-sft-huatuo",
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
    data_collator=collate_fn,
    optimizers=(optimizer, None)
)

trainer.train()
trainer.save_model("./output/qwen3-1.7b-sft-huatuo")

# plot the loss curve
train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# load test set
ds_test = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="test")
ds_test = ds_test.map(preprocess_example, remove_columns=ds_test.column_names)
ds_test = ds_test.filter(lambda x: x is not None)
ds_test = ds_test.shuffle(seed=100).select(range(400))
ds_test = ds_test.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])

inputs = [x['input_ids'] for x in ds_test]
references = [x['labels'] for x in ds_test] 
questions = [] 
questions_text = [] # questions in the test set
references_text = [] # answers
for i in range(len(ds_test)):
    label_ids = [id for id in references[i] if id != -100]
    references_text.append(tokenizer.decode(label_ids, skip_special_tokens=True))
    question = []
    for j,id in enumerate(references[i]):
        if id == -100:
            question.append(inputs[i][j])
    questions.append(question)
    questions_text.append(tokenizer.decode(question, skip_special_tokens=True))

# evaluation metrics
def eval_generation(preds, refs, lang="zh", use_bleurt=True):
    assert len(preds) == len(refs)

    # BERTScore
    bertscore = evaluate.load("bertscore")
    bs = bertscore.compute(predictions=preds, references=refs, lang=lang)
    bs_f1 = float(np.mean(bs["f1"]))

    # BLEURT
    bleurt_score = None
    if use_bleurt:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
        except Exception:
            pass

        bleurt = evaluate.load("bleurt", config_name="BLEURT-20")
        bl = bleurt.compute(predictions=preds, references=refs)
        bleurt_score = float(np.mean(bl["scores"]))

    final_score = 0.6 * bs_f1 + 0.4 * ((bleurt_score + 1) / 2)
    return {
        "BERTScore_F1": bs_f1,
        "BLEURT": bleurt_score,
        "Final_Score": final_score
    }

def collate_fn_leftpad(batch):
    batch = [torch.tensor(x, dtype=torch.long) for x in batch]
    max_len = max(len(x) for x in batch)
    pad_id = tokenizer.pad_token_id
    # left padding
    padded = torch.stack([
        torch.cat([torch.full((max_len - len(x),), pad_id, dtype=torch.long), x])
        for x in batch
    ])
    attention_mask = (padded != pad_id).long()
    return padded, attention_mask

peft_path = "./output/qwen3-1.7b-sft-huatuo"

# load PEFT configuration
config = PeftConfig.from_pretrained(peft_path)

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
model_finetune = PeftModel.from_pretrained(base_model, peft_path)

# generate answers
model_finetune.eval()
BATCH_SIZE = 16
dataloader = DataLoader(questions, batch_size=BATCH_SIZE, collate_fn=collate_fn_leftpad)
model_finetune.config.pad_token_id = tokenizer.pad_token_id
model_finetune.base_model.config.pad_token_id = tokenizer.pad_token_id

predictions_text = []
model_finetune.config.use_cache = True
for batch, attention_mask in tqdm(dataloader, desc="Generating"):
    batch = batch.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model_finetune.generate(
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

# load origninal model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model_original = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model_original.to(device)
model_original.gradient_checkpointing_enable()
model_original.config.use_cache = False
model_original = prepare_model_for_kbit_training(model_original)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model_original = get_peft_model(model_original, lora_cfg)

model_original.eval()
predictions_text_original = []
model_original.config.pad_token_id = tokenizer.pad_token_id
model_original.base_model.config.pad_token_id = tokenizer.pad_token_id
BATCH_SIZE = 16

dataloader = DataLoader(questions, batch_size=BATCH_SIZE, collate_fn=collate_fn_leftpad)

predictions_text_original = []

for batch, attention_mask in tqdm(dataloader, desc="Generating (original model)"):
    batch = batch.to(device)
    attention_mask = attention_mask.to(device)

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
results_original = eval_generation(predictions_text_original, references_text, lang="zh", use_bleurt=True)
print("baseline")
print(f"BERTScore_F1: {results_original['BERTScore_F1']:.4f}")
print(f"BLEURT: {results_original['BLEURT']:.4f}")
print(f"Final Score: {results_original['Final_Score']:.4f}")

# RAG
# load medical dataset
ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train")
docs = []
load_dict = {}
load_docs = []
for i, record in enumerate(ds):
    q = record.get("question", "")
    a = record.get("answer", "")
    if q and a:
        load_dict[a] = q
        content = a
        load_docs.append(content)
        docs.append(Document(page_content=content))
    if i >= 5000: 
        break
print(f"Loaded {len(docs)} documents.")

# generate embedding vectors and save in FAISS vector database
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={"device": "cuda"})
vectorstore = FAISS.from_documents(docs, embedding_model)

# hybrid retrieval method (BM25 + dense retrieval)
BM25_en_retriever = BM25Retriever.from_texts(load_docs, preprocess_func=lambda text: list(jieba.cut(text)))
BM25_en_retriever.k = 3
BGE_en_vectorstore = vectorstore
BGE_en_retrieverk = BGE_en_vectorstore.as_retriever(search_kwargs={"k": 3})
ensemble_retriever = EnsembleRetriever(retrievers=[BM25_en_retriever, BGE_en_retrieverk], weights=[0.5, 0.5])

# evaluate retrieval performance
def eval_one(pair):
    load_text, question = pair
    docs = ensemble_retriever.invoke(question)
    contents = [doc.page_content for doc in docs]
    if load_text in contents:
        pos = contents.index(load_text) + 1
        return (1, 1 / pos)
    return (0, 0)
pairs = list(load_dict.items())
hit_rate, mrr = 0, 0
with ThreadPoolExecutor(max_workers=24) as executor:
    futures = [executor.submit(eval_one, p) for p in pairs]
    for f in tqdm(as_completed(futures), total=len(pairs), desc="Evaluating"):
        hit, reciprocal = f.result()
        hit_rate += hit
        mrr += reciprocal
hit_rate /= len(pairs)
mrr /= len(pairs)
print(f"Hit Rate: {hit_rate:.4f}")
print(f"MRR: {mrr:.4f}")

peft_path = "./output/qwen3-1.7b-sft-huatuo"

config = PeftConfig.from_pretrained(peft_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               
    bnb_4bit_quant_type="nf4",      
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16 
)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,  
    device_map="auto",              
    trust_remote_code=True
)

model_finetune = PeftModel.from_pretrained(base_model, peft_path)
generator = pipeline(
    "text-generation",
    model=model_finetune,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.5,
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=generator)

# prompt engineering (few shot + chain of thought)
prompt_template = """
你是一名专业的医学顾问，回答问题时请严格遵循以下步骤：
1. 先思考 (“Thinking”)，对问题的关键因素进行分析。
2. 用逻辑推理逐步得出答案 (Chain-of-Thought)。
3. 给出详细答案，包括原因、机制、常见表现以及可能的注意事项。

示例：
问题：发烧可能由哪些因素引起？
思考：发烧是人体对病原体或异常状态的反应，常见原因包括病毒、细菌感染或免疫反应。不同病因可能伴随不同症状，需要结合临床表现分析。
答案：发烧可能由多种因素引起，包括：
- **病毒感染**：如流感、腮腺炎，常伴有乏力、头痛。
- **细菌感染**：如肺炎、尿路感染，常伴有局部感染症状。
- **免疫反应或炎症**：如自身免疫性疾病，可能伴随关节疼痛、皮疹。
临床上需要结合其他症状和检查来确定具体原因，并注意发热的持续时间和程度。

问题：{question}
参考资料：
{context}
思考：
"""

QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

# RAG metrics
ds_test = Dataset.from_dict({"question": questions_text})

def generate_batch(batch):
    outputs = generator(batch["question"], max_new_tokens=512, batch_size=8)
    preds = []
    for out in outputs:
        text = out[0]['generated_text']
        if "答案：" in text:
            text = text.split("答案：")[-1].split("问题")[0]
        preds.append(text)
    return {"prediction": preds}

ds_test = ds_test.map(generate_batch, batched=True, batch_size=8)
predictions_text_RAG = ds_test["prediction"]
results_RAG = eval_generation(predictions_text_RAG, references_text, lang="zh", use_bleurt=True)
print("RAG + Prompt")
print(f"BERTScore_F1: {results_RAG['BERTScore_F1']:.4f}")
print(f"BLEURT: {results_RAG['BLEURT']:.4f}")
print(f"Final Score: {results_RAG['Final_Score']:.4f}")
