from datasets import load_dataset
from tqdm import tqdm

from src.retriever.rag_chain import build_rag_chain
from src.evaluation.metrics import eval_generation
from src.data.data_utils import load_sft_test

def generate_batch(generator, batch):
    outputs = generator(batch["question"], max_new_tokens=512, batch_size=8)
    preds = []
    for out in outputs:
        text = out[0]['generated_text']
        if "答案：" in text:
            text = text.split("答案：")[-1].split("问题")[0]
        preds.append(text)
    return {"prediction": preds}
  
qa_chain = build_rag_chain(limit_docs=5000)
generator = qa_chain.llm.pipeline     
tokenizer = generator.tokenizer

_, _, questions_text, references_text = load_sft_test(tokenizer)
ds_test = Dataset.from_dict({"question": questions_text})
ds_test = ds_test.map(lambda batch: generate_batch(generator, batch), batched=True, batch_size=8)
predictions_text_RAG = ds_test["prediction"]
results_RAG = eval_generation(predictions_text_RAG, references_text, lang="zh", use_bleurt=True)
print("RAG + Prompt")
print(f"BERTScore_F1: {results_RAG['BERTScore_F1']:.4f}")
print(f"BLEURT: {results_RAG['BLEURT']:.4f}")
print(f"Final Score: {results_RAG['Final_Score']:.4f}")
