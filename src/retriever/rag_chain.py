import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftConfig, PeftModel
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from src.retriever.build_retriever import load_docs, build_hybrid_retriever
from src.retriever.rag_prompt import QA_PROMPT
from src.models.models import get_4bit_bnb_config
from src.config.config import PEFT_PATH


# create RetrievalQA chain
def build_rag_chain(limit_docs=5000):
    docs, load_doc, load_dict = load_docs(limit=limit_docs)
    retriever = build_hybrid_retriever(docs, load_doc)

    config = PeftConfig.from_pretrained(PEFT_PATH)

    bnb_config = get_4bit_bnb_config()

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

    model = PeftModel.from_pretrained(base_model, PEFT_PATH)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=generator)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return qa_chain
