from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import jieba

from src.config.config import RAG_CORPUS_NAME


def load_docs(limit=5000):
    ds = load_dataset(RAG_CORPUS_NAME, split="train")
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
        if i >= limit:
            break
    return docs, load_docs, load_dict


def build_hybrid_retriever(docs, load_docs):
    # generate embedding vectors and save in FAISS vector database
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={"device": "cuda"})
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # hybrid retrieval method (BM25 + dense retrieval)
    BM25 = BM25Retriever.from_texts(load_docs, preprocess_func=lambda text: list(jieba.cut(text)))
    BM25.k = 3
    BGE_vectorstore = vectorstore
    BGE = BGE_vectorstore.as_retriever(search_kwargs={"k": 3})
    hybrid = EnsembleRetriever(retrievers=[BM25, BGE], weights=[0.5, 0.5])
    return hybrid


def evaluate_retriever(hybrid, load_dict):
    def eval_one(pair):
        load_text, question = pair
        docs = hybrid.invoke(question)
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
    return hit_rate, mrr


docs, load_docs, load_dict = load_docs(limit=5000)

hybrid = build_hybrid_retriever(docs, load_docs)

hit_rate, mrr = evaluate_retriever(hybrid, load_dict)

print(f"Hit Rate: {hit_rate:.4f}")
print(f"MRR:     {mrr:.4f}")
