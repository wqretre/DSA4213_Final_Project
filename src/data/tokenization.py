from src.config.config import MAX_LEN

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
