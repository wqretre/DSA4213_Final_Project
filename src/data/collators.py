import torch


# pad inputs_id and lebels to the same length
def collate_fn_sft(batch, pad_id):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}


def collate_fn_leftpad(batch, pad_id):
    batch = [torch.tensor(x, dtype=torch.long) for x in batch]
    max_len = max(len(x) for x in batch)
    # left padding
    padded = torch.stack([
        torch.cat([torch.full((max_len - len(x),), pad_id, dtype=torch.long), x])
        for x in batch
    ])
    attention_mask = (padded != pad_id).long()
    return padded, attention_mask
