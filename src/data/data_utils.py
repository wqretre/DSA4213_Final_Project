from datasets import load_dataset

from src.data.preprocess import preprocess_example
from src.data.tokenization import tokenize_function
from src.config.config import TRAIN_DATASET_NAME, TEST_DATASET_NAME


# load train and validation set
def load_sft_train_val(tokenizer, train_size=10000, val_size=1000, seed=42):
    ds_train = load_dataset(TRAIN_DATASET_NAME, split="train")
    ds_val = load_dataset(TRAIN_DATASET_NAME, split="validation")

    ds_train = ds_train.map(preprocess_example, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(preprocess_example, remove_columns=ds_val.column_names)

    ds_train = ds_train.filter(lambda x: x is not None)
    ds_val = ds_val.filter(lambda x: x is not None)

    ds_train = ds_train.shuffle(seed=seed).select(range(train_size))
    ds_val = ds_val.shuffle(seed=seed).select(range(val_size))

    ds_train = ds_train.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])
    ds_val = ds_val.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])
    return ds_train, ds_val


def load_sft_test(tokenizer, test_size=400, seed=100):
    ds_test = load_dataset(TEST_DATASET_NAME, split="test")
    ds_test = ds_test.map(preprocess_example, remove_columns=ds_test.column_names)
    ds_test = ds_test.filter(lambda x: x is not None)
    ds_test = ds_test.shuffle(seed=seed).select(range(test_size))
    ds_test = ds_test.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["input", "output"])

    inputs = [x['input_ids'] for x in ds_test]
    references = [x['labels'] for x in ds_test]
    questions = []
    questions_text = []  # questions in the test set
    references_text = []  # answers
    for i in range(len(ds_test)):
        label_ids = [id for id in references[i] if id != -100]
        references_text.append(tokenizer.decode(label_ids, skip_special_tokens=True))
        question = []
        for j, id in enumerate(references[i]):
            if id == -100:
                question.append(inputs[i][j])
        questions.append(question)
        questions_text.append(tokenizer.decode(question, skip_special_tokens=True))
    return ds_test, questions, questions_text, references_text
