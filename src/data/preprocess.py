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
