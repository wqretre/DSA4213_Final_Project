import os
import numpy as np
import evaluate


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
