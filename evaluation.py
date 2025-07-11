from nltk.tokenize import word_tokenize

def exact_match(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())

def f1_score(pred, ref):
    pred_tokens = set(word_tokenize(pred.lower()))
    ref_tokens = set(word_tokenize(ref.lower()))
    overlap = pred_tokens & ref_tokens
    if not overlap:
        return 0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)