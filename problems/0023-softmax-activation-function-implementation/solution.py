import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    mx = max(scores)
    scores_exp = [math.exp(score - mx) for score in scores]
    s = sum(scores_exp)
    return [score / s for score in scores_exp]