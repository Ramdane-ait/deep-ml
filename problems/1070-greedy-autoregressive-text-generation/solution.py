import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    # Your code here
    for _ in range(max_new_tokens):
        context = idx[-context_size:]

        logits = model(np.array(context).reshape((1,-1)))
        token_pred = np.argmax(logits[:,-1,:])
        idx.append(int(token_pred))
    
    return idx
