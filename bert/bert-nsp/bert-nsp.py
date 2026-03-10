import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # YOUR CODE HERE
    examples = []
    num_positive = num_examples // 2
    num_negative = num_examples - num_positive
    # Flatten all sentences for fallback negative sampling
    all_sentences = [(doc_id, sent) for doc_id, doc in enumerate(documents) for sent in doc]

    # positive examples
    valid_docs = [doc for doc in documents if len(doc) >= 2]
    for _ in range(num_positive):
        doc = random.choice(valid_docs)
        i = random.randint(0, len(doc) - 2)
        sent_a, sent_b = doc[i], doc[i + 1]
        examples.append((sent_a, sent_b, 1))
    
    # negative examples
    for _ in range(num_negative):
        # get sent_a
        doc_a_idx = random.randrange(len(documents))
        doc_a = documents[doc_a_idx]
        sent_a = random.choice(doc_a)
        # get sent_b
        other_doc_idx = [j for j in range(len(documents)) if (j != doc_a_idx) and (len(documents[j]) > 0)]
        if len(other_doc_idx) > 0:
            doc_b_idx = random.choice(other_doc_idx)
            sent_b = random.choice(documents[doc_b_idx])
        else:
            candidate_pool = [sent for _, sent in all_sentences if sent != sent_a]
            if len(candidate_pool) == 0:
                candidate_pool = [sent for _, sent in all_sentences]
            sent_b = random.choice(candidate_pool)
        examples.append((sent_a, sent_b, 0))

    random.shuffle(examples)
    return examples

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext probability.
        """
        # YOUR CODE HERE
        return cls_hidden @ self.W + self.b

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
