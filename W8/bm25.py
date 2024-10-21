from collections import defaultdict
from math import log

def compute_bm25(query, term_freq, idf, doc_count, avgdl, doc_lengths, k1=1.5, b=0.75):
    scores = defaultdict(float)
    
    for term in query:
        if term in idf:
            for doc_id, tf in term_freq.items():
                freq = tf.get(term, 0)
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl)
                scores[doc_id] += idf[term] * numerator / denominator
    
    return scores

