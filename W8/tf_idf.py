from collections import defaultdict
from math import log

# Computing term and document frequencies
def compute_statistics(docs):
    doc_count = len(docs)
    term_doc_freq = defaultdict(int)
    term_freq = defaultdict(lambda: defaultdict(int))

    for doc_id, words in docs.items():
        word_set = set(words)
        for word in words:
            term_freq[doc_id][word] += 1
        for word in word_set:
            term_doc_freq[word] += 1

    return term_freq, term_doc_freq, doc_count

# Computing IDF
def compute_idf(term_doc_freq, doc_count, epsilon=0.1):
    idf = {}
    for term, df in term_doc_freq.items():
        idf[term] = log((doc_count - df + 0.5) / (df + 0.5) + 1) + epsilon
    return idf

# Computing TF-IDF vectors for documents
def compute_tf_idf(term_freq, idf):
    tf_idf = defaultdict(dict)
    for doc_id, terms in term_freq.items():
        for term, freq in terms.items():
            tf = freq / len(term_freq[doc_id])  # term frequency
            tf_idf[doc_id][term] = tf * idf.get(term, 0)  # TF-IDF weighting
    return tf_idf

