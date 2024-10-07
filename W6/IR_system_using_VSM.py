from flask import Flask, request, jsonify
import os
import re
import numpy as np
from collections import defaultdict
from math import log, sqrt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Preprocessing and cleaning text
def text_cleaner(text, stem='Stem'):
    text = text.lower()  # Converting to lowercase
    text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)  # Removing URLs
    text = re.sub(r"[^\w\s]", '', text)  # Removing non-word, non-whitespace characters
    text = re.sub(r"[\d]", '', text)  # Removing numbers
    cleaned_text = text.split()  # Tokenizing the text
    
    # Removing stop words
    useless_words = stopwords.words("english")
    final_text = [word for word in cleaned_text if word not in useless_words]

    # Applying stemming
    if stem == 'Stem':
        stemmer = PorterStemmer()
        final_text = [stemmer.stem(word) for word in final_text]

    return final_text

# Loading the documents
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs[filename] = text_cleaner(file.read())
    return docs

# Loading the queries
def load_queries(query_file_path):
    with open(query_file_path, 'r') as file:
        return [text_cleaner(line.strip()) for line in file.readlines()]

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

# Computing TF-IDF vectors for documents
def compute_tf_idf(term_freq, idf):
    tf_idf = defaultdict(dict)
    for doc_id, terms in term_freq.items():
        for term, freq in terms.items():
            tf = freq / len(term_freq[doc_id])  # term frequency
            tf_idf[doc_id][term] = tf * idf.get(term, 0)  # TF-IDF weighting
    return tf_idf

# Computing IDF separately to avoid 0 output
def compute_idf(term_doc_freq, doc_count, epsilon=0.1):
    idf = {}
    for term, df in term_doc_freq.items():
        idf[term] = log((doc_count - df + 0.5) / (df + 0.5) + 1) + epsilon
    return idf

# Converting text to TF-IDF vector
def query_to_vector(query, idf):
    tf = defaultdict(int)
    for term in query:
        tf[term] += 1
    
    tf_idf = {term: (tf[term] / len(query)) * idf.get(term, 0) for term in query}
    return tf_idf

# Cosine similarity computation
#def cosine_similarity(vec1, vec2):
#    intersection = set(vec1.keys()) & set(vec2.keys())
#    numerator = sum([vec1[x] * vec2[x] for x in intersection])
#
#    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
#    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
#    denominator = sqrt(sum1) * sqrt(sum2)
#
#    if not denominator:
#        return 0.0
#    else:
#        return numerator / denominator

# Retrieving and ranking documents using cosine similarity for given queries
def retrieve_documents_cosine_similarity(folder_path, query_file_path):
    docs = load_documents(folder_path)
    queries = load_queries(query_file_path)

    term_freq, term_doc_freq, doc_count = compute_statistics(docs)
    tf_idf_docs, idf = compute_tf_idf(term_freq, term_doc_freq, doc_count)

    for query in queries:
        query_vector = query_to_vector(query, idf)
        scores = {doc_id: cosine_similarity(query_vector, tf_idf_docs[doc_id]) for doc_id in tf_idf_docs}

        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
       
# Computing BM25 scores
#def compute_bm25(query, term_freq, idf, doc_count, avgdl, doc_lengths, k1=1.5, b=0.75):
#    scores = defaultdict(float)
#    
#    for term in query:
#        if term in idf:
#            for doc_id, tf in term_freq.items():
#                freq = tf.get(term, 0)
#                numerator = freq * (k1 + 1)
#                denominator = freq + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl)
#                scores[doc_id] += idf[term] * numerator / denominator
#    
#    return scores

# Adjust this accordingly
folder_path = './Dataset' 
docs = load_documents(folder_path)
term_freq, term_doc_freq, doc_count = compute_statistics(docs)
doc_lengths = {doc_id: len(words) for doc_id, words in docs.items()}
avgdl = np.mean(list(doc_lengths.values()))
idf = compute_idf(term_doc_freq, doc_count)


# Cosine similarity computation
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)
    
    return numerator / denominator if denominator else 0.0

# BM25 computation
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

# Compute the TF-IDF vectors for the documents
tf_idf_docs = compute_tf_idf(term_freq, idf)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Preprocess the query
    cleaned_query = text_cleaner(query)
    
    # Compute BM25 and Cosine Similarity scores
    bm25_scores = compute_bm25(cleaned_query, term_freq, idf, doc_count, avgdl, doc_lengths)
    
    query_vector = query_to_vector(cleaned_query, idf)
    cosine_scores = {doc_id: cosine_similarity(query_vector, tf_idf_docs[doc_id]) for doc_id in tf_idf_docs}
    
    # Combine scores with a weight
    alpha = 0.7  # Adjust weight for BM25 (70% BM25, 30% cosine similarity)
    final_scores = {doc_id: alpha * bm25_scores[doc_id] + (1 - alpha) * cosine_scores[doc_id] 
                    for doc_id in bm25_scores}
    
    # Rank documents by combined score
    ranked_docs = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Format the results for the response
    results = [
        {"document": doc_id, "score": round(score, 4)}
        for doc_id, score in ranked_docs if score > 0  # Filtering out zero scores
    ]

    return jsonify({"query": query, "results": results})

if __name__ == '__main__':
    app.run(debug=True)

