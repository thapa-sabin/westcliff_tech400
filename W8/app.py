from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import numpy as np
from collections import defaultdict
from math import log, sqrt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)

# Sample ground truth data for MAP calculation
ground_truth = {
    "query1": ["Grimms' Fairy Tales.txt", "Middlemarch.txt"],
    "query2": ["The Prince.txt", "The Iliad.txt"],
    "query3": ["Romeo and Juliet.txt", "The Count of Monte Cristo.txt"],
}

# Document ID mapping for images
doc_id_mapping = {
    "The Prince.txt": "https://www.gutenberg.org/cache/epub/1232/pg1232.cover.medium.jpg",
    "Ulysses.txt": "https://www.gutenberg.org/cache/epub/4300/pg4300.cover.medium.jpg",
    "The Count of Monte Cristo.txt": "https://www.gutenberg.org/cache/epub/1184/pg1184.cover.medium.jpg",
    "The Iliad.txt": "https://www.gutenberg.org/cache/epub/6130/pg6130.cover.medium.jpg",
    "The Adventures of Tom Sawyer.txt": "https://www.gutenberg.org/cache/epub/74/pg74.cover.medium.jpg",
    "Second Treatise of Government.txt": "https://www.gutenberg.org/cache/epub/7370/pg7370.cover.medium.jpg",
    "Romeo and Juliet.txt": "https://www.gutenberg.org/cache/epub/1513/pg1513.cover.medium.jpg",
    "Middlemarch.txt": "https://www.gutenberg.org/cache/epub/145/pg145.cover.medium.jpg",
    "Grimms' Fairy Tales.txt": "https://www.gutenberg.org/cache/epub/2591/pg2591.cover.medium.jpg",
    "Frankenstein: Or, The Modern Prometheus.txt": "https://www.gutenberg.org/cache/epub/42324/pg42324.cover.medium.jpg",
}

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

# Loading documents
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs[filename] = text_cleaner(file.read())
    return docs

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

# Computing IDF
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
folder_path = './Dataset' 
docs = load_documents(folder_path)
term_freq, term_doc_freq, doc_count = compute_statistics(docs)
doc_lengths = {doc_id: len(words) for doc_id, words in docs.items()}
avgdl = np.mean(list(doc_lengths.values()))
idf = compute_idf(term_doc_freq, doc_count)
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
        {
            "document": doc_id, 
            "score": round(score, 4),
            "snippet": extract_snippet(doc_id, cleaned_query),  # Get snippet
            "image_url": doc_id_mapping.get(doc_id, None)  # Get image URL
        }
        for doc_id, score in ranked_docs if score > 0  # Filtering out zero scores
    ]

    return jsonify({"query": query, "results": results})

def extract_snippet(doc_id, query):
    # Load the document and extract a relevant snippet
    with open(os.path.join(folder_path, doc_id), 'r') as file:
        content = file.read()
        for line in content.split('\n'):
            if any(term in line.lower() for term in query):
                return line.strip()  # Return the first line containing a query term
    return "No relevant snippet found."

if __name__ == '__main__':
    app.run(debug=True)

