import os
import re
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

