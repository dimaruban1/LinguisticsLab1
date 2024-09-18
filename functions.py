import string
import nltk
import os
import numpy as np
from collections import Counter
from lists import stopwords

nltk.download('punkt_tab')

uk_stopwords = set(stopwords)
unique_words = set()


def get_author_texts(author):
    print("READING PLEASE WAIT...")
    texts = []
    for filename in os.listdir('source\\' + author):
        print("READING PLEASE WAIT...")
        file_path = os.path.join(author, filename)
        with open('source\\' + file_path, 'r', encoding='utf-8') as file:
            text = ''.join(file.readlines())
            texts.append(text.lower())
            print(f"author: {author}, text: {filename} read")
    return texts


def get_bow(words):
    tokens = [word for word in words if word.isalpha() and word not in uk_stopwords]
    bow = Counter(tokens)
    return bow


def process_text(texts):
    print("PROCESSING PLEASE WAIT... 100%")
    avg_sentence_length = 0

    words = []
    sentences = []
    sentences_length = 0
    for text in texts:
        print("PROCESSING PLEASE WAIT... 100%")
        sentences.extend(nltk.sent_tokenize(text.lower()))
        words.extend(nltk.word_tokenize(text.lower()))

        for sentence in sentences:
            sentences_length += len(sentence)
        for word in words:
            unique_words.add(word)
        print("another text processed")

    if len(sentences) != 0:
        avg_sentence_length = sentences_length / len(sentences)

    print(f"{len(sentences)} sentences and {len(words)} words after processing another author")

    return words, avg_sentence_length


def get_vector_from_dick(dictionary, base_list):
    return np.array([dictionary.get(word, 0) for word in base_list])


def get_frequency_vectors(words):
    punctuation_vectors = dict.fromkeys(list(string.punctuation), 0)
    stopwords_vectors = dict.fromkeys(uk_stopwords, 0)
    punctuation_list = list(string.punctuation)
    for word in words:
        if word in punctuation_list:
            punctuation_vectors[word] += 1
        elif word in uk_stopwords:
            stopwords_vectors[word] += 1

    bow = get_bow(words)
    vectors = {
        "punctuation": punctuation_vectors,
        "stopwords": stopwords_vectors,
        "bow": bow,
    }

    return vectors


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_authors(vectors_a, vectors_b, avg_sentence_len_a, avg_sentence_len_b):
    # Calculate cosine similarity for the 3 feature vectors
    similarities = []
    for i in range(len(vectors_a)):
        sim = cosine_similarity(vectors_a[i].reshape(1, -1), vectors_b[i].reshape(1, -1))[0][0]
        similarities.append(sim)

    # Normalize the sentence length difference
    sentence_length_diff = abs(avg_sentence_len_a - avg_sentence_len_b)

    # Combine the similarities (higher is better for cosine) and penalize sentence length differences
    # You could adjust the weights based on how much importance you want to give to sentence length vs vectors
    combined_score = np.mean(similarities) - sentence_length_diff

    return combined_score
