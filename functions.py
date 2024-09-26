import string
import nltk
import os
import pymorphy2
import time
import numpy as np
from collections import Counter
from lists import stopwords

nltk.download('punkt_tab')

uk_stopwords = set(stopwords)
unique_words = set()


def get_author_texts(author):
    print("READING PLEASE WAIT...")
    texts = []

    workdir = "source"

    for filename in os.listdir(f'{workdir}\\{author}'):
        print("READING PLEASE WAIT...")
        file_path = os.path.join(author, filename)
        with open(f'{workdir}\\{file_path}', 'r', encoding='utf-8') as file:
            text = ''.join(file.readlines())
            texts.append(text.lower())
            print(f"author: {author}, text: {filename} read")

    return texts


def get_bow(words):
    tokens = [word for word in words if word.isalpha() and word not in uk_stopwords]
    bow = Counter(tokens)
    return bow


def get_vectors_from_text(text, skip_normalize=True):
    words, avg_sent_len = process_text(text, skip_normalize)
    dicts = get_frequency_dicts(words)
    a = np.concatenate((get_vector_from_dick(dicts["punctuation"], list(string.punctuation)),
                    get_vector_from_dick(dicts["stopwords"], uk_stopwords),
                    get_vector_from_dick(dicts["bow"], unique_words)))
    return np.append(a, avg_sent_len)


def process_text(text, skip_normalize=True):
    avg_sentence_length = 0

    words = []
    normalized_words = []
    sentences = []
    sentences_length = 0

    print("PROCESSING PLEASE WAIT... 100%")
    sentences.extend(nltk.sent_tokenize(text.lower()))
    words = (nltk.word_tokenize(text.lower()))
    i = 0
    start = time.time()

    for sentence in sentences:
        sentences_length += len(sentence)
    for word in words:
        i += 1
        if i % 1000 == 0:
            end = time.time()
            t = end - start
            speed = 1000 / t
            start = time.time()
            print(f"speed is {speed} symbols per second")

        print(f'.{word}   ;   {i} / {len(words)}')
        if word in list(string.punctuation) or word in stopwords or word in unique_words:
            continue

        normalized_word = word

        if not skip_normalize:
            normalized_word = normalize_word(word)

        unique_words.add(normalized_word)
        normalized_words.append(normalized_word)
    print("another text processed")

    if len(sentences) != 0:
        avg_sentence_length = sentences_length / len(sentences)

    print(f"{len(sentences)} sentences and {len(words)} words after processing another author")

    return words, avg_sentence_length


def get_vector_from_dick(dictionary, base_list):
    return np.array([dictionary.get(word, 0) for word in base_list])


def get_frequency_dicts(words):
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
    print(a, b)
    if len(b) != len(a):
        a = np.pad(a, (0, max(len(a), len(b)) - len(a)), mode='constant')
        b = np.pad(b, (0, max(len(a), len(b)) - len(b)), mode='constant')

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize_word(word):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    p = morph.parse(word)[0].normal_form
    return p


def compare_authors(vectors_a, vectors_b, avg_sentence_len_a, avg_sentence_len_b):
    # Calculate cosine similarity for the 3 feature vectors
    similarities = []

    for key in vectors_a.keys():
        sim = cosine_similarity(vectors_a[key], vectors_b[key])
        similarities.append(sim)

    # Normalize the sentence length difference
    sentence_length_diff = abs(avg_sentence_len_a - avg_sentence_len_b)

    # Combine the similarities (higher is better for cosine) and penalize sentence length differences
    # You could adjust the weights based on how much importance you want to give to sentence length vs vectors
    print(similarities)
    print(sentence_length_diff)
    combined_score = similarities[0] / 1000 + similarities[1] / 10 + similarities[2] - sentence_length_diff / 10000

    return combined_score


def get_author_dataset(author):
    texts = get_author_texts(author)
    batches = []
    for text in texts:
        batch_text = ""
        sentences = nltk.sent_tokenize(text.lower())
        i = 0
        while i < len(sentences):
            batch_text += sentences[i]
            if len(batch_text) < 30000:
                batches.append(batch_text)
                batch_text = ""

            i += 1

    return batches
