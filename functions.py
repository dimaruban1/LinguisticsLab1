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


def get_all_words(authors):
    for author in authors:
        texts = get_author_texts(author)
        for text in texts:
            words = (nltk.word_tokenize(text.lower()))
            for word in words:
                unique_words.add(word)


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
    return a


def process_text(text, skip_normalize=True):
    avg_sentence_length = 0

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


def normalize_word(word):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    p = morph.parse(word)[0].normal_form
    return p


def get_author_dataset(author):
    texts = get_author_texts(author)
    batches = []
    sentences_count = []

    for text in texts:
        batch_text = ""
        sentences = nltk.sent_tokenize(text.lower())
        i = 0
        j = 0
        while i < len(sentences):
            batch_text += sentences[i]
            j += 1
            if len(batch_text) > 2000:
                batches.append(batch_text)
                batch_text = ""
                sentences_count.append(30000 / j)
                j = 0

            i += 1

    return batches, sentences_count
