import numpy as np
from lists import authors
from functions import *
import pymorphy2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

scaler = StandardScaler()

author_labels = []

dataset_punctuation = []
dataset_stopwords = []
dataset_ngrams = []
dataset_avg_len = []

scaled_features = []
unique_ngrams = {}
top_ngrams = {}

# n stands for n-gram
n = 1
limit = 30000


def get_all_ngrams(authors, skip_normalize=True):
    for author in authors:
        texts = get_author_texts(author)
        for text in texts:
            words = nltk.word_tokenize(text.lower())
            ngrams = nltk.ngrams(words, n)
            for ngram in ngrams:
                if ngram not in unique_ngrams:
                    unique_ngrams[ngram] = 0
                unique_ngrams[ngram] += 1

    top_ngrams = sorted(unique_ngrams.items(), key=lambda x: x[1], reverse=True)[:limit]
    return top_ngrams


def process_text(text, skip_normalize=True):
    avg_sentence_length = 0
    sentences = []
    sentences_length = 0

    print("PROCESSING PLEASE WAIT... 100%")
    sentences.extend(nltk.sent_tokenize(text.lower()))
    words = (nltk.word_tokenize(text.lower()))

    for sentence in sentences:
        sentences_length += len(sentence)

    ngrams = nltk.ngrams(words, n)

    if len(sentences) != 0:
        avg_sentence_length = sentences_length / len(sentences)

    print(f"{len(sentences)} sentences and {len(words)} words after processing another author")

    return ngrams, avg_sentence_length


def get_vectors_from_text(text, skip_normalize=True):
    def get_vector_from_dick(dictionary, base_list):
        return np.array([dictionary.get(word, 0) for word in base_list])

    punctuation_vectors = dict.fromkeys(list(string.punctuation), 0)
    stopwords_vectors = dict.fromkeys(uk_stopwords, 0)

    ngrams, avg_sent_len = process_text(text, skip_normalize)
    words = nltk.word_tokenize(text.lower())

    for word in words:
        if word in string.punctuation:
            punctuation_vectors[word] += 1
        elif word in uk_stopwords:
            stopwords_vectors[word] += 1

    bow = Counter(ngrams)

    punctuation_vector = get_vector_from_dick(punctuation_vectors, list(string.punctuation))
    stopwords_vector = get_vector_from_dick(stopwords_vectors, uk_stopwords)
    ngrams = get_vector_from_dick(bow, unique_ngrams)

    return {"punctuation": punctuation_vector, "stopword": stopwords_vector, "ngrams": ngrams, "avg":  [avg_sent_len]}


def get_dataset():
    for author in authors:
        texts, sent_counts = get_author_batches(author)
        author_labels.extend([author] * len(texts))

        for i, text in enumerate(texts):
            vectors = get_vectors_from_text(text)
            dataset_punctuation.append(vectors["punctuation"])
            dataset_stopwords.append(vectors["stopword"])
            dataset_ngrams.append(vectors["ngrams"])
            dataset_avg_len.append(vectors["avg"])
    # scaled_features.append(scaler.fit_transform(dataset_vectors))


def get_author_dataset(author):
    texts, sent_counts = get_author_batches(author)
    author_labels.extend([author] * len(texts))

    for i, text in enumerate(texts):
        vectors = get_vectors_from_text(text)
        dataset_punctuation.append(vectors["punctuation"])
        dataset_stopwords.append(vectors["stopword"])
        dataset_ngrams.append(vectors["ngrams"])
        dataset_avg_len.append(vectors["avg"])
    # scaled_features.append(scaler.fit_transform(dataset_vectors))


def get_rf_classifier(data):
    x_train, x_test, y_train, y_test = train_test_split(data, author_labels, test_size=0.4, random_state=42)

    rf_feature = RandomForestClassifier()
    rf_feature.fit(x_train, y_train)

    return rf_feature, [x_test, y_test]


def train():
    rf_punctuation, data1 = get_rf_classifier(dataset_punctuation)
    rf_stopwords, data2 = get_rf_classifier(dataset_stopwords)
    rf_ngrams, data3 = get_rf_classifier(dataset_ngrams)
    rf_avg_len, data4 = get_rf_classifier(dataset_avg_len)
    return [[rf_punctuation, data1],
            [rf_stopwords, data2],
            [rf_ngrams, data3],
            [rf_avg_len, data4]]


def normalize_word(word):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    p = morph.parse(word)[0].normal_form
    return p

