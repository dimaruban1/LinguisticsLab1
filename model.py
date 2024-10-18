import numpy as np

import lists
from functions import *
import pymorphy2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import stanza
import nltk

stanza.download('uk')
nlp = stanza.Pipeline(lang='uk', processors='tokenize, mwt, pos, lemma', use_gpu=True)


scaler = StandardScaler()

author_labels = []

dataset_punctuation = []
dataset_stopwords = []
dataset_ngrams = []
dataset_avg_len = []
dataset_POS_ngrams = []

unique_words_list = []
unique_pos_ngrams_list = []

stopwords_set = set(lists.stopwords)
punctuation_set = set(lists.punctuation)
# n stands for n-gram
n = 4
limit = 30000


def filter_words(words):
    filtered_words = []
    for word in words:
        if word.text in stopwords_set or word.text in punctuation_set:
            continue
        else:
            filtered_words.append(word)
    return filtered_words


def get_pos_ngrams(words):
    pos = [w.upos for w in words]
    pos_ngrams = nltk.ngrams(pos, n)
    return [n_gram for n_gram in pos_ngrams]


def get_all_unique_words(texts):
    global unique_words_list
    global unique_pos_ngrams_list

    processed_unique_ngrams = []
    normalized_words = []
    start_time = time.time()
    update_interval = 10
    texts_processed = 0
    total_texts_processed = 0
    # start = time.time()
    # in_docs = [stanza.Document([], text=d) for d in texts]  # Wrap each document with a stanza.Document object
    # out_docs = nlp(in_docs)
    # end = time.time()
    # print(f"spent {(end - start) / 60} minutes doing this stupid shit")

    for text in texts:
        doc = nlp(text)
        sentences = doc.sentences
        for sentence in sentences:
            filtered_words = filter_words(sentence.words)
            normalized_words.extend([w.lemma for w in filtered_words])
            processed_unique_ngrams.extend(get_pos_ngrams(filtered_words))
        # benchmarking
        texts_processed += 1
        if texts_processed > 0 and (time.time() - start_time) > update_interval:
            current_time = time.time() - start_time
            total_texts_processed += texts_processed
            speed = texts_processed / current_time  # Words per second
            time_left = (len(texts) - total_texts_processed) / speed  # Estimated time left in seconds
            print(f"Processed {total_texts_processed}/{len(texts)} texts at {speed:.2f} texts/sec")
            print(f"Estimated time left: {time_left / 60:.2f} minutes")

            # Reset start time for the next interval
            start_time = time.time()
            texts_processed = 0
    bow = Counter(normalized_words)
    pos_dict = Counter(processed_unique_ngrams)

    top_words = sorted(bow.items(), key=lambda x: x[1], reverse=True)[:limit]
    unique_words_list = [word for word, freq in top_words]
    top_pos_ngrams = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
    unique_pos_ngrams_list = [pos_ngram for pos_ngram, freq in top_pos_ngrams]
    return unique_words_list


def process_text(text):
    print("PROCESSING PLEASE WAIT... 100%")

    avg_sentence_length = 0
    sentences_length = 0
    pos_sentence_ngrams_list = []

    doc = nlp(text.lower())
    sentences = doc.sentences
    punctuation_vector = dict.fromkeys(punctuation_set, 0)
    stopwords_vector = dict.fromkeys(uk_stopwords, 0)
    normalized_words = []
    # for sentence in sentences:
    #     filtered_words = filter_words(sentence.words)
    #     normalized_words.extend([w.lemma for w in filtered_words])
    #     pos = [w.upos for w in filtered_words]
    #     pos_ngrams = nltk.ngrams(pos, n)
    #     processed_unique_ngrams.extend([n_gram for n_gram in pos_ngrams])
    for sentence in sentences:
        sentences_length += len(sentence.words)
        words = sentence.words
        filtered_words = []

        for w in words:
            if w.text in uk_stopwords:
                stopwords_vector[w.text] += 1
            elif w.text in punctuation_set:
                punctuation_vector[w.text] += 1
            else:
                filtered_words.append(w)
        normalized_words.extend(w.lemma for w in filtered_words)
        pos_sentence_ngrams_list.extend(get_pos_ngrams(filtered_words))

    if len(sentences) != 0:
        avg_sentence_length = len(normalized_words) / len(sentences)

    print(f"{len(sentences)} sentences and {len(normalized_words)} words after processing another text")
    return normalized_words, pos_sentence_ngrams_list, stopwords_vector, punctuation_vector, avg_sentence_length


def get_vectors_from_text(text, skip_normalize=True):
    def get_vector_from_dick(dictionary, base_list):
        return np.array([dictionary.get(word, 0) for word in base_list])

    normalized_words, pos_sentence_ngrams_list, stopwords_vector, punctuation_vector, avg_sentence_length = process_text(text)
    bow = Counter(normalized_words)
    pos_ngrams = Counter(pos_sentence_ngrams_list)

    punctuation_vector1 = get_vector_from_dick(punctuation_vector, list(string.punctuation))
    stopwords_vector1 = get_vector_from_dick(stopwords_vector, uk_stopwords)
    words_vector = get_vector_from_dick(bow, unique_words_list)
    pos_vector1 = get_vector_from_dick(pos_ngrams, unique_pos_ngrams_list)

    return {"punctuation": punctuation_vector1,
            "stopword": stopwords_vector1,
            "words": words_vector,
            "pos": pos_vector1,
            "avg":  [avg_sentence_length],
            }


def expand_dataset_with_vectors(vectors):
    dataset_punctuation.append(vectors["punctuation"])
    dataset_stopwords.append(vectors["stopword"])
    dataset_ngrams.append(vectors["words"])
    dataset_POS_ngrams.append(vectors["pos"])
    dataset_avg_len.append(vectors["avg"])


def get_rf_classifier(data):
    x_train, x_test, y_train, y_test = train_test_split(data, author_labels, test_size=0.4, random_state=42)

    rf_feature = RandomForestClassifier(n_estimators=20)
    rf_feature.fit(x_train, y_train)

    return rf_feature, [x_test, y_test]


def train():
    rf_punctuation, data1 = get_rf_classifier(dataset_punctuation)
    rf_stopwords, data2 = get_rf_classifier(dataset_stopwords)
    rf_ngrams, data3 = get_rf_classifier(dataset_ngrams)
    rf_pos, data4 = get_rf_classifier(dataset_POS_ngrams)
    rf_avg_len, data5 = get_rf_classifier(dataset_avg_len)
    return [[rf_punctuation, data1],
            [rf_stopwords, data2],
            [rf_ngrams, data3],
            [rf_pos, data4],
            [rf_avg_len, data5]]


def predict():
    rf_models = train()

    models = {
        "punctuation": rf_models[0][0],
        "stopwords": rf_models[1][0],
        "ngrams": rf_models[2][0],
        "pos": rf_models[3][0],
        "avg_len": rf_models[4][0],
    }
    y_test = rf_models[0][1][1]

    predictions = []
    features = ["punctuation", "stopwords", "ngrams", "pos", "average sentence length"]
    for idx, rf_model in enumerate(models):
        y_pred_prob = models[rf_model].predict_proba(rf_models[idx][1][0])
        predictions.append(y_pred_prob)
    prediction_matrix = np.hstack(predictions)
    meta_classifier = RandomForestClassifier(n_estimators=20)
    meta_classifier.fit(prediction_matrix, y_test)
    final_predictions = meta_classifier.predict(prediction_matrix)
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Final Accuracy: {accuracy:.2f}")

    return final_predictions, y_test


def normalize_word(word):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    p = morph.parse(word)[0].normal_form
    return p


def process_sentence_words(sentence_words):
    normalized_words = []
    pos_list = []
    morph = pymorphy2.MorphAnalyzer(lang='uk')

    for i, word in enumerate(sentence_words):
        parsed_word = morph.parse(word)[0]
        normalized_words.append(parsed_word.normal_form)
        if parsed_word.tag.POS is None:
            continue
        pos_list.append(parsed_word.tag.POS)
    return normalized_words, pos_list
