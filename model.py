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
dataset_vectors = []
scaled_features = []


def get_dataset():
    for author in authors:
        texts = get_author_dataset(author)
        author_labels.extend(author * len(texts))

        for text in texts:
            dataset_vectors.append(get_vectors_from_text(text))

    scaled_features.append(scaler.fit_transform(dataset_vectors))
    return scaled_features, author_labels


def save_all():
    for i in range(len(author_labels)):
        np.savetxt(f'vectors\\{author_labels[i]}\\{i}.txt', dataset_vectors[i], fmt='%d')
