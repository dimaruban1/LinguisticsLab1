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
        texts, sent_counts = get_author_dataset(author)
        author_labels.extend([author] * len(texts))

        for i, text in enumerate(texts):
            dataset_vectors.append(np.append(get_vectors_from_text(text), sent_counts[i]))

    # scaled_features.append(scaler.fit_transform(dataset_vectors))
    return scaled_features, author_labels


def train():
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset_vectors, author_labels, test_size=0.3, random_state=42)

    # Initialize the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    print(classification_report(y_test, y_pred))


def save_all():
    for i in range(len(author_labels)):
        np.savetxt(f'vectors\\{author_labels[i]}\\{i}.txt', dataset_vectors[i], fmt='%d')
        print(f'vectors\\{author_labels[i]}\\{i}.txt')
