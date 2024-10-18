import random

from functions import *
import numpy as np
from lists import authors
import model
from sklearn.metrics import accuracy_score, classification_report


def get_batches_txt():
    for author in authors:
        batches, length = get_author_batches(author)
        for i, batch in enumerate(batches):
            file_path = f"batches\\{author}\\{i}.txt"
            with open(file_path, 'w', encoding="utf-8") as f:
                for line in batch:
                    f.write(f"{line}\n")


def get_file_path():
    default_filename = "input.txt"

    while True:
        file_path = input("Please enter the file path or press 'F' to skip: ")

        if file_path.lower() == 'f':
            print("File input skipped.")
            return default_filename

        try:
            with open(file_path, 'r'):
                print(f"File '{file_path}' found.")
                return file_path
        except FileNotFoundError:
            print("File not found. Please try again.")


def taranukha_have_mercy_on_my_soul(authors):
    texts = []
    for author in authors:
        batches = get_author_batches(author)
        random.shuffle(batches)
        texts.extend(batches[:100])
    model.get_all_unique_words(texts)

    for i, author in enumerate(authors):
        batches = get_author_batches(author)
        random.shuffle(batches)
        for text in batches[:200]:
            vectors = model.get_vectors_from_text(text)
            model.expand_dataset_with_vectors(vectors)
            model.author_labels.append(author)
        print(f"{i+1}/{len(authors)} authors processed")
    y_pred, y_test = model.predict()
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))


def main():
    taranukha_have_mercy_on_my_soul(authors)
    # batches = get_author_batches("vyshnia")
    # ls = model.get_all_unique_words(batches[:2])
    # print(ls)
    # print(len(ls))
    #
    # for text in batches:
    #     model.process_text(text)


main()
