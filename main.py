from functions import *
import numpy as np
from lists import authors
import model
from sklearn.metrics import accuracy_score


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


def main():
    top_ngrams = model.get_all_ngrams(authors)
    model.unique_ngrams = top_ngrams

    model.get_dataset()
    rf_models = model.train()

    for rf_model in rf_models:
        y_pred = rf_model[0].predict(rf_model[1][0])
        # Evaluate the model's performance
        accuracy = accuracy_score(rf_model[1][1], y_pred)
        print(f"Accuracy: {accuracy}")


main()
