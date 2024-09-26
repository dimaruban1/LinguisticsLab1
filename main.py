from functions import *
import numpy as np
from lists import authors
import model


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
    get_all_words(authors)

    model.get_dataset()
    model.train()


main()
