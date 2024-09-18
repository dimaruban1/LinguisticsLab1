from functions import *
import numpy as np
from lists import authors

author_values = {}


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


def accept_input():
    path = get_file_path()
    text = ""
    with open(path, 'r', encoding='utf-8') as file:
        text = ''.join(file.readlines())
    text = text.lower()
    words, avg_sentence_length = process_text(text)
    vectors = get_frequency_vectors(author_words)
    return [vectors, avg_sentence_length]


for author in authors:
    print(f"AUTHOR: {author}")
    author_texts = get_author_texts(author)
    author_words, avg_sentence_length = process_text(author_texts)
    vectors = get_frequency_vectors(author_words)
    author_values[author] = [vectors, avg_sentence_length]


def debug():
    for author in authors:
        value = author_values[author]
        vectors = value[0]
        avg_len = value[1]
        print(avg_len)
        print(vectors["punctuation"])
        print(vectors["stopwords"])
        print(len(unique_words))
        print(len(vectors["bow"]))


debug()

