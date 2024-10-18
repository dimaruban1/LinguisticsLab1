import string
import os
import pymorphy2
import time
import numpy as np
from collections import Counter
from lists import stopwords
import nltk

uk_stopwords = set(stopwords)


def get_all_pos_sentences(authors):
    for author in authors:
        texts = get_author_texts(author)
        for text in texts:
            nltk.sent_tokenize(text, )


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


BATCH_SIZE = 500


def get_author_batches(author):
    texts = get_author_texts(author)
    batches = []

    for text in texts:
        i = 0
        while i < len(text):
            batch_text = text[i:i + BATCH_SIZE]  # Get the first 5000 characters
            i += BATCH_SIZE

            # Continue until we find punctuation or reach the end of the string
            while i < len(text) and text[i] not in string.punctuation:
                batch_text += text[i]
                i += 1

            # Add the last punctuation if it exists
            if i < len(text):
                batch_text += text[i]
                i += 1

            batches.append(batch_text)

    return batches
