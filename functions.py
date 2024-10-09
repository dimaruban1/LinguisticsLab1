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


def get_author_batches(author):
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
