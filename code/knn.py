import glob
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from sklearn.model_selection import train_test_split
from collections import Counter


def load_scripts(path):
    scripts = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            scripts.append(content)
    return scripts

def preprocessing_data(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def get_n_grams_frequencies(texts, categories, n):
    total_grams = 0
    n_gram_category = {}
    n_gram_counter = {}
    for text, category in zip(texts, categories):
        total_line_length = len(text)
        line_length = total_line_length - n + 1
        for line in range(line_length):
            n_gram_word = text[line:n + line]
            total_grams += 1
            if n_gram_word+"@"+category in n_gram_category:
                n_gram_category[n_gram_word+"@"+category] += 1
            else:
                n_gram_category[n_gram_word+"@"+category] = 1
            if n_gram_word in n_gram_counter:
                n_gram_counter[n_gram_word] += 1
            else:
                n_gram_counter[n_gram_word] = 1

    counter = Counter(n_gram_counter)
    keys_sorted_by_value = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)

    unique_categories = categories.unique()
    frequency_dict = {}
    for n_gram_sorted in keys_sorted_by_value:
        for category in unique_categories:
            key = n_gram_sorted + "@" + category
            if key in n_gram_category.keys():
                frequency_dict[key] = n_gram_category[key] / total_grams
                # print(frequency_dict[key])
            else:
                frequency_dict[key] = 0

    sorted_frequency = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    # for idx, (word, freq) in enumerate(sorted_frequency[:100], start=1):
    #     print(f"{idx}. {word}: {freq}")
    counter = 0
    zero_counter = 0
    non_zero_counter = 0
    for key, value in sorted_frequency:
        # print(value)
        if value == 0:
            zero_counter += 1
        else:
            non_zero_counter += 1

    print("Non zero: " + str(non_zero_counter))
    print("Zero counter: " + str(zero_counter))

    return sorted_frequency




if __name__ == "__main__":
    # data = load_scripts("scripts")
    # # print(preprocessing_data("L.R. Brane loves his life - his car, his apartment, his job, but especially his girlfriend, Vespa. One day while showering, Vespa runs out of shampoo. L.R. runs across the street to a convenience store to buy some more, a quick trip of no more than a few minutes. When he returns, Vespa is gone and every trace of her existence has been wiped out. L.R.'s life becomes a tortured existence as one strange event after another occurs to confirm in his mind that a conspiracy is working against his finding Vespa."))
    # # print("")
    # directory = 'scripts'
    # df = process_all_scripts(directory)
    #
    # # Prikaz prvih nekoliko redova DataFrame-a
    # print(df)

    # Učitajte CSV fajl
    df = pd.read_csv('new/ecommerceDataset.csv')
    df.columns = ['category', 'text']
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(preprocessing_data)

    X = df['text']
    # print(df['text'])
    # print("--------")
    # print(df[['text']])
    # input()
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    frequencies = get_n_grams_frequencies(X_train, y_train, 3)
    print(frequencies)

    # knn = KNNClassifier(k=4000)
    # knn.fit(frequencies, y_train)

    # x_test = test_data_preprocessing(X_test, 3)
    # print(x_test)
    # y_pred = knn.predict(x_test)
    # accuracy = (y_pred == y_test_tensor).sum().item() / len(y_test_tensor)
    # print("Tačnost:", accuracy)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy * 100)
    # print(datetime.datetime.now())


