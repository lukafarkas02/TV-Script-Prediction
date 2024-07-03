import glob
import os
import nltk
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import string
import re
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import datetime
import math
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

stopword_list = stopwords.words("english")


class KNNClassifier():
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def calculate_distance(self, first_k_frequency, X_test):
        distance = defaultdict(float)
        test_dict = {key: val for key, val in X_test}
        for word, value in first_k_frequency:
            n_gram, category = word.split("@")
            frequency_test = test_dict.get(n_gram, 0)
            distance[category] += (value - frequency_test) ** 2
        for category in distance:
            distance[category] = math.sqrt(distance[category])
        return distance

    def predict(self, X):
        Y_pred = [None] * len(X)
        first_k_frequency = self.X[:self.k]

        for index, X_test in enumerate(X):
            distance = self.calculate_distance(first_k_frequency, X_test)
            Y_pred[index] = min(distance, key=distance.get)

        return Y_pred


def pretprocessing(X_test, n):
    all_instances = []
    for text in X_test:
        all_grams = {}
        frequency = {}
        total_line_length = len(text)
        line_length = total_line_length - n + 1
        for line in range(line_length):
            n_gram_word = text[line:n + line]
            if n_gram_word in all_grams.keys():
                all_grams[n_gram_word] += 1
            else:
                all_grams[n_gram_word] = 1
        for key, value in all_grams.items():
            frequency[key] = value / line_length

        all_instances.append(list(frequency.items()))
    return all_instances


def get_n_grams_frequencies(texts, categories, n):
    n_gram_counter = {}
    n_gram_category = {}
    total_grams = 0
    for text, category in zip(texts, categories):
        total_line_length = len(text)
        line_length = total_line_length - n + 1
        # total_grams += line_length
        for line in range(line_length):
            n_gram_word = text[line:n + line]
            total_grams += 1
            if n_gram_word + "@" + str(category) in n_gram_category:
                n_gram_category[n_gram_word + "@" + str(category)] += 1
            else:
                n_gram_category[n_gram_word + "@" + str(category)] = 1
            if n_gram_word in n_gram_counter:
                n_gram_counter[n_gram_word] += 1
            else:
                n_gram_counter[n_gram_word] = 1

    counter = Counter(n_gram_counter)
    keys_sorted_by_value = sorted(n_gram_counter.items(), key=lambda item: item[1], reverse=True)
    counter = 0
    zero_counter = 0
    non_zero_counter = 0
    for key, value in keys_sorted_by_value:
        # print(value)
        if value == 0:
            zero_counter += 1
        else:
            non_zero_counter += 1

    keys_sorted_by_value = [key for key, value in keys_sorted_by_value]

    print("Non zero: " + str(non_zero_counter))
    print("Zero counter: " + str(zero_counter))
    unique_categories = categories.unique()
    frequency_dict = {}
    for n_gram_sorted in keys_sorted_by_value:
        for category in unique_categories:
            key = n_gram_sorted + "@" + str(category)
            if key in n_gram_category.keys():
                frequency_dict[key] = n_gram_category[n_gram_sorted + "@" + str(category)] / total_grams
                # print(frequency_dict[key])
            else:
                frequency_dict[key] = 0
    return list(frequency_dict.items())


def parse_script(script):
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'Written by:.*?\n', '', script)
    script = re.sub(r'Opening Credits', '', script)
    lines = re.findall(r'([A-Z][a-zA-Z]+): (.*?)\n', script)
    df = pd.DataFrame(lines, columns=['role', 'text'])
    df['role'] = df['role'].str.title()
    return df


def load_scripts(path):
    scripts = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            scripts.append(content)
    return scripts


def clean_text(data):
    tokens = word_tokenize(data)
    clean_data = [word.lower() for word in tokens if
                  (word.lower() not in string.punctuation) and (word.lower() not in stopword_list) and (
                              len(word) > 2) and (word.isalpha())]
    return clean_data


def vectorizer(final_text, model):
    features = []
    for review in final_text:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for word in review:
            try:
                if word in model.wv:
                    vectors.append(model.wv[word])
            except KeyError:
                continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


if __name__ == "__main__":
    print(datetime.datetime.now())

    # pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_csv('new/ecommerceDataset.csv')
    df.columns = ['category', 'text']
    # df = df.copy(deep=True)
    df['text'] = df['text'].astype(str)

    # print(df['text'])

    df['cleaned_text'] = df['text'].apply(clean_text)

    model = Word2Vec(sentences=df['cleaned_text'], vector_size=100, window=5, min_count=1, workers=4)

    df['vectorized_text'] = vectorizer(df['cleaned_text'], model)
    # df = df.copy(deep=True)
    df['category'] = df['category'].replace({"Household": 0, "Books": 1, "Electronics": 2, "Clothing & Accessories": 3})

    x_train, x_test, y_train, y_test = train_test_split(df['vectorized_text'].tolist(), df['category'].values,
                                                        test_size=0.2)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")
    # print("Model evaluation on Test data")
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print()
    # print("Classification Report:\n", classification_report(y_test, y_pred))
    # print()
    acc_test = accuracy_score(y_test, y_pred) * 100
    print('Accuracy for KNN Model on Test Data is:', acc_test)
    print("********************************************************")

    episodes = load_scripts("scripts")
    print(len(episodes))
    print(datetime.datetime.now())
    dfs = [parse_script(script) for script in episodes]
    all_dialogues_df = pd.concat(dfs, ignore_index=True)
    count_role = {}
    for index, row in all_dialogues_df.iterrows():
        text = row['text']
        role = row['role']
        if role in count_role:
            count_role[role] += 1
        else:
            count_role[role] = 1

    dict_data = Counter(count_role)
    # print(dict_data)
    top_6 = Counter(dict_data).most_common(6)
    top_6_roles = [word for word, count in top_6]

    filtered_df = all_dialogues_df[all_dialogues_df['role'].isin(top_6_roles)]
    filtered_df['text'] = filtered_df['text'].astype(str)
    filtered_df['cleaned_text'] = filtered_df['text'].apply(clean_text)

    model = Word2Vec(sentences=filtered_df['cleaned_text'], vector_size=50, window=2, min_count=2, workers=4)
    print(model.vector_size)

    filtered_df['vectorized_text'] = vectorizer(filtered_df['cleaned_text'], model)

    filtered_df['role'] = filtered_df['role'].replace({"Joey": 0, "Monica": 1, "Rachel": 2, "Chandler": 3,
                                                       "Ross": 4, "Phoebe": 5})

    # print(filtered_df['role'].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(filtered_df['vectorized_text'].tolist(),
                                                        filtered_df['role'].values,
                                                        test_size=0.2)

    classifier = KNeighborsClassifier(n_neighbors=13)
    classifier.fit(x_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(x_test)

    acc_test = accuracy_score(y_test, y_pred) * 100
    print('Accuracy for KNN Model on Test Data is:', acc_test)
    print("********************************************************")

    #     MY CUSTOM KNN WITH N_GRAMS
    dfNew = pd.read_csv('new/ecommerceDataset.csv')
    dfNew.columns = ['category', 'text']
    dfNew['text'] = dfNew['text'].astype(str)

    # dfNew['cleaned_text'] = dfNew['text'].apply(clean_text)
    # print(dfNew['cleaned_text'])

    # dfNew['category'] = dfNew['category'].replace({"Household": 0, "Books": 1, "Electronics": 2, "Clothing & Accessories": 3})

    x_train, x_test, y_train, y_test = train_test_split(dfNew['text'], dfNew['category'],
                                                        test_size=0.2, random_state=42)

    frequencies = get_n_grams_frequencies(x_train, y_train, 3)
    #      list of tuples where each tuple has pair of (n_gram@category, frequency)
    classifier = KNNClassifier(3501)
    proccessed_x_test = pretprocessing(x_test, 3)

    classifier.fit(frequencies, y_train)

    y_pred = classifier.predict(proccessed_x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Model evaluation on Test data")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print()