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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

stopword_list = stopwords.words("english")


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
    clean_data = [word.lower() for word in tokens if (word.lower() not in string.punctuation) and (word.lower() not in stopword_list) and (len(word)>2) and (word.isalpha())]
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

    df = pd.read_csv('new/ecommerceDataset.csv')
    df.columns = ['category', 'text']
    df['text'] = df['text'].astype(str)

    print(df['text'])

    df['cleaned_text'] = df['text'].apply(clean_text)

    model = Word2Vec(sentences=df['cleaned_text'], vector_size=100, window=5, min_count=1, workers=4)

    df['vectorized_text'] = vectorizer(df['cleaned_text'], model)

    df['category'] = df['category'].replace({"Household": 0, "Books": 1, "Electronics": 2, "Clothing & Accessories": 3})

    x_train, x_test, y_train, y_test = train_test_split(df['vectorized_text'].tolist(), df['category'].values, test_size=0.2)

    classifier = KNeighborsClassifier(n_neighbors=5)  # Example: k=5
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
    print(dict_data)
    top_6 = Counter(dict_data).most_common(6)
    top_6_roles = [word for word, count in top_6]

    filtered_df = all_dialogues_df[all_dialogues_df['role'].isin(top_6_roles)]
    print(filtered_df['text'])
    filtered_df['text'] = filtered_df['text'].astype(str)
    filtered_df['cleaned_text'] = filtered_df['text'].apply(clean_text)

    model = Word2Vec(sentences=filtered_df['cleaned_text'], vector_size=50, window=2, min_count=2, workers=4)
    print(model.vector_size)


    filtered_df['vectorized_text'] = vectorizer(filtered_df['cleaned_text'], model)

    filtered_df['role'] = filtered_df['role'].replace({"Joey": 0, "Monica": 1, "Rachel": 2, "Chandler": 3,
                                                               "Ross": 4, "Phoebe": 5})

    print(filtered_df['role'].value_counts())
    print(input())

    x_train, x_test, y_train, y_test = train_test_split(filtered_df['vectorized_text'].tolist(), filtered_df['role'].values,
                                                        test_size=0.2)

    classifier = KNeighborsClassifier(n_neighbors=13)  # Example: k=5
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


