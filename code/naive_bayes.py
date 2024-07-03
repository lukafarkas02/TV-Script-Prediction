from collections import Counter

import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, recall_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Instantiate the TweetTokenizer class
tokenizer = TweetTokenizer(
    preserve_case=False,  # Convert all text to lowercase
    strip_handles=True,    # Remove Twitter handles (user mentions)
    reduce_len=True        # Normalize repeated characters
)


stopwords = set(stopwords.words('english'))
# Instantiate the PorterStemmer for word stemming
stemmer = PorterStemmer()


def cleaning_data(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = text.split()
    text = [word for word in tokens if word not in stopwords]
    return text


def preprocess(sample):
    sample = str(sample)
    sample = sample.lower()
    sample = re.sub('https?:\/\/.*[\r\n]*', ' ', sample)
    sample = re.sub(r'[#:-]', ' ', sample)
    sample = tokenizer.tokenize(sample)

    output = []

    for i in sample:
        if i not in stopwords and i not in string.punctuation and i.isnumeric() == False and len(i) > 1:
            output.append(stemmer.stem(i))

    return output


# if __name__ == "__main__":
#     df = pd.read_csv('new/ecommerceDataset.csv')
#     df.dropna(inplace=True)
#     df.columns = ['category', 'text']
#     # test = df.tail(3)
#     # data = df.head(50420)


class NB_Text_Classifier:
    def __init__(self, texts: list, categories: list) -> None:
        assert len(texts) == len(categories), 'Za svaki tekst postoji tačno jedna kategorija.'
        self.texts = texts
        self.categories = categories
        self.word_counts = {}
        self.text_counts = {}
        self.n_words = {}
        self.prior = {}
        self.unique_categories = set(categories)  # Set svih kategorija

        for category in self.unique_categories:
            self.word_counts[category] = {}
            self.text_counts[category] = 0
            self.n_words[category] = 0
            self.prior[category] = 0

    def _preprocess(self, text: str) -> list:
        '''Preprocess and returns text.'''
        import re
        text = re.sub(r'[^\w\s]', '', text)  # uklonimo znakove
        words = text.lower().split()  # svedemo na mala slova i podelimo na reči
        return words

    def fit(self) -> None:
        for text, category in zip(self.texts, self.categories):
            words = self._preprocess(text)
            for word in words:
                self.word_counts[category][word] = self.word_counts[category].get(word, 0) + 1
            self.text_counts[category] += 1

        total_texts = len(self.texts)

        for category in self.unique_categories:
            self.n_words[category] = sum(self.word_counts[category].values())
            self.prior[category] = self.text_counts[category] / total_texts

    def predict(self, text: str) -> dict:
        '''Returns a dictionary of probabilities for each category.'''
        words = self._preprocess(text)
        category_probs = {}

        for category in self.unique_categories:
            p_words_given_category = []
            for word in words:
                # Verovatnoća da se reč nađe u recenziji određene kategorije
                p_word_given_category = (self.word_counts[category].get(word, 0) + 1) / (
                        self.n_words[category] + len(self.word_counts[category]))  # Laplace Smoothing
                p_words_given_category.append(p_word_given_category)

            # Računamo P(text|category) tako što pomnožimo verovatnoću za svaku reč
            p_text_given_category = np.prod(p_words_given_category)

            # Iskoristimo Bajesovu formulu da nađemo P(category|text)
            category_probs[category] = self.prior[category] * p_text_given_category

        return category_probs

    def predict_category(self, text: str) -> str:
        '''Predicts the category for a given text.'''
        category_probs = self.predict(text)
        return max(category_probs, key=category_probs.get)


if __name__ == '__main__':
    df = pd.read_csv('new/ecommerceDataset.csv')
    df.columns = ['category', 'text']

    df['text'] = df['text'].fillna('')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_texts = train_df['text'].tolist()
    train_categories = train_df['category'].tolist()

    clf = NB_Text_Classifier(train_texts, train_categories)
    clf.fit()

    test_texts = test_df['text'].tolist()
    test_categories = test_df['category'].tolist()

    # Make predictions on the test set
    predictions = [clf.predict_category(text) for text in test_texts]

    # Calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(test_categories, predictions)

    print(f'Accuracy: {accuracy:.2f}')

    print("Bayes as python library")
    vectorizer = CountVectorizer(analyzer=cleaning_data)
    X_counts = vectorizer.fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(X_counts, df['category'], test_size=0.2, random_state=42)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_score = accuracy_score(y_test, y_pred_nb)
    print('Accuracy for NaiveBayes Model on Test Data is:', accuracy_score)
    print("********************************************************")


