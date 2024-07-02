import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, recall_score

stopwords = set(stopwords.words('english'))


def cleaning_data(text):
    text = ''.join([word.lower() for word in text if word not in string.punctuation])
    tokens = text.split()
    text = [word for word in tokens if word not in stopwords]
    return text


if __name__ == "__main__":
    df = pd.read_csv('new/ecommerceDataset.csv')
    df.dropna(inplace=True)
    df.columns = ['category', 'text']
    # test = df.tail(3)
    # data = df.head(50420)
    vectorizer = CountVectorizer(analyzer=cleaning_data)
    X_counts = vectorizer.fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(X_counts, df['category'], test_size=0.2, random_state=42)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_score = accuracy_score(y_test, y_pred_nb)
    print('Accuracy for NaiveBayes Model on Test Data is:', accuracy_score)
    print("********************************************************")