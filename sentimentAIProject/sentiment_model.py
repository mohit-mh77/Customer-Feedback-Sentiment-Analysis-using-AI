import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# dataset load
train_data = pd.read_csv("Train.csv")

X = train_data["text"]
y = train_data["label"]

# vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# model train
model = MultinomialNB()
model.fit(X_vec, y)

# prediction function
def predict_sentiment(review):
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)

    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


