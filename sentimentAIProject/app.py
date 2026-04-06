import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentiment_model import predict_sentiment

# Load dataset
data = pd.read_csv("Train.csv")

st.title("AI Sentiment Analysis Dashboard")

st.write("Analyze movie reviews using AI")

# -------------------------
# Sentiment Distribution
# -------------------------

st.subheader("Sentiment Distribution")

sentiment_counts = data["label"].value_counts()

fig, ax = plt.subplots()

ax.bar(sentiment_counts.index, sentiment_counts.values)

ax.set_xlabel("Sentiment (0 = Negative, 1 = Positive)")
ax.set_ylabel("Number of Reviews")

st.pyplot(fig)

# -------------------------
# Word Cloud
# -------------------------

st.subheader("Word Cloud")

text = " ".join(data["text"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

fig2, ax2 = plt.subplots()

ax2.imshow(wordcloud)

ax2.axis("off")

st.pyplot(fig2)

# -------------------------
# Sentiment Prediction
# -------------------------

st.subheader("Check Review Sentiment")

review = st.text_input("Enter Movie Review", key="review_input")

# Session state counters
if "positive_count" not in st.session_state:
    st.session_state.positive_count = 0

if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0

if st.button("Predict Sentiment"):

    result = predict_sentiment(review)

    st.success(result)

    if "Positive" in result:
        st.session_state.positive_count += 1
    else:
        st.session_state.negative_count += 1

# -------------------------
# Live Dashboard
# -------------------------

st.subheader("Live Sentiment Dashboard")

labels = ["Positive", "Negative"]

values = [
    st.session_state.positive_count,
    st.session_state.negative_count
]

fig3, ax3 = plt.subplots()

ax3.pie(values, labels=labels, autopct="%1.1f%%")

ax3.set_title("Live Sentiment Results")

st.pyplot(fig3)

# -------------------------
# CSV Upload Analysis
# -------------------------

st.subheader("Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV with 'text' column")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "text" in df.columns:

        df["Sentiment"] = df["text"].apply(predict_sentiment)

        st.write(df)

    else:

        st.error("CSV file must contain a 'text' column")
