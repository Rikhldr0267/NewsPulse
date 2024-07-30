import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Initialize tools and libraries
cohere_api_key = st.secrets["cohere_api_key"]
co = cohere.Client(api_key=cohere_api_key)
summarizer = pipeline("summarization")
sia = SentimentIntensityAnalyzer()
nltk.download('punkt')

def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', ' ').replace('\r', ' '). replace('\t', ' ')
    return text.strip()

def fetch_text(url):
    response = requests.get(url)
    return clean_text(response.text)

def generate_embeddings(texts):
    response = co.embed(texts=texts, model="embed-english-v2.0")
    embeddings = np.array(response.embeddings)
    return embeddings

def fetch_and_embed(text):
    embedding = co.embed(model="large", texts=[text])
    return np.array(embedding.embeddings[0])

def find_similar_words(text1, text2):
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    common_words = words1.intersection(words2)
    return list(common_words)

def visualize_embeddings(embedding1, embedding2):
    embeddings = np.array([embedding1, embedding2])
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    fig = px.scatter(
        x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
        text=["Article 1", "Article 2"],
        title="PCA of Article Embeddings",
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )
    fig.update_traces(textposition='top center')
    return fig

def generate_answer(query, context):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Question: {query}\n\nContext: {context}\n\nAnswer:",
        max_tokens=300,
        temperature=0.5
    )
    return response.generations[0].text.strip()

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores

def main():
    st.title("News Analysis Tool")

    # Sidebar options
    option = st.sidebar.selectbox("Choose an option:", ["Ask about an article", "Compare two articles"])

    if option == "Ask about an article":
        st.sidebar.header("Input URL")
        url_input = st.sidebar.text_input("Enter URL for the article:")

        if 'article' not in st.session_state:
            st.session_state.article = ""

        if st.sidebar.button("Process URL"):
            st.session_state.article = fetch_text(url_input)
            st.success("Article fetched and processed!")

        response_length = st.sidebar.slider("Set response max length", 50, 500, 150)
        include_sentiment = st.sidebar.checkbox("Include sentiment analysis", True)

        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if st.session_state.article:
                context = st.session_state.article
                answer = generate_answer(query, context)
                st.write("Answer:", answer[:response_length])
                if include_sentiment:
                    sentiment = analyze_sentiment(answer)
                    st.write("Sentiment Analysis:", sentiment)

                st.write("Article URL:")
                st.write(url_input)
            else:
                st.error("Please process the URL first.")

    elif option == "Compare two articles":
        st.sidebar.header("Input URLs")
        url1 = st.sidebar.text_input("Enter URL for Article 1:")
        url2 = st.sidebar.text_input("Enter URL for Article 2:")
    
        if st.sidebar.button("Compare Articles"):
            if url1 and url2:
                text1 = fetch_text(url1)
                text2 = fetch_text(url2)
                embedding1 = fetch_and_embed(text1)
                embedding2 = fetch_and_embed(text2)
                similarity = cosine_similarity([embedding1, embedding2])[0, 1]
                common_words = find_similar_words(text1, text2)

                st.write("### Similarity Score")
                st.write(f"The similarity score between the articles is: {similarity:.2f}")
                st.write("### Similar Words in Both Articles")
                st.write(', '.join(common_words))

                # Visualize embeddings
                fig = visualize_embeddings(embedding1, embedding2)
                st.plotly_chart(fig)

            else:
                st.error("Please enter URLs for both articles.")

if __name__ == "__main__":
    main()
