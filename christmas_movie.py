import streamlit as st
import requests
import json
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Jina AI SeFo API key and URL
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.getenv("JINA_API_KEY")  # Fetch the API key from the environment variables

# Function to embed movie data using Jina AI
def get_embeddings(data, model="jina-embeddings-v3"):
    if not JINA_API_KEY:
        st.error("JINA_API_KEY not found. Please check your .env file.")
        return None
    
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "data": data}
    response = requests.post(JINA_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        st.error("Failed to fetch embeddings. Check your API key or input.")
        return None

# Load precomputed embeddings (or calculate them if needed)
def load_movie_data():
    try:
        with open("movie_data.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Movie data file not found. Precompute embeddings first.")
        return None

# Find the most similar movie
def find_top_match(query_embedding, movie_data):
    similarities = []
    for movie in movie_data:
        similarity = np.dot(query_embedding, movie["embedding"]) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(movie["embedding"])
        )
        similarities.append((similarity, movie))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[0][1] if similarities else None

# Streamlit app layout
def main():
    st.title("🎄 Christmas Movie Prediction App")
    st.markdown(
        """
        Use AI to find the best Christmas movie!  
        Enter a short description or your favorite Christmas movie theme, and we'll match it with the most relevant movie in our list.
        """
    )

    # User input
    user_input = st.text_area("Describe your favorite Christmas movie or theme", "")
    
    # Load movie data
    movie_data = load_movie_data()

    if st.button("Find My Movie 🎥") and user_input:
        # Get user query embedding
        query_embedding = get_embeddings([user_input])
        if query_embedding:
            query_embedding = query_embedding[0]
            # Find top match
            top_movie = find_top_match(query_embedding, movie_data)
            if top_movie:
                st.success(f"Top Matched Movie: **{top_movie['title']}**")
                st.write(f"**Description:** {top_movie['description']}")
                if "poster_url" in top_movie:
                    st.image(top_movie["poster_url"], caption=top_movie["title"])
            else:
                st.warning("No match found. Try a different description.")

if __name__ == "__main__":
    main()
