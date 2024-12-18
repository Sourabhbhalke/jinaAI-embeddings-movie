import streamlit as st
import pandas as pd
import numpy as np
import requests
from numpy.linalg import norm
from datasets import load_dataset

# Define cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Jina API key
API_KEY = 'jina_4d9174d9a85a4a06a62ddcaf7ee59fa7RjwQ9weOg0_rrIIKAhV3BzbfXBiX'

# Load the dataset using Hugging Face Datasets
ds = load_dataset("jinaai/jina-weaviate-hackson-movie")

# Sample title and wikipedia_link from the dataset
titles = ds['train']['title'][:50]  # First 50 entries in the title column
wikipedia_links = ds['train']['wikipedia_link'][:50]  # First 50 entries in the wikipedia_link column

# API endpoint for embeddings
url = 'https://api.jina.ai/v1/embeddings'

# Define headers with API key
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Prepare the data for the request
data = {
    'input': [
        {"text": title, "url": link}
        for title, link in zip(titles, wikipedia_links)
    ],
    'model': 'jina-clip-v2',
    'encoding_type': 'float',
    'dimensions': '768'
}

# Send request to Jina API to generate embeddings
response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    embeddings = response.json()['data']
    
    # Create a DataFrame for easy access to embeddings and movie info
    embedding_df = pd.DataFrame(embeddings)
    embedding_df['title'] = titles
    embedding_df['wikipedia_link'] = wikipedia_links
else:
    st.error("Error fetching embeddings, please try again.")
    embedding_df = pd.DataFrame()

# Streamlit App UI
st.title("🎄 Bombastic Christmas Movie Recommender 🎄")

st.markdown("""
    Welcome to the **Christmas Movie Recommender**! Ask a question like:
    * "that one movie where that one boy fights intruders!"
    * "A Christmas adventure with an elf!"
    And we'll find the perfect movie for you!
    Get ready for a bombastic recommendation experience!
""")

# User query input
user_query = st.text_input("Enter a question or description about a movie 🎥", "")

if user_query:
    # Send the user's input to the API to get its embedding
    data_query = {
        'input': [{"text": user_query}],
        'model': 'jina-clip-v2',
        'encoding_type': 'float',
        'dimensions': '768'
    }

    query_response = requests.post(url, headers=headers, json=data_query)

    if query_response.status_code == 200:
        query_embedding = np.array(query_response.json()['data'][0]['embedding'])
        
        # Calculate similarity of the user's query with all other movies
        similarities = []
        for embedding in embeddings:
            movie_embedding = np.array(embedding['embedding'])
            similarities.append(cos_sim(query_embedding, movie_embedding))

        # Get the top 5 most similar movies
        top_similar_idx = np.argsort(similarities)[-6:][::-1]  # Excluding the movie itself (most similar)

        st.subheader(f"Movies similar to your question: **'{user_query}'**")

        # Display the recommended movies in bombastic style with links and images
        for idx in top_similar_idx:
            movie_title = embedding_df['title'][idx]
            movie_link = embedding_df['wikipedia_link'][idx]
            st.markdown(f"🔗 **{movie_title}**\n[Click here for more info]({movie_link})")
            st.image(f"https://via.placeholder.com/250x350.png?text={movie_title.replace(' ', '+')}", caption=f"**{movie_title}**", width=250)
            st.markdown("---")
    else:
        st.error("Error fetching query embedding, please try again.")
else:
    st.markdown("""
        Type a question or description about a movie in the text box above to get recommendations!
    """)
