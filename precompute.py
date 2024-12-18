import requests
import numpy as np
from numpy.linalg import norm
import json

# Cosine Similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Function to get content from Reader API
def get_cleaned_content_from_url(url):
    reader_url = f"https://r.jina.ai/{url}"
    response = requests.get(reader_url)
    if response.status_code == 200:
        return response.json()['content']
    else:
        print(f"Error fetching content from {url}")
        return None

# Function to get embeddings using Jina API
def get_embeddings(text):
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer <YOUR_JINA_AI_API_KEY>'
    }
    data = {
        'input': [{"text": text}],
        'model': 'jina-clip-v2',
        'encoding_type': 'float',
        'dimensions': '768'
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return np.array(response.json()['data'][0]['embedding'])
    else:
        print("Error generating embeddings.")
        return None

# Example usage for fetching and embedding a movie URL
def process_movie_url(url):
    print(f"Fetching content from: {url}")
    content = get_cleaned_content_from_url(url)
    if content:
        print("Content fetched successfully.")
        embedding = get_embeddings(content)
        if embedding is not None:
            print("Embedding generated.")
            # Save or process embedding further as needed
        else:
            print("Failed to generate embedding.")
    else:
        print("Failed to fetch content.")
        
# Example: Replace this with actual movie URLs
movie_url = "https://arxiv.org/abs/2310.19923"
process_movie_url(movie_url)
