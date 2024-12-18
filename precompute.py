import requests
import numpy as np
import pandas as pd
from numpy.linalg import norm
from datasets import load_dataset

# Define cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

# Jina API key
API_KEY = 'jina_4d9174d9a85a4a06a62ddcaf7ee59fa7RjwQ9weOg0_rrIIKAhV3BzbfXBiX'

# Load the dataset using Hugging Face Datasets
ds = load_dataset("jinaai/jina-weaviate-hackson-movie")

# Sample title and wikipedia_link from the dataset (use the available columns)
titles = ds['train']['title'][:5]  # First 5 entries in the title column
wikipedia_links = ds['train']['wikipedia_link'][:5]  # First 5 entries in the wikipedia_link column

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

# Check for successful response
if response.status_code == 200:
    embeddings = response.json()['data']
    
    # Calculate cosine similarity between first title-link pair (you can adjust this as needed)
    title_embedding = np.array(embeddings[0]['embedding'])
    link_embedding = np.array(embeddings[1]['embedding'])
    sim = cos_sim(title_embedding, link_embedding)
    print(f"Cosine title<->link similarity: {sim}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Optionally save embeddings to a file
# Convert embeddings into a DataFrame
embedding_df = pd.DataFrame(embeddings)
embedding_df.to_parquet('embeddings_output.parquet')
