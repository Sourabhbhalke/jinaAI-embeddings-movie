import json
from christmas_movie import get_embeddings  # Assuming this function is available
from datasets import load_dataset

# Load the datasets
movie_dataset = load_dataset("jinaai/jina-weaviate-hackson-movie")

# Precompute movie embeddings
movie_data = []
for item in movie_dataset['train']:
    movie_entry = {
        "title": item["title"],  # Use the title of the movie
        "wikipedia_link": item["wikipedia_link"],  # Optionally, you can fetch data from this link
        "embedding": None  # Placeholder for embedding
    }
    movie_data.append(movie_entry)

# Generate embeddings for movie descriptions
movie_descriptions = [movie["title"] for movie in movie_data]
movie_embeddings = get_embeddings(movie_descriptions)

if movie_embeddings is not None:
    for i, embedding in enumerate(movie_embeddings):
        movie_data[i]["embedding"] = embedding
else:
    print("Error: Movie embeddings were not generated.")

# Save the movie data with embeddings to a file
with open("movie_data_with_embeddings.json", "w") as file:
    json.dump(movie_data, file)

print("Embeddings for movies have been computed and saved!")
