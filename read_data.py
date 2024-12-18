from datasets import load_dataset

def explore_dataset(dataset_name):
    """
    Function to load and explore a Hugging Face dataset.
    It will print out the structure and the first few entries of the dataset.
    """
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name)

        # Print the available splits (train, test, etc.)
        print(f"Available splits in {dataset_name}: {dataset.keys()}\n")

        # Inspect the first few entries in the training split
        print(f"First 3 entries from the 'train' split in {dataset_name}:")
        for i, entry in enumerate(dataset['train'][:3]):
            print(f"\nEntry {i+1}: {entry}")

    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")


if __name__ == "__main__":
    # Explore the Christmas Movie dataset
    print("Exploring Christmas Movie Dataset:\n")
    explore_dataset("jinaai/jina-weaviate-hackson-movie")

    # Explore the Christmas Cookie dataset
    print("\nExploring Christmas Cookie Dataset:\n")
    explore_dataset("jinaai/jina-weaviate-hackson-cookie")
