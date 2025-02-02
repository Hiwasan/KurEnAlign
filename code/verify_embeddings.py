import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_embeddings(file_path):
    """
    Load embeddings from a file.
    """
    return torch.load(file_path)

def inspect_embeddings(embeddings, num_sentences=5, num_words=5):
    """
    Inspect a subset of the embeddings.
    """
    print("Number of sentences:", len(embeddings))
    print("Number of words in the first sentence:", len(embeddings[0]))

    for i, embedding in enumerate(embeddings):
        if i < num_sentences:  # Limit to the first num_sentences sentences
            print(f"Sentence {i}:")
            full_word = ""
            for word, emb in embedding[:num_words]:  # Limit to the first num_words words
                if word.startswith("▁"):
                    if full_word:
                        print(f"  Word: {full_word}, Embedding: {emb[:5]}...")  # Print first 5 dimensions of the embedding
                    full_word = word[1:]  # Remove the leading underscore
                else:
                    full_word += word
            if full_word:
                print(f"  Word: {full_word}, Embedding: {emb[:5]}...")  # Print the last word

def visualize_embeddings(embeddings):
    """
    Visualize embeddings using PCA.
    """
    # Flatten the embeddings for visualization
    flattened_embeddings = [emb for sentence in embeddings for _, emb in sentence]

    # Convert to numpy array
    flattened_embeddings = torch.stack(flattened_embeddings).detach().numpy()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_embeddings)

    # Plot the PCA result
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title("PCA of Word Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def main():
    # Load the embeddings
    embeddings = load_embeddings('output.pt')

    # Inspect the embeddings
    inspect_embeddings(embeddings)

    # Visualize the embeddings
    visualize_embeddings(embeddings)

if __name__ == "__main__":
    main()