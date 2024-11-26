import os
import soundfile as sf
import openl3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Directory containing audio files
DATASET_FOLDER = "sounds"

# Function to process audio files and generate embeddings
def generate_embeddings(folder_path):
    embeddings = []
    file_names = []
    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.ogg', '.flac')):  # Supported formats
            file_path = os.path.join(folder_path, file)
            print(f"Processing: {file_path}")

            # Read audio file
            audio, sr = sf.read(file_path)
            
            # Generate OpenL3 embeddings
            emb, ts = openl3.get_audio_embedding(audio, sr)
            embeddings.append(emb.mean(axis=0))  # Use mean embedding for simplicity
            file_names.append(file)
    
    return embeddings, file_names

def plot_embeddings(embeddings, file_names, method="pca", save_path=None):
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced_emb = reducer.fit_transform(embeddings)
    elif method == "tsne":
        # Adjust perplexity based on the number of samples
        n_samples = len(embeddings)
        perplexity = min(30, max(1, n_samples - 1))  # t-SNE requires perplexity < n_samples
        print(f"Using perplexity={perplexity} for t-SNE")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_emb = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=np.arange(len(file_names)), cmap='viridis', s=100)
    
    # Annotate points with file names
    for i, name in enumerate(file_names):
        plt.annotate(name, (reduced_emb[i, 0], reduced_emb[i, 1]), fontsize=9)
    
    plt.title(f"Audio Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label="File Index")
    plt.grid()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Save plot as a PNG file with high resolution
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Main script
if __name__ == "__main__":
    embeddings, file_names = generate_embeddings(DATASET_FOLDER)
    
    # Convert embeddings to numpy array for processing
    embeddings = np.array(embeddings)
    
    # Plot embeddings using PCA or t-SNE
    plot_embeddings(embeddings, file_names, method="tsne", save_path="results/test_tsne_result.png")
    plot_embeddings(embeddings, file_names, method="pca", save_path="results/test_pca_result.png")
    
    
