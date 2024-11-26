import os
import soundfile as sf
import openl3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional


class AudioEmbeddingProcessor:
    def __init__(self, dataset_folder: str):
        """
        Initialize the processor with the dataset folder path.
        :param dataset_folder: Path to the folder containing audio files.
        """
        self.dataset_folder = dataset_folder
        self.embeddings = []
        self.file_names = []

    def generate_embeddings(self):
        """
        Generate embeddings for all audio files in the dataset folder.
        """
        print(f"Processing audio files in folder: {self.dataset_folder}")
        for file in os.listdir(self.dataset_folder):
            if file.endswith(('.wav', '.ogg', '.flac')):  # Supported formats
                file_path = os.path.join(self.dataset_folder, file)
                print(f"Processing: {file_path}")

                # Read audio file
                audio, sr = sf.read(file_path)
                
                # Generate OpenL3 embeddings
                emb, ts = openl3.get_audio_embedding(audio, sr)
                self.embeddings.append(emb.mean(axis=0))  # Use mean embedding for simplicity
                self.file_names.append(file)

        print(f"Processed {len(self.file_names)} files.")
    
    def plot_embeddings(
        self, 
        method: str = "pca", 
        save_path: Optional[str] = None
    ):
        """
        Visualize embeddings using PCA or t-SNE.
        :param method: Dimensionality reduction method ('pca' or 'tsne').
        :param save_path: Path to save the plot. If None, the plot will be displayed.
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Run `generate_embeddings()` first.")

        embeddings_array = np.array(self.embeddings)
        
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced_emb = reducer.fit_transform(embeddings_array)
        elif method == "tsne":
            n_samples = len(embeddings_array)
            perplexity = min(30, max(1, n_samples - 1))  # t-SNE requires perplexity < n_samples
            print(f"Using perplexity={perplexity} for t-SNE")
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_emb = reducer.fit_transform(embeddings_array)
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

        # Plot the embeddings
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=np.arange(len(self.file_names)), cmap='viridis', s=100)
        
        # Annotate points with file names
        for i, name in enumerate(self.file_names):
            plt.annotate(name, (reduced_emb[i, 0], reduced_emb[i, 1]), fontsize=9)
        
        plt.title(f"Audio Embeddings Visualization ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar(label="File Index")
        plt.grid()

        # Save or display the plot
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Initialize the processor with the dataset folder
    dataset_folder = "sounds"
    processor = AudioEmbeddingProcessor(dataset_folder)

    # Generate embeddings
    processor.generate_embeddings()

    # Create output folder for results
    os.makedirs("results", exist_ok=True)

    # Plot embeddings using t-SNE and PCA
    processor.plot_embeddings(method="tsne", save_path="results/test_tsne_result.png")
    processor.plot_embeddings(method="pca", save_path="results/test_pca_result.png")
