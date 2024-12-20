import os
import random
import soundfile as sf
import openl3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional
import sys

class EmbeddingVisualizer:
    def __init__(self, dataset_folder: str, model):
        """
        Initialize the visualizer with the dataset folder path.
        :param dataset_folder: Path to the GTZAN dataset folder.
        """
        self.dataset_folder = dataset_folder
        self.model = model
        self.embeddings = []
        self.labels = []
        self.failed_files = []

    def generate_embeddings(self, num_samples_per_genre: int = 10):
        """
        Generate embeddings for audio files in the dataset.
        :param num_samples_per_genre: Number of audio samples to process per genre.
        """
        genres = [genre for genre in os.listdir(self.dataset_folder) if os.path.isdir(os.path.join(self.dataset_folder, genre))]
        print(f"Found genres: {genres}")

        for genre in genres:
            genre_folder = os.path.join(self.dataset_folder, genre)
            files = [file for file in os.listdir(genre_folder) if file.endswith(('.wav', '.ogg', '.flac'))]

            # Randomly sample files from each genre
            random.shuffle(files)
            sampled_files = files[:num_samples_per_genre]
            total_files = len(sampled_files)
            print(f"Processing {total_files} files from genre: {genre}")

            for i, file in enumerate(sampled_files):
                file_path = os.path.join(genre_folder, file)
                try:
                    # Read audio file
                    audio, sr = sf.read(file_path)
                    
                    # Generate embedding
                    embedding, _ = openl3.get_audio_embedding(audio, sr, model=self.model, verbose=False)
                    
                    # Store the embedding and label
                    self.embeddings.append(embedding)
                    self.labels.append(genre)
                except Exception as e:
                    self.failed_files.append(file_path)

                # Update progress
                sys.stdout.write(f"\rProcessed {i + 1}/{total_files} files from genre: {genre}")
                sys.stdout.flush()

            print()  # Move to the next line after processing all files in the genre
            
        print(f"Processed {len(self.embeddings)} audio files from all genres.")
        if self.failed_files:
            print("Failed to process the following files:")
            for file in self.failed_files:
                print(file)

    def plot_embeddings(
        self, 
        method: str = "pca", 
        save_path: Optional[str] = None
    ):
        """
        Visualize embeddings using PCA or t-SNE, coloring by genre.
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
            perplexity = min(30, max(1, n_samples - 1))
            print(f"Using perplexity={perplexity} for t-SNE")
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_emb = reducer.fit_transform(embeddings_array)
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")


        # Plot the embeddings
        plt.figure(figsize=(12, 10))
        genres = np.unique(self.labels)
        for genre in genres:
            idx = np.where(np.array(self.labels) == genre)[0]
            plt.scatter(reduced_emb[idx, 0], reduced_emb[idx, 1], label=genre, s=100, alpha=0.7)

        plt.title(f"Audio Embeddings Visualization ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Genre", loc='best')
        plt.grid()


        # Save or display the plot
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Initialize the visualizer with the dataset folder
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=6144)
    dataset_folder = "gtzan_dataset/genres_original"
    visualizer = EmbeddingVisualizer(dataset_folder, model)

    # Generate embeddings for 10 random songs per genre
    visualizer.generate_embeddings(num_samples_per_genre=100)

    # Create output folder for results
    os.makedirs("results", exist_ok=True)

    # Plot embeddings using t-SNE and PCA
    visualizer.plot_embeddings(method="tsne", save_path="results/gtzan_tsne_result100.png")
    visualizer.plot_embeddings(method="pca", save_path="results/gtzan_pca_result100.png")
