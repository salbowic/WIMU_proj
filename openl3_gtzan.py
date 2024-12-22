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
                    self.embeddings.append(embedding.mean(axis=0))  # Use mean embedding for simplicity
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


    def save_embeddings(self, file_path: str):
        """
        Save embeddings and labels to a file.
        :param file_path: Path to the file where embeddings and labels will be saved.
        """
        np.savez(file_path, embeddings=self.embeddings, labels=self.labels)
        print(f"Embeddings and labels saved to {file_path}")


    def load_embeddings(self, file_path: str):
        """
        Load embeddings and labels from a file.
        :param file_path: Path to the file from which embeddings and labels will be loaded.
        """
        data = np.load(file_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        print(f"Embeddings and labels loaded from {file_path}")
        
        
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



def inspect_npz(file_path: str, num_elements: int = 5):
    """
    Inspect the structure of a .npz file and print its contents.
    :param file_path: Path to the .npz file.
    :param num_elements: Number of elements to print from each array.
    """
    data = np.load(file_path, allow_pickle=True)
    print(f"Keys in the .npz file: {list(data.keys())}")
    for key in data.keys():
        print(f"\nKey: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")
        print(f"First {num_elements} elements of {key}: {data[key][:num_elements]}")


if __name__ == "__main__":
    # # Initialize the visualizer with the dataset folder
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=6144)
    dataset_folder = "gtzan_dataset/genres_original"
    visualizer = EmbeddingVisualizer(dataset_folder, model)

    # # Generate embeddings for 10 random songs per genre
    # visualizer.generate_embeddings(num_samples_per_genre=100)

    # # Save embeddings to file
    embeddings_file = "results/gtzan_embeddings.npz"
    # visualizer.save_embeddings(embeddings_file)

    # Load embeddings from file
    visualizer.load_embeddings(embeddings_file)
    
    # Create output folder for results
    os.makedirs("results", exist_ok=True)

    # Plot embeddings using t-SNE and PCA
    visualizer.plot_embeddings(method="tsne", save_path="results/gtzan_tsne_result100.png")
    visualizer.plot_embeddings(method="pca", save_path="results/gtzan_pca_result100.png")
