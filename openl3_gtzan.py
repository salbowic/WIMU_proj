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
    def __init__(
        self,
        dataset_folder: str = None,
        model = openl3.models.load_audio_embedding_model(input_repr="mel128",
                                                         content_type="music",
                                                         embedding_size=6144)
        ):
        """
        Initialize the visualizer with the dataset folder path.
        :param dataset_folder: Path to the GTZAN dataset folder.
        """
        self.dataset_folder = dataset_folder
        self.model = model
        self.embeddings = []
        self.labels = []
        self.failed_files = []
        
    def get_dataset_folder(self):
        return self.dataset_folder
    
    def set_dataset_folder(self, dataset_folder):
        self.dataset_folder = dataset_folder
        
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model

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

            for i, file in enumerate(sampled_files):
                file_path = os.path.join(genre_folder, file)
                try:
                    # Process embedding
                    output_dir = f'results/embeddings/{genre}'
                    os.makedirs(output_dir, exist_ok=True)
                    openl3.process_audio_file(file_path, model = self.model, suffix='_emb', output_dir=f'results/embeddings/{genre}', verbose=False)
                    
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    self.failed_files.append(file_path)

                # Update progress
                sys.stdout.write(f"\rProcessed {i + 1}/{total_files} files from genre: {genre}")
                sys.stdout.flush()

            print()
            
        print(f"Processed {len(self.embeddings)} audio files from all genres.")
        if self.failed_files:
            print("Failed to process the following files:")
            for file in self.failed_files:
                print(file)


    def load_embeddings(self, output_dir: str = 'results/embeddings'):
        """
        Load embeddings and labels from the saved .npz files.
        :param output_dir: Directory where the embeddings are saved.
        """
        genres = [genre for genre in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, genre))]
        print(f"Loading found genres: {genres}")

        for genre in genres:
            genre_folder = os.path.join(output_dir, genre)
            files = [file for file in os.listdir(genre_folder) if file.endswith('_emb.npz')]

            for file in files:
                file_path = os.path.join(genre_folder, file)
                
                try:
                    # Load the saved embedding
                    data = np.load(file_path)
                    embedding = data['embedding']
                    
                    # Store the embedding and label
                    self.embeddings.append(embedding)
                    self.labels.append(genre)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        print(f"Loaded {len(self.embeddings)} embeddings from all genres.")
        
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
    visualizer = EmbeddingVisualizer()
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=6144)
    dataset_folder = "gtzan_dataset\genres_original"
    # visualizer.set_dataset_folder(dataset_folder)
    # visualizer.set_model(model)
    

    # Generate embeddings for 10 random songs per genre
    # visualizer.generate_embeddings(num_samples_per_genre=100)

    # Load embeddings from file
    visualizer.load_embeddings()
    
    # # Create output folder for results
    # os.makedirs("results", exist_ok=True)

    # # Plot embeddings using t-SNE and PCA
    visualizer.plot_embeddings(method="tsne", save_path="results/gtzan_tsne_result100v2.png")
    visualizer.plot_embeddings(method="pca", save_path="results/gtzan_pca_result100v2.png")
