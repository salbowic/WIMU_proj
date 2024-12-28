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
        self.centroids = []
        self.failed_files = []
        
    def get_dataset_folder(self):
        return self.dataset_folder
    
    def set_dataset_folder(self, dataset_folder):
        self.dataset_folder = dataset_folder
        
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model

    def generate_embeddings(self, num_samples_per_genre: int = 10, emb_dir: str = 'results/embeddings/embeddings1'):
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
                output_dir = f'{emb_dir}/{genre}'
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    # Process audio file and save embedding to disk
                    openl3.process_audio_file(file_path, model=self.model, suffix='_emb', output_dir=output_dir, verbose=False)
                    
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    self.failed_files.append(file_path)

                # Update progress
                sys.stdout.write(f"\rProcessed {i + 1}/{total_files} files from genre: {genre}")
                sys.stdout.flush()

            print()
            
        print(f"Finished processing audio files from all genres.")
        if self.failed_files:
            print("Failed to process the following files:")
            for file in self.failed_files:
                print(file)


    def load_embeddings(self, input_dir: str = 'results/embeddings/embeddings1'):
        """
        Load embeddings and labels from the saved .npz files.
        :param output_dir: Directory where the embeddings are saved.
        """
        genres = [genre for genre in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, genre))]
        print(f"Loading found genres: {genres}")

        for genre in genres:
            genre_folder = os.path.join(input_dir, genre)
            files = [file for file in os.listdir(genre_folder) if file.endswith('_emb.npz')]

            for file in files:
                file_path = os.path.join(genre_folder, file)
                
                try:
                    # Load the saved embedding
                    data = np.load(file_path)
                    embedding = data['embedding']
                    mean_emb = embedding.mean(axis=0)  # Use mean embedding for simplicity
                    
                    # Store the embedding and label
                    self.embeddings.append(mean_emb)
                    self.labels.append(genre)
                    
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        # Calculate centroids
        self._calculate_genre_centroids(input_dir=input_dir)
        
        print(f"Loaded {len(self.embeddings)} embeddings from all genres.")
  
  
    def _calculate_genre_centroids(self, input_dir: str = 'results/embeddings/embeddings1'):
        """
        Calculate and return the centroids of the genre embeddings.
        :param input_dir: Directory where the embeddings are saved.
        :return: Dictionary with genres as keys and centroid embeddings as values.
        """
        if not self.embeddings:
            self.load_embeddings(input_dir)
        
        genre_centroids = {}
        genres = np.unique(self.labels)
        
        for genre in genres:
            genre_embeddings = [self.embeddings[i] for i in range(len(self.embeddings)) if self.labels[i] == genre]
            genre_centroid = np.mean(genre_embeddings, axis=0)
            genre_centroids[genre] = genre_centroid
        
        self.centroids = genre_centroids
    
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
            reduced_centroids = reducer.fit_transform(np.array(list(self.centroids.values())))
        elif method == "tsne":
            n_samples = len(embeddings_array)
            perplexity = min(30, max(1, n_samples - 1))
            print(f"Using perplexity={perplexity} for t-SNE")
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_emb = reducer.fit_transform(embeddings_array)
            # Adjust perplexity for centroids
            centroid_perplexity = min(30, max(1, len(self.centroids) - 1))
            centroid_reducer = TSNE(n_components=2, perplexity=centroid_perplexity, random_state=42)
            reduced_centroids = centroid_reducer.fit_transform(np.array(list(self.centroids.values())))
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")


        # Plot the embeddings
        plt.figure(figsize=(12, 10))
        genres = np.unique(self.labels)
        for genre in genres:
            idx = np.where(np.array(self.labels) == genre)[0]
            plt.scatter(reduced_emb[idx, 0], reduced_emb[idx, 1], label=genre, s=100, alpha=0.7)

        # Plot centroids if available
        if self.centroids:
            for i, (genre, centroid) in enumerate(self.centroids.items()):
                reduced_centroid = reduced_centroids[i]
                plt.scatter(reduced_centroid[0], reduced_centroid[1], s=200, marker='X', edgecolors='k')
                
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
      
            