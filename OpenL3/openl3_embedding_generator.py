import os
import random
import soundfile as sf
import openl3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys
import seaborn as sns
    
class EmbeddingVisualizer:
    def __init__(
        self,
        dataset_folder: str = None,
        model = None
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


    def generate_embeddings(self, num_samples_per_genre: Optional[int] = None, emb_dir: str = 'results/embeddings/embeddings1'):
        """
        Generate embeddings for audio files in the dataset.
        :param num_samples_per_genre: Number of audio samples to process per genre. If None, process all samples.
        :param emb_dir: Directory to save the embeddings.
        """
        genres = [genre for genre in os.listdir(self.dataset_folder) if os.path.isdir(os.path.join(self.dataset_folder, genre))]
        print(f"Found genres: {genres}")

        for genre in genres:
            genre_folder = os.path.join(self.dataset_folder, genre)
            files = [file for file in os.listdir(genre_folder) if file.endswith(('.wav', '.ogg', '.flac'))]

            # If num_samples_per_genre is None, process all files
            if num_samples_per_genre is None:
                sampled_files = files
            else:
                # Randomly sample files from each genre
                random.seed(42)
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
                sys.stdout.write(f"\rProcessed {i + 1}/{total_files} files from: {genre}")
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
            files = [file for file in os.listdir(genre_folder) if file.endswith(('.npz', '.npy'))]

            for file in files:
                file_path = os.path.join(genre_folder, file)
                
                try:
                    data = np.load(file_path)
                    # Load the saved embedding
                    if file.endswith('.npz'):   
                        if 'embedding' in data:
                            embedding = data['embedding']
                        elif 'data' in data:
                            embedding = data['data']
                        else:
                            # Interpret the entire data as the embedding
                            embedding = data[list(data.keys())[0]]
                    elif file.endswith('.npy'):
                        embedding = data
                    
                    # Check if embedding has 3 axes and reduce the 1st or 2nd axis if necessary
                    if embedding.ndim == 3:
                        mean_emb = embedding.mean(axis=(0,1))
                    elif embedding.ndim == 2:
                        mean_emb = embedding.mean(axis=0)
                    else:
                        mean_emb = embedding

                    # Store the embedding and label
                    self.embeddings.append(mean_emb)
                    self.labels.append(genre)
                    
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        # Calculate centroids
        self._calculate_genre_centroids()
        
        print(f"Loaded {len(self.embeddings)} embeddings from all genres.")
  
  
    def _calculate_genre_centroids(self):
        """
        Calculate and return the centroids of the genre embeddings.
        :return: Dictionary with genres as keys and centroid embeddings as values.
        """
        genre_centroids = {}
        genres = np.unique(self.labels)
        
        for genre in genres:
            genre_embeddings = [self.embeddings[i] for i in range(len(self.embeddings)) if self.labels[i] == genre]
            
            # Ensure all embeddings have the same shape
            if len(genre_embeddings) > 0:
                first_shape = genre_embeddings[0].shape
                if all(emb.shape == first_shape for emb in genre_embeddings):
                    genre_centroid = np.mean(genre_embeddings, axis=0)
                    genre_centroids[genre] = genre_centroid
                else:
                    print(f"Inconsistent shapes found in embeddings for genre: {genre}")
                    raise ValueError(f"Inconsistent shapes found in embeddings for genre: {genre}")
            
        self.centroids = genre_centroids
        return genre_centroids


    def calculate_cosine_similarity(self):
        """
        Calculate the cosine similarity between different centroids and display the results in a table.
        :return: DataFrame containing normalized cosine similarity differences.
        """
        if not self.centroids:
            raise ValueError("No centroids available. Run `generate_embeddings()` and `load_embeddings()` first.")

        # Convert centroids to a numpy array
        centroids_array = np.array(list(self.centroids.values()))
        genre_labels = list(self.centroids.keys())

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(centroids_array)

        # Calculate differences (1 - cosine similarity)
        similarity_diff_matrix = 1 - similarity_matrix
        
        # Normalize the differences to a range of 0 to 1
        max_value = similarity_diff_matrix.max()
        similarity_diff_normalized = similarity_diff_matrix / max_value
        
        # Create a DataFrame for better visualization
        similarity_diff_df = pd.DataFrame(similarity_diff_normalized, index=genre_labels, columns=genre_labels)

        return similarity_diff_df

    def calculate_genre_variance(self):
        """
        Calculate the mean variance of embeddings for different genres.
        :return: DataFrame containing the mean variance for each genre.
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Run `generate_embeddings()` or 'load_embeddings()' first.")

        genre_variances = {}
        genres = np.unique(self.labels)

        for genre in genres:
            idx = np.where(np.array(self.labels) == genre)[0]
            genre_embeddings = np.array(self.embeddings)[idx]
            genre_variance = np.var(genre_embeddings, axis=0).mean()  # Calculate mean variance across all dimensions
            genre_variances[genre] = genre_variance

        # Create a DataFrame for better visualization
        variance_df = pd.DataFrame([genre_variances])
        variance_df.index = ["Mean Variance"]

        return variance_df
    

    def plot_embeddings(
        self, 
        method: str = "pca", 
        title: Optional[str] = "openl3",
        plot_dir: str = "results/plots",
        perplexity: int = 30
    ):
        """
        Visualize embeddings using PCA or t-SNE, coloring by genre.
        :param method: Dimensionality reduction method ('pca' or 'tsne').
        :param title: Title to be used at the beginning of the plot title and for generating the save path.
        :param plot_dir: Directory to save the plot.
        :param perplexity: Perplexity value for t-SNE.
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Run `generate_embeddings()` first.")

        embeddings_array = np.array(self.embeddings)
        centroids_array = np.array(list(self.centroids.values()))

        if method == "pca":
            reducer = PCA(n_components=2)
            reduced_emb = reducer.fit_transform(embeddings_array)
            reduced_centroids = reducer.fit_transform(np.array(list(self.centroids.values())))
            
        elif method == "tsne":
            combined_array = np.vstack((embeddings_array, centroids_array))
            n_samples = len(embeddings_array)
            perplexity = min(perplexity, max(1, n_samples - 1))
            print(f"Using perplexity={perplexity} for t-SNE")
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_combined = reducer.fit_transform(combined_array)
            reduced_emb = reduced_combined[:n_samples]
            reduced_centroids = reduced_combined[n_samples:]
            
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

        plot_title = f"{title} - Audio Embeddings Visualization ({method.upper()})"
        plt.title(plot_title)
        plt.tight_layout(pad=2.0)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Genre", loc='best')
        plt.grid()

        # Generate save path
        save_path = os.path.join(plot_dir, f"{title.replace(' ', '_')}_{method.lower()}.png")
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
      
    
    def plot_cosine_similarity(self, similarity_diff_df, title: Optional[str] = "Cosine Similarity", plot_dir: str = "results/plots"):
        """
        Plot the normalized cosine similarity differences between different centroids.
        :param similarity_diff_df: DataFrame containing the normalized cosine similarity differences.
        :param title: Title to be used for the plot title and for generating the save path.
        :param plot_dir: Directory to save the plot.
        """
        # Highlight the diagonal values in red
        mask = np.zeros_like(similarity_diff_df, dtype=bool)
        np.fill_diagonal(mask, True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_diff_df, annot=True, cmap='viridis', cbar_kws={'label': 'Normalized 1 - Cosine Similarity'}, mask=mask, linewidths=.5, linecolor='red')
        plt.title(f'{title} - Normalized Cosine Similarity Differences (1 - Cosine Similarity)')
        plt.tight_layout(pad=2.0)
        plt.xlabel('Genres')
        plt.ylabel('Genres')

        # Generate save path
        save_path = os.path.join(plot_dir, f"{title.replace(' ', '_')}_cosine_similarity.png")

        # Save or display the plot
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()