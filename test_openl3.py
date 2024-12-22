from openl3_gtzan import EmbeddingVisualizer
import os

def main():
    for i, file in enumerate(sampled_files):
        file_path = os.path.join(genre_folder, file)
        output_dir = f'results/embeddings/{genre}'
        os.makedirs(output_dir, exist_ok=True)
        

        EmbeddingVisualizer.generate_embeddings()

        # Update progress
        sys.stdout.write(f"\rProcessed {i + 1}/{total_files} files from genre: {genre}")
        sys.stdout.flush()

        print()



if __name__ == "__main__":
    genres = [genre for genre in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, genre))]
    print(f"Found genres: {genres}")
    for genre in genres:
    genre_folder = os.path.join(dataset_folder, genre)
    files = [file for file in os.listdir(genre_folder) if file.endswith(('.wav', '.ogg', '.flac'))]
        # Randomly sample files from each genre
    random.shuffle(files)
    sampled_files = files[:num_samples_per_genre]
    total_files = len(sampled_files)
                
    # Initialize the visualizer with the dataset folder
    dataset_folder = "gtzan_dataset/genres_original"
    visualizer = EmbeddingVisualizer(dataset_folder)

    # Generate embeddings for 10 random songs per genre
    visualizer.generate_embeddings(num_samples_per_genre=10)

    # Load embeddings from the saved .npz files
    visualizer.load_embeddings(input_dir='results/embeddings')

    # Create output folder for results
    os.makedirs("results", exist_ok=True)

    # Plot embeddings using t-SNE and PCA
    visualizer.plot_embeddings(method="tsne", save_path="results/gtzan_tsne_result.png")
    visualizer.plot_embeddings(method="pca", save_path="results/gtzan_pca_result.png")