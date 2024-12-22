from openl3_gtzan import EmbeddingVisualizer
import os

if __name__ == "__main__":           
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