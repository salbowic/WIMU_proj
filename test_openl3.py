from openl3_embedding_generator import EmbeddingVisualizer
import os
import openl3
import sys

def print_instructions():
    instructions = """
    Usage: python test_openl3.py [options]
    
    Options:
    --generate-embeddings <input_repr> <embedding_size> <dataset_folder> <n_samples> 
        Generate embeddings with specified parameters (default: mel128 6144 gtzan_dataset/genres_original 100)
    --embedding-dir <embedding_dir>    
        Directory to save and/or load embeddings (default: results/embeddings). Embeddings are saved in /<embedding_dir>/class
    --plot <plot_dir> <plot_name> <plot_method> 
        Directory to save plots, name of the plot file, and plot method (tsne, pca, or both) (default: results/plots plot1 both)
    --help                             
        Show this help message and exit
    """
    print(instructions)

if __name__ == "__main__":
    # Parse command line arguments
    args = sys.argv[1:]
    if "--help" in args:
        print_instructions()
        sys.exit(0)

    # Default values
    input_repr = "mel128"
    embedding_size = 6144
    dataset_folder = "gtzan_dataset/genres_original"
    n_samples_per_genre = 100
    generate_embeddings = False
    plot=False
    embedding_dir = "results/embeddings/gtzan_embeddings_mel128_6144"
    plot_dir = "results/plots"
    plot_name = "plot1"
    plot_method = "both"

    try:
        for i in range(len(args)):
            if args[i] == "--generate-embeddings":
                generate_embeddings = True
                input_repr = args[i + 1]
                if input_repr not in ["linear", "mel128", "mel256"]:
                    raise ValueError(f"Invalid input representation: {input_repr}")
                embedding_size = int(args[i + 2])
                if embedding_size not in [512, 6144]:
                    raise ValueError(f"Invalid embedding size: {embedding_size}")
                dataset_folder = args[i + 3]
                n_samples_per_genre = int(args[i + 4])
            elif args[i] == "--embedding-dir":
                embedding_dir = args[i + 1]
            elif args[i] == "--plot":
                plot = True
                plot_dir = args[i + 1]
                plot_name = args[i + 2]
                plot_method = args[i + 3]
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        print_instructions()
        sys.exit(1)

    # Create output folder for results
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Generate embeddings for the specified number of samples per genre
    if generate_embeddings:
        # Initialize the visualizer with the dataset folder
        visualizer = EmbeddingVisualizer()
        model = openl3.models.load_audio_embedding_model(input_repr=input_repr, content_type="music", embedding_size=embedding_size)
        visualizer.set_dataset_folder(dataset_folder)
        visualizer.set_model(model)
        visualizer.generate_embeddings(num_samples_per_genre=n_samples_per_genre, emb_dir=embedding_dir)

    # Load embeddings from file
    if plot:
        visualizer = EmbeddingVisualizer()
        visualizer.load_embeddings(input_dir=embedding_dir)
        # Plot embeddings using t-SNE and/or PCA
        if plot_method.lower() == "both":
            visualizer.plot_embeddings(method="tsne", save_path=f"{plot_dir}/{plot_name}_tsne.png")
            visualizer.plot_embeddings(method="pca", save_path=f"{plot_dir}/{plot_name}_pca.png")
        else:
            visualizer.plot_embeddings(method=plot_method.lower(), save_path=f"{plot_dir}/{plot_name}_{plot_method.lower()}.png")