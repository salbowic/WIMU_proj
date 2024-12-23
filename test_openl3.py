import sys

def print_instructions():
    instructions = """
    Usage: python test_openl3.py [options]
    
    Options:
    --input-repr <input_repr>          Input representation (default: mel128, accepted values: linear, mel128, mel256)
    --embedding-size <embedding_size>  Embedding size (default: 6144, accepted values: 512, 6144)
    --dataset-folder <dataset_folder>  Path to the dataset folder (default: gtzan_dataset/genres_original)
    --n-samples <n_samples>            Number of samples per genre (default: 100)
    --generate-embeddings              Generate embeddings (default: False)
    --embedding-dir <embedding_dir>    Directory to save embeddings (default: results/embeddings)
    --plot-dir <plot_dir>              Directory to save plots (default: results/plots)
    --plot-name <plot_name>            Name of the plot file (default: plot1.png)
    --plot-method <plot_method>        Plot method (tsne, pca, or both) (default: both)
    --help                             Show this help message and exit
    """
    print(instructions)

if __name__ == "__main__":

    args = sys.argv[1:]
    if "--help" in args:
        print_instructions()
        quit()


    from openl3_gtzan import EmbeddingVisualizer
    import os
    import openl3
    
    # Default values
    input_repr = "mel128"
    embedding_size = 6144
    dataset_folder = "gtzan_dataset/genres_original"
    n_samples_per_genre = 100
    generate_embeddings = False
    embedding_dir = "results/embeddings"
    plot_dir = "results/plots"
    plot_name = "plot1.png"
    plot_method = "both"
    
    # Parse command line arguments
    try:
        for i in range(len(args)):
            if args[i] == "--input-repr":
                input_repr = args[i + 1]
                if input_repr not in ["linear", "mel128", "mel256"]:
                    raise ValueError(f"Invalid input representation: {input_repr}")
            elif args[i] == "--embedding-size":
                embedding_size = int(args[i + 1])
                if embedding_size not in [512, 6144]:
                    raise ValueError(f"Invalid embedding size: {embedding_size}")
            elif args[i] == "--dataset-folder":
                dataset_folder = args[i + 1]
            elif args[i] == "--n-samples":
                n_samples_per_genre = int(args[i + 1])
            elif args[i] == "--generate-embeddings":
                generate_embeddings = True
            elif args[i] == "--embedding-dir":
                embedding_dir = args[i + 1]
            elif args[i] == "--plot-dir":
                plot_dir = args[i + 1]
            elif args[i] == "--plot-name":
                plot_name = args[i + 1]
            elif args[i] == "--plot-method":
                plot_method = args[i + 1]
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
    if not generate_embeddings:
        visualizer = EmbeddingVisualizer()
        visualizer.load_embeddings(input_dir=embedding_dir)
    
    # Plot embeddings using t-SNE and/or PCA
    if plot_method.lower() == "both":
        visualizer.plot_embeddings(method="tsne", save_path=f"{plot_dir}/{plot_name}_tsne.png")
        visualizer.plot_embeddings(method="pca", save_path=f"{plot_dir}/{plot_name}_pca.png")
    else:
        visualizer.plot_embeddings(method=plot_method.lower(), save_path=f"{plot_dir}/{plot_name}")