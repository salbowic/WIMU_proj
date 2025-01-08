from openl3_embedding_generator import EmbeddingVisualizer
import os
import openl3
import sys
import time

def print_instructions():
    instructions = """
    Usage: python test_openl3.py [options]
    
    Options:
    --generate-embeddings <input_repr> <embedding_size> <dataset_folder> <n_samples> 
        Generate embeddings with specified parameters (default: mel128 6144 gtzan_dataset/genres_original 100)
    --embedding-dir <embedding_dir>    
        Directory to save and/or load embeddings (default: results/embeddings). Embeddings are saved in /<embedding_dir>/class
    --plot <plot_title> <plot_method> <plot_dir>
        Title of the plot,  plot method (tsne, pca, or both) and directory to save plots (default: plot1 both results/plots)
    --calc-metrics [<cos_sim_filename>] [<cos_sim_plot_title>] [<cos_sim_plot_dir>] [<variance_path>]
        Calculate cosine similarity differences between centroids, save the DataFrame to the specified filename, save the plot with the specified title in the specified directory, and calculate variance of different genres and save the DataFrame to the specified path and filename (default: results/openl3_cos_sim.csv Cosine Similarity results/plots results/variance/genre_variance.csv)
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
    input_repr = "mel256"
    embedding_size = 6144
    dataset_folder = "gtzan_dataset/genres_original"
    n_samples_per_genre = None
    generate_embeddings = False
    plot = False
    embedding_dir = "results/embeddings/gtzan_embeddings_mel128_6144"
    plot_dir = "results/plots"
    plot_name = "plot1"
    plot_method = "both"
    calc_metrics = False
    cos_sim_filename = "cosine_similarity.csv"
    cos_sim_plot_title = "cosine_similarity"
    cos_sim_plot_dir = "results/plots"
    variance_path = "results/variance/genre_variance.csv"

    try:
        for i in range(len(args)):
            if args[i] == "--generate-embeddings":
                generate_embeddings = True
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    if input_repr not in ["linear", "mel128", "mel256"]:
                        raise ValueError(f"Invalid input representation: {input_repr}")
                    input_repr = args[i + 1]
                if i + 2 < len(args) and not args[i + 2].startswith("--"):
                    if embedding_size not in [512, 6144]:
                        raise ValueError(f"Invalid embedding size: {embedding_size}")
                    embedding_size = int(args[i + 2])
                if i + 3 < len(args) and not args[i + 3].startswith("--"):
                    dataset_folder = args[i + 3]
                if i + 4 < len(args) and not args[i + 4].startswith("--"):
                    n_samples_per_genre = int(args[i + 4])
            elif args[i] == "--embedding-dir":
                embedding_dir = args[i + 1]
            elif args[i] == "--plot":
                plot = True
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    plot_title = args[i + 1]
                if i + 2 < len(args) and not args[i + 2].startswith("--"):
                    plot_method = args[i + 2]
                if i + 3 < len(args) and not args[i + 3].startswith("--"):
                    plot_dir = args[i + 3]
                    
            elif args[i] == "--calc-metrics":
                calc_metrics = True
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    cos_sim_filename = args[i + 1]
                if i + 2 < len(args) and not args[i + 2].startswith("--"):
                    cos_sim_plot_title = args[i + 2]
                if i + 3 < len(args) and not args[i + 3].startswith("--"):
                    cos_sim_plot_dir = args[i + 3]
                if i + 4 < len(args) and not args[i + 4].startswith("--"):
                    variance_path = args[i + 4]

    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        print_instructions()
        sys.exit(1)
        
    # Ensure embedding_dir is an absolute path
    embedding_dir = os.path.abspath(embedding_dir)
    plot_dir = os.path.abspath(plot_dir)
    cos_sim_plot_dir = os.path.abspath(cos_sim_plot_dir)
    variance_path = os.path.abspath(variance_path)

    # Generate embeddings for the specified number of samples per genre
    if generate_embeddings:
        # Create output folder for results
        os.makedirs(embedding_dir, exist_ok=True)
        # Initialize the visualizer with the dataset folder
        visualizer = EmbeddingVisualizer()
        model = openl3.models.load_audio_embedding_model(input_repr=input_repr, content_type="music", embedding_size=embedding_size)
        visualizer.set_dataset_folder(dataset_folder)
        visualizer.set_model(model)
        
        start_time = time.time()
        visualizer.generate_embeddings(num_samples_per_genre=n_samples_per_genre, emb_dir=embedding_dir)
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Convert the elapsed time to hours, minutes, and seconds
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Print the elapsed time in H:M:S format
        print(f"Embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")

    # Load embeddings from file
    if plot:
        os.makedirs(plot_dir, exist_ok=True)
        visualizer = EmbeddingVisualizer()
        visualizer.load_embeddings(input_dir=embedding_dir)
        
        # Plot embeddings using t-SNE and/or PCA
        if plot_method.lower() == "both":
            visualizer.plot_embeddings(method="tsne", title=plot_title, plot_dir=plot_dir)
            visualizer.plot_embeddings(method="pca", title=plot_title, plot_dir=plot_dir)
        else:
            visualizer.plot_embeddings(method=plot_method.lower(), title=plot_title, plot_dir=plot_dir)
    
    # Calculate and display cosine similarity between centroids
    if calc_metrics:
        os.makedirs(cos_sim_plot_dir, exist_ok=True)
        if not plot:
            visualizer = EmbeddingVisualizer()
            visualizer.load_embeddings(input_dir=embedding_dir)
            
        similarity_df = visualizer.calculate_cosine_similarity()
        if cos_sim_filename:
            if not cos_sim_filename.endswith('.csv'):
                cos_sim_filename += '.csv'
            similarity_df.to_csv(cos_sim_filename, sep=';')
            print(f"Cosine similarity DataFrame saved to {cos_sim_filename}")
        
        if cos_sim_plot_title:
            visualizer.plot_cosine_similarity(similarity_df, title=cos_sim_plot_title, plot_dir=cos_sim_plot_dir)
        
        # Calculate and save the variance of different genres
        variance_df = visualizer.calculate_genre_variance()
        
        # Ensure the directory for the variance path exists
        variance_dir = os.path.dirname(variance_path)
        os.makedirs(variance_dir, exist_ok=True)
    
        variance_df.to_csv(variance_path, sep=';')
        print(f"Variance DataFrame saved to {variance_path}")
        print(variance_df)
        