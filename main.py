import argparse
import os
import sys
import time
import gc
import openl3
import contextlib

# Add the OpenL3 and Clamp2 directories to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenL3')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Clamp2')))

from openl3_embedding_generator import EmbeddingVisualizer
from clamp2_embedding_generator import Clamp2EmbeddingGenerator

def print_instructions():
    instructions = """
    Usage: python main.py [options] [--params-file <file>]
    
    Options:
    --dataset <dataset_dir> 
        Path to the dataset directory containing genre subdirectories.
    --emb-methods <methods> 
        Embedding methods to use (clamp2, openl3, or both).
    --emb-dir <embedding_dir> 
        Directory to save embeddings (with added _clamp2 or _openl3).
    --input-repr <input_repr> 
        Input representation for OpenL3 (default: mel256).
    --embedding-size <embedding_size> 
        Embedding size for OpenL3 (default: 6144).
    --plot [<plot_title>] [<plot_method>] [<plot_dir>] 
        Title of the plot, plot method (tsne, pca, or both), and directory to save plots.
    --calc-metrics [<cos_sim_plot_dir>] [<variance_dir>]
        Calculate cosine similarity differences between centroids, save the plot with the specified title in the specified directory, and calculate variance of different genres and save the DataFrame to the specified path and filename.
    --params-file <file>
        Path to a file containing parameters.
    --help                             
        Show this help message and exit.
    """
    print(instructions)

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings and perform analysis.')
    parser.add_argument('--dataset', type=str, help='Path to the dataset directory containing genre subdirectories')
    parser.add_argument('--emb-methods', type=str, choices=['clamp2', 'openl3', 'both'], help='Embedding methods to use (clamp2, openl3, or both)')
    parser.add_argument('--emb-dir', type=str, help='Directory to save embeddings (with added _clamp2 or _openl3)')
    parser.add_argument('--input-repr', type=str, default='mel256', help='Input representation for OpenL3 (default: mel256)')
    parser.add_argument('--embedding-size', type=int, default=6144, help='Embedding size for OpenL3 (default: 6144)')
    parser.add_argument('--plot', nargs=3, help='Plot title, plot method (tsne, pca, or both), and directory to save plots')
    parser.add_argument('--calc-metrics', nargs=2, help='Cosine similarity plot directory and variance directory')
    parser.add_argument('--params-file', type=str, help='Path to a file containing parameters')
    args, unknown = parser.parse_known_args()

    if args.params_file:
        with open(args.params_file, 'r') as f:
            file_args = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    file_args.extend(line.split())
        args = parser.parse_args(file_args + unknown)

    dataset_dir = args.dataset
    emb_methods = args.emb_methods
    emb_dir = args.emb_dir
    input_repr = args.input_repr
    embedding_size = args.embedding_size
    plot_args = args.plot
    calc_metrics_args = args.calc_metrics

    openl3_elapsed_time = None
    clamp2_elapsed_time = None

    if emb_methods in ['openl3', 'both']:
        openl3_emb_dir = f"{emb_dir}_openl3_{input_repr}_{embedding_size}"
        os.makedirs(openl3_emb_dir, exist_ok=True)
        
        # Initialize the visualizer with the dataset folder
        visualizer = EmbeddingVisualizer()
        model = openl3.models.load_audio_embedding_model(input_repr=input_repr, content_type="music", embedding_size=embedding_size)
        visualizer.set_dataset_folder(dataset_dir)
        visualizer.set_model(model)
        
        start_time = time.time()
        visualizer.generate_embeddings(emb_dir=openl3_emb_dir)
        end_time = time.time()
        
        # Calculate the elapsed time
        openl3_elapsed_time = end_time - start_time

        if plot_args:
            plot_title, plot_method, plot_dir = plot_args
            plot_dir_openl3 = os.path.join(plot_dir, "openl3")
            os.makedirs(plot_dir_openl3, exist_ok=True)
            visualizer.load_embeddings(input_dir=openl3_emb_dir)
            if plot_method.lower() == "both":
                visualizer.plot_embeddings(method="tsne", title=f"{plot_title}_openl3_{input_repr}_{embedding_size}", plot_dir=plot_dir_openl3)
                visualizer.plot_embeddings(method="pca", title=f"{plot_title}_openl3_{input_repr}_{embedding_size}", plot_dir=plot_dir_openl3)
            else:
                visualizer.plot_embeddings(method=plot_method.lower(), title=f"{plot_title}_openl3_{input_repr}_{embedding_size}", plot_dir=plot_dir_openl3)

        if calc_metrics_args:
            cos_sim_plot_dir, variance_dir = calc_metrics_args
            cos_sim_plot_dir_openl3 = os.path.join(cos_sim_plot_dir, "openl3")
            cos_sim_plot_title_openl3 = f"openl3_{input_repr}_{embedding_size}"
            variance_path_openl3 = f"{variance_dir}/variance_openl3_{input_repr}_{embedding_size}.csv"
            os.makedirs(cos_sim_plot_dir_openl3, exist_ok=True)
            if not plot_args:
                visualizer = EmbeddingVisualizer()
                visualizer.load_embeddings(input_dir=openl3_emb_dir)
            similarity_diff_df = visualizer.calculate_cosine_similarity()
            visualizer.plot_cosine_similarity(similarity_diff_df, title=cos_sim_plot_title_openl3, plot_dir=cos_sim_plot_dir_openl3)
            variance_df = visualizer.calculate_genre_variance()
            os.makedirs(os.path.dirname(variance_path_openl3), exist_ok=True)
            variance_df.to_csv(variance_path_openl3, sep=';')

        # Delete the visualizer and model to free up memory
        del visualizer
        del model
        gc.collect()

    if emb_methods in ['clamp2', 'both']:
        clamp2_emb_dir = f"{emb_dir}_clamp2"
        os.makedirs(clamp2_emb_dir, exist_ok=True)
        
        # Initialize the Clamp2 embedding generator
        clamp2_generator = Clamp2EmbeddingGenerator(dataset_dir, clamp2_emb_dir)
        
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            start_time = time.time()
            clamp2_generator.generate_embeddings_for_dataset()
            end_time = time.time()
        
        # Delete the clamp2_generator to free up memory
        del clamp2_generator
        gc.collect()
        
        # Calculate the elapsed time
        clamp2_elapsed_time = end_time - start_time

        if plot_args:
            plot_title, plot_method, plot_dir = plot_args
            plot_dir_clamp2 = os.path.join(plot_dir, "clamp2")
            os.makedirs(plot_dir_clamp2, exist_ok=True)
            # Initialize the visualizer with the dataset folder
            visualizer = EmbeddingVisualizer()
            visualizer.load_embeddings(input_dir=clamp2_emb_dir)
            if plot_method.lower() == "both":
                visualizer.plot_embeddings(method="tsne", title=f"{plot_title}_clamp2", plot_dir=plot_dir_clamp2)
                visualizer.plot_embeddings(method="pca", title=f"{plot_title}_clamp2", plot_dir=plot_dir_clamp2)
            else:
                visualizer.plot_embeddings(method=plot_method.lower(), title=f"{plot_title}_clamp2", plot_dir=plot_dir_clamp2)

        if calc_metrics_args:
            cos_sim_plot_dir, variance_dir = calc_metrics_args
            cos_sim_plot_dir_clamp2 = os.path.join(cos_sim_plot_dir, "clamp2")
            cos_sim_plot_title_clamp2 = "clamp2"
            variance_path_clamp2 = f"{variance_dir}/variance_clamp2.csv"
            os.makedirs(cos_sim_plot_dir_clamp2, exist_ok=True)
            if not plot_args:
                visualizer = EmbeddingVisualizer()
                visualizer.load_embeddings(input_dir=clamp2_emb_dir)
            similarity_diff_df = visualizer.calculate_cosine_similarity()
            visualizer.plot_cosine_similarity(similarity_diff_df, title=cos_sim_plot_title_clamp2, plot_dir=cos_sim_plot_dir_clamp2)
            variance_df = visualizer.calculate_genre_variance()
            os.makedirs(os.path.dirname(variance_path_clamp2), exist_ok=True)
            variance_df.to_csv(variance_path_clamp2, sep=';')

        

    # Print elapsed times at the end
    if openl3_elapsed_time is not None:
        hours, rem = divmod(openl3_elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"OpenL3 embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")

    if clamp2_elapsed_time is not None:
        hours, rem = divmod(clamp2_elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Clamp2 embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")

if __name__ == "__main__":
    main()