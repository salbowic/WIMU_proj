import argparse
import os
import sys
import time

# Add the OpenL3 and Clamp2 directories to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenL3')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Clamp2')))

from openl3_embedding_generator import EmbeddingVisualizer
from clamp2_embedding_generator import Clamp2EmbeddingGenerator

def print_instructions():
    instructions = """
    Usage: python main.py [options]
    
    Options:
    --dataset <dataset_dir> 
        Path to the dataset directory containing genre subdirectories.
    --emb-methods <methods> 
        Embedding methods to use (clamp2, openl3, or both).
    --emb-dir <embedding_dir> 
        Directory to save embeddings (with added _clamp2 or _openl3).
    --plot [<plot_title>] [<plot_method>] [<plot_dir>] 
        Title of the plot, plot method (tsne, pca, or both), and directory to save plots.
    --calc-metrics [<cos_sim_filename>] [<cos_sim_plot_title>] [<cos_sim_plot_dir>] [<variance_path>]
        Calculate cosine similarity differences between centroids, save the DataFrame to the specified filename, save the plot with the specified title in the specified directory, and calculate variance of different genres and save the DataFrame to the specified path and filename.
    --help                             
        Show this help message and exit.
    """
    print(instructions)

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings and perform analysis.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory containing genre subdirectories')
    parser.add_argument('--emb-methods', type=str, required=True, choices=['clamp2', 'openl3', 'both'], help='Embedding methods to use (clamp2, openl3, or both)')
    parser.add_argument('--emb-dir', type=str, required=True, help='Directory to save embeddings (with added _clamp2 or _openl3)')
    parser.add_argument('--plot', nargs=3, help='Plot title, plot method (tsne, pca, or both), and directory to save plots')
    parser.add_argument('--calc-metrics', nargs=4, help='Cosine similarity filename, plot title, plot directory, and variance path')
    args = parser.parse_args()

    dataset_dir = args.dataset
    emb_methods = args.emb_methods
    emb_dir = args.emb_dir
    plot_args = args.plot
    calc_metrics_args = args.calc_metrics

    if emb_methods in ['openl3', 'both']:
        openl3_emb_dir = f"{emb_dir}_openl3"
        os.makedirs(openl3_emb_dir, exist_ok=True)
        
        # Initialize the visualizer with the dataset folder
        visualizer = EmbeddingVisualizer()
        model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=6144)
        visualizer.set_dataset_folder(dataset_dir)
        visualizer.set_model(model)
        
        start_time = time.time()
        visualizer.generate_embeddings(emb_dir=openl3_emb_dir)
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Convert the elapsed time to hours, minutes, and seconds
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Print the elapsed time in H:M:S format
        print(f"OpenL3 embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")

        if plot_args:
            plot_title, plot_method, plot_dir = plot_args
            os.makedirs(plot_dir, exist_ok=True)
            visualizer.load_embeddings(input_dir=openl3_emb_dir)
            if plot_method.lower() == "both":
                visualizer.plot_embeddings(method="tsne", title=plot_title, plot_dir=plot_dir)
                visualizer.plot_embeddings(method="pca", title=plot_title, plot_dir=plot_dir)
            else:
                visualizer.plot_embeddings(method=plot_method.lower(), title=plot_title, plot_dir=plot_dir)

        if calc_metrics_args:
            cos_sim_filename, cos_sim_plot_title, cos_sim_plot_dir, variance_path = calc_metrics_args
            os.makedirs(cos_sim_plot_dir, exist_ok=True)
            visualizer.load_embeddings(input_dir=openl3_emb_dir)
            similarity_diff_df = visualizer.calculate_cosine_similarity()
            if not cos_sim_filename.endswith('.csv'):
                cos_sim_filename += '.csv'
            similarity_diff_df.to_csv(cos_sim_filename, sep=';')
            print(f"Cosine similarity DataFrame saved to {cos_sim_filename}")
            visualizer.plot_cosine_similarity(similarity_diff_df, title=cos_sim_plot_title, plot_dir=cos_sim_plot_dir)
            variance_df = visualizer.calculate_genre_variance()
            os.makedirs(os.path.dirname(variance_path), exist_ok=True)
            variance_df.to_csv(variance_path, sep=';')
            print(f"Variance DataFrame saved to {variance_path}")

    if emb_methods in ['clamp2', 'both']:
        clamp2_emb_dir = f"{emb_dir}_clamp2"
        os.makedirs(clamp2_emb_dir, exist_ok=True)
        
        # Initialize the Clamp2 embedding generator
        clamp2_generator = Clamp2EmbeddingGenerator(dataset_dir, clamp2_emb_dir, m3_compatible=True)
        
        start_time = time.time()
        clamp2_generator.generate_embeddings()
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Convert the elapsed time to hours, minutes, and seconds
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Print the elapsed time in H:M:S format
        print(f"Clamp2 embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")

        if plot_args:
            plot_title, plot_method, plot_dir = plot_args
            os.makedirs(plot_dir, exist_ok=True)
            visualizer.load_embeddings(input_dir=clamp2_emb_dir)
            if plot_method.lower() == "both":
                visualizer.plot_embeddings(method="tsne", title=plot_title, plot_dir=plot_dir)
                visualizer.plot_embeddings(method="pca", title=plot_title, plot_dir=plot_dir)
            else:
                visualizer.plot_embeddings(method=plot_method.lower(), title=plot_title, plot_dir=plot_dir)

        if calc_metrics_args:
            cos_sim_filename, cos_sim_plot_title, cos_sim_plot_dir, variance_path = calc_metrics_args
            os.makedirs(cos_sim_plot_dir, exist_ok=True)
            visualizer.load_embeddings(input_dir=clamp2_emb_dir)
            similarity_diff_df = visualizer.calculate_cosine_similarity()
            if not cos_sim_filename.endswith('.csv'):
                cos_sim_filename += '.csv'
            similarity_diff_df.to_csv(cos_sim_filename, sep=';')
            print(f"Cosine similarity DataFrame saved to {cos_sim_filename}")
            visualizer.plot_cosine_similarity(similarity_diff_df, title=cos_sim_plot_title, plot_dir=cos_sim_plot_dir)
            variance_df = visualizer.calculate_genre_variance()
            os.makedirs(os.path.dirname(variance_path), exist_ok=True)
            variance_df.to_csv(variance_path, sep=';')
            print(f"Variance DataFrame saved to {variance_path}")

if __name__ == "__main__":
    main()