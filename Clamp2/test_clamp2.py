import argparse
import os
from clamp2_embedding_generator import Clamp2EmbeddingGenerator
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for a dataset with genre subdirectories.')
    parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory containing genre subdirectories')
    parser.add_argument('embeddings_dir', type=str, help='Path to the directory where embeddings will be stored')
    parser.add_argument('--m3_compatible', action='store_true', help='Set for M3 compatibility')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    embeddings_dir = os.path.abspath(args.embeddings_dir)
    m3_compatible = args.m3_compatible

    # Ensure the embeddings directory exists
    os.makedirs(embeddings_dir, exist_ok=True)
    
    clamp2_emb_generator = Clamp2EmbeddingGenerator()
    clamp2_emb_generator.set_input_dir(dataset_dir)
    clamp2_emb_generator.set_emb_dir(embeddings_dir)
    clamp2_emb_generator.m3_compatible = m3_compatible
    
    start_time = time.time()
    clamp2_emb_generator.generate_embeddings_for_dataset()
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Convert the elapsed time to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print the elapsed time in H:M:S format
    print(f"Embeddings generation completed in {int(hours)}:{int(minutes)}:{int(seconds)}")