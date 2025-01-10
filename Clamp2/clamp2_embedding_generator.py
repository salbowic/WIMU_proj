m3_compatible = True  # Set to True for M3 compatibility; set to False to retain all MIDI information during conversion.
import os
import numpy as np
import mido
import shutil
import random
import math
import subprocess
from multiprocessing import Pool

class Clamp2EmbeddingGenerator:
    def __init__(
        self,
        input_dir: str = None,
        emb_dir: str = None
        ):
        """
        Initialize the visualizer with the dataset folder path.
        :param dataset_folder: Path to the GTZAN dataset folder.
        """
        self.input_dir = input_dir
        self.emb_dir = emb_dir
        self.mtf_dir = None
        
    def set_input_dir(self, input_dir: str):
        """
        Set the input directory.
        :param input_dir: Path to the input directory.
        """
        self.input_dir = input_dir
    
    def get_input_dir(self):
        """
        Get the input directory.
        :return: Path to the input directory.
        """
        return self.input_dir
    
    def set_emb_dir(self, emb_dir: str):
        """
        Set the embeddings directory.
        :param emb_dir: Path to the embeddings directory.
        """
        self.emb_dir = emb_dir
        
    def get_emb_dir(self):
        """
        Get the embeddings directory.
        :return: Path to the embeddings directory.
        """
        return self.emb_dir
    
    def set_mtf_dir(self, mtf_dir: str):
        """
        Set the MTF directory.
        :param mtf_dir: Path to the MTF directory.
        """
        self.mtf_dir = mtf_dir
        
    def get_mtf_dir(self):
        """
        Get the MTF directory.
        :return: Path to the MTF directory.
        """
        return self.mtf_dir
    
    def _msg_to_str(self, msg):
        str_msg = ""
        for key, value in msg.dict().items():
            str_msg += " " + str(value)
        return str_msg.strip().encode('unicode_escape').decode('utf-8')


    def _load_midi(self, filename):
        # Load a MIDI file
        mid = mido.MidiFile(filename)
        msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

        # Traverse the MIDI file
        for msg in mid.merged_track:
            if m3_compatible:
                if msg.is_meta:
                    if msg.type in ["text", "copyright", "track_name", "instrument_name", 
                                    "lyrics", "marker", "cue_marker", "device_name", "sequencer_specific"]:
                        continue
                else:
                    if msg.type in ["sysex"]:
                        continue
            str_msg = self._msg_to_str(msg)
            msg_list.append(str_msg)
        
        return "\n".join(msg_list)


    def _midi2mtf(self, file_list):
        for file in file_list:
            filename = file.split('/')[-1]
            genre_dir = os.path.basename(os.path.dirname(file))
            base_dir = os.path.dirname(os.path.dirname(file))  # Get the base directory (e.g., dataset/songs)
            mtf_dir = os.path.join(base_dir + '_mtf', genre_dir)
            os.makedirs(mtf_dir, exist_ok=True)
            
            try:
                output = self._load_midi(file)

                if output == '':
                    print(f"Empty output for file: {file}")
                    with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                        f.write(file + '\n')
                    continue
                else:
                    mtf_file_path = os.path.join(mtf_dir, ".".join(filename.split(".")[:-1]) + '.mtf')
                    with open(mtf_file_path, 'w', encoding='utf-8') as f:
                        f.write(output)
            except Exception as e:
                print(f"Error processing file: {file}, Error: {str(e)}")
                with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(file + " " + str(e) + '\n')
                pass
        
            
    def _convert_midi2mtf(self):
        file_list = []
        os.makedirs("logs", exist_ok=True)

        # Traverse the specified folder for MIDI files
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith(".mid") and not file.endswith(".midi"):
                    continue
                filename = os.path.join(root, file).replace("\\", "/")
                file_list.append(filename)

        # Prepare for multiprocessing
        file_lists = []
        random.shuffle(file_list)
        for i in range(os.cpu_count()):
            start_idx = int(math.floor(i * len(file_list) / os.cpu_count()))
            end_idx = int(math.floor((i + 1) * len(file_list) / os.cpu_count()))
            file_lists.append(file_list[start_idx:end_idx])

        pool = Pool(processes=os.cpu_count())
        pool.map(self._midi2mtf, file_lists)


    def _run_extract_m3(self, mtf_dir, emb_dir):
        """
        Run the extract_m3.py script with the specified directories.
        :param mtf_dir: Path to the MTF directory.
        :param emb_dir: Path to the embedding directory.
        """
        command = f"python Clamp2/clamp2_github/code/extract_m3.py {mtf_dir} {emb_dir}"
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")
    

    def convert_emb_to_npz(self):
        # Iterate over all files in the embedding directory
        for filename in os.listdir(self.emb_dir):
            if filename.endswith('.npy'):
                # Load the .npy file
                data = np.load(os.path.join(self.emb_dir, filename))
                
                # Define the output file path
                output_file = os.path.join(self.emb_dir, filename.replace('.npy', '.npz'))
                
                # Save the data as a .npz file
                np.savez(output_file, data=data)
                
                # Remove the original .npy file
                os.remove(os.path.join(self.emb_dir, filename))
                
    
    def generate_embeddings_for_dataset(self):
        """
        Generate embeddings for a dataset with genre subdirectories.
        :param dataset_dir: Path to the dataset directory containing genre subdirectories.
        :param embeddings_dir: Path to the directory where embeddings will be stored.
        """
        input_dir = self.input_dir
        emb_dir = self.emb_dir
        for genre in os.listdir(input_dir):
            genre_input_dir = os.path.join(input_dir, genre)
            genre_emb_dir = os.path.join(emb_dir, genre)

            if os.path.isdir(genre_input_dir):
                os.makedirs(genre_emb_dir, exist_ok=True)
                self.set_input_dir(genre_input_dir)
                self.set_emb_dir(os.path.join(emb_dir, genre))
                self.set_mtf_dir(os.path.join(input_dir + '_mtf'))

                self._convert_midi2mtf()
                
            mtf_dir = (os.path.join(self.mtf_dir, genre))
            
            num_files = len([f for f in os.listdir(genre_input_dir) if f.endswith('.mid') or f.endswith('.midi')])
            print(f"Clamp2 m3 processing {num_files} files from: {genre}")
            # Run the extract_m3.py script
            self._run_extract_m3(mtf_dir, self.emb_dir)
            
            #self.convert_emb_to_npz() # optional: convert .npy to .npz
            
            self._delete_mtf_directory()
            
            
    def _delete_mtf_directory(self):
        base_dir = os.path.dirname(self.input_dir)
        mtf_dir = base_dir + '_mtf'
        if os.path.exists(mtf_dir):
            shutil.rmtree(mtf_dir)
 