import torch 
import numpy as np 
import os 

from data_pipe import iter_wiki_sentences

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_corpus_dir = os.path.join(base_dir, "..", "data", "o_corpus.bin")
    to_save_corpus_dir = os.path.join(base_dir, "..", "data", "n_corpus.bin")
    
    

