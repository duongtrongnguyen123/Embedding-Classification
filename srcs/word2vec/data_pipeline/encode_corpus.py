import torch 
import numpy as np 
import os 

from _encode_corpus import encode_corpus

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_vocab_dir = os.path.join(base_dir, "..", "..", "data", "vocab.pt")
    saved_train_corpus_dir = os.path.join(base_dir, "..", "..", "data", "o_train_corpus.bin")
    saved_valid_corpus_dir = os.path.join(base_dir, "..", "..", "data", "o_valid_corpus.bin")
    to_save_train_corpus_dir = os.path.join(base_dir, "..", "..", "data", "n_train_corpus.bin")
    to_save_valid_corpus_dir = os.path.join(base_dir, "..", "..", "data", "n_valid_corpus.bin")
    vocab = torch.load(saved_vocab_dir)
    old2new = vocab["old2new"]
    old2new_for_pair = vocab["old2new_for_pair"]
    skip_id = vocab["skip_id"]
    negate_id = vocab["negate_id"]
    id2word = vocab["id2word"]
    word2id = vocab["word2id"]
    o_word2id = vocab["o_word2id"]
    keep_probs = vocab["keep_probs"]
    encode_corpus(old2new, negate_id, skip_id, old2new_for_pair, saved_train_corpus_dir, to_save_train_corpus_dir)
    encode_corpus(old2new, negate_id, skip_id, old2new_for_pair, saved_valid_corpus_dir, to_save_valid_corpus_dir)
