# cython: language_level=3, boundscheck=False, wraparound=False

from cpython.object cimport PyObject, PyCallable_Check 
from libcpp.unordered_map cimport unordered_map
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libc.stdio cimport FILE, fopen, fclose, fwrite
from libc.errno cimport errno

import sys, time
from itertools import chain


cimport cython
from libc.math cimport log, fmax
import numpy as np
cimport numpy as cnp
import heapq

ctypedef long long i64
ctypedef int i32
ctypedef unsigned int u32
ctypedef unsigned long long u64
ctypedef float f32  

cdef object AUX, INTENS, negate
AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
                       "pretty","rather","somewhat","kinda","sorta","at","all"}
negate = {"no", "not", "never"}


cdef inline u64 pair_id_encode(u32 a, u32 b) nogil:
    return ((<u64>a) << 32) | (<u64>b)



cdef void save_corpus(const vector[u32]& o_encode, const vector[u32]& sizes, const char* path) except*:
    cdef FILE* fp = fopen(path, "wb")
    if fp == NULL:
        raise OSError(errno, "fail open for write")
    cdef u32 n_encode = <u32> o_encode.size()
    cdef u32 n_size   = <u32> sizes.size()
    
    if fwrite(&n_encode, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant write")
    if fwrite(&o_encode[0], sizeof(u32), n_encode, fp) != n_encode:
        raise OSError(errno, "cant write")

    if fwrite(&n_size, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant write")
    if fwrite(&sizes[0], sizeof(u32), n_size, fp) != n_size:
        raise OSError(errno, "cant write")

    if fclose(fp) != 0:
        raise OSError(errno, "cant close after write")




def iter_sentences(object s):
    cdef list out = []
    for i in s:
        if i == ".":
            yield out
            out.clear()
        else:
            out.append(i)

def first_pass(object ite, int top_k, int min_pair_count, object to_save_path):

    # map ca word2id de truy cap unigram va bigram nhanh
    cdef dict o_word2id = {}
    cdef list o_id2word = []
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] unigram = np.zeros(600000, dtype=np.uint32)  # Full vocab only 580k
    cdef unordered_map[u64, u32] bigram
    cdef vector[u32] o_encode
    cdef vector[u32] o_sizes
    o_encode.reserve(600000)

    cdef object sent, w, obj        #sents:sentence, w: word, obj: map cua word2id 
    cdef i32 prev_id = -1, n      #prev_id: prev_word's id, i index in array, n = length of sent
    cdef u32 i, id                      #id: word's id
    cdef u64 pair_id                 #encode pair_id  

    for sent in ite:
        n = len(sent)
        i = 0
        prev_id = -1
        o_sizes.push_back(n)
        while i < n:
            w = sent[i]
            obj = o_word2id.get(w, None)
            if obj is None:
                id = <u32> len(o_word2id)
                o_word2id[w] = id
                o_id2word.append(w)
            else:
                id = <u32> obj
            o_encode.push_back(id)
            unigram[id] += 1
            if prev_id != -1:
                if o_id2word[prev_id] in negate:
                    while i < n - 1 and (sent[i] in AUX or sent[i] in INTENS):
                        i += 1
                        w = sent[i]
                        obj = o_word2id.get(w, None)
                        if obj is None:
                            id = <u32> len(o_word2id)
                            o_id2word.append(w)
                            o_word2id[w] = id
                        else: 
                            id = <u32> obj
                        unigram[id] += 1
                        o_encode.push_back(id)
                pair_id = pair_id_encode(prev_id, id)
                bigram[pair_id] += 1       
            prev_id = id
            i += 1

    save_corpus(o_encode, o_sizes, str(to_save_path).encode("utf-8"))




    cdef u32 total_bi = 0
    cdef unordered_map[u64, u32].iterator it = bigram.begin()
    while it != bigram.end():
        total_bi += deref(it).second
        inc(it)

    it = bigram.begin()

    total_u = unigram.sum() 
    cdef u32 count
    cdef u32 c1_id, c2_id
    cdef list scored = []
    while it != bigram.end():
        pair_id = deref(it).first
        count   = deref(it).second
        if count < min_pair_count:
            it = inc(it)
            continue
        p12 = <double> count / <double> total_bi
        c1_id = <u32> (pair_id >> 32)
        c2_id = <u32> (pair_id) 
        if (o_id2word[c1_id] == "not"):
            scored.append((1000, pair_id, count))          #luon pass
            it = inc(it)
            continue

        p1 = <double> unigram[c1_id] / <double> total_u
        p2 = <double> unigram[c2_id] / <double> total_u

        de = p1 * p2
        if de <= 0:
            it = inc(it)
            continue
        pmi = fmax(0, log(p12 / de))
        if pmi < 3:
            it = inc(it)
            continue
        scored.append((float(pmi), pair_id, count)) 
        inc(it)
        
    top = heapq.nlargest(top_k, scored, key=lambda x: (x[0], x[2]))
    result = [(x[1], x[2]) for x in top]
  
    return unigram, result, o_id2word, o_word2id


def save_valid_encode(object iter, dict o_word2id, object to_save_path):
    cdef vector[u32] o_encode
    cdef vector[u32] o_sizes

    cdef u32 id 
    cdef object sent, w, obj
    cdef u32 n, i
    cdef u32 count
    for sent in iter:
        n = len(sent)
        count = 0
        for i in range(n):
            obj = o_word2id.get(sent[i], None)
            if obj is None: 
                continue
            id = <u32> obj
            count += 1
            o_encode.push_back(id)
        o_sizes.push_back(count)

    save_corpus(o_encode, o_sizes, str(to_save_path).encode("utf-8"))




  
def build_vocab(cnp.ndarray[cnp.uint32_t, ndim=1] unigram, 
                list o_id2word, dict o_word2id, int min_count):

    mask = unigram >= min_count 
    cand_idx = np.nonzero(mask)[0] 
    
    order = np.argsort(-unigram[cand_idx])
    cand_idx = cand_idx[order]
    
    cdef dict n_word2id = {}
    cdef list n_id2word = [None] * len(cand_idx) 
    
    cdef cnp.ndarray[cnp.int32_t, ndim=1] n_counts = np.zeros(mask.sum(), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] old2new = np.full(600000, -1, dtype=np.int32)

    for new_id, old_id in enumerate(cand_idx):
        old2new[old_id] = new_id
        n_word2id[o_id2word[old_id]] = new_id 
        n_id2word[new_id] = o_id2word[old_id]
        n_counts[new_id] = unigram[old_id]

    

    return old2new, n_word2id, n_id2word, n_counts

    
    





    



