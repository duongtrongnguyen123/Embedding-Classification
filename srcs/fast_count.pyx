from libcpp.unordered_map cimport unordered_map
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc

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

cdef object AUX, INTENS, negate, BAD_PART
AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
                       "pretty","rather","somewhat","kinda","sorta","at","all"}
negate = {"no", "not", "never"}
BAD_PART = {"not", "to"}


cdef inline u64 pair_id_encode(u32 a, u32 b) nogil:
    return ((<u64>a) << 32) | (<u64>b)

def iter_sentences(object s):
    cdef list out = []
    for i in s:
        if i == ".":
            yield out
            out.clear()
        else:
            out.append(i)

def first_pass(object iter, int top_k, int min_pair_count):
    # map ca word2id de truy cap unigram va bigram nhanh 
    cdef dict o_word2id = {}
    cdef list o_id2word = []
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] unigram = np.zeros(600000, dtype=np.uint32)
    cdef unordered_map[u64, u32] bigram
    
    cdef object sents, w, tok, obj
    cdef i32 prev_idx = 0, i, n
    cdef u32 idx
    cdef u64 pair_idx

    for sents in iter:
        n = len(sents)
        i = 0
        prev_idx = -1
        while i < len(sents):
            w = sents[i]
            obj = o_word2id.get(w, None)
            if obj is None:
                idx = <u32> len(o_word2id)
                o_word2id[w] = idx
                o_id2word.append(w)
            else:
                idx = <u32> obj
            unigram[idx] += 1
            if prev_idx != -1:
                if o_id2word[prev_idx] in negate:
                    while i < n - 1 and (sents[i] in AUX or sents[i] in INTENS):
                        i += 1
                        w = sents[i]
                        obj = o_word2id.get(w, None)
                        if obj is None:
                            idx = <u32> len(o_word2id)
                            o_id2word.append(w)
                            o_word2id[w] = idx
                        else: 
                            idx = <u32> obj
                    
                pair_idx = pair_id_encode(prev_idx, idx)
                bigram[pair_idx] += 1       
            prev_idx = idx
            i += 1
    
    cdef u32 total_bi = 0
    cdef unordered_map[u64, u32].iterator it = bigram.begin()
    while it != bigram.end():
        total_bi += deref(it).second
        inc(it)

    it = bigram.begin()

    total_u = unigram.sum() 
    cdef u32 count
    cdef u32 c1_idx, c2_idx
    cdef list scored = []
    while it != bigram.end():
        pair_idx = deref(it).first
        count    = deref(it).second
        if count < min_pair_count:
            it = inc(it)
            continue
        p12 = count / total_bi
        c1_idx = <u32> (pair_idx >> 32)
        c2_idx = <u32> (pair_idx) 
        p1 = unigram[c1_idx] / total_u
        p2 = unigram[c2_idx] / total_u

        de = p1 * p2
        if de <= 0:
            continue
        pmi = fmax(0, log(p12 / de)) 
        scored.append((float(pmi), pair_idx, count)) 
        inc(it)
        
    top = heapq.nlargest(top_k, scored, key=lambda x: (x[0], x[2]))
    result = [x[1] for x in top]


    return unigram, result, o_id2word, o_word2id

  
def build_vocab(cnp.ndarray[cnp.uint32_t, ndim=1] unigram, list scored, list o_id2word, dict o_word2id, int min_count):

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

    
if __name__ == "__main__":
    

    """
    train_iter = iter_sentences(str) 
    unigram, scored, o_id2word, o_word2id = first_pass(train_iter, top_k=5000, min_pair_count=0)
    
    
    old2new, word2id, id2word, counts = build_vocab(unigram, scored, o_id2word, o_word2id, min_count=22)
    """  
 
        





    



