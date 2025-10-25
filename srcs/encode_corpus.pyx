from libc.stdint cimport uint32_t, uint64_t, int32_t
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.utility cimport pair as cpp_pair
from libc.stdio cimport FILE, fopen, fclose, fread, fwrite
from cython.operator cimport dereference as deref, preincrement as inc
from libc.errno cimport errno
from libcpp cimport bool


cimport numpy as cnp
import numpy as np

ctypedef uint32_t u32
ctypedef uint64_t u64
ctypedef int32_t  i32

cdef inline u64 encode_pair_id(u32 a, u32 b):
    return (<u64>a << 32) | (<u64>b)

cdef unordered_map[u64, u32] build_map(dict o2nfp):
    cdef unordered_map[u64, u32] res
    cdef u64 a
    cdef u32 b
    for o, n in o2nfp.items():
        a = <u64> o
        b = <u32> n
        res[a] = b

    return res

cdef unordered_set[u32] build_set(set skip_id):
    cdef unordered_set[u32] res
    for o in skip_id:
        res.insert(o)

    return res


cdef void load_corpus(vector[u32]& o_encode,                 #old encoded
                 vector[u32]& o_sizes,                    #old sents size
                 const char* path) except*:
    cdef FILE* fp = fopen(path, "rb")
    if fp == NULL:
        raise OSError(errno, "fail open for read")
    cdef u32 n_encode, n_size
    
    if fread(&n_encode, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant read")
    o_encode.resize(n_encode)
    fread(&o_encode[0], sizeof(u32), n_encode, fp)

    if fread(&n_size, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant read")
    o_sizes.resize(n_size)
    fread(&o_sizes[0], sizeof(u32), n_size, fp) 
        
    if fclose(fp) != 0:
        raise OSError(errno, "cant close after write")


cdef void save_corpus(const vector[u32]& new_encode, 
                      const vector[u32]& new_sizes, 
                      const char* path) except*:
    cdef FILE* fp = fopen(path, "wb")
    if fp == NULL:
        raise OSError(errno, "fail open for write")
    cdef u32 n_encode = <u32> new_encode.size()
    cdef u32 n_size   = <u32> new_sizes.size()
    
    if fwrite(&n_encode, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant write")
    if fwrite(&new_encode[0], sizeof(u32), n_encode, fp) != n_encode:
        raise OSError(errno, "cant write")

    if fwrite(&n_size, sizeof(u32), 1, fp) != 1:
        raise OSError(errno, "cant write")
    if fwrite(&new_sizes[0], sizeof(u32), n_size, fp) != n_size:
        raise OSError(errno, "cant write")

    if fclose(fp) != 0:
        raise OSError(errno, "cant close after write")





cdef _encode_corpus(cnp.ndarray[int32_t, ndim=1] old2new,
            set negate_id,                      # set of negate' s old id
            set skip_id,                        # set of skip_word's old id
            dict old2new_for_pair,              # pair id 2 new id 
            object load_path,
            object to_save_path):    
    cdef vector[u32] o_encode
    cdef vector[u32] o_sizes 

    cdef vector[u32] n_encode
    cdef vector[u32] n_sizes

    cdef unordered_set[u32] skip_id_std = build_set(skip_id)
    cdef unordered_set[u32] negate_id_std = build_set(negate_id)
    cdef unordered_map[u64, u32] o2n_for_pair_std = build_map(old2new_for_pair)
    load_corpus(o_encode, o_sizes, str(load_path).encode("utf-8"))
    cdef unordered_set[u32].iterator it = negate_id_std.begin()

    #idx for index in array, id for word's id
    cdef i32 prev_id, n_id                     #prev old_id (old vocab)
    cdef u32 o_id                              #old id
    cdef size_t cur_idx = 0
    cdef size_t o_sent_length, n_sent_length
    cdef u64 o_pair_id                          #encode merge by old id
    cdef bool prev_in_negate = False            
    cdef size_t i, j
    for i in range(len(o_sizes)):
        prev_id = -1            
        o_sent_length = o_sizes[i]
        n_sent_length = 0
        j = 0
        while (j < o_sent_length):
            o_id = o_encode[cur_idx + j]
            n_id = old2new[o_id]

            if (n_id < 0):
                if not prev_in_negate:
                    prev_id = -1
                j += 1
                continue

            if (prev_id != -1):
                prev_in_negate = (negate_id_std.find(prev_id) != negate_id_std.end())  
                if prev_in_negate and (skip_id_std.find(o_id) != skip_id_std.end()):
                    j += 1
                    continue
                
                o_pair_id = encode_pair_id(prev_id, o_id)
                if o2n_for_pair_std.find(o_pair_id) != o2n_for_pair_std.end():
                    n_encode.pop_back()
                    n_encode.push_back(o2n_for_pair_std[o_pair_id])
                    prev_id = -1
                    prev_in_negate = False
                    j += 1
                    continue
                
                else:
                    n_sent_length += 1
                    n_encode.push_back(n_id)
                
            else:
                n_sent_length += 1
                n_encode.push_back(n_id)
            prev_id = o_id
            j += 1
        cur_idx += o_sent_length
        n_sizes.push_back(n_sent_length)                          
       
    
    
    save_corpus(n_encode, n_sizes, str(to_save_path).encode("utf-8"))

def encode_corpus(cnp.ndarray[int32_t, ndim=1] old2new,
            set negate_id,                      # set of negate' s old id
            set skip_id,                        # set of skip_word's old id
            dict old2new_for_pair,
            object load_path,
            object to_save_path):
    return _encode_corpus(old2new, negate_id, skip_id, old2new_for_pair, load_path, to_save_path)


    







