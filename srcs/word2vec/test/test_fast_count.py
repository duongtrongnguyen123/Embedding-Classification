import pyximport
pyximport.install(language_level=3)

import fast_count
from fast_count import iter_sentences, first_pass, build_vocab
import torch
import spacy
import numpy as np
import os
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

POS_PAIR = {("PROPN","PROPN"),("ADJ","NOUN"),("NOUN","NOUN"),("PROPN","NOUN"),("NOUN","PROPN")}

negate = {"no", "not", "never"}
BAD_PART = {"not", "to"}
common_words = [
    "the","a","an","of","and","to","in","on","for","with","at","by","from",
    "about","as","is","was","are","were","be","been","being",
    "do","does","did","have","has","had",
    "that","this","these","those","it","its",
    "i","you","he","she","they","we","me","him","her","them","us",
    "my","your","his","their","our",
    "but","or","so","if","then",
    "there","here","when","where","what","which","who","whom",
    "gonna","gotta","wanna","lemme","gimme","imma","outta","kinda"
]

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
                       "pretty","rather","somewhat","kinda","sorta","at","all"}




tokens = [
    "in","the","quiet","village","by","the","river","people","woke","early","and","gathered","near","the","marketplace",".",
    "farmers","carried","baskets","of","vegetables","fishermen","laid","out","nets","children","ran","across","the","square","laughing","loudly",".",
    "the","smell","of","fresh","bread","drifted","from","the","small","bakery","where","the","old","baker","greeted","every","passerby","with","a","smile",".",
    "across","the","stone","bridge","travelers","from","distant","towns","arrived","bringing","stories","of","mountains","deserts","and","seas",".",
    "the","school","bell","rang","and","students","hurried","inside","carrying","notebooks","filled","with","scribbles","of","dreams","and","lessons","from","the","previous","day",".",
    "under","the","tall","oak","tree","elders","shared","tales","of","battles","long","past","and","wisdom","gathered","from","years","of","watching","seasons","change",".",
    "dogs","barked","in","the","distance","cats","stretched","lazily","on","rooftops","and","birds","circled","above","searching","for","crumbs","left","behind",".",
    "merchants","shouted","prices","of","spices","cloth","and","tools","while","neighbors","bargained","and","exchanged","news","about","births","marriages","and","harvests",".",
    "when","the","sun","reached","its","highest","point","the","square","shimmered","with","heat","and","people","rested","in","the","shade","sipping","cool","water",".",
    "travelers","spoke","of","kings","queens","scholars","and","explorers","who","shaped","lands","far","away","reminding","villagers","of","the","vast","world","beyond","the","river",".",
    "in","the","evening","lanterns","were","lit","casting","golden","light","across","cobbled","streets","while","music","rose","from","fiddles","drums","and","voices","singing","old","songs",".",
    "families","gathered","around","tables","eating","rice","meat","fruit","and","bread","giving","thanks","for","the","day",".",
    "children","listened","as","parents","told","stories","of","courage","friendship","and","love","passing","values","through","words","as","ancient","as","the","stars",".",
    "night","arrived","slowly","the","sky","turning","deep","blue","then","black","filled","with","constellations","known","to","shepherds","sailors","and","dreamers","alike",".",
    "the","village","grew","quiet","once","again","as","fires","dimmed","people","slept","and","the","river","whispered","softly","under","the","moon",".",
    "this","cycle","repeated","each","day","binding","generations","together","with","routine","memory","and","hope","for","tomorrow","."
]
pairs_tokens = [
    "machine","learning","is","nigga","popular","in","many","fields",".",
    "deep","learning","is","a","subfield","of","machine","learning",".",
    "data","science","not","really","often","uses","machine","learning","methods","here",".",
    "artificial","intelligence","includes","machine","learning","and","deep","learning",".",
    "neural","networks","are","the","basis","of","deep","learning",".",
    "python","is","commonly","used","for","machine","learning","and","data","science",".",
    "machine","learning","algorithms","can","be","supervised","or","unsupervised",".",
    "deep","learning","can","train","large","neural","networks","with","big","data",".",
    "data","science","combines","statistics","and","machine","learning",".",
    "machine","learning","applications","include","vision","speech","and","language",".",
    "deep","learning","applications","include","image","recognition","and","translation",".",
    "many","companies","invest","in","data","science","and","machine","learning",".",
    "researchers","publish","papers","about","deep","learning","and","machine","learning",".",
    "students","study","data","science","and","practice","machine","learning",".",
    "machine","learning","is","used","in","recommendation","systems",".",
    "deep","learning","is","used","in","autonomous","vehicles",".",
    "data","science","is","important","for","business",".",
    "machine","learning","models","require","training","data",".",
    "deep","learning","models","require","a","lot","of","data",".",
    "machine","learning","and","data","science","are","growing",".",
    "machine","learning","and","deep","learning","are","closely","related",".",
    "data","science","teams","work","with","machine","learning",".",
    "deep","learning","research","advances","every","year",".",
    "machine","learning","helps","solve","real","world","problems",".",
    "data","science","helps","companies","make","better","decisions",".",
    "deep","learning","helps","improve","speech","recognition",".",
    "machine","learning","is","used","for","fraud","detection",".",
    "data","science","is","used","for","customer","analytics",".",
    "deep","learning","is","not","used","for","medical","diagnosis",".",
    "machine","learning","requires","features","and","labels",".",
    "deep","learning","requires","powerful","hardware",".",
    "data","science","requires","data","collection",".",
    "machine","learning","is","at","the","core","of","modern","ai",".",
    "deep","learning","is","a","key","part","of","modern","ai",".",
    "data","science","is","a","key","part","of","modern","business",".",
    "machine","learning","and","deep","learning","drive","innovation",".",
    "students","learn","machine","learning","and","data","science",".",
    "researchers","explore","deep","learning","and","machine","learning",".",
    "companies","hire","data","science","teams",".",
    "machine","learning","is","taught","in","universities",".",
    "deep","learning","is","taught","in","advanced","courses",".",
    "data","science","is","taught","in","many","programs","."
]
not_check = [
  "we","do","not","often","agree",".",
  "this","is","not","really","useful",".",
  "the","result","is","not","very","good",".",
  "it","was","not","just","a","toy",".",
  "the","method","is","not","only","fast","but","also","accurate",".",
  "they","are","not","often","available",".",
  "the","model","is","not","quite","stable",".",
  "he","is","not","really","sure",".",
  "data","is","not","very","reliable",".",
  "we","are","not","often","wrong",".",
  "the","trend","is","not","so","clear",".",
  "she","is","not","too","confident",".",
  "the","system","is","not","rather","robust",".",
  "the","pipeline","is","not","pretty","simple",".",
  "they","are","not","fairly","represented",".",
  "i","am","not","often","convinced",".",
  "the","team","is","not","really","prepared",".",
  "the","assumption","is","not","very","realistic",".",
  "we","are","not","often","using","it",".",
  "the","pattern","is","not","often","observed","."
]


words = [
    "yesterday", "the", "movie", "festival", "opened", "downtown", "and", "the", "movie", "reviews", "were", "good", "overall", "with", "strong", "acting", "and", "clean", "editing", ".",
    "our", "coffee", "shop", "tested", "a", "new", "roast", "and", "the", "customer", "service", "was", "good", "while", "the", "music", "playlist", "felt", "cozy", ".",
    "the", "product", "launch", "highlighted", "battery", "life", "and", "camera", "quality", "and", "the", "user", "experience", "looked", "good", "during", "the", "demo", ".", 
    ".",                                  #test empty sents 
    "she", "built", "a", "machine", "learning", "model", "on", "a", "clean", "data", "pipeline", "and", "the", "validation", "result", "was","not", "good", "for", "early", "metrics", "not", "so", ".",  #test 'not'
    "signal", "processing", "processing", "notes", "signal", "processing", "signal", "processing", "signal", "connected", "fourier", "series", "to", "neural", "network", "intuition", "and", "the", "matrix", "calculus", "derivation", "was", "clear", "and", "good", ".",              #test duplicate
    "the", "stock", "market", "was","not", "quiet", "but", "foreign", "exchange", "volume", "spiked", "after", "policy", "news", "and", "several", "desks", "reported", "good", "liquidity", ".",                       #test 'not'
    "climate", "change", "policy", "appeared", "in", "several", "panels", "and", "the", "public", "discussion", "was", "good", "and", "surprisingly", "calm", ".",
    "later", "that", "night", "the", "game", "studio", "showed", "concept", "art", "and", "the", "game", "mechanics", "looked", "good", "with", "steady", "frame", "time", ".",
    "the", "research", "team", "cleaned", "labels", "and", "rebuilt", "the", "data", "pipeline", "so", "ingestion", "was", "reliable", "and", "the", "training", "curve", "looked", "good", ".",
    "our", "support", "desk", "improved", "customer", "service", "scripts", "and", "people", "said", "response", "time", "was", "good", "even", "during", "peak", "hours", ".",
    "the", "presentation", "explained", "neural", "network", "regularization", "and", "linked", "it", "to", "signal", "processing", "ideas", "and", "the", "examples", "were", "good", ".",
    "she", "visited", "a", "book", "fair", "and", "the", "story", "selection", "was", "good", "with", "fresh", "authors", "and", "clear", "themes", ".",
    "the", "teacher", "ran", "a", "workshop", "on", "matrix", "calculus", "and", "the", "exercises", "were", "good", "practice", "for", "gradients", "and", "hessians", ".",
    "they", "tested", "a", "new", "router", "and", "wifi", "coverage", "was", "good", "throughout", "the", "apartment", "except", "one", "corner", ".",
    "our", "friend", "organized", "a", "film", "night", "and", "the", "movie", "soundtrack", "was", "good", "and", "people", "stayed", "late", "to", "chat", ".",
    "the", "battery", "life", "on", "the", "laptop", "was", "good", "after", "tuning", "background", "apps", "and", "disabling", "heavy", "animations", ".",
    "the", "editor", "added", "grammar", "checks", "so", "the", "user", "experience", "felt", "smooth", "and", "the", "final", "layout", "looked", "good", ".",
    "our", "prototype", "game", "needed", "balancing", "but", "the", "core", "mechanics", "were", "good", "and", "players", "liked", "the", "pacing", ".",
    "the", "restaurant", "menu", "was", "diverse", "and", "the", "service", "was", "good", "even", "with", "a", "full", "house", ".",
    "a", "short", "review", "said", "the", "story", "arc", "was", "good", "but", "the", "ending", "felt", "rushed", ".",
    "for", "the", "hackathon", "she", "paired", "a", "neural", "network", "with", "a", "signal", "processing", "frontend", "and", "the", "demo", "looked", "good", ".",
    "the", "public", "garden", "planted", "new", "flowers", "and", "the", "weekend", "crowd", "was", "good", "natured", "and", "patient", ".",
    "he", "was", "not", "really", "tired", "and", "kept", "coding", "until", "midnight", "because", "the", "debugging", "progress", "was", "good", ".",
    "i", "took", "notes", "on", "foreign", "exchange", "microstructure", "and", "the", "liquidity", "session", "was", "good", "for", "practical", "tricks", ".",
    "the", "museum", "audio", "guide", "was", "clear", "and", "the", "visitor", "flow", "was", "good", "even", "with", "two", "tour", "groups", ".",
    "she", "brought", "a", "camera", "and", "the", "image", "stabilization", "was", "good", "during", "night", "shots", ".",
    "our", "local", "coffee", "shop", "added", "seats", "outside", "and", "the", "customer", "service", "remained", "good", "during", "rain", "."
]



def pair_id_decode(id):
    return (int(id>>32), int(id & 0XFFFFFFFF))

def aux_intens_id(o_word2id, a, b):
    res = set()
    for s in a:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)
    for s in b:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)
    return res

def vocab_pos(id2word, batch_size=32768, group_size=16):
    V = len(id2word)
    pos_arr = [None] * V

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    text = [" ".join(batch) for batch in batched(id2word, group_size)]

    idx = 0
    for doc in nlp.pipe(text, batch_size=batch_size):
        for tok in doc:
            if idx >= V:
                break
            pos_arr[idx] = tok.pos_
            idx += 1
    
    return pos_arr


def build_mask(id2word, pos_arr):
    adj_arr = [True if pos=="ADJ" else False for pos in pos_arr]
    adv_arr = [True if pos=="ADV" else False for pos in pos_arr]
    common_id = [word2id.get(w, None) for w in common_words]
    adj_ten = torch.tensor(adj_arr, dtype=torch.bool)
    adv_ten = torch.tensor(adv_arr, dtype=torch.bool)
    common_ten = torch.zeros(len(id2word), dtype=torch.bool)
    for w in common_id:
        if w is not None:
            common_ten[w] = True
    return adj_ten, adv_ten, common_ten
 





def encode_with_py(iter, o_word2id, old2new, old2new_for_pair, id2word):
    ids_bin = []
    sent_size = []
    skip_id = aux_intens_id(o_word2id, AUX, INTENS)
    not_id = o_word2id["not"] 
    prev_in_negate = False
    for sents in iter:
        out = []
        prev_id = -1
        for i in range(len(sents)):
            o_id = o_word2id.get(sents[i], None)
            new_id = old2new[o_id]
            if new_id < 0:
                if not prev_in_negate:
                    prev_id = -1
                continue

            if prev_id != -1:
                if prev_id == not_id:
                    prev_in_negate = (prev_id == not_id)
                    if prev_in_negate and o_id in skip_id:
                        continue
                pair_id = (np.int64(prev_id) << 32) | np.int64(o_id)
                obj = old2new_for_pair.get(pair_id, None)
                if obj is not None:
                    n_pair_id = int(obj)
                    out.pop()
                    out.append(n_pair_id)
                    prev_id = -1
                    prev_in_negate = False
                    continue
                else:
                    out.append(new_id.item())
            else:
                out.append(new_id.item())
            prev_id = o_id
        ids_bin.append(out)
        sent_size.append(len(out))
    return ids_bin, sent_size




if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    to_save_corpus_dir = os.path.join(base_dir, "..", "data", "test", "o_corpus.bin")
    to_save_vocab_dir = os.path.join(base_dir, "..", "data", "test", "vocab.pt")

    os.makedirs(os.path.dirname(to_save_vocab_dir), exist_ok=True)

    train_iter = iter_sentences(words) 
    unigram, top_pairs, o_id2word, o_word2id = first_pass(train_iter, top_k=200, min_pair_count=0, to_save_path=to_save_corpus_dir)
   
 
     
    old2new, word2id, id2word, counts = build_vocab(unigram, o_id2word, o_word2id, min_count=2)
    
        


    pos_arr = vocab_pos(id2word) 
    



    old2new_for_pair: dict[int, int] = {}
    counts_for_pair = []
    pos_arr_for_pair = []
    for pair_id, count in top_pairs:
        old_id1, old_id2 = pair_id_decode(pair_id)
        new_id1 = old2new[old_id1]
        new_id2 = old2new[old_id2]
        v = (id2word[new_id1] == "signal" and id2word[new_id2] == "processing")
        if v: 
            print("catch here")
        if new_id1 == -1 or new_id2 == -1:
            continue
        if v:
            print("catch here")
        a = (pos_arr[new_id1], pos_arr[new_id2]) in POS_PAIR 
        b = (id2word[new_id1] in negate)  
        c = (pos_arr[new_id1] == "VERB" and pos_arr[new_id2] == "PART" and id2word[new_id2] not in BAD_PART)
        if a:
            pos_arr_for_pair.append("NOUN")
        elif b:
            pos_arr_for_pair.append("ADV")
        elif c:
            pos_arr_for_pair.append("VERB")
        else:
            continue
        merge_word = id2word[new_id1] + "_" + id2word[new_id2]
        idx = len(id2word)
        id2word.append(merge_word)
        word2id[merge_word] = idx
        old2new_for_pair[pair_id] = idx
        counts_for_pair.append(count + 2)


            
    pos_arr.extend(pos_arr_for_pair)
    counts = np.concatenate((counts, counts_for_pair))
    counts = torch.from_numpy(counts)    

    adj_mask, adv_mask, common_mask = build_mask(id2word, pos_arr)
    
   

    iter = iter_sentences(words)
    ids_bin, sent_size = encode_with_py(iter, o_word2id, old2new, old2new_for_pair, id2word)
    print(ids_bin[:10])
    for arr in ids_bin:
        words = [id2word[s] for s in arr]
        print(words)
        
    #print(word2id["artificial_intelligence"])
    

    bundle = {
        "o_word2id": o_word2id,
        "o_id2word": o_id2word,
        "old2new"  : old2new,
        "id2word"  : id2word,
        "old2new_for_pair": old2new_for_pair,
    }

    torch.save(bundle, to_save_vocab_dir)
        
    
        
        
      
    
    """
    print(old2new[:10])
    print(f"len new word2id: {len(id2word)}")
    print(f"len old word2id: {len(o_id2word)}")
    print(id2word[old2new[o_word2id["many"]]])
    """
