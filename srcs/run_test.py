import pyximport
pyximport.install(language_level=3)

import fast_count
from fast_count import iter_sentences, first_pass, build_vocab

if __name__ == "__main__":
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
    "machine","learning","is","popular","in","many","fields",".",
    "deep","learning","is","a","subfield","of","machine","learning",".",
    "data","science","often","uses","machine","learning","methods",".",
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
    "deep","learning","is","used","for","medical","diagnosis",".",
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


    train_iter = iter_sentences(pairs_tokens) 
    unigram, scored, o_id2word, o_word2id = first_pass(train_iter, top_k=5000, min_pair_count=0)
   
     
     
    old2new, word2id, id2word, counts = build_vocab(unigram, scored, o_id2word, o_word2id, min_count=2)
 
    print(old2new[:10])
    print(f"len new word2id: {len(id2word)}")
    print(f"len old word2id: {len(o_id2word)}")
    print(id2word[old2new[o_word2id["many"]]])

