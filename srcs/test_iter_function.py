import re
import unicodedata
import math
from collections import Counter
from typing import Iterable

import spacy 
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_sm")

negation = ("not", "no", "never")

_tok_re = re.compile(r"[A-Za-z]+(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]|-")
_year_re = re.compile(r"^(1|2)\d{3}$")

ROMAN_SMALL = {"i","ii","iii","iv","v","vi","vii","viii","ix","x","xi","xii","xiii","xiv","xv","xvi","xvii","xviii","xix","xx"}

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly","pretty",
          "rather","somewhat","kinda","sorta","at","all"}
POS_PAIR = {("PROPN","PROPN"),("ADJ","NOUN"),("NOUN","NOUN"),("PROPN","NOUN"),("NOUN","PROPN")}
negate = {"not", "no", "never"}

def iter(s: list):
    for i in s:
        yield i


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"'))



def tokenize(s: str):
    if not s:
        return []
    s = normalize_text(s)  
    return _tok_re.findall(s.lower())


def expand_contraction(tok: str):
    t = tok.replace(" ", "")
    if t.endswith("n't") and len(t) > 3: return [t[:-3], "not"]
    if t.endswith("'re") and len(t) > 3: return [t[:-3], "are"]
    if t.endswith("'ll") and len(t) > 3: return [t[:-3], "will"]
    if t.endswith("'ve") and len(t) > 3: return [t[:-3], "have"]
    if t.endswith("'m")  and len(t) > 2: return [t[:-2], "am"]
    if t.endswith("'d")  and len(t) > 2: return [t[:-2], "would"] 
    if t.endswith("'s")  and len(t) > 2: return [t[:-2]]         
    return [t]

END = {".", "!", "?"}
BAD_PART = {"not", "to"}
def _norm_token(tok: str) -> str:
    if tok.isdigit():
        if _year_re.fullmatch(tok):
            return "<year>"
        return "<digit>" if len(tok) <= 1 else "<nums>"
    if tok in ROMAN_SMALL:
        return "<century>"
    return tok

def iter_wiki_sentences(s: str, top_pairs: set[str,str]=None):
    sent = []

    def flush():
        nonlocal sent
        if sent:
            yield list(sent)
            sent.clear()


    prev_tok = None

    for tok in tokenize(s):
        if tok in END:
            for s in flush():
                yield s
            continue
        
        
        for st in expand_contraction(tok):
            st = _norm_token(st)
            if prev_tok: 
                if prev_tok in negation:
                    if st in AUX or st in INTENS:
                        continue 
                    if (prev_tok, st) in top_pairs:
                        sent.append(prev_tok+"_"+st)
                        prev_tok = None
                        continue
                if (prev_tok, st) in top_pairs:
                    sent.pop()
                    sent.append(prev_tok+"_"+st)
                    prev_tok = None
                    continue
            sent.append(st)
            prev_tok = st


    for s in flush():
        yield s






def candidate_pairs(token_iter: Iterable[str], min_count_bi: int=9,
               min_thres: int=2, top_k: int=3000):
    unigram = Counter()
    bigram = Counter()
    total_u = 0
    total_bi = 0
    unigram["<unk>"] = 0
    for sents in token_iter:
        for i in range(len(sents)):
            if i >= 1:
                if sents[i-1] in negate:
                    j = i
                    while sents[j] in AUX or sents[j] in INTENS and j < len(sents) - 1:
                        j += 1
                    bigram[(sents[i-1], sents[j])] += 1    
                else: 
                    bigram[(sents[i-1], sents[i])] += 1
            unigram[sents[i]] += 1
            
    total_u = unigram.total()
    total_bi = bigram.total()
    
    scored: list[tuple[float, tuple[str, str], int]] = []
    for (c1, c2), count_bi in bigram.item():
        if count_bi < min_count_bi:
            continue

        p12 = count_bi / total_bi
        p1 = unigram[c1] / total_u
        p2 = unigram[c2] / total_u
        de = p1 * p2

        if de <= 0: 
            continue

        pmi = math.log2(p12 / de)
    

        if pmi < min_thres:
            continue
        scored.append((pmi, (c1, c2), count_bi))

    scored.sort(key=lambda x:(x[0], x[2 ]))

    scored = scored[:top_k]
    
    top_pairs: set[tuple[str,str]] = {p for _,p,_ in scored}
    return unigram, total_u, bigram, total_bi, top_pairs
    



if __name__ == "__main__":
    s = "In the-year 2025, pe*ople still argue that technology has not really solved the simplest problems of everyday life. Some say, “It’s never going to replace the warmth of a human smile,” while others insist it might eventually do even more. Strangely enough, those arguments often appear in online forums at 3 a.m., posted by users with names like “Dragon_77” or “FutureKingIX.” You shouldn’t believe everything you read there, but you also can’t ignore the intensity—people will definitely shout, “This is absolutely not acceptable!” when a new gadget fails after only 2 days. Numbers don’t lie, though: 123456 users reported bugs, and at least 42% admitted they didn’t even read the manual. Historians sometimes compare our current obsession with progress to the optimism of the 19th century, back in the days when the 20th century seemed so far away. Yet here we are, in the 21st century, still asking whether AI is just another tool or a dangerous experiment. Some argue that it will never truly think, that machines can’t possibly feel; others counter that such doubts aren’t different from people in the 1800s saying airplanes would never fly. It’s not only about logic; it’s about belief, repetition, and hype—sometimes exaggerated, sometimes real. And so, in 3021, maybe another historian will laugh at our endless debates, noting that humans have always been both dreamers and doubters."
    top_pairs = {("not", "believe"), ("still","argue")}


    ls = tokenize(s)

    unigram, total_u, bigram, total_bi, top_pairs = build_pairs(iter(ls))

    print(len(top_pairs))
    for u, v in top_pairs:
        print(u, v)

    #for w in iter_wiki_sentences(s, top_pairs=top_pairs):
    #   print(w)


