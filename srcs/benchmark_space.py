import time, statistics, random
import spacy

BATCH_SIZE = 16348
GROUP_SIZE = 32
ROUNDS = 3

# ---- vocab giả lập (thay bằng id2word thật của mày nếu có) ----
random.seed(0)
common = ["the","a","of","and","to","in","on","for","with","at","by","is","are",
          "i","you","he","she","they","we","not","never","no","going","book",
          "play","time","day","good","bad","great","small","big","city","school"]
rand_words = [f"tok{i}" for i in range(12000)]
id2word = common + rand_words
random.shuffle(id2word)

# ---- load spaCy: chỉ để lại tagger (nhanh hơn) ----
nlp = spacy.load("en_core_web_sm",
                 disable=["ner","parser","textcat","lemmatizer","attribute_ruler","senter"])

def bench(fn, rounds=ROUNDS, warmup=True):
    if warmup: fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return {"median": statistics.median(times), "runs": times}

def pos_each_word():
    out = 0
    for doc in nlp.pipe(id2word, batch_size=BATCH_SIZE):
        out += len(doc)
    return out

texts = [" ".join(id2word[i:i+GROUP_SIZE]) for i in range(0, len(id2word), GROUP_SIZE)]
def pos_grouped():
    out = 0
    for doc in nlp.pipe(texts, batch_size=BATCH_SIZE):
        out += len(doc)
    return out

res_each  = bench(pos_each_word)
res_group = bench(pos_grouped)

n_tokens = len(id2word)
tokps_each  = n_tokens / res_each["median"]
tokps_group = n_tokens / res_group["median"]

print(f"Tokens: {n_tokens:,}")
print(f"[each-word ] median = {res_each['median']:.3f}s  | ~{tokps_each:,.0f} tok/s")
print(f"[grouped   ] median = {res_group['median']:.3f}s | ~{tokps_group:,.0f} tok/s")
print(f"Speedup (each / grouped): ×{res_group['median']/res_each['median']:.2f}")
print(f"Runs each:  {res_each['runs']}")
print(f"Runs group: {res_group['runs']}")

