"""A minimal byte-pair-encoding tokenizer (GPT-2 style, simplified).

1. Split text into 'words', keeping the leading space attached (" ball").
2. Start from single characters.
3. Repeatedly merge the most frequent adjacent pair into a new token.
The browser re-implements `encode` exactly (src/model/tokenizer.ts).
"""
import re
from collections import Counter

BASE = list("\n abcdefghijklmnopqrstuvwxyz.,?!:'")
PAT = re.compile(r" ?[a-z]+| ?[^a-z\s]|\n| +")


def pretokenize(text):
    return PAT.findall(text)


def train_bpe(text, vocab_size):
    words = Counter(pretokenize(text))
    vocab = ["<unk>"] + BASE
    seqs = {w: [c for c in w] for w in words}
    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        (a, b), n = pairs.most_common(1)[0]
        if n < 2:
            break
        merges.append([a, b]); vocab.append(a + b)
        for w in words:
            s = seqs[w]; i = 0; out = []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(s[i]); i += 1
            seqs[w] = out
    return vocab, merges


def encode(text, vocab, merges):
    idx = {t: i for i, t in enumerate(vocab)}
    rank = {(a, b): i for i, (a, b) in enumerate(merges)}
    ids = []
    for w in pretokenize(text):
        s = list(w)
        while len(s) > 1:
            best = min(((rank.get((s[i], s[i + 1]), 1e9), i) for i in range(len(s) - 1)))
            if best[0] == 1e9:
                break
            i = best[1]
            s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        ids.extend(idx.get(t, 0) for t in s)
    return ids
