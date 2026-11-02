// Per-chapter teaching content. Filled in by the curriculum author; keyed by chapter id (see curriculum.ts).
// Numbers about TinyGPT come from the trained model (public/model) and train/train.py; code snippets were all run.
export interface QuizQuestion { q: string; options: string[]; answer: number; why: string }
export interface ChapterContent {
  /** One sentence: what you'll understand after this chapter. */
  promise: string;
  /** 3–5 visual-first "key idea" cards. */
  keyIdeas: { title: string; body: string; tryIt?: string }[];
  /** 3 multiple-choice checks; `answer` is the index of the correct option. */
  quiz: QuizQuestion[];
  /** For people who want the mechanism: plain-text math (no LaTeX) and a short PyTorch snippet. */
  deeper: { math: string; code: string; codeCaption: string };
  /** Common misconception and the correction. */
  pitfall?: { myth: string; truth: string };
  /** Next steps beyond this app (well-known resources only). */
  further?: { title: string; url: string; note: string }[];
}

export const CONTENT: Record<string, ChapterContent> = {
  "big-picture": {
    promise: "After this you'll be able to trace one prediction through a real GPT, from text in to next token out, and name every stage on the way.",
    keyIdeas: [
      { title: "It does exactly one thing", body: "A GPT reads text and outputs a probability for every possible next token. That's all. Given “…now tom has the”, TinyGPT says “ red” 87.1%.", tryIt: "Open the X-ray with the first preset and press play." },
      { title: "Text → numbers → vectors", body: "The text is chopped into tokens (23 for the mia/tom story). Each token becomes a list of 64 numbers, plus a second list that encodes its position." },
      { title: "3 layers of “look back, then think”", body: "In every layer, attention lets each token gather information from earlier tokens, then an MLP processes what it gathered. TinyGPT stacks 3 such layers with 4 attention heads each." },
      { title: "Score, pick, append, repeat", body: "The final vector is scored against all 256 tokens, softmax turns the scores into probabilities, one token is picked and appended, and the whole model runs again. A sentence is many predictions in a row." },
      { title: "Same recipe as ChatGPT, just tiny", body: "TinyGPT has 170,560 parameters. Large chat models use the same basic design with billions of parameters, vastly more data, and extra training to follow instructions." },
    ],
    quiz: [
      {
        q: "What does TinyGPT output each time it runs?",
        options: ["A whole sentence at once", "A probability for every one of its 256 tokens", "The single correct word, looked up in a database", "A grammar tree of the prompt"],
        answer: 1,
        why: "One forward pass produces one probability distribution over the vocabulary. Whole sentences come from repeating predict, pick, append.",
      },
      {
        q: "How does a GPT write a 10-token reply?",
        options: ["It runs once and outputs all 10 tokens in parallel", "It retrieves a stored reply that matches the prompt", "It runs 10 times; each picked token is appended to the input for the next run", "It writes the last token first and works backwards"],
        answer: 2,
        why: "Generation is autoregressive: every new token requires another full pass over the (now longer) context.",
      },
      {
        q: "Where does TinyGPT's knowledge that tom ends up with the red ball come from?",
        options: ["The tokenizer", "Hand-written rules about who gives what to whom", "A search through its training text at answer time", "170,560 numbers adjusted during training on example stories"],
        answer: 3,
        why: "Everything it knows is encoded in its trained parameters. Nothing is looked up when it answers; training on tiny-world stories shaped those numbers.",
      },
    ],
    deeper: {
      math: String.raw`ids = tokenize(text)                      23 integers in [0, 256)
x₀[i] = E_tok[ids[i]] + E_pos[i]          23 × 64
repeat for 3 layers:
  x ← x + Attn(LN(x))                     4 heads × 16 dims, causal
  x ← x + MLP(LN(x))                      64 → 256 → 64, GELU
logits = LN_f(x[last]) · E_tokᵀ           256 scores (weight tying)
p = softmax(logits / T);  next ~ p        then append and run again`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
V, T, D, H, L, M = 256, 64, 64, 4, 3, 256        # TinyGPT: vocab, context, width, heads, layers, MLP
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, M), nn.GELU(approximate="tanh"), nn.Linear(M, D))
    def forward(s, x):
        B, N, _ = x.shape
        q, k, v = (t.view(B, N, H, D // H).transpose(1, 2) for t in s.qkv(s.ln1(x)).split(D, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, N, D))   # attention: look back
        return x + s.mlp(s.ln2(x))                            # MLP: think
tok, pos = nn.Embedding(V, D), nn.Embedding(T, D)
blocks, ln_f = nn.Sequential(*[Block() for _ in range(L)]), nn.LayerNorm(D)
def gpt(ids):                                     # ids: (batch, n) token ids
    x = tok(ids) + pos(torch.arange(ids.shape[1]))
    return ln_f(blocks(x)) @ tok.weight.T         # weight-tied read-out → (batch, n, 256) logits
print(sum(p.numel() for m in (tok, pos, blocks, ln_f) for p in m.parameters()))  # 170560
print(gpt(torch.randint(V, (1, 23))).shape)       # torch.Size([1, 23, 256])`,
      codeCaption: "TinyGPT's exact architecture in about 20 lines of PyTorch (untrained here): it has the same 170,560 parameters and turns 23 token ids into 23 × 256 next-token scores.",
    },
    pitfall: { myth: "The model plans the whole answer first, then types it out.", truth: "It computes one next-token distribution per pass. Whatever “plan” exists has to live inside the vectors computed for the current position; there is no separate script being read off." },
    further: [
      { title: "3Blue1Brown: Transformers, the tech behind LLMs", url: "https://www.youtube.com/watch?v=wjZofJX0v4M", note: "The best visual tour of the whole pipeline." },
      { title: "Andrej Karpathy: Let's build GPT, from scratch, in code", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "Builds essentially TinyGPT, live, in code." },
      { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762", note: "The original transformer paper." },
    ],
  },
  "tokens": {
    promise: "After this you'll be able to predict how text gets split into tokens and explain why a model sees token ids, not letters.",
    keyIdeas: [
      { title: "Models read ids, not letters", body: "Text is chopped into pieces called tokens, and each piece is replaced by its id number. “mia has a red ball.” becomes 6 tokens: mia · ␣has · ␣a · ␣red · ␣ball · .", tryIt: "In the X-ray, open Tokens and type your own sentence." },
      { title: "TinyGPT knows 256 pieces", body: "Its vocabulary is 1 <unk> token, 34 single characters (a–z, space, newline and punctuation) and 221 learned merges such as “ the”, “ has” and “ kite”." },
      { title: "Merges are learned from frequency", body: "Byte-pair encoding (BPE) starts from single characters and repeatedly glues the most frequent neighbouring pair into a new token. Common words end up as one token; rare words stay in pieces." },
      { title: "Rare words get shredded", body: "“ kite” is one token, but “ elephant” (never in the tiny world) splits into 7: ␣e · le · p · h · a · n · t. The model has to rebuild meaning from fragments.", tryIt: "Type “the elephant has a kite” in the Tokens view." },
      { title: "The space belongs to the word", body: "Tokens carry their leading space: “ red” is one token, while “red” with no space before it isn't in the vocabulary and splits into r · ed. That's why the X-ray shows ␣ marks." },
    ],
    quiz: [
      {
        q: "Why does TinyGPT's tokenizer keep “ kite” whole but split “ elephant” into 7 pieces?",
        options: ["“ kite” is shorter", "“ kite” is frequent in the training text, so its pieces got merged; “ elephant” never appears", "Tokenizers always split words longer than 5 letters", "Animal names get a special rule"],
        answer: 1,
        why: "BPE only creates merges for pairs that are frequent in its training text. Words it rarely or never saw stay as small pieces.",
      },
      {
        q: "A tokenizer with more merges (a bigger vocabulary) will usually…",
        options: ["need fewer tokens for the same text", "need more tokens for the same text", "read letters directly", "remove the need for embeddings"],
        answer: 0,
        why: "More merges mean longer pieces, so the same text becomes fewer tokens and more of it fits in the context window. The price is a bigger embedding table.",
      },
      {
        q: "Why can a language model struggle to count the letters in a word?",
        options: ["It forgets the alphabet during training", "Counting is impossible for neural networks", "Its context window is too short", "It sees token ids, and a whole word may be a single opaque token"],
        answer: 3,
        why: "If “ kite” is one id, the letters k-i-t-e are never shown to the model directly; it can only learn spelling indirectly.",
      },
    ],
    deeper: {
      math: String.raw`training BPE:
  vocab = single characters;  text = list of characters
  repeat until |vocab| = 256:
    (a, b) = most frequent adjacent pair
    add token "ab";  replace every a,b with ab
encoding new text: apply the learned merges in the order they were learned
TinyGPT: text is first split into words (leading space attached),
         so merges never cross word boundaries
vocab = 1 <unk> + 34 characters + 221 merges = 256`,
      code: String.raw`from collections import Counter
def train_bpe(text, n_merges):
    seq, merges = list(text), []                 # start from single characters
    for _ in range(n_merges):
        (a, b), n = Counter(zip(seq, seq[1:])).most_common(1)[0]   # most frequent neighbour pair
        if n < 2: break
        merges.append(a + b)
        out, i = [], 0
        while i < len(seq):                      # glue every (a, b) into one new token
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == (a, b):
                out.append(a + b); i += 2
            else:
                out.append(seq[i]); i += 1
        seq = out
    return merges, seq
merges, toks = train_bpe("the cat has the hat. the cat has the bat.", 6)
print(merges)   # learned merges, most frequent first
print(toks)     # the text as tokens`,
      codeCaption: "A from-scratch BPE trainer on a toy sentence. TinyGPT's real one (train/bpe.py) splits text into words first, so its spaces attach to the front of a word (“ the”) instead of the back (“the ”).",
    },
    pitfall: { myth: "One token = one word.", truth: "A token is any frequent chunk: a whole word (“ kite”), part of a word (TinyGPT splits “ wednesday” into ␣w · ed · n · esday), or a single punctuation mark." },
    further: [
      { title: "Andrej Karpathy: Let's build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE", note: "Builds a GPT-style BPE tokenizer step by step." },
      { title: "minbpe (GitHub)", url: "https://github.com/karpathy/minbpe", note: "Minimal, readable BPE code to study and modify." },
      { title: "Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)", url: "https://arxiv.org/abs/1508.07909", note: "The paper that brought BPE to neural language models." },
    ],
  },
  "embeddings": {
    promise: "After this you'll be able to explain how a token id becomes a vector of 64 numbers and why tokens with similar roles end up close together.",
    keyIdeas: [
      { title: "A big lookup table", body: "Each of the 256 tokens owns a row of 64 numbers. Turning a token into a vector just means reading its row; no calculation yet." },
      { title: "Meaning = location", body: "Training moves the rows so that tokens used in similar ways end up close. In TinyGPT the nearest neighbours of “ red” are “ purple” (cosine 0.92), “ white” and “ black”; “ cat” sits near “ bear”, “ fish” and “ dog”.", tryIt: "In the X-ray, open Vectors and explore the meaning map." },
      { title: "A second table for position", body: "Attention on its own has no built-in sense of word order. So a second learned table (64 positions × 64 numbers) adds a position vector: x = token vector + position vector." },
      { title: "Neighbours explain mistakes", body: "After “now tom has the”, TinyGPT's runner-up guesses are “ white” and “ purple”: red's closest neighbours. The output scores are computed against the same table (weight tying), so similar vectors get similar scores." },
    ],
    quiz: [
      {
        q: "Two token vectors have cosine similarity 0.92. What does that most likely mean?",
        options: ["They are spelled similarly", "They have consecutive ids", "The model uses them in similar contexts", "They always appear next to each other"],
        answer: 2,
        why: "Embeddings are shaped by usage: tokens that can replace each other (like colors) get pushed toward similar directions.",
      },
      {
        q: "Why does TinyGPT add a position vector to each token vector?",
        options: ["To make the vectors longer", "To normalize the vectors", "To store the token's spelling", "Attention has no built-in sense of order; position vectors tell it where each token sits"],
        answer: 3,
        why: "Attention weighs tokens by content, not location. Giving each of the 64 slots its own learned signature makes order visible, so “mia gives tom” differs from “tom gives mia”.",
      },
      {
        q: "Computationally, embedding a token is…",
        options: ["a lookup: copy one row of a 256 × 64 table", "a full pass through all 3 layers", "a dictionary definition encoded as numbers", "a fresh random draw each time"],
        answer: 0,
        why: "It's equivalent to multiplying a one-hot vector by the table, which just selects a row.",
      },
    ],
    deeper: {
      math: String.raw`E_tok ∈ ℝ^(256×64),  E_pos ∈ ℝ^(64×64)
x₀[i] = E_tok[id_i] + E_pos[i]  =  onehot(id_i) · E_tok + E_pos[i]
cosine(a, b) = (a · b) / (‖a‖ ‖b‖)
weight tying: logits = LN_f(x) · E_tokᵀ  (the same table reads out)
parameters: 256·64 + 64·64 = 16,384 + 4,096 = 20,480`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
tok = nn.Embedding(256, 64)                 # 256 tokens × 64 numbers = 16,384 weights
pos = nn.Embedding(64, 64)                  # one learned vector per position 0..63
ids = torch.tensor([246, 46, 37, 157])      # TinyGPT ids for "mia", " has", " a", " red"
x = tok(ids) + pos(torch.arange(len(ids)))  # what layer 1 receives
print(x.shape)                              # torch.Size([4, 64])
# A lookup is just "one-hot row × table": no arithmetic magic.
onehot = F.one_hot(ids, 256).float()
print(torch.allclose(onehot @ tok.weight, tok(ids)))   # True
def nearest(i, k=3):                        # cosine-similarity neighbours of token i
    sims = F.cosine_similarity(tok.weight[i][None], tok.weight, dim=-1)
    return sims.topk(k + 1).indices[1:].tolist()        # skip the token itself
print(nearest(157))   # random now; in trained TinyGPT " red" → " purple", " white", " black"`,
      codeCaption: "Embedding lookups with TinyGPT's shapes and real token ids. The table here is untrained, so its neighbours are random until training arranges them.",
    },
    pitfall: { myth: "Each of the 64 numbers means one thing, like “is a color”.", truth: "Meaning is spread across directions in the space, usually combinations of many numbers. The 2-D meaning map is a projection (PCA) that keeps only the two biggest directions, so some neighbours that look close are not." },
    further: [
      { title: "3Blue1Brown: Transformers, the tech behind LLMs", url: "https://www.youtube.com/watch?v=wjZofJX0v4M", note: "Covers embeddings and the famous vector arithmetic intuition." },
      { title: "Efficient Estimation of Word Representations in Vector Space (word2vec, 2013)", url: "https://arxiv.org/abs/1301.3781", note: "The classic paper showing that word vectors capture meaning." },
    ],
  },
  "attention": {
    promise: "After this you'll be able to read an attention pattern and explain how a head copies information from earlier tokens.",
    keyIdeas: [
      { title: "Every token looks back", body: "In each layer, each token gathers information from earlier tokens, never later ones. How much it takes from each is a set of weights that add up to 100%." },
      { title: "Query, key, value", body: "Each token makes a query (what am I looking for?), a key (what do I contain?) and a value (what I hand over). Query·key scores set the weights; the values are mixed with those weights." },
      { title: "A real copying head", body: "In TinyGPT, layer 2 head 4 puts 93.2% of the last “ the”'s attention on “ red” (position 3, raw score 4.28). It finds the ball's color and copies it forward.", tryIt: "In the X-ray, open Attention and pick layer 2, head 4." },
      { title: "12 heads, 12 habits", body: "TinyGPT has 3 layers × 4 heads = 12 heads, each working in 16 dimensions. Each head learns its own pattern of where to look.", tryIt: "Click through all 12 head thumbnails in the Attention view." },
      { title: "No peeking: the causal mask", body: "Scores for future tokens are set to −∞ before softmax, so they get exactly 0% weight. That's what keeps a next-token predictor from cheating during training." },
    ],
    quiz: [
      {
        q: "In layer 2 head 4, the last “ the” puts 93.2% of its attention on “ red”. What happens as a result?",
        options: ["Mostly the value vector of “ red” is mixed into the last token's vector", "“ red” is deleted from the context", "The model outputs “ red” immediately and skips layer 3", "“ red” starts attending back to “ the”"],
        answer: 0,
        why: "A head's output is the weighted mix of value vectors, added into the querying token's residual stream. Later layers and the readout can then use that color information.",
      },
      {
        q: "Why are attention scores divided by √16 = 4 in TinyGPT?",
        options: ["To make it faster", "To average over the 4 heads", "To keep the dot products in a sane range so softmax doesn't become extremely peaked", "To hide future tokens"],
        answer: 2,
        why: "A dot product of two 16-dimensional vectors with unit-variance entries has variance 16. Dividing by √16 brings it back to about unit scale, keeping softmax soft and gradients healthy.",
      },
      {
        q: "What would go wrong if you trained without the causal mask?",
        options: ["Nothing; the mask only saves compute", "Attention weights would no longer sum to 1", "The model would have fewer parameters", "Each position could see the token it is supposed to predict, so it would learn to copy instead of predict"],
        answer: 3,
        why: "Training targets are the next tokens of the same sequence. Without the mask, the answer is visible, and the model is useless at generation time when the future really is unknown.",
      },
    ],
    deeper: {
      math: String.raw`per head (16 dims):  q_i = W_q x_i,  k_j = W_k x_j,  v_j = W_v x_j
score(i, j) = q_i · k_j / √16        for j ≤ i   (−∞ for j > i)
w(i, j) = exp(score(i, j)) / Σ_{j′ ≤ i} exp(score(i, j′))
head(i) = Σ_j w(i, j) · v_j
out(i) = W_o · concat(head₁(i), …, head₄(i))      4 × 16 = 64
cost: every pair of tokens → O(T²) per layer`,
      code: String.raw`import torch, torch.nn.functional as F
torch.manual_seed(0)
T, d = 5, 16                                  # 5 tokens, head size 16 (= 64 / 4 heads)
q, k, v = torch.randn(3, T, d).unbind(0)      # in a real model: three linear maps of x
scores = q @ k.T / d ** 0.5                   # (T, T): how well each query matches each key
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(future, float("-inf"))   # causal mask: no peeking ahead
w = scores.softmax(dim=-1)                    # each row sums to 1
out = w @ v                                   # each token gets a weighted mix of values
print(w.round(decimals=2))
ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
print(torch.allclose(out, ref, atol=1e-6))    # True: matches PyTorch's fused version`,
      codeCaption: "One causal attention head written out by hand and checked against PyTorch's built-in scaled_dot_product_attention.",
    },
    pitfall: { myth: "Attention weights tell you exactly why the model made its choice.", truth: "They show where information was read from, not what was done with it. A head can attend strongly to a token whose value carries little, and MLPs and other heads matter too. Treat patterns as clues, then test them, for example by switching the head off." },
    further: [
      { title: "3Blue1Brown: Attention in transformers, step-by-step", url: "https://www.youtube.com/watch?v=eMlx5fFNoYc", note: "The clearest visual explanation of queries, keys and values." },
      { title: "Attention Is All You Need (Vaswani et al., 2017)", url: "https://arxiv.org/abs/1706.03762", note: "Where multi-head attention was introduced." },
      { title: "A Mathematical Framework for Transformer Circuits (Anthropic, 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html", note: "How to reverse-engineer what attention heads do." },
    ],
  },
  "mlp": {
    promise: "After this you'll be able to explain what the MLP in each layer does and why most of a GPT's parameters live there.",
    keyIdeas: [
      { title: "Attention gathers, the MLP thinks", body: "After attention brings in context, the MLP processes each token's vector on its own, the same way at every position: expand 64 → 256, apply GELU, compress 256 → 64." },
      { title: "256 pattern detectors", body: "Each hidden neuron checks the vector for one pattern (a dot product). When it matches, the neuron fires and adds its own direction back into the vector.", tryIt: "In the X-ray, open Layers and watch the MLP neuron grid; switch between layers 1, 2 and 3." },
      { title: "Most parameters live here", body: "Each MLP has 33,088 parameters versus 16,640 for attention in the same layer. Across TinyGPT's 3 layers, MLPs hold 99,264 of the 170,560 parameters (58%)." },
      { title: "Facts as key → value memories", body: "One leading theory: a neuron's input weights act as a key (a pattern to detect) and its output weights as a value (what to add). Facts like “the cat says meow” may be spread across many such pairs." },
      { title: "The bend matters", body: "Without GELU's bend, 64 → 256 → 64 would collapse into one plain 64 → 64 linear map. The nonlinearity is what lets neurons act as on/off detectors." },
    ],
    quiz: [
      {
        q: "How do attention and the MLP differ in what they process?",
        options: ["Only the MLP has learned weights", "The MLP mixes positions; attention works on each position separately", "Both mix positions in the same way", "Attention mixes information across positions; the MLP works on each position separately"],
        answer: 3,
        why: "Attention is the only place where tokens exchange information. The MLP applies the same function to every position independently.",
      },
      {
        q: "What happens if you delete the GELU and keep the two linear layers?",
        options: ["Nothing important changes", "The two layers collapse into one linear map, so neurons can no longer act as on/off detectors", "The MLP gets more parameters", "Attention stops working"],
        answer: 1,
        why: "A product of linear maps is linear: W_out·(W_in·x) = (W_out·W_in)·x. Without the nonlinearity the extra width adds no expressive power.",
      },
      {
        q: "Why does each TinyGPT MLP have about twice as many parameters as the attention in the same layer?",
        options: ["It has more layers", "It stores the vocabulary", "Attention has no weights", "Its hidden layer is 4× wider: 2 · 64 · 256 weights versus 4 · 64 · 64 for attention's Q, K, V and output"],
        answer: 3,
        why: "2 · 64 · 256 = 32,768 weights versus 4 · 64 · 64 = 16,384 (plus biases: 33,088 vs 16,640).",
      },
    ],
    deeper: {
      math: String.raw`h = GELU(W_in · LN(x) + b_in)          W_in: 256 × 64
out = W_out · h + b_out                 W_out: 64 × 256
x ← x + out
  = x + b_out + Σ_{i=1..256} h_i · W_out[:, i]    each neuron writes its own direction
GELU(z) ≈ 0.5 · z · (1 + tanh(√(2/π) · (z + 0.044715 z³)))
params per layer: 64·256 + 256 + 256·64 + 64 = 33,088`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
D, M = 64, 256                                 # TinyGPT: 64 → 256 → 64
up, down = nn.Linear(D, M), nn.Linear(M, D)
x = torch.randn(D)                             # one token's (normalized) residual vector
h = F.gelu(up(x), approximate="tanh")          # 256 neuron activations
out = down(h)                                  # the update added back to the residual stream
# Same computation, read as "neuron i detects up.weight[i] and writes down.weight[:, i]":
out2 = down.bias + sum(h[i] * down.weight[:, i] for i in range(M))
print(torch.allclose(out, out2, atol=1e-5))    # True
print("neurons firing > 0.5:", (h > 0.5).sum().item(), "of", M)
print("params:", sum(p.numel() for p in [*up.parameters(), *down.parameters()]))  # 33088`,
      codeCaption: "An MLP with TinyGPT's shapes; the second computation gives the same output written as a sum of 256 neuron “write” vectors.",
    },
    pitfall: { myth: "Each fact is stored in one neuron.", truth: "Usually not. Neurons are often polysemantic (they fire for several unrelated things), and a fact tends to be spread across many neurons and layers. Clean one-neuron-one-concept cases are the exception." },
    further: [
      { title: "3Blue1Brown: How might LLMs store facts", url: "https://www.youtube.com/watch?v=9-Jl0dxWQs8", note: "A visual walk through how MLPs might store facts." },
      { title: "Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)", url: "https://arxiv.org/abs/2012.14913", note: "The key-value memory view of MLP layers." },
      { title: "Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)", url: "https://arxiv.org/abs/1606.08415", note: "The activation function TinyGPT uses (tanh approximation)." },
    ],
  },
  "residual-stream": {
    promise: "After this you'll be able to describe the residual stream as a shared workspace and use the logit lens to watch a prediction form layer by layer.",
    keyIdeas: [
      { title: "One vector, edited by addition", body: "Each token's 64-number vector flows from the embedding to the output. Every attention and MLP block reads it and adds a change; nothing overwrites it." },
      { title: "A shared whiteboard", body: "Because everything is added into the same stream, later layers can read what earlier ones wrote. Layer 2 head 4 (which, on stories like this one, fetches the handed-over item's color) writes “red” information that layer 3 and the readout then use." },
      { title: "The logit lens: peek mid-thought", body: "Apply the final readout to the stream after any layer to see what the model would predict if it stopped right there.", tryIt: "In the X-ray, open Layers with the mia/tom preset and watch the guess change at each step." },
      { title: "Watching “ red” emerge", body: "On the last “ the”, the lens reads “ the” (33%) after layer 1, “ white” (12%) after layer 2, and “ red” (87%) after layer 3. The answer is assembled step by step." },
      { title: "A highway for gradients", body: "Addition passes gradients straight through, so even layer 1 gets a clear training signal. This is a big reason deep transformers can be trained at all." },
    ],
    quiz: [
      {
        q: "After layer 2 the logit lens shows “ white” at 12%, but the final answer is “ red” at 87%. What does that tell you?",
        options: ["The model is broken", "The logit lens shows random noise", "Layer 3 replaced the stream with a brand-new vector", "The answer is built up across layers, and layer 3 still adds crucial information"],
        answer: 3,
        why: "Each layer adds to the same running sum. Here layer 3's contribution is what pushes “ red” to the top.",
      },
      {
        q: "In a pre-LN transformer like TinyGPT, how does each block update the residual stream x?",
        options: ["x ← Block(x)", "x ← x × Block(LN(x))", "x ← x + Block(LN(x))", "x ← LN(x + Block(x))"],
        answer: 2,
        why: "Each block reads a normalized copy and adds its output; the stream itself is never normalized in place. The last option is the older post-LN layout.",
      },
      {
        q: "Why is the logit lens only an approximation of what an early layer “thinks”?",
        options: ["Early layers may store information in a form the final readout can't decode yet", "It uses a different vocabulary", "It only works on the first token", "It changes the weights"],
        answer: 0,
        why: "The readout was trained on the final stream. Intermediate streams can hold useful information in directions the readout ignores.",
      },
    ],
    deeper: {
      math: String.raw`x⁰ = E_tok[id] + E_pos[i]
x^(l+½) = x^l + Attn_l(LN(x^l))
x^(l+1) = x^(l+½) + MLP_l(LN(x^(l+½)))
final:  x³ = x⁰ + Σ_{l=1..3} (attn_l + mlp_l)       a sum of 7 contributions
logit lens after step s:  p_s = softmax(LN_f(x^s) · E_tokᵀ)`,
      code: String.raw`import torch, torch.nn.functional as F
torch.manual_seed(0)
E = 0.2 * torch.randn(256, 64)                 # a stand-in embedding table (256 tokens × 64)
THE, RED = 38, 157                             # TinyGPT ids for " the" and " red"
def lens(x):                                   # logit lens: read the stream out as if it were the end
    return F.softmax(F.layer_norm(x, (64,)) @ E.T, dim=-1)
x = E[THE].clone()                             # the stream starts as the token's embedding
writes = [0.05 * torch.randn(64), 0.8 * E[RED], 2.0 * E[RED]]   # toy layer outputs
for l, delta in enumerate(writes, 1):
    x = x + delta                              # each layer ADDS; nothing is overwritten
    p = lens(x)
    print(f"after layer {l}: p(' the')={p[THE]:.2f}  p(' red')={p[RED]:.2f}")
print(torch.allclose(x, E[THE] + sum(writes)))  # final stream = embedding + sum of all writes`,
      codeCaption: "A toy residual stream with stand-in weights (not TinyGPT's): each layer adds a write, and the logit lens shows the top guess flipping from “ the” to “ red”, much like the real model's the → white → red.",
    },
    pitfall: { myth: "Each layer transforms the input into a completely new representation.", truth: "In a residual network each layer only adds an update to a running sum. The original embedding is still part of the final vector, which is why decoding intermediate states works at all." },
    further: [
      { title: "A Mathematical Framework for Transformer Circuits (Anthropic, 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html", note: "Introduces the residual stream as a communication channel between layers." },
      { title: "3Blue1Brown: Transformers, the tech behind LLMs", url: "https://www.youtube.com/watch?v=wjZofJX0v4M", note: "Visual overview of how vectors flow through the layers." },
    ],
  },
  "next-token": {
    promise: "After this you'll be able to turn logits into probabilities and explain how temperature and top-k change what the model writes.",
    keyIdeas: [
      { title: "One score per token", body: "The final vector is compared with all 256 token vectors, giving 256 scores called logits. For the mia/tom prompt: “ red” 10.37, “ white” 7.68, “ purple” 7.53." },
      { title: "Softmax: scores → percentages", body: "Exponentiate each score and divide by the total. Gaps get amplified: a 2.7-point lead turns into “ red” 87.1% versus “ white” 5.9%." },
      { title: "Temperature turns the dial", body: "Divide the logits by T before softmax. At T = 0.5 “ red” gets 99.2%; at T = 2 it drops to 32.4%. As T → 0 the model always picks its top token.", tryIt: "In the X-ray, open Probabilities and drag the what-if temperature slider." },
      { title: "Sampling is a weighted dice roll", body: "The X-ray samples at temperature 0.8 from the 8 most likely tokens (top-k = 8). Same prompt, different roll, possibly a different story." },
      { title: "Real coin flips", body: "For “zoe has a yellow shoe. q: what color is the shoe? a: the shoe is”, TinyGPT gives “ yellow” 47.1% and “ pink” 45.9%. Sampling answers pink almost half the time.", tryIt: "Choose the zoe preset, go to Pick next and press Re-roll a few times." },
    ],
    quiz: [
      {
        q: "Two logits differ by 2.7. At temperature 1, how do their probabilities compare?",
        options: ["About 2.7× apart", "Equal", "It depends on the other tokens", "About e^2.7 ≈ 15× apart"],
        answer: 3,
        why: "Softmax ratios depend only on logit differences: p_a / p_b = exp(z_a − z_b). TinyGPT's “ red” (87.1%) vs “ white” (5.9%) is exactly this ≈15× ratio.",
      },
      {
        q: "What does raising the temperature from 1 to 2 do?",
        options: ["Flattens the distribution, giving unlikely tokens more chances", "Sharpens it toward the top token", "Changes which token has the highest score", "Makes the model know more"],
        answer: 0,
        why: "Dividing logits by 2 halves every gap, so probabilities move closer together. The ranking never changes.",
      },
      {
        q: "With top-k = 8, can the sampler ever output the 9th most likely token?",
        options: ["Yes, just rarely", "Only if its probability is above 5%", "No: top-k drops everything outside the 8 highest-scoring tokens before sampling", "Only for the first generated token"],
        answer: 2,
        why: "Top-k sets all other logits to −∞, so they get exactly zero probability.",
      },
    ],
    deeper: {
      math: String.raw`logits  z = LN_f(x_last) · E_tokᵀ           256 numbers
softmax p_i = exp(z_i / T) / Σ_j exp(z_j / T)
ratio   p_a / p_b = exp((z_a − z_b) / T)
top-k:  keep the k largest z_i, set the rest to −∞, renormalize
greedy (T → 0): next = argmax_i z_i`,
      code: String.raw`import torch
logits = torch.tensor([10.37, 7.68, 7.53])     # TinyGPT's top 3: " red", " white", " purple"
for T in (0.5, 1.0, 2.0):                      # temperature reshapes the distribution
    print(T, (logits / T).softmax(-1).round(decimals=3).tolist())
def sample(logits, temperature=0.8, top_k=8, gen=None):
    if temperature == 0:
        return int(logits.argmax())            # greedy: always the favourite
    z = logits / temperature
    kth = z.topk(min(top_k, z.numel())).values[-1]
    z = z.masked_fill(z < kth, float("-inf"))  # top-k: drop everything outside the k best
    return int(torch.multinomial(z.softmax(-1), 1, generator=gen))  # weighted dice roll
g = torch.Generator().manual_seed(7)
print([sample(logits, 1.0, 2, g) for _ in range(20)])   # mostly 0 (" red"), sometimes 1`,
      codeCaption: "Temperature and top-k sampling on TinyGPT's real top-3 logits. With only 3 tokens the shares differ slightly from the full 256-token softmax (88.8% vs 87.1% at T = 1).",
    },
    pitfall: { myth: "The probabilities are the model's confidence that an answer is true.", truth: "They estimate how likely each token is to come next in text like the training data. A model can be confidently wrong, and a near coin-flip like yellow 47.1% vs pink 45.9% means that sampling at temperature 1 picks the wrong color about 46% of the time." },
    further: [
      { title: "3Blue1Brown: Transformers, the tech behind LLMs", url: "https://www.youtube.com/watch?v=wjZofJX0v4M", note: "Shows how the final vector becomes a probability distribution." },
      { title: "The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)", url: "https://arxiv.org/abs/1904.09751", note: "Why pure sampling and greedy decoding both go wrong, and top-p sampling." },
    ],
  },
  "how-it-learned": {
    promise: "After this you'll be able to describe the training loop that turned 170,560 random numbers into TinyGPT.",
    keyIdeas: [
      { title: "It started as noise", body: "At step 0 every parameter was random. Given “mia has a red ball. tom” it continued with gibberish like “sh hill w timog say”.", tryIt: "On the Training page, click the step-0 marker, then the step-3000 one." },
      { title: "The loop: guess, score, blame, nudge", body: "Show it 64 windows of 64 tokens, let it predict every next token, measure how surprised it was (the loss), work out which way each parameter should move (gradients), and nudge them all a little. Repeat 3,000 times." },
      { title: "Loss fell from 5.56 to about 0.46", body: "5.56 is what pure guessing among 256 tokens scores (ln 256 ≈ 5.55). By the end its average surprise per token was about 0.46." },
      { title: "Skills appear in order", body: "In sampled text, by step 300 it wrote fluent tiny-world sentences; at step 1000 it still said tom has the “yellow ball”; by step 3000 it tracked who has what and wrote “red ball”." },
      { title: "80 seconds, same recipe as the giants", body: "TinyGPT trained in about 80 seconds on a CPU. Large models run essentially the same loop for weeks or months on thousands of accelerators." },
    ],
    quiz: [
      {
        q: "Why was TinyGPT's loss at step 0 about 5.56?",
        options: ["A random model spreads probability roughly evenly over 256 tokens, and −ln(1/256) ≈ 5.55", "5.56 is the number of training minutes", "The learning rate was too high", "Loss always starts at 5.56"],
        answer: 0,
        why: "Before training, every token is about equally likely, so the surprise on the true token is ln 256. Seeing this number confirms the setup is right.",
      },
      {
        q: "What exactly is TinyGPT trained to do?",
        options: ["Answer questions correctly", "Predict the next token at every position of its training text", "Copy its input", "Classify stories by topic"],
        answer: 1,
        why: "The only objective is next-token prediction. It answers “q: … a:” questions only because that pattern appears in its training text.",
      },
      {
        q: "What does each of the 3,000 training steps do to the parameters?",
        options: ["Replaces them with new random values", "Sets them to the best values found by searching", "Moves every parameter a small amount in the direction that lowers the loss on that batch", "Changes only one layer at a time"],
        answer: 2,
        why: "Every step updates all 170,560 parameters a little, using that batch's gradients and the AdamW optimizer.",
      },
    ],
    deeper: {
      math: String.raw`for step t = 0 … 3000:
  x, y = 64 random windows × 64 tokens       (y = x shifted by one)
  L = mean over 4,096 positions of −ln p_θ(y | x)
  g = ∂L/∂θ   (backprop), clipped to norm ≤ 1.0
  θ ← AdamW(θ, g, lr_t)    lr: 100-step warm-up to 3e-3, then cosine decay
tokens seen ≈ 3,000 × 4,096 ≈ 12.3M  (≈ 7–8 passes over the 1.6M training tokens)`,
      code: String.raw`import math, torch, torch.nn.functional as F
text = "mia has a red ball. tom has a blue kite. " * 50
chars = sorted(set(text)); V = len(chars)
data = torch.tensor([chars.index(c) for c in text])
x, y = data[:-1], data[1:]                         # target = the next character
W = torch.zeros(V, V, requires_grad=True)          # tiny "model": next-char scores per char
opt = torch.optim.AdamW([W], lr=0.1)
for step in range(201):
    loss = F.cross_entropy(W[x], y)                # 1. predict  2. measure surprise
    opt.zero_grad(); loss.backward()               # 3. blame: gradient for every weight
    opt.step()                                     # 4. nudge every weight a little
    if step % 50 == 0: print(step, round(loss.item(), 3))
print("uniform guess = ln V =", round(math.log(V), 3))   # step 0 starts exactly here`,
      codeCaption: "The same four-step loop on a toy next-character model: the loss starts exactly at ln V (uniform guessing) and falls as the weights learn which character follows which.",
    },
    pitfall: { myth: "Training copies the training text into the model.", truth: "Training adjusts parameters so likely continuations get high probability. TinyGPT's loss on held-out text (about 0.46) matches its training loss, and freshly generated stories it has never seen score about 0.45, so it learned patterns that generalize rather than a copy. Big models can memorize some passages, but mostly they learn patterns." },
    further: [
      { title: "3Blue1Brown: But what is a neural network?", url: "https://www.youtube.com/watch?v=aircAruvnKk", note: "What a neural network is, from the ground up." },
      { title: "3Blue1Brown: Gradient descent, how neural networks learn", url: "https://www.youtube.com/watch?v=IHZwWFHWa-w", note: "How the nudging works: gradient descent." },
      { title: "Andrej Karpathy: Let's build GPT, from scratch, in code", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "Writes and trains a model like TinyGPT, live." },
    ],
  },
  "loss": {
    promise: "After this you'll be able to compute cross-entropy loss by hand and read a loss curve like a researcher.",
    keyIdeas: [
      { title: "Loss = surprise", body: "For each position, loss = −ln(probability given to the true next token). 87.1% on “ red” costs 0.14; 5.9% would cost 2.83; 100% costs 0." },
      { title: "The starting line: ln 256", body: "Uniform guessing over 256 tokens scores ln 256 ≈ 5.55. TinyGPT's step-0 loss was 5.56, so a fresh model really knows nothing." },
      { title: "An average over thousands of guesses", body: "Each step averages the loss over 64 × 64 = 4,096 predictions. TinyGPT ended near 0.46: about as unsure as choosing between 1.6 equally likely options (e^0.46 ≈ 1.58, the “perplexity”)." },
      { title: "Two curves, one story", body: "Training loss is measured on text the model learns from; validation loss on held-out text (TinyGPT keeps 5% aside). Both ended near 0.46. Short template lines like \u201cthe sky is blue.\u201d repeat in the tiny world, so not every held-out line is new, but freshly generated stories also score about 0.45: it generalizes.", tryIt: "On the Training page, compare the solid (training) and dashed (validation) curves." },
      { title: "Zero isn't the goal", body: "Real text is partly unpredictable (which name starts the next story?), so even a perfect model has a loss floor above 0." },
    ],
    quiz: [
      {
        q: "The model gives the true next token probability 0.5. What is the loss at that position?",
        options: ["0.5", "0", "2", "−ln 0.5 ≈ 0.69"],
        answer: 3,
        why: "Cross-entropy is −ln(p_true). Halving your confidence in the truth always costs ln 2 ≈ 0.69 extra.",
      },
      {
        q: "Training loss keeps dropping while validation loss starts rising. What's happening?",
        options: ["The validation set is too big", "The learning rate is too low", "Underfitting", "Overfitting: the model is memorizing training text instead of learning general patterns"],
        answer: 3,
        why: "Improving only on seen data while getting worse on unseen data is the signature of overfitting.",
      },
      {
        q: "Why does cross-entropy punish confident mistakes so harshly?",
        options: ["Because it squares the error", "Because −ln p shoots up as p approaches 0: 1% on the truth costs 4.6, 0.01% costs 9.2", "Because it counts characters", "It doesn't; every mistake costs the same"],
        answer: 1,
        why: "The log makes the cost unbounded as the probability on the true token goes to zero, so the model learns never to be certain about the wrong thing.",
      },
    ],
    deeper: {
      math: String.raw`per position:   ℓ = −ln p(y_true)
batch loss:     L = −(1/N) Σ_t ln p(y_t | tokens before t)      N = 4,096 in TinyGPT
uniform guess:  −ln(1/256) = ln 256 ≈ 5.545
perplexity:     PPL = e^L          L = 0.46 → PPL ≈ 1.58
gradient w.r.t. the logits:  ∂ℓ/∂z_i = p_i − [i = y_true]`,
      code: String.raw`import math, torch, torch.nn.functional as F
print(-math.log(0.871))   # 0.138: TinyGPT's loss if the truth is " red" (it gave 87.1%)
print(-math.log(0.059))   # 2.83: if the truth had been " white" (5.9%)
print(math.log(256))      # 5.545: a uniform guess over 256 tokens (TinyGPT at step 0: 5.56)
print(math.exp(0.46))     # ≈ 1.58: perplexity at the final loss of 0.46
torch.manual_seed(0)
logits = torch.randn(4, 256)                  # scores at 4 positions for 256 tokens
targets = torch.tensor([157, 183, 29, 38])    # the true next tokens
manual = -logits.log_softmax(-1)[torch.arange(4), targets].mean()
print(torch.allclose(manual, F.cross_entropy(logits, targets)))   # True`,
      codeCaption: "Cross-entropy by hand with TinyGPT's real numbers, then checked against PyTorch's F.cross_entropy.",
    },
    pitfall: { myth: "A loss of 0.46 means the model is wrong 46% of the time.", truth: "Loss is average surprise in nats (natural-log units), not an error rate. 0.46 means the true token gets about e^−0.46 ≈ 63% probability on average (a geometric mean)." },
    further: [
      { title: "3Blue1Brown: Gradient descent, how neural networks learn", url: "https://www.youtube.com/watch?v=IHZwWFHWa-w", note: "Explains the cost function that training minimizes." },
      { title: "Andrej Karpathy: Let's build GPT, from scratch, in code", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "Implements and watches this exact loss while training a GPT." },
    ],
  },
  "backprop": {
    promise: "After this you'll be able to follow how one loss number sends a “blame” signal back to every one of 170,560 parameters.",
    keyIdeas: [
      { title: "The question backprop answers", body: "For every parameter: if I nudge it up a tiny bit, does the loss go up or down, and by how much? That number is its gradient." },
      { title: "The chain rule as a relay", body: "Each operation only needs its own local slope. Multiplying local slopes along the path from the loss back to a parameter gives that parameter's gradient.", tryIt: "On the Backprop page, step through the backward pass one operation at a time." },
      { title: "One sweep gets every gradient", body: "Instead of testing 170,560 nudges one by one, a single backward pass computes all gradients, at roughly twice the cost of a forward pass." },
      { title: "Gradients add where paths split", body: "When a value is used in several places (like the residual stream feeding every layer), its gradient is the sum of the gradients flowing back from each use." },
      { title: "The output signal is simple", body: "For softmax + cross-entropy, each logit's gradient is its probability minus 1 for the true token (minus 0 for the rest). If “ red” at 87.1% is the truth: −0.129 on “ red”, +0.059 on “ white”." },
    ],
    quiz: [
      {
        q: "A parameter's gradient is −0.2. With plain gradient descent at learning rate 0.1, what happens to it?",
        options: ["It decreases by 0.02", "It increases by 0.02", "It is set to −0.2", "It stays the same"],
        answer: 1,
        why: "θ ← θ − lr · g = θ − 0.1 · (−0.2) = θ + 0.02. A negative gradient means increasing the parameter lowers the loss.",
      },
      {
        q: "Why is backprop far faster than nudging each parameter and re-measuring the loss?",
        options: ["It skips most parameters", "It uses random guesses", "It reuses intermediate results so one backward sweep yields all gradients", "It only works on small models"],
        answer: 2,
        why: "Nudging would need one extra forward pass per parameter (170,560 of them). Backprop shares the work through the chain rule.",
      },
      {
        q: "If y = f(g(x)), then dy/dx equals…",
        options: ["f′(g(x)) · g′(x)", "f′(x) + g′(x)", "f(x) · g(x)", "f′(x) / g′(x)"],
        answer: 0,
        why: "The chain rule: the outer slope (evaluated at the inner value) times the inner slope. Backprop applies it repeatedly from the loss backwards.",
      },
    ],
    deeper: {
      math: String.raw`chain rule: y = f(g(x))  ⇒  dy/dx = f′(g(x)) · g′(x)
linear layer z = W a + b, given δ = ∂L/∂z:
  ∂L/∂W = δ aᵀ,    ∂L/∂b = δ,    ∂L/∂a = Wᵀ δ
softmax + cross-entropy:  ∂L/∂z_i = p_i − [i = y]
branching: if x feeds two places, ∂L/∂x = sum of both contributions
cost: backward ≈ 2 × forward  →  training ≈ 3 × forward per token`,
      code: String.raw`import math, torch
x, y = 2.0, 1.0                                 # input and target
w = torch.tensor(-1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
z = w * x + b                                   # forward pass: build the graph
a = torch.tanh(z)
loss = (a - y) ** 2
loss.backward()                                 # autograd walks the graph backwards
# The same gradients by hand with the chain rule:
a_ = math.tanh(w.item() * x + b.item())
dL_da = 2 * (a_ - y)                            # ∂loss/∂a
da_dz = 1 - a_ ** 2                             # ∂tanh(z)/∂z
print(w.grad.item(), dL_da * da_dz * x)         # ∂loss/∂w = dL/da · da/dz · dz/dw (= x)
print(b.grad.item(), dL_da * da_dz * 1)         # ∂loss/∂b = dL/da · da/dz · 1`,
      codeCaption: "PyTorch's autograd and the hand-applied chain rule give the same gradients for a one-neuron model.",
    },
    pitfall: { myth: "Backpropagation is the learning algorithm.", truth: "Backprop only computes the gradients. The optimizer (SGD, Adam, AdamW…) decides how to use them to change the weights." },
    further: [
      { title: "3Blue1Brown: Backpropagation, intuitively", url: "https://www.youtube.com/watch?v=Ilg3gGewQ5U", note: "The intuition, visually." },
      { title: "Andrej Karpathy: Building micrograd", url: "https://www.youtube.com/watch?v=VMj-3S1tku0", note: "Builds backprop from scratch in about 100 lines." },
      { title: "micrograd (GitHub)", url: "https://github.com/karpathy/micrograd", note: "The tiny autograd engine from that video." },
    ],
  },
  "optimizers": {
    promise: "After this you'll be able to explain how SGD, momentum and AdamW turn gradients into weight updates, and why the learning rate is the knob that matters most.",
    keyIdeas: [
      { title: "Walk downhill", body: "Gradient descent moves each parameter a small step against its gradient: θ ← θ − lr · g. The learning rate (lr) sets the step size." },
      { title: "Too small crawls, too big explodes", body: "A tiny lr wastes steps; a huge one overshoots the valley so the loss bounces or blows up. TinyGPT used a peak lr of 3e-3.", tryIt: "In the Train lab, try a much bigger and a much smaller learning rate and compare the curves." },
      { title: "Momentum: a heavy ball", body: "Keep a running average of recent gradients and step along it. Directions that stay consistent speed up; zig-zags cancel out." },
      { title: "Adam: a step size per parameter", body: "Adam also tracks each parameter's typical gradient size and divides by it, so parameters with tiny gradients still move. AdamW adds weight decay (TinyGPT: 0.05) that gently shrinks weights toward zero." },
      { title: "Warm up, then cool down", body: "TinyGPT ramped its lr up over the first 100 steps, then let it follow a cosine curve down toward zero by step 3000: bold steps early, careful steps late." },
    ],
    quiz: [
      {
        q: "Right after starting, the loss shoots up to huge values or NaN. What's the most likely fix?",
        options: ["Train for longer", "Lower the learning rate", "Use a smaller validation set", "Remove the warm-up"],
        answer: 1,
        why: "Exploding loss usually means steps are too large and overshoot. Lowering the lr (or adding warm-up and gradient clipping) is the first thing to try.",
      },
      {
        q: "What does Adam divide each parameter's update by?",
        options: ["The number of parameters", "The current loss", "A running estimate of that parameter's gradient size, √v̂", "The batch size"],
        answer: 2,
        why: "Dividing by √v̂ normalizes the step, so each parameter moves at roughly the learning rate's scale regardless of its raw gradient size.",
      },
      {
        q: "What's the key difference between Adam with L2 regularization and AdamW?",
        options: ["AdamW applies weight decay directly to the weights, separately from the adaptive gradient scaling", "AdamW has no momentum", "AdamW needs a bigger learning rate", "There is no difference"],
        answer: 0,
        why: "In Adam + L2 the decay term gets divided by √v̂ like any gradient, weakening it unevenly. AdamW decouples it, which works better in practice.",
      },
    ],
    deeper: {
      math: String.raw`SGD:       θ ← θ − η g
Momentum:  m ← μ m + g;     θ ← θ − η m
Adam:      m ← β₁ m + (1−β₁) g;     v ← β₂ v + (1−β₂) g²
           m̂ = m / (1−β₁ᵗ);   v̂ = v / (1−β₂ᵗ);   θ ← θ − η m̂ / (√v̂ + ε)
AdamW:     θ ← θ − η λ θ, then the Adam step
TinyGPT:   η_peak = 3e-3, λ = 0.05, β = (0.9, 0.999)
schedule:  η_t = 3e-3 · min(1, (t+1)/100) · ½ (1 + cos(π t / 3000))`,
      code: String.raw`import torch
torch.manual_seed(0)
lr, b1, b2, eps, wd = 3e-3, 0.9, 0.999, 1e-8, 0.05   # TinyGPT: AdamW, lr 3e-3, weight decay 0.05
p = torch.randn(5, requires_grad=True)
q = p.detach().clone()                               # our hand-written copy
opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
m, v = torch.zeros(5), torch.zeros(5)
for t in range(1, 101):
    g = 2 * (q - 1)                       # gradient of Σ(q − 1)², by hand
    q -= lr * wd * q                      # decoupled weight decay (the "W" in AdamW)
    m = b1 * m + (1 - b1) * g             # momentum: running average of gradients
    v = b2 * v + (1 - b2) * g * g         # running average of squared gradients
    q -= lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
    opt.zero_grad(); ((p - 1) ** 2).sum().backward(); opt.step()
print(torch.allclose(p.detach(), q, atol=1e-6))      # True: identical to torch.optim.AdamW`,
      codeCaption: "AdamW written out by hand produces exactly the same parameters as torch.optim.AdamW, using TinyGPT's settings.",
    },
    pitfall: { myth: "With Adam you don't need to tune anything.", truth: "AdamW is the standard choice for transformers because it copes well with gradients of very different sizes, but no optimizer rescues a bad learning rate. The lr and its schedule are still the first knobs to tune." },
    further: [
      { title: "Why Momentum Really Works (Distill, 2017)", url: "https://distill.pub/2017/momentum/", note: "An interactive explanation of momentum." },
      { title: "Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)", url: "https://arxiv.org/abs/1412.6980", note: "The original Adam paper." },
      { title: "Decoupled Weight Decay Regularization (AdamW, Loshchilov & Hutter, 2017)", url: "https://arxiv.org/abs/1711.05101", note: "Why decoupled weight decay works better." },
    ],
  },
  "train-your-own": {
    promise: "After this you'll be able to set up data, batches and a validation split, and tell underfitting from overfitting.",
    keyIdeas: [
      { title: "Data is the curriculum", body: "A model can only learn patterns that are in its data. TinyGPT's 5.2M characters of tiny-world stories are why it knows cats say meow, and why it has never met an elephant." },
      { title: "Batches: many windows at once", body: "Each step picks 64 random windows of 64 tokens. The targets are the same windows shifted by one token, so every position is a training example: 4,096 predictions per step." },
      { title: "Hold some data back", body: "TinyGPT keeps the last 5% of its tokens for validation and never trains on them. Watch for leakage: in the tiny world, short template sentences repeat, so some held-out lines also appear in the training text. Real projects deduplicate before splitting." },
      { title: "Underfitting vs overfitting", body: "Both losses high and still falling: underfitting, so train longer or go bigger. Training loss falling while validation rises: overfitting, so add data, shrink the model, or stop earlier.", tryIt: "Train a model in the Train lab and watch the gap between the training and validation curves." },
      { title: "Start small, check everything", body: "Before a long run: check the step-0 loss is about ln(vocab size), make sure the model can overfit one small batch, and read samples regularly. Most bugs show up in those three checks." },
    ],
    quiz: [
      {
        q: "In TinyGPT's batches, how are the targets y made from the inputs x?",
        options: ["y is x reversed", "y is the same window shifted one token later", "y is a different random window", "y is a topic label for the window"],
        answer: 1,
        why: "Position t of y holds the token that follows position t of x, so one window gives 64 next-token examples.",
      },
      {
        q: "Training loss is 0.10 and validation loss is 1.90 and rising. What's the best next move?",
        options: ["Train longer", "Make the model bigger", "Raise the learning rate", "Get more data, add regularization, or stop earlier"],
        answer: 3,
        why: "A large and growing gap means memorization. More data, regularization (like weight decay or dropout) or early stopping all target that.",
      },
      {
        q: "Why check that the step-0 loss is close to ln(vocab size)?",
        options: ["It confirms the model starts out unbiased and the loss is wired up correctly", "PyTorch requires it", "It sets the learning rate", "It guarantees training will succeed"],
        answer: 0,
        why: "A freshly initialized model should be nearly uniform. A much higher starting loss points to bad initialization or a bug in how targets line up.",
      },
    ],
    deeper: {
      math: String.raw`split:  train = tokens[: −5%],   val = tokens[−5% :]
batch:  x_b = tokens[i_b : i_b + 64],   y_b = tokens[i_b + 1 : i_b + 65]
tokens per step = B × T = 64 × 64 = 4,096
passes over the data ≈ 3,000 × 4,096 / 1.6M ≈ 7.7
generalization gap = L_val − L_train`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
data = torch.randint(256, (10_000,))           # stand-in for a tokenized corpus
n_val = len(data) // 20                         # hold out the last 5%, like TinyGPT
train, val = data[:-n_val], data[-n_val:]
def get_batch(d, B=64, T=64):                   # TinyGPT: 64 windows × 64 tokens
    ix = torch.randint(len(d) - T - 1, (B,))
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + T] for i in ix])   # targets = inputs shifted by one
    return x, y
x, y = get_batch(train)
print(x.shape, (x[:, 1:] == y[:, :-1]).all().item())    # torch.Size([64, 64]) True
# Overfitting in miniature: a big net memorizes 64 examples of pure noise.
X, Y, Xv, Yv = torch.randn(64, 32), torch.randint(10, (64,)), torch.randn(64, 32), torch.randint(10, (64,))
net = nn.Sequential(nn.Linear(32, 512), nn.ReLU(), nn.Linear(512, 10))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
for step in range(500):
    loss = F.cross_entropy(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    print(f"train {loss.item():.3f}  val {F.cross_entropy(net(Xv), Yv).item():.3f}")  # train ≈ 0, val > ln 10`,
      codeCaption: "TinyGPT-style batching (targets = inputs shifted by one), and overfitting in miniature: a big net memorizes 64 random labels (training loss ≈ 0) yet does worse than guessing on fresh data (validation above ln 10 ≈ 2.3).",
    },
    pitfall: { myth: "If training loss keeps going down, the model keeps getting better.", truth: "Only validation loss tells you about new text. Once it stops falling while training loss keeps dropping, extra training is mostly memorization." },
    further: [
      { title: "nanoGPT (GitHub)", url: "https://github.com/karpathy/nanoGPT", note: "The simplest real codebase for training GPTs; TinyGPT follows the same shape." },
      { title: "Karpathy: A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/", note: "Practical habits for training without fooling yourself." },
      { title: "Andrej Karpathy: Let's build GPT, from scratch, in code", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "Trains a small GPT from scratch, step by step." },
    ],
  },
  "depth": {
    promise: "After this you'll be able to explain why residual connections and LayerNorm are what make deep networks trainable.",
    keyIdeas: [
      { title: "Deep stacks used to break", body: "Stack many layers that each replace the previous output, and the training signal fades or explodes on its way back. In a toy 50-layer tanh stack, the gradient reaching layer 1 shrinks to about 10⁻¹⁴." },
      { title: "Residuals: add, don't replace", body: "x ← x + f(x). Each layer only learns a correction, and the + passes gradients straight back. This trick from ResNet (2015) is in every transformer." },
      { title: "LayerNorm keeps numbers in range", body: "Before each block, TinyGPT rescales the token's vector to mean 0 and variance 1, then applies a learned scale and shift. No layer receives wildly sized inputs." },
      { title: "Pre-LN: normalize the input, not the stream", body: "TinyGPT, like GPT-2, normalizes what goes into each block and leaves the residual stream untouched (plus one final LayerNorm). This trains more stably than the original 2017 post-LN layout." },
      { title: "Break it to believe it", body: "Removing residuals or LayerNorm and retraining is the fastest way to feel why they're there.", tryIt: "In the Architecture lab, remove the residual connections and retrain." },
    ],
    quiz: [
      {
        q: "Why do residual connections help gradients reach early layers?",
        options: ["The derivative of x + f(x) with respect to x contains an identity term, so the signal gets through even when f's slope is tiny", "They make the model smaller", "They normalize the activations", "They remove the need for a loss"],
        answer: 0,
        why: "∂(x + f(x))/∂x = I + ∂f/∂x. The I term gives the gradient a direct path through every layer.",
      },
      {
        q: "What does LayerNorm do to a token's 64-number vector?",
        options: ["Sorts it", "Shifts and scales it to mean 0 and variance 1, then applies a learned per-dimension scale and shift", "Clips every value to [−1, 1]", "Averages it with the other tokens' vectors"],
        answer: 1,
        why: "The statistics are computed over the 64 features of one token, so each token is normalized independently.",
      },
      {
        q: "In a pre-LN block, where is LayerNorm applied?",
        options: ["After adding to the residual stream", "Only at the very end", "To the input of each attention and MLP block, leaving the residual stream itself unnormalized", "To the gradients"],
        answer: 2,
        why: "x ← x + F(LN(x)). Keeping the stream un-normalized preserves the clean identity path for gradients.",
      },
    ],
    deeper: {
      math: String.raw`residual: x_{l+1} = x_l + F_l(x_l)   ⇒   ∂x_{l+1}/∂x_l = I + ∂F_l/∂x_l
plain:    x_{l+1} = F_l(x_l)         ⇒   gradient = product of L Jacobians (can vanish or explode)
LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β      μ, σ² over the 64 features, ε = 1e-5
pre-LN (TinyGPT, GPT-2):   x ← x + F(LN(x)),  final LN_f before the readout
post-LN (2017 Transformer): x ← LN(x + F(x))`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
D, L = 64, 50                                   # a deliberately deep stack: 50 layers
layers = [nn.Linear(D, D) for _ in range(L)]
def grad_at_layer1(residual, norm):
    x = torch.randn(32, D)
    for f in layers:
        h = torch.tanh(f(F.layer_norm(x, (D,)) if norm else x))
        x = x + h if residual else h            # residual: ADD the update instead of replacing
    for f in layers: f.zero_grad()
    x.pow(2).mean().backward()
    return layers[0].weight.grad.norm().item()
for residual in (False, True):
    for norm in (False, True):
        print(f"residual={residual!s:5} layernorm={norm!s:5} → grad at layer 1: {grad_at_layer1(residual, norm):.1e}")`,
      codeCaption: "A toy 50-layer tanh stack: without residuals or normalization the gradient reaching layer 1 vanishes (about 10⁻¹⁴); either trick keeps it alive, and TinyGPT uses both.",
    },
    pitfall: { myth: "Deeper is always better if you can afford the compute.", truth: "Depth only helps if the signal can flow. Without residuals and normalization, extra layers can make training slower or impossible; even with them, depth has to be balanced against width and data." },
    further: [
      { title: "Deep Residual Learning for Image Recognition (ResNet, He et al., 2015)", url: "https://arxiv.org/abs/1512.03385", note: "Introduced residual connections." },
      { title: "Layer Normalization (Ba et al., 2016)", url: "https://arxiv.org/abs/1607.06450", note: "Introduced Layer Normalization." },
      { title: "On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)", url: "https://arxiv.org/abs/2002.04745", note: "Why pre-LN transformers train more stably than post-LN." },
    ],
  },
  "arch-lab": {
    promise: "After this you'll be able to design an ablation: remove one part, retrain, and measure what it was really doing.",
    keyIdeas: [
      { title: "Ablation = remove and measure", body: "To learn what a part does, take it out (or zero it), keep everything else identical, and compare validation loss. The damage tells you what the part was worth.", tryIt: "In the Architecture lab, switch off one component, retrain, and compare with the baseline curve." },
      { title: "Predict before you run", body: "Without attention, each token sees only itself, so the model collapses to guessing from the current token (roughly a bigram model). Without position embeddings, you'd expect word order to be much harder to use. Guess the damage first, then check: TinyGPT's real result may surprise you." },
      { title: "Not all heads are equal", body: "Studies of trained transformers found many heads can be removed with little loss while a few are critical. TinyGPT's layer 2 head 4 is a good candidate for a critical one: zeroing it drops p(\u201c red\u201d) on the probe from 87% to about 2%." },
      { title: "One change, several seeds", body: "Random initialization alone moves the final loss a little. Run each variant with a few seeds and only trust differences bigger than that spread." },
    ],
    quiz: [
      {
        q: "Removing a component raises validation loss slightly in one run. Before concluding anything, you should…",
        options: ["Rerun both versions with several seeds to see whether the gap beats the noise", "Publish the result", "Double the learning rate for the ablated model only", "Compare training losses instead"],
        answer: 0,
        why: "A single run can't separate a real effect from seed-to-seed noise. Paired runs over a few seeds can.",
      },
      {
        q: "If you remove every attention layer, what can TinyGPT use to predict the next token?",
        options: ["The whole prompt, just less efficiently", "Only the current token and its position", "Nothing at all", "Only the first token"],
        answer: 1,
        why: "Attention is the only part that moves information between positions. Without it each position's prediction depends only on its own token and position.",
      },
      {
        q: "Why must data, steps and learning rate stay the same in an ablation?",
        options: ["To save money", "PyTorch requires it", "Otherwise you can't tell whether the removed part or something else caused the difference", "To make the plots line up"],
        answer: 2,
        why: "A controlled comparison changes exactly one thing, so any difference can be attributed to it.",
      },
    ],
    deeper: {
      math: String.raw`effect(part) = L_val(model without part) − L_val(full model)
noise:  σ = std of L_val across seeds for the same config
trust Δ only if it is clearly bigger than the noise (e.g. paired over seeds, Δ > 2·SE)
zero-ablation of head h: head_h ← 0;   mean-ablation: head_h ← average of head_h
no attention ⇒ p(next | token_i, position i) only`,
      code: String.raw`import torch, torch.nn as nn, torch.nn.functional as F
X = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]]); Y = torch.tensor([0, 1, 1, 0])   # XOR
def run(act, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 16), act, nn.Linear(16, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=0.05)
    for _ in range(300):
        loss = F.cross_entropy(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
for name, act in [("with GELU", nn.GELU()), ("GELU ablated", nn.Identity())]:
    losses = torch.tensor([run(act, seed) for seed in range(5)])     # several seeds, not one
    print(f"{name:13s} final loss {losses.mean():.3f} ± {losses.std():.3f}")
# Without the nonlinearity the two layers collapse into one linear map: stuck at ln 2 ≈ 0.693.`,
      codeCaption: "A minimal ablation with seeds: take the nonlinearity out of a tiny MLP and it can no longer learn XOR, stuck at ln 2 ≈ 0.693 on every seed.",
    },
    pitfall: { myth: "If removing a part barely hurts, that part was useless.", truth: "Other parts can compensate when the model is retrained without it, and average loss can hide effects on specific behaviors. Check targeted prompts as well as the overall number." },
    further: [
      { title: "Are Sixteen Heads Really Better than One? (Michel et al., 2019)", url: "https://arxiv.org/abs/1905.10650", note: "Many attention heads can be pruned; a few cannot." },
      { title: "A Mathematical Framework for Transformer Circuits (Anthropic, 2021)", url: "https://transformer-circuits.pub/2021/framework/index.html", note: "Tools for figuring out what each component computes." },
      { title: "Karpathy: A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/", note: "How to run careful, debuggable experiments." },
    ],
  },
  "scaling": {
    promise: "After this you'll be able to read a scaling-law plot and explain how loss improves predictably with parameters, data and compute.",
    keyIdeas: [
      { title: "Straight lines on log-log plots", body: "Across many orders of magnitude, loss falls as a power law in model size, data and compute. Each doubling buys a small but predictable improvement." },
      { title: "Compute ≈ 6 × parameters × tokens", body: "Training costs about 6 floating-point operations per parameter per token (forward plus backward). TinyGPT: 6 × 170,560 × 12.3M ≈ 1.3 × 10¹³ FLOPs." },
      { title: "Balance model and data", body: "For a fixed compute budget, the Chinchilla study (2022) found parameters and training tokens should grow together, at roughly 20 tokens per parameter. TinyGPT saw about 72 tokens per parameter." },
      { title: "Small runs predict big ones", body: "Labs fit the curve on cheap small models, then extrapolate to decide how big to go.", tryIt: "In the Scaling lab, fit the line through the small runs and predict the next size." },
      { title: "Loss isn't everything", body: "Smooth loss curves can hide abrupt-looking jumps on specific tasks, and there's a floor set by how unpredictable the text is. Scaling laws say how much better, not which skills appear." },
    ],
    quiz: [
      {
        q: "On a log-log plot of loss against parameters, a power law looks like…",
        options: ["An exponential curve", "A staircase", "A circle", "A straight line"],
        answer: 3,
        why: "If L = c · N^(−α), then log L = log c − α · log N: a straight line with slope −α.",
      },
      {
        q: "By the Chinchilla rule of thumb, roughly how many training tokens suit a 1-billion-parameter model?",
        options: ["1 billion", "20 million", "20 billion", "1 trillion"],
        answer: 2,
        why: "About 20 tokens per parameter: 20 × 1B = 20B tokens for compute-optimal training.",
      },
      {
        q: "Why train many small models before one big one?",
        options: ["To build the tokenizer", "Small models are always better", "Big models can't be trained directly", "To fit a scaling curve and predict the big run's loss before paying for it"],
        answer: 3,
        why: "Fitting the trend on cheap runs lets you choose model size and data budget for the expensive run with evidence.",
      },
    ],
    deeper: {
      math: String.raw`Kaplan et al. 2020 (N = non-embedding parameters, D = tokens):
  L(N) ≈ (N_c / N)^α_N,  α_N ≈ 0.076;    L(D) ≈ (D_c / D)^α_D,  α_D ≈ 0.095
Chinchilla 2022:  L(N, D) = E + A / N^α + B / D^β     (fit: α ≈ 0.34, β ≈ 0.28)
compute:          C ≈ 6 · N · D FLOPs
compute-optimal:  N_opt and D_opt both grow ≈ C^0.5   (≈ 20 tokens per parameter)
power law in logs: log L = α · log N_c − α · log N`,
      code: String.raw`import numpy as np
rng = np.random.default_rng(0)
N = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6])          # model sizes (parameters)
L = (8.8e13 / N) ** 0.076 * np.exp(rng.normal(0, 0.005, N.size))   # toy data, Kaplan-style constants
slope, intercept = np.polyfit(np.log(N), np.log(L), 1)    # a power law is a straight line in log-log
print(f"fitted exponent α ≈ {-slope:.3f}")               # ≈ 0.076
print("extrapolated loss at 1e9 params:", round(float(np.exp(intercept + slope * np.log(1e9))), 3))
# Training compute rule of thumb: C ≈ 6 · N · D FLOPs
N_tiny, D_tiny = 170_560, 3000 * 64 * 64
print(f"TinyGPT: {D_tiny:,} tokens seen, ≈ {6 * N_tiny * D_tiny:.1e} FLOPs, {D_tiny / N_tiny:.0f} tokens/param")`,
      codeCaption: "Fitting a power law as a straight line in log-log space (toy data generated from Kaplan-style constants), plus TinyGPT's own compute budget.",
    },
    pitfall: { myth: "Scaling laws say a bigger model is always the better buy.", truth: "They describe the best loss for a given compute budget. A model too big for its data wastes compute, and bigger models cost more every time they run, which is why many deployed models are trained on far more than 20 tokens per parameter." },
    further: [
      { title: "Scaling Laws for Neural Language Models (Kaplan et al., 2020)", url: "https://arxiv.org/abs/2001.08361", note: "The original language-model scaling laws." },
      { title: "Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022)", url: "https://arxiv.org/abs/2203.15556", note: "Revised the balance between model size and data." },
    ],
  },
  "chatgpt": {
    promise: "After this you'll be able to explain the steps that turn a next-token predictor like TinyGPT into a helpful chat assistant.",
    keyIdeas: [
      { title: "Step 1: pretraining (what TinyGPT did)", body: "Predict the next token on a huge pile of text. The result is a base model that continues documents. It knows a lot but doesn't reliably follow instructions." },
      { title: "Chat is just formatted text", body: "TinyGPT's corpus has “q: … a: …” lines, so after “q: what does the cat say? a:” it writes “the cat says meow”. Chat models work the same way, with special tokens marking who is speaking.", tryIt: "In the X-ray, run the preset “q: what does the owl say? a:”." },
      { title: "Step 2: supervised fine-tuning", body: "Keep training the same model on example conversations that show ideal answers. Same loss, different data: now the most likely continuation is a helpful reply." },
      { title: "Step 3: learning from preferences", body: "People compare two answers and pick the better one. RLHF trains a reward model on those choices and optimizes the chat model against it; DPO learns from the preference pairs directly." },
      { title: "Same engine underneath", body: "It still generates one token at a time with softmax and sampling. The big differences are scale (billions of parameters, trillions of tokens), data, and these post-training steps." },
    ],
    quiz: [
      {
        q: "What does a base model (pretraining only) do with the prompt “Write a poem about cats”?",
        options: ["Always writes a polished poem", "Refuses", "Continues the text in whatever way is most likely, which might be more instructions or a list of prompts", "Asks a clarifying question"],
        answer: 2,
        why: "A base model imitates documents. Following instructions is a behavior that fine-tuning makes reliable.",
      },
      {
        q: "What changes between pretraining and supervised fine-tuning?",
        options: ["The data: curated example conversations instead of raw text; the next-token loss stays the same", "The architecture is replaced", "The tokenizer is retrained from scratch", "Gradients are no longer used"],
        answer: 0,
        why: "SFT is ordinary next-token training on different data, usually counting the loss only on the assistant's tokens.",
      },
      {
        q: "In RLHF, what is the reward model trained on?",
        options: ["Raw internet text", "Human comparisons saying which of two responses is better", "The chat model's own loss curve", "Token frequencies"],
        answer: 1,
        why: "It learns to score responses the way people ranked them; the chat model is then optimized to get high scores while staying close to its starting point.",
      },
    ],
    deeper: {
      math: String.raw`pretraining:  L = −Σ_t ln p_θ(x_t | x_<t)                   all tokens
SFT:          L = −Σ_{t ∈ answer} ln p_θ(x_t | x_<t)        answer tokens only
reward model: L_RM = −ln σ(r(x, y_good) − r(x, y_bad))
RLHF:         maximize  E[r(x, y)] − β · KL(π_θ ‖ π_SFT)
DPO:          L = −ln σ(β [(ln π_θ(y_w|x) − ln π_ref(y_w|x)) − (ln π_θ(y_l|x) − ln π_ref(y_l|x))])`,
      code: String.raw`import torch, torch.nn.functional as F
torch.manual_seed(0)
# 1) Supervised fine-tuning (SFT): the same next-token loss, but only on the answer tokens.
tokens = torch.randint(256, (1, 10))              # e.g. "q: what does the owl say? a: hoot"
logits = torch.randn(1, 10, 256)                  # model output at each position
n_prompt = 6                                      # the first 6 tokens are the question
targets = tokens[:, 1:].clone()                   # position t predicts token t+1
targets[:, : n_prompt - 1] = -100                 # ignore: don't learn to write questions
sft = F.cross_entropy(logits[:, :-1].reshape(-1, 256), targets.reshape(-1), ignore_index=-100)
# 2) Preference tuning (DPO): raise the preferred answer relative to a frozen reference model.
def dpo_loss(lp_good, lp_bad, ref_good, ref_bad, beta=0.1):   # summed log-probs of each answer
    return -F.logsigmoid(beta * ((lp_good - ref_good) - (lp_bad - ref_bad)))
t = torch.tensor
print(round(sft.item(), 3), round(dpo_loss(t(-12.0), t(-15.0), t(-13.0), t(-13.0)).item(), 3))`,
      codeCaption: "SFT is the usual next-token loss with the prompt masked out (ignore_index=-100); DPO turns a preference pair into a loss using a frozen reference model.",
    },
    pitfall: { myth: "ChatGPT looks up its answers on the internet or in a database.", truth: "The core model generates text from its parameters, one token at a time, just like TinyGPT. Some products add tools such as web search, but that's an extra system around the model, not how the model itself works." },
    further: [
      { title: "Andrej Karpathy: Intro to Large Language Models", url: "https://www.youtube.com/watch?v=zjkBMFhNj_g", note: "How real LLMs are trained and turned into assistants." },
      { title: "Training language models to follow instructions with human feedback (InstructGPT, 2022)", url: "https://arxiv.org/abs/2203.02155", note: "The SFT + RLHF recipe behind the first ChatGPT-style models." },
      { title: "Direct Preference Optimization (Rafailov et al., 2023)", url: "https://arxiv.org/abs/2305.18290", note: "A simpler alternative to RLHF." },
    ],
  },
  "research-loop": {
    promise: "After this you'll be able to run a fair experiment on a model idea: hypothesis, baseline, one change, several seeds, honest conclusion.",
    keyIdeas: [
      { title: "Start from a working baseline", body: "Every idea is measured against something that already works, such as TinyGPT's 3-layer recipe with validation loss around 0.46. No baseline, no evidence." },
      { title: "Change one thing", body: "Swap one component (an activation, a normalization, a learning rate) and keep data, steps and evaluation identical. Change two things at once and you can't tell which one mattered." },
      { title: "Predict first, then run", body: "Write down what you expect and why. A surprise is where you learn the most, but only if you made a prediction.", tryIt: "On the Invent page, write a hypothesis, run it against the baseline, and record the result." },
      { title: "Noise is real: use seeds", body: "Rerun each version with a few random seeds. Trust a gain only if it's clearly bigger than the run-to-run spread." },
      { title: "Small and fast beats big and slow", body: "Many quick runs teach more than one long one. Iterate at small scale (TinyGPT trains in about 80 seconds), then check that the winner still wins when scaled up." },
    ],
    quiz: [
      {
        q: "Your new activation beats the baseline by 0.01 validation loss in one run. The spread across seeds is ±0.03. What's your conclusion?",
        options: ["It's a clear win", "Not distinguishable from noise yet; run more seeds", "The baseline is broken", "The activation is worse"],
        answer: 1,
        why: "A difference smaller than the seed-to-seed spread could easily be luck. More paired runs shrink the uncertainty.",
      },
      {
        q: "Your variant trained for twice as many steps as the baseline and won. What's the problem?",
        options: ["Nothing; it won", "It should have trained 4× longer", "The comparison is unfair: it got more compute, so the gain may just come from training longer", "Validation loss can't be compared between models"],
        answer: 2,
        why: "Compare at equal compute (or plot both against compute). Otherwise you are measuring the budget, not the idea.",
      },
      {
        q: "Why judge ideas by validation loss rather than training loss?",
        options: ["Validation loss measures how well the model handles text it never trained on", "It's always lower", "Training loss isn't computed", "It's faster to compute"],
        answer: 0,
        why: "An idea that only lowers training loss may just memorize better; validation loss tracks generalization.",
      },
    ],
    deeper: {
      math: String.raw`hypothesis: "change X lowers L_val at equal compute, because …"
arms: baseline B and variant V, same seeds s = 1 … n
paired gain: g_s = L_val(B, s) − L_val(V, s)
mean ḡ, standard error SE = std(g) / √n
call it real only if ḡ > ~2·SE, then re-test at a larger scale`,
      code: String.raw`import statistics as st, torch, torch.nn as nn, torch.nn.functional as F
def run(cfg, seed, steps=200):
    torch.manual_seed(seed)                            # same seed → same data for both arms
    X, Xv = torch.randn(512, 20), torch.randn(512, 20)
    Y, Yv = (X[:, :5].sum(1) > 0).long(), (Xv[:, :5].sum(1) > 0).long()
    net = nn.Sequential(nn.Linear(20, 32), nn.GELU(), nn.Linear(32, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"])
    for _ in range(steps):                             # identical budget for every run
        loss = F.cross_entropy(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.cross_entropy(net(Xv), Yv).item()     # judge on held-out data
baseline, idea = dict(lr=1e-3), dict(lr=1e-2)          # change ONE thing
a = [run(baseline, s) for s in range(5)]
b = [run(idea, s) for s in range(5)]
gain = [x - y for x, y in zip(a, b)]                   # paired by seed
se = st.stdev(gain) / len(gain) ** 0.5
print(f"baseline {st.mean(a):.3f}  idea {st.mean(b):.3f}  gain {st.mean(gain):.3f} ± {se:.3f}")
print("bigger than noise?", abs(st.mean(gain)) > 2 * se)`,
      codeCaption: "A paired, multi-seed comparison on a toy task. Here the “idea” (a 10× higher learning rate) does not beat the noise, which is the most common honest outcome.",
    },
    pitfall: { myth: "If my idea gets a lower loss, it works.", truth: "Rule out unfair comparisons first: more parameters, more compute, a different data split, a lucky seed, or a bug that leaks validation data. Many apparent wins disappear under a fair rerun." },
    further: [
      { title: "Karpathy: A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/", note: "Habits that keep experiments honest." },
      { title: "Google Research: Deep Learning Tuning Playbook", url: "https://github.com/google-research/tuning_playbook", note: "A systematic guide to tuning and comparing runs." },
      { title: "nanoGPT (GitHub)", url: "https://github.com/karpathy/nanoGPT", note: "A clean baseline to fork for your own ideas." },
    ],
  },
  "frontier": {
    promise: "After this you'll be able to name the main open directions in model design and pick one to prototype at TinyGPT scale.",
    keyIdeas: [
      { title: "Beyond quadratic attention", body: "Attention compares every token with every other, so cost grows with the square of the context length. Linear attention and state-space models like Mamba keep a fixed-size memory instead, with constant cost per new token." },
      { title: "Mixture of experts", body: "Replace one MLP with many expert MLPs plus a router that sends each token to only a few of them. You get far more parameters without more compute per token." },
      { title: "Better ways to encode position", body: "TinyGPT learns one vector per position, up to 64. Many modern models use rotary embeddings (RoPE), which build relative distance directly into the attention dot product." },
      { title: "Thinking longer at test time", body: "Models that write out intermediate reasoning before answering can solve harder problems, trading extra inference compute for accuracy." },
      { title: "Understanding what's inside", body: "Interpretability hunts for circuits (like TinyGPT's layer 2 head 4, which fetches the item's color on stories like the probe) and features (directions in the residual stream). Sparse autoencoders try to split polysemantic neurons into cleaner features.", tryIt: "On the Invent page, pick one of these ideas and test it against the TinyGPT baseline." },
    ],
    quiz: [
      {
        q: "Why does standard attention get expensive for very long contexts?",
        options: ["It compares every pair of tokens, so cost grows with the square of the length", "The vocabulary grows", "The embeddings get longer", "It needs a new tokenizer"],
        answer: 0,
        why: "T tokens means T² query-key scores per head per layer. Doubling the context quadruples that work.",
      },
      {
        q: "What is the main appeal of mixture-of-experts layers?",
        options: ["Fewer total parameters", "They need no training", "Many more parameters, while each token uses only a few experts, so compute per token stays modest", "They remove attention"],
        answer: 2,
        why: "Capacity grows with the number of experts, but each token only pays for the experts its router picks.",
      },
      {
        q: "In the linear-attention recurrence S_t = S_{t−1} + k_t v_tᵀ, what stays constant as the sequence grows?",
        options: ["The output", "The size of the state S (d × d)", "The number of tokens", "The loss"],
        answer: 1,
        why: "All past tokens are folded into one fixed-size matrix, so each new token costs the same no matter how long the context is.",
      },
    ],
    deeper: {
      math: String.raw`softmax attention: o_t = Σ_{j≤t} softmax_j(q_t · k_j) v_j            O(T²) for T tokens
linear attention:  o_t = q_t S_t,   S_t = S_{t−1} + k_t v_tᵀ          O(T), fixed d × d state
state-space (Mamba): h_t = Ā_t h_{t−1} + B̄_t x_t,  y_t = C_t h_t   (Ā_t, B̄_t, C_t depend on the input)
mixture of experts:  y = Σ_{i ∈ top-k(router(x))} g_i(x) · Expert_i(x)
RoPE: rotate q and k by angles ∝ position, so q_m · k_n depends on m − n`,
      code: String.raw`import torch
torch.manual_seed(0)
T, d = 6, 8
q, k, v = torch.randn(3, T, d).unbind(0)
# Parallel form: attention WITHOUT softmax. Cost grows with T² like normal attention.
out_parallel = ((q @ k.T) * torch.tril(torch.ones(T, T))) @ v
# Recurrent form: the same outputs from a fixed-size d×d memory, constant cost per token.
S, outs = torch.zeros(d, d), []
for t in range(T):
    S = S + torch.outer(k[t], v[t])       # write: add key ⊗ value to the memory
    outs.append(q[t] @ S)                 # read: ask the memory with the query
print(torch.allclose(out_parallel, torch.stack(outs), atol=1e-5))   # True`,
      codeCaption: "Causal linear attention computed two ways, as a masked T × T product and as an RNN with a fixed d × d memory, gives identical outputs. Real linear-attention models add feature maps, normalization or decay.",
    },
    pitfall: { myth: "The transformer is finished, and testing new ideas needs giant compute.", truth: "Many architecture and optimizer ideas can get a first test on small models. A fair, multi-seed comparison at TinyGPT scale is a legitimate first filter, though some effects only show up at scale, so winners need re-checking bigger." },
    further: [
      { title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)", url: "https://arxiv.org/abs/2312.00752", note: "Selective state-space models as an attention alternative." },
      { title: "Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2021)", url: "https://arxiv.org/abs/2101.03961", note: "A simple, influential mixture-of-experts design." },
      { title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Anthropic, 2023)", url: "https://transformer-circuits.pub/2023/monosemantic-features", note: "Sparse autoencoders for finding interpretable features." },
    ],
  },
};
