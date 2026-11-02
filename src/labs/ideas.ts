// Frontier directions for the "Invent your own" idea board. Only well-known arXiv references.
import type { LucideIcon } from "lucide-react";
import { GraduationCap, Hourglass, Library, RotateCw, Sparkles, Split, Waves, Zap } from "lucide-react";
import type { ChangeId } from "./pyscript";

export interface Idea {
  id: string; icon: LucideIcon; title: string; idea: string; problem: string; tiny: string;
  refs: { label: string; url: string }[];
  change: ChangeId; custom?: string; hypothesis: string;
}

export const IDEAS: Idea[] = [
  {
    id: "moe", icon: Split, title: "Mixture of experts",
    idea: "Many MLPs per block; a small router sends each token to one (or a few). Lots of parameters, but each token only pays for one expert.",
    problem: "Bigger models are better, but every parameter normally costs compute on every token.",
    tiny: "2 experts per block with top-1 routing. Compare loss at the same steps, and watch whether the router uses both experts.",
    refs: [{ label: "Switch Transformer (Fedus et al. 2021)", url: "https://arxiv.org/abs/2101.03961" }, { label: "Sparsely-gated MoE (Shazeer et al. 2017)", url: "https://arxiv.org/abs/1701.06538" }],
    change: "moe2", hypothesis: "Two experts per block lower val loss at the same number of steps, because each token gets a more specialised MLP.",
  },
  {
    id: "ssm", icon: Waves, title: "State-space models",
    idea: "Replace attention with a recurrent state that is updated token by token. Selective SSMs (Mamba) let the input decide what to keep or forget.",
    problem: "Attention compares every token with every earlier token: cost grows with the square of the context length.",
    tiny: "Swap attention in one block for a gated recurrence h = a·h + b·x (a, b computed from x). Does the color-copy prompt still work?",
    refs: [{ label: "Mamba (Gu & Dao 2023)", url: "https://arxiv.org/abs/2312.00752" }],
    change: "custom", custom: "Replace self-attention in block 1 with a gated linear recurrence: h_t = sigmoid(W_a x_t) * h_{t-1} + (W_b x_t), output W_o h_t.",
    hypothesis: "A gated recurrence in place of attention in one block costs little loss, because short stories need little long-range lookup.",
  },
  {
    id: "linear", icon: Zap, title: "Linear attention",
    idea: "Drop the softmax: with a feature map φ, attention becomes φ(Q)·(running sum of φ(K)ᵀV). It can run like an RNN, linear in length.",
    problem: "Softmax attention is quadratic in context length, and its memory grows as the context grows.",
    tiny: "Use φ(x) = elu(x) + 1 with causal cumulative sums. Compare loss and ms/step at context 64 and 256.",
    refs: [{ label: "Transformers are RNNs (Katharopoulos et al. 2020)", url: "https://arxiv.org/abs/2006.16236" }],
    change: "custom", custom: "Linear attention: replace softmax(QK^T)V with phi(Q) cumsum(phi(K)^T V) / phi(Q) cumsum(phi(K)), phi(x) = elu(x) + 1.",
    hypothesis: "Linear attention is faster per step at long context but reaches a higher val loss, because it can't focus as sharply as softmax.",
  },
  {
    id: "rope", icon: RotateCw, title: "RoPE & long context",
    idea: "Encode position by rotating queries and keys. The attention score then depends on the distance between tokens, not their absolute slots.",
    problem: "Learned position vectors only exist for positions seen in training; models stumble past that length.",
    tiny: "Swap learned positions for RoPE. Then train at context 64 and evaluate at 128.",
    refs: [{ label: "RoFormer (Su et al. 2021)", url: "https://arxiv.org/abs/2104.09864" }],
    change: "rope", hypothesis: "RoPE matches learned positions at context 64 and degrades less when evaluated at context 128.",
  },
  {
    id: "retrieval", icon: Library, title: "Retrieval",
    idea: "Look up relevant chunks of text from a database and let the model read them. Knowledge lives outside the weights.",
    problem: "Storing every fact in the weights needs huge models, and facts go stale.",
    tiny: "For each q: question, prepend the matching fact sentence found by keyword search. Does a 1-layer model answer better?",
    refs: [{ label: "RETRO (Borgeaud et al. 2021)", url: "https://arxiv.org/abs/2112.04426" }, { label: "RAG (Lewis et al. 2020)", url: "https://arxiv.org/abs/2005.11401" }],
    change: "custom", custom: "Retrieval: before each training sequence, prepend the corpus sentence with the most word overlap (keyword search), separated by a newline.",
    hypothesis: "Prepending a retrieved fact sentence lowers loss on q:/a: lines more than adding a layer does.",
  },
  {
    id: "diffusion", icon: Sparkles, title: "Diffusion language models",
    idea: "Start from a fully masked sequence and fill it in over several denoising steps, all positions in parallel, instead of left to right.",
    problem: "Left-to-right decoding is one token per step and can never revise what it already wrote.",
    tiny: "Mask a random fraction of tokens, predict them with bidirectional attention, then sample by unmasking in 8 steps.",
    refs: [{ label: "D3PM: discrete diffusion (Austin et al. 2021)", url: "https://arxiv.org/abs/2107.03006" }],
    change: "custom", custom: "Masked diffusion: no causal mask; each batch masks a random fraction t~U(0,1) of tokens with a [MASK] id and trains to predict them.",
    hypothesis: "A masked-diffusion model of the same size writes grammatical tiny-world sentences in 8 parallel steps.",
  },
  {
    id: "ttc", icon: Hourglass, title: "Test-time compute",
    idea: "Spend more compute when answering: sample several answers and vote, or let the model write intermediate steps before the answer.",
    problem: "One forward pass is a fixed amount of thinking; some questions need more.",
    tiny: "On \"q: what color is the …? a:\" prompts, sample 8 answers and take a majority vote. Compare accuracy with greedy decoding.",
    refs: [{ label: "Chain-of-thought (Wei et al. 2022)", url: "https://arxiv.org/abs/2201.11903" }, { label: "Self-consistency (Wang et al. 2022)", url: "https://arxiv.org/abs/2203.11171" }],
    change: "custom", custom: "No training change: at eval time sample 8 answers per q:/a: prompt at temperature 0.8 and majority-vote; report accuracy vs greedy.",
    hypothesis: "Majority voting over 8 samples fixes near coin-flip mistakes like \"yellow\" vs \"pink\".",
  },
  {
    id: "distill", icon: GraduationCap, title: "Distillation",
    idea: "Train a small student to match a big teacher's whole probability distribution (soft targets), not just the one right token.",
    problem: "Big models are expensive to run. Small ones trained from scratch learn less from the same data.",
    tiny: "Teacher: a 6-layer TinyGPT. Student: 1 layer, trained on KL to the teacher's softmax at temperature 2 vs plain labels.",
    refs: [{ label: "Distilling the Knowledge in a Neural Network (Hinton et al. 2015)", url: "https://arxiv.org/abs/1503.02531" }],
    change: "custom", custom: "Distillation: load a trained 6-layer teacher; loss = KL(student/T || teacher/T) * T^2 with T = 2, mixed 50/50 with cross-entropy.",
    hypothesis: "A 1-layer student distilled from a 6-layer teacher beats the same student trained on labels alone.",
  },
];
