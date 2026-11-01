// Byte-pair-encoding tokenizer — mirrors train/bpe.py exactly.
export interface TokenPiece {
  id: number;
  text: string;
  word: number; // which pre-token ("word") it came from
  merges: string[]; // merge steps that built this token, e.g. ["b+a", "ba+ll"]
}

const PAT = / ?[a-z]+| ?[^a-z\s]|\n| +/g;

export class Tokenizer {
  vocab: string[];
  index = new Map<string, number>();
  rank = new Map<string, number>();
  constructor(vocab: string[], merges: [string, string][]) {
    this.vocab = vocab;
    vocab.forEach((t, i) => this.index.set(t, i));
    merges.forEach(([a, b], i) => this.rank.set(a + "\u0000" + b, i));
  }

  /** Normalize to the tiny model's alphabet (lowercase, known punctuation). */
  normalize(text: string) {
    return text.toLowerCase().replace(/[^\na-z .,?!:']/g, " ");
  }

  encode(text: string): TokenPiece[] {
    const out: TokenPiece[] = [];
    const words = this.normalize(text).match(PAT) ?? [];
    words.forEach((w, wi) => {
      let s = [...w].map((c) => ({ t: c, m: [] as string[] }));
      while (s.length > 1) {
        let best = Infinity, bi = -1;
        for (let i = 0; i < s.length - 1; i++) {
          const r = this.rank.get(s[i].t + "\u0000" + s[i + 1].t);
          if (r !== undefined && r < best) { best = r; bi = i; }
        }
        if (bi < 0) break;
        const a = s[bi], b = s[bi + 1];
        const merged = { t: a.t + b.t, m: [...a.m, ...b.m, `${show(a.t)} + ${show(b.t)}`] };
        s = [...s.slice(0, bi), merged, ...s.slice(bi + 2)];
      }
      for (const p of s) out.push({ id: this.index.get(p.t) ?? 0, text: p.t, word: wi, merges: p.m });
    });
    return out;
  }

  decode(ids: number[]) {
    return ids.map((i) => this.vocab[i]).join("");
  }
}

/** Make whitespace visible: " ball" → "␣ball", "\n" → "⏎" */
export function show(t: string) {
  return t.replace(/ /g, "␣").replace(/\n/g, "⏎");
}
