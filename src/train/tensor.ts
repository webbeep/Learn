// A tiny reverse-mode autograd engine on Float32Array tensors, just big enough for a GPT.
// Every op computes its output and, if a tape is recording and an input needs gradients,
// pushes a closure that adds d(loss)/d(input) into input.grad. backward() replays the tape
// in reverse. All tensors are row-major 2-D [rows, cols] (or 1-D vectors).

export class Tensor {
  grad: Float32Array | null = null;
  constructor(public data: Float32Array, public shape: number[], public requiresGrad = false) {}
  get size() { return this.data.length; }
  /** The gradient buffer, allocated on first use. */
  g() { return (this.grad ??= new Float32Array(this.data.length)); }
  static zeros(shape: number[], requiresGrad = false) { return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape, requiresGrad); }
}

let tape: (() => void)[] | null = null;

/** Run fn while recording a tape; returns whatever fn returns. */
export function recording<T>(fn: () => T): T { tape = []; return fn(); }
/** Run fn without recording anything (eval / sampling). */
export function noGrad<T>(fn: () => T): T { const saved = tape; tape = null; try { return fn(); } finally { tape = saved; } }

/** Backpropagate from `loss` (seeded with 1s, or with `seed` = d(objective)/d(loss) for a non-scalar output). */
export function backward(loss: Tensor, seed?: Float32Array) {
  if (!tape) throw new Error("backward() called without a recorded tape");
  if (seed) loss.g().set(seed); else loss.g().fill(1);
  for (let i = tape.length - 1; i >= 0; i--) tape[i]();
  tape = null;
}

function track(out: Tensor, inputs: (Tensor | null | undefined)[], fn: () => void): Tensor {
  if (tape && inputs.some((t) => t?.requiresGrad)) { out.requiresGrad = true; tape.push(() => { if (out.grad) fn(); }); }
  return out;
}

// ---------------------------------------------------------------- ops

/** Gather rows of W[V, D] by integer ids → [N, D]. */
export function embedding(W: Tensor, ids: ArrayLike<number>): Tensor {
  const D = W.shape[1], N = ids.length, y = new Float32Array(N * D), w = W.data;
  for (let n = 0; n < N; n++) y.set(w.subarray(ids[n] * D, ids[n] * D + D), n * D);
  const out = new Tensor(y, [N, D]);
  return track(out, [W], () => {
    const dy = out.grad!, dW = W.g();
    for (let n = 0; n < N; n++) { const o = ids[n] * D, p = n * D; for (let d = 0; d < D; d++) dW[o + d] += dy[p + d]; }
  });
}

/** out[n, m] (+)= Σ_k A[n, k]·B[m, k] — every inner loop is a contiguous dot product, 2×2 register-blocked. */
export function mmT(A: Float32Array, B: Float32Array, out: Float32Array, N: number, M: number, K: number, acc = false) {
  let n = 0;
  for (; n + 1 < N; n += 2) {
    const a0 = n * K, a1 = a0 + K, o0 = n * M, o1 = o0 + M; let m = 0;
    for (; m + 1 < M; m += 2) {
      const b0 = m * K, b1 = b0 + K; let s00 = 0, s01 = 0, s10 = 0, s11 = 0;
      for (let k = 0; k < K; k++) { const x0 = A[a0 + k], x1 = A[a1 + k], y0 = B[b0 + k], y1 = B[b1 + k]; s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1; }
      if (acc) { out[o0 + m] += s00; out[o0 + m + 1] += s01; out[o1 + m] += s10; out[o1 + m + 1] += s11; }
      else { out[o0 + m] = s00; out[o0 + m + 1] = s01; out[o1 + m] = s10; out[o1 + m + 1] = s11; }
    }
    if (m < M) { const b0 = m * K; let s0 = 0, s1 = 0; for (let k = 0; k < K; k++) { s0 += A[a0 + k] * B[b0 + k]; s1 += A[a1 + k] * B[b0 + k]; } if (acc) { out[o0 + m] += s0; out[o1 + m] += s1; } else { out[o0 + m] = s0; out[o1 + m] = s1; } }
  }
  if (n < N) {
    const a0 = n * K, o0 = n * M;
    for (let m = 0; m < M; m++) { const b0 = m * K; let s = 0; for (let k = 0; k < K; k++) s += A[a0 + k] * B[b0 + k]; if (acc) out[o0 + m] += s; else out[o0 + m] = s; }
  }
}
export function transpose(x: Float32Array, R: number, Cc: number) {
  const y = new Float32Array(R * Cc);
  for (let r = 0; r < R; r++) { const o = r * Cc; for (let c = 0; c < Cc; c++) y[c * R + r] = x[o + c]; }
  return y;
}

/** y = x · Wᵀ + b, with x [N, K], W [M, K] (PyTorch layout), b [M]. */
export function linear(x: Tensor, W: Tensor, b?: Tensor | null): Tensor {
  const [N, K] = x.shape, M = W.shape[0], y = new Float32Array(N * M);
  mmT(x.data, W.data, y, N, M, K);
  if (b) { const bd = b.data; for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) y[n * M + m] += bd[m]; }
  const out = new Tensor(y, [N, M]);
  return track(out, [x, W, b], () => {
    const dy = out.grad!;
    if (x.requiresGrad) mmT(dy, transpose(W.data, M, K), x.g(), N, K, M, true); // dx = dy · W
    if (W.requiresGrad) mmT(transpose(dy, N, M), transpose(x.data, N, K), W.g(), M, K, N, true); // dW = dyᵀ · x
    if (b?.requiresGrad) { const db = b.g(); for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) db[m] += dy[n * M + m]; }
  });
}

/** Elementwise a + b. If b has fewer elements it is broadcast over rows (b [cols]). */
export function add(a: Tensor, b: Tensor): Tensor {
  const n = a.size, m = b.size, y = new Float32Array(n), ad = a.data, bd = b.data;
  if (m === n) for (let i = 0; i < n; i++) y[i] = ad[i] + bd[i];
  else for (let i = 0; i < n; i++) y[i] = ad[i] + bd[i % m];
  const out = new Tensor(y, a.shape.slice());
  return track(out, [a, b], () => {
    const dy = out.grad!;
    if (a.requiresGrad) { const da = a.g(); for (let i = 0; i < n; i++) da[i] += dy[i]; }
    if (b.requiresGrad) { const db = b.g(); for (let i = 0; i < n; i++) db[i % m] += dy[i]; }
  });
}

const C = Math.sqrt(2 / Math.PI);
/** GELU, tanh approximation (as in GPT-2 and the real TinyGPT). */
export function gelu(x: Tensor): Tensor {
  const n = x.size, xd = x.data, y = new Float32Array(n), th = new Float32Array(n);
  for (let i = 0; i < n; i++) { const v = xd[i], t = Math.tanh(C * (v + 0.044715 * v * v * v)); th[i] = t; y[i] = 0.5 * v * (1 + t); }
  const out = new Tensor(y, x.shape.slice());
  return track(out, [x], () => {
    const dy = out.grad!, dx = x.g();
    for (let i = 0; i < n; i++) {
      const v = xd[i], t = th[i];
      dx[i] += dy[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * C * (1 + 3 * 0.044715 * v * v));
    }
  });
}

/** LayerNorm over the last dim: (x − mean)/√(var + 1e-5) · g + b. */
export function layerNorm(x: Tensor, g: Tensor, b: Tensor): Tensor {
  const [N, D] = x.shape, xd = x.data, gd = g.data, bd = b.data;
  const y = new Float32Array(N * D), xh = new Float32Array(N * D), inv = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    const o = n * D; let mu = 0; for (let d = 0; d < D; d++) mu += xd[o + d]; mu /= D;
    let va = 0; for (let d = 0; d < D; d++) { const c = xd[o + d] - mu; va += c * c; } va /= D;
    const r = 1 / Math.sqrt(va + 1e-5); inv[n] = r;
    for (let d = 0; d < D; d++) { const h = (xd[o + d] - mu) * r; xh[o + d] = h; y[o + d] = h * gd[d] + bd[d]; }
  }
  const out = new Tensor(y, [N, D]);
  return track(out, [x, g, b], () => {
    const dy = out.grad!, dx = x.requiresGrad ? x.g() : null, dg = g.requiresGrad ? g.g() : null, db = b.requiresGrad ? b.g() : null;
    for (let n = 0; n < N; n++) {
      const o = n * D; let s1 = 0, s2 = 0;
      for (let d = 0; d < D; d++) {
        const dyv = dy[o + d], h = xh[o + d];
        if (dg) dg[d] += dyv * h; if (db) db[d] += dyv;
        const dh = dyv * gd[d]; s1 += dh; s2 += dh * h;
      }
      if (dx) { const r = inv[n] / D; for (let d = 0; d < D; d++) dx[o + d] += r * (D * dy[o + d] * gd[d] - s1 - xh[o + d] * s2); }
    }
  });
}

/**
 * Causal multi-head self-attention, fused. qkv [B·T, 3D] holds q | k | v per row.
 * Returns the concatenated head outputs [B·T, D] and the attention weights [B, H, T, T].
 */
export function attention(qkv: Tensor, B: number, T: number, H: number): { out: Tensor; probs: Float32Array } {
  const D = qkv.shape[1] / 3, hd = D / H, S = 3 * D, scale = 1 / Math.sqrt(hd), q = qkv.data;
  const P = new Float32Array(B * H * T * T), y = new Float32Array(B * T * D);
  for (let bb = 0; bb < B; bb++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
    const qo = (bb * T + i) * S + h * hd, po = ((bb * H + h) * T + i) * T;
    let mx = -Infinity;
    for (let j = 0; j <= i; j++) {
      const ko = (bb * T + j) * S + D + h * hd; let s = 0;
      for (let c = 0; c < hd; c++) s += q[qo + c] * q[ko + c];
      s *= scale; P[po + j] = s; if (s > mx) mx = s;
    }
    let z = 0; for (let j = 0; j <= i; j++) { const e = Math.exp(P[po + j] - mx); P[po + j] = e; z += e; }
    const yo = (bb * T + i) * D + h * hd;
    for (let j = 0; j <= i; j++) {
      const p = (P[po + j] /= z), vo = (bb * T + j) * S + 2 * D + h * hd;
      for (let c = 0; c < hd; c++) y[yo + c] += p * q[vo + c];
    }
  }
  const out = new Tensor(y, [B * T, D]);
  track(out, [qkv], () => {
    const dy = out.grad!, dq = qkv.g(), dP = new Float32Array(T);
    for (let bb = 0; bb < B; bb++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
      const qo = (bb * T + i) * S + h * hd, po = ((bb * H + h) * T + i) * T, yo = (bb * T + i) * D + h * hd;
      let dot = 0;
      for (let j = 0; j <= i; j++) {
        const vo = (bb * T + j) * S + 2 * D + h * hd, p = P[po + j]; let s = 0;
        for (let c = 0; c < hd; c++) { s += dy[yo + c] * q[vo + c]; dq[vo + c] += p * dy[yo + c]; }
        dP[j] = s; dot += p * s;
      }
      for (let j = 0; j <= i; j++) {
        const ds = P[po + j] * (dP[j] - dot) * scale; if (ds === 0) continue;
        const ko = (bb * T + j) * S + D + h * hd;
        for (let c = 0; c < hd; c++) { dq[qo + c] += ds * q[ko + c]; dq[ko + c] += ds * q[qo + c]; }
      }
    }
  });
  return { out, probs: P };
}

/** Mean softmax cross-entropy of logits [N, V] against integer targets → scalar [1]. */
export function crossEntropy(logits: Tensor, targets: ArrayLike<number>): Tensor {
  const [N, V] = logits.shape, x = logits.data;
  const lse = new Float32Array(N); let loss = 0;
  for (let n = 0; n < N; n++) {
    const o = n * V; let mx = -Infinity; for (let v = 0; v < V; v++) if (x[o + v] > mx) mx = x[o + v];
    let z = 0; for (let v = 0; v < V; v++) z += Math.exp(x[o + v] - mx);
    lse[n] = mx + Math.log(z); loss += lse[n] - x[o + targets[n]];
  }
  const out = new Tensor(Float32Array.of(loss / N), [1]);
  return track(out, [logits], () => {
    const s = out.grad![0] / N, dl = logits.g();
    for (let n = 0; n < N; n++) {
      const o = n * V, l = lse[n];
      for (let v = 0; v < V; v++) dl[o + v] += s * Math.exp(x[o + v] - l);
      dl[o + targets[n]] -= s;
    }
  });
}

/** Rows [start, start+count) of a [N, D] tensor (differentiable). */
export function rows(x: Tensor, start: number, count: number): Tensor {
  const D = x.shape[1], out = new Tensor(x.data.slice(start * D, (start + count) * D), [count, D]);
  return track(out, [x], () => { const dy = out.grad!, dx = x.g(); for (let i = 0; i < count * D; i++) dx[start * D + i] += dy[i]; });
}

export function l2(a: Float32Array | null) { if (!a) return 0; let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); }
