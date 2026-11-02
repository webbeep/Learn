// A tiny scalar autograd engine (micrograd-style), written for the Backprop lab.
// Every Value remembers which Values produced it and how to push its gradient back to them.

export type Op = "" | "+" | "×" | "−" | "^" | "tanh" | "relu" | "exp" | "log" | "neg" | "σ";

let nextId = 0, visitEpoch = 0;

const noop = () => {};

export class Value {
  data: number;
  grad = 0;
  readonly prev: Value[];
  readonly op: Op;
  label: string;
  readonly id: number;
  /** For "^": the exponent. */
  readonly k: number;
  _backward: () => void = noop;
  _seen = 0;

  constructor(data: number, prev: Value[] = [], op: Op = "", label = "", k = 0) {
    this.data = data; this.prev = prev; this.op = op; this.label = label; this.k = k; this.id = nextId++;
  }

  static of(x: Value | number): Value { return x instanceof Value ? x : new Value(x); }

  add(o: Value | number, label = ""): Value {
    const b = Value.of(o), out = new Value(this.data + b.data, [this, b], "+", label);
    out._backward = () => { this.grad += out.grad; b.grad += out.grad; };
    return out;
  }
  mul(o: Value | number, label = ""): Value {
    const b = Value.of(o), out = new Value(this.data * b.data, [this, b], "×", label);
    out._backward = () => { this.grad += b.data * out.grad; b.grad += this.data * out.grad; };
    return out;
  }
  sub(o: Value | number, label = ""): Value {
    const b = Value.of(o), out = new Value(this.data - b.data, [this, b], "−", label);
    out._backward = () => { this.grad += out.grad; b.grad -= out.grad; };
    return out;
  }
  pow(k: number, label = ""): Value {
    const out = new Value(this.data ** k, [this], "^", label, k);
    out._backward = () => { this.grad += k * this.data ** (k - 1) * out.grad; };
    return out;
  }
  neg(label = ""): Value {
    const out = new Value(-this.data, [this], "neg", label);
    out._backward = () => { this.grad -= out.grad; };
    return out;
  }
  tanh(label = ""): Value {
    const t = Math.tanh(this.data), out = new Value(t, [this], "tanh", label);
    out._backward = () => { this.grad += (1 - t * t) * out.grad; };
    return out;
  }
  relu(label = ""): Value {
    const out = new Value(Math.max(0, this.data), [this], "relu", label);
    out._backward = () => { this.grad += (this.data > 0 ? 1 : 0) * out.grad; };
    return out;
  }
  exp(label = ""): Value {
    const e = Math.exp(this.data), out = new Value(e, [this], "exp", label);
    out._backward = () => { this.grad += e * out.grad; };
    return out;
  }
  log(label = ""): Value {
    const out = new Value(Math.log(this.data), [this], "log", label);
    out._backward = () => { this.grad += out.grad / this.data; };
    return out;
  }
  sigmoid(label = ""): Value {
    const s = 1 / (1 + Math.exp(-this.data)), out = new Value(s, [this], "σ", label);
    out._backward = () => { this.grad += s * (1 - s) * out.grad; };
    return out;
  }

  /** Children-before-parents ordering of every Value this one depends on. */
  topo(): Value[] {
    const stamp = ++visitEpoch, order: Value[] = [], stack: Value[] = [this], idx: number[] = [0];
    this._seen = stamp;
    while (stack.length) {
      const v = stack[stack.length - 1], i = idx[idx.length - 1];
      if (i < v.prev.length) {
        idx[idx.length - 1] = i + 1;
        const c = v.prev[i];
        if (c._seen !== stamp) { c._seen = stamp; stack.push(c); idx.push(0); }
      } else { order.push(v); stack.pop(); idx.pop(); }
    }
    return order;
  }

  backward(): void {
    const order = this.topo();
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) order[i]._backward();
  }
}

export const sum = (vs: Value[]) => vs.slice(1).reduce((a, b) => a.add(b), vs[0]);

// ---------- tiny MLP -----------------------------------------------------------
export type Act = "tanh" | "relu";
export class Neuron {
  w: Value[]; b: Value;
  constructor(nin: number, rand: () => number, public act: Act | "linear") {
    const s = 1 / Math.sqrt(nin);
    this.w = Array.from({ length: nin }, () => new Value((rand() * 2 - 1) * s * 1.5));
    this.b = new Value(0);
  }
  call(x: Value[]): Value {
    let s = this.b;
    for (let i = 0; i < x.length; i++) s = s.add(this.w[i].mul(x[i]));
    return this.act === "tanh" ? s.tanh() : this.act === "relu" ? s.relu() : s;
  }
  params() { return [...this.w, this.b]; }
}
export class MLP {
  layers: Neuron[][];
  constructor(sizes: number[], rand: () => number, act: Act = "tanh") {
    this.layers = [];
    for (let l = 0; l < sizes.length - 1; l++)
      this.layers.push(Array.from({ length: sizes[l + 1] }, () => new Neuron(sizes[l], rand, l === sizes.length - 2 ? "linear" : act)));
  }
  call(x: Value[]): Value[] { let h = x; for (const L of this.layers) h = L.map((n) => n.call(h)); return h; }
  params() { return this.layers.flatMap((L) => L.flatMap((n) => n.params())); }
  /** Fast plain-number forward (no graph), for drawing fits / decision boundaries. */
  eval(x: number[]): number {
    let h = x;
    for (const L of this.layers) h = L.map((n) => {
      let s = n.b.data;
      for (let i = 0; i < h.length; i++) s += n.w[i].data * h[i];
      return n.act === "tanh" ? Math.tanh(s) : n.act === "relu" ? Math.max(0, s) : s;
    });
    return h[0];
  }
}

/** Small deterministic PRNG (mulberry32). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gauss(r: () => number) { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
