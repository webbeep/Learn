// Toy datasets + a training step for the "fit a tiny dataset" mode, using the scalar autograd.
import { MLP, Value, gauss, rng } from "./autograd";

export type DatasetId = "sine" | "moons" | "circles" | "xor";
export interface Dataset { id: DatasetId; dim: 1 | 2; X: number[][]; Y: number[]; name: string; blurb: string }

export const DATASETS: { id: DatasetId; name: string; blurb: string }[] = [
  { id: "sine", name: "Wiggly curve", blurb: "1-D regression: learn y = sin(1.5x) from 24 noisy points" },
  { id: "moons", name: "Two moons", blurb: "2-D classification: two interlocking crescents" },
  { id: "circles", name: "Ring", blurb: "2-D classification: a ring around a blob" },
  { id: "xor", name: "XOR", blurb: "2-D classification: opposite corners belong together" },
];

export function makeData(id: DatasetId, seed = 1): Dataset {
  const r = rng(seed), meta = DATASETS.find((d) => d.id === id)!;
  const X: number[][] = [], Y: number[] = [];
  if (id === "sine") {
    for (let i = 0; i < 24; i++) { const x = -3 + (6 * (i + 0.5)) / 24 + (r() - 0.5) * 0.15; X.push([x]); Y.push(Math.sin(1.5 * x) + gauss(r) * 0.06); }
    return { ...meta, dim: 1, X, Y };
  }
  const n = 60;
  for (let i = 0; i < n; i++) {
    const c = i % 2 === 0 ? 1 : -1;
    let x = 0, y = 0;
    if (id === "moons") {
      const t = Math.PI * r();
      if (c === 1) { x = Math.cos(t); y = Math.sin(t); } else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
      x = (x - 0.5) * 1.3 + gauss(r) * 0.1; y = (y - 0.25) * 1.3 + gauss(r) * 0.1;
    } else if (id === "circles") {
      const t = 2 * Math.PI * r(), rad = c === 1 ? 0.35 * Math.sqrt(r()) : 0.9 + gauss(r) * 0.08;
      x = rad * Math.cos(t); y = rad * Math.sin(t);
    } else {
      const qx = r() < 0.5 ? -1 : 1, qy = c === 1 ? qx : -qx;
      x = qx * 0.6 + gauss(r) * 0.2; y = qy * 0.6 + gauss(r) * 0.2;
    }
    X.push([x, y]); Y.push(c);
  }
  return { ...meta, dim: 2, X, Y };
}

export function makeNet(ds: Dataset, hidden: number, seed = 2) {
  return new MLP([ds.dim, hidden, hidden, 1], rng(seed), "tanh");
}

export type Opt = "sgd" | "adam";
/** Full-batch trainer: forward through the autograd graph, backward, then an SGD or Adam update. */
export class Trainer {
  m: Float64Array; v: Float64Array; t = 0; params;
  constructor(public net: MLP, public ds: Dataset) {
    this.params = net.params(); this.m = new Float64Array(this.params.length); this.v = new Float64Array(this.params.length);
  }
  /** One step. Returns the loss before the update. */
  step(opt: Opt, lr: number) {
    const { ds, net } = this;
    let total = new Value(0);
    for (let i = 0; i < ds.X.length; i++) {
      const s = net.call(ds.X[i].map((v) => new Value(v)))[0];
      // regression: squared error; classification: logistic loss log(1 + e^(−y·s))
      total = total.add(ds.dim === 1 ? s.sub(ds.Y[i]).pow(2) : s.mul(-ds.Y[i]).exp().add(1).log());
    }
    const loss = total.mul(1 / ds.X.length);
    loss.backward();
    // clip the global gradient norm at 1 (TinyGPT's training does the same)
    let n2 = 0; for (const p of this.params) n2 += p.grad * p.grad;
    const k = Math.min(1, 1 / (Math.sqrt(n2) + 1e-12));
    this.t++;
    const b1 = 0.9, b2 = 0.999;
    this.params.forEach((p, i) => {
      const g = p.grad * k;
      if (opt === "sgd") { p.data -= lr * g; return; }
      this.m[i] = b1 * this.m[i] + (1 - b1) * g; this.v[i] = b2 * this.v[i] + (1 - b2) * g * g;
      const mh = this.m[i] / (1 - b1 ** this.t), vh = this.v[i] / (1 - b2 ** this.t);
      p.data -= lr * mh / (Math.sqrt(vh) + 1e-8);
    });
    return loss.data;
  }
}

export function accuracy(net: MLP, ds: Dataset) {
  if (ds.dim === 1) return NaN;
  let ok = 0;
  for (let i = 0; i < ds.X.length; i++) if (Math.sign(net.eval(ds.X[i])) === ds.Y[i]) ok++;
  return ok / ds.X.length;
}
