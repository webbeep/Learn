// Preset computational graphs for the Backprop lab, plus layout + step-by-step simulation.
import { Value } from "./autograd";

export type LeafKind = "param" | "input" | "target";
export interface LeafDef { key: string; init: number; kind: LeafKind; min: number; max: number; hint: string }
export interface Preset {
  id: string; name: string; blurb: string; lossLabel: string;
  leaves: LeafDef[];
  build: (v: Record<string, number>) => Value;
  /** Short formula shown above the graph. */
  formula: string;
}

const L = (key: string, init: number, kind: LeafKind, hint: string, min = -4, max = 4): LeafDef => ({ key, init, kind, min, max, hint });

export const PRESETS: Preset[] = [
  {
    id: "expr", name: "Expression", lossLabel: "L",
    blurb: "Three multiplies and an add. The smallest graph where the chain rule does something.",
    formula: "L = (a·b + c) · f",
    leaves: [L("a", 2, "param", "a number we can change"), L("b", -3, "param", "a number we can change"),
      L("c", 10, "param", "a number we can change", -12, 12), L("f", -2, "param", "a number we can change")],
    build: (v) => {
      const a = new Value(v.a, [], "", "a"), b = new Value(v.b, [], "", "b"), c = new Value(v.c, [], "", "c"), f = new Value(v.f, [], "", "f");
      const e = a.mul(b, "e"), d = e.add(c, "d");
      return d.mul(f, "L");
    },
  },
  {
    id: "neuron", name: "One neuron", lossLabel: "loss",
    blurb: "Two inputs, two weights, a bias and a tanh: one neuron. The loss is how far its output is from the target.",
    formula: "o = tanh(x₁w₁ + x₂w₂ + b),  loss = (o − y)²",
    leaves: [L("x1", 2, "input", "input (data, not learned)"), L("w1", -3, "param", "weight (learned)"),
      L("x2", 0, "input", "input (data, not learned)"), L("w2", 1, "param", "weight (learned)"),
      L("b", 6.88, "param", "bias (learned)", -8, 8), L("y", -0.5, "target", "target: the answer we want")],
    build: (v) => {
      const x1 = new Value(v.x1, [], "", "x1"), w1 = new Value(v.w1, [], "", "w1"), x2 = new Value(v.x2, [], "", "x2"), w2 = new Value(v.w2, [], "", "w2");
      const b = new Value(v.b, [], "", "b"), y = new Value(v.y, [], "", "y");
      const s1 = x1.mul(w1, "x1w1"), s2 = x2.mul(w2, "x2w2"), s = s1.add(s2, "s"), n = s.add(b, "n"), o = n.tanh("o");
      return o.sub(y, "err").pow(2, "loss");
    },
  },
  {
    id: "net", name: "Tiny 2-layer net", lossLabel: "loss",
    blurb: "One input, two hidden tanh neurons, one output. Trained on a single example (x → y).",
    formula: "ŷ = v₁·tanh(w₁x + b₁) + v₂·tanh(w₂x + b₂) + c,  loss = (ŷ − y)²",
    leaves: [L("x", 1, "input", "input (data)"), L("w1", 0.8, "param", "layer-1 weight"), L("w2", -1.2, "param", "layer-1 weight"),
      L("b1", 0.1, "param", "layer-1 bias"), L("b2", 0.4, "param", "layer-1 bias"), L("v1", 0.9, "param", "layer-2 weight"),
      L("v2", -0.6, "param", "layer-2 weight"), L("c", 0.2, "param", "layer-2 bias"), L("y", -0.8, "target", "target output")],
    build: (v) => {
      const x = new Value(v.x, [], "", "x"), y = new Value(v.y, [], "", "y");
      const P = (k: string) => new Value(v[k], [], "", k);
      const h1 = x.mul(P("w1"), "m1").add(P("b1"), "z1").tanh("h1");
      const h2 = x.mul(P("w2"), "m2").add(P("b2"), "z2").tanh("h2");
      const yh = h1.mul(P("v1"), "p1").add(h2.mul(P("v2"), "p2"), "q").add(P("c"), "ŷ");
      return yh.sub(y, "err").pow(2, "loss");
    },
  },
];

// ---------- layout ------------------------------------------------------------
export interface GNode { v: Value; layer: number; slot: number; x: number; y: number }
export interface GEdge { from: string; to: string } // child label → parent label (forward direction)

export function layout(root: Value) {
  const topo = root.topo();
  const parents = new Map<Value, Value[]>();
  for (const v of topo) for (const c of v.prev) parents.set(c, [...(parents.get(c) ?? []), v]);
  const depth = new Map<Value, number>();
  for (const v of topo) if (v.prev.length) depth.set(v, 1 + Math.max(...v.prev.map((c) => (c.prev.length ? depth.get(c)! : 0))));
  for (const v of topo) if (!v.prev.length) depth.set(v, Math.min(...(parents.get(v) ?? [root]).map((p) => depth.get(p)!)) - 1);
  const nL = Math.max(...depth.values()) + 1;
  const layers: Value[][] = Array.from({ length: nL }, () => []);
  for (const v of topo) layers[depth.get(v)!].push(v);
  // barycenter sweeps to reduce crossings
  const pos = new Map<Value, number>();
  const setPos = () => layers.forEach((Ly) => Ly.forEach((v, i) => pos.set(v, i - (Ly.length - 1) / 2)));
  setPos();
  for (let it = 0; it < 4; it++) {
    for (let l = nL - 2; l >= 0; l--) {
      const bc = (v: Value) => { const ps = parents.get(v) ?? []; return ps.length ? ps.reduce((s, p) => s + pos.get(p)!, 0) / ps.length : pos.get(v)!; };
      layers[l].sort((a, b) => bc(a) - bc(b)); setPos();
    }
    for (let l = 1; l < nL; l++) {
      const bc = (v: Value) => (v.prev.length ? v.prev.reduce((s, c) => s + pos.get(c)!, 0) / v.prev.length : pos.get(v)!);
      layers[l].sort((a, b) => bc(a) - bc(b)); setPos();
    }
  }
  const maxW = Math.max(...layers.map((l) => l.length));
  const nodes: GNode[] = [];
  layers.forEach((Ly, l) => Ly.forEach((v, i) => nodes.push({ v, layer: l, slot: i - (Ly.length - 1) / 2, x: 0, y: 0 })));
  const edges: GEdge[] = topo.flatMap((v) => v.prev.map((c) => ({ from: c.label, to: v.label })));
  return { nodes, edges, nLayers: nL, maxWidth: maxW, topo };
}

// ---------- step-by-step simulation ------------------------------------------
export const OP_NAME: Record<string, string> = { "+": "+", "×": "×", "−": "−", "^": "²", tanh: "tanh", relu: "relu", exp: "exp", log: "log", neg: "−", "σ": "σ" };

export function forwardText(v: Value) {
  const [a, b] = v.prev, f = (x: number) => fmt(x);
  switch (v.op) {
    case "+": return `${v.label} = ${a.label} + ${b.label} = ${f(a.data)} + ${f(b.data)} = ${f(v.data)}`;
    case "×": return `${v.label} = ${a.label} × ${b.label} = ${f(a.data)} × ${f(b.data)} = ${f(v.data)}`;
    case "−": return `${v.label} = ${a.label} − ${b.label} = ${f(a.data)} − ${f(b.data)} = ${f(v.data)}`;
    case "^": return `${v.label} = ${a.label}² = ${f(a.data)}² = ${f(v.data)}`;
    case "tanh": return `${v.label} = tanh(${a.label}) = tanh(${f(a.data)}) = ${f(v.data)}`;
    default: return `${v.label} = ${f(v.data)}`;
  }
}

/** Local derivative d(parent)/d(child) as [formula, value]. */
export function localRule(v: Value, child: Value): [string, number] {
  const [a, b] = v.prev;
  switch (v.op) {
    case "+": return ["1", 1];
    case "×": { const o = child === a ? b : a; return [o.label, o.data]; }
    case "−": return child === a ? ["1", 1] : ["−1", -1];
    case "^": return [`${v.k}·${a.label}`, v.k * a.data ** (v.k - 1)];
    case "tanh": return [`1 − ${v.label}²`, 1 - v.data * v.data];
    case "relu": return [`[${a.label} > 0]`, a.data > 0 ? 1 : 0];
    default: return ["?", NaN];
  }
}

export function backwardText(v: Value, lossLabel: string) {
  return v.prev.map((c) => {
    const [rule, val] = localRule(v, c);
    return `∂${lossLabel}/∂${c.label} += ${rule} × ∂${lossLabel}/∂${v.label} = ${fmt(val)} × ${fmt(v.grad)} = ${fmt(val * v.grad)}`;
  });
}

export interface Sim {
  ops: Value[]; // op nodes in forward order
  /** grads[k] = grad of every node (by label) after k backward steps (k = 0: nothing; 1: loss.grad = 1; 1+i: after ops reversed[i]). */
  grads: Record<string, number>[];
  /** reached[k] = labels whose grad has been touched after k steps. */
  reached: Set<string>[];
}

export function simulate(root: Value, topo: Value[]): Sim {
  const ops = topo.filter((v) => v.prev.length);
  for (const v of topo) v.grad = 0;
  const snap = () => Object.fromEntries(topo.map((v) => [v.label, v.grad]));
  const grads = [snap()], reached = [new Set<string>()];
  root.grad = 1; grads.push(snap()); reached.push(new Set([root.label]));
  for (let i = ops.length - 1; i >= 0; i--) {
    ops[i]._backward(); grads.push(snap());
    reached.push(new Set([...reached[reached.length - 1], ...ops[i].prev.map((c) => c.label)]));
  }
  return { ops, grads, reached };
}

export function fmt(x: number, d = 2) {
  if (!Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a < 1e-6) return (0).toFixed(d);
  if (a !== 0 && (a < 0.01 || a >= 1000)) return x.toExponential(1).replace("e", "e").replace("-", "−");
  return x.toFixed(d).replace("-", "−");
}
