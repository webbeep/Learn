import { useCallback, useEffect, useState } from "react";
import { setScreen } from "../lib/store";
import { GraphLab } from "../labs/BackpropGraph";
import { FitLab } from "../labs/BackpropFit";
import { RealGrads } from "../labs/BackpropReal";

export function Backprop() {
  const [scene, setScene] = useState("Graph");
  const [parts, setParts] = useState<{ graph: string; fit: string; real: string }>({ graph: "", fit: "", real: "" });
  const onGraph = useCallback((sc: string, s: string) => { setScene(sc); setParts((p) => (p.graph === s ? p : { ...p, graph: s })); }, []);
  const onFit = useCallback((s: string) => setParts((p) => (p.fit === s ? p : { ...p, fit: s })), []);
  const onReal = useCallback((s: string) => setParts((p) => (p.real === s ? p : { ...p, real: s })), []);
  useEffect(() => {
    setScreen("Backprop, visualized", scene, [
      "Backpropagation lab: a scalar autograd engine (micrograd-style) drawn as a computational graph. Gradients are computed by the chain rule from the output back to every leaf.",
      parts.graph, parts.fit, parts.real,
    ].filter(Boolean).join("\n\n"));
  }, [scene, parts]);

  return (
    <div className="space-y-6 pb-10">
      <h1 className="text-3xl font-semibold tracking-tight">Backprop, visualized</h1>
      <GraphLab onSummary={onGraph} />
      <FitLab onSummary={onFit} />
      <RealGrads onSummary={onReal} />
    </div>
  );
}
