// Starter questions tailored to what kind of element is selected.
const BY_PREFIX: [RegExp, string[]][] = [
  [/^(tok|vocab)/, ["Why was the text split this way?", "What does the model actually receive for this token?", "How do real LLMs like GPT-4 tokenize?"]],
  [/^(emb|pca)/, ["What do these numbers mean?", "Why are these words close together on the map?", "Where did these vectors come from?"]],
  [/^att/, ["Why does it pay attention here?", "What job might this head be doing?", "What are queries, keys and values here?"]],
  [/^(lens|resid)/, ["What changed between these layers?", "Why does the answer appear only late?", "What is the residual stream?"]],
  [/^mlp/, ["What does this neuron respond to?", "What does the MLP add that attention can't?", "How do real models store facts in MLPs?"]],
  [/^(prob|logit)/, ["Why is this token so likely?", "What would temperature do here?", "What's the difference between logits and probabilities?"]],
  [/^pick/, ["Why did it pick this token?", "Is the model deterministic?", "How do chatbots choose tokens?"]],
  [/^(loss|snap|train)/, ["Why does the loss drop like this?", "What is the model learning at this point?", "How would I train my own?"]],
  [/^video/, ["Explain this moment in plain words", "How does this connect to the tiny model in the app?", "What should I try in the X-ray after this?"]],
];
const DEFAULT = ["Explain what I'm looking at", "Why does this matter if I want to build my own model?", "What's the one thing to notice here?"];

export function suggestionsFor(ids: string[]) {
  const id = ids[0] ?? "";
  for (const [re, qs] of BY_PREFIX) if (re.test(id)) return qs;
  return DEFAULT;
}
