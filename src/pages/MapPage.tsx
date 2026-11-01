import { useEffect } from "react";
import { addWish, learner, setScreen, useStore } from "../lib/store";

export const LEVELS = [
  { n: 0, href: "#/xray", title: "Watch a real model think", sub: "Inference X-ray", status: "open",
    blurb: "Type a sentence, press play, and watch a real (tiny) GPT turn it into the next word — tokens, vectors, attention, layers, probabilities, dice roll. Pause anywhere, click anything, ask." },
  { n: 1, href: "#/training", title: "Where the numbers came from", sub: "Training peek", status: "open",
    blurb: "The loss curve of the exact model you just watched, and what it wrote at each stage of its 80-second life." },
  { n: 2, title: "Train it yourself", sub: "In-browser training", status: "next",
    blurb: "Hit train, watch the loss fall live, see gradients flow backward through the same X-ray. Change the data and watch it learn something new." },
  { n: 3, title: "Backprop, visualized", sub: "Dive down when you need it", status: "later",
    blurb: "Why does each weight move the way it does? The chain rule as a picture — only when you get curious." },
  { n: 4, title: "Architecture lab", sub: "Rewire the model", status: "later",
    blurb: "Remove attention. Add layers. Swap the MLP. Retrain and compare — learn design by breaking things." },
  { n: 5, title: "Scaling", sub: "Why bigger works", status: "later",
    blurb: "Train 5 sizes, plot loss vs. compute, find the scaling law yourself. GPUs, batching, memory — when you hit the wall." },
  { n: 6, title: "Invent your own", sub: "A new kind of model", status: "later",
    blurb: "Your idea, your training loop, your framework — with everything above as your toolbox." },
];

export function MapPage() {
  const wishes = useStore(learner, (s) => s.wishes);
  const profile = useStore(learner, (s) => s.profile);
  useEffect(() => setScreen("Journey map", "", `The learning journey has levels: ${LEVELS.map((l) => `${l.n}. ${l.title} (${l.status})`).join("; ")}.`), []);
  return (
    <div className="page map">
      <section className="hero">
        <h1>Learn AI top-down.</h1>
        <p className="lead">Start with a real model working in front of you. Poke it. Break it. Ask about anything you see — every pixel is clickable and the tutor sees exactly what you're pointing at. Dive into the fundamentals only when you hit a question. This app grows with your questions.</p>
        <div className="row"><a className="btn primary big" href="#/xray">▶ Watch a model think</a><a className="btn" href="#/watch">🎬 Videos</a></div>
      </section>
      {profile?.next?.length ? (
        <section className="card">
          <div className="lane-label">Suggested for you (from your questions)</div>
          <div className="next-list">{profile.next.slice(0, 3).map((n, i) => (
            <a key={i} className="next-item" href={n.where === "training" ? "#/training" : n.where === "watch" ? "#/watch" : "#/xray"}><b>{n.title}</b><span className="muted small">{n.why}</span></a>
          ))}</div>
        </section>
      ) : null}
      <section className="levels">
        {LEVELS.map((l) => {
          const wanted = wishes.some((w) => w.text.startsWith(`Level ${l.n}:`));
          return (
            <div key={l.n} className={`level ${l.status}`} data-ask-id={`level-${l.n}`} data-ask={`Level ${l.n}: ${l.title} — ${l.blurb} (${l.status})`}>
              <div className="lvl-n">{l.n}</div>
              <div className="lvl-body">
                <div className="muted small">{l.sub}</div>
                <h3>{l.title}</h3>
                <p>{l.blurb}</p>
                {l.href ? <a className="btn primary" href={l.href}>Open</a> :
                  <button disabled={wanted} onClick={() => addWish(`Level ${l.n}: ${l.title} — ${l.blurb}`, "journey map")}>{wanted ? "✓ on your build list" : "⭐ I want this next"}</button>}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
