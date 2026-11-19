import React, { useState } from "react";
import SiteHeader from "./components/SiteHeader";
import EvolutionPlayer from "./components/EvolutionPlayer";
import InfoPanel from "./components/InfoPanel";
import { useInfo } from "./context/InfoContext";
import { MODELS } from "./models";
import { TRANSITIONS } from "./evolution";

const TEXT = {
  en: {
    eyebrow: "Architecture evolution",
    title: "From GPT-2 to today's LLMs, one change at a time",
    intro: "Watch the GPT-2 diagram turn into newer models. Each step changes one component and explains why; open the notes for the details.",
    models: "Explore a model",
    open: "Diagram, code and notes →",
    subtitle: "Watch the architectures evolve · open any model to read its code and notes",
  },
  zh: {
    eyebrow: "架构演变",
    title: "从 GPT-2 到今天的大模型，一次只改一处",
    intro: "看 GPT-2 的架构图怎样一步步变成更新的模型。每一步只改一个组件，并说明为什么这样改；想看细节就打开对应的 note。",
    models: "逐个模型细看",
    open: "架构图、代码和 notes →",
    subtitle: "看架构怎样演变 · 打开任意模型看代码和 notes",
  },
};

const BLURB = {
  gpt2: { en: "The reference point: 124M parameters, learned positions, LayerNorm, GELU.", zh: "参照点：1.24 亿参数，学出来的位置向量，LayerNorm，GELU。" },
  gpt3: { en: "GPT-2's design at 175B parameters, with alternating sparse attention.", zh: "GPT-2 的设计放大到 1750 亿参数，并交替使用稀疏 attention。" },
  llama: { en: "LLaMA 3 8B: RMSNorm, RoPE, grouped-query attention, SwiGLU.", zh: "LLaMA 3 8B：RMSNorm、RoPE、分组查询注意力、SwiGLU。" },
};

// The landing page: the architecture-evolution animation, then links to each model page.
export default function HomePage() {
  const { lang } = useInfo();
  const t = TEXT[lang] ?? TEXT.en;
  const [current, setCurrent] = useState(0);
  const transition = TRANSITIONS[current];

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800">
      <SiteHeader badge="LX" currentId="home" subtitle={t.subtitle} />

      <main className="mx-auto max-w-[1680px] px-4 py-8 sm:px-6">
        <div className="mb-6 max-w-3xl">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">{t.eyebrow}</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">{t.title}</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 sm:text-base">{t.intro}</p>
        </div>

        {TRANSITIONS.length > 1 && (
          <div role="tablist" className="mb-4 inline-flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs font-medium shadow-sm">
            {TRANSITIONS.map((tr, i) => (
              <button key={tr.id} type="button" role="tab" aria-selected={i === current} onClick={() => setCurrent(i)}
                className={`rounded-md px-2.5 py-1 ${i === current ? "bg-slate-800 text-white" : "text-slate-600 hover:text-slate-900"}`}>
                {tr.from} → {tr.to}
              </button>
            ))}
          </div>
        )}

        <EvolutionPlayer key={transition.id} transition={transition} />

        <h2 className="mb-3 mt-10 text-base font-semibold text-slate-900">{t.models}</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MODELS.map((m) => (
            <a key={m.id} href={`#/${m.id}`} className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm hover:border-indigo-300 hover:shadow">
              <div className="flex items-center gap-3">
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-[13px] font-bold text-white">{m.badge}</div>
                <h3 className="text-[15px] font-semibold text-slate-900">{m.name}</h3>
              </div>
              {BLURB[m.id] && <p className="mt-3 text-sm text-slate-600">{BLURB[m.id][lang] ?? BLURB[m.id].en}</p>}
              <p className="mt-3 text-xs font-medium text-indigo-600 group-hover:text-indigo-700">{t.open}</p>
            </a>
          ))}
        </div>
      </main>

      <InfoPanel />
    </div>
  );
}
