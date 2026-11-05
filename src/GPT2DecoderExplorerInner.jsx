import React, { useMemo, useState } from "react";
import Node from "./components/diagram/Node";
import Arrow from "./components/diagram/Arrow";
import GPT2OuterFrame from "./components/diagram/GPT2OuterFrame";
import TransformerBlockGroup from "./components/diagram/TransformerBlockGroup";
import MLPSubDiagram from "./components/diagram/MLPSubDiagram";
import AttnSubDiagram from "./components/diagram/AttnSubDiagram";
import { SelectionContext } from "./components/diagram/SelectionContext";
import { CATEGORY, DETAIL_LINE, LINE } from "./components/diagram/palette";
import CodeWithActions from "./components/CodeWithActions";
import InfoPanel from "./components/InfoPanel";
import { buildCodeHtml } from "./utils/highlight";
import { attnCode, content, gpt2TrainingCode, mlpCode, transformerBlockCode } from "./content/snippets";
import { useInfo } from "./context/InfoContext";

const FILE_NAMES = new Map([
  [gpt2TrainingCode, "train_gpt2.py"],
  [transformerBlockCode, "block.py"],
  [attnCode, "attention.py"],
  [mlpCode, "mlp.py"],
]);

const REPO_URL = "https://github.com/chengdihan1979/gpt2-explorer";

function Legend() {
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1.5">
      {Object.entries(CATEGORY).map(([key, c]) => (
        <li key={key} className="flex items-center gap-1.5 text-[11px] text-slate-500">
          <span className="h-2.5 w-2.5 rounded-[3px] border" style={{ background: c.fill, borderColor: c.stroke }} />
          {c.label}
        </li>
      ))}
    </ul>
  );
}

export default function GPT2DecoderExplorerInner() {
  const { close } = useInfo();
  const [selectedId, setSelectedId] = useState("token_embed");

  const handleSelect = (id) => {
    setSelectedId(id);
    close();
  };

  const effectiveId = ["mlp_linear1", "mlp_linear2", "mlp_gelu", "mlp_dropout"].includes(selectedId)
    ? "mlp_subdiagram"
    : (selectedId === "emb_dropout"
        || selectedId === "gpt2_blocks"
        || selectedId === "final_ln"
        || selectedId === "lm_head"
        ? "gpt2"
        : (selectedId === "block_attn_plus" ? "block_attn" : (selectedId === "block_mlp_plus" ? "block_mlp" : selectedId)));

  const component = content[effectiveId] || { title: "Unknown", summary: "No data", code: "", anchors: [] };

  const codeHtml = useMemo(() => buildCodeHtml({ component, selectedId }), [component, selectedId]);
  const lineCount = component.code ? component.code.split("\n").length : 0;
  const fileName = FILE_NAMES.get(component.code) ?? "code.py";

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800">
      <header className="sticky top-0 z-30 h-16 border-b border-slate-200 bg-white/85 backdrop-blur">
        <div className="mx-auto flex h-full max-w-[1680px] items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-[13px] font-bold text-white shadow-sm">
              G2
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-[15px] font-semibold tracking-tight text-slate-900">GPT-2 Decoder Explorer</h1>
              <p className="hidden truncate text-xs text-slate-500 sm:block">
                Click a block to see its code · click underlined code to open the notes
              </p>
            </div>
          </div>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm hover:border-slate-300 hover:text-slate-900"
          >
            <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1680px] grid-cols-[minmax(0,1fr)] gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">Architecture</p>
              <h2 className="text-base font-semibold text-slate-900">Decoder data flow</h2>
            </div>
            <Legend />
          </div>

          <SelectionContext.Provider value={selectedId}>
            <svg viewBox="0 0 836 848" className="h-auto w-full" role="group" aria-label="GPT-2 architecture diagram">
              <defs>
                <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                  <polygon points="0 0, 8 3, 0 6" fill={LINE} />
                </marker>
                <marker id="arrowhead-detail" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                  <polygon points="0 0, 8 3, 0 6" fill={DETAIL_LINE} />
                </marker>
                <filter id="node-shadow" x="-10%" y="-30%" width="120%" height="160%">
                  <feDropShadow dx="0" dy="1.5" stdDeviation="2" floodColor="#0f172a" floodOpacity="0.18" />
                </filter>
              </defs>

              <GPT2OuterFrame onClick={handleSelect} />

              {/* Input data and tokenizer stacked directly above embeddings */}
              <Node id="input_data" label="Input Data (Text)" x={20} y={50} w={180} h={40} onClick={handleSelect} />
              <Arrow x1={110} y1={90} x2={110} y2={110} />
              <Node id="tokenizer" label="Tokenizer" x={20} y={110} w={180} h={40} onClick={handleSelect} />

              <Arrow x1={110} y1={150} x2={110} y2={170} />
              <Node id="token_embed" label="Token Embeddings" x={20} y={170} w={180} h={40} onClick={handleSelect} />
              <text x={210} y={190} textAnchor="middle" dominantBaseline="central" fontSize={16} fontWeight={500} fill="#64748B">+</text>
              <Node id="pos_embed" label="Positional Embeddings" x={220} y={170} w={180} h={40} onClick={handleSelect} />

              <Arrow x1={210} y1={210} x2={210} y2={240} />
              <Node id="emb_dropout" label="Dropout" x={20} y={240} w={380} h={40} onClick={handleSelect} />

              <Arrow x1={210} y1={280} x2={210} y2={300} />
              <TransformerBlockGroup x={20} y={300} onClick={handleSelect} />
              {/* Residual connection from the embedding Dropout output into the attention residual add */}
              <polyline points="215,290 500,290 500,455 220,455" stroke={LINE} strokeWidth={1.25} fill="none" markerEnd="url(#arrowhead)" />

              <Arrow x1={210} y1={590} x2={210} y2={620} />
              <Node id="final_ln" label="Final LayerNorm" x={20} y={620} w={380} h={40} onClick={handleSelect} />

              <Arrow x1={210} y1={660} x2={210} y2={700} />
              <Node id="lm_head" label="LM Head (Linear Layer)" x={20} y={700} w={380} h={40} onClick={handleSelect} />

              <Arrow x1={210} y1={740} x2={210} y2={780} />
              <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={780} w={380} h={40} onClick={handleSelect} />

              <MLPSubDiagram x={600} y={530} onClick={handleSelect} />
              <AttnSubDiagram x={600} y={40} onClick={handleSelect} />
              {/* Attention block expands into the Causal Self-Attention sub-diagram */}
              <Arrow x1={400} y1={410} x2={600} y2={120} variant="detail" />
            </svg>
          </SelectionContext.Provider>
        </section>

        <section className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:sticky lg:top-[88px] lg:h-[calc(100vh-112px)]">
          <div className="mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">Code</p>
            <h2 className="text-base font-semibold text-slate-900">{component.title}</h2>
            <p className="mt-0.5 text-sm text-slate-500">{component.summary}</p>
          </div>
          {codeHtml ? (
            <div className="flex max-h-[75vh] min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-slate-800 bg-[#0d1117] lg:max-h-none">
              <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
                <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
                <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
                <span className="ml-2 font-mono text-xs text-slate-400">{fileName}</span>
              </div>
              <CodeWithActions html={codeHtml} lineCount={lineCount} onSelect={setSelectedId} />
            </div>
          ) : (
            <p className="text-sm text-slate-400">No code for this block.</p>
          )}
        </section>
      </main>

      <InfoPanel />
    </div>
  );
}
