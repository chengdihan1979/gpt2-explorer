import React, { useMemo, useState } from "react";
import { SelectionContext } from "./components/diagram/SelectionContext";
import { DiagramSpecContext } from "./components/diagram/DiagramSpecContext";
import { CATEGORY, DETAIL_LINE, LINE } from "./components/diagram/palette";
import CodeWithActions from "./components/CodeWithActions";
import InfoPanel from "./components/InfoPanel";
import { buildCodeHtml } from "./utils/highlight";
import { useInfo } from "./context/InfoContext";
import SiteHeader from "./components/SiteHeader";

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

// One model's page: architecture diagram on the left, its code on the right, notes in a drawer.
// Everything model-specific comes from the `model` config (see src/models/).
export default function ModelExplorer({ model }) {
  const { close } = useInfo();
  const [selectedId, setSelectedId] = useState(model.defaultSelected);

  const handleSelect = (id) => {
    setSelectedId(id);
    close();
  };

  const component = model.content[model.resolveCodeId(selectedId)] || { title: "Unknown", summary: "No data", code: "", anchors: [] };
  const Diagram = model.diagram.Component;

  const codeHtml = useMemo(() => buildCodeHtml({ component, selectedId, noteOverrides: model.noteOverrides }), [component, selectedId, model.noteOverrides]);
  const lineCount = component.code ? component.code.split("\n").length : 0;
  const fileName = model.fileNames.get(component.code) ?? "code.py";

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800">
      <SiteHeader badge={model.badge} currentId={model.id} subtitle="Click a block to see its code · click underlined code to open the notes" />

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
            <svg viewBox={model.diagram.viewBox} className="h-auto w-full" role="group" aria-label={model.diagram.label}>
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

              <DiagramSpecContext.Provider value={model.diagram.spec}>
                <Diagram onSelect={handleSelect} />
              </DiagramSpecContext.Provider>
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
