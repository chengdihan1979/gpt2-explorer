import React, { useCallback, useEffect, useRef, useState } from "react";
import { CATEGORY, DETAIL_LINE, LINE, getCategory } from "./diagram/palette";
import { TYPE_ORDER, frameAt, linkGeometry } from "../evolution/scene";
import { useInfo } from "../context/InfoContext";

const MOVE_MS = 1800; // one transition between steps (see frameAt for its phases)
const DWELL_MS = 6000; // time on each step while playing
const HIGHLIGHT = "#F59E0B";
const REMOVED = "#F43F5E";

const UI = {
  en: { step: "Step", play: "Play", pause: "Pause", prev: "Previous", next: "Next", replay: "Replay", note: "Read the note", page: "Open the page", steps: "All steps" },
  zh: { step: "第", play: "播放", pause: "暂停", prev: "上一步", next: "下一步", replay: "重新播放", note: "看 note", page: "打开页面", steps: "全部步骤" },
};

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

// Text that cross-fades when its label changed during the current transition.
function Label({ e, x, y, size = 12, weight = 500, fill, anchor = "middle" }) {
  const common = { x, y, fontSize: size, fontWeight: weight, fill, textAnchor: anchor, dominantBaseline: "central", className: "select-none" };
  if (e.prevLabel && e.labelT < 1) {
    return (
      <>
        <text {...common} opacity={1 - e.labelT}>{e.prevLabel}</text>
        <text {...common} opacity={e.labelT}>{e.label}</text>
      </>
    );
  }
  return <text {...common}>{e.label}</text>;
}

const pillWidth = (label) => label.length * 7.2 + 20;

function Element({ e, frame, highlighted }) {
  const removed = e.status === "removed";
  switch (e.type) {
    case "frame": {
      const outer = e.id === "outer";
      return (
        <rect x={e.x} y={e.y} width={e.w} height={e.h} rx={outer ? 18 : 12}
          fill={outer ? "none" : "#F8FAFC"} stroke={e.dashed ? "#C7D2FE" : "#CBD5E1"} strokeDasharray={e.dashed ? "5 4" : undefined} />
      );
    }
    case "title": {
      const w0 = pillWidth(e.prevLabel ?? e.label);
      const w = e.prevLabel ? w0 + (pillWidth(e.label) - w0) * e.labelT : pillWidth(e.label);
      const left = e.anchor === "start" ? e.x - 9 : e.x - w / 2;
      return (
        <g>
          <rect x={left} y={e.y - 12} width={w} height={24} rx={12} fill="#fff"
            stroke={highlighted ? HIGHLIGHT : "#E2E8F0"} strokeWidth={highlighted ? 2 : 1} className={highlighted ? "evo-pulse" : undefined} />
          <Label e={e} x={e.x} y={e.y} size={13} weight={600} fill="#0F172A" anchor={e.anchor ?? "middle"} />
        </g>
      );
    }
    case "node": {
      if (e.hidden) return null;
      const c = CATEGORY[e.cat ?? getCategory(e.id)];
      return (
        <g>
          {highlighted && (
            <rect x={e.x - 4} y={e.y - 4} width={e.w + 8} height={e.h + 8} rx={11} fill="none" stroke={HIGHLIGHT} strokeWidth={2.5} className="evo-pulse" />
          )}
          <rect x={e.x} y={e.y} width={e.w} height={e.h} rx={8} fill={c.fill}
            stroke={removed ? REMOVED : c.stroke} strokeWidth={removed ? 2 : 1} strokeDasharray={removed ? "4 3" : undefined} />
          <Label e={e} x={e.x + e.w / 2} y={e.y + e.h / 2} fill={c.text} />
        </g>
      );
    }
    case "link": {
      const g = linkGeometry(e, frame);
      return g && <line {...g} stroke={LINE} strokeWidth={1.25} markerEnd="url(#evo-arrow)" />;
    }
    case "line": {
      const detail = e.variant === "detail";
      return (
        <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={detail ? DETAIL_LINE : LINE} strokeWidth={1.25}
          strokeDasharray={detail ? "4 4" : undefined} markerEnd={detail ? "url(#evo-arrow-detail)" : "url(#evo-arrow)"} />
      );
    }
    case "poly": {
      const tie = e.variant === "tie";
      return (
        <polyline points={e.points.map((p) => p.join(",")).join(" ")} fill="none"
          stroke={tie ? (removed ? REMOVED : "#D97706") : LINE} strokeWidth={tie ? 1.5 : 1.25}
          strokeDasharray={tie ? "5 4" : undefined} markerEnd={tie ? undefined : "url(#evo-arrow)"} />
      );
    }
    case "plus":
      return (
        <g stroke={LINE} strokeWidth={1.25}>
          <circle cx={e.cx} cy={e.cy} r={8} fill="#fff" />
          <line x1={e.cx - 4.5} y1={e.cy} x2={e.cx + 4.5} y2={e.cy} />
          <line x1={e.cx} y1={e.cy - 4.5} x2={e.cx} y2={e.cy + 4.5} />
        </g>
      );
    case "text":
      return (
        <g transform={e.rotate ? `rotate(${e.rotate} ${e.x} ${e.y})` : undefined}>
          <Label e={e} x={e.x} y={e.y} size={e.size ?? 12} fill={removed ? REMOVED : e.color ?? "#64748B"} />
        </g>
      );
    default:
      return null;
  }
}

function Diagram({ frame, highlight, viewBox, label }) {
  const els = Object.values(frame).sort((a, b) => TYPE_ORDER.indexOf(a.type) - TYPE_ORDER.indexOf(b.type));
  return (
    <svg viewBox={viewBox} className="h-auto w-full" role="img" aria-label={label}>
      <defs>
        <marker id="evo-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <polygon points="0 0, 8 3, 0 6" fill={LINE} />
        </marker>
        <marker id="evo-arrow-detail" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <polygon points="0 0, 8 3, 0 6" fill={DETAIL_LINE} />
        </marker>
      </defs>
      {els.map((e) => (
        <g key={e.id} opacity={e.opacity}>
          <Element e={e} frame={frame} highlighted={highlight.has(e.id) && e.status !== "removed"} />
        </g>
      ))}
    </svg>
  );
}

// Plays one model-to-model transition (see src/evolution/): the diagram morphs step by step,
// each step with a caption, a highlight on what changed, and a link to the matching note.
export default function EvolutionPlayer({ transition }) {
  const { lang, open } = useInfo();
  const t = UI[lang] ?? UI.en;
  const { steps, scenes } = transition;
  const last = steps.length - 1;

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  // The drawing: tweening from scene `from` to scene `index`, `progress` 0..1.
  const [anim, setAnim] = useState({ from: 0, progress: 1 });
  const raf = useRef(0);
  const indexRef = useRef(0);

  const goTo = useCallback((i) => {
    const cur = indexRef.current;
    if (i === cur) return;
    cancelAnimationFrame(raf.current);
    indexRef.current = i;
    setIndex(i);
    const duration = prefersReducedMotion() ? 0 : MOVE_MS;
    const start = performance.now();
    const tick = (now) => {
      const p = duration ? Math.min(1, (now - start) / duration) : 1;
      setAnim({ from: cur, progress: p });
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    setAnim({ from: cur, progress: 0 });
    raf.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  // Autoplay: after each transition, wait, then go to the next step; stop at the end.
  useEffect(() => {
    if (!playing || anim.progress < 1) return;
    if (index >= last) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => goTo(index + 1), index === 0 ? 2500 : DWELL_MS);
    return () => clearTimeout(id);
  }, [playing, anim.progress, index, last, goTo]);

  const frame = frameAt(scenes[anim.from], scenes[index], anim.progress);
  const step = steps[index];
  const highlight = new Set(step.highlight ?? []);

  const manual = (i) => {
    setPlaying(false);
    goTo(Math.max(0, Math.min(last, i)));
  };
  const togglePlay = () => {
    if (playing) return setPlaying(false);
    if (index >= last) goTo(0);
    setPlaying(true);
  };

  const base = "inline-flex items-center justify-center rounded-lg border px-3 py-1.5 text-xs font-medium shadow-sm disabled:cursor-not-allowed disabled:opacity-40";
  const btn = `${base} border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:text-slate-900`;
  const linkBtn = `${base} border-slate-200 bg-white text-indigo-700 hover:border-indigo-300`;
  const primary = `${base} min-w-[84px] border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700`;

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
        <Diagram frame={frame} highlight={anim.progress < 1 ? new Set() : highlight} viewBox={transition.viewBox}
          label={`${transition.from} to ${transition.to} architecture`} />
      </div>

      <div className="flex flex-col gap-4">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">
            {lang === "zh" ? `第 ${index} / ${last} 步` : `${t.step} ${index} / ${last}`}
          </p>
          <h3 className="mt-1 text-lg font-semibold text-slate-900">{step.title[lang] ?? step.title.en}</h3>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">{step.body[lang] ?? step.body.en}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {step.note && (
              <button type="button" className={linkBtn} onClick={() => open(step.note)}>{t.note} →</button>
            )}
            {step.page && <a className={linkBtn} href={`#/${step.page}`}>{t.page} →</a>}
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-4">
            <button type="button" className={btn} onClick={() => manual(index - 1)} disabled={index === 0}>← {t.prev}</button>
            <button type="button" className={primary} onClick={togglePlay}>
              {playing ? t.pause : index >= last ? t.replay : t.play}
            </button>
            <button type="button" className={btn} onClick={() => manual(index + 1)} disabled={index === last}>{t.next} →</button>
          </div>
          <div className="mt-3 h-1 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
            <div className="h-full rounded-full bg-indigo-500 transition-[width] duration-500" style={{ width: `${(index / last) * 100}%` }} />
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{t.steps}</p>
          <ol className="space-y-0.5">
            {steps.map((s, i) => (
              <li key={i}>
                <button type="button" onClick={() => manual(i)} aria-current={i === index ? "step" : undefined}
                  className={`flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm ${i === index ? "bg-indigo-50 font-medium text-indigo-800" : i < index ? "text-slate-700 hover:bg-slate-50" : "text-slate-400 hover:bg-slate-50 hover:text-slate-700"}`}>
                  <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-semibold ${i <= index ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-500"}`}>{i}</span>
                  {s.title[lang] ?? s.title.en}
                </button>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}
