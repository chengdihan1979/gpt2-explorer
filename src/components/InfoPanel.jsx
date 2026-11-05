import React, { useState, useRef, useLayoutEffect, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import { useInfo } from "../context/InfoContext";
import { INFO } from "../content/info";

// Inline code written as `⟦Hello⟧⟦,⟧⟦⎵world⟧` renders as a row of boxed token chunks.
function NoteCode({ children, className, ...props }) {
  delete props.node; // react-markdown passes the AST node; keep it off the DOM
  const text = Array.isArray(children) ? children.join("") : children;
  if (!className && typeof text === "string" && text.startsWith("⟦") && text.endsWith("⟧")) {
    const pieces = text.slice(1, -1).split("⟧⟦");
    return (
      <span className="chunk-row">
        {pieces.map((p, i) => (
          <span key={i} className="chunk">
            {[...p].map((ch, j) => (ch === "⎵" ? <span key={j} className="chunk-space">⎵</span> : ch))}
          </span>
        ))}
      </span>
    );
  }
  return <code className={className} {...props}>{children}</code>;
}

export default function InfoPanel() {
  const { active, previous, close, open, back } = useInfo();
  const bodyRef = useRef(null);
  const [hoverInfoId, setHoverInfoId] = useState(null);
  const [hoverPos, setHoverPos] = useState({ top: 0, left: 0 });
  const popupRef = useRef(null);
  const anchorRectRef = useRef(null);

  const recomputePopup = useCallback(() => {
    if (!hoverInfoId || !popupRef.current || !anchorRectRef.current) return;
    const r = popupRef.current.getBoundingClientRect();
    const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
    const vh = typeof window !== "undefined" ? window.innerHeight : 800;
    const ar = anchorRectRef.current;
    const m = 8;
    // Candidates: above, below, right, left
    const candidates = [];
    // Above
    candidates.push({ top: ar.top - r.height - m, left: Math.min(Math.max(ar.left, m), vw - r.width - m), fits: () => ar.top - r.height - m >= m });
    // Below
    candidates.push({ top: ar.bottom + m, left: Math.min(Math.max(ar.left, m), vw - r.width - m), fits: () => ar.bottom + r.height + m <= vh - m });
    // Right
    candidates.push({ top: Math.min(Math.max(ar.top, m), vh - r.height - m), left: ar.right + m, fits: () => ar.right + r.width + m <= vw - m });
    // Left
    candidates.push({ top: Math.min(Math.max(ar.top, m), vh - r.height - m), left: ar.left - r.width - m, fits: () => ar.left - r.width - m >= m });
    let placed = null;
    for (const c of candidates) {
      if (c.fits()) { placed = { top: Math.max(m, Math.min(c.top, vh - r.height - m)), left: Math.max(m, Math.min(c.left, vw - r.width - m)) }; break; }
    }
    if (!placed) {
      placed = { top: Math.max(m, Math.min(ar.bottom + m, vh - r.height - m)), left: Math.max(m, Math.min(ar.left, vw - r.width - m)) };
    }
    setHoverPos(placed);
  }, [hoverInfoId]);

  useLayoutEffect(() => { recomputePopup(); }, [recomputePopup]);

  useLayoutEffect(() => {
    if (!popupRef.current) return;
    const ro = new ResizeObserver(() => recomputePopup());
    ro.observe(popupRef.current);
    const onResize = () => recomputePopup();
    window.addEventListener('resize', onResize);
    return () => { ro.disconnect(); window.removeEventListener('resize', onResize); };
  }, [recomputePopup, popupRef]);
  // Esc closes the drawer (or the hover popup first, if one is open).
  useEffect(() => {
    if (!active) return;
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      if (hoverInfoId) setHoverInfoId(null);
      else close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, hoverInfoId, close]);

  // Each new note starts at the top, and any hover popup from the previous note goes away.
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    setHoverInfoId(null);
  }, [active]);

  if (!active) return null;

  return (
    <aside
      className="notes-drawer fixed inset-x-0 bottom-0 top-16 z-40 flex flex-col border-r border-slate-200 bg-white shadow-2xl lg:right-auto lg:w-1/2"
      aria-label="Notes"
    >
      <header className="flex items-center gap-3 border-b border-slate-200 px-6 py-3">
        {previous && (
          <button
            type="button"
            onClick={back}
            className="inline-flex max-w-[45%] items-center gap-1 truncate rounded-md px-2 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            title={`Back to ${INFO[previous]?.title ?? "previous note"}`}
          >
            <span aria-hidden="true">←</span>
            <span className="truncate">{INFO[previous]?.title ?? "Back"}</span>
          </button>
        )}
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">Notes</p>
          <h2 className="truncate text-base font-semibold text-slate-900">{INFO[active]?.title ?? "Info"}</h2>
        </div>
        <button
          type="button"
          onClick={close}
          className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          aria-label="Close notes"
          title="Close (Esc)"
        >
          <svg viewBox="0 0 20 20" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <path d="M5 5l10 10M15 5L5 15" />
          </svg>
        </button>
      </header>
      <div ref={bodyRef} className="note-body prose prose-slate prose-sm max-w-none flex-1 overflow-y-auto px-6 py-5">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
        components={{
          code: NoteCode,
          a: ({ href, children, ...props }) => {
            if (href && href.startsWith("#info:")) {
              const id = href.slice("#info:".length);
              return (
                <a href={href} onClick={(e) => { e.preventDefault(); e.stopPropagation(); if (INFO[id]) open(id); }} className="note-link" {...props}>
                  {children}
                </a>
              );
            }
            if (href && href.startsWith("#hover:")) {
              const id = href.slice("#hover:".length);
              return (
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const rect = e.currentTarget.getBoundingClientRect();
                    anchorRectRef.current = rect;
                    // Seed a provisional position; final position will be adjusted by recomputePopup
                    const left = Math.min(rect.left, (typeof window !== "undefined" ? window.innerWidth : 1200) - 360);
                    const top = Math.max(8, rect.top - 16);
                    setHoverPos({ top, left });
                    anchorRectRef.current = rect;
                    setHoverInfoId((prev) => (prev === id ? null : id));
                    // In next frame, recompute with actual popup size
                    setTimeout(() => recomputePopup(), 0);
                  }}
                  className="note-link"
                  {...props}
                >
                  {children}
                </a>
              );
            }
            if (href === "#cdf") {
              return (
                <a href={href} onClick={(e) => { e.preventDefault(); e.stopPropagation(); open("cdf"); }} className="note-link" {...props}>
                  {children}
                </a>
              );
            }
            const isExternal = /^https?:\/\//.test(href || "");
            return (
              <a href={href} className="note-link" {...(isExternal ? { target: "_blank", rel: "noopener noreferrer" } : {})} {...props}>
                {children}
              </a>
            );
          },
        }}
      >
        {INFO[active]?.md || "_No content_"}
      </ReactMarkdown>
      {hoverInfoId && INFO[hoverInfoId] && (
        <div
          ref={popupRef}
          className="note-popup prose prose-slate prose-sm fixed z-50 overflow-auto rounded-xl border border-slate-200 bg-white p-4 shadow-xl"
          style={{ top: hoverPos.top, left: hoverPos.left, maxWidth: 720, maxHeight: '70vh', minWidth: 360 }}
          onClick={(e) => { e.stopPropagation(); }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
          >
            {INFO[hoverInfoId]?.md || "_No content_"}
          </ReactMarkdown>
        </div>
      )}
      </div>
    </aside>
  );
}
