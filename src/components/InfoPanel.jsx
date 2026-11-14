import React, { useState, useRef, useLayoutEffect, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import { useInfo } from "../context/InfoContext";
import { INFO } from "../content/info";
import { INFO_ZH } from "../content/info_zh";

// The note to show in the chosen language; a note with no Chinese version falls back to English.
function getNote(id, lang) {
  const en = INFO[id];
  const zh = lang === "zh" ? INFO_ZH[id] : null;
  return { title: zh?.title ?? en?.title, md: zh?.md ?? en?.md, missingZh: lang === "zh" && !zh };
}

// GFM adds tables; singleTilde is off so a lone "~" (as in "~10×") never becomes strikethrough.
const REMARK_PLUGINS = [remarkMath, [remarkGfm, { singleTilde: false }]];

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

// One click popup (a "#hover:" note). Popups can stack: a "#hover:" link inside a popup opens another one on top.
function Popup({ id, anchor, level, lang, components }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(() => ({
    top: Math.max(8, anchor.top - 16),
    left: Math.max(8, Math.min(anchor.left, (typeof window !== "undefined" ? window.innerWidth : 1200) - 360)),
  }));

  // Place the popup next to its link: above, below, right or left, whichever fits first.
  const place = useCallback(() => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const ar = anchor;
    const m = 8;
    const clampLeft = (l) => Math.min(Math.max(l, m), vw - r.width - m);
    const clampTop = (t) => Math.min(Math.max(t, m), vh - r.height - m);
    const candidates = [
      { top: ar.top - r.height - m, left: clampLeft(ar.left), fits: ar.top - r.height - m >= m },
      { top: ar.bottom + m, left: clampLeft(ar.left), fits: ar.bottom + r.height + m <= vh - m },
      { top: clampTop(ar.top), left: ar.right + m, fits: ar.right + r.width + m <= vw - m },
      { top: clampTop(ar.top), left: ar.left - r.width - m, fits: ar.left - r.width - m >= m },
    ];
    const c = candidates.find((c) => c.fits);
    setPos(c
      ? { top: Math.max(m, Math.min(c.top, vh - r.height - m)), left: Math.max(m, Math.min(c.left, vw - r.width - m)) }
      : { top: Math.max(m, Math.min(ar.bottom + m, vh - r.height - m)), left: Math.max(m, Math.min(ar.left, vw - r.width - m)) });
  }, [anchor]);

  useLayoutEffect(() => {
    place();
    const ro = new ResizeObserver(place);
    if (ref.current) ro.observe(ref.current);
    window.addEventListener("resize", place);
    return () => { ro.disconnect(); window.removeEventListener("resize", place); };
  }, [place]);

  return (
    <div
      ref={ref}
      data-popup-level={level}
      className="note-popup prose prose-slate prose-sm fixed overflow-auto rounded-xl border border-slate-200 bg-white p-4 shadow-xl"
      style={{ zIndex: 50 + level, top: pos.top, left: pos.left, maxWidth: "min(720px, calc(100vw - 16px))", maxHeight: "70vh", minWidth: "min(360px, calc(100vw - 16px))" }}
      onClick={(e) => e.stopPropagation()}
    >
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
        components={components}
      >
        {getNote(id, lang).md || "_No content_"}
      </ReactMarkdown>
    </div>
  );
}

export default function InfoPanel() {
  const { active, previous, close, open, back, lang } = useInfo();
  const bodyRef = useRef(null);
  // Open popups, bottom to top: [{ id, anchor }]. Index = level; the note body is level -1.
  const [popups, setPopups] = useState([]);

  // Esc closes the top popup first, then the drawer.
  useEffect(() => {
    if (!active) return;
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      if (popups.length) setPopups((ps) => ps.slice(0, -1));
      else close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, popups.length, close]);

  // A click inside a popup closes the popups above it; a click outside all popups closes them all.
  // (A click on a "#hover:" link is handled by the link itself.)
  useEffect(() => {
    if (!popups.length) return;
    const onDown = (e) => {
      if (e.target.closest?.('a[href^="#hover:"]')) return;
      const el = e.target.closest?.("[data-popup-level]");
      const level = el ? Number(el.dataset.popupLevel) : -1;
      setPopups((ps) => ps.slice(0, level + 1));
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [popups.length]);

  // Each new note starts at the top, and popups from the previous note go away.
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    setPopups([]);
  }, [active]);

  // Markdown components for the note body (level -1) or a popup at the given level.
  const makeComponents = (level) => ({
    code: NoteCode,
    a: ({ href, children, ...props }) => {
      delete props.node;
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
              const anchor = e.currentTarget.getBoundingClientRect();
              // Keep the popups up to this level; toggle the one directly above it.
              setPopups((ps) => {
                const base = ps.slice(0, level + 1);
                return ps[level + 1]?.id === id ? base : [...base, { id, anchor }];
              });
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
  });

  if (!active) return null;

  const note = getNote(active, lang);
  const prevTitle = previous ? getNote(previous, lang).title : null;

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
            title={`Back to ${prevTitle ?? "previous note"}`}
          >
            <span aria-hidden="true">←</span>
            <span className="truncate">{prevTitle ?? "Back"}</span>
          </button>
        )}
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">Notes</p>
          <h2 className="truncate text-base font-semibold text-slate-900">{note.title ?? "Info"}</h2>
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
      <div ref={bodyRef} lang={lang === "zh" && !note.missingZh ? "zh-CN" : "en"} className="note-body prose prose-slate prose-sm max-w-none flex-1 overflow-y-auto px-6 py-5">
      {note.missingZh && (
        <p className="not-prose mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          这篇笔记的中文版暂缺，下面显示英文原文。
        </p>
      )}
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
        components={makeComponents(-1)}
      >
        {note.md || "_No content_"}
      </ReactMarkdown>
      {popups.map((pp, i) =>
        INFO[pp.id] ? <Popup key={`${i}:${pp.id}`} id={pp.id} anchor={pp.anchor} level={i} lang={lang} components={makeComponents(i)} /> : null
      )}
      </div>
    </aside>
  );
}
