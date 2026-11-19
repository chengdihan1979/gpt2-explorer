import React from "react";
import { useInfo } from "../context/InfoContext";
import { MODELS } from "../models";

const REPO_URL = "https://github.com/chengdihan1979/llm-explorer";

// Top navigation between model pages ("#/<id>"); the current one is highlighted.
function ModelNav({ currentId }) {
  return (
    <nav aria-label="Models" className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs font-medium shadow-sm">
      {[{ id: "home", name: "Home" }, ...MODELS].map((m) => (
        <a
          key={m.id}
          href={m.id === "home" ? "#/" : `#/${m.id}`}
          aria-current={m.id === currentId ? "page" : undefined}
          className={`rounded-md px-1.5 py-1 sm:px-2.5 ${m.id === currentId ? "bg-slate-800 text-white" : "text-slate-600 hover:text-slate-900"}`}
        >
          {m.name}
        </a>
      ))}
    </nav>
  );
}

// The sticky top bar shared by every page: badge and title, page navigation, notes language, GitHub link.
export default function SiteHeader({ badge, currentId, subtitle }) {
  const { lang, setLang } = useInfo();
  return (
    <header className="sticky top-0 z-30 h-16 border-b border-slate-200 bg-white/85 backdrop-blur">
      <div className="mx-auto flex h-full max-w-[1680px] items-center justify-between gap-2 px-3 sm:gap-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-[13px] font-bold text-white shadow-sm">
            {badge}
          </div>
          <div className="hidden min-w-0 sm:block">
            <h1 className="truncate text-[15px] font-semibold tracking-tight text-slate-900">LLM Explorer</h1>
            <p className="hidden truncate text-xs text-slate-500 sm:block">
              {subtitle}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-3">
        <ModelNav currentId={currentId} />
        <div
          role="group"
          aria-label="Notes language"
          title="Notes language"
          className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs font-medium shadow-sm"
        >
          {[["en", "EN"], ["zh", "中文"]].map(([code, label]) => (
            <button
              key={code}
              type="button"
              onClick={() => setLang(code)}
              aria-pressed={lang === code}
              className={`rounded-md px-1.5 py-1 sm:px-2.5 ${lang === code ? "bg-indigo-600 text-white" : "text-slate-600 hover:text-slate-900"}`}
            >
              {label}
            </button>
          ))}
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
      </div>
    </header>
  );
}
