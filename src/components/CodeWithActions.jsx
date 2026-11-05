import React, { useEffect, useRef } from "react";
import { useInfo } from "../context/InfoContext";

export default function CodeWithActions({ html, lineCount, onSelect }) {
  const { open } = useInfo();
  const codeRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = codeRef.current;
    if (!el) return;

    const onClick = (e) => {
      const t = e.target.closest("[data-action]");
      if (!t) return;
      const action = t.getAttribute("data-action");
      if (!action) return;

      if (action.startsWith("open:")) {
        e.preventDefault();
        const id = action.split(":")[1];
        if (id) open(id);
      } else if (action.startsWith("select:")) {
        e.preventDefault();
        const id = action.split(":")[1];
        if (id && onSelect) onSelect(id);
      }
    };

    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [open, onSelect]);

  // Bring the highlighted code into view whenever the selection changes.
  useEffect(() => {
    const box = scrollRef.current;
    const mark = codeRef.current?.querySelector("mark");
    if (!box) return;
    if (!mark) {
      box.scrollTo({ top: 0 });
      return;
    }
    const top = mark.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
    box.scrollTo({ top: Math.max(0, top - box.clientHeight / 3), behavior: "smooth" });
  }, [html]);

  if (!html) return null;

  const gutter = Array.from({ length: lineCount }, (_, i) => i + 1).join("\n");

  return (
    <div ref={scrollRef} className="code-surface relative flex-1 min-h-0 overflow-auto rounded-xl">
      <div className="flex min-w-max py-4 font-mono text-[13px] leading-6">
        <pre aria-hidden="true" className="code-gutter sticky left-0 z-10 select-none pl-4 pr-4 text-right">
          {gutter}
        </pre>
        <pre className="pr-6">
          <code ref={codeRef} dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
    </div>
  );
}
