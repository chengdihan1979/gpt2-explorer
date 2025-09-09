import React, { useEffect, useRef } from "react";
import { useInfo } from "../context/InfoContext";

export default function CodeWithActions({ html, onSelect }) {
  const { open } = useInfo();
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
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

  if (!html) return null;

  return (
    <pre className="text-sm overflow-x-auto border rounded p-2 bg-slate-50">
      <code ref={ref} dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}
