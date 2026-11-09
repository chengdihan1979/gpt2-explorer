import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export const InfoContext = createContext(null);

const LANG_KEY = "gpt2-explorer:notes-lang";

function readLang() {
  try {
    return window.localStorage.getItem(LANG_KEY) === "zh" ? "zh" : "en";
  } catch {
    return "en";
  }
}

// Notes are kept as a history stack so "Back" returns to whichever note linked here.
// `lang` is the notes language ("en" | "zh"), remembered per browser.
export function InfoProvider({ children }) {
  const [stack, setStack] = useState([]);
  const [lang, setLang] = useState(readLang);

  useEffect(() => {
    try {
      window.localStorage.setItem(LANG_KEY, lang);
    } catch {
      // storage unavailable (private mode etc.): the choice just isn't remembered
    }
  }, [lang]);

  const open = useCallback((id) => {
    setStack((s) => (s[s.length - 1] === id ? s : [...s, id]));
  }, []);
  const back = useCallback(() => setStack((s) => s.slice(0, -1)), []);
  const close = useCallback(() => setStack([]), []);

  const value = useMemo(
    () => ({
      active: stack[stack.length - 1] ?? null,
      previous: stack[stack.length - 2] ?? null,
      open,
      back,
      close,
      lang,
      setLang,
    }),
    [stack, open, back, close, lang]
  );

  return <InfoContext.Provider value={value}>{children}</InfoContext.Provider>;
}

export function useInfo() {
  const ctx = useContext(InfoContext);
  if (!ctx) throw new Error("useInfo must be used within <InfoProvider>");
  return ctx;
}
