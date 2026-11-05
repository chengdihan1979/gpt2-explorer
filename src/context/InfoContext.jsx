import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

export const InfoContext = createContext(null);

// Notes are kept as a history stack so "Back" returns to whichever note linked here.
export function InfoProvider({ children }) {
  const [stack, setStack] = useState([]);

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
    }),
    [stack, open, back, close]
  );

  return <InfoContext.Provider value={value}>{children}</InfoContext.Provider>;
}

export function useInfo() {
  const ctx = useContext(InfoContext);
  if (!ctx) throw new Error("useInfo must be used within <InfoProvider>");
  return ctx;
}
