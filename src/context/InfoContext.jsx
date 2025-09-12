import React, { createContext, useContext, useState } from "react";

export const InfoContext = createContext(null);

export function InfoProvider({ children }) {
  const [active, setActive] = useState(null);
  const [previous, setPrevious] = useState(null);
  const open = (id) => {
    setPrevious(active);
    setActive(id);
  };
  const close = () => {
    setActive(null);
    setPrevious(null);
  };
  return (
    <InfoContext.Provider value={{ active, previous, open, close }}>
      {children}
    </InfoContext.Provider>
  );
}

export function useInfo() {
  const ctx = useContext(InfoContext);
  if (!ctx) throw new Error("useInfo must be used within <InfoProvider>");
  return ctx;
}
