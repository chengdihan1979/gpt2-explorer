import { createContext, useContext } from "react";

// Per-model tweaks to a shared diagram, so one drawing can serve several models:
// - labels: { [nodeId]: text } replaces a node's or frame title's default label
// - newIds: Set of node ids to mark 🆕 (changed compared with the previous model)
const EMPTY = { labels: {}, newIds: new Set() };

export const DiagramSpecContext = createContext(EMPTY);

export function useDiagramSpec() {
  return useContext(DiagramSpecContext) ?? EMPTY;
}
