import { createContext, useContext } from "react";

// Lets every diagram node know which block is currently selected without prop drilling.
export const SelectionContext = createContext(null);

export function useSelectedId() {
  return useContext(SelectionContext);
}
