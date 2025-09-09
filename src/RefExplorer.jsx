import React from "react";
import { InfoProvider } from "./context/InfoContext";
import GPT2DecoderExplorerInner from "./GPT2DecoderExplorerInner";

export default function RefExplorer() {
  return (
    <InfoProvider>
      <GPT2DecoderExplorerInner />
    </InfoProvider>
  );
}
