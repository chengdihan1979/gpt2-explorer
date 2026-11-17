import React from "react";
import { InfoProvider } from "./context/InfoContext";
import ModelExplorer from "./ModelExplorer";
import { DEFAULT_MODEL_ID, findModel } from "./models";
import { useHashRoute } from "./useHashRoute";

// Picks the model page from the URL hash ("#/gpt2"); unknown or missing routes show the default model.
export default function RefExplorer() {
  const route = useHashRoute();
  const model = findModel(route) ?? findModel(DEFAULT_MODEL_ID);
  return (
    <InfoProvider>
      <ModelExplorer key={model.id} model={model} />
    </InfoProvider>
  );
}
