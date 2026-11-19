import React from "react";
import { InfoProvider } from "./context/InfoContext";
import ModelExplorer from "./ModelExplorer";
import HomePage from "./HomePage";
import { findModel } from "./models";
import { useHashRoute } from "./useHashRoute";

// Picks the page from the URL hash: "#/<model id>" shows that model; no route (or an unknown one) shows the home page.
export default function RefExplorer() {
  const route = useHashRoute();
  const model = findModel(route);
  return (
    <InfoProvider>
      {model ? <ModelExplorer key={model.id} model={model} /> : <HomePage />}
    </InfoProvider>
  );
}
