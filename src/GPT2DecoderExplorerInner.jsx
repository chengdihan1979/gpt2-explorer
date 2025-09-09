import React, { useMemo, useState } from "react";
import Node from "./components/diagram/Node";
import Arrow from "./components/diagram/Arrow";
import GPT2OuterFrame from "./components/diagram/GPT2OuterFrame";
import TransformerBlockGroup from "./components/diagram/TransformerBlockGroup";
import MLPSubDiagram from "./components/diagram/MLPSubDiagram";
import CodeWithActions from "./components/CodeWithActions";
import InfoPanel from "./components/InfoPanel";
import { buildCodeHtml } from "./utils/highlight";
import { content } from "./content/snippets";
import { useInfo } from "./context/InfoContext";

export default function GPT2DecoderExplorerInner() {
  const { close } = useInfo();
  const [selectedId, setSelectedId] = useState("token_embed");

  const handleSelect = (id) => {
    setSelectedId(id);
    close();
  };

  const effectiveId = ["mlp_linear1", "mlp_linear2", "mlp_gelu", "mlp_dropout"].includes(selectedId)
    ? "mlp_subdiagram"
    : (selectedId === "token_embed"
        || selectedId === "pos_embed"
        || selectedId === "emb_dropout"
        || selectedId === "gpt2_blocks"
        || selectedId === "final_ln"
        || selectedId === "lm_head"
        ? "gpt2"
        : selectedId);

  const component = content[effectiveId] || { title: "Unknown", summary: "No data", code: "", anchors: [] };

  const codeHtml = useMemo(() => buildCodeHtml({ component, selectedId }), [component, selectedId]);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">GPT-2 Decoder Explorer</h1>

      <div className="grid md:grid-cols-2 gap-6">
        <section className="bg-white rounded-xl shadow p-4">
          <svg viewBox="0 0 900 900" className="w-full h-auto">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" />
              </marker>
            </defs>

            <GPT2OuterFrame onClick={handleSelect} />

            <Node id="token_embed" label="Token Embeddings" x={20} y={40} w={180} h={40} onClick={handleSelect} />
            <text x={210} y={65} textAnchor="middle" className="text-lg fill-slate-800">+</text>
            <Node id="pos_embed" label="Positional Embeddings" x={220} y={40} w={180} h={40} onClick={handleSelect} />
            <Arrow x1={210} y1={80} x2={210} y2={105} />
            <Node id="emb_dropout" label="Dropout" x={20} y={110} w={380} h={40} onClick={handleSelect} />

            <Arrow x1={210} y1={150} x2={210} y2={175} />
            <TransformerBlockGroup x={20} y={180} onClick={handleSelect} />

            <Arrow x1={210} y1={470} x2={210} y2={515} />
            <Node id="final_ln" label="Final LayerNorm" x={20} y={520} w={380} h={40} onClick={handleSelect} />

            <Arrow x1={210} y1={560} x2={210} y2={605} />
            <Node id="lm_head" label="LM Head (Linear Layer)" x={20} y={610} w={380} h={40} onClick={handleSelect} />

            <Arrow x1={210} y1={650} x2={210} y2={695} />
            <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={700} w={380} h={40} onClick={handleSelect} />

            <MLPSubDiagram x={600} y={280} onClick={handleSelect} />
          </svg>
        </section>

        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-medium mb-2">{component.title}</h2>
          <p className="text-sm text-slate-600 mb-4">{component.summary}</p>
          {codeHtml && <CodeWithActions html={codeHtml} onSelect={setSelectedId} />}
          <InfoPanel />
        </section>
      </div>
    </div>
  );
}
