import React, { useMemo, useState } from "react";
import Node from "./components/diagram/Node";
import Arrow from "./components/diagram/Arrow";
import GPT2OuterFrame from "./components/diagram/GPT2OuterFrame";
import TransformerBlockGroup from "./components/diagram/TransformerBlockGroup";
import MLPSubDiagram from "./components/diagram/MLPSubDiagram";
import AttnSubDiagram from "./components/diagram/AttnSubDiagram";
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
    : (selectedId === "emb_dropout"
        || selectedId === "gpt2_blocks"
        || selectedId === "final_ln"
        || selectedId === "lm_head"
        ? "gpt2"
        : (selectedId === "block_attn_plus" ? "block_attn" : (selectedId === "block_mlp_plus" ? "block_mlp" : selectedId)));

  const component = content[effectiveId] || { title: "Unknown", summary: "No data", code: "", anchors: [] };

  const codeHtml = useMemo(() => buildCodeHtml({ component, selectedId }), [component, selectedId]);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">GPT-2 Decoder Explorer</h1>

      <div className="grid md:grid-cols-2 gap-6">
        <section className="bg-white rounded-xl shadow p-4">
          <svg viewBox="0 0 900 900" className="w-full h-auto">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" markerUnits="strokeWidth">
                <polygon points="0 0, 10 3.5, 0 7" fill="black" />
              </marker>
            </defs>

            <GPT2OuterFrame onClick={handleSelect} />

            {/* Input data and tokenizer stacked directly above embeddings */}
            <Node id="input_data" label="Input Data (Text)" x={20} y={50} w={180} h={40} onClick={handleSelect} />
            <Arrow x1={110} y1={90} x2={110} y2={110} />
            <Node id="tokenizer" label="Tokenizer" x={20} y={110} w={180} h={40} onClick={handleSelect} />

            {/* Token and positional embeddings (left row) */}
            {/* Diagonal or short connector from tokenizer to token embeddings */}
            <Arrow x1={110} y1={150} x2={110} y2={170} />
            <Node id="token_embed" label="Token Embeddings" x={20} y={170} w={180} h={40} onClick={handleSelect} />
            <text x={210} y={195} textAnchor="middle" className="text-lg fill-slate-800">+</text>
            <Node id="pos_embed" label="Positional Embeddings" x={220} y={170} w={180} h={40} onClick={handleSelect} />

            {/* Down to embedding dropout */}
            <Arrow x1={210} y1={210} x2={210} y2={240} />
            <Node id="emb_dropout" label="Dropout" x={20} y={240} w={380} h={40} onClick={handleSelect} />

            {/* Transformer blocks */}
            <Arrow x1={210} y1={280} x2={210} y2={300} />
            <TransformerBlockGroup x={20} y={300} onClick={handleSelect} />
            {/* Residual connection from embedding Dropout output (210,280) to Attention output (210,440) */}
            <polyline points="215,290 500,290 500,455 220,455" stroke="black" fill="none" markerEnd="url(#arrowhead)" />

            {/* Final LN, LM head, loss with extra spacing */}
            <Arrow x1={210} y1={590} x2={210} y2={620} />
            <Node id="final_ln" label="Final LayerNorm" x={20} y={620} w={380} h={40} onClick={handleSelect} />

            <Arrow x1={210} y1={660} x2={210} y2={700} />
            <Node id="lm_head" label="LM Head (Linear Layer)" x={20} y={700} w={380} h={40} onClick={handleSelect} />

            <Arrow x1={210} y1={740} x2={210} y2={780} />
            <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={780} w={380} h={40} onClick={handleSelect} />

            {/* MLP sub-diagram nudged up; keep clear of attention box (ends ~y=520) */}
            <MLPSubDiagram x={600} y={530} onClick={handleSelect} />
            {/* Attention sub-diagram inside GPT-2 frame */}
            <AttnSubDiagram x={600} y={40} onClick={handleSelect} />
            {/* Arrow from Masked Multi-Head Attention node to the Causal Self-Attention diagram */}
            <Arrow x1={400} y1={410} x2={600} y2={120} />
          </svg>
          {/* Move notes under the left SVG */}
          <div className="mt-4">
            <InfoPanel />
          </div>
        </section>

        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-medium mb-2">{component.title}</h2>
          <p className="text-sm text-slate-600 mb-4">{component.summary}</p>
          {codeHtml && <CodeWithActions html={codeHtml} onSelect={setSelectedId} />}
        </section>
      </div>
    </div>
  );
}
