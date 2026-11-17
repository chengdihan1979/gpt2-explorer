import React from "react";
import Node from "../../../components/diagram/Node";
import Arrow from "../../../components/diagram/Arrow";
import { LINE } from "../../../components/diagram/palette";
import GPT2OuterFrame from "./GPT2OuterFrame";
import TransformerBlockGroup from "./TransformerBlockGroup";
import MLPSubDiagram from "./MLPSubDiagram";
import AttnSubDiagram from "./AttnSubDiagram";

// The GPT-2 architecture drawn inside the shared <svg> (viewBox 0 0 836 848).
// Every node id here must have a matching entry in ../snippets.js `content`, or map to one via resolveCodeId.
export default function GPT2Diagram({ onSelect }) {
  return (
    <>
      <GPT2OuterFrame onClick={onSelect} />

      {/* Input data and tokenizer stacked directly above embeddings */}
      <Node id="input_data" label="Input Data (Text)" x={20} y={50} w={180} h={40} onClick={onSelect} />
      <Arrow x1={110} y1={90} x2={110} y2={110} />
      <Node id="tokenizer" label="Tokenizer" x={20} y={110} w={180} h={40} onClick={onSelect} />

      <Arrow x1={110} y1={150} x2={110} y2={170} />
      <Node id="token_embed" label="Token Embeddings" x={20} y={170} w={180} h={40} onClick={onSelect} />
      <text x={210} y={190} textAnchor="middle" dominantBaseline="central" fontSize={16} fontWeight={500} fill="#64748B">+</text>
      <Node id="pos_embed" label="Positional Embeddings" x={220} y={170} w={180} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={210} x2={210} y2={240} />
      <Node id="emb_dropout" label="Dropout" x={20} y={240} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={280} x2={210} y2={300} />
      <TransformerBlockGroup x={20} y={300} onClick={onSelect} />
      {/* Residual connection from the embedding Dropout output into the attention residual add */}
      <polyline points="215,290 500,290 500,455 220,455" stroke={LINE} strokeWidth={1.25} fill="none" markerEnd="url(#arrowhead)" />

      <Arrow x1={210} y1={590} x2={210} y2={620} />
      <Node id="final_ln" label="Final LayerNorm" x={20} y={620} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={660} x2={210} y2={700} />
      <Node id="lm_head" label="LM Head (Linear Layer)" x={20} y={700} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={740} x2={210} y2={780} />
      <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={780} w={380} h={40} onClick={onSelect} />

      <MLPSubDiagram x={600} y={530} onClick={onSelect} />
      <AttnSubDiagram x={600} y={40} onClick={onSelect} />
      {/* Attention block expands into the Causal Self-Attention sub-diagram */}
      <Arrow x1={400} y1={410} x2={600} y2={120} variant="detail" />
    </>
  );
}
