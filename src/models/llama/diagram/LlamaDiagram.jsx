import React from "react";
import Node from "../../../components/diagram/Node";
import Arrow from "../../../components/diagram/Arrow";
import { LINE } from "../../../components/diagram/palette";
import GPT2OuterFrame from "../../gpt2/diagram/GPT2OuterFrame";
import TransformerBlockGroup from "../../gpt2/diagram/TransformerBlockGroup";
import AttnSubDiagram from "../../gpt2/diagram/AttnSubDiagram";
import SwiGLUSubDiagram from "./SwiGLUSubDiagram";

// Attention steps for LLaMA: separate (fewer) K/V heads, RoPE on q and k, K/V heads repeated for GQA, no dropout.
const ATTN_STEPS = [
  ["attn_linear_qkv", "Linear for Q, K, V"],
  ["attn_rope", "RoPE (Q, K)"],
  ["attn_repeat_kv", "Repeat K, V heads"],
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
];

// Same layout as the GPT-2 diagram (viewBox 0 0 836 848), with LLaMA's parts:
// no position embedding and no dropout; RMSNorm, GQA with RoPE, and a SwiGLU MLP.
// Labels of the shared groups (frame title, block, norms) come from the model's diagram spec.
export default function LlamaDiagram({ onSelect }) {
  return (
    <>
      <GPT2OuterFrame onClick={onSelect} />

      <Node id="input_data" label="Input Data (Text)" x={20} y={50} w={180} h={40} onClick={onSelect} />
      <Arrow x1={110} y1={90} x2={110} y2={110} />
      <Node id="tokenizer" label="Tokenizer" x={20} y={110} w={180} h={40} onClick={onSelect} />

      <Arrow x1={110} y1={150} x2={110} y2={170} />
      <Node id="token_embed" label="Token Embeddings (no position embedding)" x={20} y={170} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={210} x2={210} y2={300} />
      <TransformerBlockGroup x={20} y={300} onClick={onSelect} />
      {/* Residual connection from the embeddings into the attention residual add */}
      <polyline points="215,290 500,290 500,455 220,455" stroke={LINE} strokeWidth={1.25} fill="none" markerEnd="url(#arrowhead)" />

      <Arrow x1={210} y1={590} x2={210} y2={620} />
      <Node id="final_ln" label="Final RMSNorm" x={20} y={620} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={660} x2={210} y2={700} />
      <Node id="lm_head" label="LM Head (Linear, not tied)" x={20} y={700} w={380} h={40} onClick={onSelect} />

      <Arrow x1={210} y1={740} x2={210} y2={780} />
      <Node id="loss_fn" label="Training Objective (CrossEntropyLoss)" x={20} y={780} w={380} h={40} onClick={onSelect} />

      <SwiGLUSubDiagram x={600} y={530} onClick={onSelect} />
      <AttnSubDiagram x={600} y={40} onClick={onSelect} steps={ATTN_STEPS} />
      {/* Attention block expands into the attention sub-diagram */}
      <Arrow x1={400} y1={410} x2={600} y2={120} variant="detail" />
    </>
  );
}
