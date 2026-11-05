import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";
import FrameTitle from "./FrameTitle";

const STEPS = [
  ["attn_linear_qkv", "Linear for Q, K, V"],
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_dropout", "Attn dropout"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
  ["attn_resid_dropout", "Residual dropout"],
];

export default function AttnSubDiagram({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={220} height={480} rx={12} fill="#F8FAFC" stroke="#C7D2FE" strokeDasharray="5 4" />
      <FrameTitle id="attn_class" label="Causal Self-Attention" x={x + 110} y={y + 24} onClick={onClick} />
      {STEPS.map(([id, label], i) => {
        const top = y + 50 + i * 54;
        return (
          <React.Fragment key={id}>
            <Node id={id} label={label} x={x + 20} y={top} w={180} h={28} onClick={onClick} />
            {i < STEPS.length - 1 && <Arrow x1={x + 110} y1={top + 28} x2={x + 110} y2={top + 54} />}
          </React.Fragment>
        );
      })}
    </g>
  );
}
