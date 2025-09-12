import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";

export default function AttnSubDiagram({ x, y, onClick }) {
  const node = (id, label, offsetY) => (
    <Node id={id} label={label} x={x + 20} y={y + offsetY} w={180} h={28} onClick={onClick} />
  );

  return (
    <g>
      <rect x={x} y={y} width={220} height={480} rx={12} className="fill-slate-50 stroke-slate-400" />
      <text
        x={x + 110}
        y={y + 25}
        textAnchor="middle"
        className="text-[11px] fill-slate-700 cursor-pointer"
        style={{ textDecoration: "underline" }}
        onClick={() => onClick("attn_class")}
      >
        Causal Self-Attention
      </text>

      {node("attn_linear_qkv", "Linear for Q, K, V", 50)}
      <Arrow x1={x + 110} y1={y + 78} x2={x + 110} y2={y + 104} />

      {node("attn_matmul_qk", "Matmul (Q × Kᵀ)", 104)}
      <Arrow x1={x + 110} y1={y + 132} x2={x + 110} y2={y + 158} />

      {node("attn_masked", "Masked Attention", 158)}
      <Arrow x1={x + 110} y1={y + 186} x2={x + 110} y2={y + 212} />

      {node("attn_softmax", "Softmax", 212)}
      <Arrow x1={x + 110} y1={y + 240} x2={x + 110} y2={y + 266} />

      {node("attn_dropout", "Attn dropout", 266)}
      <Arrow x1={x + 110} y1={y + 294} x2={x + 110} y2={y + 320} />

      {node("attn_matmul_v", "Matmul (att × V)", 320)}
      <Arrow x1={x + 110} y1={y + 348} x2={x + 110} y2={y + 374} />

      {node("attn_linear_out", "Linear", 374)}
      <Arrow x1={x + 110} y1={y + 402} x2={x + 110} y2={y + 428} />

      {node("attn_resid_dropout", "Residual dropout", 428)}
    </g>
  );
}


