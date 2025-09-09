import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";

export default function TransformerBlockGroup({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={380} height={290} rx={16} className="fill-slate-50 stroke-slate-400" />
      <text x={x + 190} y={y + 20} textAnchor="middle" className="text-[11px] fill-slate-700 cursor-pointer" style={{ textDecoration: "underline" }} onClick={() => onClick("gpt2_blocks")} underline="true">
        Transformer Block (×12)
      </text>

      <Node id="block_ln" label="LayerNorm" x={x + 20} y={y + 50} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 80} x2={x + 190} y2={y + 105} />

      <Node id="block_attn" label="Masked Multi-Head Attention" x={x + 20} y={y + 110} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 140} x2={x + 190} y2={y + 165} />

      <Node id="block_ln2" label="LayerNorm" x={x + 20} y={y + 170} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 200} x2={x + 190} y2={y + 225} />

      <Node id="block_mlp" label="MLP / Feed-Forward" x={x + 20} y={y + 230} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 365} y1={y + 245} x2={590} y2={y + 245} />
    </g>
  );
}
