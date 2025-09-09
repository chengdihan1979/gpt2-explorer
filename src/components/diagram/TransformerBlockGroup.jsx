import React from "react";
import { useInfo } from "../../context/InfoContext";
import Node from "./Node";
import Arrow from "./Arrow";

export default function TransformerBlockGroup({ x, y, onClick }) {
  const { open } = useInfo();
  return (
    <g>
      <rect x={x} y={y} width={380} height={290} rx={16} className="fill-slate-50 stroke-slate-400" />
      {/* <text x={x + 190} y={y + 20} textAnchor="middle" className="text-[11px] fill-slate-700 cursor-pointer" style={{ textDecoration: "underline" }} onClick={() => open("highlevel_transformer_introduction_notes")} underline="true">
        Transformer Block (×12)
      </text> */}

      <text x={x + 190} y={y + 20} textAnchor="middle" className="text-[11px] fill-slate-700 cursor-pointer" style={{ textDecoration: "underline" }} onClick={() => onClick("stack")} underline="true">
        Transformer Block (×12)
      </text>
      <Node id="block_ln" label="LayerNorm" x={x + 20} y={y + 50} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 80} x2={x + 190} y2={y + 110} />

      <Node id="block_attn" label="Masked Multi-Head Attention" x={x + 20} y={y + 110} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 140} x2={x + 190} y2={y + 170} />

      {/* Encircled plus between Attention and LayerNorm (residual indicator) */}
      <g className="cursor-pointer" onClick={() => onClick("block_attn_plus")}>
        <circle cx={x + 190} cy={y + 152} r={8} className="fill-white stroke-slate-600" />
        <line x1={x + 190 - 5} y1={y + 152} x2={x + 190 + 5} y2={y + 152} stroke="black" />
        <line x1={x + 190} y1={y + 152 - 5} x2={x + 190} y2={y + 152 + 5} stroke="black" />
      </g>

      <Node id="block_ln2" label="LayerNorm" x={x + 20} y={y + 170} w={340} h={30} onClick={onClick} underline="true" />
      <Arrow x1={x + 190} y1={y + 200} x2={x + 190} y2={y + 230} />

      <Node id="block_mlp" label="MLP / Feed-Forward" x={x + 20} y={y + 230} w={340} h={30} onClick={onClick} underline="true" />
      {/* Encircled plus at bottom of MLP (residual indicator) */}
      <g className="cursor-pointer" onClick={() => onClick("block_mlp_plus")}>
        <circle cx={x + 190} cy={y + 275} r={8} className="fill-white stroke-slate-600" />
        <line x1={x + 190 - 5} y1={y + 275} x2={x + 190 + 5} y2={y + 275} stroke="black" />
        <line x1={x + 190} y1={y + 275 - 5} x2={x + 190} y2={y + 275 + 5} stroke="black" />
      </g>
      {/* Residual straight-line path: right from LN2, down, then left into MLP output */}
      <polyline
        points={`${x + 195},${y + 215} ${x + 480},${y + 215} ${x + 480},${y + 275} ${x + 200},${y + 275}`}
        stroke="black"
        fill="none"
        markerEnd="url(#arrowhead)"
      />
      <Arrow x1={x + 365} y1={y + 245} x2={590} y2={y + 245} />
    </g>
  );
}
