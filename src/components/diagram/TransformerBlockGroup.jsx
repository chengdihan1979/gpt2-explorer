import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";
import FrameTitle from "./FrameTitle";
import { LINE } from "./palette";
import { useSelectedId } from "./SelectionContext";

// Encircled plus marking a residual add; clickable.
function ResidualPlus({ id, cx, cy, onClick }) {
  const selected = useSelectedId() === id;
  return (
    <g
      className={`diagram-plus${selected ? " is-selected" : ""}`}
      onClick={() => onClick(id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick(id);
        }
      }}
      role="button"
      tabIndex={0}
      aria-label="Residual add"
      aria-pressed={selected}
    >
      <circle cx={cx} cy={cy} r={8} />
      <line x1={cx - 4.5} y1={cy} x2={cx + 4.5} y2={cy} />
      <line x1={cx} y1={cy - 4.5} x2={cx} y2={cy + 4.5} />
    </g>
  );
}

export default function TransformerBlockGroup({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={380} height={290} rx={14} fill="#F8FAFC" stroke="#CBD5E1" />
      <FrameTitle id="stack" label="Transformer Block × 12" x={x + 190} y={y + 22} onClick={onClick} />

      <Node id="block_ln" label="LayerNorm" x={x + 20} y={y + 50} w={340} h={30} onClick={onClick} />
      <Arrow x1={x + 190} y1={y + 80} x2={x + 190} y2={y + 110} />

      <Node id="block_attn" label="Masked Multi-Head Attention" x={x + 20} y={y + 110} w={340} h={30} onClick={onClick} />
      <Arrow x1={x + 190} y1={y + 140} x2={x + 190} y2={y + 170} />

      <ResidualPlus id="block_attn_plus" cx={x + 190} cy={y + 152} onClick={onClick} />

      <Node id="block_ln2" label="LayerNorm" x={x + 20} y={y + 170} w={340} h={30} onClick={onClick} />
      <Arrow x1={x + 190} y1={y + 200} x2={x + 190} y2={y + 230} />

      <Node id="block_mlp" label="MLP / Feed-Forward" x={x + 20} y={y + 230} w={340} h={30} onClick={onClick} />
      <ResidualPlus id="block_mlp_plus" cx={x + 190} cy={y + 275} onClick={onClick} />

      {/* Residual path: right from LN2 input, down, then left into the MLP residual add */}
      <polyline
        points={`${x + 195},${y + 215} ${x + 480},${y + 215} ${x + 480},${y + 275} ${x + 200},${y + 275}`}
        stroke={LINE}
        strokeWidth={1.25}
        fill="none"
        markerEnd="url(#arrowhead)"
      />
      {/* MLP block expands into the MLP sub-diagram */}
      <Arrow x1={x + 365} y1={y + 245} x2={590} y2={y + 245} variant="detail" />
    </g>
  );
}
