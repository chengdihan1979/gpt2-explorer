import React from "react";
import { useSelectedId } from "./SelectionContext";

// Clickable heading of a group frame (GPT-2, Transformer Block, sub-diagrams).
export default function FrameTitle({ id, label, x, y, anchor = "middle", onClick }) {
  const selected = useSelectedId() === id;
  const width = label.length * 7.6 + 20;
  const left = anchor === "middle" ? x - width / 2 : x - 9;

  return (
    <g
      className={`diagram-title${selected ? " is-selected" : ""}`}
      onClick={() => onClick(id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick(id);
        }
      }}
      role="button"
      tabIndex={0}
      aria-label={label}
      aria-pressed={selected}
    >
      <rect x={left} y={y - 12} width={width} height={24} rx={12} className="diagram-title-pill" />
      <text x={x} y={y} textAnchor={anchor} dominantBaseline="central" fontSize={13} fontWeight={600} className="select-none">
        {label}
      </text>
    </g>
  );
}
