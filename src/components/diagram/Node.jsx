import React from "react";
import { CATEGORY, getCategory } from "./palette";
import { useSelectedId } from "./SelectionContext";

export default function Node({ id, label, x, y, w, h, onClick }) {
  const selected = useSelectedId() === id;
  const c = CATEGORY[getCategory(id)];
  const activate = () => onClick(id);

  return (
    <g
      className={`diagram-node${selected ? " is-selected" : ""}`}
      onClick={activate}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate();
        }
      }}
      role="button"
      tabIndex={0}
      aria-label={label}
      aria-pressed={selected}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={selected ? c.active : c.fill}
        stroke={c.stroke}
        strokeWidth={selected ? 2 : 1}
        filter={selected ? "url(#node-shadow)" : undefined}
      />
      <text
        x={x + w / 2}
        y={y + h / 2}
        dominantBaseline="central"
        textAnchor="middle"
        fontSize={13}
        fontWeight={selected ? 600 : 500}
        fill={c.text}
        className="select-none"
      >
        {label}
      </text>
    </g>
  );
}
