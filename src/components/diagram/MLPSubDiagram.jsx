import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";

export default function MLPSubDiagram({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={200} height={300} rx={12} className="fill-slate-50 stroke-slate-400" />
      <text
        x={x + 100}
        y={y + 25}
        textAnchor="middle"
        className="text-[11px] fill-slate-700 cursor-pointer"
        style={{ textDecoration: "underline" }}
        onClick={() => onClick("mlp_subdiagram")}
      >
        MLP Architecture
      </text>
      <Node id="mlp_linear1" label="Linear" x={x + 20} y={y + 50} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 80} x2={x + 100} y2={y + 110} />
      <Node id="mlp_gelu" label="GELU" x={x + 20} y={y + 115} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 145} x2={x + 100} y2={y + 175} />
      <Node id="mlp_linear2" label="Linear" x={x + 20} y={y + 180} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 210} x2={x + 100} y2={y + 240} />
      <Node id="mlp_dropout" label="Dropout" x={x + 20} y={y + 245} w={160} h={30} onClick={onClick} />
    </g>
  );
}
