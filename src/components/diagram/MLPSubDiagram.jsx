import React from "react";
import Node from "./Node";
import Arrow from "./Arrow";
import FrameTitle from "./FrameTitle";

export default function MLPSubDiagram({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={200} height={300} rx={12} fill="#F8FAFC" stroke="#C7D2FE" strokeDasharray="5 4" />
      <FrameTitle id="mlp_subdiagram" label="MLP Architecture" x={x + 100} y={y + 24} onClick={onClick} />
      <Node id="mlp_linear1" label="Linear" x={x + 20} y={y + 50} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 80} x2={x + 100} y2={y + 115} />
      <Node id="mlp_gelu" label="GELU" x={x + 20} y={y + 115} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 145} x2={x + 100} y2={y + 180} />
      <Node id="mlp_linear2" label="Linear" x={x + 20} y={y + 180} w={160} h={30} onClick={onClick} />
      <Arrow x1={x + 100} y1={y + 210} x2={x + 100} y2={y + 245} />
      <Node id="mlp_dropout" label="Residual Dropout" x={x + 20} y={y + 245} w={160} h={30} onClick={onClick} />
    </g>
  );
}
