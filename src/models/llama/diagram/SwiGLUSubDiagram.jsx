import React from "react";
import Node from "../../../components/diagram/Node";
import Arrow from "../../../components/diagram/Arrow";
import FrameTitle from "../../../components/diagram/FrameTitle";

// LLaMA's MLP: two parallel projections (gate, up); SiLU on the gate; multiply element-wise; project down.
export default function SwiGLUSubDiagram({ x, y, onClick }) {
  return (
    <g>
      <rect x={x} y={y} width={220} height={280} rx={12} fill="#F8FAFC" stroke="#C7D2FE" strokeDasharray="5 4" />
      <FrameTitle id="mlp_subdiagram" label="SwiGLU MLP" x={x + 110} y={y + 24} onClick={onClick} />

      <Node id="mlp_linear_gate" label="Linear (gate)" x={x + 10} y={y + 50} w={95} h={30} onClick={onClick} />
      <Node id="mlp_linear_up" label="Linear (up)" x={x + 115} y={y + 50} w={95} h={30} onClick={onClick} />

      <Arrow x1={x + 57.5} y1={y + 80} x2={x + 57.5} y2={y + 110} />
      <Node id="mlp_silu" label="SiLU" x={x + 10} y={y + 110} w={95} h={30} onClick={onClick} />

      <Arrow x1={x + 57.5} y1={y + 140} x2={x + 57.5} y2={y + 170} />
      <Arrow x1={x + 162.5} y1={y + 80} x2={x + 162.5} y2={y + 170} />
      <Node id="mlp_mul" label="Multiply (⊙)" x={x + 20} y={y + 170} w={180} h={30} onClick={onClick} />

      <Arrow x1={x + 110} y1={y + 200} x2={x + 110} y2={y + 230} />
      <Node id="mlp_linear_down" label="Linear (down)" x={x + 20} y={y + 230} w={180} h={30} onClick={onClick} />
    </g>
  );
}
