import React from "react";
import FrameTitle from "./FrameTitle";

export default function GPT2OuterFrame({ onClick }) {
  return (
    <g>
      <rect x={8} y={2} width={820} height={840} rx={18} fill="none" stroke="#CBD5E1" pointerEvents="none" />
      <FrameTitle id="gpt2" label="GPT-2" x={26} y={22} anchor="start" onClick={onClick} />
    </g>
  );
}
