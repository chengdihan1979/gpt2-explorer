import React from "react";

export default function GPT2OuterFrame({ onClick }) {
  return (
    <g>
      <rect x={8} y={2} width={820} height={760} rx={18} className="fill-transparent stroke-slate-700" pointerEvents="none" />
      <text x={18} y={22} className="text-[13px] fill-slate-800 cursor-pointer" style={{ textDecoration: "underline", fontWeight: 600 }} onClick={() => onClick("gpt2")}>
        GPT-2
      </text>
    </g>
  );
}
