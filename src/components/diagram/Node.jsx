import React from "react";

const getFillColor = (id) => {
  if (id.includes("embed")) return "fill-amber-300";
  if (id.includes("ln")) return "fill-purple-300";
  if (id === "mlp_dropout" || id === "emb_dropout") return "fill-red-300";
  if (id.includes("attn")) return "fill-green-300";
  if (id.includes("mlp_linear") || id.includes("lm_head")) return "fill-indigo-300";
  if (id.includes("mlp")) return "fill-pink-300";
  if (id.includes("loss")) return "fill-yellow-300";
  return "fill-slate-200";
};

export default function Node({ id, label, x, y, w, h, onClick, underline }) {
  return (
    <g onClick={() => onClick(id)} className="cursor-pointer">
      <rect x={x} y={y} width={w} height={h} rx={10} className={`transition-all duration-200 ${getFillColor(id)} stroke-slate-500`} />
      <text x={x + w / 2} y={y + h / 2} dominantBaseline="middle" textAnchor="middle" className="select-none text-[10px] fill-slate-800" style={underline ? { textDecoration: "underline" } : undefined}>
        {label}
      </text>
    </g>
  );
}
