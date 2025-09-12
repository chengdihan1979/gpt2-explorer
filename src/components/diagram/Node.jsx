import React from "react";

const getFillColor = (id) => {
  if (id.includes("embed")) return "fill-amber-300";
  if (id.includes("ln")) return "fill-purple-300";
  // Unify all dropout nodes to the same color as MLP dropout
  if (id.includes("dropout")) return "fill-red-300";
  // Use the same distinct blue for both Matmul steps in attention
  if (id === "attn_matmul_qk" || id === "attn_matmul_v") return "fill-blue-300";
  // Give Softmax a unique color not used elsewhere
  if (id === "attn_softmax") return "fill-teal-300";
  // Match Linear color for MLP and reuse it for attention linear steps
  if (id.includes("mlp_linear") || id.includes("lm_head") || id.includes("attn_linear")) return "fill-indigo-300";
  if (id.includes("attn")) return "fill-green-300";
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
