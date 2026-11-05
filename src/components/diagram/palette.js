// Semantic color categories for diagram nodes.
// fill = resting tint, active = selected tint, stroke = outline, text = label color.
export const CATEGORY = {
  input:   { label: "Input",          fill: "#F1F5F9", active: "#E2E8F0", stroke: "#94A3B8", text: "#334155" },
  embed:   { label: "Embedding",      fill: "#FEF3C7", active: "#FDE68A", stroke: "#F59E0B", text: "#78350F" },
  norm:    { label: "Normalization",  fill: "#EDE9FE", active: "#DDD6FE", stroke: "#8B5CF6", text: "#4C1D95" },
  attn:    { label: "Attention",      fill: "#D1FAE5", active: "#A7F3D0", stroke: "#10B981", text: "#064E3B" },
  linear:  { label: "Linear",         fill: "#E0E7FF", active: "#C7D2FE", stroke: "#6366F1", text: "#312E81" },
  matmul:  { label: "Matmul",         fill: "#E0F2FE", active: "#BAE6FD", stroke: "#0EA5E9", text: "#0C4A6E" },
  softmax: { label: "Softmax",        fill: "#CCFBF1", active: "#99F6E4", stroke: "#14B8A6", text: "#134E4A" },
  mlp:     { label: "MLP / GELU",     fill: "#FCE7F3", active: "#FBCFE8", stroke: "#EC4899", text: "#831843" },
  dropout: { label: "Dropout",        fill: "#FFE4E6", active: "#FECDD3", stroke: "#F43F5E", text: "#881337" },
  loss:    { label: "Loss",           fill: "#FEF9C3", active: "#FEF08A", stroke: "#EAB308", text: "#713F12" },
};

export function getCategory(id) {
  if (id.includes("embed")) return "embed";
  if (id === "input_data" || id === "tokenizer") return "input";
  if (id.includes("ln")) return "norm";
  if (id.includes("dropout")) return "dropout";
  if (id === "attn_matmul_qk" || id === "attn_matmul_v") return "matmul";
  if (id === "attn_softmax") return "softmax";
  if (id.includes("mlp_linear") || id.includes("lm_head") || id.includes("attn_linear")) return "linear";
  if (id.includes("attn")) return "attn";
  if (id.includes("mlp")) return "mlp";
  if (id.includes("loss")) return "loss";
  return "input";
}

export const LINE = "#94A3B8";
export const DETAIL_LINE = "#A5B4FC";
