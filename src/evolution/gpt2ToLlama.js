import { relabel, remove, replaceGroup, scene } from "./scene";

// GPT-2 small → LLaMA 3 8B, one architectural change per step.
// Layout follows the GPT-2 / LLaMA pages (viewBox 0 0 836 848), with a margin on the left for the weight-tying link.

// ---------- Sub-diagrams, laid out from a list so they can shrink and grow ----------

// Attention steps stacked in a dashed frame at (600, 40).
function attention(steps, title) {
  const els = [
    { id: "attn:frame", type: "frame", x: 600, y: 40, w: 220, h: 102 + (steps.length - 1) * 54, dashed: true },
    { id: "attn:title", type: "title", x: 710, y: 64, label: title },
  ];
  steps.forEach(([id, label], i) => {
    els.push({ id, type: "node", x: 610, y: 90 + i * 54, w: 200, h: 28, label });
    if (i > 0) els.push({ id: `attn:link:${steps[i - 1][0]}>${id}`, type: "link", from: steps[i - 1][0], to: id });
  });
  return els;
}

// GPT-2's MLP: one column in a dashed frame at (600, 530).
function gpt2Mlp(steps) {
  const els = [
    { id: "mlp:frame", type: "frame", x: 600, y: 530, w: 200, h: 105 + (steps.length - 1) * 65, dashed: true },
    { id: "mlp:title", type: "title", x: 700, y: 554, label: "MLP Architecture" },
  ];
  steps.forEach(([id, label], i) => {
    els.push({ id, type: "node", x: 620, y: 580 + i * 65, w: 160, h: 30, label });
    if (i > 0) els.push({ id: `mlp:link:${steps[i - 1][0]}>${id}`, type: "link", from: steps[i - 1][0], to: id });
  });
  return els;
}

// LLaMA's SwiGLU MLP. The GPT-2 nodes keep their ids so they morph: Linear → gate, GELU → SiLU, Linear → down.
function swiglu() {
  return [
    { id: "mlp:frame", type: "frame", x: 600, y: 530, w: 220, h: 280, dashed: true },
    { id: "mlp:title", type: "title", x: 710, y: 554, label: "SwiGLU MLP" },
    { id: "mlp_linear1", type: "node", x: 610, y: 580, w: 95, h: 30, label: "Linear (gate)" },
    { id: "mlp_linear_up", type: "node", x: 715, y: 580, w: 95, h: 30, label: "Linear (up)" },
    { id: "mlp_gelu", type: "node", x: 610, y: 640, w: 95, h: 30, label: "SiLU" },
    { id: "mlp_mul", type: "node", x: 620, y: 700, w: 180, h: 30, label: "Multiply (⊙)", cat: "mlp" },
    { id: "mlp_linear2", type: "node", x: 620, y: 760, w: 180, h: 30, label: "Linear (down)" },
    { id: "mlp:link:mlp_linear1>mlp_gelu", type: "link", from: "mlp_linear1", to: "mlp_gelu" },
    { id: "mlp:link:mlp_gelu>mlp_mul", type: "link", from: "mlp_gelu", to: "mlp_mul" },
    { id: "mlp:link:mlp_linear_up>mlp_mul", type: "link", from: "mlp_linear_up", to: "mlp_mul" },
    { id: "mlp:link:mlp_mul>mlp_linear2", type: "link", from: "mlp_mul", to: "mlp_linear2" },
  ];
}

const GPT2_ATTN = [
  ["attn_linear_qkv", "Linear for Q, K, V"],
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_dropout", "Attn dropout"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
  ["attn_resid_dropout", "Residual dropout"],
];

const GPT2_MLP = [
  ["mlp_linear1", "Linear"],
  ["mlp_gelu", "GELU"],
  ["mlp_linear2", "Linear"],
  ["mlp_dropout", "Residual Dropout"],
];

const withoutDropout = (steps) => steps.filter(([id]) => !id.includes("dropout"));

// ---------- The starting scene: GPT-2 small ----------

const GPT2 = scene([
  { id: "outer", type: "frame", x: 8, y: 2, w: 820, h: 840 },
  { id: "model_title", type: "title", x: 26, y: 22, label: "GPT-2 small", anchor: "start" },

  { id: "input_data", type: "node", x: 20, y: 50, w: 180, h: 40, label: "Input Data (Text)" },
  { id: "tokenizer", type: "node", x: 20, y: 110, w: 180, h: 40, label: "Tokenizer · 50K vocab" },
  { id: "token_embed", type: "node", x: 20, y: 170, w: 180, h: 40, label: "Token Embeddings" },
  { id: "embed_plus", type: "text", x: 210, y: 190, label: "+", size: 16, color: "#64748B" },
  { id: "pos_embed", type: "node", x: 220, y: 170, w: 180, h: 40, label: "Positional Embeddings" },
  { id: "emb_dropout", type: "node", x: 20, y: 240, w: 380, h: 40, label: "Dropout" },
  { id: "link:input>tok", type: "link", from: "input_data", to: "tokenizer" },
  { id: "link:tok>emb", type: "link", from: "tokenizer", to: "token_embed" },
  { id: "link:emb>drop", type: "link", from: "token_embed", to: "emb_dropout", x: 210 },
  { id: "link:drop>block", type: "link", from: "emb_dropout", to: "block_frame_top", x: 210 },

  // Transformer block group (block_frame_top is an invisible anchor for links into the group)
  { id: "block_frame", type: "frame", x: 20, y: 300, w: 380, h: 290 },
  { id: "block_frame_top", type: "node", x: 20, y: 300, w: 380, h: 0, hidden: true },
  { id: "block_title", type: "title", x: 210, y: 322, label: "Transformer Block × 12" },
  { id: "block_ln", type: "node", x: 40, y: 350, w: 340, h: 30, label: "LayerNorm" },
  { id: "block_attn", type: "node", x: 40, y: 410, w: 340, h: 30, label: "Masked Multi-Head Attention" },
  { id: "block_ln2", type: "node", x: 40, y: 470, w: 340, h: 30, label: "LayerNorm" },
  { id: "block_mlp", type: "node", x: 40, y: 530, w: 340, h: 30, label: "MLP / Feed-Forward" },
  { id: "link:ln>attn", type: "link", from: "block_ln", to: "block_attn" },
  { id: "link:attn>ln2", type: "link", from: "block_attn", to: "block_ln2" },
  { id: "link:ln2>mlp", type: "link", from: "block_ln2", to: "block_mlp" },
  { id: "plus_attn", type: "plus", cx: 210, cy: 452 },
  { id: "plus_mlp", type: "plus", cx: 210, cy: 575 },
  { id: "resid_emb", type: "poly", points: [[215, 290], [500, 290], [500, 455], [220, 455]] },
  { id: "resid_mlp", type: "poly", points: [[215, 515], [500, 515], [500, 575], [220, 575]] },

  { id: "final_ln", type: "node", x: 20, y: 620, w: 380, h: 40, label: "Final LayerNorm" },
  { id: "lm_head", type: "node", x: 20, y: 700, w: 380, h: 40, label: "LM Head (tied to Token Embeddings)" },
  { id: "loss_fn", type: "node", x: 20, y: 780, w: 380, h: 40, label: "Training Objective (CrossEntropyLoss)" },
  { id: "link:block>final", type: "line", x1: 210, y1: 590, x2: 210, y2: 620 },
  { id: "link:final>head", type: "link", from: "final_ln", to: "lm_head" },
  { id: "link:head>loss", type: "link", from: "lm_head", to: "loss_fn" },

  // Weight tying: the LM head reuses the token embedding table
  { id: "tie", type: "poly", variant: "tie", points: [[20, 196], [-8, 196], [-8, 714], [20, 714]] },
  { id: "tie_label", type: "text", x: -16, y: 455, label: "same weights", size: 11, color: "#B45309", rotate: -90 },

  ...attention(GPT2_ATTN, "Causal Self-Attention"),
  ...gpt2Mlp(GPT2_MLP),
  { id: "detail_attn", type: "line", x1: 400, y1: 410, x2: 600, y2: 120, variant: "detail" },
  { id: "detail_mlp", type: "line", x1: 385, y1: 545, x2: 590, y2: 545, variant: "detail" },
]);

const LLAMA_ATTN = [
  ["attn_linear_qkv", "Linear for Q, K, V"],
  ["attn_rope", "RoPE (Q, K)"],
  ["attn_matmul_qk", "Matmul (Q × Kᵀ)"],
  ["attn_masked", "Masked Attention"],
  ["attn_softmax", "Softmax"],
  ["attn_matmul_v", "Matmul (att × V)"],
  ["attn_linear_out", "Linear"],
];

const LINEARS = ["attn_linear_qkv", "attn_linear_out", "mlp_linear1", "mlp_linear_up", "mlp_linear2"];

// Each step: what changes (`change`), which elements to highlight, and the caption.
// `note` opens that note in the drawer; `page` links to a model page.
const STEPS = [
  {
    title: { en: "GPT-2 small (2019)", zh: "起点：GPT-2 small（2019）" },
    body: {
      en: "12 Transformer blocks, width 768, 124M parameters. Step through the changes below to turn it into LLaMA 3 8B, one part at a time.",
      zh: "12 个 Transformer block，宽度 768，1.24 亿参数。往下一步步点，看它怎样一处一处变成 LLaMA 3 8B。",
    },
    page: "gpt2",
  },
  {
    title: { en: "A bigger vocabulary", zh: "词表变大" },
    body: {
      en: "The tokenizer grows from GPT-2's 50,257 tokens to 128,256. The same text splits into fewer tokens, especially non-English text and code.",
      zh: "tokenizer 的词表从 GPT-2 的 50,257 个 token 扩大到 128,256 个。同样一段文字切出来的 token 更少，中文等非英文文本和代码尤其明显。",
    },
    change: (s) => relabel(s, { tokenizer: "Tokenizer · 128K vocab" }),
    highlight: ["tokenizer"],
    note: "llama_tokenizer_notes",
  },
  {
    title: { en: "No position embedding", zh: "去掉位置 embedding" },
    body: {
      en: "GPT-2 adds a learned position vector to every token. LLaMA drops that table, so x holds only the token. Position comes back inside attention, as RoPE (a later step).",
      zh: "GPT-2 给每个 token 加上一个学出来的位置向量。LLaMA 去掉了这张表，x 里只有 token 本身。位置信息会在 attention 里面加回来，也就是后面要讲的 RoPE。",
    },
    change: (s) => {
      remove(s, ["pos_embed", "embed_plus"]);
      s.token_embed = { ...s.token_embed, w: 380, label: "Token Embeddings (no position embedding)" };
    },
    highlight: ["token_embed"],
    note: "llama_embedding_notes",
  },
  {
    title: { en: "No dropout", zh: "去掉 dropout" },
    body: {
      en: "All four dropouts go: after the embeddings, on the attention weights, and on the attention and MLP outputs. LLaMA trains on trillions of tokens and sees most of them only once, so overfitting is not the main worry.",
      zh: "四处 dropout 全部去掉：embedding 之后、attention 权重上、attention 输出和 MLP 输出上。LLaMA 用上万亿个 token 训练，大部分只看一遍，过拟合不是主要问题。",
    },
    change: (s) => {
      remove(s, ["emb_dropout", "link:emb>drop", "link:drop>block"]);
      s["link:emb>block"] = { id: "link:emb>block", type: "link", from: "token_embed", to: "block_frame_top", x: 210 };
      remove(s, ["attn_dropout", "attn_resid_dropout", "mlp_dropout"]);
      replaceGroup(s, "attn:", attention(withoutDropout(GPT2_ATTN), "Causal Self-Attention"));
      replaceGroup(s, "mlp:", gpt2Mlp(withoutDropout(GPT2_MLP)));
    },
    highlight: [],
  },
  {
    title: { en: "LayerNorm → RMSNorm", zh: "LayerNorm 换成 RMSNorm" },
    body: {
      en: "Every LayerNorm becomes RMSNorm: divide by the root mean square, with no mean subtraction and no bias. It sits in the same pre-norm places as in GPT-2.",
      zh: "所有 LayerNorm 都换成 RMSNorm：只除以均方根，不减均值，也没有 bias。位置和 GPT-2 一样，还是 pre-norm。",
    },
    change: (s) => relabel(s, { block_ln: "RMSNorm", block_ln2: "RMSNorm", final_ln: "Final RMSNorm" }),
    highlight: ["block_ln", "block_ln2", "final_ln"],
    note: "rmsnorm_notes",
  },
  {
    title: { en: "RoPE inside attention", zh: "在 attention 里加入 RoPE" },
    body: {
      en: "Right after q and k are computed, each pair of their dimensions is rotated by an angle set by the token's position. The score q·k then depends only on how far apart two tokens are. v is not rotated.",
      zh: "q 和 k 算出来之后，把它们的维度两两分对，按 token 的位置各转一个角度。这样 q·k 的分数只取决于两个 token 相隔多远。v 不转。",
    },
    change: (s) => {
      replaceGroup(s, "attn:", attention(LLAMA_ATTN, "Causal Self-Attention"));
    },
    highlight: ["attn_rope"],
    note: "rope_notes",
  },
  {
    title: { en: "Grouped-query attention", zh: "分组查询注意力（GQA）" },
    body: {
      en: "q keeps 32 heads, but k and v get only 8; each k/v head is shared by 4 query heads. This cuts the KV cache to a quarter, which saves memory and time when generating.",
      zh: "q 还是 32 个 head，k 和 v 只剩 8 个，每个 k/v head 由 4 个 query head 共用。KV cache 因此只有原来的四分之一，生成时省内存，也更快。",
    },
    change: (s) => {
      const steps = [...LLAMA_ATTN];
      steps[0] = ["attn_linear_qkv", "Linear Q, K, V (8 K/V heads)"];
      steps.splice(2, 0, ["attn_repeat_kv", "Repeat K, V heads × 4"]);
      replaceGroup(s, "attn:", attention(steps, "Grouped-Query Attention"));
      relabel(s, { block_attn: "Grouped-Query Attention" });
    },
    highlight: ["attn_linear_qkv", "attn_repeat_kv", "block_attn"],
    note: "gqa_notes",
  },
  {
    title: { en: "SwiGLU MLP", zh: "MLP 换成 SwiGLU" },
    body: {
      en: "The MLP gets two input projections. One goes through SiLU (replacing GELU) and acts as a gate: it is multiplied element by element with the other, then projected back down.",
      zh: "MLP 的输入投影变成两个。其中一路经过 SiLU（取代 GELU）当作\"门\"，和另一路逐元素相乘，再投影回原来的宽度。",
    },
    change: (s) => {
      replaceGroup(s, "mlp:", swiglu());
      relabel(s, { block_mlp: "SwiGLU MLP" });
    },
    highlight: ["mlp_linear1", "mlp_linear_up", "mlp_gelu", "mlp_mul", "mlp_linear2", "block_mlp"],
    note: "swiglu_notes",
  },
  {
    title: { en: "No biases", zh: "去掉 bias" },
    body: {
      en: "Every Linear layer drops its bias: the q/k/v and output projections in attention, and all three projections in the MLP. RMSNorm has no bias either.",
      zh: "所有 Linear 层都去掉了 bias：attention 里 q/k/v 的投影和输出投影，以及 MLP 里的三个投影。RMSNorm 也没有 bias。",
    },
    highlight: LINEARS,
  },
  {
    title: { en: "A separate LM head", zh: "LM head 不再共享权重" },
    body: {
      en: "GPT-2 reuses the token embedding table as the LM head (weight tying). LLaMA 3 8B gives the LM head its own weights: another 525M parameters, 6.5% of the model.",
      zh: "GPT-2 把 token embedding 表直接当作 LM head 的权重（weight tying）。LLaMA 3 8B 给 LM head 单独一份权重，多了 5.25 亿参数，占整个模型的 6.5%。",
    },
    change: (s) => {
      remove(s, ["tie", "tie_label"]);
      relabel(s, { lm_head: "LM Head (its own weights)" });
    },
    highlight: ["lm_head"],
    note: "llama_lm_head_notes",
  },
  {
    title: { en: "Scale up: LLaMA 3 8B", zh: "放大规模：LLaMA 3 8B" },
    body: {
      en: "Same skeleton, much bigger: 32 blocks instead of 12, width 4096 instead of 768, an 8K context instead of 1,024 tokens, and about 8B parameters instead of 124M.",
      zh: "骨架不变，规模大得多：block 从 12 个变成 32 个，宽度从 768 变成 4096，上下文从 1,024 变成 8K，参数从 1.24 亿变成约 80 亿。",
    },
    change: (s) => relabel(s, { model_title: "LLaMA 3 8B", block_title: "Transformer Block × 32" }),
    highlight: ["model_title", "block_title"],
    page: "llama",
  },
];

// Precompute the scene after each step.
function buildScenes(base, steps) {
  const out = [base];
  for (const step of steps.slice(1)) {
    const next = { ...out[out.length - 1] };
    if (step.change) step.change(next);
    out.push(next);
  }
  return out;
}

export default {
  id: "gpt2-llama",
  from: "GPT-2",
  to: "LLaMA 3",
  viewBox: "-30 0 866 848",
  steps: STEPS,
  scenes: buildScenes(GPT2, STEPS),
};
