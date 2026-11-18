import LlamaDiagram from "./diagram/LlamaDiagram";
import { attnCode, blockCode, content, llamaTrainingCode, mlpCode } from "./snippets";

// LLaMA 3 8B. Shared groups (outer frame, transformer block) are relabeled through the diagram spec;
// 🆕 marks the parts that changed compared with GPT-2.
export default {
  id: "llama",
  name: "LLaMA",
  badge: "L3",
  diagram: {
    Component: LlamaDiagram,
    viewBox: "0 0 836 848",
    label: "LLaMA architecture diagram",
    spec: {
      labels: {
        gpt2: "LLaMA 3 8B",
        stack: "Transformer Block × 32",
        block_ln: "RMSNorm",
        block_ln2: "RMSNorm",
        block_attn: "Grouped-Query Attention",
        block_mlp: "SwiGLU MLP",
        attn_class: "Grouped-Query Attention",
      },
      newIds: new Set(["block_ln", "block_ln2", "final_ln", "block_attn", "block_mlp", "attn_rope", "attn_repeat_kv", "mlp_silu", "mlp_mul"]),
    },
  },
  defaultSelected: "token_embed",
  content,
  resolveCodeId: (id) => id,
  // Shared code-link rules that should open a LLaMA note instead of the GPT-2 one.
  noteOverrides: {
    embedding_tok: "llama_embedding_notes",
    lm_head_notes: "llama_lm_head_notes",
  },
  fileNames: new Map([
    [llamaTrainingCode, "train_llama.py"],
    [blockCode, "block.py"],
    [attnCode, "attention.py"],
    [mlpCode, "mlp.py"],
  ]),
};
