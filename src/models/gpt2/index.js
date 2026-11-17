import GPT2Diagram from "./diagram/GPT2Diagram";
import { attnCode, content, gpt2TrainingCode, mlpCode, transformerBlockCode } from "./snippets";

// Diagram nodes that show the code of a larger enclosing block instead of their own.
const CODE_ID = {
  mlp_linear1: "mlp_subdiagram",
  mlp_linear2: "mlp_subdiagram",
  mlp_gelu: "mlp_subdiagram",
  mlp_dropout: "mlp_subdiagram",
  emb_dropout: "gpt2",
  gpt2_blocks: "gpt2",
  final_ln: "gpt2",
  lm_head: "gpt2",
  block_attn_plus: "block_attn",
  block_mlp_plus: "block_mlp",
};

// Everything the shared ModelExplorer page needs to show one model.
export default {
  id: "gpt2",
  name: "GPT-2",
  badge: "G2",
  diagram: { Component: GPT2Diagram, viewBox: "0 0 836 848", label: "GPT-2 architecture diagram" },
  defaultSelected: "token_embed",
  content,
  resolveCodeId: (id) => CODE_ID[id] ?? id,
  fileNames: new Map([
    [gpt2TrainingCode, "train_gpt2.py"],
    [transformerBlockCode, "block.py"],
    [attnCode, "attention.py"],
    [mlpCode, "mlp.py"],
  ]),
};
