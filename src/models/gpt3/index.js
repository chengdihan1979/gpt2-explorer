import gpt2 from "../gpt2";
import { attnCode, content, gpt3TrainingCode, mlpCode, transformerBlockCode } from "./snippets";

// GPT-3 reuses GPT-2's diagram: only the labels that differ and the 🆕 marks are given here.
export default {
  id: "gpt3",
  name: "GPT-3",
  badge: "G3",
  diagram: {
    ...gpt2.diagram,
    label: "GPT-3 architecture diagram",
    spec: {
      labels: {
        gpt2: "GPT-3",
        stack: "Transformer Block × 96",
        attn_masked: "Masked Attention",
      },
      newIds: new Set(["attn_masked"]),
    },
  },
  defaultSelected: "attn_masked",
  content,
  resolveCodeId: gpt2.resolveCodeId,
  fileNames: new Map([
    [gpt3TrainingCode, "train_gpt3.py"],
    [transformerBlockCode, "block.py"],
    [attnCode, "attention.py"],
    [mlpCode, "mlp.py"],
  ]),
};
