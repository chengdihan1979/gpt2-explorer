// GPT-3 code = the GPT-2 code with GPT-3's sizes, plus the one architectural change:
// every other layer uses "locally banded" attention (each token sees only the last `window` tokens).
// Built from the GPT-2 snippets with exact replacements; a replacement that no longer matches throws,
// so a change to the GPT-2 code cannot silently drift away from this page.
import { attnCode as gpt2AttnCode, content as gpt2Content, gpt2TrainingCode, mlpCode as gpt2MlpCode, transformerBlockCode } from "../gpt2/snippets";

function patch(code, replacements) {
  return replacements.reduce((out, [from, to]) => {
    if (!out.includes(from)) throw new Error(`GPT-3 snippet patch no longer matches: ${from}`);
    return out.replace(from, to);
  }, code);
}

const LOCAL_LAYERS = `        # GPT-3: attention alternates between dense and locally banded layers
        for i, blk in enumerate(self.blocks):
            if i % 2 == 1:
                blk.attn.make_local(window=256)  # window size is illustrative; the paper does not give it
`;

export const gpt3TrainingCode = patch(gpt2TrainingCode, [
  ["class GPT2(nn.Module):", "class GPT3(nn.Module):"],
  ["        # V = vocab_size = 50304, C = n_embd = 768", "        # V = vocab_size = 50304, C = n_embd = 12288"],
  ["# weight (1024, C)", "# weight (2048, C)"],
  ["            for _ in range(n_layer)\n        ])\n", "            for _ in range(n_layer)\n        ])\n" + LOCAL_LAYERS],
  ["    # B = batch, T = sequence length, C = 768, V = 50304", "    # B = batch, T = sequence length, C = 12288, V = 50304"],
  [
    "model = GPT2(vocab_size=50304, n_layer=12, n_embd=768, n_head=12, max_toks=1024, dropout=0.1).to(device)",
    "# GPT-3 175B: GPT-2's model with bigger settings (the training loop below is still the small demo)\n" +
      "model = GPT3(vocab_size=50304, n_layer=96, n_embd=12288, n_head=96, max_toks=2048, dropout=0.1).to(device)",
  ],
]);

const MAKE_LOCAL = `    def make_local(self, window):
        # GPT-3 "locally banded" layer: each token sees only itself and the previous window - 1 tokens
        band = torch.triu(torch.ones(2048, 2048), diagonal=-(window - 1))  # 1 where j >= i - (window - 1)
        self.mask = self.mask * band                                        # still causal, now also local

`;

export const attnCode = patch(gpt2AttnCode, [
  [
    '        # mask: (1, 1, 1024, 1024)\n        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))\n\n',
    '        # mask: (1, 1, 2048, 2048)\n        self.register_buffer("mask", torch.tril(torch.ones(2048, 2048)).unsqueeze(0).unsqueeze(0))\n\n' + MAKE_LOCAL,
  ],
  ["# 64 (768 // 12)", "# 128 (12288 // 96)"],
]);

export const mlpCode = patch(gpt2MlpCode, [["# C = n_embd = 768, 4C = 3072", "# C = n_embd = 12288, 4C = 49152"]]);

export { transformerBlockCode };

// Same entries as GPT-2, pointing at the GPT-3 code; a few titles, summaries and anchors change.
const CODE = new Map([
  [gpt2TrainingCode, gpt3TrainingCode],
  [gpt2AttnCode, attnCode],
  [gpt2MlpCode, mlpCode],
]);

const OVERRIDES = {
  gpt2: { title: "GPT-3 (model & training)", summary: "GPT-2's model class at GPT-3 size, with alternating dense / locally banded attention." },
  attn_class: {
    anchors: [
      { id: "class_def", label: "class", match: "class CausalSelfAttention(nn.Module):" },
      { id: "mask", label: "mask", match: 'self.register_buffer("mask", torch.tril(torch.ones(2048, 2048)).unsqueeze(0).unsqueeze(0))' },
      { id: "local_band", label: "local band", match: "    def make_local(self, window):" },
    ],
  },
  attn_masked: {
    title: "Masked Attention (dense / local, alternating)",
    summary: "Causal mask in every layer; in every other layer the mask is also cut to a local window.",
    anchors: [{ id: "local_band", label: "local band", match: "    def make_local(self, window):" }],
  },
};

export const content = Object.fromEntries(
  Object.entries(gpt2Content).map(([id, entry]) => {
    const code = CODE.get(entry.code) ?? entry.code;
    const anchors = (entry.anchors || []).filter((a) => !a.match || code.includes(a.match));
    return [id, { ...entry, code, anchors, ...OVERRIDES[id] }];
  })
);
// The model-level entry also highlights the loop that makes every other layer local.
content.gpt2.anchors = [...content.gpt2.anchors, { id: "local_layers", label: "local layers", match: "        for i, blk in enumerate(self.blocks):" }];
