import { ORIGINAL_INFO } from "../GPT2DecoderExplorer";

export const INFO = {
  ...(ORIGINAL_INFO || {}),
  mlp_fc_in_notes: {
    title: "MLP expansion: nn.Linear(n_embd, 4 * n_embd)",
    md: `
**What this layer does**

- Projects each token's hidden vector from size $n\\_embd$ up to a larger **feed-forward** size $4 * n\\_embd$.
- This is the first linear in the Transformer block's MLP (a.k.a. FFN):
  $$
  h' = \\phi(\\underbrace{W_1 h + b_1}_{\\text{this layer}}), \\quad \\
  y = W_2 h' + b_2
  $$

**Why 4X? (common design in Transformers)**

- The MLP is the main place where non-linearity and per-token mixing capacity live.
- Using a wider hidden dimension increases expressive power at modest depth.
- Empirically, 4x is a strong trade-off between quality and compute/memory for GPT-2 style models. Larger ratios (e.g., 8x) can improve quality but cost more; smaller (2x) may underfit.

**Shapes**

$$
\\boxed{\\begin{array}{l}
- \\text{Input per token}: h \\in \\mathbb{R}^{n\\_embd} \\\\[6pt]
- \\text{After this layer}: h_1 = W_1 h + b_1 \\in \\mathbb{R}^{4 \\times n\\_embd} \\\\[6pt]
- \\text{After activation (GELU): same shape} (4 \\times n\\_embd) \\\\[6pt]
- \\text{Second linear maps back}: (4 \\times n\\_embd) \\to (n\\_embd)
\\end{array}}
$$


**Parameter count (this layer only)**

$$
\\boxed{\\begin{array}{l}
- \\text{Weights}: 4 \\times n\\_embd \\times n\\_embd = 4 \\times n\\_embd^2 \\\\[6pt]
- \\text{Bias}: 4 \\times n\\_embd \\\\[6pt]
- \\text{Total}: ~4 \\times n\\_embd^2 + 4 \\times n\\_embd
\\end{array}}
$$


**Alternatives / notes**

- Some variants use ratios other than 4, gated linear units (GEGLU/SwishGLU), or different activations; the idea is similar: widen, nonlinearity, project back.
- In Pre-LN GPT-2 blocks, this comes after LayerNorm and parallel to attention in a residual structure.
    `,
  },
  mlp_fc_out_notes: {
    title: "MLP projection back: nn.Linear(4 * n_embd, n_embd)",
    md: `
**What this layer does**

- Projects features back from $4 * n\\_embd$ to $n\\_embd$ so the residual add aligns with the block input.
- FFN pattern recap:
  $$
  h' = \\phi(W_1 h + b_1), \\quad y = \\underbrace{W_2 h' + b_2}_{\\text{this layer}}
  $$

&nbsp;

**Why project back?**

- Residual connections need matching dimensions; compressing to $n\\_embd$ keeps compute stable across layers.

&nbsp;

**Shapes**

$$
\\boxed{\\begin{array}{l}
- \\text{Input}: h' \\in \\mathbb{R}^{4 \\times n\\_embd} \\\\[6pt]
- \\text{Output}: y \\in \\mathbb{R}^{n\\_embd}
\\end{array}}
$$

**Parameter count (this layer only)**

$$
\\boxed{\\begin{array}{l}
- \\text{Weights}: n\\_embd \\times (4 \\times n\\_embd) = 4 \\times n\\_embd^2 \\\\[6pt]
- \\text{Bias}: n\\_embd \\\\[6pt]
- \\text{Total}: ~4 \\times n\\_embd^2 + n\\_embd
\\end{array}}
$$

    `,
  },
};
