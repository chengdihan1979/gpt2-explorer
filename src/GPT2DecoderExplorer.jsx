import React, {
  useMemo,
  useState,
  useEffect,
  useRef,
  createContext,
  useContext,
} from "react";

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug"; // adds id=... to headings
import "katex/dist/katex.min.css";

/** =========================
 *  Info Registry (Scalable)
 *  ========================= */
const INFO = {
  dimension_change_notes: {
    title: "Dimension changes during forward",
    md: `
$$
\\text{Let } B=\\text{batch size},\\; T=\\text{sequence length},\\; C=n\\_embd,\\; V=\\text{vocab\\_size}. \\\\[6pt]
$$

$$
\\boxed{\\begin{array}{l}
\\textbf{Inputs and helpers}\\\\[6pt]
\\mathtt{idx} : (B, T) \\text{(Long tensor of token IDs)} \\\\[6pt]
\\mathtt{pos}=\\mathtt{arange}(0, T) : (T)\\\\[6pt]
\\textbf{Embeddings and sum}\\\\[6pt]
\\mathtt{tok\\_emb(idx)} : (B, T, C)\\\\[6pt]
\\mathtt{pos\\_emb(pos)} : (T, C)\\\\[6pt]
\\mathtt{pos\\_emb(pos)[None,\\ :,\\ :]} : (1, T, C) → \\text{broadcasts in the sum}\\\\[6pt]
x=\\mathtt{tok\\_emb(idx)} + \\mathtt{pos\\_emb(pos)[None,\\ :,\\ :]} : (B, T, C)\\\\[6pt]
\\textbf{Dropout and blocks}\\\\[6pt]
x=\\mathtt{drop}(x) : (B, T, C) \\text{(same shape; values randomly zeroed in train mode)}\\\\[6pt]
\\text{for } \\mathtt{blk} \\text{ in } \\mathtt{blocks:}\\; x=\\mathtt{blk}(x) : (B, T, C)\\\\[6pt]
\\textbf{Final norm and head}\\\\[6pt]
x=\\mathtt{ln\\_f}(x) : (B, T, C)\\\\[6pt]
\\mathtt{logits}=\\mathtt{lm\\_head}(x) : (B, T, V)\\\\[6pt]
\\textbf{Targets and loss (if provided)}\\\\[6pt]
\\mathtt{targets} : (B, T)\\\\[6pt]
\\mathtt{logits.view(-1,\\ logits.size(-1))} : (B\\!\\cdot\\!T, V)\\\\[6pt]
\\mathtt{targets.view(-1)} : (B\\!\\cdot\\!T)\\\\[6pt]
\\mathtt{loss}=\\mathtt{F.cross\\_entropy}((B\\!\\cdot\\!T, V),\\ (B\\!\\cdot\\!T)) : \\text{scalar () (a single number)}\\\\[6pt]
\\textbf{Return} :\\ \\mathtt{logits} \\ (B, T, V),\\ \\mathtt{loss}\\ \\text{(scalar or None)}\\\\[6pt]
\\end{array}}
$$

    `,
  },

  lm_head_notes: {
    title: "lm_head",
    md: `
$$
\\boxed{\\begin{array}{l}
\\textbf{What it is:}\\ \\text{a linear projection that turns each hidden vector (size } n\\_embd \\text{) into a vector of} \\\\
\\text{vocabulary logits (size } \\texttt{vocab\\_size}\\text{).} \\\\[8pt]
\\textbf{Shape-wise:}\\ \\text{if hidden states } x \\in \\mathbb{R}^{B \\times T \\times n\\_embd},\\ \\text{ then} \\\\[4pt]
\\mathtt{logits = lm\\_head(x)}\\ \\ \\#\\ \\mathbb{R}^{B \\times T \\times \\texttt{vocab\\_size}} \\\\[8pt]
\\text{Each row of the weight matrix is a learned “prototype” for one token; the logit for token } v \\text{ is the} \\\\
\\text{dot product between the hidden state and that token’s embedding.} \\\\[12pt]
\\textbf{Why no bias (}\\mathtt{bias=False}\\textbf{):} \\\\[6pt]
\\quad 1.\\ \\textbf{Weight tying.}\\ \\text{GPT-2 typically ties the output weights to the input embedding table:} \\\\[4pt]
\\quad \\ \\ \\ \\mathtt{self.lm\\_head.weight = self.tok\\_emb.weight}\\ \\ \\#\\ \\text{same parameters} \\\\[4pt]
\\quad \\ \\ \\ \\text{This saves parameters and often improves perplexity. The tied matrix matches in shape,} \\\\
\\quad \\ \\ \\ \\text{but a separate output bias isn’t needed (and is commonly omitted).} \\\\[6pt]
\\quad 2.\\ \\textbf{Little benefit, lots of params.}\\ \\text{A bias would add } \\texttt{vocab\\_size} \\text{ more parameters;} \\\\
\\quad \\ \\ \\ \\text{in practice it yields negligible gains here, so many implementations drop it.}
\\end{array}}
$$

    `,
  },


  linear: {
    title: "Linear",
    md: `
**torch.nn.Linear** — fully-connected (affine) layer

Given input $x \\in \\mathbb{R}^{*,\\,\\text{in}}$,
$$
y = x W^\\top + b
$$
where $W \\in \\mathbb{R}^{\\text{out}\\times\\text{in}},\\ b \\in \\mathbb{R}^{\\text{out}}$.

- Maps \`in_features -> out_features\`.
- Supports leading-dim broadcasting: \`(N, ..., in) -> (N, ..., out)\`.
- Docs: [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).
`,
  },
  gelu: {
    title: "GELU",
    md: `
**GELU (Gaussian Error Linear Unit)**

$$
\\mathrm{GELU}(x) = x \\, \\Phi(x)
$$

The GELU comes from the paper [Hendrycks & Gimpel (2016)](https://arxiv.org/abs/1606.08415)  

where $\\Phi(x)$ is the [CDF of the standard normal distribution](#cdf).

**Tanh approximation (used in GPT-2):**

$$
\\mathrm{GELU}(x) \\approx 0.5\\,x\\,\\left(1 + \\tanh\\!\\left(\\sqrt{\\tfrac{2}{\\pi}}\\,(x + 0.044715\\,x^3)\\right)\\right)
$$

The reason behind the approximation comes from the [github issue](https://github.com/pytorch/pytorch/issues/39853).

Historically, evaluating erf (needed by exact GELU) was slower than a few multiplies + a tanh, especially on older GPU kernels and before [common fusions](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/). So many models (including GPT-2) used the Hendrycks-Gimpel tanh approximation for speed.

**TensorFlow today:** $tf.nn.gelu(x, approximate=False)$ computes the exact erf form; $approximate=True$ uses the tanh form. With modern TF/XLA, both paths are vectorized and often fused; the exact version is usually fast enough unless you're ultra latency-sensitive

**PyTorch today:** $torch.nn.GELU() / F.gelu$ support $approximate='none'$ (exact) or $approximate='tanh'$. Same idea: exact is generally fine; tanh can be a tiny bit faster on some hardware. PyTorch also has “FastGELU” in some transformer libs that's just the tanh form.

**Why does GPT-2 use GeLU instead of ReLU?**

ReLU definition:
$$
ReLU(x)=max(0,x).
$$
So the gradient behavior of ReLU is:

For $x>0$: gradient = 1 (flows without scaling).

For $x<0$: gradient = 0 (dies out).

This makes optimization straightforward and avoids vanishing gradients that plagued sigmoid/tanh in early days of deep learning if nets are deep.
But it also introduces the “dying ReLU” problem: if a neuron's weights push it into negative inputs too often, its gradient is always zero → it can “die” and never update again.
GELU doesn't discard negatives outright. It smoothly scales them down using the normal CDF.

`,
  },
  cdf: {
    title: "CDF",
    md: `
**CDF of the standard normal distribution** $(\\mu = 0, \\sigma = 1)$

The CDF is

$$
\\Phi(x) = \\frac{1}{\\sqrt{2\\pi}} \\int_{-\\infty}^{x} e^{-t^{2}/2} \\, dt
$$

with properties: $\\Phi(0)=0.5$, it is monotonically increasing, and its derivative is the standard normal PDF $\\phi(x)$.
`,
  },

  layernorm: {
    title: "LayerNorm",
    md: `
**nn.LayerNorm** — applies layer normalization over the last dimension of the input (here, the embedding dimension $n\\_embd$).

Mathematically, given any input vector $h \\in \\mathbb{R}^{d}$, LayerNorm computes
$$
\\text{LN}(h) = \\gamma \\odot \\frac{h - \\mu(h)}{\\sigma(h) + \\epsilon} + \\beta
$$
where $\\mu,\\sigma$ are the mean and std over the last (feature) dimension, and $\\gamma,\\beta$ are learned scale/shift.

&nbsp;

Here is sometimes called the “final layer norm”, which applies to each token individually, across its embedding dimensions before the final projection layer (lm_head).

&nbsp;

**Example**

&nbsp;

For a token embedding vector

$$
x = [5.0,\\; 4.5,\\; 6.2,\\; 4.8],
$$

LayerNorm computes the mean and standard deviation over the 4 dimensions:

$$
\\mu = 5.125, \\quad \\sigma \\approx 0.64
$$

and normalizes each element:

$$
\\hat{x}_i = \\frac{x_i - \\mu}{\\sigma + \\epsilon}.
$$

So the normalized output becomes approximately

$$
\\hat{x} = [-0.20,\\; -0.98,\\; 1.68,\\; -0.49].
$$

&nbsp;

---

&nbsp;

**Why It’s Used in GPT-Style Models**

&nbsp;

1. **Stabilizes training**  
   Prevents exploding/vanishing activations by keeping values centered and scaled.  
   For each token embedding $x \\in \\mathbb{R}^D$:

   $$
   \\text{LayerNorm}(x) = \\frac{x - \\mu}{\\sigma + \\epsilon} \\cdot \\gamma + \\beta
   $$

   where $\\mu$ is the mean of $x$, $\\sigma$ its standard deviation, and $\\gamma, \\beta$ are learnable parameters.

&nbsp;

2. **Smooth gradients**  
   Since each token embedding is normalized:

   $$
   \\mathbb{E}[x] \\approx 0, \\quad \\text{Var}(x) \\approx 1
   $$

   This keeps gradients well-conditioned and avoids scale mismatches across dimensions.

&nbsp;

3. **Consistent scale across layers**  
   Regardless of depth or input distribution, the hidden states entering the next layer have a stable range:

   $$
   x_{\\text{normalized}} \\in \\mathbb{R}^D \\quad \\text{with mean} \\; 0 \\; \\text{and std} \\; 1.
   $$

   This prevents certain dimensions from dominating and helps training convergence.


- Docs: [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
`,
  },

  embedding_tok: {
    title: "Token Embedding",
    md: `
  **Maps token IDs to dense vectors.**
  
  &nbsp;

  **Weight matrix:**
  $$
  W \\in \\mathbb{R}^{\\text{vocab\\_size} \\times \\text{n\\_embd}}
  $$
  **Input IDs:**
  $$
  \\text{input\\_ids} \\in \\mathbb{Z}^{B \\times T}
  $$

  **Embedding lookup:**
  $$
  \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
  $$

  **Parameter count:**
  $$
  \\text{params} = \\text{vocab\\_size} \\times n\\_embd
  $$
  (e.g. GPT-2 small: $50{,}257 \\times 768 \\approx 3.86 \\times 10^7$)

  &nbsp;
  
  ---

  &nbsp;

  **Training**

  &nbsp;

  - $W$ is learned via backprop.
  - Only rows for tokens in the batch get updated.
  - GPT-2 often ties $W$ with the output softmax weights. ([What is weight tying?](#info:weight_tying_notes))

  &nbsp;

  ---

  &nbsp;

  **Why use embedding? Not one hot encoding.**  

  &nbsp;

  A one-hot token vector would be length $vocab\\_size$ and extremely sparse. $nn.Embedding$ is equivalent to multiplying a one-hot vector by $W$, but it:

  - avoids creating huge sparse vectors,

  - learns dense, semantic representations where related tokens end up close in the embedding space.

    `,
  },
  
  embedding_pos: {
    title: "Positional Embedding",
    md: `
  Transformers need order information because self-attention has no sense of sequence.
  Positional embeddings give each token a unique vector based on its position.
  
  Weight matrix:
  $$
  P \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
  $$
  
  - Here, \`max_toks\` is the maximum sequence length (e.g. 1024 for GPT-2).
  - Each row $P_j$ corresponds to position $j$ in the sequence.
  
  &nbsp;
  
  ---
  
  &nbsp;
  
  **Runtime usage**
  
  &nbsp;
  
  1. Token IDs → token embeddings:  
  $$
  \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
  $$
  
  2. Positions $0,1,\\dots,T-1$ → positional embeddings:  
  $$
  \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
  $$
  
  3. Add them together via broadcasting: ([What is broadcasting?](#info:broadcasting)) 
  $$
  x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
  $$
  
  ---
  
  &nbsp;
  
  **Parameters**
  
  $$
  \\text{params} = \\text{max\\_toks} \\times n\\_embd
  $$
  
  For GPT-2 small:  
  $$
  1024 \\times 768 \\approx 7.86 \\times 10^5
  $$
  
  Much smaller than the token embedding table (~38M params).
  
  &nbsp;
  
  ---
  
  &nbsp;
  
  **Shapes at a glance**
  
  &nbsp;

  - Weight: $[\\text{max\\_toks}, n\\_embd]$  
  - Positions: $[T]$  
  - Pos embeddings: $[T, n\\_embd]$  
  - Final sum: $[B, T, n\\_embd]$  
    `,
  },


  weight_tying_notes: {
    title: "Weight Tying",
    md: `
  **Weight tying (shared input/output embeddings)**
  
  &nbsp;

  The idea comes from the paper [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
    
  Weight tying sets the output softmax weights equal to the input token embedding weights:
  
  &nbsp;

  \`\`\`python
  # tie lm_head weight to tok_emb
  self.lm_head.weight = self.tok_emb.weight
  \`\`\`
  
  &nbsp;
  
  **Why do this?**
  
  &nbsp;
  
  - Fewer parameters (≈ halves \`V*d\` + \`d*V\` to just \`V*d\`).
  - Acts as a helpful regularizer and often improves perplexity.
  
  &nbsp;
  
  **Notes:**

  &nbsp;

  - Many GPT-2 implementations use \`bias=False\` on \`lm_head\` when tying.
  - Tying requires identical shapes: \`lm_head.weight.shape == tok_emb.weight.shape == (vocab_size, n_embd)\`.
  `,
  },

  broadcasting: {
    title: "PyTorch Broadcasting",
    md: `
  Broadcasting is the set of rules PyTorch (and NumPy) uses to perform operations 
  on tensors of different shapes without explicitly copying data.

  ---

  **Basic rule**

  Two tensors are compatible if, for each dimension (from the end):

  - they are equal, or  
  - one of them is $1$.

  The smaller tensor is "stretched" (virtually repeated) to match the larger shape.

  ---

  **Example 1: Vector + Matrix**

  $$
  A \\in \\mathbb{R}^{3 \\times 4}, \\quad b \\in \\mathbb{R}^{1 \\times 4}
  $$

  Operation:
  $$
  C = A + b
  $$

  Here, $b$ is broadcast along the first dimension, so:
  $$
  C \\in \\mathbb{R}^{3 \\times 4}
  $$

  ---

  **Example 2: Column vector + Row vector**

  $$
  u \\in \\mathbb{R}^{3 \\times 1}, \\quad v \\in \\mathbb{R}^{1 \\times 4}
  $$

  Operation:
  $$
  M = u + v
  $$

  Shapes expand to $(3,4)$:
  $$
  M \\in \\mathbb{R}^{3 \\times 4}
  $$

  ---

  **Formal rule**

  If we have shapes:
  $$
  (s_1, s_2, \\dots, s_k), \\quad (t_1, t_2, \\dots, t_k)
  $$

  then they are broadcast-compatible if
  $$
  s_i = t_i \\quad \\text{or} \\quad s_i = 1 \\quad \\text{or} \\quad t_i = 1
  $$
  for all $i$ (padding shorter shapes with $1$ on the left if needed).

  ---

  **Key benefits**

  - Saves memory (no explicit tiling).
  - Makes code concise.
  - Heavily used in deep learning for adding biases, positional embeddings, etc.
  `,
  },

  mlp_dropout_notes: {
    title: "MLP Dropout",
    md: `
**Remainder: What Happens in the MLP Step**

Inside a Transformer block, the feed-forward MLP is:

$$
\\text{MLP}(x) = W_2 \\cdot \\phi(W_1 x + b_1) + b_2
$$

where:

- $W_1, W_2$ are weight matrices  
- $b_1, b_2$ are bias vectors  
- $\\phi$ is a non-linear activation function (e.g. GELU in GPT-2)

This expands the hidden dimension with $W_1$, applies the non-linearity $\\phi$, and then projects back down with $W_2$.

&nbsp;

---

&nbsp;

**Why Dropout Is Inserted After MLP**

1. **Regularization**  
   The MLP produces high-dimensional features after the nonlinearity.  
   Dropout prevents overfitting by randomly zeroing activations:

   $$
   h' = h \\odot m \\cdot \\frac{1}{1-p}
   $$

   where $h$ is the activation, $m$ is a Bernoulli mask, and $p$ is the dropout probability.

&nbsp;

2. **Preventing co-adaptation**  
   Without dropout, neurons may rely on specific others.  
   Dropout forces the network to learn redundant, robust representations.

&nbsp;

3. **Stabilizing optimization**  
   Adding stochastic noise to activations prevents weights from growing too aggressively.  
   This is important since the feed-forward MLP usually has more parameters than the attention layer.

&nbsp;

---

&nbsp;

**Example**

If the MLP output for one token is

$$
h = [0.8,\\; -0.5,\\; 1.2,\\; 0.0,\\; -0.3,\\; 0.9],
$$

Suppose the 3rd dimension (1.2) is consistently high for certain tokens.

The next projection layer $W_2$ could just learn to “look at” the 3rd neuron and ignore the rest.

In other words, the model develops a shortcut: “dimension 3 tells me what I need.”

&nbsp;

**How Dropout Disrupts This Shortcut**

With dropout ($p = 0.5$), a random mask $m$ is applied:

$$
m = [1,\\; 0,\\; 0,\\; 1,\\; 1,\\; 1],
$$

and the output becomes

$$
h' = h \\odot m \\cdot \\frac{1}{1-p}
   = [1.6,\\; 0.0,\\; 0.0,\\; 0.0,\\; -0.6,\\; 1.8].
$$

Here the third neuron has been dropped.  
As a result, $W_2$ cannot always depend on $h_3$ being active,  
so it must learn to combine information from multiple dimensions.  
This encourages redundancy and more robust feature representations.


    `,
  },

  emb_dropout_notes: {
    title: "Embedding Dropout",
    md: `
  After the sum of token + positional embeddings, GPT-2 applies a dropout layer. This is done before feeding the representation into the first Transformer block.

  &nbsp;

  **Why Dropout Here?**

  &nbsp;
  
  Prevents overfitting by randomly zeroing out parts of the embedding vectors during training.

  Forces the model not to rely too heavily on specific embedding dimensions.

  Provides regularization right at the input stage, complementing dropout inside Transformer blocks (e.g., after attention and feed-forward layers).

  &nbsp;

  ---

  &nbsp;
  
  **Example**

  &nbsp;
  
  Suppose after summing token embeddings and positional embeddings, one token has this 6-dimensional vector:

  $$
  h = [0.8, -0.5, 1.2, 0.0, -0.3, 0.9]
  $$

  This is the raw input to the first Transformer block.

  &nbsp;

  **Apply Dropout (say $p=0.5$)**

  &nbsp;
  
  Dropout randomly “masks” some dimensions with probability $p$. The surviving dimensions are rescaled by $1/(1-p)$ to keep expectation consistent.

  Dropout mask example:

  $$
  m = [1, 0, 1, 0, 1, 1]
  $$

  Apply mask and scale (scale factor = 2 because $1/(1-0.5)=2$):

  $$
  h' = h \\odot m \\times 2 = [1.6, 0.0, 2.4, 0.0, -0.6, 1.8]
  $$

  So the second and fourth dimensions are “dropped” (zeroed), while the rest are doubled.

  &nbsp;

  ---

  &nbsp;

  **Why This Helps**

  &nbsp;
  
  **Increase generalization and Prevents overfitting**: Helps model learn more robust patterns by not “wiring” itself too closely to specific dimensions. If the model learned to rely too heavily on, say, the 3rd dimension, dropout forces it to also use other dimensions, because the 3rd dimension might vanish in training.

  &nbsp;
  
  **Regularizes embeddings**: Token embeddings won't encode fragile, single-dimension “shortcuts” — they need to spread useful signals more evenly.

`,
},

  cross_entropy_notes: {
    title: "Cross Entropy",
    md: `
Cross-entropy loss is a measure of how well a model predicts the target distribution. This computes the negative log-likelihood of the correct next token at every position.

Docs: [CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

&nbsp;

---

&nbsp;

**Shape of Logits and Targets in Language Models**

In GPT-style training:

- **Logits** (raw predictions before softmax):  

$$
(B,T,V)
$$

where:  
  - $B$: batch size  
  - $T$: sequence length  
  - $V$: vocabulary size

- **Targets** (ground truth token indices):  

$$
(B,T)
$$

Example:
For $B=2$, $T=4$, $V=10$:

$$
Logits → shape = (2, 4, 10)
$$

$$
Targets → shape = (2, 4)
$$

---

&nbsp;

**What CrossEntropy Expects**

\`torch.nn.functional.cross_entropy\` expects:

**Input**: 2D tensor of shape $(N, C)$

where $N$ = number of samples, $C$ = number of classes.

**Target**: 1D tensor of shape $(N,)$ with integer class indices.

So we need to flatten both logits and targets.

&nbsp;

---

&nbsp;

**Why .view(-1, logits.size(-1))?**

\`logits.view(-1, V)\` reshapes $(B, T, V) \\to (B \\cdot T, V)$.

→ Each row is one token prediction across the vocabulary.

\`targets.view(-1)\` reshapes $(B, T) \\to (B \\cdot T,)$.

→ Each entry is the correct class index for one token.

Thus, every token in the batch is treated as an independent classification example.

&nbsp;

---

&nbsp;

**Putting It Together**

\`\`\`python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # shape (B*T, V)
    targets.view(-1)                    # shape (B*T,)
)
\`\`\`
  

This computes the **average negative log-likelihood** across all tokens in the batch.


    `,
  },

  key_view_notes: {
    title: "Key View",
    md: `
**Key projection**

$self.key(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

**Split channels into heads**

$.view(B, T, n_{\\text{head}}, h_s)$ — splits the last dim $C$ into $n_{\\text{head}}\\times h_s$ with
$h_s = \\left\\lfloor \\dfrac{C}{n_{\\text{head}}} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n_{\\text{head}}, h_s).
$$

**Move heads dimension next to batch**

$.transpose(1, 2)$ — swaps the $T$ and $n_{\\text{head}}$ axes:

$$
(B, T, n_{\\text{head}}, h_s) \\;\\rightarrow\\; (B, n_{\\text{head}}, T, h_s).
$$
    `,
  },

  query_view_notes: {
    title: "Query View",
    md: `
**Query projection**

$self.query(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

**Split channels into heads**

$.view(B, T, n_{\\text{head}}, h_s)$ — splits the last dim $C$ into $n_{\\text{head}}\\times h_s$ with
$h_s = \\left\\lfloor \\dfrac{C}{n_{\\text{head}}} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n_{\\text{head}}, h_s).
$$

**Move heads dimension next to batch**

$.transpose(1, 2)$ — swaps the $T$ and $n_{\\text{head}}$ axes:

$$
(B, T, n_{\\text{head}}, h_s) \\;\\rightarrow\\; (B, n_{\\text{head}}, T, h_s).
$$
    `,
  },

  value_view_notes: {
    title: "Value View",
    md: `
**Value projection**

$self.value(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

**Split channels into heads**

$.view(B, T, n_{\\text{head}}, h_s)$ — splits the last dim $C$ into $n_{\\text{head}}\\times h_s$ with
$h_s = \\left\\lfloor \\dfrac{C}{n_{\\text{head}}} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n_{\\text{head}}, h_s).
$$

**Move heads dimension next to batch**

$.transpose(1, 2)$ — swaps the $T$ and $n_{\\text{head}}$ axes:

$$
(B, T, n_{\\text{head}}, h_s) \\;\\rightarrow\\; (B, n_{\\text{head}}, T, h_s).
$$
    `,
  },

  q_k_product_notes: {
    title: "Q @ K Product",
    md: `

**Explanation of Scale dot-product attention**

&nbsp;

q, k have shape $(B, n\\_head, T, hs)$ (queries/keys per head).

&nbsp;

$k.\\mathrm{transpose}(-2, -1) \\;\\rightarrow\\; (B, n\\_head, hs, T)$ swap the last two axes of keys, so each query can dot with every key at all time steps.

&nbsp;

\\@ is batched matrix multiply:

$(B, n\\_head, T, hs)\\ @\\ (B, n\\_head, hs, T) \\;\\rightarrow\\; (B, n\\_head, T, T)$

&nbsp;

Result: for each head, a $T \\times T$ matrix of raw attention scores (one row per query position, one column per key position).

&nbsp;

\\* (1.0 / hs**0.5) multiplies by $1/\\sqrt{hs}$ (with $hs$ = head size). This is the scale from “Scaled Dot-Product Attention.”

&nbsp;


**Scaled dot-product attention logits in math notation**

&nbsp;

Given
$$
q, k \\in \\mathbb{R}^{B \\times n\\_head \\times T \\times h_s},
$$
the attention logits are computed as
$$
\\mathrm{att} \\;=\\; \\bigl(q \\; k^\\top\\bigr)\\;\\frac{1}{\\sqrt{h_s}},
$$
where the transpose is over the last two axes of $k$:
$$
k^\\top \\in \\mathbb{R}^{B \\times n\\_head \\times h_s \\times T},
\\qquad
q\\,k^\\top \\in \\mathbb{R}^{B \\times n\\_head \\times T \\times T}.
$$

Elementwise, for batch $b$, head $h$, query position $t_q$, and key position $t_k$:
$$
\\mathrm{att}_{b,h,t_q,t_k}
\\;=\\;
\\frac{ q_{b,h,t_q,:}\\cdot k_{b,h,t_k,:} }{\\sqrt{h_s}}.
$$

**Why the $1/\\sqrt{h_s}$ scale**

&nbsp;

If the components of $q$ and $k$ are roughly zero-mean, unit-variance, then
$$
\\operatorname{Var}\\!\\bigl(q_{b,h,t_q,:}\\cdot k_{b,h,t_k,:}\\bigr) \\;\\propto\\; h_s.
$$
Dividing by $\\sqrt{h_s}$ keeps the logits’ variance approximately constant (≈1), which helps prevent softmax becomes overly peaky (saturation) and stabilizes gradients.

After this step, a causal/padding mask is applied and then a softmax over the last dimension produces attention probabilities.

    `,
  },

  torch_ones_notes: {
    title: "torch.ones",
    md: `

$\\text{Creates a tensor filled with 1s of the given shape.}$

$\\mathtt{torch.ones(*sizes,\\ dtype=None,\\ device=None,\\ requires\\_grad=False)}$

$\\text{Examples: }\\; \\mathtt{torch.ones(2,3)},\\ \\mathtt{torch.ones\\_like(x)}.$

&nbsp;

Docs: [torch.ones](https://docs.pytorch.org/docs/stable/generated/torch.ones.html)
    `,
  },

  torch_tril_notes: {
    title: "torch.tril",
    md: `
$\\text{Returns the lower-triangular part of a matrix; zeros out elements above the specified diagonal.}$

$$
\\mathtt{torch.tril(input,\\ diagonal=0)}
$$

$\\text{If }\\ diagonal=0:\\ \\text{keep main diagonal and below};\\quad$

$\\text{if }\\ diagonal>0:\\ \\text{a positive value includes just as many diagonals above the main diagonal};\\quad$

$\\text{if }\\ diagonal<0:\\ \\text{a negative value excludes just as many diagonals below the main diagonal}$

&nbsp;

Docs: [torch.tril](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)
    `,
  },

  unsqueeze_notes: {
    title: "unsqueeze",
    md: `
$\\text{Inserts a dimension of size }1\\text{ at position }dim\\text{ in the tensor's shape (no data copy).}$

$$
\\mathtt{torch.unsqueeze(input,\\ dim)}\\quad\\text{or}\\quad\\mathtt{tensor.unsqueeze(dim)}
$$

$$
\\text{Shape examples:}\\quad
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(0)}\\ }\\ (1,\\ B,\\ T),\\qquad
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(-1)}\\ }\\ (B,\\ T,\\ 1).
$$

$$
\\text{Valid }dim\\in[-(n{+}1),\\ n+1]\\ \\text{for an }n\\text{-D input; negative }dim\\text{ counts from the end.}
$$

&nbsp;

Docs: [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)

    `,
  },

  causal_mask_notes: {
    title: "Causal Mask",
    md: `
That line applies the **causal (look-back only) mask** to the attention logits so each token can attend only to itself and earlier tokens.

$$
\\mathtt{att = att.masked\\_fill(self.mask[:, :, :T, :T] == 0,\\ float('-inf'))}
$$

$$
\\text{att: } (B,\\ n\\_head,\\ T,\\ T)
\\qquad
\\text{self.mask: } (1,\\ 1,\\ 1024,\\ 1024)\\ \\text{lower-triangular 1s, else 0s}
$$

&nbsp;

---

&nbsp;

**What's happening step by step**

&nbsp;

1\\. **What $\\mathtt{self.mask}$ is**  
   &nbsp;

   Earlier we had:
   $$
   \\mathtt{self.register\\_buffer("mask",\\ torch.tril(torch.ones(1024,\\ 1024)).unsqueeze(0).unsqueeze(0))}
   $$
   [torch.tril](#info:torch_tril_notes)([torch.ones](#info:torch_ones_notes)(1024,1024)) makes a **lower-triangular** matrix of size $1024 \\times 1024$ with $1$s on/under the main diagonal and $0$s above it (future positions).  
   [unsqueeze(0)](#info:unsqueeze_notes).unsqueeze(0) turns it into shape $(1,\\ 1,\\ 1024,\\ 1024)$ so it can **broadcast** across batch and heads.  
   Stored as a **buffer** so it moves with the module to GPU/CPU and is saved, but not trained.

   &nbsp;

2\\. **Slicing to current length**  
  &nbsp;

   $$
   \\mathtt{self.mask[:, :, :T, :T]} \\;\\to\\; (1,\\ 1,\\ T,\\ T)
   $$
   matching the current sequence length.

   &nbsp;

3\\. **Build a boolean mask of disallowed links**  
  &nbsp;

   $$
   \\mathtt{==\\ 0}
   $$
   converts the \\(0/1\\) matrix into a boolean mask: **True** where the entry is $0$ (i.e., future positions), **False** otherwise.

   &nbsp;

4\\. **$\\mathtt{masked\\_fill}$ on the logits**  
  &nbsp;

   $\\text{att}$ has shape $(B,\\ n\\_head,\\ T,\\ T)$ and holds **raw attention logits** $\\bigl(q\\,k^\\top/\\sqrt{h_s}\\bigr)$.  
   $\\mathtt{.masked\\_fill(bool\\_mask,\\ -\\infty)}$ writes $-\\infty$ into $\\text{att}$ **where the mask is True** (i.e., where attending would look **ahead**).  
   Broadcasting lets a $(1,\\ 1,\\ T,\\ T)$ mask apply to all batches and heads.

   &nbsp;

5\\. **Why $-inf$?**  
  &nbsp;

   Next you do $\\mathtt{att = F.softmax(att,\\ dim=-1)}$.  
   $\\exp(-\\infty) = 0$,
   so masked positions get **exactly zero probability**, and the softmax **renormalizes only over allowed (past/self) positions**.  
   If you zeroed *after* softmax, probabilities would still be influenced by future tokens during normalization. **Masking before softmax** fixes that.

   &nbsp;

   ---

&nbsp;

**Intuition (per token row)**  
&nbsp;

For query position $t$, this sets
$$
\\text{att}_{t,j} = -\\infty \\quad \\text{for } j > t,
$$
so after softmax, the row can only distribute probability over keys $j \\le t$ (itself and earlier tokens). That’s exactly **causal self-attention**.

&nbsp;

**Notes**
- This mask enforces **causality**; it does **not** handle padding on its own. If you have padded tokens, you typically *also* add a padding mask (another $\\mathtt{masked\\_fill}$) so padding positions get $-\\infty$ too.
- Using a big negative constant (e.g. $-10^9$) is common; $-\\infty$ is cleaner and numerically robust in PyTorch softmax.

    `,
  },

  register_buffer_notes: {
    title: "register_buffer",
    md: `
$\\text{Registers a tensor as a \\emph{buffer} on a Module: it moves with the module (}\\mathtt{to/cuda/cpu}\\text{),}$
$\\text{is saved/restored with }\\mathtt{state\\_dict}\\text{ (by default), and is \\emph{not} a learnable parameter.}$

$$
\\mathtt{Module.register\\_buffer(name,\\ tensor,\\ persistent = True)}
$$

$\\text{Not a Parameter: excluded from }\\mathtt{parameters()}\\text{ and optimizer updates.}$

$\\text{Persistence: included in }\\mathtt{state\\_dict}\\text{ if }\\mathtt{persistent = True;}\\ \\text{set }\\mathtt{persistent = False}\\text{ to exclude.}$

$\\text{Device/Dtype: buffer follows }\\mathtt{module.to()/cuda()/cpu()}\\text{ just like parameters.}$

$\\text{Use cases: masks (e.g., causal masks), running statistics, indices, constants tied to the model.}$

$$
\\text{Example: }\\quad
\\mathtt{self.register\\_buffer("mask",\\ torch.tril(torch.ones(1024,\\ 1024)).unsqueeze(0).unsqueeze(0))}
$$

Docs: [register_buffer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

    `,
  },

  softmax_notes: {
    title: "softmax",
    md: `

**F.softmax explanation**

&nbsp;

1\\. $\\text{Converts the raw attention scores in }\\;\\mathtt{att}\\;\\text{ into probabilities along the last dimension (keys).}$

&nbsp;

2\\. $\\text{For each query token (and each head), the resulting row tells how much attention that query token should pay to}$
$\\text{each allowed previous token — higher value means more attention, zero means ignore. (e.g., masked future/padding)}$

&nbsp;

3\\. $\\text{Each row sums to 1, so the weights are comparable and interpretable.}$

&nbsp;

4\\. $\\text{The output has the same shape as the input; only the values are rescaled to valid probabilities.}$

&nbsp;

5\\. $\\text{These probabilities are then used to take a weighted average of the value vectors for that query token.}$

&nbsp;

---

&nbsp;

**Math Intepretation**

&nbsp;

$\\text{Applies softmax along the last dimension of }\\;\\mathtt{att}\\;\\text{to turn logits into probabilities.}$

$$
\\text{Shape: }\\; att \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}
\\quad\\xrightarrow{\\ \\mathtt{softmax(\\cdot,\\ dim=-1)}\\ }\\quad
P \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}.
$$

$$
\\text{For fixed }(b,h,t_q):\\quad
P_{b,h,t_q,j}
\\;=\\;
\\frac{\\exp\\big(att_{b,h,t_q,j}\\big)}
{\\sum_{k=1}^{T}\\exp\\big(att_{b,h,t_q,k}\\big)},
\\qquad
\\sum_{j=1}^{T} P_{b,h,t_q,j} = 1.
$$

$$
\\text{Effect of the causal mask: if }\\ att_{b,h,t_q,j}=-\\infty\\ \\Rightarrow\\ P_{b,h,t_q,j}=0
\\;\\text{ (future positions get zero probability).}
$$

$$
\\text{Notes: }\\;
\\mathtt{dim=-1}\\;\\text{means “over the last axis” (the keys dimension).}\\; 
\\mathtt{F.softmax}\\;\\text{is numerically stable (effectively subtracts a row max).}
$$

    `,
  },

  att_v_product_notes: {
    title: "att @ v",
    md: `

**Explanation and Intuition**

&nbsp;

1\\. $\\mathtt{att}\\ \\text{holds, for each token (and each head), how much attention to pay to every allowed token in the sequence.}$

&nbsp;

2\\. $\\mathtt{v}\\ \\text{holds the information content for each token (the pieces worth copying).}$

&nbsp;

$\\text{Intuition: the operation says for each token, mix together other tokens' information according to those attention weights.}$

&nbsp;

$\\text{Tokens with higher weights contribute more; tokens with zero weight do not contribute.}$

&nbsp;

$\\text{The result }\\mathtt{y}\\text{ is a context-aware summary for each token (per head) that pulls in the most relevant information.}$

&nbsp;

---

&nbsp;

**Math Intepretation**

$$
\\mathtt{y\\ =\\ att\\ @\\ v}
$$

$$
att \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times T},\\quad
v \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times hs},\\quad
y \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times hs}.
$$

&nbsp;

$$
\\text{For a fixed batch } b \\text{ and head } h:\\quad
y^{(b,h)} \\,=\\, att^{(b,h)}\\, v^{(b,h)},\\quad
att^{(b,h)}\\in\\mathbb{R}^{T\\times T},\\;
v^{(b,h)}\\in\\mathbb{R}^{T\\times hs},\\;
y^{(b,h)}\\in\\mathbb{R}^{T\\times hs}.
$$

$$
\\text{Elementwise (per query position } t\\text{):}\\quad
y_{b,h,t,:}
\\;=\\;
\\sum_{j=1}^{T} att_{b,h,t,j}\\; v_{b,h,j,:}
$$

$$
\\boxed{\\begin{array}{l}
\\textbf{Attention row:}\\ \\text{each element is the probability of attending to a specific key position} \\\\[6pt]
\\text{(i.e., a specific token in the allowed context—self and previous tokens in causal attention). Masked tokens (future/padding) have probability 0.)}\\\\[6pt]
\\textbf{Value vectors:}\\ \\text{there is one value vector per source token (same positions as the keys). These vectors carry the content to be mixed.}\\\\[6pt]
\\textbf{Result:}\\ \\text{for a given query token } t,\\ \\text{the output is a weighted mix of those value vectors,} \\\\[6pt]
\\text{using the attention row as weights—tokens with higher probability contribute more, zero means no contribution.}
\\end{array}}
$$

&nbsp;

$$
\\boxed{\\begin{array}{l}
\\textbf{Concrete mini-example:}\\\\[6pt]
\\text{Say the attention row for token } t \\text{ is: } [0.5,\\ 0.3,\\ 0.2].\\\\[6pt]
\\text{The value vectors at positions } 1,\\ 2,\\ 3 \\text{ are } v_1,\\ v_2,\\ v_3\\; (\\text{each is a length } h\\_s \\text{ vector}).\\\\[6pt]
\\text{Then the output vector for token } t \\text{ is: } 0.5\\,v_1 + 0.3\\,v_2 + 0.2\\,v_3\\; \\text{— a single vector (length } h\\_s \\text{) that mixes information using that row's weights.}
\\end{array}}
$$

&nbsp;

$$
\\text{Causality: }\\; att_{b,h,t,j}=0 \\;\\text{for } j>t\\;
\\;\\Rightarrow\\;
\\text{the sum only ranges over allowed (past/self) positions.}
$$

$$
\\text{Since } att_{b,h,t,j}\\ge 0 \\;\\text{and}\\; \\sum_{j} att_{b,h,t,j}=1
\\;\\Rightarrow\\;
y_{b,h,t,:}\\;\\text{is a convex combination of the } v_{b,h,j,:}
$$

    `,
  },

  reorder_merge_heads_notes: {
    title: "reorder_merge_heads",
    md: `

$\\textbf{Explanation and Intuition}$

$$
\\mathtt{y.transpose(1,\\ 2).contiguous().view(B,\\ T,\\ C)}
$$

$$
\\boxed{\\begin{array}{l}
\\textbf{transpose(1,\\ 2)}: \\text{you currently have per-head outputs arranged as } (B,\\ n\\_head,\\ T,\\ h\\_s).\\\\[6pt]
\\text{ This swap moves }\\textbf{time}\\text{ next to }\\textbf{batch}\\text{ → } (B,\\ T,\\ n\\_head,\\ h\\_s),\\ \\text{ so for each token you can gather all heads together.}\\\\[12pt]
\\textbf{contiguous()}: \\text{after a transpose, the tensor's memory is strided (non-contiguous). } \\mathtt{view}\\ \\text{ needs a contiguous layout,}\\\\[6pt]
\\text{ so this makes a compact copy if necessary.}\\\\[12pt]
\\textbf{view(B,\\ T,\\ C)}: \\text{finally, }\\textbf{stitch all heads back together}\\text{ by flattening } (n\\_head,\\ h\\_s) \\text{ into the model dimension } \\mathtt{C = n\\_head * hs,}\\\\[6pt]
\\text{giving one vector per token → shape } (B,\\ T,\\ C).\\\\[12pt]
\\text{In plain terms: reorder to time-major per token → ensure memory layout is safe → merge the heads so each token has a single C-dim embedding again,}\\\\[6pt]
\\text{ready for the output projection.}
\\end{array}}
$$

&nbsp;

More on contiguous: [contiguous](#info:contiguous_notes)

    `,
  },

  contiguous_notes: {
    title: "contiguous",
    md: `
$$
\\text{A tensor is contiguous if its elements are laid out in memory in the usual row-major order for its current shape;}\\\\[6pt]
\\text{some ops (like transpose/advanced slicing) change the view/strides so the tensor becomes non-contiguous.}\\\\[6pt]
\\mathtt{.contiguous()}\\;\\text{ makes a compact copy in the new order so things like }\\mathtt{.view(...)}\\;\\text{ can work.}
$$

&nbsp;

---

&nbsp;

$\\textbf{Example 1 — A normal tensor is contiguous}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{a\\ =\\ torch.arange(12).view(3,\\ 4)\\ \\#\\ shape\\ (3,4)}\\\\[6pt]
\\mathtt{print(a)}\\\\[6pt]
\\mathtt{print("shape:".\\ a.shape,\\ "is\\_contiguous:".\\ a.is\\_contiguous(),\\ "strides:".\\ a.stride())}\\\\[6pt]
\\end{array}}
$$

$\\textbf{What you'll see}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{is\\_contiguous:\\ True}\\\\[6pt]
\\text{strides: }(4,\\ 1)\\ \\rightarrow\\ \\text{ move 1 step along dim1 (columns) = +1 in memory; move 1 step along dim0 (rows) = +4 in memory.}\\\\[6pt]
\\text{ That's standard row-major layout.}
\\end{array}}
$$

&nbsp;

---

&nbsp;

$\\textbf{Example 2 — Transpose makes it non-contiguous}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{b\\ =\\ a.transpose(0,\\ 1)\\ \\#\\ shape\\ (4,3)}\\\\[6pt]
\\mathtt{print(b)}\\\\[6pt]
\\mathtt{print("is\\_contiguous:".\\ b.is\\_contiguous(),\\ "strides:".\\ b.stride())}
\\end{array}}
$$

$\\textbf{What you'll see}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{is\\_contiguous:\\ False}\\\\[6pt]
\\text{strides: }(1,\\ 4)\\ \\rightarrow\\ \\text{ now moving along rows is the "fast" axis; memory jumps don't match the }(4,3)\\ \\text{ row-major order.}
\\end{array}}
$$

&nbsp;

---

&nbsp;

$\\textbf{Example 3 — Why }\\mathtt{.view(...)}\\ \\text{ can fail on non-contiguous}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{\\text{try:}} \\\\[6pt]
\\mathtt{\\ \\ \\ \\ flat\\ =\\ b.view(-1)\\ \\#\\ will\\ error:\\ not\\ contiguous} \\\\[6pt]
\\mathtt{\\text{except RuntimeError as e:}} \\\\[6pt]
\\mathtt{\\ \\ \\ \\ print("view\\ error:".\\ e)}
\\end{array}}
$$

$\\texttt{.view}\\ \\text{ only works when data are laid out contiguously for the new shape. A transposed tensor isn't.}$

$\\textbf{Two fixes:}$

$$
\\boxed{\\begin{array}{l}
\\mathtt{flat\\_ok1\\ =\\ b.contiguous().view(-1)\\ \\#\\ make\\ a\\ compact\\ copy,\\ then\\ view} \\\\[6pt]
\\mathtt{flat\\_ok2\\ =\\ b.reshape(-1)\\ \\#\\ reshape\\ will\\ copy\\ if\\ needed}\\\\[6pt]
\\mathtt{.contiguous()}\\ \\text{ copies data into a compact row-major buffer for the current ordering.}\\\\[6pt]
\\mathtt{.reshape}\\ \\text{ is more flexible: it returns a view if possible, or silently copies if not.} \\\\[6pt]
\\end{array}}
$$



    `,
  },
};

// Expose original notes for refactored InfoPanel fallback
export const ORIGINAL_INFO = INFO;


/** =========================
 *  Info Context (Single Source of Truth)
 *  ========================= */
const InfoContext = createContext(null);

function InfoProvider({ children }) {
  const [active, setActive] = useState(null); // 'gelu' | 'cdf' | 'layernorm' | null
  const open = (id) => setActive(id);
  const close = () => setActive(null);
  return (
    <InfoContext.Provider value={{ active, open, close }}>
      {children}
    </InfoContext.Provider>
  );
}

function useInfo() {
  const ctx = useContext(InfoContext);
  if (!ctx) throw new Error("useInfo must be used within <InfoProvider>");
  return ctx;
}

/** =========================
 *  CodeWithActions (event delegation for data-action clicks)
 *  ========================= */
function CodeWithActions({ html, onSelect }) {
  const { open } = useInfo();
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const onClick = (e) => {
      const t = e.target.closest("[data-action]");
      if (!t) return;
      const action = t.getAttribute("data-action");
      if (!action) return;

      if (action.startsWith("open:")) {
        // open info panel entries (gelu, layernorm, etc.)
        e.preventDefault();
        const id = action.split(":")[1];
        if (id) open(id);
      } else if (action.startsWith("select:")) {
        // switch right-panel code to a content entry id
        e.preventDefault();
        const id = action.split(":")[1];
        if (id && onSelect) onSelect(id);
      }
    };

    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [open, onSelect]);

  return (
    <pre className="text-sm overflow-x-auto border rounded p-2 bg-slate-50">
      <code ref={ref} dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}


/** =========================
 *  InfoPanel (renders active info; intercepts #cdf)
 *  ========================= */
function InfoPanel() {
  const { active, close, open } = useInfo();
  if (!active) return null;

  return (
    <div className="mt-4 pt-3 border-t text-xs text-slate-700 prose prose-sm max-w-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          {active === "weight_tying" && (
            <button
              className="text-xs underline"
              onClick={() => open("embedding_tok")}
            >
              ← Back
            </button>
          )}
          {active === "broadcasting" && (
            <button
              className="text-xs underline"
              onClick={() => open("embedding_pos")}
            >
              ← Back
            </button>
          )}
          {(active === "torch_ones_notes" || active === "torch_tril_notes" || active === "unsqueeze_notes") && (
            <button
              className="text-xs underline"
              onClick={() => open("causal_mask_notes")}
            >
              ← Back
            </button>
          )}
          {active === "contiguous_notes" && (
            <button
              className="text-xs underline"
              onClick={() => open("reorder_merge_heads_notes")}
            >
              ← Back
            </button>
          )}
          <h3 className="font-medium my-0">
            {INFO[active]?.title ?? "Info"}
          </h3>
        </div>
        <button className="text-xs underline" onClick={close}>
          Close
        </button>
      </div>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[
          [rehypeKatex, {
            trust: (ctx) => ctx.command === "\\href",
            strict: "ignore",
            fleqn: true,                // <— left-align display math at the KaTeX renderer level
          }],
          rehypeSlug,
        ]}
        components={{
          a: ({ href, children, ...props }) => {
            // Internal jump to another INFO entry: #info:<id>
            if (href && href.startsWith("#info:")) {
              const id = href.slice("#info:".length); // e.g., "weight_tying"
              return (
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (INFO[id]) open(id);
                  }}
                  className="underline cursor-pointer"
                  {...props}
                >
                  {children}
                </a>
              );
            }
        
            // legacy internal link you already had
            if (href === "#cdf") {
              return (
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    open("cdf");
                  }}
                  className="underline cursor-pointer"
                  {...props}
                >
                  {children}
                </a>
              );
            }
        
            // external links
            const isExternal = /^https?:\/\//.test(href || "");
            return (
              <a
                href={href}
                className="underline cursor-pointer"
                {...(isExternal
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                {...props}
              >
                {children}
              </a>
            );
          },
        }}        
      >
        {INFO[active]?.md || "_No content_"}
      </ReactMarkdown>
    </div>
  );
}

/** =========================
 *  Diagram + Helpers
 *  ========================= */

const gpt2TrainingCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Core model -----
class GPT2(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, max_toks, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_toks, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: LM head shares weights with token embeddings
        self.lm_head.weight = self.tok_emb.weight

    // [Dimension changes during forward]
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----- Simple training loop -----
model = GPT2(vocab_size=50304, n_layer=12, n_embd=768, n_head=12, max_toks=1024, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

for step, (x, y) in enumerate(dataloader):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)              # teacher-forced next-token loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")
`;


const attnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.key   = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # causal mask for up to 1024 tokens (adjust as needed)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        # project and split into heads: (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # causal mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                       # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reassemble heads
        y = self.resid_drop(self.proj(y))
        return y`;

const mlpCode = `import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd)  # GPT-2 uses 4x expansion
        # GELU (tanh approx like GPT-2 impl)
        try:
            self.act = nn.GELU(approximate="tanh")
        except TypeError:
            self.act = nn.GELU()
        self.fc_out = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc_out(self.act(self.fc_in(x))))`;

const transformerBlockCode = `import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        # Pre-LN pattern (GPT-2)
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)
        
    def forward(self, x):
        # Attention path
        x = x + self.attn(self.ln1(x))
        # MLP path
        x = x + self.mlp(self.ln2(x))
        return x`;

const content = {
  gpt2: {
    title: "GPT-2 (model & training)",
    summary: "Full model definition and a minimal training loop.",
    code: gpt2TrainingCode,
    anchors: [
      { id: "tok_emb",   label: "tok_emb",   match: "self.tok_emb = nn.Embedding(vocab_size, n_embd)" },
      { id: "tok_apply", label: "use tok",   match: "self.tok_emb(idx)" },
      { id: "pos_emb",     label: "pos_emb",     match: "self.pos_emb = nn.Embedding(max_toks, n_embd)" },
      { id: "pos_apply", label: "use pos",   match: "self.pos_emb(pos)[None, :, :]" },
      { id: "drop_def",   label: "define drop", match: "self.drop = nn.Dropout(dropout)" },
      { id: "drop_apply", label: "apply drop",  match: "x = self.drop(x)" },
      { id: "lnf_def",     label: "final ln def", match: "self.ln_f = nn.LayerNorm(n_embd)" },
      { id: "lnf_apply",   label: "apply ln_f",   match: "x = self.ln_f(x)" },
      { id: "lm_head_def",   label: "lm_head def",   match: "self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)" },
      { id: "lm_head_apply", label: "apply lm_head", match: "logits = self.lm_head(x)" },
      { id: "blocks_assign", label: "blocks assign", match: "self.blocks = nn.ModuleList([" },
      { id: "blocks_ctor",   label: "TransformerBlock call", match: "TransformerBlock(n_embd=n_embd, n_head=n_head, dropout=dropout)" },
      { id: "blocks_for",    label: "for loop",   match: "for _ in range(n_layer)" },
      { id: "blocks_close",  label: "close",      match: "])" },
      { id: "blocks_use_for",   label: "iterate blocks", match: "for blk in self.blocks:" },
      { id: "blocks_use_apply", label: "apply block",    match: "x = blk(x)" },
    ],
  },
  token_embed: {
    title: "Token Embeddings",
    summary: "Maps token indices to dense vectors.",
    code: "",
    anchors: [],
  },
  pos_embed: {
    title: "Positional Embeddings",
    summary: "Learned positional embeddings.",
    code: "",
    anchors: [],
  },
  emb_dropout: {
    title: "Embedding Dropout",
    summary: "Dropout applied to token+pos embeddings before Transformer blocks.",
    code: "",
    anchors: [],
  },
  stack: {
    title: "Transformer Block",
    summary: "GPT-2 repeats the Transformer Block many times.",
    code: transformerBlockCode,
    anchors: [
      { id: "class_def", label: "class", match: "class TransformerBlock(nn.Module):" },
      { id: "forward", label: "forward", match: "def forward(self, x):" }
    ]
  },
  block_ln: {
    title: "LayerNorm before Attention",
    summary: "Normalization before self-attention.",
    code: transformerBlockCode,
    anchors: [
      { id: "ln1", label: "ln1", match: "self.ln1 = nn.LayerNorm(n_embd)" },
    ],
  },
  block_attn: {
    title: "Masked Multi-Head Attention",
    summary: "Performs causal self-attention.",
    code: transformerBlockCode,
    anchors: [
      {
        id: "attn_assign",
        label: "self.attn assignment",
        match: "self.attn = CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)",
      },
    ],
  },
  attn_class: {
    title: "Causal Self-Attention",
    summary: "Implementation of masked self-attention.",
    code: attnCode,
    anchors: [
      { id: "class_def", label: "class",  match: "class CausalSelfAttention(nn.Module):" },
      { id: "mask",      label: "mask",   match: 'self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))' },
    ],
  },
  block_dropout_attn: {
    title: "Dropout after Attention",
    summary: "Dropout regularization.",
    code: transformerBlockCode,
    anchors: [
      {
        id: "dropout1",
        label: "dropout1",
        match: "self.drop = nn.Dropout(dropout)"
      }
    ],
  },
  block_ln2: {
    title: "LayerNorm before MLP",
    summary: "Normalization before feed-forward MLP.",
    code: transformerBlockCode,
    anchors: [
      { id: "ln2", label: "ln2", match: "self.ln2 = nn.LayerNorm(n_embd)" },
    ],
  },
  block_mlp: {
    title: "MLP / Feed-Forward",
    summary: "Two linear layers with GELU in between.",
    code: transformerBlockCode,
    anchors: [
      {
        id: "mlp_assign",
        label: "self.mlp assignment",
        match: "self.mlp = MLP(n_embd=n_embd, dropout=dropout)",
      },
    ],
  },
  block_dropout_mlp: {
    title: "Dropout after MLP",
    summary: "Dropout regularization.",
    code: transformerBlockCode,
    anchors: [
      {
        id: "dropout2",
        label: "dropout2",
        match: "self.drop = nn.Dropout(dropout)"
      }
    ],
  },
  final_ln: {
    title: "Final LayerNorm",
    summary: "Normalizes hidden states.",
    code: "",
    anchors: [],
  },
  lm_head: {
    title: "LM Head (Linear Layer)",
    summary: "Linear projection to vocab logits.",
    code: "",
    anchors: [],
  },
  loss_fn: {
    title: "Training Loss",
    summary: "CrossEntropy loss for next-token prediction.",
    code: gpt2TrainingCode,                           // ← show GPT-2 code
    anchors: [
      {
        id: "loss_line",
        label: "loss line",
        match:
          "loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))",
      },
    ],
  },
  mlp_subdiagram: {
    title: "MLP Architecture",
    summary: "Linear → GELU → Linear → Dropout.",
    code: mlpCode,
    anchors: [
      { id: "mlp_linear1", label: "fc_in",  match: "self.fc_in = nn.Linear(n_embd, 4 * n_embd)" },
      { id: "mlp_fc_in_use", label: "fc_in(x) use", match: "self.fc_in(x)" }, // ⬅️ add this
      { id: "mlp_gelu",    label: "act",    match: '        try:\n            self.act = nn.GELU(approximate="tanh")\n        except TypeError:\n            self.act = nn.GELU()' },
      { id: "mlp_gelu_in_use",    label: "act use",    match: 'self.act(self.fc_in(x))' },
      { id: "mlp_linear2", label: "fc_out", match: "self.fc_out = nn.Linear(4 * n_embd, n_embd)" },
      { id: "mlp_fc_out_use",  label: "fc_out use",  match: "self.fc_out(self.act(self.fc_in(x)))" },   // ⬅️ new
      { id: "mlp_dropout", label: "dropout", match: "self.drop = nn.Dropout(dropout)" },
      { id: "mlp_dropout_in_use", label: "dropout use", match: "self.drop(self.fc_out(self.act(self.fc_in(x))))" },
    ],
  },
};

const getFillColor = (id) => {
  if (id.includes("embed")) return "fill-amber-300";
  if (id.includes("ln")) return "fill-purple-300";
  if (id === "mlp_dropout" || id === "emb_dropout")
    return "fill-red-300";
  if (id.includes("attn")) return "fill-green-300";
  if (id.includes("mlp_linear") || id.includes("lm_head"))
    return "fill-indigo-300";
  if (id.includes("mlp")) return "fill-pink-300";
  if (id.includes("loss")) return "fill-yellow-300";
  return "fill-slate-200";
};

const Node = ({ id, label, x, y, w, h, onClick, underline }) => (
  <g onClick={() => onClick(id)} className="cursor-pointer">
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={10}
      className={`transition-all duration-200 ${getFillColor(
        id
      )} stroke-slate-500`}
    />
    <text
      x={x + w / 2}
      y={y + h / 2}
      dominantBaseline="middle"
      textAnchor="middle"
      className="select-none text-[10px] fill-slate-800"
      style={underline ? { textDecoration: "underline" } : undefined}
    >
      {label}
    </text>
  </g>
);

const Arrow = ({ x1, y1, x2, y2 }) => (
  <line
    x1={x1}
    y1={y1}
    x2={x2}
    y2={y2}
    stroke="black"
    markerEnd="url(#arrowhead)"
  />
);

function GPT2OuterFrame({ onClick }) {
  // Frame sized to surround the whole left-hand diagram + MLP subdiagram
  // Adjust width/height if you move things.
  return (
    <g>
      {/* Outer frame first so inner nodes are on top; pointer-events none so it doesn't block clicks */}
      <rect
        x={8}
        y={2}
        width={820}
        height={760}
        rx={18}
        className="fill-transparent stroke-slate-700"
        pointerEvents="none"
      />
      {/* Clickable title */}
      <text
        x={18}
        y={22}
        className="text-[13px] fill-slate-800 cursor-pointer"
        style={{ textDecoration: "underline", fontWeight: 600 }}
        onClick={() => onClick("gpt2")}
      >
        GPT-2
      </text>
    </g>
  );
}


const TransformerBlockGroup = ({ x, y, onClick }) => (
  <g>
    <rect x={x} y={y} width={380} height={290} rx={16} className="fill-slate-50 stroke-slate-400" />
    <text
      x={x + 190}
      y={y + 20}
      textAnchor="middle"
      className="text-[11px] fill-slate-700 cursor-pointer"
      style={{ textDecoration: "underline" }}
      onClick={() => onClick("gpt2_blocks")}
      underline="true"
    >
      Transformer Block (×12)
    </text>

    {/* Underline the LayerNorm nodes to look link-like */}
    <Node
      id="block_ln"
      label="LayerNorm"
      x={x + 20}
      y={y + 50}
      w={340}
      h={30}
      onClick={onClick}
      underline="true"
    />
    <Arrow x1={x + 190} y1={y + 80} x2={x + 190} y2={y + 105} />

    <Node
      id="block_attn"
      label="Masked Multi-Head Attention"
      x={x + 20}
      y={y + 110}
      w={340}
      h={30}
      onClick={onClick}
      underline="true"
    />
    <Arrow x1={x + 190} y1={y + 140} x2={x + 190} y2={y + 165} />

    <Node
      id="block_ln2"
      label="LayerNorm"
      x={x + 20}
      y={y + 170}
      w={340}
      h={30}
      onClick={onClick}
      underline="true"
    />
    <Arrow x1={x + 190} y1={y + 200} x2={x + 190} y2={y + 225} />

    <Node
      id="block_mlp"
      label="MLP / Feed-Forward"
      x={x + 20}
      y={y + 230}
      w={340}
      h={30}
      onClick={onClick}
      underline="true"
    />
    
    <Arrow x1={x + 365} y1={y + 245} x2={590} y2={y + 245} />
  </g>
);

const MLPSubDiagram = ({ x, y, onClick }) => (
  <g>
    <rect x={x} y={y} width={200} height={300} rx={12} className="fill-slate-50 stroke-slate-400" />
    <text
      x={x + 100}
      y={y + 25}
      textAnchor="middle"
      className="text-[11px] fill-slate-700 cursor-pointer"
      style={{ textDecoration: "underline" }}
      onClick={() => onClick("mlp_subdiagram")}
    >
      MLP Architecture
    </text>
    <Node
      id="mlp_linear1"
      label="Linear"
      x={x + 20}
      y={y + 50}
      w={160}
      h={30}
      onClick={onClick}
    />
    <Arrow x1={x + 100} y1={y + 80} x2={x + 100} y2={y + 110} />
    <Node
      id="mlp_gelu"
      label="GELU"
      x={x + 20}
      y={y + 115}
      w={160}
      h={30}
      onClick={onClick}
    />
    <Arrow x1={x + 100} y1={y + 145} x2={x + 100} y2={y + 175} />
    <Node
      id="mlp_linear2"
      label="Linear"
      x={x + 20}
      y={y + 180}
      w={160}
      h={30}
      onClick={onClick}
    />
    <Arrow x1={x + 100} y1={y + 210} x2={x + 100} y2={y + 240} />
    <Node
      id="mlp_dropout"
      label="Dropout"
      x={x + 20}
      y={y + 245}
      w={160}
      h={30}
      onClick={onClick}
    />
  </g>
);

/** =========================
 *  Utilities
 *  ========================= */
function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** =========================
 *  Inner Component (uses Info Center)
 *  ========================= */
function GPT2DecoderExplorerInner() {
  const { close } = useInfo();

  const [selectedId, setSelectedId] = useState("token_embed");
  const handleSelect = (id) => {
    setSelectedId(id);
    // Optional: close info when selecting other diagram parts
    close();
  };

  const effectiveId = ["mlp_linear1", "mlp_linear2", "mlp_gelu", "mlp_dropout"].includes(selectedId)
  ? "mlp_subdiagram"
  : (selectedId === "token_embed"
     || selectedId === "pos_embed"
     || selectedId === "emb_dropout"
     || selectedId === "gpt2_blocks"
     || selectedId === "final_ln"
     || selectedId === "lm_head"
     ? "gpt2"
     : selectedId);

  const component =
    content[effectiveId] || {
      title: "Unknown",
      summary: "No data",
      code: "",
      anchors: [],
    };

  const codeHtml = useMemo(() => {
    if (!component.code) return null;
    let html = escapeHtml(component.code);

    // Highlight anchors depending on which LayerNorm (or MLP sub-node) was clicked
    if (
      (selectedId === "block_ln" || selectedId === "block_ln2" || selectedId === "block_dropout_attn" || selectedId === "block_dropout_mlp") &&
      component.anchors?.length
    ) {
      component.anchors.forEach((a) => {
        // For block_ln -> ln1, block_ln2 -> ln2
        if (
          (selectedId === "block_ln" && a.id === "ln1") ||
          (selectedId === "block_ln2" && a.id === "ln2") ||
          (selectedId === "block_dropout_attn" && a.id === "dropout1")
        ) {
          const escaped = escapeHtml(a.match);
          html = html.replace(escaped, `<mark>${escaped}</mark>`);
        }
      });
    } else if (
      ["mlp_linear1", "mlp_linear2", "mlp_gelu", "mlp_dropout"].includes(selectedId) &&
      component.anchors
    ) {
      component.anchors.forEach((a) => {
        const shouldMark =
          (selectedId === "mlp_linear1" && (a.id === "mlp_linear1" || a.id === "mlp_fc_in_use")) ||
          (selectedId === "mlp_linear2" && (a.id === "mlp_linear2" || a.id === "mlp_fc_out_use")) ||
          (selectedId === "mlp_gelu" && (a.id === "mlp_gelu" || a.id === "mlp_gelu_in_use")) ||
          (selectedId === "mlp_dropout" && (a.id === "mlp_dropout" || a.id === "mlp_dropout_in_use"));
    
        if (shouldMark) {
          const esc = escapeHtml(a.match);
          html = html.replaceAll
            ? html.replaceAll(esc, `<mark>${esc}</mark>`)
            : html.replace(esc, `<mark>${esc}</mark>`);
        }
      });
    }

    {
      const attnAssignRaw =
        "CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)";
      const attnAssignEsc = escapeHtml(attnAssignRaw);
    
      const linked =
        selectedId === "block_attn"
          ? `<span data-action="select:attn_class" style="text-decoration: underline; cursor: pointer;"><mark>${attnAssignEsc}</mark></span>`
          : `<span data-action="select:attn_class" style="text-decoration: underline; cursor: pointer;">${attnAssignEsc}</span>`;
    
      // replace *every* occurrence just in case (use replaceAll if available)
      html = html.replaceAll ? html.replaceAll(attnAssignEsc, linked) : html.replace(attnAssignEsc, linked);
    }

    {
      const mlpAssignRaw = 'MLP(n_embd=n_embd, dropout=dropout)';
      const esc = escapeHtml(mlpAssignRaw);
    
      const wrapped =
        `<span data-action="select:mlp_subdiagram" ` +
        `style="text-decoration: underline; cursor: pointer;">` +
        // highlight when the MLP node is selected
        (selectedId === "block_mlp" ? `<mark>${esc}</mark>` : esc) +
        `</span>`;
    
      html = html.replaceAll
        ? html.replaceAll(esc, wrapped)
        : html.replace(esc, wrapped);
    }

    if (selectedId === "token_embed" && component.anchors?.length) {
      ["tok_emb", "tok_apply"].forEach(key => {     // highlight both def + use
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }

    if (selectedId === "pos_embed" && component.anchors?.length) {
      ["pos_emb", "pos_apply"].forEach(key => {
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }

    if (selectedId === "emb_dropout" && component.anchors?.length) {
      ["drop_def", "drop_apply"].forEach((key) => {
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }

    if (selectedId === "final_ln" && component.anchors?.length) {
      ["lnf_def", "lnf_apply"].forEach((key) => {
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }

    if (selectedId === "lm_head" && component.anchors?.length) {
      ["lm_head_def", "lm_head_apply"].forEach((key) => {
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }

    if (selectedId === "loss_fn" && component.anchors?.length) {
      const a = component.anchors.find((x) => x.id === "loss_line");
      if (a) {
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      }
    }

    {
      // Keep constructor clickable
      const ctor = component.anchors.find(a => a.id === "blocks_ctor");
      if (ctor) {
        const escCtor = escapeHtml(ctor.match);
        html = html.replace(
          escCtor,
          `<span data-action="select:stack" style="text-decoration: underline; cursor: pointer;">${escCtor}</span>`
        );
      }
    }

    if (selectedId === "gpt2_blocks" && component.anchors?.length) {
      // Highlight the rest of the block (definition + usage)
      [
        "blocks_ctor",
        "blocks_assign",
        "blocks_for",
        "blocks_close",
        "blocks_use_for",    // NEW
        "blocks_use_apply",  // NEW
      ].forEach(key => {
        const a = component.anchors.find(x => x.id === key);
        if (!a) return;
        const esc = escapeHtml(a.match);
        html = html.replace(esc, `<mark>${esc}</mark>`);
      });
    }
    
    if (selectedId === "emb_dropout" && component.anchors?.length) {
      html = html.replace(
        /nn\.Dropout\(dropout\)/g,
        '<span data-action="open:emb_dropout_notes" style="text-decoration: underline; cursor: pointer;">nn.Dropout(dropout)</span>'
      );
    }
    
    if (selectedId === "mlp_dropout" && component.anchors?.length) {
      html = html.replace(
        /nn\.Dropout\(dropout\)/g,
        '<span data-action="open:mlp_dropout_notes" style="text-decoration: underline; cursor: pointer;">nn.Dropout(dropout)</span>'
      );
    }

    // add link on "Dimension changes during forward"
    html = html.replace(
      /Dimension\s+changes\s+during\s+forward/g,
      '<span data-action="open:dimension_change_notes" style="text-decoration: underline; cursor: pointer;">Dimension changes during forward</span>'
    );

    // Make tokens clickable via data-action
    html = html.replace(
      /nn\.GELU\(approximate=\"tanh\"\)/g,
      '<span data-action="open:gelu" style="text-decoration: underline; cursor: pointer;">nn.GELU(approximate="tanh")</span>'
    );
    html = html.replace(
      /nn\.GELU\(\)/g,
      '<span data-action="open:gelu" style="text-decoration: underline; cursor: pointer;">nn.GELU()</span>'
    );

    // link on text "weight tying"
    html = html.replace(
      /weight\s+tying/g,
      '<span data-action="open:weight_tying_notes" style="text-decoration: underline; cursor: pointer;">weight tying</span>'
    );

    // link on text "nn.Linear(n_embd, vocab_size, bias=False)"
    html = html.replace(
      /nn\.Linear\([^)]*\)/g,
      (m) =>
        `<span data-action="open:linear" style="text-decoration: underline; cursor: pointer;">${m}</span>`
    );

    html = html.replace(
      /nn\.Linear\(n_embd, vocab_size, bias=False\)/g,
      '<span data-action="open:lm_head_notes" style="text-decoration: underline; cursor: pointer;">nn.Linear(n_embd, vocab_size, bias=False)</span>'
    );

    html = html.replace(
      /nn\.LayerNorm\([^)]*\)/g,
      (m) =>
        `<span data-action="open:layernorm" style="text-decoration: underline; cursor: pointer;">${m}</span>`
    );
    html = html.replace(
      /nn\.Embedding\(vocab_size,\s*n_embd\)/g,
      '<span data-action="open:embedding_tok" style="text-decoration: underline; cursor: pointer;">nn.Embedding(vocab_size, n_embd)</span>'
    );
    
    html = html.replace(
      /nn\.Embedding\(max_toks,\s*n_embd\)/g,
      '<span data-action="open:embedding_pos" style="text-decoration: underline; cursor: pointer;">nn.Embedding(max_toks, n_embd)</span>'
    );
    html = html.replace(
      /F\.cross_entropy(\([^)]*\))/g,
      '<span data-action="open:cross_entropy_notes" style="text-decoration: underline; cursor: pointer;">F.cross_entropy($1)</span>'
    );

    // multihead attention
    html = html.replace(
      /k\s*=\s*self\.key\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g,
      '<span data-action="open:key_view_notes" style="text-decoration: underline; cursor: pointer;">k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>'
    );
    html = html.replace(
      /q\s*=\s*self\.query\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g,
      '<span data-action="open:query_view_notes" style="text-decoration: underline; cursor: pointer;">q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>'
    );
    html = html.replace(
      /v\s*=\s*self\.value\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g,
      '<span data-action="open:value_view_notes" style="text-decoration: underline; cursor: pointer;">v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>'
    );
    
    // att = (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)
    html = html.replace(
      /\s*\(\s*q\s*@\s*k\.transpose\(\s*-2\s*,\s*-1\s*\)\s*\)\s*\*\s*\(\s*1\.0\s*\/\s*hs\*\*0\.5\s*\)/g,
      '<span data-action="open:q_k_product_notes" style="text-decoration: underline; cursor: pointer;"> (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)</span>'
    );

    // att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    html = html.replace(
      /\s*att\.masked_fill\(\s*self\.mask\s*\[\s*:\s*,\s*:\s*,\s*:\s*T\s*,\s*:\s*T\s*\]\s*==\s*0\s*,\s*float\(\s*'-inf'\s*\)\s*\)/g,
      '<span data-action="open:causal_mask_notes" style="text-decoration: underline; cursor: pointer;"> att.masked_fill(self.mask[:, :, :T, :T] == 0, float(\'-inf\'))</span>'
    );

    html = html.replace(
      /self\.attn_drop\s*=\s*nn\.Dropout\(dropout\)/g,
      '<span data-action="open:attn_dropout_notes" style="text-decoration: underline; cursor: pointer;">self.attn_drop = nn.Dropout(dropout)</span>'
    );
    html = html.replace(
      /self\.resid_drop\s*=\s*nn\.Dropout\(dropout\)/g,
      '<span data-action="open:resid_dropout_notes" style="text-decoration: underline; cursor: pointer;">self.resid_drop = nn.Dropout(dropout)</span>'
    );
    
    // create the link on only text "self.register_buffer"
    html = html.replace(
      /self\.register_buffer/g,
      '<span data-action="open:register_buffer_notes" style="text-decoration: underline; cursor: pointer;">self.register_buffer</span>'
    );

    // create the link on only text "torch.tril"
    html = html.replace(
      /torch\.tril/g,
      '<span data-action="open:torch_tril_notes" style="text-decoration: underline; cursor: pointer;">torch.tril</span>'
    );

    // create the link on only text "torch.ones"
    html = html.replace(
      /torch\.ones/g,
      '<span data-action="open:torch_ones_notes" style="text-decoration: underline; cursor: pointer;">torch.ones</span>'
    );

    // create the link on only text "unsqueeze"
    html = html.replace(
      /unsqueeze/g,
      '<span data-action="open:unsqueeze_notes" style="text-decoration: underline; cursor: pointer;">unsqueeze</span>'
    );

    // create the link on only text "F.softmax(att, dim=-1)"
    html = html.replace(
      /F\.softmax\(att, dim=-1\)/g,
      '<span data-action="open:softmax_notes" style="text-decoration: underline; cursor: pointer;">F.softmax(att, dim=-1)</span>'
    );

    // create the link on only text "y = att @ v"
    html = html.replace(
      /y = att @ v/g,
      '<span data-action="open:att_v_product_notes" style="text-decoration: underline; cursor: pointer;">y = att @ v</span>'
    );

    // create the link on only text " y.transpose(1, 2).contiguous().view(B, T, C)"
    html = html.replace(
      /y\.transpose\(1, 2\)\.contiguous\(\)\.view\(B, T, C\)/g,
      '<span data-action="open:reorder_merge_heads_notes" style="text-decoration: underline; cursor: pointer;"> y.transpose(1, 2).contiguous().view(B, T, C)</span>'
    );

    return html;
  }, [component.code, selectedId, component.anchors]);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">GPT-2 Decoder Explorer</h1>

      <div className="grid md:grid-cols-2 gap-6">
        <section className="bg-white rounded-xl shadow p-4">
          <svg viewBox="0 0 900 900" className="w-full h-auto">
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="0"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" />
              </marker>
            </defs>

            <GPT2OuterFrame onClick={handleSelect} />

            <Node
              id="token_embed"
              label="Token Embeddings"
              x={20}
              y={40}
              w={180}
              h={40}
              onClick={handleSelect}
            />
            <text
              x={210}
              y={65}
              textAnchor="middle"
              className="text-lg fill-slate-800"
            >
              +
            </text>
            <Node
              id="pos_embed"
              label="Positional Embeddings"
              x={220}
              y={40}
              w={180}
              h={40}
              onClick={handleSelect}
            />
            <Arrow x1={210} y1={80} x2={210} y2={105} />
            <Node
              id="emb_dropout"
              label="Dropout"
              x={20}
              y={110}
              w={380}
              h={40}
              onClick={handleSelect}
            />

            <Arrow x1={210} y1={150} x2={210} y2={175} />
            <TransformerBlockGroup x={20} y={180} onClick={handleSelect} />

            <Arrow x1={210} y1={470} x2={210} y2={515} />
            <Node
              id="final_ln"
              label="Final LayerNorm"
              x={20}
              y={520}
              w={380}
              h={40}
              onClick={handleSelect}
            />

            <Arrow x1={210} y1={560} x2={210} y2={605} />
            <Node
              id="lm_head"
              label="LM Head (Linear Layer)"
              x={20}
              y={610}
              w={380}
              h={40}
              onClick={handleSelect}
            />

            <Arrow x1={210} y1={650} x2={210} y2={695} />
            <Node
              id="loss_fn"
              label="Training Objective (CrossEntropyLoss)"
              x={20}
              y={700}
              w={380}
              h={40}
              onClick={handleSelect}
            />

            <MLPSubDiagram x={600} y={280} onClick={handleSelect} />
          </svg>
        </section>

        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-medium mb-2">{component.title}</h2>
          <p className="text-sm text-slate-600 mb-4">{component.summary}</p>

          {/* Code with actionable spans */}
          {codeHtml && <CodeWithActions html={codeHtml} onSelect={setSelectedId} />}


          {/* Centralized Info Panel (GELU / CDF / LayerNorm, etc.) */}
          <InfoPanel />
        </section>
      </div>
    </div>
  );
}

/** =========================
 *  Default export with Provider
 *  ========================= */
export default function GPT2DecoderExplorer() {
  // Runtime guard for content keys
  [
    "gpt2",
    "token_embed",
    "pos_embed",
    "emb_dropout",
    "block_attn",
    "block_mlp",
    "block_ln",
    "block_ln2",
    "block_dropout_attn",
    "block_dropout_mlp",
    "final_ln",
    "lm_head",
    "loss_fn",
    "stack",
    "mlp_subdiagram",
    "mlp_linear1",
    "mlp_linear2",
    "mlp_gelu",
  ].forEach((k) => {
    if (!content[k] && !["mlp_linear1", "mlp_linear2", "mlp_gelu"].includes(k))
      throw new Error(`Test failed: ${k} missing`);
  });

  return (
    <InfoProvider>
      <GPT2DecoderExplorerInner />
    </InfoProvider>
  );
}
