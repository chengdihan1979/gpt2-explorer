const INFO = {
  dimension_change_notes: {
    title: "Dimension changes during forward",
    md: `
**How the tensor shapes change during the forward pass.**

Notation:

- $B$ = batch size
- $T$ = sequence length
- $C$ = $n\\_embd$
- $V$ = \`vocab_size\`

## Inputs and helpers

\`\`\`text
idx                          (B, T)     Long tensor of token IDs
pos = arange(0, T)           (T)
\`\`\`

## Embeddings and sum

\`\`\`text
tok_emb(idx)                 (B, T, C)
pos_emb(pos)                 (T, C)
pos_emb(pos)[None, :, :]     (1, T, C)  broadcasts in the sum
x = tok_emb(idx) + pos_emb(pos)[None, :, :]
                             (B, T, C)
\`\`\`

## Dropout and blocks

\`\`\`text
x = emb_drop(x)              (B, T, C)  same shape; values randomly zeroed in train mode
for blk in blocks: x = blk(x)
                             (B, T, C)
\`\`\`

## Final norm and head

\`\`\`text
x = ln_f(x)                  (B, T, C)
logits = lm_head(x)          (B, T, V)
\`\`\`

## Targets and loss (if provided)

\`\`\`text
targets                              (B, T)
logits.view(-1, logits.size(-1))     (B*T, V)
targets.view(-1)                     (B*T)
loss = F.cross_entropy((B*T, V), (B*T))
                                     scalar ()  a single number
\`\`\`

## Return

- \`logits\`: $(B, T, V)$
- \`loss\`: a scalar, or \`None\`
`,
  },

  lm_head_notes: {
    title: "lm_head",
    md: `
**\`lm_head\`** turns each final hidden vector into a score for every token in the vocabulary.

## What it is conceptually

A linear map that converts each final hidden vector of size $n\\_embd$ into a logits vector of the vocabulary size (\`vocab_size\`). Each logit is the score for predicting a particular token in the vocabulary.

## Shapes

- **Hidden states:** $x \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$
- **\`lm_head.weight\`:** $\\mathbb{R}^{\\text{vocab\\_size} \\times n\\_embd}$ (the \`lm_head\` weight matrix is tied to the \`tok_emb\` weight matrix)
- **Logits:** \`logits = lm_head(x)\` $\\in \\mathbb{R}^{B \\times T \\times \\text{vocab\\_size}}$

> The \`lm_head\` weight matrix is just the vocabulary's token embedding matrix because of [weight tying](#info:weight_tying_notes).

## What it does intuitively

Each row of the weight matrix is a learned embedding vector for one vocabulary token.

The logit for token $v$ at position $t$ is the dot product between the hidden state $x$ at that position and the embedding row for token $v$.

- **During training,** the logits are compared with the ground-truth next token to compute the cross-entropy loss. This is done for all positions across the whole batch, against all vocabulary tokens, in one single batched matmul.
- **During inference,** the logits are turned into probabilities and used to pick the next token.

> [Why do the logits predict the next token during inference? Deeper insights!](#info:lm_head_insights_notes)

## Why no bias (\`bias=False\`)

1. **Weight tying.** GPT-2 ties the output weights to the input embedding table:

   \`\`\`python
   self.lm_head.weight = self.tok_emb.weight  # same parameters
   \`\`\`

   This saves parameters and often improves perplexity. The tied matrix matches in shape, but a separate output bias isn't needed (and is commonly omitted).

2. **Little benefit, lots of params.** A bias would add \`vocab_size\` more parameters; in practice it yields negligible gains here, so many implementations drop it.
`,
  },

  lm_head_insights_notes: {
    title: "lm_head_insights_notes",
    md: `
**Why does a dot product with the output embeddings predict the next token?**

## Setup

- $x \\in \\mathbb{R}^{n\\_embd}$ is the final hidden state of a token at some position.
- $e_v \\in \\mathbb{R}^{n\\_embd}$ is the output weight vector of a vocabulary token $v$ (with weight tying, it's also the input embedding for $v$).

First of all, we need to understand what training achieves:

1. **The hidden state $x$** captures the context of all tokens at previous and current positions, and it also encodes the features the next token should have.

   For example, the input is "The dogs that barked loudly". After training, the hidden state at position 5 captures the context "plural subject", learned from the tokens at positions 1 to 5. It also encodes the feature that the next token should most likely be a verb.

2. **$e_v$ is a global classifier weight vector for token $v$:** contexts that should choose $v$ align with this direction. It's context-independent (it doesn't change moment by moment); it's the general meaning of the token $v$ learned from the whole corpus.

3. **The cross-entropy loss sculpts both sides.** It pushes:
   - the hidden state $x$ to align with the correct token's $e_v$;
   - $e_v$ toward the direction that best separates contexts where $v$ is correct from others. Over time, contexts where $v$ is correct land near $e_v$.

With these in mind, let's continue.

## Dot-product form

The LM head gives a logit $\\ell_v$ for token $v$ by a dot product between $x$ and $e_v$:

$$
\\ell_v = x \\cdot e_v
$$

## Cosine form

By the definition of cosine in $\\mathbb{R}^{d}$ we also get:

$$
\\ell_v = x \\cdot e_v = \\lVert x \\rVert \\, \\lVert e_v \\rVert \\, \\cos\\theta(x, e_v)
$$

### Three-factor view

So each token's logit $\\ell_v$ factors into:

1. the magnitude of $x$;
2. the magnitude of $e_v$;
3. the direction of $e_v$, which determines the context-dependent cosine similarity between $x$ and $e_v$.

## Effect of the final LayerNorm

With the final layer norm, the magnitude of $x$ is nearly constant, within a small range. This makes the other two factors, the magnitude of $e_v$ and the cosine similarity, the dominant factors in determining the logit.

> At a high level, you can think of factor 3, **the direction of $e_v$**, as capturing the context, and factor 2, **the magnitude of $e_v$**, as capturing the frequency of the token.

- **Direction of $e_v$:** the closer $x$ points in the direction of $e_v$, the higher the third factor, and the more it contributes to the logit. Semantically, this means token $v$ bears a stronger resemblance to the context the hidden state $x$ holds, so it will be predicted with high probability.
- **Magnitude of $e_v$:** the token embedding matrix in a large language model (LLM) is correlated with token frequency: frequent tokens generally develop larger embedding norms, leading to higher predicted probabilities for those tokens.

## Why do frequent tokens get larger norms?

Here is an intuitive explanation.

During training we minimize a loss $\\mathcal{L}$ and update the parameters by

$$
\\Delta\\theta = -\\eta \\, \\nabla_\\theta \\mathcal{L}
$$

The parameters tied to a specific token $v$ (its vector $e_v$, and bias $b_v$ if present) receive gradient on steps where that token participates in the loss.

### Gradients for the LM head vector (bias-free head)

$$
\\ell = x E^\\top, \\quad p = \\mathrm{softmax}(\\ell), \\quad \\frac{\\partial \\mathcal{L}}{\\partial e_v} = \\big(p_v(x) - \\mathbf{1}\\{y = v\\}\\big) \\, x
$$

Across a dataset:

$$
\\nabla_{e_v} \\mathcal{L} = \\sum_{t} \\big(p_v(x_t) - \\mathbf{1}\\{y_t = v\\}\\big) \\, x_t
$$

### Frequent tokens "get more gradient"

1. If token $v$ appears $C_v$ times as the target, there are $C_v$ strong terms (the $-\\mathbf{1}\\{y_t = v\\}$ part) that pull $e_v$ toward the hidden state $x$ preceding $v$.
2. At other positions, $e_v$ contributes via $p_v(x_t)$, yielding smaller "push-away" terms when $v$ is not the target.
3. Over many examples, the first effect is directional alignment with the hidden state $x$.
4. Once the direction is reasonably aligned, increasing the magnitude of $e_v$ becomes an efficient way to lift $v$'s logit across many occurrences, i.e. it acts like a global gain/prior. Frequent tokens supply more such updates, so they tend to end up with larger norms.

> **Rule of thumb:** more occurrences $\\Rightarrow$ more (and larger) gradient contributions $\\Rightarrow$ stronger adaptation of that token's parameters.

## Final notes

- **The token embedding $e_v$** (from \`wte.weight\`) is context-independent. It captures a token's type-level tendencies: rough, averaged semantics/syntax learned from how that token appears across the whole corpus (e.g. that "dogs" is plural, noun-ish, often near verbs like "are/were"). Think of it as a baseline prototype.
- **The specific meaning in a sentence**, the token-in-context, lives in the hidden state $x_t$ after the token has gone through positional encoding + self-attention + MLPs. That's the **contextualized representation** the LM head uses to pick the next token.

> So the embedding carries **general meaning**, not the **moment-by-moment context**. The context is injected by the transformer and reflected in $x_t$, not in $e_v$.
`,
  },

  linear: {
    title: "Linear",
    md: `
**\`torch.nn.Linear\`**: a fully-connected (affine) layer.

## The formula

Given input $x \\in \\mathbb{R}^{*,\\,\\text{in}}$,

$$
y = x W^\\top + b
$$

where $W \\in \\mathbb{R}^{\\text{out}\\times\\text{in}},\\ b \\in \\mathbb{R}^{\\text{out}}$.

## Notes

- Maps \`in_features -> out_features\`.
- Supports leading-dim broadcasting: \`(N, ..., in) -> (N, ..., out)\`.
- Docs: [nn.Linear ↗](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
`,
  },
  gelu: {
    title: "GELU",
    md: `
**GELU (Gaussian Error Linear Unit)**

## The formula

$$
\\mathrm{GELU}(x) = x \\, \\Phi(x)
$$

where $\\Phi(x)$ is the [CDF of the standard normal distribution](#cdf).

> The GELU comes from the paper [Hendrycks & Gimpel (2016) ↗](https://arxiv.org/abs/1606.08415).

## Tanh approximation (used in GPT-2)

$$
\\mathrm{GELU}(x) \\approx 0.5\\,x\\,\\left(1 + \\tanh\\!\\left(\\sqrt{\\tfrac{2}{\\pi}}\\,(x + 0.044715\\,x^3)\\right)\\right)
$$

The reason behind the approximation comes from this [GitHub issue ↗](https://github.com/pytorch/pytorch/issues/39853).

Historically, evaluating erf (needed by exact GELU) was slower than a few multiplies + a tanh, especially on older GPU kernels and before [common fusions ↗](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/). So many models (including GPT-2) used the Hendrycks-Gimpel tanh approximation for speed.

### Today

- **TensorFlow:** \`tf.nn.gelu(x, approximate=False)\` computes the exact erf form; \`approximate=True\` uses the tanh form. With modern TF/XLA, both paths are vectorized and often fused; the exact version is usually fast enough unless you're ultra latency-sensitive.
- **PyTorch:** \`torch.nn.GELU()\` / \`F.gelu\` support \`approximate='none'\` (exact) or \`approximate='tanh'\`. Same idea: exact is generally fine; tanh can be a tiny bit faster on some hardware. Some transformer libraries also have a "FastGELU", which is just the tanh form.

## Why does GPT-2 use GELU instead of ReLU?

ReLU is defined as:

$$
\\mathrm{ReLU}(x) = \\max(0, x)
$$

So the gradient behavior of ReLU is:

- For $x > 0$: gradient = 1 (flows without scaling).
- For $x < 0$: gradient = 0 (dies out).

This makes optimization straightforward and avoids the vanishing gradients that plagued sigmoid/tanh in deep nets in the early days of deep learning.

But it also introduces the **"dying ReLU"** problem: if a neuron's weights push it into negative inputs too often, its gradient is always zero, so it can "die" and never update again.

> GELU doesn't discard negatives outright. It smoothly scales them down using the normal CDF.
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

  embedding_tok: {
    title: "Token Embedding",
    md: `
**Maps token IDs to dense vectors.**

## Shapes

- **Weight matrix:** $W \\in \\mathbb{R}^{\\text{vocab\\_size} \\times n\\_embd}$, one row per token in the vocabulary.
- **Input IDs:** $\\text{input\\_ids} \\in \\mathbb{Z}^{B \\times T}$
- **Embedding lookup:** $\\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$

### Parameter count

$$
\\text{params} = \\text{vocab\\_size} \\times n\\_embd
$$

> GPT-2 small: $50{,}257 \\times 768 \\approx 3.86 \\times 10^7$

## Training

- $W$ is learned via backprop.
- Only rows for tokens in the batch get updated.
- GPT-2 ties $W$ with the output softmax weights (\`lm_head\`). [What is weight tying?](#info:weight_tying_notes)

## Why use an embedding, not one-hot encoding?

A one-hot token vector would be length $\\text{vocab\\_size}$ and extremely sparse. \`nn.Embedding\` is equivalent to multiplying a one-hot vector by $W$, but it:

- avoids creating huge sparse vectors,
- learns dense, semantic representations where related tokens end up close in the embedding space.

## References

1. **PyTorch docs**

   \`nn.Embedding\` defines an embedding as a lookup table that "stores embeddings of a fixed dictionary and size" and is "retrieved using indices."

   [nn.Embedding ↗](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

2. **GPT-2 technical report (Input Representation)**

   Describes GPT-2's inputs as token embeddings + position embeddings (sum), establishing that token ids are first converted to embeddings before entering the stack.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

  `,
  },
  


  weight_tying_notes: {
    title: "Weight Tying",
    md: `
**Weight tying (shared input/output embeddings)**

Weight tying sets the output softmax weights equal to the input token embedding weights:

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# tie lm_head weight to tok_emb
self.lm_head.weight = self.tok_emb.weight
\`\`\`

> The idea comes from the paper [Using the Output Embedding to Improve Language Models ↗](https://arxiv.org/abs/1608.05859)

## What it does

After the last block, each position has a final vector $h \\in \\mathbb{R}^{n\\_embd}$. \`lm_head\` gives every token in the vocabulary a score (logit) by taking a dot product with that token's row:

$$
\\text{logit}_i = h \\cdot W[i]
$$

The more $h$ points in the same direction as row $i$, the higher token $i$ scores. With tying, row $i$ plays two roles:

- **Input:** the vector that represents token $i$ when it is read in.
- **Output:** the template the model matches against to predict token $i$ next.

> To predict \`Ġcat\`, the model has to move $h$ toward the embedding of \`Ġcat\` itself.

## Why do this?

- **Fewer parameters:** two matrices, \`V*d\` + \`d*V\`, become one \`V*d\`, roughly halving them. For GPT-2 small that saves $50{,}257 \\times 768 \\approx 38.6\\text{M}$ parameters: the model has about 124M instead of about 163M.
- **Better generalization:** acts as a helpful regularizer and often improves perplexity.

## How gradients flow

The assignment does not copy anything: both layers point to the **same \`Parameter\` object**, so the shared matrix receives the **sum** of the gradients from both uses.

- **Input side:** only the rows of tokens that appear in the batch get a gradient.
- **Output side:** the softmax gives a gradient to **all 50,257 rows** every step, pushing down the scores of the wrong tokens.

So with tying, even a rare token's embedding keeps getting trained, through the output side.

## Notes

- Many GPT-2 implementations use \`bias=False\` on \`lm_head\` when tying.
- Tying requires identical shapes: \`lm_head.weight.shape == tok_emb.weight.shape == (vocab_size, n_embd)\`.
- The shapes line up without a transpose because \`nn.Linear(n_embd, vocab_size)\` stores its weight as \`(out, in)\` = \`(vocab_size, n_embd)\`, the same layout as \`nn.Embedding(vocab_size, n_embd)\`.
- The original GPT-2 ties too: OpenAI's code computes the logits with the token embedding table \`wte\` itself: \`logits = tf.matmul(h_flat, wte, transpose_b=True)\`.

## Recent developments

Tying is a trade-off. The input row should say "what this token is"; the output row should say "what context comes before this token". Forcing one vector to do both is a constraint.

- **Large models usually don't tie:** Llama 3.1 8B and 70B, Qwen2.5 7B and 72B. At that scale the embedding table is a small share of the parameters, so the saving matters less.
- **Small models often do:** Llama 3.2 1B and 3B, Qwen2.5 0.5B and 3B, Gemma 2. Here the embedding table is a large share of the parameters.

> In Hugging Face model configs this is the \`tie_word_embeddings\` flag.

## References

1. **Using the Output Embedding to Improve Language Models**

   Press & Wolf, 2016. Shows that tying input and output embeddings reduces parameters and improves perplexity.

   [arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)

2. **Tying Word Vectors and Word Classifiers**

   Inan, Khosravi & Socher, 2016. Proposes the same idea independently, motivated by a loss framework for language modeling.

   [arXiv:1611.01462 ↗](https://arxiv.org/abs/1611.01462)

3. **GPT-2 reference model code**

   OpenAI's original model.py, where the logits are computed with \`wte\`.

   [openai/gpt-2 model.py ↗](https://github.com/openai/gpt-2/blob/master/src/model.py)

  `,
  },

  broadcasting: {
    title: "PyTorch Broadcasting",
    md: `
**Broadcasting** is the set of rules PyTorch (and NumPy) uses to perform operations on tensors of different shapes without explicitly copying data.

## Basic rule

Two tensors are compatible if, for each dimension (from the end):

- they are equal, or
- one of them is $1$.

The smaller tensor is "stretched" (virtually repeated) to match the larger shape.

## Examples

### Example 1: Vector + Matrix

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

### Example 2: Column vector + Row vector

$$
u \\in \\mathbb{R}^{3 \\times 1}, \\quad v \\in \\mathbb{R}^{1 \\times 4}
$$

Operation:

$$
M = u + v
$$

Shapes expand to $(3, 4)$:

$$
M \\in \\mathbb{R}^{3 \\times 4}
$$

## Formal rule

If we have shapes

$$
(s_1, s_2, \\dots, s_k), \\quad (t_1, t_2, \\dots, t_k)
$$

then they are broadcast-compatible if

$$
s_i = t_i \\quad \\text{or} \\quad s_i = 1 \\quad \\text{or} \\quad t_i = 1
$$

for all $i$ (padding shorter shapes with $1$ on the left if needed).

## Key benefits

- Saves memory (no explicit tiling).
- Makes code concise.
- Heavily used in deep learning for adding biases, positional embeddings, etc.
`,
  },

  mlp_dropout_notes: {
    title: "Residual Dropout in MLP",
    md: `
**Residual dropout** is applied to the output of each sublayer (attention branch and MLP branch) before adding it back via the residual path.

## Reminder: what happens in the MLP step

Inside a Transformer block, the feed-forward MLP is:

$$
\\text{MLP}(x) = W_2 \\cdot \\phi(W_1 x + b_1) + b_2
$$

where:

- $W_1, W_2$ are weight matrices;
- $b_1, b_2$ are bias vectors;
- $\\phi$ is a non-linear activation function (e.g. GELU in GPT-2).

This expands the hidden dimension with $W_1$, applies the non-linearity $\\phi$, and then projects back down with $W_2$.

## Why residual dropout

### 1. Regularization

The MLP produces high-dimensional features after the nonlinearity. Dropout prevents overfitting by randomly zeroing activations:

$$
h' = h \\odot m \\cdot \\frac{1}{1-p}
$$

where $h$ is the activation, $m$ is a Bernoulli mask, and $p$ is the dropout probability (e.g. $p = 0.1$ for GPT-2).

### 2. Preventing co-adaptation

Without dropout, neurons may rely on specific others. Residual dropout prevents the model from over-relying on any one feature/channel, encouraging redundant, distributed cues. The network learns to succeed even when some of its "favorite" features are momentarily missing.

### 3. Stabilizing optimization

During training, dropout randomly zeros parts of the activations, making any single neuron or path unreliable. As a result, the model learns backup cues and spreads information across features instead of cranking a few channels way up.

In the residual branches specifically, this "residual dropout" doesn't cap weights directly: it injects stochastic noise into the branch update, which discourages overly aggressive weight growth. To perform well under this noise, the network develops redundant, distributed representations and typically ends up with smaller, more balanced weights.

> This regularization helps both the attention and MLP residual branches, and is especially valuable for the feed-forward MLP, which usually has many more parameters than the attention layer.

## Example

If the MLP output for one token is

$$
h = [0.8,\\; -0.5,\\; 1.2,\\; 0.0,\\; -0.3,\\; 0.9],
$$

suppose the 3rd dimension (1.2) is consistently high for certain tokens. The next projection layer $W_2$ could just learn to "look at" the 3rd neuron and ignore the rest.

In other words, the model develops a shortcut: "dimension 3 tells me what I need."

### How dropout disrupts this shortcut

With dropout ($p = 0.5$), a random mask $m$ is applied:

$$
m = [1,\\; 0,\\; 0,\\; 1,\\; 1,\\; 1],
$$

and the output becomes

$$
h' = h \\odot m \\cdot \\frac{1}{1-p} = [1.6,\\; 0.0,\\; 0.0,\\; 0.0,\\; -0.6,\\; 1.8].
$$

Here the third neuron has been dropped. As a result, $W_2$ cannot always depend on $h_3$ being active, so it must learn to combine information from multiple dimensions.

> This encourages redundancy and more robust feature representations.

## Train vs. eval

- In **train** mode, dropout is active, so the model sees different activations each batch.
- In **eval** mode, dropout is disabled.

## References

1. **Original Transformer paper (where residual dropout is defined)**

   Vaswani et al., "Attention Is All You Need," §5.4 "Regularization": applies dropout to the output of each sublayer before adding it to the residual (and also to embeddings/attention probs). Good primary citation for what and where dropout is applied.

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 specifics (configs & terminology)**

   Hugging Face docs show GPT-2's three dropouts: \`resid_pdrop\` (residual branches), \`attn_pdrop\` (attention probs), \`embd_pdrop\` (embeddings). Useful for mapping concepts to code.

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

3. **General dropout theory (why it helps)**

   Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR 2014): canonical reference for dropout as stochastic regularization/model averaging intuition.

   [Dropout: A Simple Way to Prevent Neural Networks from Overfitting ↗](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

4. **Related/contrastive regularizer (dropping whole residual paths)**

   Huang et al., "Deep Networks with Stochastic Depth" (CVPR 2016): drops entire residual blocks during training ("drop-path"). A useful contrast with residual element-wise dropout.

   [Deep Networks with Stochastic Depth ↗](https://arxiv.org/abs/1603.09382)
`,
  },

  emb_dropout_notes: {
    title: "Embedding Dropout",
    md: `
**After the sum of token + positional embeddings, GPT-2 applies a dropout layer**, before feeding the representation into the first Transformer block.

## Why dropout here?

- Prevents overfitting by randomly zeroing out parts of the embedding vectors during training.
- Forces the model not to rely too heavily on specific embedding dimensions.
- Provides regularization right at the input stage, complementing dropout inside the Transformer blocks (e.g. after attention and feed-forward layers).

## Example

Suppose that after summing token embeddings and positional embeddings, one token has this 6-dimensional vector:

$$
h = [0.8, -0.5, 1.2, 0.0, -0.3, 0.9]
$$

This is the raw input to the first Transformer block.

### Apply dropout (say $p = 0.5$)

Dropout randomly "masks" some dimensions with probability $p$. The surviving dimensions are rescaled by $1/(1-p)$ to keep the expectation consistent.

Dropout mask example:

$$
m = [1, 0, 1, 0, 1, 1]
$$

Apply the mask and scale (scale factor = 2 because $1/(1-0.5) = 2$):

$$
h' = h \\odot m \\times 2 = [1.6, 0.0, 2.4, 0.0, -0.6, 1.8]
$$

So the second and fourth dimensions are "dropped" (zeroed), while the rest are doubled.

## Why this helps

- **Increases generalization and prevents overfitting:** helps the model learn more robust patterns by not "wiring" itself too closely to specific dimensions. If the model learned to rely too heavily on, say, the 3rd dimension, dropout forces it to also use other dimensions, because the 3rd dimension might vanish in training.
- **Regularizes embeddings:** token embeddings won't encode fragile, single-dimension "shortcuts"; they need to spread useful signals more evenly.

## References

1. **Original Transformer paper (where dropout is defined)**

   Vaswani et al. state explicitly: "we apply dropout to the sums of the embeddings and the positional encodings." GPT-2 follows this practice (with learned absolute positions).

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)
`,
},

  cross_entropy_notes: {
    title: "Cross Entropy",
    md: `
**Cross-entropy loss** is a measure of how well a model predicts the target distribution. Here it computes the negative log-likelihood of the correct next token at every position.

Docs: [CrossEntropy ↗](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

## Shape of logits and targets in language models

In GPT-style training:

- **Logits** (raw predictions before softmax): $(B, T, V)$
- **Targets** (ground-truth token indices): $(B, T)$

where:

- $B$: batch size
- $T$: sequence length
- $V$: vocabulary size

Example: for $B = 2$, $T = 4$, $V = 10$:

\`\`\`text
Logits shape  = (2, 4, 10)
Targets shape = (2, 4)
\`\`\`

## What cross-entropy expects

\`torch.nn.functional.cross_entropy\` expects:

- **Input:** a 2D tensor of shape $(N, C)$, where $N$ = number of samples and $C$ = number of classes.
- **Target:** a 1D tensor of shape $(N,)$ with integer class indices.

So we need to flatten both logits and targets.

## Why \`.view(-1, logits.size(-1))\`?

- \`logits.view(-1, V)\` reshapes $(B, T, V) \\to (B \\times T, V)$. Each row corresponds to one token position $(b, t)$ in the batch, i.e. the model's distribution over the vocabulary for that position.
- \`targets.view(-1)\` reshapes $(B, T) \\to (B \\times T,)$. Each entry is the correct class index for the next token in the sequence.

> Every token in the batch is treated as an independent classification example. The cross-entropy loss is computed for all sequences in the current batch at once.

## A tiny numerical example

Suppose a 4-class model outputs these prediction probabilities for one position:

$$
p = [0.1, 0.2, 0.6, 0.1]
$$

The true class is class 3 (0-indexed: index 2), i.e. one-hot $y = [0, 0, 1, 0]$.

> In practice (e.g. GPT-2), the target is not one-hot but the class index, i.e. 2. It's equivalent to multiplying by a one-hot, but far more memory/compute efficient.

Cross-entropy for this example:

$$
CE = -\\sum_{i=1}^4 y_i \\log p_i = -\\log p_3 = -\\log 0.6 \\approx 0.5108
$$

## References

1. **PyTorch docs**

   CrossEntropyLoss (multiclass softmax CE; targets are class indices, not one-hot).

   [PyTorch Docs ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

2. **CS231n Softmax Classifier**

   Notes on the derivation of softmax + cross-entropy and its gradients.

   [CS231n Softmax Classifier ↗](https://cs231n.github.io/linear-classify/)
`,
  },

  key_view_notes: {
    title: "Key View",
    md: `
## Key projection

\`k\` is the second slice of the fused projection: \`qkv = self.c_attn(x)\` maps every token from $C$ to $3C$, and \`q, k, v = qkv.split(C, dim=2)\` cuts it into three parts. Each part is equivalent to its own linear layer applied to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## Split channels into heads

\`.view(B, T, n_head, hs)\` splits the last dim $C$ into $n\\_head \\times hs$, with $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

In GPT-2 small, $n\\_head = 12$ and $hs = 768 / 12 = 64$, so each token's 768-dim k is cut **in order** into 12 chunks of 64:

\`\`\`text
k of one token (768 dims):
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

A smaller example: a 6-dim k split into 3 heads of 2 dims each.

\`\`\`text
k = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **The values don't change.** \`view\` only changes how the same numbers are indexed: no computation and no copy. The 768 numbers are the same 768 numbers, now seen as 12 rows of 64 instead of one long row.

The q and v tensors are split the same way, so each token ends up with 12 small q's, 12 small k's and 12 small v's of 64 dims each. Head $i$ uses chunk $i$ of each.

## Move heads dimension next to batch

\`.transpose(1, 2)\` swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

In GPT-2 small this is $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$.

### Same data, grouped differently

An example with $T = 3$ tokens and 2 heads (each entry is one 64-dim vector):

\`\`\`text
Before (B, T, n_head, hs): grouped by token
token 0: [head 0's k, head 1's k]
token 1: [head 0's k, head 1's k]
token 2: [head 0's k, head 1's k]

After (B, n_head, T, hs): grouped by head
head 0: [token 0's k, token 1's k, token 2's k]
head 1: [token 0's k, token 1's k, token 2's k]
\`\`\`

The numbers don't change; only the grouping does, from "which heads does each token have" to "which tokens does each head have".

### Why group by head?

Attention runs **within each head**, **across tokens**: in head 0, every token's q is compared with every token's k. After the transpose, the last two dims are $(T, hs)$, a $T \\times 64$ matrix holding all tokens' vectors for one head, so the next step \`q @ k.transpose(-2, -1)\` is a plain matrix multiply on those two dims. PyTorch treats the leading $(B, n\\_head)$ as batch dims and does that multiply for every sequence and every head, in parallel:

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head
\`\`\`
`,
  },

  query_view_notes: {
    title: "Query View",
    md: `
## Query projection

\`q\` is the first slice of the fused projection: \`qkv = self.c_attn(x)\` maps every token from $C$ to $3C$, and \`q, k, v = qkv.split(C, dim=2)\` cuts it into three parts. Each part is equivalent to its own linear layer applied to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## Split channels into heads

\`.view(B, T, n_head, hs)\` splits the last dim $C$ into $n\\_head \\times hs$, with $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

In GPT-2 small, $n\\_head = 12$ and $hs = 768 / 12 = 64$, so each token's 768-dim q is cut **in order** into 12 chunks of 64:

\`\`\`text
q of one token (768 dims):
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

A smaller example: a 6-dim q split into 3 heads of 2 dims each.

\`\`\`text
q = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **The values don't change.** \`view\` only changes how the same numbers are indexed: no computation and no copy. The 768 numbers are the same 768 numbers, now seen as 12 rows of 64 instead of one long row.

k and v are split the same way, so each token ends up with 12 small q's, 12 small k's and 12 small v's of 64 dims each. Head $i$ uses chunk $i$ of each.

## Move heads dimension next to batch

\`.transpose(1, 2)\` swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

In GPT-2 small this is $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$.

### Same data, grouped differently

An example with $T = 3$ tokens and 2 heads (each entry is one 64-dim vector):

\`\`\`text
Before (B, T, n_head, hs): grouped by token
token 0: [head 0's q, head 1's q]
token 1: [head 0's q, head 1's q]
token 2: [head 0's q, head 1's q]

After (B, n_head, T, hs): grouped by head
head 0: [token 0's q, token 1's q, token 2's q]
head 1: [token 0's q, token 1's q, token 2's q]
\`\`\`

The numbers don't change; only the grouping does, from "which heads does each token have" to "which tokens does each head have".

### Why group by head?

Attention runs **within each head**, **across tokens**: in head 0, every token's q is compared with every token's k. After the transpose, the last two dims are $(T, hs)$, a $T \\times 64$ matrix holding all tokens' vectors for one head, so the next step \`q @ k.transpose(-2, -1)\` is a plain matrix multiply on those two dims. PyTorch treats the leading $(B, n\\_head)$ as batch dims and does that multiply for every sequence and every head, in parallel:

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head
\`\`\`
`,
  },

  value_view_notes: {
    title: "Value View",
    md: `
## Value projection

\`v\` is the third slice of the fused projection: \`qkv = self.c_attn(x)\` maps every token from $C$ to $3C$, and \`q, k, v = qkv.split(C, dim=2)\` cuts it into three parts. Each part is equivalent to its own linear layer applied to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## Split channels into heads

\`.view(B, T, n_head, hs)\` splits the last dim $C$ into $n\\_head \\times hs$, with $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

In GPT-2 small, $n\\_head = 12$ and $hs = 768 / 12 = 64$, so each token's 768-dim v is cut **in order** into 12 chunks of 64:

\`\`\`text
v of one token (768 dims):
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

A smaller example: a 6-dim v split into 3 heads of 2 dims each.

\`\`\`text
v = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **The values don't change.** \`view\` only changes how the same numbers are indexed: no computation and no copy. The 768 numbers are the same 768 numbers, now seen as 12 rows of 64 instead of one long row.

The q and k tensors are split the same way, so each token ends up with 12 small q's, 12 small k's and 12 small v's of 64 dims each. Head $i$ uses chunk $i$ of each.

## Move heads dimension next to batch

\`.transpose(1, 2)\` swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

In GPT-2 small this is $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$.

### Same data, grouped differently

An example with $T = 3$ tokens and 2 heads (each entry is one 64-dim vector):

\`\`\`text
Before (B, T, n_head, hs): grouped by token
token 0: [head 0's v, head 1's v]
token 1: [head 0's v, head 1's v]
token 2: [head 0's v, head 1's v]

After (B, n_head, T, hs): grouped by head
head 0: [token 0's v, token 1's v, token 2's v]
head 1: [token 0's v, token 1's v, token 2's v]
\`\`\`

The numbers don't change; only the grouping does, from "which heads does each token have" to "which tokens does each head have".

### Why group by head?

Attention runs **within each head**, **across tokens**: in head 0, every token's q is compared with every token's k. After the transpose, the last two dims are $(T, hs)$, a $T \\times 64$ matrix holding all tokens' vectors for one head, so the next step \`q @ k.transpose(-2, -1)\` is a plain matrix multiply on those two dims. PyTorch treats the leading $(B, n\\_head)$ as batch dims and does that multiply for every sequence and every head, in parallel:

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head
\`\`\`
`,
  },

  q_k_product_notes: {
    title: "Q @ K Product",
    md: `
## Explanation of scaled dot-product attention

\`q\` and \`k\` have shape $(B, n\\_head, T, hs)$ (queries/keys per head).

- \`k.transpose(-2, -1)\` $\\rightarrow (B, n\\_head, hs, T)$ swaps the last two axes of the keys, so each query can dot with every key at all time steps.
- \`@\` is batched matrix multiply:

  $$
  (B, n\\_head, T, hs)\\ @\\ (B, n\\_head, hs, T) \\;\\rightarrow\\; (B, n\\_head, T, T)
  $$

  Result: for each head, a $T \\times T$ matrix of raw attention scores (one row per query position, one column per key position).
- \`* (1.0 / hs**0.5)\` multiplies by $1/\\sqrt{hs}$ (with $hs$ = head size). This is the scale from "Scaled Dot-Product Attention."

## What the $T \\times T$ matrix means

Entry $(i, j)$ is the dot product of token $i$'s q with token $j$'s k, scaled by $1/\\sqrt{hs}$ (why: see below): a single number saying how interested position $i$ is in position $j$:

$$
\\mathrm{att}[i][j] = \\frac{q_i \\cdot k_j}{\\sqrt{hs}}
$$

This is computed separately within each sequence and each head.

An example for "the river bank" ($T = 3$) in one head (made-up numbers, to show the structure):

\`\`\`text
             k_the   k_river   k_bank
q_the     [   1.0     0.2       0.5  ]   <- the's scores for each word
q_river   [   0.3     1.2       0.4  ]   <- river's scores for each word
q_bank    [   0.1     2.5       0.8  ]   <- bank's scores: river is highest
\`\`\`

- **Each row:** one token as the asker, scoring every token.
- **Each column:** one token as the one being looked up, scored by every token.

> **This is the first time positions interact.** \`c_attn\`, \`split\`, \`view\` and \`transpose\` all work on each token separately. At \`q @ k.transpose(-2, -1)\`, token $i$'s q meets **other** tokens' k for the first time.

Two things to note:

- The matrix is still the **full** $T \\times T$, including scores for looking ahead (e.g. the → bank). The [causal mask](#info:causal_mask_notes) removes those next.
- Each head has its own $T \\times T$ matrix, so there are 12 of them per sequence.

> 🆕 Most LLMs today don't add position embeddings to $x$. Instead they rotate q and k by position right before this step, so each score depends only on the distance between $i$ and $j$: [RoPE](#info:rope_notes).

## Why the $1/\\sqrt{hs}$ scale

In GPT-2 small $hs = 64$, so this step simply divides every score by $\\sqrt{64} = 8$.

### The problem: dot products grow with dimension

A score is a sum of $hs$ products:

$$
q \\cdot k = q_1 k_1 + q_2 k_2 + \\cdots + q_{hs} k_{hs}
$$

Assume the components of $q$ and $k$ are independent, with mean 0 and variance 1 (roughly the scale after LayerNorm). Then each term $q_i k_i$ has variance 1, and the sum of $hs$ of them has variance $hs$:

$$
\\operatorname{Var}(q \\cdot k) = hs, \\qquad \\operatorname{std}(q \\cdot k) = \\sqrt{hs}
$$

For $hs = 64$ the standard deviation is 8, so scores of $\\pm 8$ or $\\pm 16$ are common. The more dimensions, the bigger the scores.

### Why big scores hurt: softmax saturates

- scores $[8,\\; 0,\\; -8]$ → softmax $\\approx [0.9997,\\; 0.0003,\\; 0.0000]$: nearly all weight on one token;
- scores $[1,\\; 0,\\; -1]$ → softmax $\\approx [0.665,\\; 0.245,\\; 0.090]$: weight is spread out.

A saturated softmax causes two problems:

- it is almost one-hot, so a position can only look at one token;
- its gradients are close to zero, so the model can hardly adjust where it attends.

### The fix: divide by $\\sqrt{hs}$

Dividing by $\\sqrt{hs}$ brings the standard deviation back to 1:

$$
\\operatorname{std}\\!\\left(\\frac{q \\cdot k}{\\sqrt{hs}}\\right) = 1
$$

The $[8, 0, -8]$ above, divided by 8, becomes exactly $[1, 0, -1]$.

> Whatever the head size, scores stay on a scale with standard deviation about 1, so the softmax doesn't start out saturated. This is the "scaled" in **Scaled Dot-Product Attention**. The Transformer paper gives the same reason: "for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients" ([Vaswani et al., 2017 ↗](https://arxiv.org/abs/1706.03762), §3.2.1).

### How this relates to LayerNorm

- **LayerNorm** controls the scale of the **input** $x$.
- **$1/\\sqrt{hs}$** controls the scale of the **sum** $q \\cdot k$, which grows with the number of terms even when the inputs are already normalized.

The two work together; neither replaces the other.

After this step, the causal mask is applied, and then a softmax over the last dimension produces attention probabilities.
`,
  },

  torch_ones_notes: {
    title: "torch.ones",
    md: `
**Creates a tensor filled with 1s of the given shape.**

## Signature

\`\`\`python
torch.ones(*sizes, dtype=None, device=None, requires_grad=False)
\`\`\`

## Examples

- \`torch.ones(2, 3)\`
- \`torch.ones_like(x)\`

[torch.ones docs ↗](https://docs.pytorch.org/docs/stable/generated/torch.ones.html)
`,
  },

  torch_tril_notes: {
    title: "torch.tril",
    md: `
**Returns the lower-triangular part of a matrix; zeros out elements above the specified diagonal.**

## Signature

\`\`\`python
torch.tril(input, diagonal=0)
\`\`\`

## The \`diagonal\` argument

- If \`diagonal = 0\`: keep the main diagonal and below.
- If \`diagonal > 0\`: a positive value includes just as many diagonals above the main diagonal.
- If \`diagonal < 0\`: a negative value excludes just as many diagonals below the main diagonal.

Docs: [torch.tril ↗](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)
`,
  },

  unsqueeze_notes: {
    title: "unsqueeze",
    md: `
**Inserts a dimension of size 1 at position \`dim\` in the tensor's shape (no data copy).**

## Signature

\`\`\`python
torch.unsqueeze(input, dim)
# or
tensor.unsqueeze(dim)
\`\`\`

## Shape examples

$$
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(0)}\\ }\\ (1,\\ B,\\ T),\\qquad
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(-1)}\\ }\\ (B,\\ T,\\ 1)
$$

## Valid \`dim\`

Valid $dim \\in [-(n+1),\\ n]$ for an $n$-D input; a negative \`dim\` counts from the end.

Docs: [unsqueeze ↗](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)
`,
  },

  causal_mask_notes: {
    title: "Causal Mask",
    md: `
**This line applies the causal (look-back only) mask to the attention scores, so each token can attend only to itself and earlier tokens.**

\`\`\`python
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
\`\`\`

- \`att\`: $(B,\\ n\\_head,\\ T,\\ T)$, the scaled scores $q\\,k^\\top/\\sqrt{hs}$ from the [previous step](#info:q_k_product_notes)
- \`self.mask\`: $(1,\\ 1,\\ 1024,\\ 1024)$, 1s on and below the diagonal, 0s above

## Why a mask is needed

The score matrix from $q\\,k^\\top$ is the **full** $T \\times T$: every position has scored every other position, **including the ones after it**. With the 4 tokens \`I love black cats\` (rows = the query, "who is looking"; columns = the key, "who is looked at"):

\`\`\`text
            I     love   black   cats
I        [  ✓      ✓       ✓      ✓  ]
love     [  ✓      ✓       ✓      ✓  ]
black    [  ✓      ✓       ✓      ✓  ]
cats     [  ✓      ✓       ✓      ✓  ]
\`\`\`

Look at the \`love\` row. Its output is used to predict the next token, \`black\`, yet it has already scored \`black\`. That is **peeking at the answer**.

- **Training:** the whole sequence is processed in parallel, and every position predicts its next token. If \`love\` could see \`black\`, predicting \`black\` would just be copying, and the model would learn nothing useful.
- **Inference:** text is generated left to right, so the later tokens **do not exist yet**. Hiding the future during training makes training match how the model is used.

The fix: before softmax, block everything above the diagonal, so each row sees only **itself and earlier positions**.

## Step by step

### 1. What \`self.mask\` is

It is built once, in \`__init__\`:

\`\`\`python
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))
\`\`\`

- [torch.ones](#info:torch_ones_notes)(1024, 1024) is all 1s; [torch.tril](#info:torch_tril_notes) keeps the **lower triangle** (on and below the diagonal) and sets everything above it to 0. At size 4:

  \`\`\`text
  torch.tril(torch.ones(4, 4))
  [[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]
  \`\`\`

  Row $t$ has 1s in columns $0 \\dots t$: position $t$ may look at itself and everything before it.
- [unsqueeze(0)](#info:unsqueeze_notes)\`.unsqueeze(0)\` adds two leading dims, giving $(1,\\ 1,\\ 1024,\\ 1024)$, so it can **broadcast** over batch and heads (how this works: see **Broadcasting** at the end of step 4 below).
- It is a [buffer](#info:register_buffer_notes): it moves with the module to GPU/CPU and is saved with it, but it is not a parameter and is never trained.

### 2. Slice to the current length

The buffer is sized for the maximum length, 1024, but the current sequence has $T$ tokens:

\`\`\`text
self.mask[:, :, :T, :T]   →   (1, 1, T, T)
\`\`\`

This takes the top-left $T \\times T$ corner. The top-left corner of a lower-triangular matrix is itself lower-triangular, so for $T = 4$ it is exactly the $4 \\times 4$ matrix above.

### 3. Turn it into "which links are forbidden"

\`== 0\` turns the 0/1 matrix into booleans: **True** where the entry is 0, i.e. the future positions to block.

\`\`\`text
mask == 0
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
\`\`\`

### 4. \`masked_fill\`: write \`-inf\` where it is True

\`att.masked_fill(bool_mask, -inf)\` writes $-\\infty$ into \`att\` wherever the mask is True, and leaves every other score unchanged. One head's scores for \`I love black cats\`, before and after:

\`\`\`text
before                                    after
          I     love  black  cats                   I     love  black  cats
I      [ 2.1,   0.3,  1.1,   0.6 ]      I      [ 2.1,   -inf, -inf,  -inf ]
love   [ 0.5,   1.3,  1.7,   2.5 ]      love   [ 0.5,   1.3,  -inf,  -inf ]
black  [ 0.2,   0.9,  1.7,   1.4 ]      black  [ 0.2,   0.9,  1.7,   -inf ]
cats   [ 0.1,   0.4,  2.5,   0.8 ]      cats   [ 0.1,   0.4,  2.5,   0.8  ]
\`\`\`

Now \`I\` sees only itself, \`love\` sees \`I\` and itself, and \`cats\` sees all four. In general, for query position $t$:

$$
\\text{att}_{t,j} = -\\infty \\quad \\text{for } j > t
$$

**Broadcasting.** The mask is $(1,\\ 1,\\ T,\\ T)$ while \`att\` is $(B,\\ n\\_head,\\ T,\\ T)$. The size-1 dims are stretched, so the **same** triangle is applied to every sequence in the batch and every head. That is correct: "don't look ahead" depends only on positions, not on the content or the head.

### 5. Why \`-inf\`, not \`0\`?

The next line is \`att = F.softmax(att, dim=-1)\`. Softmax raises $e$ to each score and divides by the row's sum:

$$
\\text{softmax}(s)_i = \\frac{e^{s_i}}{\\sum_j e^{s_j}}
$$

Take the \`love\` row from step 4.

**Filling \`-inf\`: future weights become exactly 0.** Since $e^{-\\infty} = 0$:

\`\`\`text
scores:   [ 0.5,   1.3,   -inf,  -inf ]
e^s:      [ 1.65,  3.67,  0,     0    ]    sum 5.32
softmax:  [ 0.31,  0.69,  0,     0    ]
\`\`\`

The future gets no weight, and the allowed positions are **renormalized automatically**: they still sum to 1.

**Filling \`0\` would not work.** A score of 0 does not mean "ignore"; it is an ordinary score, and $e^0 = 1$:

\`\`\`text
scores:   [ 0.5,   1.3,   0,     0     ]
softmax:  [ 0.225, 0.501, 0.137, 0.137 ]
\`\`\`

The two future positions still receive 27% of the attention: the model is still peeking.

**Why not zero the weights after softmax?** Then the row no longer sums to 1. With the unmasked \`love\` row \`[0.5, 1.3, 1.7, 2.5]\`:

\`\`\`text
softmax (no mask):  [ 0.072, 0.16, 0.238, 0.53 ]
zero the future:    [ 0.072, 0.16, 0,     0    ]    sum only 0.23
\`\`\`

It would need a second division to fix the sum. Filling \`-inf\` **before** softmax gives "future = 0" and "row sums to 1" in one step.

> In one sentence: $e^{-\\infty} = 0$, so softmax treats a \`-inf\` score as if that position did not exist.

## Notes

- This mask enforces **causality** only; it does not handle padding. If a batch contains padded tokens, a padding mask is usually applied too (another \`masked_fill\`), so padding positions also get $-\\infty$.
- Some code uses a large negative constant (e.g. $-10^9$) instead of $-\\infty$; after softmax its weight is also effectively 0.
`,
  },

  register_buffer_notes: {
    title: "register_buffer",
    md: `
**A buffer is a tensor that belongs to the model but is not trained.** It moves with the model (\`.to()\`, \`.cuda()\`, \`.cpu()\`) and is saved and loaded with it (\`state_dict\`), but it is not a parameter, so the optimizer never updates it.

## Parameter vs plain attribute vs buffer

There are three ways to keep a tensor on an \`nn.Module\`. Take the causal mask:

\`\`\`python
# 1. parameter
self.mask = nn.Parameter(torch.tril(torch.ones(1024, 1024)))

# 2. plain attribute
self.mask = torch.tril(torch.ones(1024, 1024))

# 3. buffer
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))
\`\`\`

They differ in three ways:

|  | trained? | moves with \`.to("cuda")\`? | saved in \`state_dict\`? |
| --- | :---: | :---: | :---: |
| Parameter | yes | yes | yes |
| plain attribute | no | no | no |
| buffer | **no** | **yes** | **yes** |

> **Analogy.** The model is a toolbox. Parameters are knives to be sharpened (training sharpens them). A buffer is the ruler in the box: it never needs sharpening, but it moves house with the box. A plain attribute is a sticky note on the old wall: it gets left behind.

## Why the causal mask is a buffer

- **Not a Parameter:** the mask is a fixed rule; "don't look ahead" should not be learned. As a Parameter, the optimizer would update it every step, and a 0 above the diagonal could drift to 0.03: the model would start peeking.
- **Not a plain attribute:** \`model.to("cuda")\` moves only Parameters and buffers. A plain attribute stays on the CPU, so in forward \`att\` (on GPU) and \`mask\` (on CPU) would meet and raise a device mismatch error.
- **A buffer fits exactly:** never trained, but moved to the GPU and saved/loaded together with the model.

## Using \`register_buffer\`

\`\`\`python
Module.register_buffer(name, tensor, persistent=True)
\`\`\`

- **name:** the buffer becomes an attribute with this name. After \`self.register_buffer("mask", ...)\`, forward uses it simply as \`self.mask\`.
- **persistent:** with the default \`True\`, the buffer is saved in \`state_dict\`. With \`False\`, it still moves with the model but is not saved; useful for things that are cheap to rebuild.
- **Listing:** buffers appear in \`model.buffers()\`, not in \`model.parameters()\`. Since the optimizer is built from \`model.parameters()\`, it never sees them.

"Not trained" means not updated by gradients. A buffer can still change if code writes to it: BatchNorm keeps its \`running_mean\` and \`running_var\` as buffers and updates them during forward. The causal mask never changes.

**Typical uses:** masks (like the causal mask), running statistics, fixed index tables, and other constants tied to the model.

## In GPT-2

\`\`\`python
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))
\`\`\`

This stores the $(1,\\ 1,\\ 1024,\\ 1024)$ lower-triangular mask once, in \`__init__\`. Every forward slices it to the current length; see [Causal Mask](#info:causal_mask_notes).

> 🆕 Later implementations (e.g. newer nanoGPT) mostly call \`F.scaled_dot_product_attention(..., is_causal=True)\`, which applies the causal mask inside the kernel, so no 1024 × 1024 mask buffer is stored.

Docs: [register_buffer ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
`,
  },

  softmax_notes: {
    title: "softmax",
    md: `
## F.softmax explanation

1. Converts the raw attention scores in \`att\` into probabilities along the last dimension (keys dimension).
2. For each query token (and each head), the resulting row tells how much attention that query token should pay to each allowed previous token: a higher value means more attention, zero means ignore (e.g., masked future/padding).
3. Each row sums to 1, so the weights are comparable and interpretable.
4. The output has the same shape as the input; only the values are rescaled to valid probabilities.
5. These probabilities are then used to take a weighted average of the value vectors for that query token.

## Math interpretation

Applies softmax along the last dimension of \`att\` to turn logits into probabilities.

**Shape:**

$$
att \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}
\\quad\\xrightarrow{\\ \\mathtt{softmax(\\cdot,\\ dim=-1)}\\ }\\quad
P \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}
$$

For fixed $(b,h,t_q)$:

$$
P_{b,h,t_q,j}
\\;=\\;
\\frac{\\exp\\big(att_{b,h,t_q,j}\\big)}
{\\sum_{k=1}^{T}\\exp\\big(att_{b,h,t_q,k}\\big)},
\\qquad
\\sum_{j=1}^{T} P_{b,h,t_q,j} = 1.
$$

**Effect of the causal mask:**

$$
att_{b,h,t_q,j}=-\\infty \\;\\Rightarrow\\; P_{b,h,t_q,j}=0
$$

(future positions get zero probability).

## Notes

- \`dim=-1\` means "over the last axis" (the keys dimension).
- \`F.softmax\` is numerically stable (it effectively subtracts a row max).
`,
  },

  att_v_product_notes: {
    title: "att @ v",
    md: `
## Explanation and intuition

1. \`att\` holds, for each token (and each head), how much attention to pay to every allowed token in the sequence.
2. \`v\` holds the content for each token: what is worth copying "if you attend to me."

**Intuition:** for each token, the operation mixes together other tokens' information according to those attention weights.

- Tokens with higher weights contribute more; tokens with zero weight do not contribute.
- The result \`y\` is a context-aware summary for each token (per head) that pulls in the most relevant information.

## Math interpretation

\`\`\`python
y = att @ v
\`\`\`

$$
att \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times T},\\quad
v \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times hs},\\quad
y \\in \\mathbb{R}^{B\\times n\\_head \\times T\\times hs}
$$

For a fixed batch $b$ and head $h$:

$$
y^{(b,h)} \\,=\\, att^{(b,h)}\\, v^{(b,h)},\\quad
att^{(b,h)}\\in\\mathbb{R}^{T\\times T},\\;
v^{(b,h)}\\in\\mathbb{R}^{T\\times hs},\\;
y^{(b,h)}\\in\\mathbb{R}^{T\\times hs}
$$

Elementwise (per query position $t$):

$$
y_{b,h,t,:}
\\;=\\;
\\sum_{j=1}^{T} att_{b,h,t,j}\\; v_{b,h,j,:}
$$

> - **Attention row:** each element is the probability of attending to a specific key position (i.e., a specific token in the allowed context: self and previous tokens in causal attention). Masked tokens (future/padding) have probability 0.
> - **Value vectors:** there is one value vector per source token (same positions as the keys). These vectors carry the content to be mixed.
> - **Result:** for a given query token $t$, the output is a weighted mix of those value vectors, using the attention row as weights. Tokens with higher probability contribute more; zero means no contribution.

### Concrete mini-example

Say the attention row for token $t$ is $[0.5,\\ 0.3,\\ 0.2]$. The value vectors at positions 1, 2, 3 are $v_1, v_2, v_3$ (each is a length-$hs$ vector). Then the output vector for token $t$ is

$$
0.5\\,v_1 + 0.3\\,v_2 + 0.2\\,v_3,
$$

a single vector (length $hs$) that mixes information using that row's weights.

### Causality

$$
att_{b,h,t,j}=0 \\;\\text{ for } j>t
\\;\\Rightarrow\\;
\\text{the sum only ranges over allowed (past/self) positions.}
$$

### Convex combination

$$
att_{b,h,t,j}\\ge 0 \\;\\text{ and }\\; \\sum_{j} att_{b,h,t,j}=1
\\;\\Rightarrow\\;
y_{b,h,t,:}\\;\\text{is a convex combination of the } v_{b,h,j,:}
$$
`,
  },

  reorder_merge_heads_notes: {
    title: "reorder_merge_heads",
    md: `
## Explanation and intuition

\`\`\`python
y.transpose(1, 2).contiguous().view(B, T, C)
\`\`\`

### \`transpose(1, 2)\`

You currently have per-head outputs arranged as $(B,\\ n\\_head,\\ T,\\ hs)$. This swap moves **time** next to **batch** → $(B,\\ T,\\ n\\_head,\\ hs)$, so for each token you can gather all heads together.

### \`contiguous()\`

After a transpose, the tensor's memory is strided (non-contiguous). \`view\` needs a contiguous layout, so this makes a compact copy if necessary.

### \`view(B, T, C)\`

Finally, **stitch all heads back together** by flattening $(n\\_head,\\ hs)$ into the model dimension \`C = n_head * hs\`, giving one vector per token → shape $(B,\\ T,\\ C)$.

> In plain terms: reorder to time-major per token → ensure the memory layout is safe → merge the heads so each token has a single $C$-dim embedding again, ready for the output projection.

More on contiguous: [contiguous](#info:contiguous_notes)
`,
  },

  contiguous_notes: {
    title: "contiguous",
    md: `
**A tensor is contiguous if its elements are laid out in memory in the usual row-major order for its current shape.**

Some ops (like transpose, or strided slicing such as \`a[:, ::2]\`) change the view/strides so the tensor becomes non-contiguous. \`.contiguous()\` makes a compact copy in the new order so things like \`.view(...)\` can work.

## Example 1: a normal tensor is contiguous

\`\`\`python
a = torch.arange(12).view(3, 4)  # shape (3,4)
print(a)
print("shape:", a.shape, "is_contiguous:", a.is_contiguous(), "strides:", a.stride())
\`\`\`

### What you'll see

\`\`\`text
is_contiguous: True
\`\`\`

Strides: $(4,\\ 1)$ → moving 1 step along dim 1 (columns) = +1 in memory; moving 1 step along dim 0 (rows) = +4 in memory. That's the standard row-major layout.

## Example 2: transpose makes it non-contiguous

\`\`\`python
b = a.transpose(0, 1)  # shape (4,3)
print(b)
print("is_contiguous:", b.is_contiguous(), "strides:", b.stride())
\`\`\`

### What you'll see

\`\`\`text
is_contiguous: False
\`\`\`

Strides: $(1,\\ 4)$ → now moving along rows is the "fast" axis; the memory jumps don't match the $(4,3)$ row-major order.

## Example 3: why \`.view(...)\` can fail on non-contiguous tensors

\`\`\`python
try:
    flat = b.view(-1)  # will error: not contiguous
except RuntimeError as e:
    print("view error:", e)
\`\`\`

\`.view\` only works when the data are laid out contiguously for the new shape. A transposed tensor isn't.

### Two fixes

\`\`\`python
flat_ok1 = b.contiguous().view(-1)  # make a compact copy, then view
flat_ok2 = b.reshape(-1)            # reshape will copy if needed
\`\`\`

- \`.contiguous()\` copies the data into a compact row-major buffer for the current ordering.
- \`.reshape\` is more flexible: it returns a view if possible, or silently copies if not.
`,
  },
};

// Expose original notes for refactored InfoPanel fallback
export const ORIGINAL_INFO = INFO;
