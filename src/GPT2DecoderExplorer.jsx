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
$$\\textbf{What it is conceptually:}\\\\[4pt]$$
A linear map that converts each final hidden vector of size $n\\_embd$ into the logits vector of the
vocabulary size ($vocab\\_size$). Each logit is the score for predicting a particular token in the vocabulary.

&nbsp;

$$\\textbf{Shapes:}\\\\[4pt]$$
$$
\\bullet\\ \\text{Hidden states } x \\in \\mathbb{R}^{B \\times T \\times n\\_embd} \\\\[4pt]
\\bullet\\ \\text{lm\\_head.weight} \\in \\mathbb{R}^{\\texttt{vocab\\_size} \\times n\\_embd} \\text{ (lm\\_head weight matrix tied to tok\\_emb weight matrix)} \\\\[4pt]
\\textit{lm\\_head weight matrix is just the vocabulary's token embedding matrix because of \\href{#info:weight_tying_notes}{\\text{weight tying}}.} \\\\[4pt]
\\bullet\\ \\mathtt{logits = lm\\_head(x)} \\in \\mathbb{R}^{B \\times T \\times \\texttt{vocab\\_size}} \\\\[8pt]
$$

$$\\textbf{What it does intuitively:}\\\\[4pt]$$
Each row of the weight matrix is a learned embedding vector for one vocabulary token.

The logit for token $v$ at position $t$ is the dot product between the hidden state $x$ at that position and the embedding row for token $v$.

During training, the logit will be used to compare with the ground truth next token to compute the cross entropy loss. And this operation is performed over all tokens at all positions across the whole batch with all vocabulary tokens
in one single batched matmul.


During inference, the logit is turned into probabilities and used to pick the next token.

&nbsp;

$$\\href{#info:lm_head_insights_notes}{\\text{Why does logit predict the next token during inference, deeper insights!}}$$

&nbsp;

$$\\textbf{Why no bias (}\\mathtt{bias=False}\\textbf{):}\\\\[4pt]$$
$$
1.\\ \\textbf{Weight tying.}\\ \\text{GPT-2 typically ties the output weights to the input embedding table:} \\\\[4pt]
\\ \\ \\ \\mathtt{self.lm\\_head.weight = self.tok\\_emb.weight}\\ \\ \\#\\ \\text{same parameters} \\\\[4pt]
\\ \\ \\ \\text{This saves parameters and often improves perplexity. The tied matrix matches in shape,} \\\\
\\ \\ \\ \\text{but a separate output bias isn't needed (and is commonly omitted).} \\\\[6pt]
2.\\ \\textbf{Little benefit, lots of params.}\\ \\text{A bias would add } \\texttt{vocab\\_size} \\text{ more parameters;} \\\\
\\ \\ \\ \\text{in practice it yields negligible gains here, so many implementations drop it.}
$$
    `,
  },

  lm_head_insights_notes: {
    title: "lm_head_insights_notes",
    md: `
$$
\\textbf{Setup}\\\\[6pt]
\\text{Assume x} \\in \\mathbb{R}^{n\\_embd} \\text{ is the final hidden state of a token at some position.} \\\\[4pt]
\\text{Assume } e_v \\in \\mathbb{R}^{n\\_embd} \\text{ is the output weight vector of a vocabulary token v. (with weight tying, it's also the input embedding for v)} \\\\[8pt]
\\text{First of all, we need to understand the following things training achieves:} \\\\[4pt]
\\textbf{1.}\\text{The hidden state x captures the context of all tokens at previous and current positions, and it also encodes the features}\\\\
\\text{the next token should have.}\\\\[4pt]
\\text{For example, the input is "The dogs that barked loudly".} \\\\[4pt]
\\text{The hidden state at position 5 captures the context "plural subject" learned from tokens from position 1 to 5 after training, } \\\\[4pt]
\\text{it also encodes the feature that the next token should be most likely a verb.} \\\\[4pt]
\\textbf{2. } e_v \\text{ is a global classifier weight vector for token v: contexts that should choose v align with this direction.}\\\\
\\text{It's moment-by-moment context-independent, it's the general meaning of the token v learned from the whole corpus.} \\\\[4pt]
\\textbf{3.}\\text{The cross-entropy loss sculpts both sides. The cross-entropy loss pushes:} \\\\[4pt]
\\bullet\\ \\text{Hidden state x to align with the correct token's } e_v \\\\[4pt]
\\bullet\\ e_v \\text{ toward the direction that best separates contexts where } \\mathtt{v} \\text{ is correct from others. Over time, contexts where}\\\\
\\mathtt{v} \\text{ is correct land near } e_v \\text{.} \\\\[8pt]
\\text{With these bear in mind, let's continue.} \\\\[4pt]
$$

$$
\\textbf{Dot-product form}\\\\[4pt]
\\text{The LM head gives a logit } \\ell_v \\text{ for token v by a dot product between x and} e_v \\\\[4pt]
\\ell_v\\text{ = x} \\cdot e_v \\\\[8pt]
$$

$$
\\textbf{Cosine form}\\\\[4pt]
\\text{Also by definition of cosine in } \\mathbb{R}^{d} \\text{ we get:} \\\\[4pt]

\\ell_v\\text{ = x} \\cdot e_v = \\lVert x \\rVert\\, \\lVert e_v \\rVert\\, \\cos\\theta(x, e_v),\\\\[4pt]

\\textbf{Three-factor view}\\\\[4pt]
\\text{So each token's logit } \\ell_v \\text{ factors into:} \\\\[4pt]
\\bullet\\ \\text{factor 1: the magnitude of x} \\\\[4pt]
\\bullet\\ \\text{factor 2: the magnitude of } e_v \\\\[4pt]
\\bullet\\ \\text{factor 3: the direction of } e_v \\text{ which determines the context-dependent cosine similarity between x and } e_v \\\\[4pt]
$$

$$
\\textbf{Effect of final LayerNorm}\\\\[6pt]
\\text{With final layer norm, the magnitude of x is a nearly constant within a small range.} \\\\[4pt]
\\text{This makes the other two factors, the magnitude of } e_v \\text{ and the cosine similarity, the dominant factors in determining the logit.} \\\\[4pt]
\\textbf{On the high level, you can think of factor 3 - the direction of } e_v \\textbf{ captures the context and factor 2 - the magnitude of } e_v \\\\[4pt]
\\textbf{captures the frequency of the token.} \\\\[8pt]
\\textbf{1. Direction of } e_v \\text{: The closer x points in the direction of } e_v \\text{, the higher the third factor will be and it will contribute more to the logit.} \\\\[4pt]
\\text{Semantically this means token v bears a stronger resemblance to the context the hidden state x holds so it will be predicted with high probability.} \\\\[8pt]
\\textbf{2. Magnitude of } e_v \\text{: The token embedding matrix in a large language model (LLM) is correlated with token frequency, } \\\\[4pt]
\\text{with frequent tokens generally developing larger embedding norms and, leading to higher predicted probabilities for those tokens.} \\\\[4pt]
$$

$$
\\textbf{Why frequent tokens get larger norms? Here I'll give a intuitive explanation.} \\\\[6pt]
\\text{During training we minimize a loss } \\mathcal{L} \\text{ and update parameters by } \\Delta\\theta=-\\eta\\,\\nabla\\_\\theta \\mathcal{L}\\text{.}\\\\[4pt]
\\text{The parameters tied to a specific token }v\\text{ (its vector }e_v\\text{, and bias }b_v\\text{ if present) receive gradient on steps where that token participates in the loss.}\\\\[4pt]
\\textbf{Gradients for the LM head vector (bias-free head)} \\\\[4pt]
\\ell = xE^\\top,\\quad p=\\mathrm{softmax}(\\ell),\\quad \\frac{\\partial \\mathcal{L}}{\\partial e_v} \\;=\\; \\big(p_v(x)-\\mathbf{1}\\{y=v\\}\\big)\\,x.\\\\[4pt]
\\text{Across a dataset: }\\quad \\nabla_{e_v}\\mathcal{L} \\;=\\; \\sum_{t}\\big(p_v(x_t)-\\mathbf{1}\\{y_t=v\\}\\big)\\,x_t.\\\\[4pt]
\\textbf{Frequent tokens “get more gradient”:} \\\\[4pt]
\\text{1. If token }v\\text{ appears }C_v\\text{ times as the target, there are }C_v\\text{ strong terms (the }-\\mathbf{1}\\{y_t=v\\}\\text{ part) that pull }e_v\\text{ toward the current hidden} \\\\[4pt]
\\text{state }x\\text{'s preceding }v\\text{.}\\\\[4pt]
\\text{2. On other positions, }e_v\\text{ contributes via }p_v(x_t)\\text{, yielding smaller “push-away” terms when }v\\text{ is not the target.}\\\\[4pt]
\\text{3. Over many examples, the first effect is directional alignment with the current hidden state x}\\\\[4pt]
\\text{4. Once the direction is reasonably aligned, increasing the magnitude of } e_v \\text{ becomes an efficient way to lift } v \\text{'s logit across many}  \\\\[4pt]
\\text{occurrences-i.e., it acts like a global gain/prior. Frequent tokens supply more such updates, so they tend to end up with larger norms.}\\\\[4pt]
\\textbf{Rule of thumb: } \\text{More occurrences } \\Rightarrow \\text{ more (and larger) gradient contributions } \\Rightarrow \\text{ stronger adaptation of that token's parameters.} \\\\[4pt]
$$

$$
\\textbf{Final notes:} \\\\[4pt]
\\bullet\\ \\textbf{Token embedding } e_v \\text{ (from } \\mathtt{wte.weight} \\text{) is context-independent. It captures a token's type-level tendencies—} \\\\
\\text{rough, averaged semantics/syntax learned from how that token appears across the whole corpus } \\\\
\\text{(e.g., that "dogs" is plural, noun-ish, often near verbs like "are/were"). Think of it as a baseline prototype.} \\\\[4pt]
\\bullet\\ \\text{The } \\textbf{specific meaning in a sentence} \\text{---the token-in-context---lives in the hidden state } x_t \\text{ after the token has gone } \\\\
\\text{through positional encoding + self-attention + MLPs. That’s the } \\textbf{contextualized representation} \\text{ the LM head} \\\\
\\text{uses to pick the next token.} \\\\[4pt]
\\text{So: the embedding carries } \\textbf{general meaning} \\text{, not the } \\textbf{moment-by-moment context}. \\text{ The context is injected by} \\\\
\\text{the transformer and reflected in } x_t \\text{, not in } e_v. \\\\[4pt]
$$

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

  embedding_tok: {
    title: "Token Embedding",
    md: `
  $$\\textbf{Maps token IDs to dense vectors.}\\\\[8pt]$$
  
  $$\\textbf{Weight matrix:}$$
  $$
  W \\in \\mathbb{R}^{\\text{vocab\\_size} \\times \\text{n\\_embd}}
  $$
  $$\\textbf{Input IDs:}$$
  $$
  \\text{input\\_ids} \\in \\mathbb{Z}^{B \\times T}
  $$

  $$\\textbf{Embedding lookup:}$$
  $$
  \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
  $$

  $$\\textbf{Parameter count:}$$
  $$
  \\text{params} = \\text{vocab\\_size} \\times n\\_embd
  $$
  (e.g. GPT-2 small: $50{,}257 \\times 768 \\approx 3.86 \\times 10^7$)

  &nbsp;
  
  ---

  &nbsp;

  $$\\textbf{Training}\\\\[4pt]$$

  $$\\text{- W is learned via backprop.}\\\\[2pt]$$
  $$\\text{- Only rows for tokens in the batch get updated.}\\\\[2pt]$$
  $$\\text{- GPT-2 often ties W with the output softmax weights.}$$ ([What is weight tying?](#info:weight_tying_notes))
  
  &nbsp;

  ---

  &nbsp;

  $$\\textbf{Why use embedding? Not one hot encoding.}\\\\[4pt]$$

  A one-hot token vector would be length $vocab\\_size$ and extremely sparse. $nn.Embedding$ is equivalent to multiplying a one-hot vector by $W$, but it:

  $$\\bullet\\ \\text{avoids creating huge sparse vectors,}$$

  $$\\bullet\\ \\text{learns dense, semantic representations where related tokens end up close in the embedding space.}$$

  &nbsp;

  ---

  &nbsp;

  $$\\textbf{References:}\\\\[4pt]$$
  $$\\textbf{1. PyTorch docs:}$$
  nn.Embedding doc: Defines an embedding as a lookup table that “stores embeddings of a fixed dictionary and size” and is “retrieved using indices,”
  $$\\href{https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html}{\\texttt{nn.Embedding}}$$
  
  $$\\textbf{2. GPT-2 technical report (Input Representation):}$$
  Describes GPT-2’s inputs as token embeddings + position embeddings (sum), establishing that token ids are first converted to embeddings before entering the stack.
  $$\\href{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com}{\\texttt{GPT-2 Technical Report}}$$
  `,
  },
  


  weight_tying_notes: {
    title: "Weight Tying",
    md: `
  $$\\textbf{Weight tying (shared input/output embeddings)}\\\\[4pt]$$
  
  The idea comes from the paper [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
    
  Weight tying sets the output softmax weights equal to the input token embedding weights:
  
  &nbsp;

  \`\`\`python
  # tie lm_head weight to tok_emb
  self.lm_head.weight = self.tok_emb.weight
  \`\`\`
  
  &nbsp;
  
  $$\\textbf{Why do this?}\\\\[4pt]$$
    
  - Fewer parameters (≈ halves \`V*d\` + \`d*V\` to just \`V*d\`).
  - Acts as a helpful regularizer and often improves perplexity.
  
  &nbsp;
  
  $$\\textbf{Notes:}\\\\[4pt]$$

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
    title: "Residual Dropout in MLP",
    md: `
$$\\textbf{Remainder: What Happens in the MLP Step}\\\\[4pt]$$

$$\\text{Inside a Transformer block, the feed-forward MLP is:}$$
$$
\\text{MLP}(x) = W_2 \\cdot \\phi(W_1 x + b_1) + b_2
$$
$$\\text{where:}$$
$$
\\text{- $W_1, W_2$ are weight matrices  }\\\\[2pt]
\\text{- $b_1, b_2$ are bias vectors  }\\\\[2pt]
\\text{- $\\phi$ is a non-linear activation function (e.g. GELU in GPT-2)  }\\\\[2pt]
$$
$$\\text{This expands the hidden dimension with $W_1$, applies the non-linearity $\\phi$, and then projects back down with $W_2$.}\\\\[8pt]$$

---

&nbsp;

$$\\textbf{What it is:}\\\\[4pt]$$

Dropout applied to the output of each sublayer (attention branch and MLP branch) before adding it back via the residual path. 

&nbsp;

---

&nbsp;

$$\\textbf{Why Residual Dropout}\\\\[4pt]$$


$$1. \\textbf{Regularization}\\\\[4pt]$$
The MLP produces high-dimensional features after the nonlinearity. Dropout prevents overfitting by randomly zeroing activations:

$$
h' = h \\odot m \\cdot \\frac{1}{1-p}
$$

where $h$ is the activation, $m$ is a Bernoulli mask, and $p$ is the dropout probability. (e.g. $p=0.1$ for GPT-2)

&nbsp;

$$2. \\textbf{Preventing co-adaptation}\\\\[4pt]$$
Without dropout, neurons may rely on specific others. 
Residual dropout prevents the model from over-relying on any one feature/channel, encouraging redundant, distributed cues. The network learns to succeed even when some of its “favorite” features are momentarily missing.

&nbsp;

$$3. \\textbf{Stabilizing optimization}\\\\[4pt]$$
During training, dropout randomly zeros parts of the activations, making any single neuron or path unreliable; as a result, 
the model learns backup cues and spreads information across features instead of cranking a few channels way up. In the residual branches specifically, 
this “residual dropout” doesn’t cap weights directly—it injects stochastic noise into the branch update, which discourages overly aggressive weight growth. 
To perform well under this noise, the network develops redundant, distributed representations and typically ends up with smaller, more balanced weights. 
This regularization helps both the attention and MLP residual branches, and is especially valuable for the feed-forward MLP, which usually has many more parameters than the attention layer.

&nbsp;

---

&nbsp;

$$\\textbf{Example}\\\\[4pt]$$

If the MLP output for one token is

$$
h = [0.8,\\; -0.5,\\; 1.2,\\; 0.0,\\; -0.3,\\; 0.9],
$$

Suppose the 3rd dimension (1.2) is consistently high for certain tokens.

The next projection layer $W_2$ could just learn to “look at” the 3rd neuron and ignore the rest.

In other words, the model develops a shortcut: “dimension 3 tells me what I need.”

&nbsp;

$$\\textbf{How Dropout Disrupts This Shortcut}\\\\[4pt]$$

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

&nbsp;

---

&nbsp;

$$\\textbf{Train vs. Eval}\\\\[4pt]$$

In train mode, dropout is active, so the model sees different activations each batch.

In eval mode, dropout is disabled.

&nbsp;

---

&nbsp;

$$\\textbf{References:}\\\\[4pt]$$

$1.$ $$\\textbf{Original Transformer paper (where residual dropout is defined).}$$
Vaswani et al., Attention Is All You Need, §5.4 “Regularization”: applies dropout to the output of each sublayer before adding it to the residual (and also to embeddings/attention probs). Good primary citation for what and where dropout is applied. 
$$\\href{https://arxiv.org/pdf/1706.03762v2}{\\texttt{Attention Is All You Need}}$$

$2.$ $$\\textbf{GPT-2 specifics (configs \\& terminology).}$$
Hugging Face docs show GPT-2's three dropouts: resid_pdrop (residual branches), attn_pdrop (attention probs), embd_pdrop (embeddings). Useful for mapping concepts to code.
$$\\href{https://huggingface.co/docs/transformers/main/model_doc/gpt2?utm_source=chatgpt.com}{\\texttt{GPT-2 Configs}}$$

$3.$ $$\\textbf{General dropout theory (why it helps).}$$
Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR 2014): canonical reference for dropout as stochastic regularization/model averaging intuition.
$$\\href{https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf}{\\texttt{Dropout: A Simple Way to Prevent Neural Networks from Overfitting}}$$

$4.$ $$\\textbf{Related/contrastive regularizer (dropping whole residual paths).}$$
Huang et al., Deep Networks with Stochastic Depth (CVPR 2016): dropping entire residual blocks during training (“drop-path”)—useful contrast with residual element-wise dropout.
$$\\href{https://arxiv.org/abs/1603.09382}{\\texttt{Deep Networks with Stochastic Depth}}$$
`,
  },

  emb_dropout_notes: {
    title: "Embedding Dropout",
    md: `
  After the sum of token + positional embeddings, GPT-2 applies a dropout layer. This is done before feeding the representation into the first Transformer block.

  &nbsp;

  $$\\textbf{Why Dropout Here?}\\\\[8pt]$$
  
  Prevents overfitting by randomly zeroing out parts of the embedding vectors during training.

  Forces the model not to rely too heavily on specific embedding dimensions.

  Provides regularization right at the input stage, complementing dropout inside Transformer blocks (e.g., after attention and feed-forward layers).

  &nbsp;

  ---

  &nbsp;
  
  $$\\textbf{Example}\\\\[8pt]$$
  
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

  $$\\textbf{Why This Helps}\\\\[8pt]$$

  
  **Increase generalization and Prevents overfitting**: Helps model learn more robust patterns by not “wiring” itself too closely to specific dimensions. If the model learned to rely too heavily on, say, the 3rd dimension, dropout forces it to also use other dimensions, because the 3rd dimension might vanish in training.

  &nbsp;

  **Regularizes embeddings**: Token embeddings won't encode fragile, single-dimension “shortcuts” — they need to spread useful signals more evenly.

  &nbsp;

  ---

  &nbsp;

  $$\\textbf{References:}\\\\[4pt]$$
  $$\\textbf{1. Original Transformer paper (where dropout is defined).}$$
  Vaswani et al. explicitly: “we apply dropout to the sums of the embeddings and the positional encodings.” GPT-2 follows this practice (with learned absolute positions).
  $$\\href{https://arxiv.org/pdf/1706.03762v2}{\\texttt{Attention Is All You Need}}$$
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

$$\\textbf{Shape of Logits and Targets in Language Models}\\\\[8pt]$$

In GPT-style training:

&nbsp;

$$\\textbf{Logits (raw predictions before softmax)}\\\\[4pt]$$  

$$
(B,T,V)
$$

where:  
$$
\\text{- B: batch size  }\\\\[2pt]
\\text{- T: sequence length  }\\\\[2pt]
\\text{- V: vocabulary size  }\\\\[2pt]
$$

$$\\textbf{Targets (ground truth token indices)}\\\\[4pt]$$  

$$
(B,T)
$$

Example:
For $B=2$, $T=4$, $V=10$:

$$
\\text{Logits shape} = (2, 4, 10)
$$

$$
\\text{Targets shape} = (2, 4)
$$

---

&nbsp;

$$\\textbf{What CrossEntropy Expects}\\\\[4pt]$$

\`torch.nn.functional.cross_entropy\` expects:

&nbsp;

**Input**: 2D tensor of shape $(N, C)$

where $N$ = number of samples, $C$ = number of classes.

&nbsp;

**Target**: 1D tensor of shape $(N,)$ with integer class indices.

&nbsp;

So we need to flatten both logits and targets.

&nbsp;

---

&nbsp;

$$\\textbf{Why .view(-1, logits.size(-1))?}\\\\[4pt]$$

\`logits.view(-1, V)\` reshapes $(B, T, V) \\to (B \\times T, V)$.

→ Each row corresponds to one token position (b, t) in the batch, i.e., the model’s distribution over the vocabulary for that position.

&nbsp;

\`targets.view(-1)\` reshapes $(B, T) \\to (B \\times T,)$.

→ Each entry is the correct class index for the next token in the sequence.

&nbsp;

Thus, every token in the batch is treated as an independent classification example.
The cross-entropy loss is computed for all sequences in the current batch at once.

&nbsp;

---

&nbsp;

$\\textbf{A tiny numerical example}\\\\[4pt]$

&nbsp;

Suppose a 4-class model outputs prediction probabilities for one sequence is:
$$
\\bullet \\ p = [0.1, 0.2, 0.6, 0.1]
$$
The true class is class 3 (0-indexed: index 2), i.e. one-hot $y = [0, 0, 1, 0]$.
(In practice (e.g., GPT-2), the target is not one-hot but the class index, i.e., 2. It's equivalent to multiplying by a one-hot, but far more memory/compute efficient)

&nbsp;

Cross-entropy for this example:
$$
CE = -\\sum_{i=1}^4 y_i \\log p_i = -\\log p_3 = -\\log 0.6 \\approx 0.5108
$$

&nbsp;

---

&nbsp;

$\\textbf{References:}\\\\[4pt]$
$$\\textbf{1. PyTorch docs:}$$
CrossEntropyLoss (multiclass softmax CE; targets are class indices, not one-hot).
$$\\href{https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?utm_source=chatgpt.com}{\\texttt{PyTorch Docs}}$$
  
$$\\textbf{2. CS231n Softmax Classifier:}$$
notes (derivation of softmax + cross-entropy and gradients).
$$\\href{https://cs231n.github.io/linear-classify/?utm_source=chatgpt.com}{\\texttt{CS231n Softmax Classifier}}$$

&nbsp;

    `,
  },

  key_view_notes: {
    title: "Key View",
    md: `
$$\\textbf{Key projection}\\\\[4pt]$$

$k(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

$$\\textbf{Split channels into heads}\\\\[4pt]$$

$.view(B, T, n\\_head, hs)$ — splits the last dim $C$ into $n\\_head\\times hs$ with
$hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs).
$$

$$\\textbf{Move heads dimension next to batch}\\\\[4pt]$$

$.transpose(1, 2)$ — swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs).
$$
    `,
  },

  query_view_notes: {
    title: "Query View",
    md: `
$$\\textbf{Query projection}\\\\[4pt]$$

$q(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

$$\\textbf{Split channels into heads}\\\\[4pt]$$

$.view(B, T, n\\_head, hs)$ — splits the last dim $C$ into $n\\_head\\times hs$ with
$hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs).
$$

$$\\textbf{Move heads dimension next to batch}\\\\[4pt]$$

$.transpose(1, 2)$ — swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs).
$$
    `,
  },

  value_view_notes: {
    title: "Value View",
    md: `
$$\\textbf{Value projection}\\\\[4pt]$$

$v(x)$ — applies a linear layer to every token:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C).
$$

$$\\textbf{Split channels into heads}\\\\[4pt]$$

$.view(B, T, n\\_head, hs)$ — splits the last dim $C$ into $n\\_head\\times hs$ with
$hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$:

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs).
$$

$$\\textbf{Move heads dimension next to batch}\\\\[4pt]$$

$.transpose(1, 2)$ — swaps the $T$ and $n\\_head$ axes:

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs).
$$
    `,
  },

  q_k_product_notes: {
    title: "Q @ K Product",
    md: `

$$\\textbf{Explanation of Scale dot-product attention}\\\\[4pt]$$

$q, k$ have shape $(B, n\\_head, T, hs)$ (queries/keys per head).$\\\\[4pt]$

$k.\\mathrm{transpose}(-2, -1) \\;\\rightarrow\\; (B, n\\_head, hs, T)$ swap the last two axes of keys, so each query can dot with every key at all time steps.$\\\\[4pt]$

\\@ is batched matrix multiply:$\\\\[4pt]$

$(B, n\\_head, T, hs)\\ @\\ (B, n\\_head, hs, T) \\;\\rightarrow\\; (B, n\\_head, T, T)\\\\[4pt]$

Result: for each head, a $T \\times T$ matrix of raw attention scores (one row per query position, one column per key position).$\\\\[4pt]$

$\\mathtt{* (1.0 / \\sqrt{hs})}$ (with $hs$ = head size). This is the scale from “Scaled Dot-Product Attention.”$\\\\[4pt]$

$$\\textbf{Scaled dot-product attention logits in math notation}\\\\[4pt]$$

Given
$$
q, k \\in \\mathbb{R}^{B \\times n\\_head \\times T \\times hs},
$$
the attention logits are computed as
$$
\\mathrm{att} \\;=\\; \\bigl(q \\; k^\\top\\bigr)\\;\\frac{1}{\\sqrt{hs}},
$$
where the transpose is over the last two axes of $k$:
$$
k^\\top \\in \\mathbb{R}^{B \\times n\\_head \\times hs \\times T},
\\qquad
q\\,k^\\top \\in \\mathbb{R}^{B \\times n\\_head \\times T \\times T}.
$$

Elementwise, for batch $b$, head $h$, query position $t_q$, and key position $t_k$:
$$
\\mathrm{att}_{b,h,t_q,t_k}
\\;=\\;
\\frac{ q_{b,h,t_q,:}\\cdot k_{b,h,t_k,:} }{\\sqrt{hs}}.
$$

$$\\textbf{Why the $1/\\sqrt{hs}$ scale}\\\\[4pt]$$

If the components of $q$ and $k$ are roughly zero-mean, unit-variance, then
$$
\\operatorname{Var}\\!\\bigl(q_{b,h,t_q,:}\\cdot k_{b,h,t_k,:}\\bigr) \\;\\propto\\; hs.
$$
Dividing by $\\sqrt{hs}$ keeps the logits' variance approximately constant (≈1), which helps prevent softmax becomes overly peaky (saturation) and stabilizes gradients.

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

$$\\textbf{What's happening step by step}\\\\[8pt]$$
$$\\textbf{1. What $\\mathtt{self.mask}$ is}\\\\[4pt]$$

Earlier we had:
$$
\\mathtt{self.register\\_buffer("mask",\\ torch.tril(torch.ones(1024,\\ 1024)).unsqueeze(0).unsqueeze(0))}
$$
[torch.tril](#info:torch_tril_notes)([torch.ones](#info:torch_ones_notes)(1024,1024)) makes a **lower-triangular** matrix of size $1024 \\times 1024$ with $1$s on/under the main diagonal and $0$s above it (future positions).  
[unsqueeze(0)](#info:unsqueeze_notes).unsqueeze(0) turns it into shape $(1,\\ 1,\\ 1024,\\ 1024)$ so it can **broadcast** across batch and heads.  
Stored as a **buffer** so it moves with the module to GPU/CPU and is saved, but not trained.

&nbsp;

$$\\textbf{2. Slicing to current length}\\\\[4pt]$$
$$
\\mathtt{self.mask[:, :, :T, :T]} \\;\\to\\; (1,\\ 1,\\ T,\\ T)
$$
matching the current sequence length.

&nbsp;

$$\\textbf{3. Build a boolean mask of disallowed links}\\\\[4pt]$$
$$
\\mathtt{==\\ 0}
$$
converts the \\(0/1\\) matrix into a boolean mask: **True** where the entry is $0$ (i.e., future positions), **False** otherwise.

&nbsp;

$$\\textbf{4. $\\mathtt{masked\\_fill}$ on the logits}\\\\[4pt]$$

$\\text{att}$ has shape $(B,\\ n\\_head,\\ T,\\ T)$ and holds **raw attention logits** $\\bigl(q\\,k^\\top/\\sqrt{hs}\\bigr)$.  
$\\mathtt{.masked\\_fill(bool\\_mask,\\ -\\infty)}$ writes $-\\infty$ into $\\text{att}$ **where the mask is True** (i.e., where attending would look **ahead**).  
Broadcasting lets a $(1,\\ 1,\\ T,\\ T)$ mask apply to all batches and heads.

&nbsp;

$$\\textbf{5. Why $-inf$?}\\\\[4pt]$$

Next you do $\\mathtt{att = F.softmax(att,\\ dim=-1)}$.  
$\\exp(-\\infty) = 0$,
so masked positions get **exactly zero probability**, and the softmax **renormalizes only over allowed (past/self) positions**.  
If you zeroed *after* softmax, probabilities would still be influenced by future tokens during normalization. **Masking before softmax** fixes that.

&nbsp;

---

&nbsp;

$$\\textbf{Intuition (per token row)}\\\\[4pt]$$

For query position $t$, this sets
$$
\\text{att}_{t,j} = -\\infty \\quad \\text{for } j > t,
$$
so after softmax, the row can only distribute probability over keys $j \\le t$ (itself and earlier tokens). That’s exactly **causal self-attention**.

&nbsp;

$$\\textbf{Notes}\\\\[4pt]$$

$-$ This mask enforces **causality**; it does **not** handle padding on its own. If you have padded tokens, you typically *also* add a padding mask (another $\\mathtt{masked\\_fill}$) so padding positions get $-\\infty$ too.$\\\\[4pt]$
$-$ Using a big negative constant (e.g. $-10^9$) is common; $-\\infty$ is cleaner and numerically robust in PyTorch softmax.

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
$$\\textbf{F.softmax explanation}\\\\[8pt]$$

$1.$ Converts the raw attention scores in $\\mathtt{att}$ into probabilities along the last dimension (keys dimension).$\\\\[4pt]$

$2.$ For each query token (and each head), the resulting row tells how much attention that query token should pay to
$\\text{each allowed previous token — higher value means more attention, zero means ignore. (e.g., masked future/padding)}\\\\[4pt]$

$3.$ Each row sums to 1, so the weights are comparable and interpretable.$\\\\[4pt]$

$4.$ The output has the same shape as the input; only the values are rescaled to valid probabilities.$\\\\[4pt]$

$5.$ These probabilities are then used to take a weighted average of the value vectors for that query token.$\\\\[4pt]$

&nbsp;

---

&nbsp;

$$\\textbf{Math Intepretation}\\\\[8pt]$$

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
\\text{Notes: }\\\\[4pt]
\\mathtt{dim=-1}\\;\\text{means “over the last axis” (the keys dimension).}\\\\[4pt]
\\mathtt{F.softmax}\\;\\text{is numerically stable (effectively subtracts a row max).}
$$

    `,
  },

  att_v_product_notes: {
    title: "att @ v",
    md: `

$$\\textbf{Explanation and Intuition}\\\\[8pt]$$

1\\. $\\mathtt{att}\\ \\text{holds, for each token (and each head), how much attention to pay to every allowed token in the sequence.}$

&nbsp;

2\\. $\\mathtt{v}\\ \\text{holds the content for each token which are worth copying if you attend to me.}$

&nbsp;

$\\text{Intuition: the operation says for each token, mix together other tokens' information according to those attention weights.}$

&nbsp;

$\\text{Tokens with higher weights contribute more; tokens with zero weight do not contribute.}$

&nbsp;

$\\text{The result }\\mathtt{y}\\text{ is a context-aware summary for each token (per head) that pulls in the most relevant information.}$

&nbsp;

---

&nbsp;

$$\\textbf{Math Intepretation}\\\\[8pt]$$

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
\\text{The value vectors at positions } 1,\\ 2,\\ 3 \\text{ are } v_1,\\ v_2,\\ v_3\\; (\\text{each is a length } hs \\text{ vector}).\\\\[6pt]
\\text{Then the output vector for token } t \\text{ is: } 0.5\\,v_1 + 0.3\\,v_2 + 0.2\\,v_3\\; \\text{— a single vector (length } hs \\text{) that mixes information using that row's weights.}
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
\\textbf{transpose(1,\\ 2)}: \\text{you currently have per-head outputs arranged as } (B,\\ n\\_head,\\ T,\\ hs).\\\\[6pt]
\\text{ This swap moves }\\textbf{time}\\text{ next to }\\textbf{batch}\\text{ → } (B,\\ T,\\ n\\_head,\\ h\\_s),\\ \\text{ so for each token you can gather all heads together.}\\\\[12pt]
\\textbf{contiguous()}: \\text{after a transpose, the tensor's memory is strided (non-contiguous). } \\mathtt{view}\\ \\text{ needs a contiguous layout,}\\\\[6pt]
\\text{ so this makes a compact copy if necessary.}\\\\[12pt]
\\textbf{view(B,\\ T,\\ C)}: \\text{finally, }\\textbf{stitch all heads back together}\\text{ by flattening } (n\\_head,\\ hs) \\text{ into the model dimension } \\mathtt{C = n\\_head * hs,}\\\\[6pt]
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
