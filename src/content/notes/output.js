// Notes: LM head and the loss.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  lm_head_notes: {
    en: {
      title: "lm_head",
      md: `
**\`lm_head\` gives every word in the vocabulary a score (a logit) at every position.**

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # 768 → 50304
logits = self.lm_head(x)                                   # x comes from ln_f
\`\`\`

## The weight matrix: one row per word

\`lm_head.weight\` has shape **(50304, 768)**: one row per vocabulary word, each row a 768-dim vector. Because of [weight tying](#hover:weight_tying_notes), this is the very same matrix as the token embedding table \`tok_emb.weight\`.

## One logit = one dot product

For each position, \`lm_head\` takes the dot product of $x$ (the output of \`ln_f\`) with every row:

\`\`\`text
logits[word i] = x · W[i]  =  Σ_j  x[j] × W[i][j]      (i = 0 … 50303, j = 0 … 767)
\`\`\`

The more a word's vector points in the same direction as $x$, the bigger the dot product and the higher its score. So in effect \`lm_head\` asks: **which word does $x$ look most like?**

## Shapes

\`\`\`text
x:       (B, T, 768)
W:       (50304, 768)
logits:  (B, T, 50304)      ← a score for every word, at every position
\`\`\`

**Every position** gets its own 50304 scores. The logits at position t predict token t+1.

- **During training,** all T sets of logits go into the [cross-entropy](#hover:cross_entropy_notes) loss, each compared with its true next token. This is done for all positions of the whole batch in one batched matmul.
- **During generation,** only the last position's logits are used to pick the next token.

> [Why does a dot product with the word vectors predict the next token? Deeper insights](#hover:lm_head_insights_notes)

## Details

- **Parameters:** 768 × 50304 = 38,633,472, about 38.6M, roughly 30% of GPT-2 small (about 124M). With weight tying they are stored only once.
- **Why 50304 and not GPT-2's 50257?** This is a nanoGPT trick: the vocabulary is padded up to a multiple of 64 (50304 = 64 × 786), which makes GPU matmuls faster. The 47 extra "words" never appear in the training data, so training pushes their logits down.
- **Why \`bias=False\`?**
  1. *Weight tying.* The output weights are the input embedding table (\`self.lm_head.weight = self.tok_emb.weight\`). The embedding has no bias, and a separate output bias isn't needed.
  2. *Little benefit.* A bias would add \`vocab_size\` more parameters for negligible gain, so the logits are pure dot products.
`,
    },
    zh: {
      title: "lm_head",
      md: `
**\`lm_head\` 在每个位置，给词表里的每个词打一个分（logit）。**

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # 768 → 50304
logits = self.lm_head(x)                                   # x 来自 ln_f
\`\`\`

## 权重矩阵：每个词一行

\`lm_head.weight\` 的形状是 **(50304, 768)**：词表里每个词一行，每行是一个 768 维向量。因为 [weight tying](#hover:weight_tying_notes)，它和 token embedding 表 \`tok_emb.weight\` 是**同一个**矩阵。

## 一个 logit = 一次点积

对每个位置，\`lm_head\` 把 $x$（\`ln_f\` 的输出）和每一行做点积：

\`\`\`text
logits[word i] = x · W[i]  =  Σ_j  x[j] × W[i][j]      (i = 0 … 50303, j = 0 … 767)
\`\`\`

一个词的向量和 $x$ 方向越接近，点积越大，分数就越高。所以 \`lm_head\` 本质上是在问：**$x$ 最像哪个词？**

## 形状

\`\`\`text
x:       (B, T, 768)
W:       (50304, 768)
logits:  (B, T, 50304)      # 每个位置，对每个词都有一个分数
\`\`\`

**每个位置**都有自己的 50304 个分数。位置 t 的 logits 用来预测第 t+1 个 token。

- **训练时**，这 T 组 logits 全部进入 [cross-entropy](#hover:cross_entropy_notes) loss，各自和真实的下一个 token 比较。整个 batch 所有位置一次批量矩阵乘法算完。
- **生成时**，只用最后一个位置的 logits 来选下一个 token。

> [为什么和词向量做点积就能预测下一个 token？更深入的解释](#hover:lm_head_insights_notes)

## 细节

- **参数量：** 768 × 50304 = 38,633,472，约 3860 万，大约占 GPT-2 small（约 1.24 亿）的三成。因为 weight tying，这些参数只存一份。
- **为什么是 50304 而不是 GPT-2 原本的 50257？** 这是 nanoGPT 的做法：把词表补到 64 的倍数（50304 = 64 × 786），GPU 矩阵乘法会更快。多出来的 47 个"词"永远不会出现在训练数据里，训练会把它们的 logits 压低。
- **为什么 \`bias=False\`？**
  1. *Weight tying。* 输出权重就是输入的 embedding 表（\`self.lm_head.weight = self.tok_emb.weight\`）。embedding 没有 bias，也不需要单独的输出 bias。
  2. *收益很小。* bias 会多出 \`vocab_size\` 个参数，几乎没有好处，所以 logits 就是纯点积。
`,
    },
  },

  llama_lm_head_notes: {
    en: {
      title: "LLaMA LM head: not tied to the embedding",
      md: `
**LLaMA's LM head does the same job as [GPT-2's](#info:lm_head_notes): it takes the dot product of the final vector with one row per token, which gives every token a score. The difference is that its weight matrix is its own, not shared with the [token embedding](#info:llama_embedding_notes).**

## In the code

\`\`\`python
# GPT-2: one matrix, used at both ends
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight

# LLaMA 3 8B: the same layer, without the tying line
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # weight (128256, 4096)
logits = self.lm_head(x)                                   # (B, T, 4096) -> (B, T, 128256)
\`\`\`

The LM head's weight and the embedding table have the same shape, $(V, C) = (128256, 4096)$, about 525M parameters each. In LLaMA they are two separate parameters, each trained on its own.

## Why not tie: the two ends do different jobs

Row $i$ has a different job in each matrix:

- **Token embedding:** what token $i$ means when it is read as **context**, for the tokens after it to use.
- **LM head:** it **scores** token $i$: the dot product of the final vector with this row is token $i

The two jobs are related (tokens with similar meanings tend to follow similar contexts), but they are not the same. Tying forces one vector to do both; untying lets each end learn the vector that suits its own job. (The [weight tying](#info:weight_tying_notes) note shows how a tied row gets gradients from both ends.)

## Why GPT-2 tied: what the saving is worth

Tying removes one $V \\times C$ matrix. How much that saves depends on the size of the rest of the model:

| Model | Tied? | One $V \\times C$ matrix | Saved by tying |
|---|---|---|---|
| GPT-2 small | yes | 38.6M | 24% (124M instead of 163M) |
| LLaMA 3.2 1B | yes | 263M | 18% (1.24B instead of 1.50B) |
| LLaMA 3.2 3B | yes | 394M | 11% (3.2B instead of 3.6B) |
| LLaMA 3 8B | no | 525M | would be 6.5% (7.5B instead of 8.0B) |

Each percentage is one matrix divided by the parameter count without tying, e.g. 38.6M / 163M ≈ 24%.

- **Small models:** the embedding is a large part of the model, so tying saves up to a quarter of the parameters. GPT-2 ties, and so do LLaMA 3.2 1B and 3B, the small models meant to run on phones and other devices.
- **Large models:** almost all the parameters are in the transformer blocks (about 7.0B of LLaMA 3 8B's 8.0B). Saving 6.5% is not worth making both ends share one vector, so LLaMA 1, 2 and 3 at 7B/8B and above do not tie.

Other model families make the same choice by size; see "Recent developments" in the [weight tying](#info:weight_tying_notes) note.

## A side effect on the gradients

The LM head's softmax has an output for every token, so **every row of the LM head** gets a gradient at every step, as in GPT-2.

Without tying, none of that reaches the embedding table. An embedding row gets a gradient only when its token appears in the input batch. So for a rare token, the output row is trained every step, but the input row only now and then (see [LLaMA token embedding](#info:llama_embedding_notes)).

## References

1. **Using the Output Embedding to Improve Language Models.** Press & Wolf, 2016. [arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)
2. **The Llama 3 Herd of Models.** Llama Team, Meta, 2024. [arXiv:2407.21783 ↗](https://arxiv.org/abs/2407.21783)
s score. So this row has to learn what the final vector looks like when the next token should be $i$.

The two jobs are related (tokens with similar meanings tend to follow similar contexts), but they are not the same. Tying forces one vector to do both; untying lets each end learn the vector that suits its own job. (The [weight tying](#info:weight_tying_notes) note shows how a tied row gets gradients from both ends.)

## Why GPT-2 tied: what the saving is worth

Tying removes one $V \\times C$ matrix. How much that saves depends on the size of the rest of the model:

| Model | Tied? | One $V \\times C$ matrix | Saved by tying |
|---|---|---|---|
| GPT-2 small | yes | 38.6M | 24% (124M instead of 163M) |
| LLaMA 3.2 1B | yes | 263M | 18% (1.24B instead of 1.50B) |
| LLaMA 3.2 3B | yes | 394M | 11% (3.2B instead of 3.6B) |
| LLaMA 3 8B | no | 525M | would be 6.5% (7.5B instead of 8.0B) |

Each percentage is one matrix divided by the parameter count without tying, e.g. 38.6M / 163M ≈ 24%.

- **Small models:** the embedding is a large part of the model, so tying saves up to a quarter of the parameters. GPT-2 ties, and so do LLaMA 3.2 1B and 3B, the small models meant to run on phones and other devices.
- **Large models:** almost all the parameters are in the transformer blocks (about 7.0B of LLaMA 3 8B's 8.0B). Saving 6.5% is not worth making both ends share one vector, so LLaMA 1, 2 and 3 at 7B/8B and above do not tie.

Other model families make the same choice by size; see "Recent developments" in the [weight tying](#info:weight_tying_notes) note.

## A side effect on the gradients

The LM head's softmax has an output for every token, so **every row of the LM head** gets a gradient at every step, as in GPT-2.

Without tying, none of that reaches the embedding table. An embedding row gets a gradient only when its token appears in the input batch. So for a rare token, the output row is trained every step, but the input row only now and then (see [LLaMA token embedding](#info:llama_embedding_notes)).

## References

1. **Using the Output Embedding to Improve Language Models.** Press & Wolf, 2016. [arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)
2. **The Llama 3 Herd of Models.** Llama Team, Meta, 2024. [arXiv:2407.21783 ↗](https://arxiv.org/abs/2407.21783)
`,
    },
    zh: {
      title: "LLaMA 的 LM head：不和 embedding 共享权重",
      md: `
**LLaMA 的 LM head 做的事和 [GPT-2 的](#info:lm_head_notes)一样：用最后的向量和每个 token 的一行做点积，给每个 token 一个分数。区别是它的权重矩阵是自己的，不和 [token embedding](#info:llama_embedding_notes) 共享。**

## 在代码里

\`\`\`python
# GPT-2：一个矩阵，两端都用
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight

# LLaMA 3 8B：同样的层，只是没有共享那一行
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # weight (128256, 4096)
logits = self.lm_head(x)                                   # (B, T, 4096) -> (B, T, 128256)
\`\`\`

LM head 的权重和 embedding 表形状一样，都是 $(V, C) = (128256, 4096)$，各约 5.25 亿参数。在 LLaMA 里它们是两套独立的参数，各自训练。

## 为什么不共享：两端做的事不一样

第 $i$ 行在两个矩阵里的作用不同：

- **token embedding：** token $i$ **作为上下文**被读进来时是什么意思，供它后面的 token 使用。
- **LM head：** 用来给 token $i$ **打分**：最后的向量和这一行做点积，得到的就是 token $i$ 的分数。所以这一行要学的是：当下一个 token 应该是 $i$ 时，最后的向量是什么样子。

这两件事相关（意思相近的词，往往出现在相似的上下文之后），但并不相同。共享权重，就是强迫一个向量同时做两件事；不共享，每一端都能学到最适合自己的向量。（[weight tying](#info:weight_tying_notes) note 里讲了共享时，一行怎样同时从两端拿到梯度。）

## GPT-2 为什么共享：省下的参数值多少

共享省掉的是一个 $V \\times C$ 矩阵。它值多少，要看模型其余部分有多大：

| 模型 | 共享？ | 一个 $V \\times C$ 矩阵 | 共享省下的比例 |
|---|---|---|---|
| GPT-2 small | 是 | 3860 万 | 24%（1.24 亿，不共享是 1.63 亿） |
| LLaMA 3.2 1B | 是 | 2.63 亿 | 18%（12.4 亿，不共享是 15.0 亿） |
| LLaMA 3.2 3B | 是 | 3.94 亿 | 11%（32 亿，不共享是 36 亿） |
| LLaMA 3 8B | 否 | 5.25 亿 | 共享的话能省 6.5%（75 亿，现在是 80 亿） |

比例 = 一个矩阵的参数量 ÷ 不共享时的总参数量，例如 3860 万 ÷ 1.63 亿 ≈ 24%。

- **小模型：** embedding 占了模型的很大一块，共享能省下最多四分之一的参数。GPT-2 共享；LLaMA 3.2 1B 和 3B（面向手机等设备的小模型）也共享。
- **大模型：** 参数几乎都在 transformer block 里（LLaMA 3 8B 的 80 亿里约 70 亿）。为了省 6.5% 而让两端共用一个向量，不划算，所以 LLaMA 1、2、3 的 7B/8B 及更大的模型都不共享。

其他模型家族也是按大小来选，见 [weight tying](#info:weight_tying_notes) note 里的 "🆕 近期发展" 一节。

## 对梯度的一个影响

LM head 的 softmax 对每个 token 都有输出，所以 **LM head 的每一行**每一步都会拿到梯度，这和 GPT-2 一样。

不共享时，这些梯度都不会传到 embedding 表。embedding 的一行只有在它的 token 出现在这个 batch 的输入里时才有梯度。所以对一个很少出现的 token，输出端那一行每一步都在训练，输入端那一行却只是偶尔被训练（见 [LLaMA 的 token embedding](#info:llama_embedding_notes)）。

## 参考资料

1. **Using the Output Embedding to Improve Language Models.** Press & Wolf，2016。[arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)
2. **The Llama 3 Herd of Models.** Llama Team，Meta，2024。[arXiv:2407.21783 ↗](https://arxiv.org/abs/2407.21783)
`,
    },
  },

  lm_head_insights_notes: {
    en: {
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
    zh: {
      title: "lm_head 深入解释",
      md: `
**为什么和输出 embedding 做点积，就能预测下一个 token？**

## 设定

- $x \\in \\mathbb{R}^{n\\_embd}$ 是某个位置上 token 的最终 hidden state。
- $e_v \\in \\mathbb{R}^{n\\_embd}$ 是词表中 token $v$ 的输出权重向量（因为 weight tying，它同时也是 $v$ 的输入 embedding）。

首先要理解训练达成了什么：

1. **hidden state $x$** 汇总了当前位置及之前所有 token 的上下文，同时也编码了下一个 token 应该具备的特征。

   例如输入是 "The dogs that barked loudly"。训练之后，第 5 个位置的 hidden state 从位置 1 到 5 的 token 中学到了"主语是复数"这个上下文，同时也编码了"下一个 token 很可能是动词"这个特征。

2. **$e_v$ 是 token $v$ 的一个全局分类器权重向量：** 应该选 $v$ 的上下文，会和这个方向对齐。它与上下文无关（不会随时变化），是从整个语料中学到的 token $v$ 的一般含义。

3. **cross-entropy loss 同时塑造两边。** 它推动：
   - hidden state $x$ 去对齐正确 token 的 $e_v$；
   - $e_v$ 朝着"最能把 $v$ 是正确答案的上下文和其他上下文区分开"的方向移动。久而久之，$v$ 是正确答案的那些上下文，就会落在 $e_v$ 附近。

记住这几点，我们继续。

## 点积形式

LM head 用 $x$ 和 $e_v$ 的点积，给 token $v$ 一个 logit $\\ell_v$：

$$
\\ell_v = x \\cdot e_v
$$

## 余弦形式

根据 $\\mathbb{R}^{d}$ 中余弦的定义，还可以写成：

$$
\\ell_v = x \\cdot e_v = \\lVert x \\rVert \\, \\lVert e_v \\rVert \\, \\cos\\theta(x, e_v)
$$

### 三个因子

所以每个 token 的 logit $\\ell_v$ 可以拆成三部分：

1. $x$ 的大小；
2. $e_v$ 的大小；
3. $e_v$ 的方向，它决定了 $x$ 和 $e_v$ 之间随上下文变化的余弦相似度。

## 最后一个 LayerNorm 的作用

有了最后的 layer norm（\`ln_f\`），$x$ 的大小几乎是常数，只在很小的范围内变化。于是另外两个因子，$e_v$ 的大小和余弦相似度，成了决定 logit 的主要因素。

> 粗略地说，可以把因子 3（**$e_v$ 的方向**）看作捕捉上下文，把因子 2（**$e_v$ 的大小**）看作捕捉这个 token 的出现频率。

- **$e_v$ 的方向：** $x$ 的方向越接近 $e_v$，第三个因子越大，对 logit 的贡献也越大。从语义上说，这意味着 token $v$ 和 hidden state $x$ 所包含的上下文更相符，所以会以较高的概率被预测出来。
- **$e_v$ 的大小：** 大语言模型（LLM）的 token embedding 矩阵与 token 频率相关：常见 token 的 embedding 范数一般更大，因此这些 token 的预测概率也更高。

## 为什么常见 token 的范数更大？

下面是一个直观的解释。

训练时我们最小化 loss $\\mathcal{L}$，参数按下式更新：

$$
\\Delta\\theta = -\\eta \\, \\nabla_\\theta \\mathcal{L}
$$

和某个 token $v$ 绑定的参数（它的向量 $e_v$，如果有 bias 还包括 $b_v$），只在这个 token 参与 loss 的那些步里得到梯度。

### LM head 向量的梯度（无 bias 的 head）

$$
\\ell = x E^\\top, \\quad p = \\mathrm{softmax}(\\ell), \\quad \\frac{\\partial \\mathcal{L}}{\\partial e_v} = \\big(p_v(x) - \\mathbf{1}\\{y = v\\}\\big) \\, x
$$

在整个数据集上：

$$
\\nabla_{e_v} \\mathcal{L} = \\sum_{t} \\big(p_v(x_t) - \\mathbf{1}\\{y_t = v\\}\\big) \\, x_t
$$

### 常见 token "得到更多梯度"

1. 如果 token $v$ 作为目标出现了 $C_v$ 次，就有 $C_v$ 个强项（$-\\mathbf{1}\\{y_t = v\\}$ 那部分）把 $e_v$ 拉向 $v$ 之前的 hidden state $x$。
2. 在其他位置，$e_v$ 通过 $p_v(x_t)$ 参与，当 $v$ 不是目标时，产生较小的"推开"项。
3. 经过大量样本，第一个效果是让 $e_v$ 的方向和 hidden state $x$ 对齐。
4. 方向基本对齐之后，增大 $e_v$ 的大小就成了一种高效的办法，能在许多次出现中一起抬高 $v$ 的 logit，相当于一个全局增益或先验。常见 token 提供的这类更新更多，所以范数往往更大。

> **经验法则：** 出现次数越多 $\\Rightarrow$ 梯度贡献越多（越大）$\\Rightarrow$ 这个 token 的参数被调整得越充分。

## 最后

- **token embedding $e_v$**（来自 \`wte.weight\`）与上下文无关。它捕捉的是 token 类型层面的倾向：从这个 token 在整个语料中出现的方式里，学到的大致、平均的语义和句法（比如 "dogs" 是复数、偏名词、常出现在 "are/were" 这类动词附近）。可以把它看作一个基准原型。
- **它在某句话里的具体含义**，也就是"上下文中的 token"，存在 hidden state $x_t$ 里，这是 token 经过位置编码 + self-attention + MLP 之后得到的。这就是 LM head 用来选下一个 token 的**上下文化表示**。

> 所以 embedding 承载的是**一般含义**，而不是**随时变化的上下文**。上下文由 transformer 注入，体现在 $x_t$ 中，而不是 $e_v$ 中。
`,
    },
  },

  cross_entropy_notes: {
    en: {
      title: "Cross Entropy",
      md: `
**Cross-entropy loss** measures how much probability the model gave to the correct answer. In GPT-2 it is computed at every position, for the true next token, and then averaged into one number.

Docs: [CrossEntropy ↗](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

## A running example

Take a tiny vocabulary of 3 words. At one position the model outputs these logits, and the correct next word is word 2:

\`\`\`text
logits z = [0.5, 0, 0]
correct answer t = 2
\`\`\`

## Step 1: logits → probabilities (softmax)

Softmax exponentiates every logit and divides by the total, so the results are positive and sum to 1:

\`\`\`text
exp(0.5) = 1.6487
exp(0)   = 1
exp(0)   = 1
total S  = 1.6487 + 1 + 1 = 3.6487

p = [1.6487 / S, 1 / S, 1 / S]
  = [0.452, 0.274, 0.274]
\`\`\`

## Step 2: loss = −log(probability of the correct answer)

$$
\\text{loss} = -\\log p_t
$$

Only the correct word's probability enters the formula. Here $t = 2$:

\`\`\`text
p[2] = 0.27407
loss = −ln(0.27407) = 1.2944
\`\`\`

The log is the natural log (base $e$), the same one PyTorch uses.

The wrong words still matter indirectly: softmax divides by the total $S$, so a higher logit for a wrong word makes $p_t$ smaller and the loss larger.

### The same formula with a one-hot target

Write the correct answer as a one-hot vector $y$: 1 at the correct word, 0 everywhere else. Here $y = [0, 0, 1]$. Then

$$
\\text{loss} = -\\sum_i y_i \\log p_i
$$

Every term with $y_i = 0$ vanishes, leaving only $-\\log p_t$, the formula above.

> In practice (e.g. GPT-2), the target is passed as the class index (here 2), not as a one-hot vector. The result is the same, but it uses far less memory and compute.

## What the numbers mean

| Probability of the correct answer | loss |
|---|---|
| 1.0 (certain and right) | 0 |
| 0.5 | 0.693 |
| 1/3 (uniform guess over 3 words) | 1.099 |
| 0.274 (the example) | 1.294 |
| 0.01 | 4.605 |

The example's loss (1.294) is worse than a uniform guess (1.099), because the model gave its highest logit to word 0, not to the correct word 2.

A model that guesses uniformly over $V$ words has $p_t = 1/V$, so its loss is $\\ln V$. For GPT-2, $\\ln 50{,}304 \\approx 10.83$. A model whose initial logits are all small (as with GPT-2's $\\mathcal{N}(0, 0.02)$ init) starts training near this value.

## Computing it directly from the logits

Substitute the softmax into the loss:

$$
\\text{loss} = -\\log \\frac{e^{z_t}}{S} = \\log S - z_t, \\qquad S = \\sum_j e^{z_j}
$$

For the example: $\\ln 3.6487 - 0 = 1.2944$, the same number.

PyTorch's \`F.cross_entropy\` takes the raw logits and computes the loss in this form (log-sum-exp), without first computing $p$ and then taking its log. A tiny probability can round to 0 in floating point, and $\\log 0 = -\\infty$; the log-sum-exp form avoids that. This is why the model passes logits, not probabilities, to \`F.cross_entropy\`.

## The gradient with respect to the logits: $p - y$

Backpropagation starts from this gradient, so it is worth deriving. Start from $\\text{loss} = \\log S - z_t$ and differentiate each term with respect to one logit $z_i$.

**First term, $\\log S$.** By the chain rule, the derivative of $\\log$ is $1/S$, and the only term of $S$ that contains $z_i$ is $e^{z_i}$:

$$
\\frac{\\partial \\log S}{\\partial z_i} = \\frac{1}{S} \\cdot e^{z_i} = p_i
$$

That is exactly the definition of softmax.

**Second term, $-z_t$.** It depends only on the correct logit:

$$
\\frac{\\partial (-z_t)}{\\partial z_i} = \\begin{cases} -1 & i = t \\\\ 0 & i \\ne t \\end{cases} \\;=\\; -y_i
$$

**Add them:**

$$
\\frac{\\partial\\, \\text{loss}}{\\partial z_i} = p_i - y_i
$$

For the example:

\`\`\`text
p     = [0.452, 0.274,  0.274]
y     = [0,     0,      1    ]
p − y = [0.452, 0.274, -0.726]
\`\`\`

Checked in PyTorch two ways: with autograd, and by nudging each logit by 0.001 and measuring how much the loss changes.

\`\`\`text
autograd z.grad                  = [ 0.4519,  0.2741, -0.7259]
nudge logit 0: change / 0.001    =   0.4518
nudge logit 1: change / 0.001    =   0.2741
nudge logit 2: change / 0.001    =  -0.7260
\`\`\`

What the gradient does (an update subtracts the gradient):

- **Wrong words** get $+p_i$, so their logits go **down**. The more probability a wrong word took, the harder it is pushed down.
- **The correct word** gets $p_t - 1$, which is always negative, so its logit goes **up**. The further $p_t$ is from 1, the harder it is pushed up.
- When $p = y$ (the correct word has probability 1), the gradient is 0 and nothing changes.

The gradient is this simple because the $\\exp$ in softmax and the $\\log$ in cross-entropy cancel when differentiated. This is another reason PyTorch fuses softmax and cross-entropy into one function.

## In GPT-2: from $(B, T, V)$ logits to one number

The model computes the loss in one line:

\`\`\`python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
\`\`\`

### Targets are the input shifted by one

Each position's correct answer is the token that comes right after it. \`get_batch\` builds \`y\` this way:

\`\`\`text
a slice of tokens:  [464, 3290, 3332, 319, 262]     (ids are made up)

idx (x)     = [464, 3290, 3332, 319]     fed to the model
targets (y) = [3290, 3332, 319, 262]     the next token at each position
\`\`\`

So the shapes are:

- **Logits** (raw scores before softmax): $(B, T, V)$
- **Targets** (true token ids, not one-hot): $(B, T)$

where $B$ is the batch size, $T$ the sequence length and $V$ the vocabulary size.

### Flattening: \`.view(-1, V)\` and \`.view(-1)\`

\`F.cross_entropy\` expects:

- **Input:** a 2D tensor $(N, C)$: $N$ samples, each with scores for $C$ classes.
- **Target:** a 1D tensor $(N,)$ of integer class indices.

It knows nothing about $B$ and $T$, so both are merged into one dimension:

- \`logits.view(-1, V)\`: $(B, T, V) \\to (B \\times T, V)$. Each row is one position $(b, t)$: the model's scores over the vocabulary there.
- \`targets.view(-1)\`: $(B, T) \\to (B \\times T,)$. Each entry is the correct next-token id for that row.

> Every position in the batch becomes an independent classification problem: $B \\times T$ problems in total.

### Averaging over all positions

Each of the $B \\times T$ rows gets its own loss, $-\\log p_t$. By default (\`reduction='mean'\`) they are averaged into one number. Checked in PyTorch with $B = 2$, $T = 4$, $V = 5$ and random logits:

\`\`\`text
logits.view(-1, V).shape = (8, 5)
targets.view(-1).shape   = (8,)

loss at each of the 8 positions:
[1.916, 1.822, 2.035, 3.344, 2.511, 1.021, 2.102, 0.378]

mean                     = 1.8913
F.cross_entropy result   = 1.8913   ✓
\`\`\`

Because the loss is a mean, each position's logits gradient is also divided by $B \\times T$:

$$
\\frac{\\partial\\, \\text{loss}}{\\partial\\, \\text{logits}[b, t]} = \\frac{p - y}{B \\times T}
$$

So the size of the gradient does not grow with the batch, and the learning rate does not have to change when the batch size does.

> One forward pass over a sequence of length $T$ yields $T$ next-token problems at once: every token is a training signal. The [causal mask](#info:causal_mask_notes) is what makes this valid: position $t$ cannot see tokens after it, so it cannot peek at its own answer.

## References

1. **PyTorch docs**

   CrossEntropyLoss (multiclass softmax CE; targets are class indices, not one-hot).

   [PyTorch Docs ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

2. **CS231n Softmax Classifier**

   Notes on the derivation of softmax + cross-entropy and its gradients.

   [CS231n Softmax Classifier ↗](https://cs231n.github.io/linear-classify/)
`,
    },
    zh: {
      title: "Cross Entropy（交叉熵）",
      md: `
**Cross-entropy loss** 衡量模型给正确答案分了多少概率。GPT-2 在每个位置上，对真实的下一个 token 算一次，再把所有位置平均成一个数。

文档：[CrossEntropy ↗](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

## 贯穿全文的例子

取一个只有 3 个词的词表。某个位置上模型输出的 logits 如下，正确的下一个词是 word 2：

\`\`\`text
logits z = [0.5, 0, 0]
正确答案 t = 2
\`\`\`

## 第 1 步：logits → 概率（softmax）

softmax 对每个 logit 取 exp，再除以总和，得到的结果都是正数、加起来等于 1：

\`\`\`text
exp(0.5) = 1.6487
exp(0)   = 1
exp(0)   = 1
总和 S   = 1.6487 + 1 + 1 = 3.6487

p = [1.6487 / S, 1 / S, 1 / S]
  = [0.452, 0.274, 0.274]
\`\`\`

## 第 2 步：loss = −log(正确答案的概率)

$$
\\text{loss} = -\\log p_t
$$

公式里只出现正确答案的概率。这里 $t = 2$：

\`\`\`text
p[2] = 0.27407
loss = −ln(0.27407) = 1.2944
\`\`\`

这里的 log 是自然对数（底数 $e$），PyTorch 用的也是它。

错误的词也会间接起作用：softmax 要除以总和 $S$，所以错误词的 logit 越高，$p_t$ 就越小，loss 就越大。

### 用 one-hot 写同一个公式

把正确答案写成 one-hot 向量 $y$：正确的词是 1，其他都是 0。这里 $y = [0, 0, 1]$。于是

$$
\\text{loss} = -\\sum_i y_i \\log p_i
$$

$y_i = 0$ 的项全部消失，只剩 $-\\log p_t$，就是上面的公式。

> 实际中（比如 GPT-2），target 传的是类别索引（这里是 2），不是 one-hot 向量。结果一样，但省内存、省计算得多。

## 这些数字是什么意思

| 正确答案的概率 | loss |
|---|---|
| 1.0（完全确定且答对） | 0 |
| 0.5 | 0.693 |
| 1/3（3 个词里均匀乱猜） | 1.099 |
| 0.274（这个例子） | 1.294 |
| 0.01 | 4.605 |

这个例子的 loss（1.294）比均匀乱猜（1.099）还高：模型把最大的 logit 给了 word 0，而不是正确答案 word 2。

在 $V$ 个词里均匀乱猜时 $p_t = 1/V$，loss 就是 $\\ln V$。对 GPT-2 来说，$\\ln 50{,}304 \\approx 10.83$。如果模型初始的 logits 都很小（比如 GPT-2 用 $\\mathcal{N}(0, 0.02)$ 初始化），训练刚开始时 loss 就在这个值附近。

## 直接从 logits 算

把 softmax 代进 loss：

$$
\\text{loss} = -\\log \\frac{e^{z_t}}{S} = \\log S - z_t, \\qquad S = \\sum_j e^{z_j}
$$

代入例子：$\\ln 3.6487 - 0 = 1.2944$，结果一样。

PyTorch 的 \`F.cross_entropy\` 接收原始 logits，按这种形式（log-sum-exp）计算，不会先算出 $p$ 再取 log。很小的概率在浮点数里可能被舍入成 0，而 $\\log 0 = -\\infty$；log-sum-exp 的写法避开了这个问题。所以模型传给 \`F.cross_entropy\` 的是 logits，不是概率。

## 对 logits 的梯度：$p - y$

反向传播就是从这个梯度开始的，所以值得推一遍。从 $\\text{loss} = \\log S - z_t$ 出发，对某一个 logit $z_i$ 分别求两项的导数。

**第一项 $\\log S$。** 用链式法则：$\\log$ 的导数是 $1/S$；$S$ 里只有 $e^{z_i}$ 这一项含 $z_i$：

$$
\\frac{\\partial \\log S}{\\partial z_i} = \\frac{1}{S} \\cdot e^{z_i} = p_i
$$

结果正好是 softmax 的定义。

**第二项 $-z_t$。** 它只和正确答案的 logit 有关：

$$
\\frac{\\partial (-z_t)}{\\partial z_i} = \\begin{cases} -1 & i = t \\\\ 0 & i \\ne t \\end{cases} \\;=\\; -y_i
$$

**两项相加：**

$$
\\frac{\\partial\\, \\text{loss}}{\\partial z_i} = p_i - y_i
$$

代入例子：

\`\`\`text
p     = [0.452, 0.274,  0.274]
y     = [0,     0,      1    ]
p − y = [0.452, 0.274, -0.726]
\`\`\`

在 PyTorch 里用两种方法核对：一是 autograd，二是把每个 logit 加 0.001，看 loss 变化了多少。

\`\`\`text
autograd z.grad                  = [ 0.4519,  0.2741, -0.7259]
logit 0 加 0.001：变化 / 0.001   =   0.4518
logit 1 加 0.001：变化 / 0.001   =   0.2741
logit 2 加 0.001：变化 / 0.001   =  -0.7260
\`\`\`

这个梯度的作用（更新时减去梯度）：

- **错误的词**梯度是 $+p_i$，所以 logit **下降**。错误词占的概率越多，被压得越狠。
- **正确的词**梯度是 $p_t - 1$，总是负数，所以 logit **上升**。$p_t$ 离 1 越远，往上推得越狠。
- 当 $p = y$（正确答案概率为 1）时，梯度为 0，不再更新。

梯度这么简单，是因为 softmax 里的 $\\exp$ 和 cross-entropy 里的 $\\log$ 求导时互相抵消了。这也是 PyTorch 把 softmax 和 cross-entropy 合成一个函数的原因之一。

## 在 GPT-2 里：从 $(B, T, V)$ 的 logits 到一个数

模型用一行代码算 loss：

\`\`\`python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
\`\`\`

### targets 就是输入往后错一位

每个位置的正确答案是紧跟在它后面的那个 token。\`get_batch\` 就是这样构造 \`y\` 的：

\`\`\`text
一段 token：  [464, 3290, 3332, 319, 262]     （编号是随便写的）

idx (x)     = [464, 3290, 3332, 319]     喂给模型
targets (y) = [3290, 3332, 319, 262]     每个位置的下一个 token
\`\`\`

所以形状是：

- **Logits**（softmax 之前的原始分数）：$(B, T, V)$
- **Targets**（真实的 token 编号，不是 one-hot）：$(B, T)$

其中 $B$ 是 batch size，$T$ 是序列长度，$V$ 是词表大小。

### 展平：\`.view(-1, V)\` 和 \`.view(-1)\`

\`F.cross_entropy\` 要求：

- **Input：** 2 维张量 $(N, C)$：$N$ 个样本，每个样本对 $C$ 个类别打分。
- **Target：** 1 维张量 $(N,)$，里面是整数类别索引。

它不认识 $B$ 和 $T$，所以要把这两个维度合成一个：

- \`logits.view(-1, V)\`：$(B, T, V) \\to (B \\times T, V)$。每一行是一个位置 $(b, t)$：模型在这里对整个词表的打分。
- \`targets.view(-1)\`：$(B, T) \\to (B \\times T,)$。每一项是这一行对应的正确下一个 token 的编号。

> batch 里的每个位置都变成一道独立的分类题，一共 $B \\times T$ 道。

### 对所有位置取平均

$B \\times T$ 行各算出一个 loss，$-\\log p_t$。默认（\`reduction='mean'\`）把它们平均成一个数。用 $B = 2$、$T = 4$、$V = 5$ 和随机 logits 在 PyTorch 里验证：

\`\`\`text
logits.view(-1, V).shape = (8, 5)
targets.view(-1).shape   = (8,)

8 个位置各自的 loss：
[1.916, 1.822, 2.035, 3.344, 2.511, 1.021, 2.102, 0.378]

平均                     = 1.8913
F.cross_entropy 的结果   = 1.8913   ✓
\`\`\`

因为 loss 是平均值，每个位置的 logits 梯度也要除以 $B \\times T$：

$$
\\frac{\\partial\\, \\text{loss}}{\\partial\\, \\text{logits}[b, t]} = \\frac{p - y}{B \\times T}
$$

这样梯度的大小不会随 batch 变大而变大，换 batch size 时 learning rate 也不用跟着改。

> 一个长度为 $T$ 的序列，一次 forward 就得到 $T$ 道"预测下一个词"的题：每个 token 都是训练信号。之所以能这样做，靠的是 [causal mask](#info:causal_mask_notes)：位置 $t$ 看不到它后面的 token，也就偷看不到自己的答案。

## 参考资料

1. **PyTorch 文档**

   CrossEntropyLoss（多分类 softmax CE；targets 是类别索引，不是 one-hot）。

   [PyTorch Docs ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

2. **CS231n Softmax Classifier**

   讲解 softmax + cross-entropy 的推导及其梯度。

   [CS231n Softmax Classifier ↗](https://cs231n.github.io/linear-classify/)
`,
    },
  },

  gradient_loss_to_logits_proof_notes: {
    en: {
      title: "Gradient of loss to logits",
      md: `
**Proof that the gradient of softmax cross-entropy w.r.t. the logits is $p - y$.**

## Setup

- Logits: $z \\in \\mathbb{R}^C$
- Softmax: $p_i = \\dfrac{e^{z_i}}{\\sum_{k=1}^C e^{z_k}}$
- Target: $y$ (usually one-hot, but it can be any distribution, e.g. for label smoothing)
- Cross-entropy loss: $L = -\\sum_{i=1}^C y_i \\log p_i$

## Proof

Use the identity

$$
\\log p_i = z_i - \\log \\sum_{k=1}^C e^{z_k}
$$

(take the log of both sides of the softmax definition). Plug it into the loss:

$$
L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i \\left(z_i - \\log \\sum_{k=1}^C e^{z_k}\\right) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
$$

Because $y$ is a probability distribution, $\\sum_{i=1}^C y_i = 1$, so

$$
L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
$$

Differentiate w.r.t. $z_j$ (in the first sum only the $i = j$ term contains $z_j$; for the second term use the rule that the derivative of $\\log u$ is $u'/u$):

$$
\\frac{\\partial L}{\\partial z_j} = -y_j + \\frac{e^{z_j}}{\\sum_{k=1}^C e^{z_k}} = p_j - y_j
$$
`,
    },
    zh: {
      title: "loss 对 logits 的梯度",
      md: `
**证明：softmax cross-entropy 对 logits 的梯度是 $p - y$。**

## 设定

- Logits：$z \\in \\mathbb{R}^C$
- Softmax：$p_i = \\dfrac{e^{z_i}}{\\sum_{k=1}^C e^{z_k}}$
- 目标：$y$（通常是 one-hot，但也可以是任意概率分布，比如 label smoothing 时）
- Cross-entropy loss：$L = -\\sum_{i=1}^C y_i \\log p_i$

## 证明

利用恒等式

$$
\\log p_i = z_i - \\log \\sum_{k=1}^C e^{z_k}
$$

（对 softmax 的定义两边取 log 就得到它。）代入 loss：

$$
L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i \\left(z_i - \\log \\sum_{k=1}^C e^{z_k}\\right) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
$$

因为 $y$ 是概率分布，$\\sum_{i=1}^C y_i = 1$，所以

$$
L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
$$

对 $z_j$ 求导（第一项里只有 $i = j$ 那一项含 $z_j$；第二项用 $\\log$ 的求导法则，$\\log u$ 的导数是 $u'/u$）：

$$
\\frac{\\partial L}{\\partial z_j} = -y_j + \\frac{e^{z_j}}{\\sum_{k=1}^C e^{z_k}} = p_j - y_j
$$
`,
    },
  },

  label_smoothing_notes: {
    en: {
      title: "Label Smoothing",
      md: `
**Label smoothing replaces the one-hot target with a soft target, so the gradient $\\nabla_z L = p - y$ never becomes exactly zero, even when the model is extremely confident.**

## Why one-hot targets collapse

With standard (one-hot) cross-entropy, if the model is right and saturated:

$$
p \\approx [1, 0, \\ldots] \\text{ and } y = [1, 0, \\ldots] \\Rightarrow p - y \\approx [0, 0, \\ldots]
$$

That is a near-zero gradient (collapse).

## The smoothed target

With label smoothing $\\epsilon$, set the target for the correct class $c$ and the other $C - 1$ classes to

$$
y_c^{LS} = 1 - \\epsilon, \\qquad y_{j \\neq c}^{LS} = \\frac{\\epsilon}{C-1}
$$

Then, even if $p \\approx [1, 0, \\ldots]$, we get

$$
\\nabla_z L = p - y^{LS} \\approx \\left[\\epsilon, -\\frac{\\epsilon}{C-1}, \\ldots\\right]
$$

which is non-zero, so the logits still receive useful updates.

> Intuitively, label smoothing reduces overconfidence by nudging the top logit down a bit and pushing up the others slightly. This prevents complete gradient shutdown when the model is already confident, keeping training dynamics active and reducing brittle overfitting.

## Tiny example ($C = 3$, $\\epsilon = 0.1$)

- Smoothed target: $y^{LS} = [0.9, 0.05, 0.05]$.
- If $p = [0.999, 0.0005, 0.0005]$ (overconfident), then

$$
\\nabla_z L = [0.999 - 0.9,\\; 0.0005 - 0.05,\\; 0.0005 - 0.05] = [0.099, -0.0495, -0.0495]
$$

This is non-zero, so the weights still get updated.

## References

1. **Original paper**

   Rethinking the Inception Architecture for Computer Vision (Inception-v3). Introduces and motivates label smoothing as a regularizer.

   [Rethinking the Inception Architecture for Computer Vision ↗](https://arxiv.org/pdf/1512.00567)

2. **Transformer training detail**

   Attention Is All You Need. Explicitly states that they use label smoothing with $\\epsilon = 0.1$ and notes its effects.

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

3. **Theory & empirical study**

   When Does Label Smoothing Help? (Müller, Kornblith, Hinton, NeurIPS 2019). Effects on generalization, calibration, and why it can hurt distillation.

   [When Does Label Smoothing Help? ↗](https://arxiv.org/pdf/1906.02629)

4. **Related regularizer (confidence penalty) and its connection to LS**

   Pereyra et al., ICLR 2017 workshop. Penalizes low-entropy outputs; shows the relation to label smoothing via the direction of the KL divergence.

   [Regularizing Neural Networks by Penalizing Confident Output Distributions ↗](https://arxiv.org/pdf/1701.06548)

5. **Calibration context**

   Guo et al., On Calibration of Modern Neural Networks (ICML 2017). Shows that modern networks tend to be overconfident (their confidence is higher than their accuracy) and proposes temperature scaling as a fix. Useful background for why reducing overconfidence, as label smoothing does, matters.

   [On Calibration of Modern Neural Networks ↗](https://arxiv.org/pdf/1706.04599)
`,
    },
    zh: {
      title: "Label Smoothing",
      md: `
**Label smoothing 把 one-hot 目标换成一个"软"目标，这样即使模型非常有把握，梯度 $\\nabla_z L = p - y$ 也永远不会正好变成 0。**

## 为什么 one-hot 目标会让梯度塌缩

用标准的（one-hot）cross-entropy 时，如果模型预测对了而且已经饱和：

$$
p \\approx [1, 0, \\ldots] \\text{ 且 } y = [1, 0, \\ldots] \\Rightarrow p - y \\approx [0, 0, \\ldots]
$$

梯度接近 0（塌缩）。

## 平滑后的目标

用 label smoothing 系数 $\\epsilon$，把正确类别 $c$ 和其他 $C - 1$ 个类别的目标设成

$$
y_c^{LS} = 1 - \\epsilon, \\qquad y_{j \\neq c}^{LS} = \\frac{\\epsilon}{C-1}
$$

这样即使 $p \\approx [1, 0, \\ldots]$，也有

$$
\\nabla_z L = p - y^{LS} \\approx \\left[\\epsilon, -\\frac{\\epsilon}{C-1}, \\ldots\\right]
$$

它不是 0，所以 logits 仍然能得到有用的更新。

> 直观地说，label smoothing 会把最高的 logit 往下压一点，把其他 logits 往上推一点，从而减少过度自信。这样模型已经很有把握时，梯度也不会完全停掉，训练能继续进行，也减少了脆弱的过拟合。

## 小例子（$C = 3$，$\\epsilon = 0.1$）

- 平滑后的目标：$y^{LS} = [0.9, 0.05, 0.05]$。
- 如果 $p = [0.999, 0.0005, 0.0005]$（过度自信），那么

$$
\\nabla_z L = [0.999 - 0.9,\\; 0.0005 - 0.05,\\; 0.0005 - 0.05] = [0.099, -0.0495, -0.0495]
$$

不是 0，所以权重仍然会被更新。

## 参考资料

1. **原始论文**

   Rethinking the Inception Architecture for Computer Vision（Inception-v3）。提出了 label smoothing 作为一种正则化方法，并说明了动机。

   [Rethinking the Inception Architecture for Computer Vision ↗](https://arxiv.org/pdf/1512.00567)

2. **Transformer 训练细节**

   Attention Is All You Need。明确写了训练时使用 $\\epsilon = 0.1$ 的 label smoothing，并说明了它的效果。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

3. **理论和实验研究**

   When Does Label Smoothing Help?（Müller、Kornblith、Hinton，NeurIPS 2019）。研究了它对泛化和校准的影响，以及为什么它会损害知识蒸馏。

   [When Does Label Smoothing Help? ↗](https://arxiv.org/pdf/1906.02629)

4. **相关的正则化方法（confidence penalty）以及它和 LS 的关系**

   Pereyra 等人，ICLR 2017 workshop。惩罚低熵（过于自信）的输出；通过 KL 散度的方向说明了它和 label smoothing 的关系。

   [Regularizing Neural Networks by Penalizing Confident Output Distributions ↗](https://arxiv.org/pdf/1701.06548)

5. **校准（calibration）的背景**

   Guo 等人，On Calibration of Modern Neural Networks（ICML 2017）。说明了现代神经网络往往过度自信（给出的置信度高于实际准确率），并提出用 temperature scaling 修正。这是理解为什么要减少过度自信（label smoothing 也在做这件事）的背景资料。

   [On Calibration of Modern Neural Networks ↗](https://arxiv.org/pdf/1706.04599)
`,
    },
  },
};
