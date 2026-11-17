// Notes: Positional information: learned, sinusoidal, RoPE.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  embedding_pos: {
    en: {
      title: "Positional Embedding",
      md: `
## What it is

Transformers need order information because self-attention has no sense of sequence. Positional embeddings give each token a unique vector based on its position.

> GPT-2's positional embeddings are **learnable parameters** in a lookup table, trained by backpropagation exactly like any other weight. They are not the [fixed sinusoids](#info:sinusoidal_pos_notes) of the original Transformer paper.

The shape of the positional embedding weight matrix is:

$$
W_{pos} \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
$$

- $\\text{max\\_toks}$ is the maximum sequence length (1024 for GPT-2).
- Each row $p_j$ corresponds to position $j$ in the sequence.

## Why do we need positional embeddings?

Intuitively, they give the model a sense of order and position in the sequence. But how exactly does that work?

### Without position information

Suppose we only have the token embeddings $X \\in \\mathbb{R}^{T \\times n\\_embd}$ (ignoring the batch dimension for simplicity). Self-attention computes:

$$
Q = X W_q, \\quad K = X W_k, \\quad \\text{scores} = Q K^\\top
$$

Now shuffle the sequence with a permutation matrix $P$. The new input is $X' = P X$, so

$$
Q' = X' W_q = P X W_q = P Q \\\\[2pt]
K' = X' W_k = P X W_k = P K
$$

and the scores become:

$$
\\text{scores}' = Q' K'^\\top = (P Q)(P K)^\\top = P \\, (Q K^\\top) \\, P^\\top
$$

These are the same numbers as before, with rows and columns reordered.

> Without position information (and without a mask), self-attention is **permutation-equivariant**: shuffling the inputs just shuffles the attention matrix the same way. The model can "see" which tokens are present, but not their order.

### With position information

Adding positional embeddings $p_t$ breaks that symmetry. Swapping two tokens now also swaps which $p_t$ each one gets, so $Q$ and $K$ change. The dot products include position-dependent terms, so $QK^\\top$ changes in value, not just by a row/column reordering. That's how the model becomes sensitive to order.

Still confused? See the [tiny example](#info:positional_embeddings_example_notes).

### What about the causal mask?

The argument above assumes every token can attend to every other token. GPT-2 uses a [causal mask](#info:causal_analogy_notes): position $t$ only attends to positions $0, \\dots, t$. The mask by itself carries order information in two ways:

- **Which tokens each position sees changes.** Shuffle \`A B C\` into \`C A B\`: \`B\` used to see $\\{A, B\\}$, now it sees $\\{C, A, B\\}$, so its output changes. The result is no longer just a reordering, so causal attention is already sensitive to order.
- **How many tokens each position sees differs.** Position $t$ always sees $t + 1$ tokens. Even in \`a a a a\`, where every position sees the same kind of token, the count still differs (e.g. averaging over the visible tokens gives each one a weight of $\\frac{1}{t+1}$). This lets the model infer its absolute position.

Haviv et al. (2022) trained causal language models with **no positional encoding at all** and found they were still competitive with standard models. Probing showed these models learn an implicit sense of absolute position; the authors conjecture that the causal mask lets each token infer how many tokens come before it.

> So "the model cannot see order" holds for attention without a mask. GPT-2 still adds explicit positional embeddings, and today's LLMs still use explicit position information (usually RoPE).

## Runtime usage

1. **Token IDs → token embeddings**

   $$
   \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
   $$

2. **Positions $0, 1, \\dots, T-1$ → positional embeddings**

   $$
   \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
   $$

3. **Add them together via broadcasting** ([What is broadcasting?](#info:broadcasting))

   $$
   x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
   $$

### Example with $n\\_embd = 4$

A toy vocabulary of 3 tokens, at most 3 positions, 4-dimensional vectors. The numbers are made up to keep the arithmetic easy; in a real model they start random and are learned.

\`\`\`text
tok_emb (3 x 4)                      pos_emb (3 x 4)
id 0  dog    [1.0, 0.0, 0.5, 0.0]    pos 0  [0.1, 0.1, 0.0, 0.0]
id 1  bites  [0.0, 1.0, 0.0, 0.5]    pos 1  [0.0, 0.1, 0.1, 0.0]
id 2  man    [0.5, 0.0, 1.0, 0.0]    pos 2  [0.0, 0.0, 0.1, 0.1]
\`\`\`

**Sentence 1:** \`dog bites man\`, ids = [0, 1, 2]

\`\`\`text
pos 0  dog    [1.0, 0.0, 0.5, 0.0] + p0 = [1.1, 0.1, 0.5, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  man    [0.5, 0.0, 1.0, 0.0] + p2 = [0.5, 0.0, 1.1, 0.1]
\`\`\`

**Sentence 2:** \`man bites dog\`, ids = [2, 1, 0]

\`\`\`text
pos 0  man    [0.5, 0.0, 1.0, 0.0] + p0 = [0.6, 0.1, 1.0, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  dog    [1.0, 0.0, 0.5, 0.0] + p2 = [1.0, 0.0, 0.6, 0.1]
\`\`\`

What the result shows:

- **Same token, different position → different vector.** \`dog\` at position 0 in sentence 1 is [1.1, 0.1, 0.5, 0.0]; at position 2 in sentence 2 it is [1.0, 0.0, 0.6, 0.1]. Similar (both are \`dog\`), but not identical.
- **Same token, same position → identical vector.** \`bites\` is at position 1 in both sentences and gets [0.0, 1.1, 0.1, 0.5] both times.
- **Without positional embeddings**, both sentences would give the same three vectors {dog, bites, man}, only in a different order. Now the inputs for "dog bites man" and "man bites dog" really differ.

Shapes, with both sentences in one batch ($B = 2$):

\`\`\`text
idx           = [[0, 1, 2],
                 [2, 1, 0]]       (2, 3)
tok_emb(idx)                      (2, 3, 4)
pos_emb(pos)   pos = [0, 1, 2]    (3, 4)
  [None, :, :]                    (1, 3, 4)   broadcast to every sequence in the batch
x = sum                           (2, 3, 4)
\`\`\`

> Both sentences add the **same** $p_0, p_1, p_2$: a positional vector depends only on the position, not on the sentence.

## Parameters

$$
\\text{params} = \\text{max\\_toks} \\times n\\_embd
$$

> GPT-2 small: $1024 \\times 768 \\approx 7.86 \\times 10^5$, much smaller than the token embedding table (~38M params).

## Shapes at a glance

- **Weight:** $[\\text{max\\_toks}, n\\_embd]$
- **Positions:** $[T]$
- **Pos embeddings:** $[T, n\\_embd]$
- **Final sum:** $[B, T, n\\_embd]$

## Recent developments

Learned absolute positional embeddings, as used in GPT-2, are not widely used in the newest state-of-the-art LLMs. These have largely moved to relative positional methods such as [RoPE (Rotary Position Embeddings) ↗](https://arxiv.org/pdf/2104.09864) or [ALiBi (Attention with Linear Biases) ↗](https://arxiv.org/pdf/2108.12409).

How RoPE works, step by step: [RoPE](#info:rope_notes)

## References

1. **Transformer positional encodings**

   Vaswani et al., "Attention Is All You Need," §3.5. Introduces adding position information to token embeddings via fixed sinusoids.

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 uses learned absolute positional embeddings**

   The Hugging Face GPT-2 docs explicitly note that GPT-2 uses absolute position embeddings (not sinusoidal), hence their advice to pad inputs on the right.

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2)

3. **GPT-2 technical report (input representation)**

   Radford et al., "Language Models are Unsupervised Multitask Learners." Describes the inputs as token + position embeddings summed before the stack.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

4. **Relative position encodings**

   Shaw et al., "Self-Attention with Relative Position Representations." Introduces relative positional embeddings as an alternative to absolute ones.

   [Self-Attention with Relative Position Representations ↗](https://arxiv.org/pdf/1803.02155)

5. **RoPE (Rotary Position Embeddings)**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding." Encodes relative positions in self-attention by rotating queries and keys.

   [RoPE: Rotary Position Embedding ↗](https://arxiv.org/pdf/2104.09864)

6. **ALiBi (Attention with Linear Biases)**

   Press et al. Adds distance-proportional biases to attention scores; enables length extrapolation.

   [ALiBi: Attention with Linear Biases ↗](https://arxiv.org/pdf/2108.12409)

7. **Causal LMs without positional encodings**

   Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information," Findings of EMNLP 2022. Shows that causal LMs with no positional encoding remain competitive and learn an implicit sense of position, likely through the causal mask.

   [arXiv:2203.16634 ↗](https://arxiv.org/abs/2203.16634)

  `,
    },
    zh: {
      title: "位置嵌入（Positional Embedding）",
      md: `
## 它是什么

Transformer 需要顺序信息，因为 self-attention 本身没有"序列"的概念。位置嵌入根据每个 token 所在的位置，给它一个独一无二的向量。

> GPT-2 的位置嵌入是查找表里的**可学习参数**，和其他权重一样通过反向传播训练。它们不是原始 Transformer 论文里的[固定正弦函数](#info:sinusoidal_pos_notes)。

位置嵌入权重矩阵的形状是：

$$
W_{pos} \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
$$

- $\\text{max\\_toks}$ 是最大序列长度（GPT-2 是 1024）。
- 每一行 $p_j$ 对应序列中的位置 $j$。

## 为什么需要位置嵌入？

直观上，它们让模型对序列中的顺序和位置有感知。但具体是怎么做到的？

### 没有位置信息时

假设我们只有 token embedding $X \\in \\mathbb{R}^{T \\times n\\_embd}$（为简单起见忽略 batch 维度）。Self-attention 计算：

$$
Q = X W_q, \\quad K = X W_k, \\quad \\text{scores} = Q K^\\top
$$

现在用一个置换矩阵 $P$ 打乱序列。新的输入是 $X' = P X$，于是

$$
Q' = X' W_q = P X W_q = P Q \\\\[2pt]
K' = X' W_k = P X W_k = P K
$$

分数变成：

$$
\\text{scores}' = Q' K'^\\top = (P Q)(P K)^\\top = P \\, (Q K^\\top) \\, P^\\top
$$

这些数和之前完全一样，只是行和列的顺序变了。

> 没有位置信息（也没有 mask）时，self-attention 是**置换等变的（permutation-equivariant）**：打乱输入，只会以同样的方式打乱 attention 矩阵。模型能"看到"有哪些 token，却看不到它们的顺序。

### 有位置信息时

加上位置嵌入 $p_t$ 就打破了这种对称性。现在交换两个 token，也会交换它们各自拿到的 $p_t$，所以 $Q$ 和 $K$ 会变。点积里包含了依赖位置的项，因此 $QK^\\top$ 的数值本身会变，而不只是行/列重新排序。模型就是这样对顺序变得敏感的。

还是不太明白？看这个[小例子](#info:positional_embeddings_example_notes)。

### 那 causal mask 呢？

上面的论证假设每个 token 都能 attend 到所有其他 token。GPT-2 用的是 [causal mask](#info:causal_analogy_notes)：位置 $t$ 只能 attend 到位置 $0, \\dots, t$。mask 本身就以两种方式携带了顺序信息：

- **每个位置能看到哪些 token 会变。** 把 \`A B C\` 打乱成 \`C A B\`：\`B\` 原来看到的是 $\\{A, B\\}$，现在看到的是 $\\{C, A, B\\}$，所以它的输出变了。结果不再只是重新排序，因此 causal attention 本来就对顺序敏感。
- **每个位置能看到的 token 数量不同。** 位置 $t$ 总是看到 $t + 1$ 个 token。即使在 \`a a a a\` 中，每个位置看到的 token 种类都一样，数量仍然不同（比如对可见 token 取平均，每个 token 的权重是 $\\frac{1}{t+1}$）。这让模型可以推断出自己的绝对位置。

Haviv et al. (2022) 训练了**完全没有位置编码**的 causal 语言模型，发现它们仍能和标准模型相媲美。探测（probing）显示，这些模型学到了一种隐式的绝对位置感；作者推测，是 causal mask 让每个 token 能推断出自己前面有多少个 token。

> 所以，"模型看不到顺序"这句话只对没有 mask 的 attention 成立。GPT-2 仍然加上了显式的位置嵌入，今天的 LLM 也仍然使用显式的位置信息（通常是 RoPE）。

## 运行时的用法

1. **Token ID → token embedding**

   $$
   \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
   $$

2. **位置 $0, 1, \\dots, T-1$ → 位置嵌入**

   $$
   \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
   $$

3. **通过广播把两者相加**（[什么是广播？](#info:broadcasting)）

   $$
   x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
   $$

### 例子：$n\\_embd = 4$

一个只有 3 个 token 的玩具词表，最多 3 个位置，向量是 4 维的。这些数字是编出来的，方便计算；在真实模型里它们从随机值开始，通过学习得到。

\`\`\`text
tok_emb (3 x 4)                      pos_emb (3 x 4)
id 0  dog    [1.0, 0.0, 0.5, 0.0]    pos 0  [0.1, 0.1, 0.0, 0.0]
id 1  bites  [0.0, 1.0, 0.0, 0.5]    pos 1  [0.0, 0.1, 0.1, 0.0]
id 2  man    [0.5, 0.0, 1.0, 0.0]    pos 2  [0.0, 0.0, 0.1, 0.1]
\`\`\`

**句子 1：** \`dog bites man\`，ids = [0, 1, 2]

\`\`\`text
pos 0  dog    [1.0, 0.0, 0.5, 0.0] + p0 = [1.1, 0.1, 0.5, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  man    [0.5, 0.0, 1.0, 0.0] + p2 = [0.5, 0.0, 1.1, 0.1]
\`\`\`

**句子 2：** \`man bites dog\`，ids = [2, 1, 0]

\`\`\`text
pos 0  man    [0.5, 0.0, 1.0, 0.0] + p0 = [0.6, 0.1, 1.0, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  dog    [1.0, 0.0, 0.5, 0.0] + p2 = [1.0, 0.0, 0.6, 0.1]
\`\`\`

结果说明了什么：

- **同一个 token，不同位置 → 不同的向量。** 句子 1 中位置 0 的 \`dog\` 是 [1.1, 0.1, 0.5, 0.0]；句子 2 中位置 2 的 \`dog\` 是 [1.0, 0.0, 0.6, 0.1]。两者相似（都是 \`dog\`），但不完全相同。
- **同一个 token，同一位置 → 完全相同的向量。** \`bites\` 在两个句子里都在位置 1，两次得到的都是 [0.0, 1.1, 0.1, 0.5]。
- **如果没有位置嵌入**，两个句子会得到同样的三个向量 {dog, bites, man}，只是顺序不同。现在 "dog bites man" 和 "man bites dog" 的输入真正不同了。

形状（两个句子放在一个 batch 里，$B = 2$）：

\`\`\`text
idx           = [[0, 1, 2],
                 [2, 1, 0]]       (2, 3)
tok_emb(idx)                      (2, 3, 4)
pos_emb(pos)   pos = [0, 1, 2]    (3, 4)
  [None, :, :]                    (1, 3, 4)   broadcast to every sequence in the batch
x = sum                           (2, 3, 4)
\`\`\`

> 两个句子加上的是**同样的** $p_0, p_1, p_2$：位置向量只取决于位置，与句子无关。

## 参数量

$$
\\text{params} = \\text{max\\_toks} \\times n\\_embd
$$

> GPT-2 small：$1024 \\times 768 \\approx 7.86 \\times 10^5$，比 token embedding 表（约 38M 参数）小得多。

## 形状一览

- **权重：** $[\\text{max\\_toks}, n\\_embd]$
- **位置：** $[T]$
- **位置嵌入：** $[T, n\\_embd]$
- **最终的和：** $[B, T, n\\_embd]$

## 近期发展

GPT-2 所用的可学习绝对位置嵌入，在最新的 state-of-the-art LLM 中已经不常用了。这些模型大多转向了相对位置方法，比如 [RoPE（旋转位置编码）↗](https://arxiv.org/pdf/2104.09864) 或 [ALiBi（带线性偏置的 attention）↗](https://arxiv.org/pdf/2108.12409)。

RoPE 是如何一步步工作的：[RoPE](#info:rope_notes)

## 参考文献

1. **Transformer 的位置编码**

   Vaswani et al., "Attention Is All You Need," §3.5。提出用固定正弦函数把位置信息加到 token embedding 上。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 使用可学习的绝对位置嵌入**

   Hugging Face 的 GPT-2 文档明确指出，GPT-2 使用绝对位置嵌入（而不是正弦编码），所以建议在右侧填充（pad）输入。

   [GPT-2 配置 ↗](https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2)

3. **GPT-2 技术报告（输入表示）**

   Radford et al., "Language Models are Unsupervised Multitask Learners." 描述了输入是 token embedding 与位置嵌入之和，然后再送入层堆叠。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

4. **相对位置编码**

   Shaw et al., "Self-Attention with Relative Position Representations." 提出相对位置嵌入，作为绝对位置嵌入的替代方案。

   [Self-Attention with Relative Position Representations ↗](https://arxiv.org/pdf/1803.02155)

5. **RoPE（旋转位置编码）**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding." 通过旋转 query 和 key，在 self-attention 中编码相对位置。

   [RoPE: Rotary Position Embedding ↗](https://arxiv.org/pdf/2104.09864)

6. **ALiBi（带线性偏置的 attention）**

   Press et al.。给 attention 分数加上与距离成正比的偏置；使模型能外推到更长的长度。

   [ALiBi: Attention with Linear Biases ↗](https://arxiv.org/pdf/2108.12409)

7. **没有位置编码的 causal LM**

   Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information," Findings of EMNLP 2022。表明没有位置编码的 causal LM 仍然有竞争力，并且学到了隐式的位置感，很可能是通过 causal mask。

   [arXiv:2203.16634 ↗](https://arxiv.org/abs/2203.16634)

  `,
    },
  },

  positional_embeddings_example_notes: {
    en: {
      title: "Positional Embeddings Example",
      md: `
**A tiny 2-D example: without positions, swapping two tokens only permutes the attention scores; with positions, the scores themselves change.**

## Setup (no positions)

Let the token embeddings be 2-D vectors:

$$
e(A) = \\begin{bmatrix}1, 0\\end{bmatrix}, \\quad e(B) = \\begin{bmatrix}0, 1\\end{bmatrix}
$$

Use identity projections, so

$$
Q = X, \\quad K = X, \\quad S = QK^\\top = XX^\\top
$$

For the sequence $[A, B]$:

$$
X = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}, \\qquad S = XX^\\top = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}
$$

The sequence $[B, A]$ just permutes the rows of $X$. As shown in the [positional embedding notes](#info:embedding_pos), the scores become $S' = PSP^\\top$.

> These are the same numbers as $S$, just with rows/columns swapped. Self-attention can't tell $AB$ from $BA$: it only sees a permutation.

## Add positional embeddings

Let the position vectors be

$$
p_0 = \\begin{bmatrix}1, 1\\end{bmatrix}, \\quad p_1 = \\begin{bmatrix}-1, 1\\end{bmatrix}
$$

We feed $h_t = e(\\text{token}_t) + p_t$.

### Sequence $[A, B]$ ($A$ at position 0, $B$ at position 1)

$$
h_0 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}2, 1\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}-1, 2\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix} \\\\[4pt]
S = hh^\\top = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix}\\begin{bmatrix}2 & -1\\\\1 & 2\\end{bmatrix} = \\begin{bmatrix}5 & 0\\\\0 & 5\\end{bmatrix}
$$

### Sequence $[B, A]$ ($B$ at position 0, $A$ at position 1)

$$
h_0 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}1, 2\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}0, 1\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix} \\\\[4pt]
S' = hh^\\top = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix}\\begin{bmatrix}1 & 0\\\\2 & 1\\end{bmatrix} = \\begin{bmatrix}5 & 2\\\\2 & 1\\end{bmatrix}
$$

> Now $S'$ is not just a row/column permutation of $S$; the dot products changed in value. Without positions, shuffling only permutes the attention matrix; with positions, the numbers themselves change, so the model becomes order-sensitive.
`,
    },
    zh: {
      title: "位置 Embedding 示例",
      md: `
**一个很小的 2 维例子：没有位置信息时，交换两个 token 只会让 attention 分数换个排列；有了位置信息，分数本身就会改变。**

## 设定（没有位置信息）

令 token embedding 为 2 维向量：

$$
e(A) = \\begin{bmatrix}1, 0\\end{bmatrix}, \\quad e(B) = \\begin{bmatrix}0, 1\\end{bmatrix}
$$

使用恒等投影，所以

$$
Q = X, \\quad K = X, \\quad S = QK^\\top = XX^\\top
$$

对于序列 $[A, B]$：

$$
X = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}, \\qquad S = XX^\\top = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}
$$

序列 $[B, A]$ 只是把 $X$ 的行换了个顺序。如[位置 embedding 笔记](#info:embedding_pos)所示，分数变成 $S' = PSP^\\top$。

> 这些数字和 $S$ 里的一样，只是行/列换了位置。Self-attention 分不清 $AB$ 和 $BA$：它看到的只是一个置换。

## 加上位置 embedding

令位置向量为

$$
p_0 = \\begin{bmatrix}1, 1\\end{bmatrix}, \\quad p_1 = \\begin{bmatrix}-1, 1\\end{bmatrix}
$$

输入改为 $h_t = e(\\text{token}_t) + p_t$。

### 序列 $[A, B]$（$A$ 在位置 0，$B$ 在位置 1）

$$
h_0 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}2, 1\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}-1, 2\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix} \\\\[4pt]
S = hh^\\top = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix}\\begin{bmatrix}2 & -1\\\\1 & 2\\end{bmatrix} = \\begin{bmatrix}5 & 0\\\\0 & 5\\end{bmatrix}
$$

### 序列 $[B, A]$（$B$ 在位置 0，$A$ 在位置 1）

$$
h_0 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}1, 2\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}0, 1\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix} \\\\[4pt]
S' = hh^\\top = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix}\\begin{bmatrix}1 & 0\\\\2 & 1\\end{bmatrix} = \\begin{bmatrix}5 & 2\\\\2 & 1\\end{bmatrix}
$$

> 现在 $S'$ 不再只是 $S$ 的行/列置换；点积的数值本身变了。没有位置信息时，打乱顺序只会置换 attention 矩阵；有了位置信息，数字本身会变，于是模型对顺序变得敏感。
`,
    },
  },

  sinusoidal_pos_notes: {
    en: {
      title: "Sinusoidal Positional Encoding",
      md: `
**Fixed sin/cos position vectors, computed by a formula instead of learned.**

The original Transformer (Vaswani et al., 2017) adds these vectors to the token embeddings, the same way GPT-2 adds its [learned positional embeddings](#info:embedding_pos). Nothing about them is trained.

## The formula

Group the $d$ dimensions into pairs. Pair $i$ gets its own frequency $\\omega_i$:

$$
PE(pos, 2i) = \\sin(\\omega_i \\cdot pos), \\qquad PE(pos, 2i+1) = \\cos(\\omega_i \\cdot pos), \\qquad \\omega_i = \\frac{1}{10000^{2i/d}}
$$

- At $i = 0$, $\\omega = 1$: the fastest pair.
- As $i$ grows, $\\omega$ shrinks and the pair turns more slowly. The slowest pair has a wavelength of about $10000 \\cdot 2\\pi$.

## Intuition: a clock with many hands

Each pair $(\\sin, \\cos)$ is one **hand** on a circle. Each step in position turns hand $i$ by $\\omega_i$ radians.

- **Fast hands** (like a second hand) change a lot from one position to the next, so they tell neighbors apart.
- **Slow hands** (like an hour hand) only move noticeably over many positions, so they tell near from far.

Reading all the hands together pins down the position, just as the second, minute and hour hands together give the exact time.

## Example with $d = 4$

Two hands: $\\omega_0 = 1$ (fast) and $\\omega_1 = 1/10000^{2/4} = 1/100$ (slow).

\`\`\`text
         fast: sin,  cos      slow: sin,  cos
pos 0  [  0.000,  1.000,         0.000, 1.000 ]
pos 1  [  0.841,  0.540,         0.010, 1.000 ]
pos 2  [  0.909, -0.416,         0.020, 1.000 ]
pos 3  [  0.141, -0.990,         0.030, 1.000 ]
\`\`\`

- The fast hand changes a lot at every step, so neighboring positions are easy to tell apart.
- The slow hand barely moves; it takes on the order of a hundred steps to turn noticeably. It says roughly which stretch of the sequence you are in.

## Key property: shifting by $k$ is a fixed rotation

For each pair:

$$
\\begin{bmatrix} \\sin(\\omega(p+k)) \\\\ \\cos(\\omega(p+k)) \\end{bmatrix}
=
\\begin{bmatrix} \\cos \\omega k & \\sin \\omega k \\\\ -\\sin \\omega k & \\cos \\omega k \\end{bmatrix}
\\begin{bmatrix} \\sin(\\omega p) \\\\ \\cos(\\omega p) \\end{bmatrix}
$$

The matrix depends only on $k$, not on $p$. So "look $k$ positions back" is the **same linear map at every position**. The paper gives this as its reason for the design: $PE_{pos+k}$ is a linear function of $PE_{pos}$, which should make it easy for attention to work with relative positions.

> GPT-2's learned table ends up with a similar shape: neighboring positions are close, and the vectors trace smooth loops at several speeds. The difference is that GPT-2 has to learn this from data, while sinusoidal encoding builds it in.

## Compared with GPT-2's learned embeddings

- **Parameters:** sinusoidal has none; GPT-2's table has $1024 \\times 768$.
- **Beyond the training length:** the formula gives a vector for any position; GPT-2's table has no row past 1023.
- **Structure:** sinusoidal is designed (hands at many speeds); GPT-2's is learned and ends up with similar smooth loops.
- **Results:** Vaswani et al. also tried learned positional embeddings and found the two "produced nearly identical results."
- **Extrapolation:** they chose sinusoids because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." In practice this works poorly: having a vector for an unseen position does not mean the model knows how to use it. The ALiBi paper measured this directly.

## Recent developments

**RoPE** (Rotary Position Embedding), used by most LLMs today, uses the same set of frequencies $\\omega_i$. The difference is that RoPE does not add a position vector to the input. Instead, inside every attention layer it **rotates** $q$ and $k$ by angles proportional to their positions, so $q \\cdot k$ depends only on the **relative distance** between the two positions.

How RoPE works, step by step: [RoPE](#info:rope_notes)

## References

1. **Attention Is All You Need, §3.5**

   Vaswani et al., 2017. Defines the sinusoidal encoding, compares it with learned embeddings, and explains the choice.

   [arXiv:1706.03762 ↗](https://arxiv.org/abs/1706.03762)

2. **ALiBi (Attention with Linear Biases)**

   Press et al. Tests how well sinusoidal, rotary and other position methods extrapolate to longer sequences than seen in training.

   [arXiv:2108.12409 ↗](https://arxiv.org/abs/2108.12409)

3. **RoPE**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding."

   [arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)

    `,
    },
    zh: {
      title: "正弦位置编码（Sinusoidal Positional Encoding）",
      md: `
**固定的 sin/cos 位置向量，由公式算出来，而不是学出来的。**

原始 Transformer（Vaswani et al., 2017）把这些向量加到 token embedding 上，方式和 GPT-2 加它的[可学习位置嵌入](#info:embedding_pos)一样。它们没有任何部分是训练出来的。

## 公式

把 $d$ 个维度两两分成一对。第 $i$ 对有自己的频率 $\\omega_i$：

$$
PE(pos, 2i) = \\sin(\\omega_i \\cdot pos), \\qquad PE(pos, 2i+1) = \\cos(\\omega_i \\cdot pos), \\qquad \\omega_i = \\frac{1}{10000^{2i/d}}
$$

- 当 $i = 0$ 时，$\\omega = 1$：转得最快的一对。
- 随着 $i$ 增大，$\\omega$ 变小，这一对转得越来越慢。最慢的一对波长约为 $10000 \\cdot 2\\pi$。

## 直觉：一个有很多指针的时钟

每一对 $(\\sin, \\cos)$ 就是圆上的一根**指针**。位置每前进一步，指针 $i$ 就转过 $\\omega_i$ 弧度。

- **快指针**（像秒针）在相邻位置之间变化很大，所以能区分相邻的位置。
- **慢指针**（像时针）只有跨越很多位置才会明显移动，所以能区分近和远。

把所有指针一起读出来，就能确定位置，就像秒针、分针、时针合在一起给出精确时间一样。

## 例子：$d = 4$

两根指针：$\\omega_0 = 1$（快）和 $\\omega_1 = 1/10000^{2/4} = 1/100$（慢）。

\`\`\`text
         fast: sin,  cos      slow: sin,  cos
pos 0  [  0.000,  1.000,         0.000, 1.000 ]
pos 1  [  0.841,  0.540,         0.010, 1.000 ]
pos 2  [  0.909, -0.416,         0.020, 1.000 ]
pos 3  [  0.141, -0.990,         0.030, 1.000 ]
\`\`\`

- 快指针每一步都变化很大，所以相邻位置很容易区分。
- 慢指针几乎不动；要上百步才会明显转动。它大致告诉你处在序列的哪一段。

## 关键性质：平移 $k$ 是一个固定的旋转

对每一对：

$$
\\begin{bmatrix} \\sin(\\omega(p+k)) \\\\ \\cos(\\omega(p+k)) \\end{bmatrix}
=
\\begin{bmatrix} \\cos \\omega k & \\sin \\omega k \\\\ -\\sin \\omega k & \\cos \\omega k \\end{bmatrix}
\\begin{bmatrix} \\sin(\\omega p) \\\\ \\cos(\\omega p) \\end{bmatrix}
$$

这个矩阵只依赖 $k$，不依赖 $p$。所以"往回看 $k$ 个位置"在**每个位置上都是同一个线性映射**。论文给出的设计理由正是这一点：$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，这应该让 attention 很容易处理相对位置。

> GPT-2 学到的表最终也有类似的形态：相邻位置彼此接近，向量以多种速度画出平滑的环。区别在于 GPT-2 必须从数据中学出这一点，而正弦编码是直接内置的。

## 和 GPT-2 的可学习嵌入对比

- **参数：** 正弦编码没有参数；GPT-2 的表有 $1024 \\times 768$ 个。
- **超出训练长度：** 公式可以给任意位置算出向量；GPT-2 的表在 1023 之后就没有行了。
- **结构：** 正弦编码是设计出来的（多种速度的指针）；GPT-2 的是学出来的，最后也形成了类似的平滑环。
- **效果：** Vaswani et al. 也试过可学习位置嵌入，发现两者 "produced nearly identical results."
- **外推：** 他们选择正弦函数，是因为它 "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." 实际上效果很差：对一个没见过的位置有向量，并不意味着模型知道怎么用它。ALiBi 论文直接测量了这一点。

## 近期发展

**RoPE**（Rotary Position Embedding，旋转位置编码）是今天大多数 LLM 使用的方法，它用的是同一组频率 $\\omega_i$。区别在于 RoPE 不往输入里加位置向量，而是在每一个 attention 层内部，把 $q$ 和 $k$ 按与各自位置成正比的角度**旋转**，使得 $q \\cdot k$ 只依赖两个位置之间的**相对距离**。

RoPE 是如何一步步工作的：[RoPE](#info:rope_notes)

## 参考文献

1. **Attention Is All You Need, §3.5**

   Vaswani et al., 2017。定义了正弦编码，与可学习嵌入做了比较，并解释了这一选择。

   [arXiv:1706.03762 ↗](https://arxiv.org/abs/1706.03762)

2. **ALiBi（带线性偏置的 attention）**

   Press et al.。测试了正弦、旋转等位置方法外推到比训练时更长的序列时效果如何。

   [arXiv:2108.12409 ↗](https://arxiv.org/abs/2108.12409)

3. **RoPE**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding."

   [arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)

    `,
    },
  },

  rope_notes: {
    en: {
      title: "RoPE (Rotary Position Embedding)",
      md: `
**RoPE (Rotary Position Embedding) is how most LLMs today encode position. GPT-2 does not use it (it uses a [learned position embedding](#info:embedding_pos)), but it is easiest to understand right at the $q \\cdot k$ step.**

## 1. Where the position information goes

**GPT-2: added to $x$ once, at the start.**

\`\`\`text
x = token embedding + position embedding    <- position mixed into x before block 1
q = x · W_qᵀ,   k = x · W_kᵀ
score = q · k
\`\`\`

**RoPE: $x$ is left alone; $q$ and $k$ are rotated just before scoring.**

\`\`\`text
x = token embedding                          <- no position embedding
q = x · W_qᵀ,   k = x · W_kᵀ
q' = rotate q by an angle set by its position
k' = rotate k by an angle set by its position
score = q' · k'
\`\`\`

Three differences:

- RoPE **adds nothing to $x$**: the residual stream carries no position vector.
- The rotation happens **inside every attention layer**, right before $q \\cdot k$, not once at the input.
- Only **q and k** are rotated, never **v**. Position affects *who to attend to*, not *what content is taken*.

## 2. Why rotation encodes position: the 2-D picture

Think of $q$ and $k$ as two arrows in a plane. Their dot product depends on their lengths and the **angle between them**.

Now say the vector at position $m$ is rotated by $m \\times \\theta$, with $\\theta = 10°$:

- q at position 5 turns 50°, k at position 3 turns 30°: the angle between them changes by $50° - 30° = 20°$.
- Shift both far along, q at 105 and k at 103: q turns 1050°, k turns 1030°, and the change is still $20°$.

> **After rotation, the score depends only on the distance between the two positions** ($5 - 3 = 2$), not on where they are. "The previous word" or "two words back" is scored by the same rule at position 5 and at position 1000.

## 3. Why "only relative distance" is a good property

**Most rules of language are relative.**

- An adjective usually sits **one position before** its noun: \`black cat\`.
- \`it\` usually refers to something **a little earlier**.
- A verb often comes **a few positions after** its subject.

These rules say "$n$ positions back", never "position 37". The same rule holds at the start of a text and at token 800.

**With absolute positions, the same rule has to be learned many times.** Suppose \`black cat\` appears twice:

\`\`\`text
positions 3, 4      :  black cat
positions 503, 504  :  black cat
\`\`\`

- **Relative distance only (RoPE):** both pairs are 1 apart, so the position part of the score is automatically the same. Learned once, it works everywhere.
- **Absolute positions:** the model sees "3 and 4" and "503 and 504", two unrelated pairs of numbers. That they are the same relation must be learned from data.

It is like teaching "look one seat ahead", which works in every seat, versus separately training "from seat 3, look at seat 2", "from seat 503, look at seat 502", and so on.

> **Content still matters.** "Only relative distance" applies to the **position** part of the score. q and k are still computed from the tokens, so at the same distance 1, \`black → cat\` may score high and \`the → of\` low. What RoPE guarantees is that the same content at the same distance gets the same score wherever it appears.

## 4. From 2-D to 64-D: rotate in pairs

Each head's q and k have $hs = 64$ dims. RoPE groups them into **32 pairs** and treats each pair as a 2-D vector that rotates in its own plane:

\`\`\`text
q (64 dims) = [ (q0, q1), (q2, q3), (q4, q5), ..., (q62, q63) ]
                 pair 0    pair 1    pair 2         pair 31
\`\`\`

At position $m$, pair $i$ is rotated by the angle $m\\,\\theta_i$, using the ordinary 2-D rotation:

$$
(a,\\, b) \\;\\rightarrow\\; \\bigl(a\\cos(m\\theta_i) - b\\sin(m\\theta_i),\\;\\; a\\sin(m\\theta_i) + b\\cos(m\\theta_i)\\bigr)
$$

Each pair has its **own speed**:

$$
\\theta_i = \\frac{1}{10000^{2i/64}}, \\qquad i = 0, 1, \\dots, 31
$$

- **Pair 0:** $\\theta = 1$ radian per position, about 57.3°. One full turn every $2\\pi \\approx 6.3$ positions.
- **Pair 16:** $\\theta = 0.01$ radian, about 0.57°. One full turn every ≈ 628 positions.
- **Pair 31:** $\\theta \\approx 0.00013$ radian. One full turn every ≈ 47,000 positions.

> **Radians.** Angles here are in radians, as in \`torch.sin\` / \`torch.cos\`. A full turn is $360° = 2\\pi \\approx 6.283$ radians, so 1 radian $= 180°/\\pi \\approx 57.3°$: the angle subtended by an arc as long as the radius.

**Why mix fast and slow pairs?**

- **Fast pairs** turn a lot between neighbors, so they can tell "adjacent" from "two apart". But they wrap around after a few positions, so on their own they can't tell far distances apart.
- **Slow pairs** barely change between neighbors, but their angles spread out over long distances, so they can tell "100 apart" from "1000 apart".

Together they cover both near and far, like the second, minute and hour hands of a clock.

### q and k are rotated by the same rule

Both use the **same** rule with the **same** 32 speeds $\\theta_i$. Only the angle differs, because it is set by each vector's own position:

- q at position $m$: pair $i$ turns $m\\,\\theta_i$;
- k at position $n$: pair $i$ turns $n\\,\\theta_i$.

A token's own q and k sit at the same position, so they turn by the same angle.

**Why the rule must be shared.** In the 2-D example ($\\theta = 10°$), q at 5 and k at 3 turn 50° and 30°, and the difference is $(5 - 3) \\times 10° = 20°$. Both angles are "position × the same $\\theta$", so the absolute positions cancel and only the distance is left.

If q used 10° and k used 7° instead:

- q at 5 and k at 3: $50° - 21° = 29°$;
- q at 105 and k at 103: $1050° - 721° = 329°$.

Same distance, different result: the score would no longer depend only on relative distance.

> "Same rule" means the same *rotation*, not the same *vector*. q and k are still different vectors, made by \`c_attn\` with different weights; RoPE only rotates each of them according to its position afterwards.

## 5. Relation to sinusoidal encoding

The [sinusoidal encoding](#info:sinusoidal_pos_notes) of the original Transformer is:

$$
PE(pos, 2i) = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\qquad PE(pos, 2i+1) = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

The frequencies are **exactly the same**, $1/10000^{2i/d}$, also in sin/cos pairs. The difference is how they are used:

- **Sinusoidal:** the sin/cos values are **added** to $x$, once at the input, and get mixed into the residual stream.
- **RoPE:** the sin/cos values **rotate** (multiply) $q$ and $k$, in every attention layer, and touch nothing else.

The sinusoidal note shows that shifting by $k$ positions is a rotation. RoPE applies that rotation **directly to q and k**, so $q \\cdot k$ automatically depends only on relative distance.

> 🆕 **The base.** The 10000 in the formula is called the base (\`rope_theta\` in Hugging Face configs). Llama 2 uses 10000; Llama 3 raises it to 500000, which slows the slowest pairs further to support longer contexts.

## 6. Beyond the training length

What happens when the sequence is longer than the model was trained on?

### GPT-2: the table has no row for it

GPT-2 gets position from a lookup table, one learned row per position:

\`\`\`text
pos_emb = nn.Embedding(max_toks, n_embd)    # real GPT-2: 1024 rows × 768
pos_emb(pos)                                 # pos = 0, 1, ..., T-1
\`\`\`

- Positions 0 to 1023 each have their own trained row.
- Position 1024 has no row, so the lookup fails with an index error.

So GPT-2 can see **at most 1024 tokens** at once. And since each row is learned separately, adding a new row would not help: it was never trained, so the model would not know what it means.

### How GPT-2 generates past 1024 tokens: a sliding window

During generation the context also includes the tokens the model has generated so far, so prompt + generated tokens quickly reach the limit. Generation code therefore crops the input at every step to the **most recent** 1024 tokens (nanoGPT's \`generate\` crops to \`block_size\`):

\`\`\`text
idx_cond = idx[:, -1024:]     # keep only the last 1024 tokens
logits = model(idx_cond)
\`\`\`

This is **not** splitting the text into separate chunks. Chunking would cut it into non-overlapping pieces:

\`\`\`text
[0 ~ 1023] [1024 ~ 2047] [2048 ~ 2999]
\`\`\`

and token 1024 could not even see token 1023, right next to it. The sliding window instead moves forward by one token per step:

\`\`\`text
generating token 3000: sees [1976 ~ 2999]
generating token 3001: sees [1977 ~ 3000]
generating token 3002: sees [1978 ~ 3001]
\`\`\`

Consecutive windows share 1023 tokens, so nearby context is always continuous: every token sees the 1023 tokens before it.

What is lost is everything **more than 1023 tokens back**. When generating token 3000, tokens 0 to 1975 are simply invisible, not remembered vaguely. If a name was mentioned at token 100, GPT-2 cannot use it at token 3000. GPT-2 has a short-term memory of about 1024 tokens.

> 🆕 This limit is a major reason later models pushed context length much further; many current models support 128K tokens or more.

### RoPE: any position has an angle

There is no table, only the rule "pair $i$ at position $m$ turns $m\\,\\theta_i$". Position 5000 can be plugged in directly, so nothing breaks. Combined with the relative-distance property (section 3), two tokens 3 apart get the same position part of the score at position 10 and at position 5000.

**But "computable" is not "works well".** A model trained on length 4096 and run on length 32,000 usually gets noticeably worse. The cause is the slowest pairs:

- During training they only ever turned by small angles. Pair 31 turns about 0.55 radian over 4096 positions, less than a tenth of a full turn.
- Further out, they reach angles **never seen in training**, and the model does not know how to read them.

> 🆕 Long-context models therefore add extra techniques, such as a larger base (Llama 3 uses 500000) or interpolating/rescaling positions (Position Interpolation, YaRN). The idea in each case is to keep far-away angles within the range seen during training.

In short: GPT-2 can never see more than 1024 tokens at once, and a sliding window only hides that by dropping older text; a RoPE model can take longer input, but needs adjustments to stay accurate there.

## References

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)**

   Su et al. introduce RoPE and show that the rotated $q \\cdot k$ depends only on relative position.

   [RoFormer ↗](https://arxiv.org/abs/2104.09864)

2. **Extending Context Window of Large Language Models via Positional Interpolation (2023)**

   Chen et al. scale positions down so that longer inputs fall within the trained range of RoPE angles.

   [Position Interpolation ↗](https://arxiv.org/abs/2306.15595)

3. **YaRN: Efficient Context Window Extension of Large Language Models (2023)**

   Peng et al. rescale RoPE frequencies differently for fast and slow pairs to extend context length.

   [YaRN ↗](https://arxiv.org/abs/2309.00071)
`,
    },
    zh: {
      title: "RoPE（旋转位置编码）",
      md: `
**RoPE（Rotary Position Embedding，旋转位置编码）是今天大多数 LLM 编码位置的方式。GPT-2 没有用它（GPT-2 用的是[可学习位置嵌入](#info:embedding_pos)），但放在 $q \\cdot k$ 这一步来理解最容易。**

## 1. 位置信息放在哪里

**GPT-2：在一开始加到 $x$ 上，只加一次。**

\`\`\`text
x = token embedding + position embedding    <- position mixed into x before block 1 (在 block 1 之前把位置混进 x)
q = x · W_qᵀ,   k = x · W_kᵀ
score = q · k
\`\`\`

**RoPE：$x$ 保持不动；$q$ 和 $k$ 在算分数之前被旋转。**

\`\`\`text
x = token embedding                          <- no position embedding (没有位置嵌入)
q = x · W_qᵀ,   k = x · W_kᵀ
q' = rotate q by an angle set by its position
k' = rotate k by an angle set by its position
score = q' · k'
\`\`\`

三点不同：

- RoPE **不往 $x$ 里加任何东西**：残差流里没有位置向量。
- 旋转发生在**每一个 attention 层内部**，就在 $q \\cdot k$ 之前，而不是只在输入处做一次。
- 只旋转 **q 和 k**，从不旋转 **v**。位置影响的是*该关注谁*，而不是*取走什么内容*。

## 2. 为什么旋转能编码位置：二维图景

把 $q$ 和 $k$ 想成平面上的两根箭头。它们的点积取决于两者的长度和**它们之间的夹角**。

现在假设位置 $m$ 上的向量被旋转 $m \\times \\theta$，其中 $\\theta = 10°$：

- 位置 5 的 q 转 50°，位置 3 的 k 转 30°：两者之间的夹角变化了 $50° - 30° = 20°$。
- 把两者都移到很远，q 在 105、k 在 103：q 转 1050°，k 转 1030°，变化仍然是 $20°$。

> **旋转之后，分数只取决于两个位置之间的距离**（$5 - 3 = 2$），而与它们具体在哪里无关。"前一个词"或"往前数两个词"在位置 5 和位置 1000 上用的是同一条打分规则。

## 3. 为什么"只依赖相对距离"是个好性质

**语言的大多数规则都是相对的。**

- 形容词通常位于它所修饰名词的**前一个位置**：\`black cat\`。
- \`it\` 通常指代**稍早一点**出现的东西。
- 动词通常出现在主语**之后几个位置**。

这些规则说的都是"往前 $n$ 个位置"，从不说"第 37 个位置"。同一条规则在文本开头和第 800 个 token 处都成立。

**如果用绝对位置，同一条规则得学很多遍。** 假设 \`black cat\` 出现了两次：

\`\`\`text
positions 3, 4      :  black cat
positions 503, 504  :  black cat
\`\`\`

- **只看相对距离（RoPE）：** 两对的距离都是 1，所以分数中的位置部分自动相同。学一次，处处适用。
- **绝对位置：** 模型看到的是"3 和 4"以及"503 和 504"，两对毫不相关的数字。它们其实是同一种关系，这一点必须从数据中学出来。

这就像教"看你前面一个座位"——在每个座位上都适用；对比分别训练"从 3 号座位看 2 号座位"、"从 503 号座位看 502 号座位"，等等。

> **内容依然重要。** "只依赖相对距离"说的是分数中的**位置**部分。q 和 k 仍然是从 token 算出来的，所以同样是距离 1，\`black → cat\` 可能得分很高，而 \`the → of\` 得分很低。RoPE 保证的是：同样的内容、同样的距离，无论出现在哪里，都得到同样的分数。

## 4. 从二维到 64 维：两两成对旋转

每个 head 的 q 和 k 有 $hs = 64$ 维。RoPE 把它们分成 **32 对**，把每一对当作一个二维向量，在各自的平面里旋转：

\`\`\`text
q (64 dims) = [ (q0, q1), (q2, q3), (q4, q5), ..., (q62, q63) ]
                 pair 0    pair 1    pair 2         pair 31
\`\`\`

在位置 $m$，第 $i$ 对被旋转角度 $m\\,\\theta_i$，用的就是普通的二维旋转：

$$
(a,\\, b) \\;\\rightarrow\\; \\bigl(a\\cos(m\\theta_i) - b\\sin(m\\theta_i),\\;\\; a\\sin(m\\theta_i) + b\\cos(m\\theta_i)\\bigr)
$$

每一对都有**自己的速度**：

$$
\\theta_i = \\frac{1}{10000^{2i/64}}, \\qquad i = 0, 1, \\dots, 31
$$

- **第 0 对：** 每个位置 $\\theta = 1$ 弧度，约 57.3°。每 $2\\pi \\approx 6.3$ 个位置转一整圈。
- **第 16 对：** $\\theta = 0.01$ 弧度，约 0.57°。每 ≈ 628 个位置转一整圈。
- **第 31 对：** $\\theta \\approx 0.00013$ 弧度。每 ≈ 47,000 个位置转一整圈。

> **弧度。** 这里的角度用弧度表示，和 \`torch.sin\` / \`torch.cos\` 一样。一整圈是 $360° = 2\\pi \\approx 6.283$ 弧度，所以 1 弧度 $= 180°/\\pi \\approx 57.3°$：即弧长等于半径时所对的角。

**为什么要把快的对和慢的对混在一起？**

- **快的对**在相邻位置之间转得多，所以能区分"相邻"和"隔两个"。但它们转几个位置就绕回来了，所以单靠它们无法区分远距离。
- **慢的对**在相邻位置之间几乎不变，但在长距离上角度会拉开，所以能区分"相隔 100"和"相隔 1000"。

两者合在一起，近和远都能覆盖，就像时钟的秒针、分针和时针。

### q 和 k 按同一规则旋转

两者使用**同一条**规则、**同样的** 32 个速度 $\\theta_i$。只有角度不同，因为角度由各自向量的位置决定：

- 位置 $m$ 的 q：第 $i$ 对转 $m\\,\\theta_i$；
- 位置 $n$ 的 k：第 $i$ 对转 $n\\,\\theta_i$。

同一个 token 自己的 q 和 k 位于同一个位置，所以转过的角度相同。

**为什么规则必须共享。** 在二维例子中（$\\theta = 10°$），位置 5 的 q 和位置 3 的 k 分别转 50° 和 30°，差是 $(5 - 3) \\times 10° = 20°$。两个角度都是"位置 × 同一个 $\\theta$"，所以绝对位置抵消掉，只剩下距离。

如果 q 用 10°，而 k 用 7°：

- 位置 5 的 q 和位置 3 的 k：$50° - 21° = 29°$；
- 位置 105 的 q 和位置 103 的 k：$1050° - 721° = 329°$。

距离相同，结果却不同：分数就不再只依赖相对距离了。

> "同一规则"指的是同一种*旋转*，而不是同一个*向量*。q 和 k 仍然是不同的向量，由 \`c_attn\` 用不同的权重算出；RoPE 只是在之后按各自的位置分别旋转它们。

## 5. 与正弦编码的关系

原始 Transformer 的[正弦编码](#info:sinusoidal_pos_notes)是：

$$
PE(pos, 2i) = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\qquad PE(pos, 2i+1) = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

频率**完全一样**，都是 $1/10000^{2i/d}$，也都是 sin/cos 成对出现。区别在于用法：

- **正弦编码：** sin/cos 值被**加**到 $x$ 上，只在输入处加一次，然后混进残差流。
- **RoPE：** sin/cos 值用来**旋转**（相乘）$q$ 和 $k$，在每一个 attention 层都做，并且不碰其他任何东西。

正弦编码那篇笔记说明了：平移 $k$ 个位置就是一个旋转。RoPE 把这个旋转**直接作用在 q 和 k 上**，所以 $q \\cdot k$ 自动只依赖相对距离。

> 🆕 **base（底数）。** 公式中的 10000 叫作 base（在 Hugging Face 配置里是 \`rope_theta\`）。Llama 2 用 10000；Llama 3 把它提高到 500000，让最慢的几对转得更慢，以支持更长的上下文。

## 6. 超出训练长度

当序列比模型训练时的长度更长时，会发生什么？

### GPT-2：表里没有对应的行

GPT-2 从一张查找表里取位置，每个位置对应一行学出来的向量：

\`\`\`text
pos_emb = nn.Embedding(max_toks, n_embd)    # real GPT-2: 1024 rows × 768
pos_emb(pos)                                 # pos = 0, 1, ..., T-1
\`\`\`

- 位置 0 到 1023 各有自己训练好的一行。
- 位置 1024 没有对应的行，所以查表会报索引错误（index error）。

所以 GPT-2 一次**最多只能看到 1024 个 token**。而且由于每一行都是单独学出来的，加一行新的也没用：它从未被训练过，模型不会知道它是什么意思。

### GPT-2 怎样生成超过 1024 个 token：滑动窗口

生成时，上下文还包括模型目前已经生成的 token，所以 prompt + 生成的 token 很快就会达到上限。因此生成代码在每一步都把输入裁剪成**最近的** 1024 个 token（nanoGPT 的 \`generate\` 会裁剪到 \`block_size\`）：

\`\`\`text
idx_cond = idx[:, -1024:]     # keep only the last 1024 tokens (只保留最后 1024 个 token)
logits = model(idx_cond)
\`\`\`

这**不是**把文本切成互不相干的块。分块会把文本切成不重叠的片段：

\`\`\`text
[0 ~ 1023] [1024 ~ 2047] [2048 ~ 2999]
\`\`\`

这样 token 1024 甚至看不到紧挨着它的 token 1023。滑动窗口则是每一步向前移动一个 token：

\`\`\`text
generating token 3000: sees [1976 ~ 2999]
generating token 3001: sees [1977 ~ 3000]
generating token 3002: sees [1978 ~ 3001]
\`\`\`

相邻的窗口共享 1023 个 token，所以附近的上下文总是连续的：每个 token 都能看到它前面的 1023 个 token。

丢掉的是**往前超过 1023 个 token** 的所有内容。生成 token 3000 时，token 0 到 1975 直接不可见，而不是被模糊地记住。如果某个名字在 token 100 处被提到，GPT-2 在 token 3000 处就用不上它。GPT-2 的短期记忆大约只有 1024 个 token。

> 🆕 这个限制是后来的模型大幅推高上下文长度的一个主要原因；如今很多模型支持 128K 甚至更多 token。

### RoPE：任何位置都有对应的角度

没有表，只有一条规则："位置 $m$ 的第 $i$ 对转 $m\\,\\theta_i$"。位置 5000 可以直接代入，所以不会出错。再结合相对距离性质（第 3 节），两个相隔 3 的 token，无论在位置 10 还是位置 5000，分数中的位置部分都一样。

**但"能算"不等于"好用"。** 一个在长度 4096 上训练的模型，在长度 32,000 上运行时通常会明显变差。原因在于最慢的那几对：

- 训练时它们只转过很小的角度。第 31 对在 4096 个位置内大约只转 0.55 弧度，不到一整圈的十分之一。
- 再往后，它们会达到**训练中从未见过的角度**，模型不知道该如何解读。

> 🆕 因此长上下文模型会加上额外的技巧，比如更大的 base（Llama 3 用 500000），或者对位置做插值/重新缩放（Position Interpolation、YaRN）。每种方法的思路都是：让远处的角度保持在训练时见过的范围内。

简而言之：GPT-2 一次永远看不到超过 1024 个 token，滑动窗口只是通过丢弃较早的文本来掩盖这一点；RoPE 模型可以接受更长的输入，但需要做些调整才能在那里保持准确。

## 参考文献

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)**

   Su et al. 提出了 RoPE，并证明旋转后的 $q \\cdot k$ 只依赖相对位置。

   [RoFormer ↗](https://arxiv.org/abs/2104.09864)

2. **Extending Context Window of Large Language Models via Positional Interpolation (2023)**

   Chen et al. 把位置按比例缩小，让更长的输入落在 RoPE 角度已训练过的范围内。

   [Position Interpolation ↗](https://arxiv.org/abs/2306.15595)

3. **YaRN: Efficient Context Window Extension of Large Language Models (2023)**

   Peng et al. 对快的对和慢的对用不同方式重新缩放 RoPE 频率，以扩展上下文长度。

   [YaRN ↗](https://arxiv.org/abs/2309.00071)
`,
    },
  },
};
