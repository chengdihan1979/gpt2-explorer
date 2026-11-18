// Notes: Token embedding, weight tying, embedding dropout.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  embedding_tok: {
    en: {
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
    zh: {
      title: "Token Embedding",
      md: `
**把 token ID 映射成稠密向量。**

## 形状

- **权重矩阵：** $W \\in \\mathbb{R}^{\\text{vocab\\_size} \\times n\\_embd}$，词表里每个 token 一行。
- **输入 ID：** $\\text{input\\_ids} \\in \\mathbb{Z}^{B \\times T}$
- **Embedding 查表：** $\\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$

### 参数量

$$
\\text{params} = \\text{vocab\\_size} \\times n\\_embd
$$

> GPT-2 small：$50{,}257 \\times 768 \\approx 3.86 \\times 10^7$

## 训练

- $W$ 通过反向传播学习。
- 只有 batch 中出现的 token 对应的行会被更新。
- GPT-2 把 $W$ 和输出 softmax 的权重（\`lm_head\`）绑在一起。[什么是 weight tying？](#info:weight_tying_notes)

## 为什么用 embedding，而不是 one-hot 编码？

one-hot 的 token 向量长度是 $\\text{vocab\\_size}$，而且极其稀疏。\`nn.Embedding\` 等价于用 one-hot 向量乘以 $W$，但它：

- 避免创建巨大的稀疏向量，
- 学到稠密的、带语义的表示，相关的 token 在 embedding 空间里会靠得很近。

## 参考资料

1. **PyTorch 文档**

   \`nn.Embedding\` 把 embedding 定义为一个查找表，它 "stores embeddings of a fixed dictionary and size"，并且 "retrieved using indices."

   [nn.Embedding ↗](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

2. **GPT-2 技术报告（Input Representation）**

   把 GPT-2 的输入描述为 token embedding + position embedding（求和），说明 token id 在进入网络堆叠之前，先被转换成 embedding。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

`,
    },
  },

  llama_embedding_notes: {
    en: {
      title: "LLaMA token embedding: no position embedding",
      md: `
**LLaMA's token embedding works exactly like [GPT-2's](#info:embedding_tok): each token id selects one row of a table. What changed is around it: the table is much larger, and there is no position embedding at all.**

## The lookup: the same, at a larger size

\`\`\`python
self.tok_emb = nn.Embedding(vocab_size, n_embd)   # weight (V, C)
x = self.tok_emb(idx)                             # (B, T) -> (B, T, C)
\`\`\`

| | Table shape | Parameters |
|---|---|---|
| GPT-2 small | $50257 \\times 768$ | 38.6M |
| LLaMA 3 8B | $128256 \\times 4096$ | 525M |

The rows are the [bigger vocabulary](#info:llama_tokenizer_notes) (128,256 tokens); the columns are the model width (4096).

## How the table is learned

Exactly as in GPT-2:

1. **Random at the start.** Every entry is drawn from a normal distribution. This site's code uses PyTorch's default for \`nn.Embedding\` (standard deviation 1); the Hugging Face LLaMA implementation uses 0.02, as GPT-2 did. Meta's papers do not specify it.
2. **Trained together with everything else.** There is no separate step: each \`loss.backward()\` computes a gradient for the table too, and \`optimizer.step()\` updates it.
3. **After training,** tokens with similar meanings end up with nearby vectors. Nobody tells the model which tokens are similar; it falls out of next-token prediction.

LLaMA 3's new tokenizer numbers its tokens differently from LLaMA 2's, so its table was also trained from scratch, not copied from LLaMA 2.

### Where the gradient comes from: one difference from GPT-2

- **GPT-2** uses [weight tying](#info:weight_tying_notes): the embedding table is also the LM head's weight. The LM head gives **every** one of the 50,257 rows a gradient at every step (the softmax has an output for every token), so every row is updated every step.
- **LLaMA 3 8B** does not tie them: the table is used only at the input. A row gets a gradient **only when its token appears in the batch's input**; all other rows get 0 at that step.

So in LLaMA, the embedding of a rare token rarely gets a gradient. (In between, [AdamW](#info:adamw_notes) still moves it a little: its momentum $m$ fades out over the following steps, and weight decay shrinks it.) This is the "input side only touches the looked-up rows" effect from the [weight tying](#info:weight_tying_notes) note, without the output-side gradient that GPT-2 adds on top.

## No position embedding

The first line of the forward pass, side by side:

\`\`\`python
# GPT-2: token vector + position vector
x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

# LLaMA: token vector only
x = self.tok_emb(idx)
\`\`\`

GPT-2's [position embedding](#info:embedding_pos) table (1024 × 768) is gone. The $x$ that enters the first block says **which** token is at each position, but not **where** it is.

### Position information is still needed

Attention compares every pair of tokens and takes a weighted sum, and that computation by itself ignores order: reorder the earlier tokens and the result for the last token does not change. For example, with one attention layer and no position information, the last token's output is identical for the inputs \`[5, 17, 42, 8]\` and \`[42, 5, 17, 8]\`. The model could not tell "dog bites man" from "man bites dog". (The [position embedding](#info:embedding_pos) note shows why with permutation matrices, and why the causal mask alone is a weak substitute.)

So LLaMA still needs position information; it only puts it somewhere else.

### Where position goes instead: RoPE inside attention

- **GPT-2** adds a position **vector** to $x$ once, at the start. From then on, position is mixed into $x$ and flows through every layer with it.
- **LLaMA** keeps position out of $x$. In **every** attention layer, after computing $q$ and $k$, it **rotates** them by an angle that depends on the position ([RoPE](#info:rope_notes)). As a result, the score $q \\cdot k$ depends only on the **relative distance** between the two tokens: a query at position 5 and a key at position 3 give the same score as positions 12 and 10.

This is why the diagram labels the node "Token Embeddings (no position embedding)", and why RoPE appears as a step inside the attention sub-diagram.

## References

1. **LLaMA: Open and Efficient Foundation Language Models.** Touvron et al., 2023. [arXiv:2302.13971 ↗](https://arxiv.org/abs/2302.13971)
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding.** Su et al., 2021. [arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)
`,
    },
    zh: {
      title: "LLaMA 的 token embedding：没有位置 embedding",
      md: `
**LLaMA 的 token embedding 和 [GPT-2 的](#info:embedding_tok)完全一样：用每个 token id 去表里取一行。变化在它周围：表大了很多，而且完全没有位置 embedding。**

## 查表：一样的做法，更大的表

\`\`\`python
self.tok_emb = nn.Embedding(vocab_size, n_embd)   # weight (V, C)
x = self.tok_emb(idx)                             # (B, T) -> (B, T, C)
\`\`\`

| | 表的形状 | 参数量 |
|---|---|---|
| GPT-2 small | $50257 \\times 768$ | 3860 万 |
| LLaMA 3 8B | $128256 \\times 4096$ | 5.25 亿 |

行数是[更大的词表](#info:llama_tokenizer_notes)（128,256 个 token），列数是模型宽度（4096）。

## 这张表是怎么学出来的

和 GPT-2 完全一样：

1. **开始时随机初始化。** 表里每个数都从正态分布里随机抽取。这个网站的代码用的是 PyTorch \`nn.Embedding\` 的默认值（标准差 1）；Hugging Face 的 LLaMA 实现用 0.02，和 GPT-2 一样。Meta 的论文没有写明。
2. **和其他参数一起训练。** 没有单独的训练步骤：每次 \`loss.backward()\` 也会算出这张表的梯度，\`optimizer.step()\` 再更新它。
3. **训练结束后，** 意思相近的 token，向量也会靠得近。没有人告诉模型哪些词意思相近，这是 next-token prediction 训练出来的结果。

LLaMA 3 换了新的 tokenizer，token 的编号和 LLaMA 2 完全不同，所以它的表也是从头训练的，没有沿用 LLaMA 2 的。

### 梯度从哪里来：和 GPT-2 的一个区别

- **GPT-2** 用了 [weight tying](#info:weight_tying_notes)：embedding 表同时也是 LM head 的权重。LM head 每一步都会给 50,257 行中的**每一行**梯度（softmax 对每个词都有输出），所以每一行每一步都会被更新。
- **LLaMA 3 8B** 不共享：这张表只在输入端用到。一行**只有在它的 token 出现在这个 batch 的输入里时**才会拿到梯度；其他行在这一步的梯度是 0。

所以在 LLaMA 里，很少出现的 token，它的 embedding 很少拿到梯度。（没有梯度的那些步，[AdamW](#info:adamw_notes) 仍然会让它动一点：动量 $m$ 在之后几步里慢慢衰减，weight decay 也会把它缩小一点。）这就是 [weight tying](#info:weight_tying_notes) note 里讲的"输入端只碰到查到的那几行"，只是 LLaMA 没有 GPT-2 那份来自输出端的梯度来补。

## 没有位置 embedding

把 forward 的第一行并排比较：

\`\`\`python
# GPT-2：token 向量 + 位置向量
x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

# LLaMA：只有 token 向量
x = self.tok_emb(idx)
\`\`\`

GPT-2 的[位置 embedding](#info:embedding_pos) 表（1024 × 768）没有了。进入第一个 block 的 $x$ 里，只有每个位置上是**哪个** token，没有它在**第几个**位置。

### 位置信息仍然是必需的

attention 计算的是每对 token 之间的相似度，再加权求和，这个计算本身不关心顺序：把前面的 token 换个顺序，最后一个 token 的结果不变。例如，只用一层不带位置信息的 attention，输入 \`[5, 17, 42, 8]\` 和 \`[42, 5, 17, 8]\` 时，最后一个 token 的输出完全相同。模型分不清 "狗咬人" 和 "人咬狗"。（[位置 embedding](#info:embedding_pos) note 里用置换矩阵说明了原因，也解释了为什么光靠 causal mask 不够。）

所以 LLaMA 仍然需要位置信息，只是放到了别的地方。

### 位置改放在哪里：attention 里的 RoPE

- **GPT-2** 在最开始，把一个位置**向量**加到 $x$ 上，只加一次。之后位置信息混在 $x$ 里，跟着 $x$ 流过所有层。
- **LLaMA** 不把位置放进 $x$。在**每一层** attention 里，算出 $q$ 和 $k$ 之后，按位置把它们**旋转**一个角度（[RoPE](#info:rope_notes)）。这样分数 $q \\cdot k$ 只取决于两个 token 的**相对距离**：位置 5 的 query 和位置 3 的 key，算出的分数和位置 12、10 的一样。

这就是图上把这个方框标成 "Token Embeddings (no position embedding)" 的原因，也是 RoPE 出现在 attention sub-diagram 里的原因。

## 参考资料

1. **LLaMA: Open and Efficient Foundation Language Models.** Touvron 等人，2023。[arXiv:2302.13971 ↗](https://arxiv.org/abs/2302.13971)
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding.** Su 等人，2021。[arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)
`,
    },
  },

  weight_tying_notes: {
    en: {
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

A tiny example shows where each part comes from. Drop all the blocks so the embedding feeds straight into \`lm_head\`, and use a vocabulary of 3 words with \`n_embd\` = 4:

\`\`\`text
E = [[ 0.5, 0,   0.5, 0  ],    # word 0
     [ 0,   0.5, 0,   0.5],    # word 1
     [-0.5, 0,   0.5, 0  ]]    # word 2

input token = word 0   →  x = E[0] = [0.5, 0, 0.5, 0]     ← input side uses E
logits = x @ E.T = [0.5, 0, 0]                            ← output side uses E again
p = softmax(logits) = [0.452, 0.274, 0.274]
correct next token = word 2,  loss = 1.294
\`\`\`

The gradient of the loss with respect to the logits is $p - y$ ([why?](#hover:cross_entropy_notes)):

\`\`\`text
p − y = [0.452, 0.274, 0.274] − [0, 0, 1] = [0.452, 0.274, -0.726]
\`\`\`

### Output side: every row gets a gradient

Since $\\text{logit}_i = x \\cdot E[i]$, row $i$ receives $(p_i - y_i)\\, x$:

\`\`\`text
row 0:   0.452 × [0.5, 0, 0.5, 0] = [ 0.226, 0,  0.226, 0]
row 1:   0.274 × [0.5, 0, 0.5, 0] = [ 0.137, 0,  0.137, 0]
row 2:  -0.726 × [0.5, 0, 0.5, 0] = [-0.363, 0, -0.363, 0]
\`\`\`

An update subtracts the gradient. Row 2 (the correct word) has a negative gradient, so it moves **toward** $x$. Rows 0 and 1 (wrong words) are pushed **away** from $x$.

### Input side: only the looked-up row

$x$ is a lookup of row 0, so only row 0 gets this gradient. $x$ affects every logit, so its gradient is a sum over all word vectors, weighted by $p_i - y_i$:

\`\`\`text
dL/dx = 0.452 × E[0] + 0.274 × E[1] − 0.726 × E[2]
      = [0.589, 0.137, -0.137, 0.137]       → added to row 0 only
rows 1 and 2: 0
\`\`\`

### The two add up

\`\`\`text
         output side             input side                       total (E.grad)
row 0:  [ 0.226, 0, 0.226, 0] + [0.589, 0.137, -0.137, 0.137] = [ 0.815, 0.137, 0.089, 0.137]
row 1:  [ 0.137, 0, 0.137, 0] + 0                              = [ 0.137, 0,     0.137, 0    ]
row 2:  [-0.363, 0,-0.363, 0] + 0                              = [-0.363, 0,    -0.363, 0    ]
\`\`\`

Checked in PyTorch: computing the gradient with only one side using $E$, then the other, and adding the two gives exactly \`E.grad\` from the tied version.

### What this means for training

- **Input side:** only the rows of tokens that appear in the batch get a gradient.
- **Output side:** the softmax gives a gradient to **every row** of the vocabulary (50,304 in this site's code) every step. A wrong word's push is proportional to its probability $p_i$.

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
    zh: {
      title: "Weight Tying（权重共享）",
      md: `
**Weight tying（输入和输出共用同一张 embedding 表）**

Weight tying 让输出 softmax 的权重等于输入的 token embedding 权重：

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# tie lm_head weight to tok_emb    (把 lm_head 的权重绑到 tok_emb 上)
self.lm_head.weight = self.tok_emb.weight
\`\`\`

> 这个想法来自论文 [Using the Output Embedding to Improve Language Models ↗](https://arxiv.org/abs/1608.05859)

## 它做了什么

最后一个 block 之后，每个位置都有一个最终向量 $h \\in \\mathbb{R}^{n\\_embd}$。\`lm_head\` 拿 $h$ 和词表里每个词的那一行做点积，给每个词打一个分（logit）：

$$
\\text{logit}_i = h \\cdot W[i]
$$

$h$ 和第 $i$ 行的方向越接近，词 $i$ 的分数就越高。共享之后，第 $i$ 行同时扮演两个角色：

- **输入端：** 读入词 $i$ 时，用来表示它的向量。
- **输出端：** 预测下一个词是不是 $i$ 时，模型拿来比对的模板。

> 要预测 \`Ġcat\`，模型就得把 $h$ 推向 \`Ġcat\` 自己的 embedding。

## 为什么这么做

- **参数更少：** 两个矩阵 \`V*d\` + \`d*V\` 变成一个 \`V*d\`，大约少了一半。对 GPT-2 small 来说省了 $50{,}257 \\times 768 \\approx 38.6\\text{M}$ 个参数：模型约 124M，而不是约 163M。
- **泛化更好：** 起到有益的正则化作用，通常能降低 perplexity。

## 梯度怎么流

这行赋值并没有复制任何东西：两个层指向的是**同一个 \`Parameter\` 对象**，所以这个共享矩阵收到的是两处用法梯度的**和**。

用一个极简的例子看清楚每一份梯度从哪来。去掉所有 block，让 embedding 直接接 \`lm_head\`；词表只有 3 个词，\`n_embd\` = 4：

\`\`\`text
E = [[ 0.5, 0,   0.5, 0  ],    # word 0
     [ 0,   0.5, 0,   0.5],    # word 1
     [-0.5, 0,   0.5, 0  ]]    # word 2

输入 token = word 0   →  x = E[0] = [0.5, 0, 0.5, 0]     ← 输入端用了 E
logits = x @ E.T = [0.5, 0, 0]                           ← 输出端又用了 E
p = softmax(logits) = [0.452, 0.274, 0.274]
正确的下一个 token = word 2，loss = 1.294
\`\`\`

loss 对 logits 的梯度是 $p - y$（[为什么？](#hover:cross_entropy_notes)）：

\`\`\`text
p − y = [0.452, 0.274, 0.274] − [0, 0, 1] = [0.452, 0.274, -0.726]
\`\`\`

### 输出端：每一行都有梯度

因为 $\\text{logit}_i = x \\cdot E[i]$，第 $i$ 行收到的梯度是 $(p_i - y_i)\\, x$：

\`\`\`text
row 0:   0.452 × [0.5, 0, 0.5, 0] = [ 0.226, 0,  0.226, 0]
row 1:   0.274 × [0.5, 0, 0.5, 0] = [ 0.137, 0,  0.137, 0]
row 2:  -0.726 × [0.5, 0, 0.5, 0] = [-0.363, 0, -0.363, 0]
\`\`\`

更新时减去梯度。row 2（正确答案）的梯度是负的，所以它**朝 $x$ 的方向移动**；row 0、row 1 是错误答案，**被推离 $x$**。

### 输入端：只有被查的那一行

$x$ 是查表查出的 row 0，所以这份梯度只加到 row 0。$x$ 影响了每个 logit，所以它的梯度是所有词向量按 $p_i - y_i$ 加权求和：

\`\`\`text
dL/dx = 0.452 × E[0] + 0.274 × E[1] − 0.726 × E[2]
      = [0.589, 0.137, -0.137, 0.137]       → 只加到 row 0
row 1、row 2：0
\`\`\`

### 两份相加

\`\`\`text
         输出端                   输入端                           合计 (E.grad)
row 0:  [ 0.226, 0, 0.226, 0] + [0.589, 0.137, -0.137, 0.137] = [ 0.815, 0.137, 0.089, 0.137]
row 1:  [ 0.137, 0, 0.137, 0] + 0                              = [ 0.137, 0,     0.137, 0    ]
row 2:  [-0.363, 0,-0.363, 0] + 0                              = [-0.363, 0,    -0.363, 0    ]
\`\`\`

在 PyTorch 里验证过：分别只让一端用 $E$ 算出梯度，两份加起来，正好等于共享版本的 \`E.grad\`。

### 对训练意味着什么

- **输入端：** 只有在 batch 里出现过的 token，对应的行才有梯度。
- **输出端：** softmax 每一步都给词表的**每一行**梯度（这个网站的代码里是 50,304 行）。错误词被推开的力度和它的概率 $p_i$ 成正比。

所以共享之后，即使是罕见词的 embedding，也能通过输出端一直得到训练。

## 补充

- 很多 GPT-2 实现在共享权重时，\`lm_head\` 都用 \`bias=False\`。
- 共享要求形状完全相同：\`lm_head.weight.shape == tok_emb.weight.shape == (vocab_size, n_embd)\`。
- 不需要转置形状就能对上，是因为 \`nn.Linear(n_embd, vocab_size)\` 把权重存成 \`(out, in)\` = \`(vocab_size, n_embd)\`，和 \`nn.Embedding(vocab_size, n_embd)\` 的布局一样。
- 原版 GPT-2 也共享权重：OpenAI 的代码直接用 token embedding 表 \`wte\` 算 logits：\`logits = tf.matmul(h_flat, wte, transpose_b=True)\`。

## 🆕 近期发展

共享是一种取舍。输入端的那一行应该表达"这个词是什么"；输出端的那一行应该表达"什么样的上下文后面会接这个词"。强迫一个向量同时做两件事，是一种约束。

- **大模型通常不共享：** Llama 3.1 8B 和 70B，Qwen2.5 7B 和 72B。在这个规模下，embedding 表只占参数的一小部分，省下的不多。
- **小模型经常共享：** Llama 3.2 1B 和 3B，Qwen2.5 0.5B 和 3B，Gemma 2。这些模型里 embedding 表占参数的比例很大。

> 在 Hugging Face 的模型 config 里，对应的是 \`tie_word_embeddings\` 这个开关。

## 参考资料

1. **Using the Output Embedding to Improve Language Models**

   Press & Wolf，2016。证明了共享输入和输出 embedding 能减少参数并降低 perplexity。

   [arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)

2. **Tying Word Vectors and Word Classifiers**

   Inan、Khosravi & Socher，2016。从语言建模的一个 loss 框架出发，独立提出了同样的想法。

   [arXiv:1611.01462 ↗](https://arxiv.org/abs/1611.01462)

3. **GPT-2 参考模型代码**

   OpenAI 原版的 model.py，里面用 \`wte\` 计算 logits。

   [openai/gpt-2 model.py ↗](https://github.com/openai/gpt-2/blob/master/src/model.py)
`,
    },
  },

  emb_dropout_notes: {
    en: {
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
    zh: {
      title: "Embedding Dropout",
      md: `
**在 token embedding 和位置 embedding 相加之后，GPT-2 会先过一个 dropout 层**，再把这个表示送进第一个 Transformer block。

## 为什么在这里用 dropout？

- 在训练时随机把 embedding 向量的一部分置零，防止过拟合。
- 迫使模型不过分依赖某些特定的 embedding 维度。
- 在输入阶段就提供正则化，和 Transformer block 内部的 dropout（例如 attention 层和前馈层之后的 dropout）互相补充。

## 例子

假设 token embedding 和位置 embedding 相加之后，某个 token 得到这个 6 维向量：

$$
h = [0.8, -0.5, 1.2, 0.0, -0.3, 0.9]
$$

这就是第一个 Transformer block 的原始输入。

### 应用 dropout（比如 $p = 0.5$）

Dropout 以概率 $p$ 随机"遮掉"一些维度。保留下来的维度会乘以 $1/(1-p)$ 重新缩放，以保持期望不变。

Dropout mask 示例：

$$
m = [1, 0, 1, 0, 1, 1]
$$

应用 mask 并缩放（缩放因子 = 2，因为 $1/(1-0.5) = 2$）：

$$
h' = h \\odot m \\times 2 = [1.6, 0.0, 2.4, 0.0, -0.6, 1.8]
$$

所以第二维和第四维被"丢掉"（置零），其余的都翻倍。

## 为什么这有帮助

- **提高泛化能力、防止过拟合：** 不让模型把自己和特定维度"焊死"，帮助它学到更稳健的模式。如果模型学会了过分依赖，比如说，第 3 维，dropout 会迫使它也去用其他维度，因为第 3 维在训练中可能会消失。
- **对 embedding 做正则化：** token embedding 不会编码脆弱的、只靠单一维度的"捷径"；它们需要把有用的信号更均匀地分散开。

## 参考资料

1. **原始 Transformer 论文（dropout 在这里被定义）**

   Vaswani et al. 明确写道："we apply dropout to the sums of the embeddings and the positional encodings."。GPT-2 沿用了这个做法（使用学习得到的绝对位置）。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)
`,
    },
  },
};
