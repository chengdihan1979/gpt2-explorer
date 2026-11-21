// Notes: Causal self-attention.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  attention_linear_notes: {
    en: {
      title: "Attention linear",
      md: `
## What it is

\`self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)\` is a single fused projection that computes **Q, K, V** in one matmul. It is equivalent to stacking three \`Linear(n_embd, n_embd)\` layers:

$$
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}
$$

## Parameter shapes

- **Weight:** $(3 \\times n\\_embd,\\ n\\_embd)$
- **Bias:** $(3 \\times n\\_embd)$

Mathematically, $W$ maps $768 \\to 2304$, so it holds $768 \\times 2304$ numbers. PyTorch's \`nn.Linear\` stores it as **(out, in)** $= (2304, 768)$ and computes $x\\,W^\\top + b$, which is why the weight shape above looks transposed.

## One set per block, shared by all tokens

- **Separate per block.** Each of the 12 blocks has its own \`c_attn\`, with its own $W$ and $b$, initialized independently and trained independently. Nothing is shared between blocks, so each layer learns its own way of computing q, k and v.
- **Shared within a block.** Inside one block, every token, from every sequence in the batch and at every position, is multiplied by the **same** $W$ and shifted by the same $b$. \`c_attn\` has no notion of position: it applies the same rule to every position, so two positions' q, k and v differ only because their input vectors differ. For example, take the two occurrences of "the" in "the cat and the dog":
  - **In block 1**, they have the same token embedding, so the only difference in their inputs is their position embeddings.
  - **In later blocks**, each "the" has also gathered different context through attention: the first one sees only itself, the second sees "the cat and the". Their inputs drift further apart, and the difference now comes mostly from context, not just position.

## What it does to the tensor

1. Input $x \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$.
2. \`qkv = c_attn(x)\` gives $qkv \\in \\mathbb{R}^{B \\times T \\times 3n\\_embd}$. Each position's 768-dim vector is projected **on its own**, with the same $W$: no information moves between positions yet. That starts only at $q K^\\top$.
3. \`q, k, v = qkv.split(n_embd, 2)\` gives $q, k, v \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$.
4. Split q, k and v into heads with $hs = n\\_embd / n\\_head$: \`view(B, T, n_head, hs).transpose(1, 2)\` gives $(B,\\ n\\_head,\\ T,\\ hs)$.

## Why fuse Q/K/V?

Fewer kernel launches and better cache use: one big GEMM instead of three separate linears. The results are identical to three separate linears.

## Parameter count

- **Weights:** $3 \\times n\\_embd^2$
- **Bias:** $3 \\times n\\_embd$

> Example ($n\\_embd = 768$): weights $3 \\cdot 768^2 = 1{,}769{,}472$, bias $3 \\cdot 768 = 2{,}304$.

## PyTorch default initialization

By default, PyTorch's \`nn.Linear\` initializes weights with **Kaiming uniform** and biases with a plain uniform with the same bound.

### Weights

PyTorch initializes \`Linear.weight\` with Kaiming (He) uniform:

\`\`\`python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
\`\`\`

Passing \`a=math.sqrt(5)\` is a PyTorch trick so that the resulting uniform range becomes $\\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}}, \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)$, numerically the same range you'd get from a plain uniform with that bound. This was done for backward compatibility when they refactored to use \`kaiming_uniform_\` ([forum post ↗](https://discuss.pytorch.org/t/nn-linear-default-weight-initialisation-assumes-leaky-relu-activation/46712/2)).

$$
W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right), \\quad \\text{fan\\_in} = n\\_embd
$$

For $n\\_embd = 768$ the bound is $\\approx \\pm 0.0361$.

### Biases

The bias is not Kaiming; it's sampled explicitly from a plain uniform with the same bound:

\`\`\`python
fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
bound = 1 / sqrt(fan_in)
init.uniform_(self.bias, -bound, bound)
\`\`\`

$$
\\Rightarrow b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)
$$

> So: weights use Kaiming uniform configured to yield the same range as a simple uniform, while biases use that simple uniform directly.

**Note:** many Transformer/GPT-2 implementations override this and use a normal init $\\mathcal{N}(0, 0.02^2)$ for linear layers and embeddings. The original GPT-2 code (OpenAI's TensorFlow \`model.py\`) does exactly that for \`c_attn\`: weights $\\sim \\mathcal{N}(0, 0.02^2)$ and biases **all zero**. The site's code doesn't override anything, so it uses the PyTorch defaults above.

## Why random, and why small?

- **Why not all zeros?** If $W = 0$, every position gets the same q, k, v, and every entry of $W$ receives the same gradient, so the entries stay identical forever and can never learn different things. Random values break this symmetry.
- **Why small?** Small weights give small q and k, so the scores $q \\cdot k$ start near 0 and the softmax starts close to uniform. That keeps it out of saturation, so gradients flow from the first step.

Everything q, k and v come to mean is learned from these random starting points.

## References

1. **Kaiming initialization**

   Introduces the initialization for rectifiers (ReLU/LeakyReLU) and derives the variance-preserving scheme.

   [He et al., 2015 ↗](https://arxiv.org/pdf/1502.01852.pdf)

2. **PyTorch defaults**

   The PyTorch init code that shows the exact behavior.

   [PyTorch init code ↗](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50)

3. **GPT-2 / HF Transformers practice (overriding PyTorch defaults)**

   GPT-2 (HF Transformers): weight matrices are initialized from a (truncated) Normal(0, 0.02), controlled by \`initializer_range=0.02\` in the config. This is the widely used default when you instantiate GPT-2 from Transformers.

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

4. **General Transformer init policy**

   The same policy is described in general Transformer init notes (HF and others): "weights ~ Normal(0, initializer_range) (often 0.02), biases 0; LayerNorm γ=1, β=0."

   [HF Transformers Modeling Utils ↗](https://huggingface.co/transformers/v3.1.0/internal/modeling_utils.html)
`,
    },
    zh: {
      title: "Attention 的 Linear（c_attn）",
      md: `
## 它是什么

\`self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)\` 是一个合并的投影，用一次矩阵乘法同时算出 **Q、K、V**。它等价于把三个 \`Linear(n_embd, n_embd)\` 摞在一起：

$$
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}
$$

## 参数形状

- **Weight：** $(3 \\times n\\_embd,\\ n\\_embd)$
- **Bias：** $(3 \\times n\\_embd)$

从数学上看，$W$ 把 $768$ 维映射到 $2304$ 维，所以有 $768 \\times 2304$ 个数。PyTorch 的 \`nn.Linear\` 把它存成 **(out, in)** $= (2304, 768)$，计算的是 $x\\,W^\\top + b$，所以上面的 weight 形状看起来是转置过的。

## 每个 block 一套，所有 token 共用

- **每个 block 各自一套。** 12 个 block 各有自己的 \`c_attn\`，各有自己的 $W$ 和 $b$，各自独立初始化、独立训练。block 之间不共享任何东西，所以每一层都学自己计算 q、k、v 的方式。
- **同一个 block 内部共用。** 在一个 block 里，每个 token，无论来自 batch 里哪个序列、处在哪个位置，都乘以**同一个** $W$，加上同一个 $b$。\`c_attn\` 没有位置的概念：它对每个位置用同一条规则，所以两个位置的 q、k、v 之所以不同，只是因为它们的输入向量不同。比如 "the cat and the dog" 里出现了两次 "the"：
  - **在第 1 个 block 里**，两者的 token embedding 相同，所以输入的唯一区别是 position embedding。
  - **在后面的 block 里**，每个 "the" 还通过 attention 收集到了不同的上下文：第一个只看得到自己，第二个看得到 "the cat and the"。它们的输入差得越来越远，差别主要来自上下文，而不只是位置。

## 它对张量做了什么

1. 输入 $x \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$。
2. \`qkv = c_attn(x)\` 得到 $qkv \\in \\mathbb{R}^{B \\times T \\times 3n\\_embd}$。每个位置的 768 维向量都用同一个 $W$ **单独**投影：位置之间还没有任何信息流动。信息流动要到 $q K^\\top$ 才开始。
3. \`q, k, v = qkv.split(n_embd, 2)\` 得到 $q, k, v \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$。
4. 把 q、k、v 拆成多个 head，$hs = n\\_embd / n\\_head$：\`view(B, T, n_head, hs).transpose(1, 2)\` 得到 $(B,\\ n\\_head,\\ T,\\ hs)$。

## 为什么把 Q/K/V 合并？

kernel 启动次数更少，缓存利用更好：一次大的 GEMM（矩阵乘法）代替三个独立的 Linear。结果和三个独立的 Linear 完全相同。

## 参数量

- **Weights：** $3 \\times n\\_embd^2$
- **Bias：** $3 \\times n\\_embd$

> 例子（$n\\_embd = 768$）：weights $3 \\cdot 768^2 = 1{,}769{,}472$，bias $3 \\cdot 768 = 2{,}304$。

## PyTorch 的默认初始化

PyTorch 的 \`nn.Linear\` 默认用 **Kaiming uniform** 初始化 weight，bias 则用同样边界的普通均匀分布。

### Weights

PyTorch 用 Kaiming（He）uniform 初始化 \`Linear.weight\`：

\`\`\`python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
\`\`\`

传入 \`a=math.sqrt(5)\` 是 PyTorch 的一个技巧，让得到的均匀分布范围变成 $\\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}}, \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)$，在数值上和直接用这个边界的普通均匀分布一样。这是为了在重构成 \`kaiming_uniform_\` 时保持向后兼容（[论坛帖子 ↗](https://discuss.pytorch.org/t/nn-linear-default-weight-initialisation-assumes-leaky-relu-activation/46712/2)）。

$$
W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right), \\quad \\text{fan\\_in} = n\\_embd
$$

$n\\_embd = 768$ 时，边界约为 $\\pm 0.0361$。

### Biases

bias 不用 Kaiming，而是直接从同样边界的普通均匀分布里抽：

\`\`\`python
fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
bound = 1 / sqrt(fan_in)
init.uniform_(self.bias, -bound, bound)
\`\`\`

$$
\\Rightarrow b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)
$$

> 所以：weight 用的是 Kaiming uniform，但参数设置成和普通均匀分布同样的范围；bias 直接用这个普通均匀分布。

**注意：** 很多 Transformer/GPT-2 实现会覆盖这个默认值，对 Linear 层和 embedding 用正态分布 $\\mathcal{N}(0, 0.02^2)$ 初始化。原版 GPT-2 代码（OpenAI 的 TensorFlow \`model.py\`）对 \`c_attn\` 正是这样做的：weight $\\sim \\mathcal{N}(0, 0.02^2)$，bias **全为 0**。本站代码没有覆盖任何初始化，所以用的是上面的 PyTorch 默认值。

## 为什么要随机，为什么要小？

- **为什么不全设为 0？** 如果 $W = 0$，每个位置得到的 q、k、v 都一样，$W$ 的每个元素收到的梯度也一样，于是它们永远保持相同，没法学到不同的东西。随机值打破了这种对称性。
- **为什么要小？** weight 小，q 和 k 就小，分数 $q \\cdot k$ 一开始接近 0，softmax 一开始接近均匀分布。这样它就不会饱和，从第一步起梯度就能流动。

q、k、v 最终代表的一切含义，都是从这些随机起点学出来的。

## 参考资料

1. **Kaiming 初始化**

   提出了针对整流激活函数（ReLU/LeakyReLU）的初始化方法，并推导出保持方差的方案。

   [He et al., 2015 ↗](https://arxiv.org/pdf/1502.01852.pdf)

2. **PyTorch 默认值**

   PyTorch 的初始化代码，展示了确切的行为。

   [PyTorch init code ↗](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50)

3. **GPT-2 / HF Transformers 的做法（覆盖 PyTorch 默认值）**

   GPT-2（HF Transformers）：权重矩阵从（截断的）Normal(0, 0.02) 初始化，由 config 里的 \`initializer_range=0.02\` 控制。用 Transformers 实例化 GPT-2 时，这是广泛使用的默认值。

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

4. **通用的 Transformer 初始化策略**

   通用的 Transformer 初始化说明（HF 等）里也描述了同样的策略："weights ~ Normal(0, initializer_range) (often 0.02), biases 0; LayerNorm γ=1, β=0."

   [HF Transformers Modeling Utils ↗](https://huggingface.co/transformers/v3.1.0/internal/modeling_utils.html)
`,
    },
  },

  query_view_notes: {
    en: {
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
    zh: {
      title: "Query View（拆成多个 head）",
      md: `
## Query projection

\`q\` 是合并投影的第一部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 q 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
q of one token (768 dims):            # 一个 token 的 q（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 q 拆成 3 个 head，每个 2 维。

\`\`\`text
q = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

k 和 v 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's q, head 1's q]
token 1: [head 0's q, head 1's q]
token 2: [head 0's q, head 1's q]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's q, token 1's q, token 2's q]
head 1: [token 0's q, token 1's q, token 2's q]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
    },
  },

  key_view_notes: {
    en: {
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
    zh: {
      title: "Key View（拆成多个 head）",
      md: `
## Key projection

\`k\` 是合并投影的第二部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 k 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
k of one token (768 dims):            # 一个 token 的 k（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 k 拆成 3 个 head，每个 2 维。

\`\`\`text
k = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

q 和 v 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's k, head 1's k]
token 1: [head 0's k, head 1's k]
token 2: [head 0's k, head 1's k]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's k, token 1's k, token 2's k]
head 1: [token 0's k, token 1's k, token 2's k]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
    },
  },

  value_view_notes: {
    en: {
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
    zh: {
      title: "Value View（拆成多个 head）",
      md: `
## Value projection

\`v\` 是合并投影的第三部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 v 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
v of one token (768 dims):            # 一个 token 的 v（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 v 拆成 3 个 head，每个 2 维。

\`\`\`text
v = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

q 和 k 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's v, head 1's v]
token 1: [head 0's v, head 1's v]
token 2: [head 0's v, head 1's v]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's v, token 1's v, token 2's v]
head 1: [token 0's v, token 1's v, token 2's v]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
    },
  },

  q_k_product_notes: {
    en: {
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

> At initialization in this site's code, the components of $q$ and $k$ actually have variance about 1/3, not 1 (LayerNorm's output has variance 1, but q and k come after a linear layer). The conclusion is the same: the variance is still proportional to $hs$. Every variance is derived step by step in [Variance at initialization](#info:variance_init_notes).

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
    zh: {
      title: "Q @ K 乘积",
      md: `
## Scaled dot-product attention 详解

\`q\` 和 \`k\` 的形状是 $(B, n\\_head, T, hs)$（每个 head 的 query 和 key）。

- \`k.transpose(-2, -1)\` $\\rightarrow (B, n\\_head, hs, T)$ 交换 key 的最后两个轴，这样每个 query 都能和所有时间步的 key 做点积。
- \`@\` 是批量矩阵乘法：

  $$
  (B, n\\_head, T, hs)\\ @\\ (B, n\\_head, hs, T) \\;\\rightarrow\\; (B, n\\_head, T, T)
  $$

  结果：每个 head 得到一个 $T \\times T$ 的原始 attention 分数矩阵（每一行对应一个 query 位置，每一列对应一个 key 位置）。
- \`* (1.0 / hs**0.5)\` 乘以 $1/\\sqrt{hs}$（$hs$ 是 head size）。这就是 "Scaled Dot-Product Attention" 里的缩放。

## $T \\times T$ 矩阵的含义

第 $(i, j)$ 项是 token $i$ 的 q 和 token $j$ 的 k 的点积，再乘以 $1/\\sqrt{hs}$（原因见下文）：一个数，表示位置 $i$ 对位置 $j$ 有多感兴趣：

$$
\\mathrm{att}[i][j] = \\frac{q_i \\cdot k_j}{\\sqrt{hs}}
$$

这个计算在每个序列、每个 head 内部分别进行。

以 "the river bank"（$T = 3$）在某一个 head 里为例（数字是编的，只为展示结构）：

\`\`\`text
             k_the   k_river   k_bank
q_the     [   1.0     0.2       0.5  ]   # the 给每个词的分数
q_river   [   0.3     1.2       0.4  ]   # river 给每个词的分数
q_bank    [   0.1     2.5       0.8  ]   # bank 给每个词的分数：river 最高
\`\`\`

- **每一行：** 一个 token 作为提问者，给所有 token 打分。
- **每一列：** 一个 token 作为被查找的对象，被所有 token 打分。

> **这是位置之间第一次发生交互。** \`c_attn\`、\`split\`、\`view\` 和 \`transpose\` 都是对每个 token 单独操作的。到了 \`q @ k.transpose(-2, -1)\`，token $i$ 的 q 才第一次遇到**其他** token 的 k。

有两点要注意：

- 这个矩阵仍然是**完整的** $T \\times T$，包括往后看的分数（比如 the → bank）。下一步的 [causal mask](#info:causal_mask_notes) 会把这些去掉。
- 每个 head 都有自己的 $T \\times T$ 矩阵，所以每个序列有 12 个。

> 🆕 现在大多数 LLM 不再把 position embedding 加到 $x$ 上，而是在这一步之前按位置旋转 q 和 k，让每个分数只取决于 $i$ 和 $j$ 之间的距离：[RoPE](#info:rope_notes)。

## 为什么要乘 $1/\\sqrt{hs}$

在 GPT-2 small 里 $hs = 64$，所以这一步就是把每个分数除以 $\\sqrt{64} = 8$。

### 问题：点积随维度增大

一个分数是 $hs$ 个乘积之和：

$$
q \\cdot k = q_1 k_1 + q_2 k_2 + \\cdots + q_{hs} k_{hs}
$$

假设 $q$ 和 $k$ 的各分量相互独立，均值为 0、方差为 1（大致是 LayerNorm 之后的尺度）。那么每一项 $q_i k_i$ 的方差是 1，$hs$ 项加起来，方差就是 $hs$：

$$
\\operatorname{Var}(q \\cdot k) = hs, \\qquad \\operatorname{std}(q \\cdot k) = \\sqrt{hs}
$$

$hs = 64$ 时标准差是 8，所以 $\\pm 8$、$\\pm 16$ 这样的分数很常见。维度越多，分数越大。

> 在本站代码的初始化状态下，$q$ 和 $k$ 各分量的方差其实约为 1/3，而不是 1（LayerNorm 的输出方差是 1，但 q 和 k 还要经过一个线性层）。结论不变：方差仍然和 $hs$ 成正比。每个方差的逐步推导见 [初始化时的方差](#info:variance_init_notes)。

### 为什么分数大了不好：softmax 饱和

- 分数 $[8,\\; 0,\\; -8]$ → softmax $\\approx [0.9997,\\; 0.0003,\\; 0.0000]$：几乎所有权重都在一个 token 上；
- 分数 $[1,\\; 0,\\; -1]$ → softmax $\\approx [0.665,\\; 0.245,\\; 0.090]$：权重分散开。

饱和的 softmax 会带来两个问题：

- 它几乎是 one-hot，一个位置只能看到一个 token；
- 它的梯度接近 0，模型几乎没法调整注意力该放在哪里。

### 解决办法：除以 $\\sqrt{hs}$

除以 $\\sqrt{hs}$ 把标准差拉回到 1：

$$
\\operatorname{std}\\!\\left(\\frac{q \\cdot k}{\\sqrt{hs}}\\right) = 1
$$

上面的 $[8, 0, -8]$ 除以 8，正好变成 $[1, 0, -1]$。

> 不管 head size 多大，分数的标准差都保持在 1 左右，softmax 一开始就不会饱和。这就是 **Scaled Dot-Product Attention** 里 "scaled" 的含义。Transformer 论文给出的理由相同："for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients"（[Vaswani et al., 2017 ↗](https://arxiv.org/abs/1706.03762)，§3.2.1）。

### 和 LayerNorm 的关系

- **LayerNorm** 控制**输入** $x$ 的尺度。
- **$1/\\sqrt{hs}$** 控制**求和结果** $q \\cdot k$ 的尺度。即使输入已经归一化，这个和也会随项数增多而变大。

两者配合使用，谁也替代不了谁。

这一步之后，先加 causal mask，再在最后一维上做 softmax，得到 attention 概率。
`,
    },
  },

  causal_mask_notes: {
    en: {
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
    zh: {
      title: "Causal Mask（因果掩码）",
      md: `
**这一行对 attention 分数加上因果（只能往回看）掩码，让每个 token 只能关注它自己和前面的 token。**

\`\`\`python
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
\`\`\`

- \`att\`：$(B,\\ n\\_head,\\ T,\\ T)$，来自[上一步](#info:q_k_product_notes)的缩放分数 $q\\,k^\\top/\\sqrt{hs}$
- \`self.mask\`：$(1,\\ 1,\\ 1024,\\ 1024)$，对角线及其下方是 1，上方是 0

## 为什么需要掩码

$q\\,k^\\top$ 得到的分数矩阵是**完整的** $T \\times T$：每个位置都给所有其他位置打了分，**包括它后面的位置**。以 4 个 token \`I love black cats\` 为例（行 = query，"谁在看"；列 = key，"被看的是谁"）：

\`\`\`text
            I     love   black   cats
I        [  ✓      ✓       ✓      ✓  ]
love     [  ✓      ✓       ✓      ✓  ]
black    [  ✓      ✓       ✓      ✓  ]
cats     [  ✓      ✓       ✓      ✓  ]
\`\`\`

看 \`love\` 这一行。它的输出要用来预测下一个 token \`black\`，可它已经给 \`black\` 打过分了。这就是**偷看答案**。

- **训练时：** 整个序列并行处理，每个位置都预测它的下一个 token。如果 \`love\` 能看到 \`black\`，预测 \`black\` 就只是照抄，模型学不到任何有用的东西。
- **推理时：** 文本从左到右生成，后面的 token **还不存在**。训练时把未来遮住，训练才和模型的实际使用方式一致。

解决办法：在 softmax 之前，把对角线以上的部分全部挡住，让每一行只看到**它自己和更早的位置**。

## 一步步看

### 1. \`self.mask\` 是什么

它在 \`__init__\` 里只创建一次：

\`\`\`python
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))
\`\`\`

- [torch.ones](#info:torch_ones_notes)(1024, 1024) 全是 1；[torch.tril](#info:torch_tril_notes) 保留**下三角**（对角线及其下方），把上方全部设为 0。以 4×4 为例：

  \`\`\`text
  torch.tril(torch.ones(4, 4))
  [[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]
  \`\`\`

  第 $t$ 行在第 $0 \\dots t$ 列是 1：位置 $t$ 可以看它自己和它前面的所有位置。
- [unsqueeze(0)](#info:unsqueeze_notes)\`.unsqueeze(0)\` 在前面加两个维度，得到 $(1,\\ 1,\\ 1024,\\ 1024)$，这样它就能在 batch 和 head 两个维度上**广播**（原理见下面第 4 步末尾的 **广播**）。
- 它是一个 [buffer](#info:register_buffer_notes)：会跟着模块一起移到 GPU/CPU，也会一起保存，但它不是参数，从不参与训练。

### 2. 切到当前长度

buffer 按最大长度 1024 创建，但当前序列只有 $T$ 个 token：

\`\`\`text
self.mask[:, :, :T, :T]   →   (1, 1, T, T)
\`\`\`

这一步取出左上角的 $T \\times T$。下三角矩阵的左上角本身还是下三角矩阵，所以 $T = 4$ 时，它正好就是上面那个 $4 \\times 4$ 矩阵。

### 3. 变成"哪些连接被禁止"

\`== 0\` 把 0/1 矩阵变成布尔值：元素为 0 的地方是 **True**，也就是要挡住的未来位置。

\`\`\`text
mask == 0
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
\`\`\`

### 4. \`masked_fill\`：在 True 的地方写入 \`-inf\`

\`att.masked_fill(bool_mask, -inf)\` 在掩码为 True 的位置把 \`att\` 写成 $-\\infty$，其他分数保持不变。以 \`I love black cats\` 在某一个 head 里的分数为例，前后对比：

\`\`\`text
before                                    after
          I     love  black  cats                   I     love  black  cats
I      [ 2.1,   0.3,  1.1,   0.6 ]      I      [ 2.1,   -inf, -inf,  -inf ]
love   [ 0.5,   1.3,  1.7,   2.5 ]      love   [ 0.5,   1.3,  -inf,  -inf ]
black  [ 0.2,   0.9,  1.7,   1.4 ]      black  [ 0.2,   0.9,  1.7,   -inf ]
cats   [ 0.1,   0.4,  2.5,   0.8 ]      cats   [ 0.1,   0.4,  2.5,   0.8  ]
\`\`\`

现在 \`I\` 只看得到自己，\`love\` 看得到 \`I\` 和自己，\`cats\` 看得到全部四个。一般地，对 query 位置 $t$：

$$
\\text{att}_{t,j} = -\\infty \\quad \\text{for } j > t
$$

**广播。** 掩码是 $(1,\\ 1,\\ T,\\ T)$，而 \`att\` 是 $(B,\\ n\\_head,\\ T,\\ T)$。大小为 1 的维度会被拉伸，所以 batch 里每个序列、每个 head 用的都是**同一个**三角形。这是对的："不能往后看"只取决于位置，和内容、和是哪个 head 都无关。

### 5. 为什么填 \`-inf\` 而不是 \`0\`？

下一行是 \`att = F.softmax(att, dim=-1)\`。softmax 对每个分数取 $e$ 的幂，再除以这一行的总和：

$$
\\text{softmax}(s)_i = \\frac{e^{s_i}}{\\sum_j e^{s_j}}
$$

拿第 4 步里 \`love\` 这一行来看。

**填 \`-inf\`：未来位置的权重正好是 0。** 因为 $e^{-\\infty} = 0$：

\`\`\`text
scores:   [ 0.5,   1.3,   -inf,  -inf ]
e^s:      [ 1.65,  3.67,  0,     0    ]    sum 5.32
softmax:  [ 0.31,  0.69,  0,     0    ]
\`\`\`

未来位置得不到任何权重，允许的位置会**自动重新归一化**：加起来仍然是 1。

**填 \`0\` 不行。** 分数为 0 并不表示"忽略"，它只是一个普通分数，而 $e^0 = 1$：

\`\`\`text
scores:   [ 0.5,   1.3,   0,     0     ]
softmax:  [ 0.225, 0.501, 0.137, 0.137 ]
\`\`\`

两个未来位置仍然分到了 27% 的注意力：模型还在偷看。

**为什么不在 softmax 之后把权重清零？** 那样这一行加起来就不再是 1 了。用没加掩码的 \`love\` 行 \`[0.5, 1.3, 1.7, 2.5]\`：

\`\`\`text
softmax (no mask):  [ 0.072, 0.16, 0.238, 0.53 ]
zero the future:    [ 0.072, 0.16, 0,     0    ]    sum only 0.23
\`\`\`

还得再除一次才能把总和修正回 1。在 softmax **之前**填 \`-inf\`，一步就同时做到了"未来 = 0"和"每行加起来 = 1"。

> 一句话：$e^{-\\infty} = 0$，所以 softmax 把分数为 \`-inf\` 的位置当作不存在。

## 补充

- 这个掩码只负责**因果性**，不处理 padding。如果一个 batch 里有填充的 token，通常还会再加一个 padding mask（又一次 \`masked_fill\`），让 padding 位置也变成 $-\\infty$。
- 有些代码用一个很大的负数（比如 $-10^9$）代替 $-\\infty$；softmax 之后它的权重同样几乎是 0。
`,
    },
  },

  softmax_notes: {
    en: {
      title: "softmax",
      md: `
**\`F.softmax(att, dim=-1)\` turns each row of scores into attention weights: every value is between 0 and 1, and each row sums to 1.** Before this line, \`att\` holds raw scores, which can be negative or \`-inf\` and don't sum to anything in particular. The shape does not change: $(B,\\ n\\_head,\\ T,\\ T)$ in and out.

## Each row is one token's attention split

Continuing the \`I love black cats\` example from [Causal Mask](#info:causal_mask_notes) (one head; rows are queries, "who is looking"; columns are keys, "who is looked at"):

\`\`\`text
after the mask                            after softmax
          I     love  black  cats                   I     love  black  cats
I      [ 2.1,   -inf, -inf,  -inf ]      I      [ 1,     0,    0,     0    ]
love   [ 0.5,   1.3,  -inf,  -inf ]      love   [ 0.31,  0.69, 0,     0    ]
black  [ 0.2,   0.9,  1.7,   -inf ]      black  [ 0.13,  0.27, 0.60,  0    ]
cats   [ 0.1,   0.4,  2.5,   0.8  ]      cats   [ 0.07,  0.09, 0.72,  0.13 ]
\`\`\`

Working through the \`black\` row:

\`\`\`text
e^0.2 = 1.22    e^0.9 = 2.46    e^1.7 = 5.47    e^-inf = 0
sum = 1.22 + 2.46 + 5.47 = 9.15
weights = [1.22/9.15, 2.46/9.15, 5.47/9.15, 0] = [0.13, 0.27, 0.60, 0]
\`\`\`

So \`black\` gives 13% of its attention to \`I\`, 27% to \`love\` and 60% to itself; \`cats\` gets 0 because it was masked. (Values are rounded to two decimals.)

- **\`I\` row:** only itself is visible, so it puts 100% on itself.
- **\`cats\` row:** nothing is masked; 72% goes to \`black\`.

A higher weight means more attention; a weight of 0 means the key is ignored. The next line, \`att @ v\`, uses these weights to take a weighted average of the value vectors.

## Why \`dim=-1\`

The last axis of \`att\` is the **key** axis: the columns within one row. \`dim=-1\` normalizes along each row, so every query (in every head) splits its own attention.

With \`dim=-2\`, each **column** would sum to 1 instead, meaning "the attention each key receives sums to 1". That is not what we want.

## The formula

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

**Effect of the causal mask:** $att_{b,h,t_q,j}=-\\infty \\;\\Rightarrow\\; P_{b,h,t_q,j}=0$, so future positions get zero weight.

## Notes

- \`F.softmax\` is numerically stable: it effectively subtracts each row's max before exponentiating, so large scores do not overflow.
`,
    },
    zh: {
      title: "softmax",
      md: `
**\`F.softmax(att, dim=-1)\` 把每一行分数变成注意力权重：每个值都在 0 到 1 之间，每一行加起来等于 1。** 在这一行之前，\`att\` 里是原始分数，可以是负数或 \`-inf\`，加起来也不等于什么特定的值。形状不变：输入输出都是 $(B,\\ n\\_head,\\ T,\\ T)$。

## 每一行就是一个 token 分配注意力的方式

接着用 [Causal Mask](#info:causal_mask_notes) 里 \`I love black cats\` 的例子（只看一个 head；行是 query，"正在看的"；列是 key，"被看的"）：

\`\`\`text
after the mask                            after softmax
          I     love  black  cats                   I     love  black  cats
I      [ 2.1,   -inf, -inf,  -inf ]      I      [ 1,     0,    0,     0    ]
love   [ 0.5,   1.3,  -inf,  -inf ]      love   [ 0.31,  0.69, 0,     0    ]
black  [ 0.2,   0.9,  1.7,   -inf ]      black  [ 0.13,  0.27, 0.60,  0    ]
cats   [ 0.1,   0.4,  2.5,   0.8  ]      cats   [ 0.07,  0.09, 0.72,  0.13 ]
\`\`\`

一步步算 \`black\` 这一行：

\`\`\`text
e^0.2 = 1.22    e^0.9 = 2.46    e^1.7 = 5.47    e^-inf = 0
sum = 1.22 + 2.46 + 5.47 = 9.15
weights = [1.22/9.15, 2.46/9.15, 5.47/9.15, 0] = [0.13, 0.27, 0.60, 0]
\`\`\`

也就是说，\`black\` 把 13% 的注意力给 \`I\`，27% 给 \`love\`，60% 给自己；\`cats\` 被 mask 掉了，拿到 0。（数值四舍五入到两位小数。）

- **\`I\` 这一行：** 只能看到自己，所以 100% 给自己。
- **\`cats\` 这一行：** 没有被 mask 的位置；72% 给了 \`black\`。

权重越大，注意力越多；权重为 0 表示完全忽略这个 key。下一行 \`att @ v\` 就用这些权重，对 value 向量做加权平均。

## 为什么是 \`dim=-1\`

\`att\` 的最后一维是 **key**，也就是一行里的各列。\`dim=-1\` 表示沿着每一行做归一化，这样每个 query（在每个 head 里）各自把注意力分完。

如果写成 \`dim=-2\`，就变成每一**列**加起来等于 1，意思是"每个 key 被分到的注意力总和是 1"。这不是我们想要的。

## 公式

$$
att \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}
\\quad\\xrightarrow{\\ \\mathtt{softmax(\\cdot,\\ dim=-1)}\\ }\\quad
P \\in \\mathbb{R}^{B\\times n\\_head\\times T\\times T}
$$

固定 $(b,h,t_q)$：

$$
P_{b,h,t_q,j}
\\;=\\;
\\frac{\\exp\\big(att_{b,h,t_q,j}\\big)}
{\\sum_{k=1}^{T}\\exp\\big(att_{b,h,t_q,k}\\big)},
\\qquad
\\sum_{j=1}^{T} P_{b,h,t_q,j} = 1.
$$

**causal mask 的作用：** $att_{b,h,t_q,j}=-\\infty \\;\\Rightarrow\\; P_{b,h,t_q,j}=0$，所以未来位置的权重是 0。

## 备注

- \`F.softmax\` 在数值上是稳定的：它实际上会先减去每一行的最大值再取指数，所以分数很大也不会溢出。
`,
    },
  },

  attn_dropout_notes: {
    en: {
      title: "Attention dropout",
      md: `
**Attention dropout randomly zeroes some attention weights during training, after softmax and before the weighted sum with $V$.**

$$
A = \\mathrm{softmax}(S + \\text{mask}), \\quad \\tilde A = \\mathrm{Dropout}_p(A), \\quad Y = \\tilde A\\,V
$$

In code: \`att = self.attn_drop(att)\`, where \`att\` is already masked and softmaxed, and \`self.attn_drop = nn.Dropout(dropout)\` with \`dropout=0.1\` on this site. The shape does not change: $(B,\\ n\\_head,\\ T,\\ T)$ in and out.

## What it does to one row

Take the \`black\` row from the [softmax](#info:softmax_notes) example, and suppose that this time the black→black weight is the one dropped:

\`\`\`text
                    I      love   black  cats
softmax:         [ 0.13,  0.27,  0.60,  0 ]
drop one (→ 0):  [ 0.13,  0.27,  0,     0 ]
rest × 1/0.9:    [ 0.14,  0.30,  0,     0 ]
\`\`\`

For this training step, \`black\` cannot see itself and has to take its information from \`I\` and \`love\`. The next step drops a different random set.

## Why multiply by 1/(1 − p)

### On average, nothing changes

Each weight is kept with probability $1 - p = 0.9$ and then multiplied by $1/0.9 \\approx 1.11$, or zeroed with probability $0.1$. For the 0.60 weight over 10 steps, about 9 keep it and 1 drops it:

\`\`\`text
kept (9 times):    0.60 × 1.11 = 0.667
dropped (1 time):  0
average = (9 × 0.667 + 1 × 0) / 10 = 0.60
\`\`\`

On average it is exactly the original weight. In general:

$$
\\tilde A = \\frac{A \\odot M}{1 - p}, \\qquad M_{ij} \\sim \\mathrm{Bernoulli}(1 - p), \\qquad \\mathbb{E}[\\tilde A] = A
$$

So a single row's sum is no longer 1 after dropout: 0.44 in the example above (the largest weight was dropped), 1.11 if nothing is dropped, 0.97 if only the 0.13 is dropped. On average, it is 1.

### Why that matters: inference has no dropout

At inference dropout is off, and every row sums to exactly 1. Without the $1/(1-p)$ factor, rows during training would average only 0.9, so \`att @ v\` would come out about 10% smaller in training than at inference. The model would adapt to the smaller values, then see larger ones at inference. Scaling by $1/(1-p)$ restores what was dropped, so the average matches inference.

### Why not renormalize the row?

Dividing the row by its sum (0.44 above) would also make it sum to 1, but it is not needed:

- The average already matches, which is what matters.
- Dropout is meant to be noise ("a bit less information, at random"); each sample does not need to sum to exactly 1.
- A fixed factor is simple; renormalizing needs an extra sum and division per row.

So \`nn.Dropout\` uses "zero, then multiply by $1/(1-p)$" everywhere it is used (this is called **inverted dropout**).

## Why it helps

The goal is to **prevent over-reliance on any single attention edge**. Dropout does not boost small weights; it randomly cuts edges:

- Cutting a small edge (say 0.07) changes little.
- Cutting a large edge (say 0.60) removes that information for this step, and the loss gets worse.

To keep the loss low when any edge might be cut, the model learns to:

1. **Not keep important information in one place only.** E.g. "\`black\` is an adjective" should also reach later tokens through other paths.
2. **Not bet everything on one token**, so it can still decide reasonably when that token is missing.

At inference, dropout is off and the weights are whatever the model learned: if \`black\` really should mostly attend to itself, it still does.

> **Analogy.** A student always copies homework from the same classmate. The teacher sends that classmate home at random, so the student has to understand the problems or check several sources. At the exam the classmate is there, but the student no longer depends on them alone.

**Example.** Suppose a head usually puts most attention on "dogs" to decide that the verb should be "were". With attention dropout, that strongest edge is sometimes zeroed during training. The model must then use other cues (plurality elsewhere, other heads), which builds redundancy and avoids brittle shortcuts.

## When it is active

- **Training only:** enabled in \`model.train()\`.
- **Disabled at inference:** \`model.eval()\`.

## Not only in attention

GPT-2 uses the same \`nn.Dropout\` in several places: [embedding dropout](#info:emb_dropout_notes), [residual dropout](#info:resid_dropout_notes) and [MLP dropout](#info:mlp_dropout_notes). They all serve as **regularization**: random noise during training, so the model does not memorize the training data (overfit) and generalizes to text it has not seen.

## Choosing p

- Typical $p \\approx 0.1$; GPT-2 uses 0.1.
- Smaller datasets may benefit from a larger $p$.
- 🆕 Many recent LLMs (e.g. LLaMA) pretrain with dropout 0: the data is huge and usually seen about once, so overfitting is not the main problem.
`,
    },
    zh: {
      title: "Attention dropout",
      md: `
**Attention dropout 在训练时随机把一部分注意力权重置 0。它的位置在 softmax 之后、和 $V$ 做加权求和之前。**

$$
A = \\mathrm{softmax}(S + \\text{mask}), \\quad \\tilde A = \\mathrm{Dropout}_p(A), \\quad Y = \\tilde A\\,V
$$

代码里是 \`att = self.attn_drop(att)\`。这时 \`att\` 已经做过 mask 和 softmax；\`self.attn_drop = nn.Dropout(dropout)\`，本网站用 \`dropout=0.1\`。形状不变：输入输出都是 $(B,\\ n\\_head,\\ T,\\ T)$。

## 对一行做了什么

拿 [softmax](#info:softmax_notes) 例子里 \`black\` 那一行，假设这一次被抽中丢掉的是 black→black 这个权重：

\`\`\`text
                    I      love   black  cats
softmax:         [ 0.13,  0.27,  0.60,  0 ]
drop one (→ 0):  [ 0.13,  0.27,  0,     0 ]
rest × 1/0.9:    [ 0.14,  0.30,  0,     0 ]
\`\`\`

这一步训练里，\`black\` 看不到自己，只能从 \`I\` 和 \`love\` 那里取信息。下一步又会随机丢掉别的位置。

## 为什么要乘 1/(1 − p)

### 平均下来什么都没变

每个权重有 $1 - p = 0.9$ 的概率被保留，再乘以 $1/0.9 \\approx 1.11$；有 $0.1$ 的概率被置 0。拿 0.60 这个权重来说，跑 10 步，大约 9 次保留、1 次丢掉：

\`\`\`text
kept (9 times):    0.60 × 1.11 = 0.667
dropped (1 time):  0
average = (9 × 0.667 + 1 × 0) / 10 = 0.60
\`\`\`

平均下来正好是原来的权重。一般地：

$$
\\tilde A = \\frac{A \\odot M}{1 - p}, \\qquad M_{ij} \\sim \\mathrm{Bernoulli}(1 - p), \\qquad \\mathbb{E}[\\tilde A] = A
$$

所以 dropout 之后，单独某一次的一行加起来不再是 1：上面的例子里是 0.44（丢掉了最大的那个），一个都没丢时是 1.11，只丢掉 0.13 时是 0.97。但平均下来是 1。

### 为什么重要：推理时没有 dropout

推理时 dropout 是关掉的，每一行加起来正好是 1。如果没有 $1/(1-p)$ 这个因子，训练时每行平均只有 0.9，\`att @ v\` 在训练时就会比推理时小大约 10%。模型训练时适应的是偏小的数值，推理时却遇到偏大的数值。乘 $1/(1-p)$ 把丢掉的部分补回来，让平均水平和推理时一致。

### 为什么不重新归一化？

把这一行除以它的总和（上面是 0.44）也能让它加起来等于 1，但没必要：

- 平均值已经对上了，这才是关键。
- dropout 本来就是噪声（"随机少一点信息"），不需要每次都正好加起来等于 1。
- 乘一个固定的因子很简单；重新归一化的话，每一行都要多做一次求和和除法。

所以 \`nn.Dropout\` 不管用在哪里，都是"置 0，再乘 $1/(1-p)$"（这种做法叫 **inverted dropout**）。

## 为什么有用

目的是**不让模型过度依赖某一条注意力连线**。dropout 不会提高小权重，它只是随机切断连线：

- 切断一条小连线（比如 0.07），影响不大。
- 切断一条大连线（比如 0.60），这一步就拿不到那条信息，loss 会变差。

为了在"任何连线都可能被切断"的情况下 loss 仍然低，模型会学会：

1. **重要的信息不只放在一个地方。** 比如"\`black\` 是形容词"这个信息，也应该能通过其他路径传到后面的 token。
2. **不把所有赌注押在一个 token 上**，这样某个 token 看不到时，仍然能做出合理的判断。

推理时 dropout 关掉，权重就是模型学出来的样子：如果 \`black\` 真的最该看自己，它照样会大比例看自己。

> **类比。** 一个学生做作业总是抄同一个同学的。老师随机让那个同学请假，这个学生就只能自己弄懂题目，或者参考好几个人的答案。到考试时那个同学也在，但这个学生已经不再只靠他一个人了。

**例子。** 假设某个 head 通常把大部分注意力放在 "dogs" 上，来判断动词应该用 "were"。有了 attention dropout，这条最强的连线在训练时偶尔会被置 0。这时模型只能利用其他线索（别处的复数信息、其他 head），于是学会了冗余，不依赖脆弱的捷径。

## 什么时候起作用

- **只在训练时：** \`model.train()\` 时开启。
- **推理时关闭：** \`model.eval()\`。

## 不只用在 attention

GPT-2 在好几个地方用了同样的 \`nn.Dropout\`：[embedding dropout](#info:emb_dropout_notes)、[residual dropout](#info:resid_dropout_notes) 和 [MLP dropout](#info:mlp_dropout_notes)。它们的作用都是**正则化**：训练时加一点随机噪声，防止模型死记训练数据（过拟合），让它对没见过的文本也表现得好。

## 怎么选 p

- 常见 $p \\approx 0.1$；GPT-2 用 0.1。
- 数据集较小时，大一点的 $p$ 可能更好。
- 🆕 很多近期的 LLM（比如 LLaMA）预训练时 dropout 设为 0：数据量极大，通常只过一遍左右，过拟合不是主要问题。
`,
    },
  },

  att_v_product_notes: {
    en: {
      title: "att @ v",
      md: `
**[\`y = att @ v\`](#hover:matmul_example_notes) is where each token actually collects information: its output is a weighted average of the value vectors, and the weights are its row of \`att\`.** It comes right after [softmax](#info:softmax_notes) and [attention dropout](#info:attn_dropout_notes), and turns \`(B, n_head, T, T)\` weights into \`(B, n_head, T, hs)\` outputs.

## Who to look at vs. what to take

Attention splits the work between q, k and v:

- **q · k** decides *who to look at, and how much*: it produces the weights in \`att\`.
- **v** decides *what to take*: it is the content that actually gets mixed.

k and v come from the same token, but they are different vectors. A token uses k to say "what I am relevant to" and v to say "what I provide when someone attends to me", and the two can be completely different.

The result is a context-aware vector for each token (per head) that pulls in information from the tokens it attends to.

## A worked example

### One row

Take head size 2 and these value vectors for \`I love black cats\`:

\`\`\`text
v_I     = [1, 0]
v_love  = [0, 1]
v_black = [2, 2]
v_cats  = [5, 5]
\`\`\`

The \`black\` row of \`att\` after softmax is \`[0.13, 0.27, 0.60, 0]\` (ignoring dropout), so:

\`\`\`text
y_black = 0.13 × [1, 0] + 0.27 × [0, 1] + 0.60 × [2, 2] + 0 × [5, 5]
        = [0.13, 0]     + [0, 0.27]     + [1.2, 1.2]    + [0, 0]
        = [1.33, 1.47]
\`\`\`

- \`black\`'s own value has 60% of the weight, so the result is closest to \`[2, 2]\`.
- \`cats\` has weight 0, so its \`[5, 5]\` does not leak in at all. This is what the [causal mask](#info:causal_mask_notes) is for: no information from the future.

### All rows at once

One matrix multiplication computes every token's weighted average together: row $t$ of \`att\` times \`v\` gives row $t$ of \`y\`.

\`\`\`text
att (T × T)                         v (T × hs)         y (T × hs)
I      [ 1,    0,    0,    0    ]   I     [1, 0]       I      [1.00, 0.00]
love   [ 0.31, 0.69, 0,    0    ] @ love  [0, 1]   =   love   [0.31, 0.69]
black  [ 0.13, 0.27, 0.60, 0    ]   black [2, 2]       black  [1.33, 1.47]
cats   [ 0.07, 0.09, 0.72, 0.13 ]   cats  [5, 5]       cats   [2.16, 2.18]
\`\`\`

The \`I\` row copies \`v_I\` exactly (100% on itself); \`cats\` takes 72% from \`black\`.

## Shapes

\`\`\`text
att:  (B, n_head, T, T)
v:    (B, n_head, T, hs)
y:    (B, n_head, T, hs)
\`\`\`

\`@\` multiplies only the last two dimensions: $(T, T) \\times (T, hs) \\to (T, hs)$. The leading $(B, n\\_head)$ are treated as a batch: every sequence and every head is computed independently.

## The formula

For batch $b$, head $h$ and query position $t$:

$$
y_{b,h,t,:}
\\;=\\;
\\sum_{j=1}^{T} att_{b,h,t,j}\\; v_{b,h,j,:}
$$

**Causality:** $att_{b,h,t,j}=0$ for $j>t$, so the sum only ranges over allowed (past and self) positions.

**Convex combination:** $att_{b,h,t,j}\\ge 0$ and $\\sum_{j} att_{b,h,t,j}=1$, so $y_{b,h,t,:}$ is a convex combination of the $v_{b,h,j,:}$: a point "between" the value vectors. (During training, attention dropout breaks the exact sum of 1 for each sample, but not on average.)

## Next

\`y\` still has one \`hs\`-long vector per head. The next line concatenates the heads back into one \`C\`-long vector per token, \`(B, T, C)\`: see [merging the heads](#info:reorder_merge_heads_notes).
`,
    },
    zh: {
      title: "att @ v",
      md: `
**[\`y = att @ v\`](#hover:matmul_example_notes) 这一步，每个 token 才真正把信息取回来：它的输出是 value 向量的加权平均，权重就是 \`att\` 里它那一行。** 它紧跟在 [softmax](#info:softmax_notes) 和 [attention dropout](#info:attn_dropout_notes) 之后，把 \`(B, n_head, T, T)\` 的权重变成 \`(B, n_head, T, hs)\` 的输出。

## 看谁 vs. 拿什么

attention 把工作分给了 q、k、v：

- **q · k** 决定"看谁、各看多少"：算出 \`att\` 里的权重。
- **v** 决定"拿走什么"：真正被混合的内容。

k 和 v 来自同一个 token，但它们是两个不同的向量。一个 token 用 k 表示"我和什么相关"，用 v 表示"被看中时我提供什么信息"，两者可以完全不同。

结果是每个 token（每个 head 里）带上了上下文的向量，汇集了它所关注的 token 的信息。

## 例子

### 算一行

假设 head size 是 2，\`I love black cats\` 的 value 向量是：

\`\`\`text
v_I     = [1, 0]
v_love  = [0, 1]
v_black = [2, 2]
v_cats  = [5, 5]
\`\`\`

softmax 之后，\`att\` 里 \`black\` 那一行是 \`[0.13, 0.27, 0.60, 0]\`（不考虑 dropout），所以：

\`\`\`text
y_black = 0.13 × [1, 0] + 0.27 × [0, 1] + 0.60 × [2, 2] + 0 × [5, 5]
        = [0.13, 0]     + [0, 0.27]     + [1.2, 1.2]    + [0, 0]
        = [1.33, 1.47]
\`\`\`

- \`black\` 自己的 value 占了 60% 的权重，所以结果离 \`[2, 2]\` 最近。
- \`cats\` 的权重是 0，它的 \`[5, 5]\` 一点都没混进来。这正是 [causal mask](#info:causal_mask_notes) 想要的效果：未来的信息进不来。

### 一次算完所有行

一次矩阵乘法就把所有 token 的加权平均都算完了：\`att\` 的第 $t$ 行乘以 \`v\`，得到的就是 \`y\` 的第 $t$ 行。

\`\`\`text
att (T × T)                         v (T × hs)         y (T × hs)
I      [ 1,    0,    0,    0    ]   I     [1, 0]       I      [1.00, 0.00]
love   [ 0.31, 0.69, 0,    0    ] @ love  [0, 1]   =   love   [0.31, 0.69]
black  [ 0.13, 0.27, 0.60, 0    ]   black [2, 2]       black  [1.33, 1.47]
cats   [ 0.07, 0.09, 0.72, 0.13 ]   cats  [5, 5]       cats   [2.16, 2.18]
\`\`\`

\`I\` 这一行原样复制了 \`v_I\`（100% 看自己）；\`cats\` 有 72% 取自 \`black\`。

## 形状

\`\`\`text
att:  (B, n_head, T, T)
v:    (B, n_head, T, hs)
y:    (B, n_head, T, hs)
\`\`\`

\`@\` 只对最后两维做矩阵乘法：$(T, T) \\times (T, hs) \\to (T, hs)$。前面的 $(B, n\\_head)$ 当作批次：每个序列的每个 head 各自独立地算。

## 公式

对 batch $b$、head $h$、query 位置 $t$：

$$
y_{b,h,t,:}
\\;=\\;
\\sum_{j=1}^{T} att_{b,h,t,j}\\; v_{b,h,j,:}
$$

**Causality：** $j>t$ 时 $att_{b,h,t,j}=0$，所以求和只包括允许的位置（之前的和自己）。

**Convex combination：** $att_{b,h,t,j}\\ge 0$ 且 $\\sum_{j} att_{b,h,t,j}=1$，所以 $y_{b,h,t,:}$ 是各个 $v_{b,h,j,:}$ 的 convex combination，也就是落在这些 value 向量"之间"的一个点。（训练时，attention dropout 会让单次的权重和不再正好是 1，但平均下来仍然是 1。）

## 下一步

\`y\` 里每个 head 仍然各有一个长度为 \`hs\` 的向量。下一行会把各个 head 拼接回来，变成每个 token 一个长度为 \`C\` 的向量，\`(B, T, C)\`：见[合并 head](#info:reorder_merge_heads_notes)。
`,
    },
  },

  reorder_merge_heads_notes: {
    en: {
      title: "Merging the heads",
      md: `
**\`y.transpose(1, 2).contiguous().view(B, T, C)\` concatenates the heads' outputs, so that each token again has a single $C$-dim vector: $(B,\\ n\\_head,\\ T,\\ hs) \\to (B,\\ T,\\ C)$.**

## Why merge

After [att @ v](#info:att_v_product_notes), every head has produced its own $hs$-dim vector for every token:

\`\`\`text
y: (B, n_head, T, hs)      GPT-2 small: n_head = 12, hs = 64
\`\`\`

The layers that follow (the output projection and the residual add) need one $C = 768$-dim vector per token. Since $12 \\times 64 = 768$, the heads' outputs are **concatenated**.

## Example

Take 2 heads with $hs = 2$, so $C = 4$. For the token \`black\`:

\`\`\`text
head 0 output for black:  [1.33, 1.47]
head 1 output for black:  [0.50, -0.20]

concatenated:             [1.33, 1.47, 0.50, -0.20]    ← length 4 = C
\`\`\`

Every token is merged the same way. The heads are placed **side by side, not added**. Mixing information across heads happens in the next line, the [output projection](#info:attention_output_linear_notes).

## Step by step

### 1. \`transpose(1, 2)\`: group by token

\`y\` is currently ordered "by head, then by token":

\`\`\`text
(B, n_head, T, hs):
  head 0:  [I: 2 numbers]  [love: 2 numbers]  [black: 2 numbers]  [cats: 2 numbers]
  head 1:  [I: 2 numbers]  [love: 2 numbers]  [black: 2 numbers]  [cats: 2 numbers]
\`\`\`

We want each token to hold all of its heads' results together, so \`transpose(1, 2)\` swaps the \`n_head\` and \`T\` dimensions back:

\`\`\`text
(B, T, n_head, hs):
  I:      [head 0: 2 numbers]  [head 1: 2 numbers]
  love:   [head 0: 2 numbers]  [head 1: 2 numbers]
  black:  [head 0: 2 numbers]  [head 1: 2 numbers]
  cats:   [head 0: 2 numbers]  [head 1: 2 numbers]
\`\`\`

### 2. \`contiguous()\`: fix the memory order

\`transpose\` only changes how the tensor is *indexed*; the numbers in memory are still in the old order. \`view\` needs them laid out in memory in the new order, so \`.contiguous()\` copies them into that order (it does nothing if they already are). More: [contiguous](#info:contiguous_notes).

### 3. \`view(B, T, C)\`: join the heads

\`view\` flattens the last two dimensions $(n\\_head,\\ hs)$ into one dimension $C$. For each token, \`[head 0's numbers] [head 1's numbers]\` are joined end to end into one vector of length 4.

### Shapes

\`\`\`text
y                     (B, n_head, T, hs)    (B, 12, T, 64)
  .transpose(1, 2)    (B, T, n_head, hs)    (B, T, 12, 64)
  .contiguous()       (B, T, n_head, hs)    same shape, new memory order
  .view(B, T, C)      (B, T, C)             (B, T, 768)
\`\`\`

## The inverse of the split

When the heads were created (see [Query View](#info:query_view_notes)), the code did \`view(B, T, n_head, hs).transpose(1, 2)\`: **cut** $C$ into heads, then **move** the head dimension forward. Merging does the reverse: **move** it back, then **join**.

Because the cutting and the joining use the same order, head $h$'s output lands in dimensions $h \\cdot hs$ to $(h+1) \\cdot hs - 1$ of the merged vector:

\`\`\`text
merged y of one token (768 dims):
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

## Next

\`y\` is now \`(B, T, C)\`, the same shape as the attention input \`x\`. The next line, \`self.resid_drop(self.proj(y))\`, mixes the heads with a $C \\times C$ linear layer and applies dropout.
`,
    },
    zh: {
      title: "合并 head",
      md: `
**\`y.transpose(1, 2).contiguous().view(B, T, C)\` 把各个 head 的输出拼接起来，让每个 token 重新只有一个 $C$ 维的向量：$(B,\\ n\\_head,\\ T,\\ hs) \\to (B,\\ T,\\ C)$。**

## 为什么要合并

[att @ v](#info:att_v_product_notes) 算完后，每个 head 各自给每个 token 产出了一个 $hs$ 维的向量：

\`\`\`text
y: (B, n_head, T, hs)      GPT-2 small: n_head = 12, hs = 64
\`\`\`

后面的层（输出投影和残差相加）要的是每个 token 一个 $C = 768$ 维的向量。因为 $12 \\times 64 = 768$，所以把各个 head 的输出**拼接**起来。

## 例子

用 2 个 head、$hs = 2$，所以 $C = 4$。看 \`black\` 这个 token：

\`\`\`text
head 0 output for black:  [1.33, 1.47]
head 1 output for black:  [0.50, -0.20]

concatenated:             [1.33, 1.47, 0.50, -0.20]    ← length 4 = C
\`\`\`

每个 token 都这样合并。各个 head 的结果是**并排放在一起，不是相加**。不同 head 之间的信息混合，发生在下一行的[输出投影](#info:attention_output_linear_notes)里。

## 一步一步看

### 1. \`transpose(1, 2)\`：按 token 分组

\`y\` 现在的排列是"先按 head 分，再按 token 分"：

\`\`\`text
(B, n_head, T, hs):
  head 0:  [I: 2 numbers]  [love: 2 numbers]  [black: 2 numbers]  [cats: 2 numbers]
  head 1:  [I: 2 numbers]  [love: 2 numbers]  [black: 2 numbers]  [cats: 2 numbers]
\`\`\`

我们要的是每个 token 把自己所有 head 的结果放在一起，所以用 \`transpose(1, 2)\` 把 \`n_head\` 和 \`T\` 两个维度换回来：

\`\`\`text
(B, T, n_head, hs):
  I:      [head 0: 2 numbers]  [head 1: 2 numbers]
  love:   [head 0: 2 numbers]  [head 1: 2 numbers]
  black:  [head 0: 2 numbers]  [head 1: 2 numbers]
  cats:   [head 0: 2 numbers]  [head 1: 2 numbers]
\`\`\`

### 2. \`contiguous()\`：整理内存顺序

\`transpose\` 只改变了 tensor 的*索引方式*，内存里的数字还是原来的顺序。\`view\` 要求数字在内存里就是新的顺序，所以 \`.contiguous()\` 把它们按新顺序复制一份（如果本来就是这个顺序，就什么都不做）。详见 [contiguous](#info:contiguous_notes)。

### 3. \`view(B, T, C)\`：把 head 拼起来

\`view\` 把最后两维 $(n\\_head,\\ hs)$ 压成一维 $C$。对每个 token 来说，就是把 \`[head 0 的数] [head 1 的数]\` 首尾相接，变成一个长度为 4 的向量。

### 形状

\`\`\`text
y                     (B, n_head, T, hs)    (B, 12, T, 64)
  .transpose(1, 2)    (B, T, n_head, hs)    (B, T, 12, 64)
  .contiguous()       (B, T, n_head, hs)    same shape, new memory order
  .view(B, T, C)      (B, T, C)             (B, T, 768)
\`\`\`

## 拆分的逆操作

当初拆 head 时（见 [Query View](#info:query_view_notes)），代码是 \`view(B, T, n_head, hs).transpose(1, 2)\`：先把 $C$ **切**成多个 head，再把 head 这一维**换**到前面。合并正好反过来：先**换**回来，再**拼**起来。

因为切和拼用的是同一个顺序，head $h$ 的输出正好落在合并后向量的第 $h \\cdot hs$ 到第 $(h+1) \\cdot hs - 1$ 维：

\`\`\`text
merged y of one token (768 dims):
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

## 下一步

\`y\` 现在是 \`(B, T, C)\`，和 attention 的输入 \`x\` 形状一样。下一行 \`self.resid_drop(self.proj(y))\` 用一个 $C \\times C$ 的线性层把各个 head 的信息混合起来，再做 dropout。
`,
    },
  },

  attention_output_linear_notes: {
    en: {
      title: "Attention output projection",
      md: `
**The output projection lets information from different heads interact before it enters the residual stream.**

\`\`\`python
self.proj = nn.Linear(n_embd, n_embd, bias=True)    # C → C
y = self.resid_drop(self.proj(y))                   # (B, T, C)
\`\`\`

## From side by side to mixed

[Merging the heads](#info:reorder_merge_heads_notes) only places the heads' results **side by side**. Each head occupies its own slice, and the heads have not interacted yet:

\`\`\`text
black: [1.33, 1.47 | 0.50, -0.20]
         head 0       head 1
\`\`\`

\`proj\` is a $C \\to C$ linear layer: **every** output dimension is a weighted sum of **all** $C$ input dimensions, so every output dimension can use information from every head.

Continuing the $C = 4$ example, suppose the weight row for output dimension 0 is \`[0.5, 0, 1, 0]\` and the bias is 0:

\`\`\`text
out[0] = 0.5 × 1.33 + 0 × 1.47 + 1 × 0.50 + 0 × (-0.20)
       = 0.665 + 0.50
       = 1.17
         ↑ from head 0   ↑ from head 1
\`\`\`

This single output number combines both heads. The other 3 output dimensions each use their own weight row to combine them in other ways.

Intuitively: if head 0 found "the adjective before me" and head 1 found "the subject of the sentence", \`proj\` combines the two clues into one useful update.

## Why it is needed

1. **Mixing the heads.** Without \`proj\`, head 0's result would always stay in dimensions 0–63, head 1's in 64–127, and so on: the heads' findings could not be combined before the next block.
2. **Writing into the residual stream.** The attention output is added back onto \`x\` (\`x = x + attn(...)\`). \`proj\` turns the concatenated result into [a form suited to that addition](#hover:proj_residual_stream_notes).

At a high level:

- **Q/K do routing:** which tokens attend to which other tokens, within each head.
- **V carries content.**
- **The projection chooses where and how to write that content into the residual space** (a learned basis change).

> It lets the model reweight heads, rotate features, and gate what survives to the next layer.

## Shapes and parameters

\`\`\`text
input y:  (B, T, C)
output:   (B, T, C)     same shape
\`\`\`

- **Weight:** $(n\\_embd,\\ n\\_embd)$
- **Bias:** $(n\\_embd)$
- **Count:** $768 \\times 768 + 768 = 590{,}592$ parameters in GPT-2 small.

More strictly, the input dimension is $n\\_head \\cdot hs$ (the merged heads), so the \`nn.Linear\` weight, stored as (out, in), is $(n\\_embd,\\ n\\_head \\cdot hs)$. In GPT-2, $n\\_head \\cdot hs = n\\_embd$ (e.g. $12 \\times 64 = 768$), so the two are the same.

Like \`c_attn\`, \`proj\` applies the same transformation to each token independently; it does not mix information **between tokens**. That only happens in [\`att @ v\`](#info:att_v_product_notes).

## PyTorch default initialization

See the "PyTorch default initialization" section in the [Attention linear notes](#info:attention_linear_notes) for the details.

## Residual-writing weights

\`proj\` is the last matrix before the attention output is added back to the residual stream:

\`\`\`text
x = x + attn(ln1(x))
        └─ last step inside attn:  y = self.resid_drop(self.proj(y))
                                                       ↑
                              last matrix with weights before the add
\`\`\`

(\`resid_drop\` has no parameters.) So \`proj\`'s weight decides what the attention writes into the residual stream, and how much. Weights in this position are called **residual-writing weights**. The MLP has one in the same position, its final [\`fc_out\`](#info:mlp_fc_out_notes), so each block has 2, and GPT-2 small (12 blocks) has 24.

Why this matters: at initialization, some implementations scale the residual-writing weights by $\\frac{1}{\\sqrt{N}}$, where $N$ is the number of residual layers ($N = 2 \\times n\\_layer = 24$ in GPT-2 small, so the same factor is also written $\\frac{1}{\\sqrt{2\\, n\\_layer}}$). This keeps the sum of $N$ residual additions from blowing up in variance with depth (derivation: [Variance at initialization](#info:variance_init_notes), section 8). Matrices that do not write to the stream, such as \`c_attn\`, are not scaled.

Whether to use it varies between implementations. The GPT-2 paper does it, and 🆕 nanoGPT initializes these weights with standard deviation \`0.02 / sqrt(2 * n_layer)\`. The code on this site does not; it uses PyTorch's default initialization.

### Why pre-norm stacks like GPT-2 need it

In a pre-norm block, $x \\leftarrow x + f(\\mathrm{LN}(x))$: LayerNorm normalizes only the sublayer's **input**. The residual stream itself is never normalized, so every block adds to it and its variance grows with the number of residual additions. Scaling the residual-writing weights by $\\frac{1}{\\sqrt{N}}$ keeps that sum in range. This is what the GPT-2 paper does: it scales the weights of residual layers at initialization by $\\frac{1}{\\sqrt{N}}$ to account for the accumulation on the residual path with depth.

In a post-norm block, $x \\leftarrow \\mathrm{LN}(x + f(x))$, the stream is renormalized after every addition, so it cannot accumulate in the same way.

> See a tiny numeric [example](#info:post_norm_scaling_example_notes) of residual scaling in a post-norm stack.
`,
    },
    zh: {
      title: "Attention output projection",
      md: `
**输出投影让不同 head 的信息在进入残差流之前相互作用。**

\`\`\`python
self.proj = nn.Linear(n_embd, n_embd, bias=True)    # C → C
y = self.resid_drop(self.proj(y))                   # (B, T, C)
\`\`\`

## 从并排到混合

[合并 head](#info:reorder_merge_heads_notes) 只是把各个 head 的结果**并排放在一起**。每个 head 各占一段，彼此之间还没有任何交流：

\`\`\`text
black: [1.33, 1.47 | 0.50, -0.20]
         head 0       head 1
\`\`\`

\`proj\` 是一个 $C \\to C$ 的线性层：输出的**每一维**，都是输入**全部** $C$ 维的加权和。所以每个输出维度都可以用到所有 head 的信息。

接着用 $C = 4$ 的例子。假设算第 0 个输出维度的那一行权重是 \`[0.5, 0, 1, 0]\`，bias 是 0：

\`\`\`text
out[0] = 0.5 × 1.33 + 0 × 1.47 + 1 × 0.50 + 0 × (-0.20)
       = 0.665 + 0.50
       = 1.17
         ↑ from head 0   ↑ from head 1
\`\`\`

这一个输出数字就把两个 head 的信息组合在了一起。另外 3 个输出维度，各自用自己那一行权重，以别的方式组合。

直观一点说：假设 head 0 找到了"前面那个形容词"，head 1 找到了"句子的主语"，\`proj\` 负责把这两条线索组合成一个有用的更新。

## 为什么需要它

1. **混合各个 head。** 没有 \`proj\` 的话，head 0 的结果永远只待在第 0–63 维，head 1 的永远只在 64–127 维，以此类推：在进入下一个 block 之前，各个 head 的发现没法结合起来。
2. **写回残差流。** attention 的输出接下来会加回到 \`x\` 上（\`x = x + attn(...)\`）。\`proj\` 负责把拼接起来的结果，转换成[适合加到残差流里的形式](#hover:proj_residual_stream_notes)。

从更高的层面看：

- **Q/K 负责路由：** 在每个 head 里，决定哪个 token 关注哪些 token。
- **V 携带内容。**
- **投影决定把这些内容写到残差空间的哪里、怎么写**（一次学出来的基变换）。

> 它让模型可以给各个 head 重新加权、旋转特征，并控制哪些信息传到下一层。

## 形状和参数

\`\`\`text
input y:  (B, T, C)
output:   (B, T, C)     same shape
\`\`\`

- **Weight：** $(n\\_embd,\\ n\\_embd)$
- **Bias：** $(n\\_embd)$
- **数量：** GPT-2 small 里是 $768 \\times 768 + 768 = 590{,}592$ 个参数。

更严格地说，输入维度是 $n\\_head \\cdot hs$（合并后的各个 head），所以按 (out, in) 存储的 \`nn.Linear\` 权重是 $(n\\_embd,\\ n\\_head \\cdot hs)$。在 GPT-2 里 $n\\_head \\cdot hs = n\\_embd$（比如 $12 \\times 64 = 768$），所以两者一样。

和 \`c_attn\` 一样，\`proj\` 对每个 token 独立地做同样的变换，不会在 **token 之间**混合信息。token 之间的交流只发生在 [\`att @ v\`](#info:att_v_product_notes) 那一步。

## PyTorch 默认初始化

详见 [Attention linear notes](#info:attention_linear_notes) 里的 "PyTorch default initialization" 一节。

## Residual-writing weights

\`proj\` 是 attention 的输出加回残差流之前的最后一个矩阵：

\`\`\`text
x = x + attn(ln1(x))
        └─ last step inside attn:  y = self.resid_drop(self.proj(y))
                                                       ↑
                              last matrix with weights before the add
\`\`\`

（\`resid_drop\` 没有参数。）所以 \`proj\` 的权重决定了 attention 往残差流里写什么、写多少。处在这个位置的权重叫 **residual-writing weights**。MLP 里也有一个处在同样位置的矩阵，就是最后的 [\`fc_out\`](#info:mlp_fc_out_notes)，所以每个 block 有 2 个，GPT-2 small（12 个 block）一共 24 个。

为什么要特别提它：初始化时，有些实现会把 residual-writing weights 乘以 $\\frac{1}{\\sqrt{N}}$，其中 $N$ 是残差层的数量（GPT-2 small 里 $N = 2 \\times n\\_layer = 24$，所以同一个因子也写成 $\\frac{1}{\\sqrt{2\\, n\\_layer}}$）。这样 $N$ 次残差相加之后，方差不会随着深度越来越大（推导见[初始化时的方差](#info:variance_init_notes)第 8 节）。不往残差流里写的矩阵（比如 \`c_attn\`）不做这个缩放。

各个实现是否这样做不尽相同。GPT-2 论文做了；🆕 nanoGPT 用标准差 \`0.02 / sqrt(2 * n_layer)\` 来初始化这些权重。本网站的代码没有做，用的是 PyTorch 的默认初始化。

### 为什么像 GPT-2 这样的 pre-norm 结构需要它

在 pre-norm block 里，$x \\leftarrow x + f(\\mathrm{LN}(x))$：LayerNorm 只归一化子层的**输入**。残差流本身从来不被归一化，所以每个 block 都往上加一点，它的方差随着残差相加的次数增长。把 residual-writing weights 乘以 $\\frac{1}{\\sqrt{N}}$，就能让这个累加和保持在合理范围内。GPT-2 论文正是这样做的：初始化时把残差层的权重乘以 $\\frac{1}{\\sqrt{N}}$，以抵消残差路径上随深度的累积。

在 post-norm block 里，$x \\leftarrow \\mathrm{LN}(x + f(x))$，每次相加之后残差流都会重新归一化，所以不会以同样的方式累积。

> 看一个 post-norm 结构里残差缩放的小数值[例子](#info:post_norm_scaling_example_notes)。
`,
    },
  },

  proj_residual_stream_notes: {
    en: {
      title: "Writing into the residual stream (hover)",
      md: `
**What does "a form suited to the residual stream" mean?** The concatenated vector and the residual stream \`x\` are both 768-dimensional, but the same position does not mean the same thing in the two. \`proj\` translates the former into the latter's "coordinates" before they are added.

**The residual stream is a shared whiteboard**

\`x\` starts at the embedding and passes through all 12 blocks, and every block adds something to it:

\`\`\`text
x = x + attn(ln1(x))
x = x + mlp(ln2(x))
\`\`\`

So the 768 dimensions of \`x\` are **shared** by every layer. Think of a whiteboard with 768 columns: what each column (more precisely, each direction) stands for is a convention the whole model forms together during training. Every layer reads from it and writes to it.

**The concatenated vector uses each head's own coordinates**

After merging the heads, dimensions 0–63 of \`y\` are head 0's output, 64–127 are head 1's, and so on. These numbers were computed in each head's own value space; their meaning is specific to that head and has nothing to do with the convention for whiteboard columns 0–63.

Adding \`y\` to \`x\` directly, without \`proj\`, would mean:

\`\`\`text
x[0:64]    += head 0's output
x[64:128]  += head 1's output
...
\`\`\`

Head 0 could then only ever change whiteboard columns 0–63, head 1 only 64–127, and each head's output coordinates would have to line up exactly with what those columns mean. That is far too rigid.

**What proj does**

\`proj\` is a $768 \\times 768$ matrix. It lets each head's result be written **anywhere** on the whiteboard, in any amount:

\`\`\`text
what head 0 found  ──proj──→  the columns that need it
what head 1 found  ──proj──→  other columns, or the same ones as head 0
\`\`\`

As an analogy: each head writes a note in its own shorthand, and \`proj\` is the translator who rewrites the notes in the whiteboard's common format and puts them in the right columns. Later layers only understand the whiteboard's format, not each head's shorthand.

> "What each column stands for" is only a way to build intuition. In practice these meanings are learned, usually do not match concepts a person can read directly, and are often spread across many dimensions rather than one meaning per dimension.
`,
    },
    zh: {
      title: "写回残差流",
      md: `
**"适合加到残差流里的形式"是什么意思？** 拼接后的向量和残差流 \`x\` 虽然都是 768 维，但同一位置上的数字含义不一样。\`proj\` 负责把前者翻译成后者的"坐标"，然后才能相加。

**残差流是一块共用的"白板"**

\`x\` 从 embedding 开始，一路穿过 12 个 block，每个 block 都在它上面加一点东西：

\`\`\`text
x = x + attn(ln1(x))
x = x + mlp(ln2(x))
\`\`\`

所以 \`x\` 的 768 维是所有层**共用**的。可以把它想成一块有 768 栏的白板，每一栏（更准确地说，是每个方向）表示什么，是整个模型在训练中一起形成的约定。所有层读的是它，写的也是它。

**拼接后的向量用的是"head 自己的坐标"**

合并 head 之后，\`y\` 的第 0–63 维是 head 0 的输出，第 64–127 维是 head 1 的输出，以此类推。这些数字是在各个 head 自己的 value 空间里算出来的，它们的含义只对那个 head 有意义，和白板上第 0–63 栏的约定无关。

如果不经过 \`proj\`，直接把 \`y\` 加到 \`x\` 上：

\`\`\`text
x[0:64]    += head 0's output
x[64:128]  += head 1's output
...
\`\`\`

这就要求 head 0 永远只能改白板的第 0–63 栏，head 1 永远只能改第 64–127 栏，而且每个 head 的输出坐标必须恰好和那几栏的含义对上。这个限制太死了。

**proj 做的事**

\`proj\` 是一个 $768 \\times 768$ 的矩阵。它让每个 head 的结果可以写到白板上**任意**位置，写多少也由它决定：

\`\`\`text
what head 0 found  ──proj──→  the columns that need it
what head 1 found  ──proj──→  other columns, or the same ones as head 0
\`\`\`

打个比方：每个 head 用自己的速记符号写了一张便条，\`proj\` 是翻译员，把这些便条翻译成白板的统一格式，再写到合适的栏里。后面的层只看得懂白板的格式，看不懂各个 head 的速记。

> "白板上每一栏表示什么"只是帮助理解的说法。实际上这些含义是训练出来的，通常不对应人能直接读懂的概念，而且往往分散在很多维度上，而不是一维一个意思。
`,
    },
  },

  resid_dropout_notes: {
    en: {
      title: "Residual dropout in Causal Self-Attention",
      md: `
**\`resid_drop\` randomly zeros part of the attention output during training, right before that output is added to the residual stream.** It is the last step of causal self-attention.

## Where it sits

\`\`\`python
self.resid_drop = nn.Dropout(dropout)    # in __init__; this site uses dropout = 0.1
...
y = self.resid_drop(self.proj(y))        # end of forward; y: (B, T, C)
return y
\`\`\`

The returned \`y\` is what the block adds to the residual stream \`x\`:

\`\`\`python
x = x + self.attn(self.ln1(x))           # self.attn(...) is the y above
\`\`\`

So the order is: [output projection](#info:attention_output_linear_notes) \`proj\` → \`resid_drop\` → add to \`x\`. The name "resid" comes from this position: it sits directly before the residual add.

## Same rule as attention dropout

\`resid_drop\` is the same \`nn.Dropout\` as [attention dropout](#info:attn_dropout_notes) (\`attn_drop\`), with the same rule:

- **Training** (\`model.train()\`): each number is independently set to 0 with probability $p = 0.1$; every number that survives is multiplied by $1/(1-p) = 1/0.9 \\approx 1.11$. The multiplication keeps the **average** output equal to the input (why this works is explained in the attention dropout note).
- **Inference** (\`model.eval()\`): it does nothing; the output equals the input.

The only difference is **what it acts on**:

| | Acts on | Shape |
| --- | --- | --- |
| \`attn_drop\` | the attention weights after softmax ("how much to look at each token") | $(B,\\ n\\_head,\\ T,\\ T)$ |
| \`resid_drop\` | the output of \`proj\`: the update this attention layer is about to write into the residual stream | $(B,\\ T,\\ C)$ |

## A worked example

Continue the $C = 4$ example from the output projection note, for the token \`black\`. The first number 1.17 is the \`out[0]\` computed there; the other numbers, and the residual stream \`x\`, are made up for illustration:

\`\`\`text
dimension:           0      1      2      3
x (residual stream): [0.20, -0.50,  1.00,  0.30]
update = proj(y):    [1.17,  0.40, -0.30,  0.80]
\`\`\`

**Without dropout** (inference), the whole update is added:

\`\`\`text
x + update         = [1.37, -0.10,  0.70,  1.10]
\`\`\`

**With dropout** (training), suppose this time dimension 1 is drawn to be dropped:

\`\`\`text
dimension 1 → 0:     [1.17,  0,    -0.30,  0.80]
survivors × 1/0.9:   [1.30,  0,    -0.33,  0.89]    ← 1.17/0.9 = 1.30, -0.30/0.9 = -0.33, 0.80/0.9 = 0.89
x + dropped update = [1.50, -0.50,  0.67,  1.19]    ← 0.20+1.30, -0.50+0, 1.00-0.33, 0.30+0.89
\`\`\`

Look at dimension 1: the update was dropped, but \`x\`'s original **−0.50 is still there**. Dropout acts only on this layer's update; the residual stream \`x\` itself is untouched. The effect is "this time, attention did not write anything into dimension 1", not "the information in dimension 1 is lost".

The next training step draws a new random mask, so a different set of dimensions is dropped.

## Why use it

### 1. Regularization

Like every dropout, it prevents the model from relying on any single dimension of the attention output. Every number in the update may disappear in a given training step, so later layers learn not to depend on one particular write. The information has to be spread over several dimensions (and several heads), which generalizes better to unseen text.

The two dropouts in attention perturb different things:

- \`attn_drop\` perturbs the **read** side: which tokens each token takes information from.
- \`resid_drop\` perturbs the **write** side: what the layer writes back into the residual stream. Even if one head always attends to the same token, what it writes is still randomly dropped sometimes.

### 2. It does not damage the main path

Dropout is placed on the **branch** (\`attn(...)\`), not on \`x\`. The residual path, \`x\` passing straight through \`x + ...\`, always stays complete.

Suppose instead dropout were applied to \`x\` itself in every block. Then each of the 12 blocks would randomly erase part of everything the earlier layers had accumulated, and the losses would compound with depth. Deep networks would become much harder to train. Dropping only the branch update means each step loses at most this one layer's contribution.

## Not only in attention

The MLP branch has the same residual dropout at the same position: \`x = x + mlp(ln2(x))\`, where the MLP output also goes through dropout before the add. See [Residual dropout in MLP](#info:mlp_dropout_notes).

## Choosing p

- GPT-2 uses $p = 0.1$.
- 🆕 Many recent LLMs (e.g. LLaMA) pretrain with dropout 0 here too: the training data is huge and usually seen about once, so overfitting is not the main problem.
`,
    },
    zh: {
      title: "Causal Self-Attention 中的 residual dropout",
      md: `
**\`resid_drop\` 在训练时随机把 attention 输出的一部分置 0，位置就在这个输出加回残差流之前。** 它是 causal self-attention 的最后一步。

## 位置

\`\`\`python
self.resid_drop = nn.Dropout(dropout)    # 在 __init__ 里；本网站用 dropout = 0.1
...
y = self.resid_drop(self.proj(y))        # forward 的最后；y: (B, T, C)
return y
\`\`\`

返回的 \`y\` 就是 block 要加到残差流 \`x\` 上的东西：

\`\`\`python
x = x + self.attn(self.ln1(x))           # self.attn(...) 就是上面的 y
\`\`\`

所以顺序是：[output projection](#info:attention_output_linear_notes) \`proj\` → \`resid_drop\` → 加到 \`x\` 上。名字里的 "resid" 就来自这个位置：它紧挨在残差相加（residual add）之前。

## 规则和 attention dropout 一样

\`resid_drop\` 和 [attention dropout](#info:attn_dropout_notes)（\`attn_drop\`）是同一种 \`nn.Dropout\`，规则也一样：

- **训练时**（\`model.train()\`）：每个数独立地以概率 $p = 0.1$ 被置 0；没被置 0 的数都乘以 $1/(1-p) = 1/0.9 \\approx 1.11$。这个乘法让输出的**平均值**和输入相等（原因见 attention dropout 的 note）。
- **推理时**（\`model.eval()\`）：什么都不做，输出等于输入。

唯一的区别是**作用在什么上**：

| | 作用在什么上 | 形状 |
| --- | --- | --- |
| \`attn_drop\` | softmax 之后的注意力权重（"每个 token 看多少"） | $(B,\\ n\\_head,\\ T,\\ T)$ |
| \`resid_drop\` | \`proj\` 的输出，也就是这一层 attention 要写进残差流的更新 | $(B,\\ T,\\ C)$ |

## 例子

接着 output projection note 里 $C = 4$ 的例子，看 \`black\` 这个 token。第一个数 1.17 就是那里算出的 \`out[0]\`；其余的数和残差流 \`x\` 是为了说明随便取的：

\`\`\`text
dimension:           0      1      2      3
x (residual stream): [0.20, -0.50,  1.00,  0.30]
update = proj(y):    [1.17,  0.40, -0.30,  0.80]
\`\`\`

**没有 dropout**（推理时），整个更新都加上去：

\`\`\`text
x + update         = [1.37, -0.10,  0.70,  1.10]
\`\`\`

**有 dropout**（训练时），假设这一次第 1 维被抽中丢掉：

\`\`\`text
dimension 1 → 0:     [1.17,  0,    -0.30,  0.80]
survivors × 1/0.9:   [1.30,  0,    -0.33,  0.89]    ← 1.17/0.9 = 1.30, -0.30/0.9 = -0.33, 0.80/0.9 = 0.89
x + dropped update = [1.50, -0.50,  0.67,  1.19]    ← 0.20+1.30, -0.50+0, 1.00-0.33, 0.30+0.89
\`\`\`

看第 1 维：更新被丢掉了，但 \`x\` 原来的 **−0.50 还在**。dropout 只作用在这一层的更新上，残差流 \`x\` 本身不受影响。效果是"这一次 attention 没往第 1 维写东西"，而不是"第 1 维的信息没了"。

下一个训练 step 会重新随机抽一次，丢掉的就是别的维度了。

## 为什么要用

### 1. 正则化

和所有 dropout 一样，它让模型不能依赖 attention 输出里的某一个维度。更新里的每个数在某个训练 step 都可能消失，所以后面的层学会不去依赖某一个特定的写入。信息必须分散到多个维度（和多个 head）上，这样对没见过的文本泛化得更好。

attention 里的两个 dropout 扰动的是不同的东西：

- \`attn_drop\` 扰动的是**读**的一侧：每个 token 从哪些 token 取信息。
- \`resid_drop\` 扰动的是**写**的一侧：这一层往残差流里写回什么。即使某个 head 总是盯着同一个 token，它写出来的东西也仍然会时不时被随机丢掉。

### 2. 不伤主干

dropout 放在**分支**（\`attn(...)\`）上，而不是放在 \`x\` 上。残差这条路，也就是 \`x\` 直接穿过 \`x + ...\` 的那条直通道，始终是完整的。

反过来，假设每个 block 都对 \`x\` 本身做 dropout。那 12 个 block 每一个都会随机抹掉前面所有层积累下来的信息的一部分，层数越多损失叠加得越多，深层网络会难训练得多。只丢分支的更新，意味着每一步最多只损失这一层自己的贡献。

## 不只在 attention 里

MLP 分支在同样的位置也有 residual dropout：\`x = x + mlp(ln2(x))\`，其中 MLP 的输出在相加之前也要经过 dropout。见 [MLP 里的 residual dropout](#info:mlp_dropout_notes)。

## p 怎么选

- GPT-2 用 $p = 0.1$。
- 🆕 很多近期的 LLM（例如 LLaMA）预训练时这里的 dropout 也设成 0：训练数据量巨大，而且通常每条只看一遍左右，过拟合不是主要问题。
`,
    },
  },

  causal_analogy_notes: {
    en: {
      title: "Causal Analogy",
      md: `
[Causal Analogy](#info:causal_analogy_notes) · [Attention Inference Example](#info:attn_inference_example_notes)

**Below is an analogy that shows how the GPT-2 multi-head self-attention block works, to give you an intuitive understanding.**

## Setting

A row of people, one per position $t$ in the sequence. Each person holds notes $x_t$ (corresponding to the embedding of the token at position $t$).

## 1. Make three cards from your notes: \`c_attn(x) → q, k, v\`

- **Query ($q$):** what I'm looking for now (built by $W_q$).
- **Key ($k$):** what I can offer, i.e. what I can be matched on (built by $W_k$); an index used for matching only.
- **Value ($v$):** what I will share if you listen (built by $W_v$); the actual content that gets copied/mixed into someone else's representation.

### Why separate Query and Key?

Consider a simpler design with no $W_q$ or $W_k$, scoring directly with $x$: $\\text{score}(i \\to j) = x_i \\cdot x_j$. This has two problems.

**1. Attention would be symmetric.** The dot product is symmetric, $x_{\\text{bank}} \\cdot x_{\\text{river}} = x_{\\text{river}} \\cdot x_{\\text{bank}}$, so bank would have to attend to river exactly as much as river attends to bank. But relations in language are often one-way: bank needs river to tell which sense it is, while river is clear on its own. With separate q and k, the two scores use different vectors and can differ. A 2-D example:

- $q_{\\text{bank}} = [0, 2]$, $k_{\\text{bank}} = [1, 0]$
- $q_{\\text{river}} = [0, 0]$, $k_{\\text{river}} = [0, 2]$
- bank → river: $q_{\\text{bank}} \\cdot k_{\\text{river}} = 0 \\cdot 0 + 2 \\cdot 2 = 4$ (strong attention)
- river → bank: $q_{\\text{river}} \\cdot k_{\\text{bank}} = 0 \\cdot 1 + 0 \\cdot 0 = 0$ (none)

(Under the causal mask river can't see the later word bank anyway; the point is that scores between any two words can be asymmetric.)

**2. Each word would attend mostly to itself.** With raw $x$, a word's score for itself is $x \\cdot x = \\|x\\|^2$, which is usually larger than its score for any other word. Every position would lean toward itself, or toward words **similar** to it. But bank isn't looking for another bank; it needs river or money, words that are **different** and add information. Separate $W_q$ and $W_k$ pick out different features of $x$, so the model can learn "my q matches words whose k complements me".

> **In short:** separating q and k lets scores be **asymmetric** (A attending to B doesn't mean B attends to A), so the model can learn to find words that are **different from itself but useful to it**.

### Why separate Key vs Value?

Keys are for addressing (similarity search), Values are for content. You choose *who* to listen to via $q \\cdot k$ and copy what matters from those people via $v$.

Could one vector do both jobs, getting found and delivering content? No: *what gets you found* and *what you hand over* are different things. Take:

> The cat sat on the mat because it was tired.

The position of "it" needs to work out who "it" refers to:

- **it's q:** I'm a pronoun, looking for an earlier singular, animate noun.
- **cat's k:** I'm a singular, animate noun. → A good match.
- **mat's k:** I'm a singular, inanimate noun. → A weaker match.
- **cat's v:** the meaning of "cat" (an animal, furry, can get tired…). → Once matched, this is what is passed to "it".

cat's k and v carry **different things**:

- **k** holds the features that get cat **found**: singular, animate, a noun. These are matching labels.
- **v** holds what a finder actually **needs**: the meaning of "cat", so "it" learns that the one who is tired is a cat.

If $k = v$, "it" would receive only the labels "singular, animate, noun", not "cat" itself. In library terms (below): you search with the index card, but you take home the book's text.

> **In short:** q is what I'm looking for, k is what I can be found by, v is what you get once you find me. All three come from the same $x$, but each has its own $W$ that picks out the features it needs. q and k only decide **who to attend to**; v is the information that actually **gets moved**.

### A library analogy

- **Query:** your search request, e.g. "I want books on river geography".
- **Key:** each book's index card, with its title and keywords, used to **match** against your request.
- **Value:** the book's actual text. Once a book matches, this is what you **take away**.

Matching uses q and k; what you take away is v.

### Example: "I went to the river bank"

"bank" is ambiguous (riverbank or financial bank), and its token embedding is the same either way. Look at the position of "bank":

- **bank's q:** I'm an ambiguous word; I'm looking for words that tell me which sense I am.
- **river's k:** I'm a geography/nature noun. → A good match for bank's q.
- **went's k:** I'm a verb of motion. → A poor match.
- **river's v:** information about rivers, water, geography. → Because river matched, this is what gets passed to bank.

Later steps use them like this:

- step 3 scores relevance with **$q \\cdot k$**: bank's q is dotted with every earlier word's k;
- step 7 mixes the **v**'s with those scores as weights and hands the result to bank.

> Two caveats. The descriptions above ("I'm looking for…", "I'm a geography noun") are only an aid to intuition: real q, k and v are just 768 numbers each, and what they encode is learned during training, not designed by hand. And **every token has all three**: bank uses its q to ask earlier words, while its own k and v are matched and read by the words that come after it.

## 2. Split into committees (heads): reshape to $(B, n\\_head, T, hs)$

Each committee (head) works in parallel with its own $W_q, W_k, W_v$ and looks at a different perspective (syntax, long-range cues, etc.).

How each tensor is split into heads: [Query View](#info:query_view_notes) · [Key View](#info:key_view_notes) · [Value View](#info:value_view_notes)

### Why 12 heads instead of one?

**One head can only look one way.** In a single head, each position gets **one** set of attention weights (summing to 1 after softmax) and mixes the v's into **one** result. But a word often needs several kinds of information at once:

> The tired dog that chased the cat was sleeping

The position of "was" wants to know:

- **Who is the subject?** dog, skipping over "that chased the cat"; this decides *was* rather than *were*.
- **What state is the subject in?** tired.
- **What is the previous word?** cat, which helps with local structure.

With one head, these three compete for the **same** weights: more weight on dog means less on tired. The mixed result is a blur of everything, and none of the three jobs is done well.

**12 heads = 12 independent attention patterns.** Each head has its own 64-dim q, k and v and does attention on its own:

- head 3 finds the subject → almost all weight on dog;
- head 7 looks at the previous word → almost all weight on cat;
- head 9 finds modifiers → most weight on tired;
- …

Each head can **focus on one thing** without interfering with the others. Concatenating the 12 results gives "was" all of this information at once.

> **Analogy:** one person reads the sentence with one question in mind. Twelve people read the same sentence, each with a different question (find the subject, check the previous word, find modifiers…), and then pool their answers.

Two caveats:

1. **It costs nothing extra.** $12 \\times 64 = 768$, so parameters and compute are about the same as one 768-dim head. The same budget is split twelve ways to get twelve different views.
2. **The division of labor is learned, not assigned.** "Head 3 finds the subject" above is only an illustration. What each head ends up doing is decided by training, and some heads are hard to interpret. But clearly specialized heads do exist: in real GPT-2 small, head 7 of layer 0 mostly attends to the last few tokens, and it stops working when position information is removed.

## 3. Score who to listen to: \`att_logits = q @ k^T / sqrt(hs)\`

Each person compares their query with everyone's keys to get compatibility scores; dividing by $\\sqrt{hs}$ keeps them numerically calm so softmax doesn't explode.

What the $T \\times T$ score matrix looks like: [Q @ K Product](#info:q_k_product_notes)

## 4. Causal rule (no looking ahead): mask future positions to $-\\infty$

This enforces the LM factorization

$$
p(x_{1:T}) = \\prod_{t=1}^{T} p(x_t \\mid x_{\\le t-1})
$$

Why it exists:

- **No leakage during training:** even though we compute all positions in parallel for speed, the mask guarantees position $t$ never uses $t+1, \\dots$.
- **Matches generation:** at inference we generate left to right; causality during training makes the learned behavior consistent with how we use the model.
- **Prevents cheating:** without masking, the model could peek at the answer (future tokens) and cheat.

## 5. Turn scores into listening weights: \`att = softmax(att_logits, dim=-1)\`

Now each person has a probability distribution over the allowed earlier positions: *how much attention do I give to each source?* Rows sum to 1, giving interpretable mixing weights.

## 6. Practice with partial connections: \`attn_drop(att)\`

Randomly drop some links during training to build robustness and avoid over-reliance on one path.

## 7. Build your personalized summary from Values: \`y = att @ v\`

Mix others' Value vectors using your attention weights (e.g., $0.6\\,v_3 + 0.3\\,v_7 + \\cdots$).

> Remember: Keys help you choose; Values are what you take.

## 8. Reassemble committees and standardize: merge heads, then \`proj(y)\`

Stitch the head outputs back into one vector per position, project to model width, and feed the residual stream cleanly.

## Big picture

> At each position $t$: select earlier positions via Query-Key (addressing), copy useful content via Values, combine multiple perspectives (heads), and do it under a causal rule that matches left-to-right generation.
`,
    },
    zh: {
      title: "因果类比",
      md: `
[因果类比](#info:causal_analogy_notes) · [Attention 推理示例](#info:attn_inference_example_notes)

**下面用一个类比来说明 GPT-2 的多头 self-attention 块是怎么工作的，帮你建立直觉。**

## 场景

一排人，序列里每个位置 $t$ 站一个人。每个人手里拿着笔记 $x_t$（对应位置 $t$ 上那个 token 的 embedding）。

## 1. 用笔记做三张卡片：\`c_attn(x) → q, k, v\`

- **Query（$q$）：** 我现在在找什么（由 $W_q$ 生成）。
- **Key（$k$）：** 我能提供什么，也就是别人能用什么来匹配到我（由 $W_k$ 生成）；它只是用来匹配的索引。
- **Value（$v$）：** 如果你来听我，我会分享什么（由 $W_v$ 生成）；这是真正被复制/混合进别人表示里的内容。

### 为什么要把 Query 和 Key 分开？

考虑一个更简单的设计：不要 $W_q$ 和 $W_k$，直接用 $x$ 打分：$\\text{score}(i \\to j) = x_i \\cdot x_j$。这样有两个问题。

**1. Attention 会是对称的。** 点积是对称的，$x_{\\text{bank}} \\cdot x_{\\text{river}} = x_{\\text{river}} \\cdot x_{\\text{bank}}$，所以 bank 对 river 的关注程度必须和 river 对 bank 的完全一样。但语言里的关系常常是单向的：bank 需要 river 来判断自己是哪个意思，而 river 本身意思就很清楚。把 q 和 k 分开后，两个方向的分数用的是不同的向量，可以不一样。一个 2 维的例子：

- $q_{\\text{bank}} = [0, 2]$，$k_{\\text{bank}} = [1, 0]$
- $q_{\\text{river}} = [0, 0]$，$k_{\\text{river}} = [0, 2]$
- bank → river：$q_{\\text{bank}} \\cdot k_{\\text{river}} = 0 \\cdot 0 + 2 \\cdot 2 = 4$（强关注）
- river → bank：$q_{\\text{river}} \\cdot k_{\\text{bank}} = 0 \\cdot 1 + 0 \\cdot 0 = 0$（不关注）

（在 causal mask 下，river 本来就看不到后面的 bank；这里要说明的是，任意两个词之间的分数可以不对称。）

**2. 每个词会主要关注它自己。** 如果直接用 $x$，一个词对自己的分数是 $x \\cdot x = \\|x\\|^2$，通常比它对其他任何词的分数都大。每个位置都会偏向自己，或者偏向和自己**相似**的词。但 bank 要找的不是另一个 bank；它需要 river 或 money 这类**不同**的、能补充信息的词。分开的 $W_q$ 和 $W_k$ 从 $x$ 里挑出不同的特征，于是模型可以学到"我的 q 去匹配那些 k 能补充我的词"。

> **简单说：** 把 q 和 k 分开，分数就可以**不对称**（A 关注 B 不代表 B 也关注 A），模型就能学会去找那些**和自己不同、但对自己有用**的词。

### 为什么要把 Key 和 Value 分开？

Key 用来寻址（相似度搜索），Value 用来承载内容。你通过 $q \\cdot k$ 选择*听谁的*，再通过 $v$ 从这些人那里复制有用的内容。

能不能用一个向量同时干两件事：既负责被找到，又负责提供内容？不行：*让你被找到的东西*和*你交出去的东西*是两回事。看这句：

> The cat sat on the mat because it was tired.

"it" 所在的位置需要弄清楚 "it" 指的是谁：

- **it 的 q：** 我是一个代词，在找前面一个单数、有生命的名词。
- **cat 的 k：** 我是单数、有生命的名词。→ 匹配得很好。
- **mat 的 k：** 我是单数、无生命的名词。→ 匹配得比较弱。
- **cat 的 v：** "cat" 的含义（一种动物，毛茸茸，会累……）。→ 匹配上之后，传给 "it" 的就是这个。

cat 的 k 和 v 装的是**不同的东西**：

- **k** 装的是让 cat **被找到**的特征：单数、有生命、名词。这些是用来匹配的标签。
- **v** 装的是找它的人真正**需要**的东西："cat" 的含义，这样 "it" 才知道累的是一只猫。

如果 $k = v$，"it" 拿到的就只有"单数、有生命、名词"这些标签，而不是 "cat" 本身。用下面图书馆的说法：你用索引卡来搜索，但带回家的是书的正文。

> **简单说：** q 是我在找什么，k 是别人凭什么能找到我，v 是找到我之后你能拿到什么。三者都来自同一个 $x$，但各有自己的 $W$，挑出各自需要的特征。q 和 k 只决定**关注谁**；v 才是真正**被搬运**的信息。

### 图书馆类比

- **Query：** 你的检索请求，比如"我想找关于河流地理的书"。
- **Key：** 每本书的索引卡，写着书名和关键词，用来和你的请求做**匹配**。
- **Value：** 书的正文。一本书匹配上之后，你**带走**的就是它。

匹配用的是 q 和 k；带走的是 v。

### 例子："I went to the river bank"

"bank" 有歧义（河岸还是银行），而它的 token embedding 无论哪种意思都一样。看 "bank" 所在的位置：

- **bank 的 q：** 我是个有歧义的词；我在找能告诉我自己是哪个意思的词。
- **river 的 k：** 我是一个地理/自然类名词。→ 和 bank 的 q 匹配得很好。
- **went 的 k：** 我是一个表示移动的动词。→ 匹配得不好。
- **river 的 v：** 关于河流、水、地理的信息。→ 因为 river 匹配上了，传给 bank 的就是这个。

后面的步骤这样使用它们：

- 第 3 步用 **$q \\cdot k$** 给相关性打分：bank 的 q 和前面每个词的 k 做点积；
- 第 7 步以这些分数为权重，把各个 **v** 混合起来，交给 bank。

> 两点提醒。上面的描述（"我在找……"、"我是地理类名词"）只是帮助理解的比喻：真实的 q、k、v 各自只是 768 个数字，它们编码什么是训练中学出来的，不是人为设计的。而且**每个 token 都有这三样**：bank 用自己的 q 去询问前面的词，而它自己的 k 和 v 则被后面的词匹配和读取。

## 2. 分成若干委员会（heads）：reshape 成 $(B, n\\_head, T, hs)$

每个委员会（head）用自己的 $W_q, W_k, W_v$ 并行工作，各自关注不同的视角（语法、长距离线索等）。

每个张量是怎么拆分成 head 的：[Query View](#info:query_view_notes) · [Key View](#info:key_view_notes) · [Value View](#info:value_view_notes)

### 为什么用 12 个 head 而不是一个？

**一个 head 只能朝一个方向看。** 在单个 head 里，每个位置只得到**一组** attention 权重（softmax 之后和为 1），并把各个 v 混合成**一个**结果。但一个词常常同时需要好几种信息：

> The tired dog that chased the cat was sleeping

"was" 所在的位置想知道：

- **主语是谁？** dog，要跳过 "that chased the cat"；这决定了用 *was* 而不是 *were*。
- **主语处于什么状态？** tired。
- **前一个词是什么？** cat，有助于理解局部结构。

只有一个 head 时，这三件事要争**同一组**权重：给 dog 的权重多了，给 tired 的就少了。混合出来的结果是所有东西的模糊平均，三件事哪件都做不好。

**12 个 head = 12 种独立的 attention 模式。** 每个 head 有自己 64 维的 q、k、v，独立地做 attention：

- head 3 找主语 → 几乎全部权重给 dog；
- head 7 看前一个词 → 几乎全部权重给 cat；
- head 9 找修饰语 → 大部分权重给 tired；
- ……

每个 head 都可以**专注于一件事**，互不干扰。把 12 个结果拼接起来，"was" 就同时拿到了所有这些信息。

> **类比：** 一个人带着一个问题读这句话。十二个人读同一句话，每人带着不同的问题（找主语、看前一个词、找修饰语……），最后把答案汇总起来。

两点说明：

1. **不需要额外代价。** $12 \\times 64 = 768$，所以参数量和计算量与一个 768 维的 head 大致相同。同样的预算被分成十二份，换来十二种不同的视角。
2. **分工是学出来的，不是指定的。** 上面"head 3 找主语"只是举例。每个 head 最终做什么由训练决定，有些 head 很难解释。但确实存在分工明确的 head：在真实的 GPT-2 small 里，第 0 层的 head 7 主要关注最近的几个 token，去掉位置信息后它就不再起作用。

## 3. 给"听谁的"打分：\`att_logits = q @ k^T / sqrt(hs)\`

每个人拿自己的 query 和所有人的 key 比较，得到匹配分数；除以 $\\sqrt{hs}$ 让数值保持平稳，免得 softmax 爆掉。

$T \\times T$ 分数矩阵长什么样：[Q @ K 乘积](#info:q_k_product_notes)

## 4. 因果规则（不能往后看）：把未来的位置 mask 成 $-\\infty$

这保证了语言模型的分解形式

$$
p(x_{1:T}) = \\prod_{t=1}^{T} p(x_t \\mid x_{\\le t-1})
$$

为什么需要它：

- **训练时不泄漏：** 虽然为了速度我们并行计算所有位置，mask 保证位置 $t$ 永远不会用到 $t+1, \\dots$。
- **和生成方式一致：** 推理时我们从左到右生成；训练时保持因果性，学到的行为就和模型的实际使用方式一致。
- **防止作弊：** 没有 mask，模型就能偷看答案（未来的 token）来作弊。

## 5. 把分数变成"倾听权重"：\`att = softmax(att_logits, dim=-1)\`

现在每个人在允许看的更早位置上都有一个概率分布：*我给每个来源多少注意力？* 每行和为 1，得到可以解释的混合权重。

## 6. 在部分连接断开的情况下练习：\`attn_drop(att)\`

训练时随机丢掉一些连接，增强鲁棒性，避免过度依赖某一条路径。

## 7. 用 Value 拼出你的个性化总结：\`y = att @ v\`

用你的 attention 权重把别人的 Value 向量混合起来（例如 $0.6\\,v_3 + 0.3\\,v_7 + \\cdots$）。

> 记住：Key 帮你选择；Value 是你拿走的东西。

## 8. 把各委员会的结果汇总并统一格式：合并 heads，然后 \`proj(y)\`

把各个 head 的输出重新拼回每个位置一个向量，投影到模型宽度，干净地送进残差流。

## 全局视角

> 在每个位置 $t$：通过 Query-Key 选出更早的位置（寻址），通过 Value 复制有用的内容，结合多种视角（heads），并且整个过程遵守与从左到右生成相一致的因果规则。
`,
    },
  },

  attn_inference_example_notes: {
    en: {
      title: "Attention Inference Example",
      md: `
[Causal Analogy](#info:causal_analogy_notes) · [Attention Inference Example](#info:attn_inference_example_notes)

**This example shows how a single attention head can infer the correct verb form (\`were\` vs \`was\`) from the context of the sentence.**

Input sentence: **The dogs that barked loudly were tired.**

We focus on predicting the token \`were\` (position 6) from the prefix \`The dogs that barked loudly\`.

## What happens (one head, simplified)

Goal: decide between \`were\` and \`was\`.

To do this, we need to know the true subject of the clause so we can infer the correct verb form. In this example, the true subject is \`dogs\` (plural), even though there's an intervening clause "that barked loudly."

### Notes

1. **The model doesn't actually know it's a verb position.** In real models there are no POS tags or grammar rules baked in. Instead, the model learns from training to recognize contexts where a verb is likely next. The model has a hidden state $x_t$ that is a function of:

   - all previous tokens and the token itself
   - the positional embedding $p_t$
   - many layers of self-attention and feed-forward networks

   The hidden state $x_t$ encodes contextual features like "we're in a place where a verb typically follows a plural subject...", but it could also encode other things:

   - **Likely punctuation next.** In this case, the model might predict a comma instead: "The dogs that barked loudly, …".
   - **Alternate continuations.** In this case, the model might predict a different verb such as \`seemed\` or \`are\`: "The dogs that barked loudly are tired."

   But in this toy example, we focus on why the model predicts \`were\` instead of \`was\`.

2. **Assumption: all parameters are already trained.** This is an inference example, so the projections $\\mathbf{W}_q, \\mathbf{W}_k, \\mathbf{W}_v$ (and the output projection $\\mathbf{W}_o$), the MLPs, LayerNorms, embeddings, and the LM head are all fixed. The example's numbers are illustrative, but they reflect what trained parameters typically make happen at inference.

## Step A: Values carry useful features

Suppose this head's Value vectors include:

- **pluralness** (singular $-1 \\,\\dots\\, +1$ plural)
- **subjectness** (is this token a good subject head?)

> This is a **simplification**. In real models, the Value vectors are **a high-dimensional learned representation of size head size** (e.g. GPT-2 small has 768 dimensions and 12 heads, so the head size is 64). So in real models it isn't literally a 2-number feature; we just projected the idea into two dimensions to illustrate what the head might encode.

Toy values:

- \`The\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`dogs\`: $\\mathbf{V} \\approx \\begin{bmatrix}+1.0 \\\\ +0.9\\end{bmatrix}$ ← plural noun, likely subject
- \`that\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.1\\end{bmatrix}$
- \`barked\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`loudly\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$

## Step B: Queries and Keys score who to listen to

The head compares the Query $\\mathbf{q}_5$ to each Key $\\mathbf{k}_j$ ($j = 1 \\ldots 5$). Made-up attention logits (after the $1/\\sqrt{hs}$ scale):

- to \`The\`: $-0.5$
- to \`dogs\`: $+2.57$ ← strong match
- to \`that\`: $+0.3$
- to \`barked\`: $-0.2$
- to \`loudly\`: $-0.1$

> The numerical values are made up to show the pattern (e.g., the subject getting the highest score). In the real model they come from the dot products with the trained $\\mathbf{W}_q, \\mathbf{W}_k$, plus masking.

Then we apply softmax to get the attention weights (they sum to 1):

- $w(\\mathtt{The}) = 0.04$
- $w(\\mathtt{dogs}) = 0.78$
- $w(\\mathtt{that}) = 0.08$
- $w(\\mathtt{barked}) = 0.05$
- $w(\\mathtt{loudly}) = 0.05$

## Step C: Mix Values with those weights (\`att @ v\`)

The weighted average produces the head's output at position 5:

$$
\\mathbf{y}_5
= 0.04\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.78\\cdot\\begin{bmatrix}+1.0, +0.9\\end{bmatrix}
+ 0.08\\cdot\\begin{bmatrix}0, 0.1\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
\\\\[4pt]
\\mathbf{y}_5 \\approx \\begin{bmatrix}+0.78, +0.71\\end{bmatrix}
$$

> **Interpretation:** position 5 now carries a strong plural + subject cue.

## Step D: Why this flips the logits the right way

Think of the LM head as rows (one per vocab item) that dot with the final hidden vector. The rows for just the two verbs:

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4, +0.5\\end{bmatrix}$ (likes plural & subject)
- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1, +0.2\\end{bmatrix}$ (penalizes plural)

Dot products from this head's contribution:

$$
\\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot 0.78 + 0.5\\cdot 0.71 \\approx +1.45 \\\\[4pt]
\\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot 0.78 + 0.2\\cdot 0.71 \\approx -0.72
$$

After combining with other heads/layers and softmax over the vocab, \`were\` gets much higher probability than \`was\`.

## Why \`dogs\` has high influence here

- **Syntactic role:** it's the true subject of the clause containing the verb, even across "that barked loudly."
- **Training pressure:** across millions of examples, gradients push the parameters so that a verb's Query aligns strongly with the Key of its governing subject. That makes the attention weight on the subject high.
- **Value payload:** the subject's Value carries exactly the features (plurality, subjectness) that help pick the correct verb form. If the model predicts the wrong form, gradients increase $q \\cdot k$ for the right subject and shape its $\\mathbf{V}$ to be more informative.

## What if we flip to singular \`dog\`?

With \`dog\` (singular) instead of \`dogs\` (plural), the same attention head that tracks subject–verb agreement will now pass singular evidence forward, so the LM head will score \`was\` higher than \`were\`.

Here's the same toy walk-through, just flipped to singular. Prefix: \`The dog that barked loudly\`.

### A) Values carry features

Toy features: [pluralness, subjectness].

- \`The\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`dog\`: $\\mathbf{V} \\approx \\begin{bmatrix}-1.0 & +0.9\\end{bmatrix}$ ← singular noun, likely subject
- \`that\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.1\\end{bmatrix}$
- \`barked\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`loudly\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$

### B) Queries and Keys decide who to listen to

Position 5's query (for the next-token slot) compares against the keys of positions 1 to 5. As before, it strongly matches the subject:

- to \`The\`: $-0.5$
- to \`dog\`: $+2.57$ ← strong match
- to \`that\`: $+0.3$
- to \`barked\`: $-0.2$
- to \`loudly\`: $-0.1$

Softmax → attention weights:

$$
w(\\mathtt{The})=0.04,\\ \\ w(\\mathtt{dog})=0.78,\\ \\ w(\\mathtt{that})=0.08,\\ \\ w(\\mathtt{barked})=0.05,\\ \\ w(\\mathtt{loudly})=0.05
$$

### C) Mix values with those weights (\`att @ v\`)

$$
\\mathbf{y} \\approx 0.78\\cdot\\begin{bmatrix}-1.0, +0.9\\end{bmatrix} + 0.08\\cdot\\begin{bmatrix}0.0, 0.1\\end{bmatrix} \\approx \\boxed{\\begin{bmatrix}-0.78, +0.71\\end{bmatrix}}
$$

> **Interpretation:** strong singular signal plus subject cue at position 5.

### D) LM head rows (toy) and resulting logits

Think of each vocab token as a row that dots with the hidden vector.

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4 & +0.5\\end{bmatrix}$ (likes plural & subject)

  $$
  \\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot(-0.78) + 0.5\\cdot 0.71 \\approx -0.74
  $$

- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1 & +0.2\\end{bmatrix}$ (penalizes plural, okay with subject)

  $$
  \\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot(-0.78) + 0.2\\cdot 0.71 \\approx +1.00
  $$

> So $\\operatorname{logit}(\\mathtt{was}) > \\operatorname{logit}(\\mathtt{were})$. After combining all heads/layers and softmax over the full vocab, the model strongly prefers \`was\` in the singular-subject context.
`,
    },
    zh: {
      title: "Attention 推理示例",
      md: `
[因果类比](#info:causal_analogy_notes) · [Attention 推理示例](#info:attn_inference_example_notes)

**这个例子展示单个 attention head 如何根据句子上下文推断出正确的动词形式（\`were\` 还是 \`was\`）。**

输入句子：**The dogs that barked loudly were tired.**

我们关注的是：根据前缀 \`The dogs that barked loudly\` 预测 token \`were\`（位置 6）。

## 发生了什么（单个 head，简化版）

目标：在 \`were\` 和 \`was\` 之间做选择。

为此，我们需要知道从句真正的主语，才能推断出正确的动词形式。在这个例子里，真正的主语是 \`dogs\`（复数），尽管中间隔着一个从句 "that barked loudly"。

### 说明

1. **模型其实并不知道这是一个动词位置。** 真实模型里没有内置词性标签或语法规则。模型是在训练中学会识别"下一个很可能是动词"的上下文。模型有一个隐藏状态 $x_t$，它是下面这些东西的函数：

   - 之前所有的 token 和这个 token 本身
   - 位置 embedding $p_t$
   - 很多层 self-attention 和前馈网络

   隐藏状态 $x_t$ 编码了诸如"这里通常是复数主语后面跟动词的位置……"这样的上下文特征，但它也可能编码别的东西：

   - **接下来可能是标点。** 这种情况下，模型可能预测一个逗号："The dogs that barked loudly, …"。
   - **其他的续写方式。** 这种情况下，模型可能预测别的动词，比如 \`seemed\` 或 \`are\`："The dogs that barked loudly are tired."

   但在这个玩具例子里，我们只关注模型为什么预测 \`were\` 而不是 \`was\`。

2. **假设：所有参数都已训练好。** 这是一个推理的例子，所以投影矩阵 $\\mathbf{W}_q, \\mathbf{W}_k, \\mathbf{W}_v$（以及输出投影 $\\mathbf{W}_o$）、MLP、LayerNorm、embedding 和 LM head 都是固定的。例子里的数字是示意性的，但它们反映了训练好的参数在推理时通常会产生的效果。

## 步骤 A：Value 携带有用的特征

假设这个 head 的 Value 向量包含：

- **复数性**（单数 $-1 \\,\\dots\\, +1$ 复数）
- **主语性**（这个 token 适不适合做主语中心词？）

> 这是一个**简化**。在真实模型里，Value 向量是**一个维度为 head size 的、学出来的高维表示**（例如 GPT-2 small 有 768 维、12 个 head，所以 head size 是 64）。所以在真实模型里它并不真的是两个数字的特征；我们只是把这个想法投影到二维，来说明这个 head 可能编码了什么。

玩具 value：

- \`The\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`dogs\`：$\\mathbf{V} \\approx \\begin{bmatrix}+1.0 \\\\ +0.9\\end{bmatrix}$ ← 复数名词，很可能是主语
- \`that\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.1\\end{bmatrix}$
- \`barked\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`loudly\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$

## 步骤 B：Query 和 Key 给"听谁的"打分

这个 head 把 Query $\\mathbf{q}_5$ 和每个 Key $\\mathbf{k}_j$（$j = 1 \\ldots 5$）做比较。编造的 attention logits（已经乘过 $1/\\sqrt{hs}$）：

- 对 \`The\`：$-0.5$
- 对 \`dogs\`：$+2.57$ ← 强匹配
- 对 \`that\`：$+0.3$
- 对 \`barked\`：$-0.2$
- 对 \`loudly\`：$-0.1$

> 这些数值是编出来的，用来展示规律（例如主语得到最高分）。在真实模型里，它们来自与训练好的 $\\mathbf{W}_q, \\mathbf{W}_k$ 计算出的点积，再加上 mask。

然后用 softmax 得到 attention 权重（和为 1）：

- $w(\\mathtt{The}) = 0.04$
- $w(\\mathtt{dogs}) = 0.78$
- $w(\\mathtt{that}) = 0.08$
- $w(\\mathtt{barked}) = 0.05$
- $w(\\mathtt{loudly}) = 0.05$

## 步骤 C：用这些权重混合 Value（\`att @ v\`）

加权平均得到这个 head 在位置 5 的输出：

$$
\\mathbf{y}_5
= 0.04\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.78\\cdot\\begin{bmatrix}+1.0, +0.9\\end{bmatrix}
+ 0.08\\cdot\\begin{bmatrix}0, 0.1\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
\\\\[4pt]
\\mathbf{y}_5 \\approx \\begin{bmatrix}+0.78, +0.71\\end{bmatrix}
$$

> **解读：** 位置 5 现在带有很强的"复数 + 主语"信号。

## 步骤 D：为什么这会把 logits 往正确的方向推

把 LM head 看成很多行（每个词表项一行），每行和最终的隐藏向量做点积。只看这两个动词对应的行：

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4, +0.5\\end{bmatrix}$（偏好复数和主语）
- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1, +0.2\\end{bmatrix}$（惩罚复数）

这个 head 的贡献带来的点积：

$$
\\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot 0.78 + 0.5\\cdot 0.71 \\approx +1.45 \\\\[4pt]
\\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot 0.78 + 0.2\\cdot 0.71 \\approx -0.72
$$

和其他 head/层的结果合起来，再在整个词表上做 softmax 之后，\`were\` 的概率远高于 \`was\`。

## 为什么 \`dogs\` 在这里影响力很大

- **句法角色：** 它是包含这个动词的从句的真正主语，即使隔着 "that barked loudly"。
- **训练压力：** 在数以百万计的样本上，梯度推动参数，让动词的 Query 和支配它的主语的 Key 高度对齐。这使得主语上的 attention 权重很高。
- **Value 携带的内容：** 主语的 Value 恰好带有帮助选出正确动词形式的特征（复数性、主语性）。如果模型预测错了形式，梯度会增大正确主语的 $q \\cdot k$，并调整它的 $\\mathbf{V}$，让它更有信息量。

## 如果换成单数 \`dog\` 呢？

把 \`dogs\`（复数）换成 \`dog\`（单数）后，负责主谓一致的同一个 attention head 现在会把单数的证据往前传，于是 LM head 给 \`was\` 的分数会高于 \`were\`。

下面是同样的玩具推演，只是换成了单数。前缀：\`The dog that barked loudly\`。

### A) Value 携带特征

玩具特征：[复数性, 主语性]。

- \`The\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`dog\`：$\\mathbf{V} \\approx \\begin{bmatrix}-1.0 & +0.9\\end{bmatrix}$ ← 单数名词，很可能是主语
- \`that\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.1\\end{bmatrix}$
- \`barked\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`loudly\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$

### B) Query 和 Key 决定听谁的

位置 5 的 query（用于预测下一个 token）和位置 1 到 5 的 key 做比较。和前面一样，它和主语强匹配：

- 对 \`The\`：$-0.5$
- 对 \`dog\`：$+2.57$ ← 强匹配
- 对 \`that\`：$+0.3$
- 对 \`barked\`：$-0.2$
- 对 \`loudly\`：$-0.1$

Softmax → attention 权重：

$$
w(\\mathtt{The})=0.04,\\ \\ w(\\mathtt{dog})=0.78,\\ \\ w(\\mathtt{that})=0.08,\\ \\ w(\\mathtt{barked})=0.05,\\ \\ w(\\mathtt{loudly})=0.05
$$

### C) 用这些权重混合 value（\`att @ v\`）

$$
\\mathbf{y} \\approx 0.78\\cdot\\begin{bmatrix}-1.0, +0.9\\end{bmatrix} + 0.08\\cdot\\begin{bmatrix}0.0, 0.1\\end{bmatrix} \\approx \\boxed{\\begin{bmatrix}-0.78, +0.71\\end{bmatrix}}
$$

> **解读：** 位置 5 上有很强的单数信号，外加主语信号。

### D) LM head 的行（玩具）和得到的 logits

把每个词表 token 看成一行，和隐藏向量做点积。

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4 & +0.5\\end{bmatrix}$（偏好复数和主语）

  $$
  \\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot(-0.78) + 0.5\\cdot 0.71 \\approx -0.74
  $$

- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1 & +0.2\\end{bmatrix}$（惩罚复数，对主语无所谓）

  $$
  \\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot(-0.78) + 0.2\\cdot 0.71 \\approx +1.00
  $$

> 所以 $\\operatorname{logit}(\\mathtt{was}) > \\operatorname{logit}(\\mathtt{were})$。把所有 head/层合起来、在整个词表上做 softmax 之后，在单数主语的上下文里模型会明显偏向 \`was\`。
`,
    },
  },

  gqa_notes: {
    en: {
      title: "Grouped-query attention (GQA) and the KV cache",
      md: `
**🆕 In grouped-query attention (GQA), several query heads share one key head and one value head. LLaMA 3 8B has 32 query heads but only 8 key/value heads. The point is to shrink the [KV cache](#info:kv_cache_notes), the k and v that a model stores while it generates text.**

## 1. From MHA to GQA

**GPT-2: every head has its own q, k, v.** GPT-2 small has 12 heads:

\`\`\`text
head 0:   q0,  k0,  v0
head 1:   q1,  k1,  v1
...
head 11:  q11, k11, v11
\`\`\`

This is standard multi-head attention (MHA).

**LLaMA 3 8B: 32 query heads, 8 key/value heads.** Since $32 \\div 8 = 4$, every 4 query heads share one k and one v:

\`\`\`text
q0  q1  q2  q3    ->  share k0, v0
q4  q5  q6  q7    ->  share k1, v1
...
q28 q29 q30 q31   ->  share k7, v7
\`\`\`

The 32 query heads form 8 groups, hence the name.

## 2. In the code

\`\`\`python
self.n_head = n_head                         # 32 query heads
self.n_kv_head = n_kv_head                   # 8 key/value heads (GQA)
self.q_proj = nn.Linear(n_embd, n_head * hs, bias=False)     # 4096 = 32 × 128
self.k_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # 1024 =  8 × 128
self.v_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # 1024 =  8 × 128
\`\`\`

q is projected to 32 heads, k and v to only 8. Before scoring, each k and v head is repeated 4 times:

\`\`\`python
rep = self.n_head // self.n_kv_head          # 4
k = k.repeat_interleave(rep, dim=1)          # (B, 8, T, 128) -> (B, 32, T, 128)
v = v.repeat_interleave(rep, dim=1)
\`\`\`

\`repeat_interleave\` repeats each head in place:

\`\`\`text
before:  k0, k1, ..., k7
after:   k0, k0, k0, k0, k1, k1, k1, k1, ..., k7, k7, k7, k7
\`\`\`

Now k and v also have 32 heads, lined up with q, and the rest ($q \\cdot k$, softmax, the weighted sum of v) is exactly as in GPT-2.

The repeat comes **after** [RoPE](#info:rope_notes), so RoPE rotates only the 8 k heads, not 32 copies.

### The heads still differ

The 4 query heads in a group share k and v, but each has its own q. So each computes its own attention weights and can attend to different tokens. What they share is what can be looked up (k) and what is taken (v).

### Parameters saved

The k and v projections of one layer:

\`\`\`text
MHA (32 k/v heads):  2 × 4096 × 4096 = 33.6M
GQA ( 8 k/v heads):  2 × 4096 × 1024 =  8.4M
\`\`\`

That is 25.2M fewer per layer, about 805M over 32 layers, roughly 10% of the 8B model. But saving parameters is not the main reason for GQA. The main reason is the memory used during inference, explained below.

## 3. Why: a 4× smaller KV cache

When generating, the model stores the $k$ and $v$ of every earlier token in every layer, so it does not have to recompute them at each step. This is the [KV cache](#info:kv_cache_notes). Its size is proportional to the number of k/v heads, so with 8 instead of 32, GQA makes it 4 times smaller:

\`\`\`text
                          per token     8K context     128K context
MHA (32 k/v heads)        512 KB        4 GB           64 GB
GQA ( 8 k/v heads)        128 KB        1 GB           16 GB
\`\`\`

For comparison, the model weights take about 15 GB. With MHA, a 128K context would need a cache more than 4 times the size of the model itself; with GQA it needs 16 GB.

This pays off twice:

- **More requests per GPU.** Each request being served needs its own cache, so a 4× smaller cache fits about 4 times as many at once.
- **Faster generation.** Each new token requires reading the whole cache from memory, and that reading, not the arithmetic, is what mostly limits speed.

The idea of GQA in one sentence: keep 32 query heads, so each head can still choose what to attend to, but cut k and v, the part that has to be stored and read again and again, to 8 heads.

## References

1. **Fast Transformer Decoding: One Write-Head is All You Need (2019)**

   Shazeer introduces multi-query attention (MQA): all query heads share a single k/v head, the extreme case of GQA.

   [MQA ↗](https://arxiv.org/abs/1911.02150)

2. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2023)**

   Ainslie et al. propose grouped-query attention, between MHA and MQA: close to MHA's quality with nearly MQA's speed.

   [GQA ↗](https://arxiv.org/abs/2305.13245)

3. **The Llama 3 Herd of Models (2024)**

   LLaMA 3 uses GQA with 8 key/value heads in all model sizes.

   [Llama 3 ↗](https://arxiv.org/abs/2407.21783)
`,
    },
    zh: {
      title: "分组查询注意力（GQA）与 KV cache",
      md: `
**🆕 分组查询注意力（GQA）让几个 query head 共用同一组 key 和 value。LLaMA 3 8B 有 32 个 query head，但 key/value head 只有 8 个。这样做主要是为了缩小 [KV cache](#info:kv_cache_notes)，也就是模型生成文本时一直存着的那些 k 和 v。**

## 1. 从 MHA 到 GQA

**GPT-2：每个 head 都有自己的 q、k、v。** GPT-2 small 有 12 个 head：

\`\`\`text
head 0:   q0,  k0,  v0
head 1:   q1,  k1,  v1
...
head 11:  q11, k11, v11
\`\`\`

这就是普通的多头注意力（MHA）。

**LLaMA 3 8B：32 个 query head，8 个 key/value head。** $32 \\div 8 = 4$，所以每 4 个 query head 共用一组 k、v：

\`\`\`text
q0  q1  q2  q3    ->  共用 k0, v0
q4  q5  q6  q7    ->  共用 k1, v1
...
q28 q29 q30 q31   ->  共用 k7, v7
\`\`\`

32 个 query head 分成了 8 组，"分组查询"这个名字就是这么来的。

## 2. 代码里怎么做

\`\`\`python
self.n_head = n_head                         # 32 query heads
self.n_kv_head = n_kv_head                   # 8 key/value heads (GQA)
self.q_proj = nn.Linear(n_embd, n_head * hs, bias=False)     # 4096 = 32 × 128
self.k_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # 1024 =  8 × 128
self.v_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # 1024 =  8 × 128
\`\`\`

q 投影出 32 个 head，k 和 v 只投影出 8 个。算分数之前，再把每个 k head、v head 复制 4 份：

\`\`\`python
rep = self.n_head // self.n_kv_head          # 4
k = k.repeat_interleave(rep, dim=1)          # (B, 8, T, 128) -> (B, 32, T, 128)
v = v.repeat_interleave(rep, dim=1)
\`\`\`

\`repeat_interleave\` 是把每个 head 就地连续复制：

\`\`\`text
复制前：k0, k1, ..., k7
复制后：k0, k0, k0, k0, k1, k1, k1, k1, ..., k7, k7, k7, k7
\`\`\`

复制完以后 k、v 也是 32 个 head，和 q 一一对上。后面的 $q \\cdot k$、softmax、按权重取 v，就和 GPT-2 完全一样了。

复制放在 [RoPE](#info:rope_notes) **之后**，所以 RoPE 只需要转 8 个 k head，不用转 32 份。

### 各个 head 仍然不一样

同一组的 4 个 query head 共用 k 和 v，但每个 head 的 q 都不一样。所以它们各自算出的 attention 权重不同，仍然可以关注不同的 token。共用的只是"被查的内容"（k）和"被取的内容"（v）。

### 省了多少参数

每一层的 k、v 投影：

\`\`\`text
MHA（32 个 k/v head）：  2 × 4096 × 4096 = 33.6M
GQA（ 8 个 k/v head）：  2 × 4096 × 1024 =  8.4M
\`\`\`

每层少 25.2M，32 层一共少了约 805M，大约占 8B 的 10%。不过省参数并不是 GQA 的主要目的，主要是为了 inference 时省内存，下面就讲这个。

## 3. 为什么：KV cache 小了 4 倍

生成文本时，模型会把前面每个 token 在每一层的 $k$、$v$ 都存起来，这样每一步就不用重新算，这就是 [KV cache](#info:kv_cache_notes)。它的大小和 k/v head 数成正比，所以 GQA 把 32 个减到 8 个，KV cache 就小了 4 倍：

\`\`\`text
                          每个 token     8K 上下文     128K 上下文
MHA（32 个 k/v head）      512 KB         4 GB          64 GB
GQA（ 8 个 k/v head）      128 KB         1 GB          16 GB
\`\`\`

对比一下，模型权重本身大约 15 GB。如果用 MHA，128K 上下文的缓存要比模型本身大 4 倍多；用 GQA 只要 16 GB。

这样有两个好处：

- **一张 GPU 能同时服务更多请求。** 每个请求都要有自己的一份缓存，缓存小了 4 倍，同时能放下的请求也就多了大约 4 倍。
- **生成更快。** 每生成一个新 token，都要把整个缓存从显存里读一遍，速度主要就卡在这里，而不是卡在计算上。

一句话概括 GQA 的思路：q 保留 32 个 head，让每个 head 仍然能自己决定关注谁；k、v 是要一直存着、反复读取的部分，就只留 8 个 head。

## 参考资料

1. **Fast Transformer Decoding: One Write-Head is All You Need (2019)**

   Shazeer 提出多查询注意力（MQA）：所有 query head 共用唯一一组 k/v，是 GQA 的极端情况。

   [MQA ↗](https://arxiv.org/abs/1911.02150)

2. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2023)**

   Ainslie 等人提出分组查询注意力，介于 MHA 和 MQA 之间：效果接近 MHA，速度接近 MQA。

   [GQA ↗](https://arxiv.org/abs/2305.13245)

3. **The Llama 3 Herd of Models (2024)**

   LLaMA 3 所有尺寸的模型都用 GQA，key/value head 都是 8 个。

   [Llama 3 ↗](https://arxiv.org/abs/2407.21783)
`,
    },
  },

  kv_cache_notes: {
    en: {
      title: "KV cache",
      md: `
**When a model generates text, it keeps the $k$ and $v$ of every earlier token in memory, so it does not have to recompute them at every step. This store is the KV cache. It saves a great deal of computation, but at long contexts it takes a lot of memory, and many newer techniques exist just to make it smaller.**

## 1. Why cache anything

A model generates one token at a time. Each new token goes back into the input, and the model runs again to produce the next one.

**Without a cache,** every step runs the whole sequence through the network again:

\`\`\`text
step 1:  run tokens 1                     -> predict token 2
step 2:  run tokens 1, 2                  -> predict token 3
step 3:  run tokens 1, 2, 3               -> predict token 4
...
step n:  run tokens 1, 2, ..., n          -> predict token n+1
\`\`\`

Generating 1,000 tokens this way runs $1 + 2 + \\dots + 1000 = 500{,}500$ token positions through the network. Almost all of that is repeated work: token 1 is processed 1,000 times.

**With a cache,** each step runs **only the new token** through the network. In every attention layer, the new token's $q$ is scored against the stored $k$ of all earlier tokens, its weights are applied to their stored $v$, and its own $k$ and $v$ are appended to the store. Generating 1,000 tokens now runs 1,000 token positions.

## 2. Only when generating, not in training

The KV cache is used only at inference, while the model generates text one token at a time. Training does not use it.

**In training,** the whole text is known in advance. An 8K sample goes through the network in one forward pass: each layer computes $q$, $k$, $v$ for all 8,192 tokens at once, and the [causal mask](#info:causal_mask_notes) lets each token see only the tokens before it. That one pass gives the prediction at every position, and the loss is computed from all of them. Each token's $k$ and $v$ are computed exactly once per layer, so there is no repeated work for a cache to save.

**When generating,** later tokens do not exist yet. They must be produced one at a time, each fed back as input: the step-by-step loop in section 1. That loop is where the repeated work comes from, and the cache is what removes it.

🆕 **Post-training can use it too.** Reinforcement-learning methods such as RLHF and GRPO first let the model generate answers (called rollouts), then score them and update the weights. Generating the answers is inference, so it uses a KV cache, and is often handed to an inference engine such as vLLM. Updating the weights is ordinary training and does not use one.

## 3. Why the stored k and v stay valid

Caching only works if the $k$ and $v$ of earlier tokens never change when new tokens arrive. They do not, because of the [causal mask](#info:causal_mask_notes):

- In layer 1, token $i$'s $k$ and $v$ are computed from token $i$'s own embedding only.
- Token $i$'s output from layer 1 mixes in tokens $1, \\dots, i$, but never anything after $i$: the causal mask blocks later tokens.
- So token $i$'s input to layer 2, and therefore its $k$ and $v$ in layer 2, also depend only on tokens $1, \\dots, i$. The same argument repeats through every layer.

A token that arrives later cannot change anything about token $i$, in any layer. Its $k$ and $v$ are final as soon as they are computed.

$q$ is not cached: an earlier token's $q$ is only used to compute that token's own output, which was already done.

## 4. What is stored, and when

**What.** For each layer, the $k$ and $v$ of every token so far, split into heads:

\`\`\`text
per layer:  k: (B, n_kv_head, T, hs)    v: (B, n_kv_head, T, hs)
\`\`\`

In LLaMA the $k$ is stored **after** [RoPE](#info:rope_notes). A token's position never changes, so its rotation never changes either.

**When.** Generation has two phases:

- **Prefill:** the whole prompt is run through the network at once, as in training. This fills the cache with the prompt's $k$ and $v$, and predicts the first new token.
- **Decode:** after that, one token per step, each appending one entry per layer to the cache.

## 5. How big it gets

For every token, every layer stores one $k$ and one $v$ vector per k/v head:

\`\`\`text
bytes = 2 (k and v) × layers × k/v heads × head size × tokens × bytes per number
\`\`\`

With 16-bit numbers (2 bytes each):

\`\`\`text
                         layers  k/v heads  head size   per token    full context
GPT-2 small                12        12         64        36 KB      1,024 tokens:     36 MB
LLaMA 3 8B (if MHA)        32        32        128       512 KB      8K: 4 GB,  128K: 64 GB
LLaMA 3 8B (GQA)           32         8        128       128 KB      8K: 1 GB,  128K: 16 GB
\`\`\`

For comparison, LLaMA 3 8B's weights take about 15 GB.

- For GPT-2 the cache is tiny. For a modern model with a long context it can be as large as the model itself.
- It grows linearly with the context length, and again with the number of requests served at once: each request has its own cache.

## 6. It also limits speed

At every decode step, the GPU reads all the model weights and the whole KV cache from memory, but does only a little arithmetic for the one new token. So decoding speed is limited mostly by **how many bytes must be read**, not by computation. A smaller cache means faster generation, not just less memory.

## 7. 🆕 Ways to make it smaller

Every factor in the size formula above has been attacked:

**Fewer k/v heads.**

- **MQA** (2019): all query heads share a single k/v head. Smallest, with some loss in quality.
- **GQA** (2023): a few k/v heads, each shared by a group of query heads. LLaMA 3 uses 8. See [GQA](#info:gqa_notes).

**Smaller k and v.**

- **MLA** (DeepSeek-V2, 2024): store one low-dimensional compressed vector per token and reconstruct $k$ and $v$ from it when needed. DeepSeek reports a 93% smaller cache than its earlier 67B model.
- **Quantization:** store $k$ and $v$ in 8 or 4 bits instead of 16.

**Fewer tokens.**

- **Sliding-window attention** (Mistral 7B, 2023): each token attends only to the last 4,096 tokens, so the cache never holds more than 4,096.
- **Local and global layers** (Gemma 2 and 3): most layers use a short window; only the global layers need the full cache.
- **Dropping tokens** (H2O, StreamingLLM, 2023): keep the tokens that receive the most attention plus the first few, and evict the rest.

**Fewer layers with their own cache.**

- **Cross-layer sharing** (CLA, 2024): adjacent layers share one set of $k$ and $v$.

**Better memory management** (serving systems; the model is unchanged).

- **PagedAttention** (vLLM, 2023): allocate the cache in small blocks as it grows, like pages in an operating system, so little memory is wasted and more requests fit on a GPU.
- **Prefix caching:** requests that start with the same text, such as the same system prompt, share the cache for that part.

## References

1. **Fast Transformer Decoding: One Write-Head is All You Need (2019).** Shazeer introduces multi-query attention. [MQA ↗](https://arxiv.org/abs/1911.02150)
2. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2023).** Ainslie et al. [GQA ↗](https://arxiv.org/abs/2305.13245)
3. **Efficient Memory Management for Large Language Model Serving with PagedAttention (2023).** Kwon et al., the vLLM paper. [PagedAttention ↗](https://arxiv.org/abs/2309.06180)
4. **Mistral 7B (2023).** Sliding-window attention with a rolling KV cache. [Mistral 7B ↗](https://arxiv.org/abs/2310.06825)
5. **Efficient Streaming Language Models with Attention Sinks (2023).** Xiao et al., StreamingLLM. [StreamingLLM ↗](https://arxiv.org/abs/2309.17453)
6. **DeepSeek-V2 (2024).** Introduces multi-head latent attention (MLA). [DeepSeek-V2 ↗](https://arxiv.org/abs/2405.04434)
`,
    },
    zh: {
      title: "KV cache",
      md: `
**模型生成文本时，会把前面每个 token 的 $k$ 和 $v$ 存在内存里，这样每一步就不用重新算。存下来的这些东西就叫 KV cache。它省掉了大量计算，但上下文一长就非常占内存，所以后来出现了很多专门给它"瘦身"的技术。**

## 1. 为什么要缓存

模型一次只生成一个 token。新生成的 token 会接到输入后面，模型再跑一遍，生成下一个。

**不用缓存的话**，每一步都要把整个序列重新跑一遍：

\`\`\`text
第 1 步：跑 token 1                     -> 预测 token 2
第 2 步：跑 token 1, 2                  -> 预测 token 3
第 3 步：跑 token 1, 2, 3               -> 预测 token 4
...
第 n 步：跑 token 1, 2, ..., n          -> 预测 token n+1
\`\`\`

这样生成 1,000 个 token，一共要跑 $1 + 2 + \\dots + 1000 = 500{,}500$ 个 token 位置，几乎全是重复劳动：光 token 1 就被算了 1,000 遍。

**用了缓存**，每一步**只把新 token** 送进网络。在每一层 attention 里，新 token 的 $q$ 和存好的前面所有 token 的 $k$ 算分数，再按权重去取存好的 $v$，最后把新 token 自己的 $k$、$v$ 也存进去。这样生成 1,000 个 token，只要跑 1,000 个 token 位置。

## 2. 只在生成时用，训练不用

KV cache 只用在 inference，准确地说是模型一个一个生成 token 的时候。训练时用不着。

**训练时**，整段文本事先都是已知的。比如一个 8K 的样本，8,192 个 token 一次全部送进网络：每一层一次算出所有 token 的 $q$、$k$、$v$，[causal mask](#info:causal_mask_notes) 保证每个 token 只看得到它前面的 token。跑一遍 forward，每个位置的预测就都有了，再拿它们一起算 loss。每个 token 的 $k$、$v$ 在每层只算一次，本来就没有重复计算，缓存也就没什么可省的。

**生成时**，后面的 token 还不存在，只能一个一个地出：出一个，接到输入后面，再跑下一步，也就是第 1 节那个循环。重复计算正是这个循环带来的，KV cache 就是为了省掉它。

🆕 **post-training 里也会用到。** RLHF、GRPO 这类强化学习方法，要先让模型自己生成回答（叫 rollout），再给回答打分、更新参数。生成回答这一步其实就是 inference，所以会用 KV cache，通常直接交给 vLLM 这类推理引擎来跑。更新参数那一步还是普通的训练，不用缓存。

## 3. 为什么存下来的 k、v 一直有效

缓存能成立，前提是新 token 加进来以后，前面 token 的 $k$、$v$ 不会变。它们确实不会变，原因是 [causal mask](#info:causal_mask_notes)：

- 第 1 层里，token $i$ 的 $k$、$v$ 只由 token $i$ 自己的 embedding 算出来。
- token $i$ 在第 1 层的输出会混入 token $1, \\dots, i$ 的信息，但不会混入 $i$ 后面的任何 token，因为 causal mask 把后面的挡住了。
- 所以 token $i$ 进入第 2 层的输入，以及它在第 2 层的 $k$、$v$，同样只取决于 token $1, \\dots, i$。往后每一层都是同样的道理。

后来的 token 在任何一层都影响不到 token $i$，所以 token $i$ 的 $k$、$v$ 一算出来就定了。

$q$ 不用存：前面 token 的 $q$ 只用来算它自己那个位置的输出，而那个输出早就算完了。

## 4. 存什么，什么时候存

**存什么。** 每一层都存到目前为止所有 token 的 $k$ 和 $v$，按 head 分开：

\`\`\`text
每一层：  k: (B, n_kv_head, T, hs)    v: (B, n_kv_head, T, hs)
\`\`\`

LLaMA 存的 $k$ 是 [RoPE](#info:rope_notes) **转过之后**的。token 的位置不会变，所以转的角度也不会变。

**什么时候存。** 生成分两个阶段：

- **prefill（预填充）**：先把整段 prompt 一次性送进网络，和训练时一样并行计算。这一步把 prompt 的 $k$、$v$ 全部存进缓存，同时预测出第一个新 token。
- **decode（解码）**：之后每一步只处理一个 token，每层往缓存里追加一份 $k$、$v$。

## 5. 会有多大

每个 token 在每一层、每个 k/v head 上，都要存一个 $k$ 向量和一个 $v$ 向量：

\`\`\`text
字节数 = 2（k 和 v）× 层数 × k/v head 数 × head size × token 数 × 每个数的字节数
\`\`\`

用 16 位存（每个数 2 字节）：

\`\`\`text
                        层数   k/v head 数   head size   每个 token    整个上下文
GPT-2 small              12       12            64        36 KB      1,024 个 token：36 MB
LLaMA 3 8B（若用 MHA）    32       32           128       512 KB      8K：4 GB，128K：64 GB
LLaMA 3 8B（GQA）         32        8           128       128 KB      8K：1 GB，128K：16 GB
\`\`\`

对比一下，LLaMA 3 8B 的权重本身大约 15 GB。

- GPT-2 的 KV cache 很小；但现在的模型一旦上下文很长，KV cache 可能和模型本身一样大。
- 它随上下文长度线性增长，还要再乘上同时处理的请求数：每个请求都有自己的一份。

## 6. 它还决定了速度

decode 的每一步，GPU 都要从显存里把所有权重和整个 KV cache 读一遍，但只为一个新 token 做很少的计算。所以生成速度主要卡在**要读多少字节**，而不是计算量。KV cache 越小，不光省内存，生成也越快。

## 7. 🆕 给 KV cache 瘦身的方法

上面公式里的每一项，都有人想办法压缩：

**减少 k/v head 数。**

- **MQA**（2019）：所有 query head 只共用一组 k/v。最省，但效果会差一些。
- **GQA**（2023）：留几组 k/v，每组由几个 query head 共用。LLaMA 3 留了 8 组，详见 [GQA](#info:gqa_notes)。

**把 k、v 本身变小。**

- **MLA**（DeepSeek-V2，2024）：每个 token 只存一个低维的压缩向量，用的时候再从它还原出 $k$ 和 $v$。DeepSeek 说 KV cache 比他们之前的 67B 模型小了 93%。
- **量化**：把 $k$、$v$ 从 16 位改成 8 位甚至 4 位来存。

**少存一些 token。**

- **滑动窗口注意力**（Mistral 7B，2023）：每个 token 只看最近 4,096 个 token，所以缓存里最多也只存 4,096 个。
- **局部层和全局层搭配**（Gemma 2、Gemma 3）：大部分层只看一个短窗口，只有全局层需要完整的缓存。
- **丢掉不重要的 token**（H2O、StreamingLLM，2023）：只留下得到注意力最多的 token 和开头几个 token，其余的扔掉。

**让几层共用一份缓存。**

- **跨层共享**（CLA，2024）：相邻几层共用同一份 $k$ 和 $v$。

**更好地管理内存**（这是服务端的事，模型本身不变）。

- **PagedAttention**（vLLM，2023）：缓存变长时按小块分配，就像操作系统管理内存分页一样，浪费很少，一张 GPU 能同时跑更多请求。
- **前缀缓存（prefix caching）**：开头相同的请求，比如用同一段系统提示词，这一段的缓存只算一次，大家共用。

## 参考资料

1. **Fast Transformer Decoding: One Write-Head is All You Need (2019).** Shazeer 提出多查询注意力（MQA）。[MQA ↗](https://arxiv.org/abs/1911.02150)
2. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2023).** Ainslie 等人。[GQA ↗](https://arxiv.org/abs/2305.13245)
3. **Efficient Memory Management for Large Language Model Serving with PagedAttention (2023).** Kwon 等人，即 vLLM 的论文。[PagedAttention ↗](https://arxiv.org/abs/2309.06180)
4. **Mistral 7B (2023).** 滑动窗口注意力和循环使用的 KV cache。[Mistral 7B ↗](https://arxiv.org/abs/2310.06825)
5. **Efficient Streaming Language Models with Attention Sinks (2023).** Xiao 等人，即 StreamingLLM。[StreamingLLM ↗](https://arxiv.org/abs/2309.17453)
6. **DeepSeek-V2 (2024).** 提出多头潜在注意力（MLA）。[DeepSeek-V2 ↗](https://arxiv.org/abs/2405.04434)
`,
    },
  },
};
