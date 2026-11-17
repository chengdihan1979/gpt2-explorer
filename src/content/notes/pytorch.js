// Notes: PyTorch building blocks.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  linear: {
    en: {
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
    zh: {
      title: "Linear（线性层）",
      md: `
**\`torch.nn.Linear\`**：全连接（仿射）层。

## 公式

给定输入 $x \\in \\mathbb{R}^{*,\\,\\text{in}}$，

$$
y = x W^\\top + b
$$

其中 $W \\in \\mathbb{R}^{\\text{out}\\times\\text{in}},\\ b \\in \\mathbb{R}^{\\text{out}}$。

## 说明

- 把 \`in_features -> out_features\`。
- 支持前导维度的 broadcasting：\`(N, ..., in) -> (N, ..., out)\`。
- 文档：[nn.Linear ↗](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
`,
    },
  },

  linear_weight_notes: {
    en: {
      title: "How nn.Linear computes (hover)",
      md: `
**How \`nn.Linear\` computes**

\`nn.Linear(in_features, out_features)\` computes

\`\`\`text
y = x @ weight.T + bias
\`\`\`

- \`weight\` is **stored** as (out, in). For \`nn.Linear(4, 2)\` it is (2, 4).
- The multiplication uses the transpose \`weight.T\`, which is (in, out) = (4, 2), with \`x\` on the left:

\`\`\`text
x       @  weight.T  =  y
(4,)       (4, 2)       (2,)
\`\`\`

**Why store it as (out, in)? One output = one row.** Row $i$ of \`weight\` holds all the weights used by output $i$:

\`\`\`text
weight[0] = [1, 0, 1,  0]  → only used for out[0] = weight[0] · x
weight[1] = [0, 1, 1, -1]  → only used for out[1] = weight[1] · x
\`\`\`

So \`weight[i]\` gives output $i$'s weights directly. This is why each row of \`fc_in.weight\` can be read as one pattern detector. Stored as (in, out), the same weights would be a column, \`weight[:, i]\`. The results would be identical; (out, in) is a convention that matches the math y = W x.

**Why \`x @ W.T\` and not \`W x\`?** They give the same numbers; only the direction of the vectors differs. Math writes $x$ as a column, PyTorch stores each token as a **row**, and the two forms are transposes of each other: (W x)ᵀ = xᵀ Wᵀ.

| | Vector direction | Formula |
| --- | --- | --- |
| Math / papers | column | y = W x |
| PyTorch | row (one row per token) | y = x @ W.T |

Since \`x\` is (B, T, in) with one token per row, the matrix has to go on the right. \`x @ W.T\` works directly, while \`W @ x\` fails on the shapes.

**Leading dimensions are kept.** \`@\` only acts on the last dimension of \`x\`. If \`x\` is (B, T, 4), think of it as B × T separate rows of 4 numbers, each multiplied by the same (4, 2) matrix:

\`\`\`text
(B, T, 4) @ (4, 2) → (B, T, 2)
\`\`\`

**What actually runs.** \`out[i] = weight[i] · x\` is the mathematical definition, not the order of execution:

- \`.T\` copies nothing. It only swaps the strides, so \`weight.T\` reads the same memory in a different order.
- The whole layer is **one** matrix multiplication (\`addmm\`: bias + x @ weight.T). The library (BLAS on CPU, cuBLAS on GPU) splits it into tiles and computes all tokens and all outputs in parallel, not one output after another.
- The (out, in) layout is convenient: both \`x[t]\` and \`weight[i]\` are contiguous along the "in" dimension that the dot product sums over. But the libraries are fast with other layouts too, so the main reason for (out, in) is convention, not speed.

In GPT-2:

| Layer | Stored weight | Multiplication |
| --- | --- | --- |
| \`fc_in = nn.Linear(768, 3072)\` | (3072, 768) | (B, T, 768) @ (768, 3072) → (B, T, 3072) |
| \`fc_out = nn.Linear(3072, 768)\` | (768, 3072) | (B, T, 3072) @ (3072, 768) → (B, T, 768) |
`,
    },
    zh: {
      title: "nn.Linear 怎么算 (hover)",
      md: `
**\`nn.Linear\` 怎么算**

\`nn.Linear(in_features, out_features)\` 计算的是：

\`\`\`text
y = x @ weight.T + bias
\`\`\`

- \`weight\` **存储**成 (out, in)。\`nn.Linear(4, 2)\` 的 weight 是 (2, 4)。
- 乘法时用转置 \`weight.T\`，形状是 (in, out) = (4, 2)，\`x\` 在左边：

\`\`\`text
x       @  weight.T  =  y
(4,)       (4, 2)       (2,)
\`\`\`

**为什么存成 (out, in)？一个输出 = 一行。** \`weight\` 的第 $i$ 行装着第 $i$ 个输出用到的全部权重：

\`\`\`text
weight[0] = [1, 0, 1,  0]  → only used for out[0] = weight[0] · x
weight[1] = [0, 1, 1, -1]  → only used for out[1] = weight[1] · x
\`\`\`

所以 \`weight[i]\` 直接就拿到第 $i$ 个输出的权重。这也是为什么 \`fc_in.weight\` 的每一行可以看成一个模式检测器。如果存成 (in, out)，同一组权重就变成一列 \`weight[:, i]\`。结果完全一样；(out, in) 是一种约定，和数学里的 y = W x 对应。

**为什么是 \`x @ W.T\` 而不是 \`W x\`？** 两者算出来的数一样，只是向量的方向不同。数学里 $x$ 默认是列向量，PyTorch 里每个 token 存成**一行**，两种写法互为转置：(W x)ᵀ = xᵀ Wᵀ。

| | 向量方向 | 写法 |
| --- | --- | --- |
| 数学 / 论文 | 列向量 | y = W x |
| PyTorch | 行向量（每个 token 一行） | y = x @ W.T |

\`x\` 是 (B, T, in)，一个 token 一行，所以矩阵只能放在右边。\`x @ W.T\` 直接就能算，\`W @ x\` 则形状对不上，会报错。

**前面的维度原样保留。** \`@\` 只作用在 \`x\` 的最后一维上。如果 \`x\` 是 (B, T, 4)，可以把它看成 B × T 个各自独立的行，每行 4 个数，各自乘同一个 (4, 2) 的矩阵：

\`\`\`text
(B, T, 4) @ (4, 2) → (B, T, 2)
\`\`\`

**实际是怎么算的。** \`out[i] = weight[i] · x\` 是数学上的定义，不是执行顺序：

- \`.T\` 不复制任何数据，只交换 stride，\`weight.T\` 是换一种顺序读同一块内存。
- 整层就是**一次**矩阵乘法（\`addmm\`：bias + x @ weight.T）。底层库（CPU 上是 BLAS，GPU 上是 cuBLAS）把它切成小块，所有 token、所有输出并行计算，而不是一个输出接一个输出地算。
- (out, in) 布局很顺手：点积要沿 "in" 这一维求和，而 \`x[t]\` 和 \`weight[i]\` 在这一维上都是连续存放的。但底层库换成别的布局也一样快，所以选 (out, in) 主要是因为约定，而不是速度。

在 GPT-2 里：

| 层 | 存储的 weight | 乘法 |
| --- | --- | --- |
| \`fc_in = nn.Linear(768, 3072)\` | (3072, 768) | (B, T, 768) @ (768, 3072) → (B, T, 3072) |
| \`fc_out = nn.Linear(3072, 768)\` | (768, 3072) | (B, T, 3072) @ (3072, 768) → (B, T, 768) |
`,
    },
  },

  vector_1d_notes: {
    en: {
      title: "(4,) vs (1, 4) (hover)",
      md: `
**\`(4,)\` is a 1-D tensor, not a 1×4 matrix**

\`(4,)\` means one dimension of length 4. It is neither 1 × 4 nor 4 × 1:

\`\`\`text
(4,)     1-D, 4 numbers                      a.dim() = 1
(1, 4)   2-D, 1 row and 4 columns (row)      a.dim() = 2
(4, 1)   2-D, 4 rows and 1 column (column)   a.dim() = 2
\`\`\`

The comma is Python syntax: a tuple with one element is written \`(4,)\`, because \`(4)\` is just the number 4.

**How \`(4,) @ (4, 2)\` works.** For \`@\`, PyTorch temporarily treats a 1-D tensor on the **left** as a row (1, 4), multiplies, then removes the extra 1:

\`\`\`text
(4,)  →  treated as (1, 4)  @ (4, 2)  =  (1, 2)  →  drop the 1  →  (2,)
\`\`\`

The numbers are the same as with a 1 × 4 matrix; only the shape lacks the 1. Checked in PyTorch, with \`W\` the (4, 2) matrix \`weight.T\` from the example:

\`\`\`python
a.shape                        # torch.Size([4])
(a @ W).shape                  # torch.Size([2])     1-D in, 1-D out
(a.reshape(1, 4) @ W).shape    # torch.Size([1, 2])  2-D in, 2-D out
a.reshape(4, 1) @ W            # error: 4x1 and 4x2 cannot be multiplied
\`\`\`

The last line shows that a column (4, 1) would not work here. On the left of \`@\`, a 1-D tensor behaves like a row.

A 1-D tensor on the **right** of \`@\` is treated as a column instead: \`W.T @ a\` is (2, 4) @ (4,) → (2,).

In the real model, \`x\` is (B, T, 3072) rather than 1-D. Each token's 3072 numbers play the role of \`a\`.
`,
    },
    zh: {
      title: "(4,) 和 (1, 4) (hover)",
      md: `
**\`(4,)\` 是一维张量，不是 1×4 矩阵**

\`(4,)\` 表示只有一个维度，长度为 4。它既不是 1 × 4，也不是 4 × 1：

\`\`\`text
(4,)     1-D, 4 numbers                      a.dim() = 1
(1, 4)   2-D, 1 row and 4 columns (row)      a.dim() = 2
(4, 1)   2-D, 4 rows and 1 column (column)   a.dim() = 2
\`\`\`

逗号是 Python 的写法：只有一个元素的 tuple 要写成 \`(4,)\`，因为 \`(4)\` 就只是数字 4。

**\`(4,) @ (4, 2)\` 怎么乘。** 做 \`@\` 时，PyTorch 把放在**左边**的一维张量临时当成行向量 (1, 4)，乘完再把多出来的 1 去掉：

\`\`\`text
(4,)  →  treated as (1, 4)  @ (4, 2)  =  (1, 2)  →  drop the 1  →  (2,)
\`\`\`

算出来的数和用 1 × 4 矩阵完全一样，只是形状少了那个 1。在 PyTorch 里验证过（\`W\` 是例子里那个 (4, 2) 的 \`weight.T\`）：

\`\`\`python
a.shape                        # torch.Size([4])
(a @ W).shape                  # torch.Size([2])     一维进，一维出
(a.reshape(1, 4) @ W).shape    # torch.Size([1, 2])  二维进，二维出
a.reshape(4, 1) @ W            # 报错：4x1 和 4x2 不能相乘
\`\`\`

最后一行说明，如果当成列向量 (4, 1) 就乘不了。放在 \`@\` 左边时，一维张量的行为和行向量一样。

一维张量放在 \`@\` **右边**时，则被当成列向量：\`W.T @ a\` 是 (2, 4) @ (4,) → (2,)。

在真实模型里，\`x\` 是 (B, T, 3072)，不是一维的。每个 token 的那 3072 个数，就相当于这里的 \`a\`。
`,
    },
  },

  matmul_example_notes: {
    en: {
      title: "a @ b: a numerical example (hover)",
      md: `
**\`a @ b\` is matrix multiplication (\`torch.matmul\`).** Each entry of the result is a row of \`a\` times a column of \`b\`: multiply element by element, then add up.

**A 2-D example: (2 × 3) @ (3 × 2) → (2 × 2)**

\`\`\`text
a = [[1, 2, 3],        b = [[1, 0],
     [4, 5, 6]]             [0, 1],
                            [2, 2]]
\`\`\`

Entry (row $i$, column $j$) of the result = row $i$ of \`a\` · column $j$ of \`b\`:

\`\`\`text
(0, 0): row 0 · col 0 = 1×1 + 2×0 + 3×2 = 7
(0, 1): row 0 · col 1 = 1×0 + 2×1 + 3×2 = 8
(1, 0): row 1 · col 0 = 4×1 + 5×0 + 6×2 = 16
(1, 1): row 1 · col 1 = 4×0 + 5×1 + 6×2 = 17

a @ b = [[ 7,  8],
         [16, 17]]
\`\`\`

**The same thing, one row at a time**

Row $i$ of the result is a **weighted sum of the rows of \`b\`**, with weights from row $i$ of \`a\`:

\`\`\`text
row 0 of (a @ b) = 1 × [1, 0] + 2 × [0, 1] + 3 × [2, 2]
                 = [1, 0] + [0, 2] + [6, 6]
                 = [7, 8]
\`\`\`

This is exactly how \`att @ v\` works: each row of \`att\` holds the weights, and the rows of \`v\` are what gets mixed. A row like \`[1, 0, 0]\` just copies the first row of \`b\`, like the \`I\` row in attention that puts 100% on itself.

**Shape rule**

\`\`\`text
(m × n) @ (n × p) → (m × p)
\`\`\`

The inner sizes (\`n\`) must match: a row of \`a\` and a column of \`b\` need the same length. Otherwise PyTorch raises a shape error.

**More than 2 dimensions: a batch of matrix products**

With more than two dimensions, \`@\` multiplies only the **last two**; the leading dimensions are a batch, and each pair is multiplied independently:

\`\`\`text
a: (2, 2, 3)    two (2 × 3) matrices: the a above, and [[1, 0, 0], [0, 0, 1]]
b: (2, 3, 2)    two (3 × 2) matrices: the b above, twice
a @ b: (2, 2, 2)

batch 0:  [[1, 2, 3], [4, 5, 6]] @ b  = [[ 7,  8], [16, 17]]
batch 1:  [[1, 0, 0], [0, 0, 1]] @ b  = [[ 1,  0], [ 2,  2]]
\`\`\`

In GPT-2, \`att @ v\` is \`(B, n_head, T, T) @ (B, n_head, T, hs) → (B, n_head, T, hs)\`: $B \\times n\\_head$ separate products of $(T \\times T) @ (T \\times hs)$, one per sequence and head.
`,
    },
    zh: {
      title: "a @ b 数值例子",
      md: `
**\`a @ b\` 是矩阵乘法（\`torch.matmul\`）。** 结果里的每一项，都是 \`a\` 的一行乘以 \`b\` 的一列：对应元素相乘，再加起来。

**二维的例子：(2 × 3) @ (3 × 2) → (2 × 2)**

\`\`\`text
a = [[1, 2, 3],        b = [[1, 0],
     [4, 5, 6]]             [0, 1],
                            [2, 2]]
\`\`\`

结果的第 $i$ 行第 $j$ 列 = \`a\` 的第 $i$ 行 · \`b\` 的第 $j$ 列：

\`\`\`text
(0, 0): row 0 · col 0 = 1×1 + 2×0 + 3×2 = 7
(0, 1): row 0 · col 1 = 1×0 + 2×1 + 3×2 = 8
(1, 0): row 1 · col 0 = 4×1 + 5×0 + 6×2 = 16
(1, 1): row 1 · col 1 = 4×0 + 5×1 + 6×2 = 17

a @ b = [[ 7,  8],
         [16, 17]]
\`\`\`

**换个角度：一次看一行**

结果的第 $i$ 行，是 **\`b\` 各行的加权和**，权重来自 \`a\` 的第 $i$ 行：

\`\`\`text
row 0 of (a @ b) = 1 × [1, 0] + 2 × [0, 1] + 3 × [2, 2]
                 = [1, 0] + [0, 2] + [6, 6]
                 = [7, 8]
\`\`\`

\`att @ v\` 正是这样算的：\`att\` 的每一行是权重，\`v\` 的各行是被混合的内容。像 \`[1, 0, 0]\` 这样的一行，就是原样复制 \`b\` 的第一行，相当于 attention 里 \`I\` 那一行 100% 看自己。

**形状规则**

\`\`\`text
(m × n) @ (n × p) → (m × p)
\`\`\`

中间的大小（\`n\`）必须相等：\`a\` 的一行和 \`b\` 的一列要一样长。否则 PyTorch 会报形状错误。

**超过二维：一批矩阵乘法**

超过二维时，\`@\` 只对**最后两维**做矩阵乘法；前面的维度当作批次，每一对各自独立地相乘：

\`\`\`text
a: (2, 2, 3)    two (2 × 3) matrices: the a above, and [[1, 0, 0], [0, 0, 1]]
b: (2, 3, 2)    two (3 × 2) matrices: the b above, twice
a @ b: (2, 2, 2)

batch 0:  [[1, 2, 3], [4, 5, 6]] @ b  = [[ 7,  8], [16, 17]]
batch 1:  [[1, 0, 0], [0, 0, 1]] @ b  = [[ 1,  0], [ 2,  2]]
\`\`\`

在 GPT-2 里，\`att @ v\` 是 \`(B, n_head, T, T) @ (B, n_head, T, hs) → (B, n_head, T, hs)\`：一共做 $B \\times n\\_head$ 次独立的 $(T \\times T) @ (T \\times hs)$ 乘法，每个序列的每个 head 各一次。
`,
    },
  },

  broadcasting: {
    en: {
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
    zh: {
      title: "PyTorch Broadcasting（广播）",
      md: `
**Broadcasting** 是 PyTorch（以及 NumPy）用来在不同形状的张量之间做运算、又不需要显式复制数据的一套规则。

## 基本规则

两个张量兼容的条件是，对每个维度（从末尾开始比较）：

- 两者相等，或者
- 其中一个是 $1$。

较小的张量会被"拉伸"（虚拟地重复）成较大的形状。

## 例子

### 例 1：向量 + 矩阵

$$
A \\in \\mathbb{R}^{3 \\times 4}, \\quad b \\in \\mathbb{R}^{1 \\times 4}
$$

运算：

$$
C = A + b
$$

这里 $b$ 沿第一个维度被 broadcast，所以：

$$
C \\in \\mathbb{R}^{3 \\times 4}
$$

### 例 2：列向量 + 行向量

$$
u \\in \\mathbb{R}^{3 \\times 1}, \\quad v \\in \\mathbb{R}^{1 \\times 4}
$$

运算：

$$
M = u + v
$$

形状扩展为 $(3, 4)$：

$$
M \\in \\mathbb{R}^{3 \\times 4}
$$

## 形式化规则

如果有两个形状

$$
(s_1, s_2, \\dots, s_k), \\quad (t_1, t_2, \\dots, t_k)
$$

那么它们 broadcast 兼容的条件是

$$
s_i = t_i \\quad \\text{or} \\quad s_i = 1 \\quad \\text{or} \\quad t_i = 1
$$

对所有 $i$ 成立（如有需要，在较短的形状左边补 $1$）。

## 主要好处

- 节省内存（不需要显式平铺复制）。
- 让代码更简洁。
- 在深度学习中大量使用，比如加 bias、加位置 embedding 等。
`,
    },
  },

  unsqueeze_notes: {
    en: {
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
    zh: {
      title: "unsqueeze",
      md: `
**在张量形状的第 \`dim\` 个位置插入一个大小为 1 的维度（不复制数据）。**

## 函数签名

\`\`\`python
torch.unsqueeze(input, dim)
# 或者
tensor.unsqueeze(dim)
\`\`\`

## 形状示例

$$
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(0)}\\ }\\ (1,\\ B,\\ T),\\qquad
(B,\\ T)\\ \\xrightarrow{\\ \\mathtt{unsqueeze(-1)}\\ }\\ (B,\\ T,\\ 1)
$$

## 合法的 \`dim\`

对 $n$ 维输入，合法的 $dim \\in [-(n+1),\\ n]$；负的 \`dim\` 从末尾开始数。

文档：[unsqueeze ↗](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)
`,
    },
  },

  contiguous_notes: {
    en: {
      title: "contiguous",
      md: `
**A tensor is contiguous if its elements are laid out in memory in the usual row-major order for its current shape.** Some ops (like \`transpose\`, or strided slicing such as \`a[:, ::2]\`) only change how memory is read, so the result becomes non-contiguous. \`.contiguous()\` copies the data into the new order, so that ops like \`.view(...)\` can work.

## A tensor in memory is one long row

Whatever its number of dimensions, a tensor is stored in memory as one flat sequence of numbers:

\`\`\`text
a = [[1, 2, 3],
     [4, 5, 6]]        shape (2, 3)

memory:  1  2  3  4  5  6
\`\`\`

PyTorch records how to read that sequence as a multi-dimensional shape with **strides**:

\`\`\`text
a.stride() = (3, 1)
  row + 1     → jump 3 places in memory
  column + 1  → jump 1 place in memory
\`\`\`

Reading row by row walks straight through memory, so \`a.is_contiguous()\` is \`True\`.

## \`transpose\` moves no data, only the strides

\`\`\`text
t = a.T = [[1, 4],
           [2, 5],
           [3, 6]]     shape (3, 2)

memory:  still 1  2  3  4  5  6    ← the same memory; nothing moved
t.stride() = (1, 3)
\`\`\`

Only the reading rule changes, which is why \`transpose\` is so fast. The cost: reading \`t\` row by row gives 1, 4, 2, 5, 3, 6, which jumps around in memory. That is what **non-contiguous** means: \`t.is_contiguous()\` is \`False\`.

## Why \`view\` fails

\`view\` does not move data either: it reads **the same memory** under a new shape. Calling \`t.view(6)\` would read memory as 1 2 3 4 5 6, but in \`t\`'s logical order the answer should be 1 4 2 5 3 6. These do not match, so PyTorch raises an error:

\`\`\`text
t.view(6)
RuntimeError: view size is not compatible with input tensor's size and stride ...
Use .reshape(...) instead.
\`\`\`

## What \`.contiguous()\` does

\`\`\`text
c = t.contiguous()
memory:  1  4  2  5  3  6     ← a new copy, in t's logical order
c.stride() = (2, 1)
c.view(6) = [1, 4, 2, 5, 3, 6]   ✓
\`\`\`

If the tensor is already contiguous, \`.contiguous()\` just returns it, with no copy.

## In attention

When [merging the heads](#info:reorder_merge_heads_notes), \`transpose(1, 2)\` leaves each token's heads scattered in memory. \`.contiguous()\` first lays them out next to each other, so that \`view(B, T, C)\` can join them directly.

## Two fixes

\`\`\`python
flat_ok1 = t.contiguous().view(-1)  # copy into order, then view
flat_ok2 = t.reshape(-1)            # reshape copies only if needed
\`\`\`

- \`.contiguous()\` copies the data into row-major order for the current shape.
- \`.reshape\` is more flexible: it returns a view if possible, and otherwise copies. The merge line could equally be written \`y.transpose(1, 2).reshape(B, T, C)\`.
`,
    },
    zh: {
      title: "contiguous",
      md: `
**如果一个 tensor 的元素在内存里按它当前形状的常规行优先（row-major）顺序排列，它就是 contiguous（连续）的。** 有些操作（比如 \`transpose\`，或者 \`a[:, ::2]\` 这样的跳步切片）只改变读取内存的方式，所以结果会变成不连续的。\`.contiguous()\` 把数据按新的顺序复制一份，\`.view(...)\` 这类操作才能用。

## tensor 在内存里是一长串数字

不管 tensor 是几维，它在内存里都存成一维的一长串：

\`\`\`text
a = [[1, 2, 3],
     [4, 5, 6]]        shape (2, 3)

memory:  1  2  3  4  5  6
\`\`\`

PyTorch 用 **stride**（步长）记录怎么把这一长串读成多维的形状：

\`\`\`text
a.stride() = (3, 1)
  row + 1     → jump 3 places in memory
  column + 1  → jump 1 place in memory
\`\`\`

一行一行地读，正好是在内存里从头走到尾，所以 \`a.is_contiguous()\` 是 \`True\`。

## \`transpose\` 不动数据，只改 stride

\`\`\`text
t = a.T = [[1, 4],
           [2, 5],
           [3, 6]]     shape (3, 2)

memory:  still 1  2  3  4  5  6    ← the same memory; nothing moved
t.stride() = (1, 3)
\`\`\`

只是读取规则换了，所以 \`transpose\` 非常快。代价是：按 \`t\` 的行顺序读出来是 1, 4, 2, 5, 3, 6，在内存里要跳着读。这就叫**不连续**（non-contiguous）：\`t.is_contiguous()\` 是 \`False\`。

## \`view\` 为什么不行

\`view\` 也不搬数据，它只是用新的形状去解释**同一串内存**。直接调用 \`t.view(6)\`，读出来的会是内存里的 1 2 3 4 5 6，但按 \`t\` 的逻辑顺序应该是 1 4 2 5 3 6。两者对不上，所以 PyTorch 直接报错：

\`\`\`text
t.view(6)
RuntimeError: view size is not compatible with input tensor's size and stride ...
Use .reshape(...) instead.
\`\`\`

## \`.contiguous()\` 做了什么

\`\`\`text
c = t.contiguous()
memory:  1  4  2  5  3  6     ← a new copy, in t's logical order
c.stride() = (2, 1)
c.view(6) = [1, 4, 2, 5, 3, 6]   ✓
\`\`\`

如果 tensor 本来就是连续的，\`.contiguous()\` 直接返回它自己，不复制。

## 在 attention 里

[合并 head](#info:reorder_merge_heads_notes) 时，\`transpose(1, 2)\` 之后，每个 token 的各个 head 在内存里是分散的。\`.contiguous()\` 先把它们挨在一起排好，\`view(B, T, C)\` 才能直接把它们拼起来。

## 两种修法

\`\`\`python
flat_ok1 = t.contiguous().view(-1)  # copy into order, then view
flat_ok2 = t.reshape(-1)            # reshape copies only if needed
\`\`\`

- \`.contiguous()\` 把数据按当前形状的行优先顺序复制一份。
- \`.reshape\` 更灵活：能 view 就返回 view，不能就复制。所以合并 head 那一行也可以写成 \`y.transpose(1, 2).reshape(B, T, C)\`。
`,
    },
  },

  torch_ones_notes: {
    en: {
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
    zh: {
      title: "torch.ones",
      md: `
**创建一个给定形状、全部填充为 1 的张量。**

## 函数签名

\`\`\`python
torch.ones(*sizes, dtype=None, device=None, requires_grad=False)
\`\`\`

## 例子

- \`torch.ones(2, 3)\`
- \`torch.ones_like(x)\`

[torch.ones 文档 ↗](https://docs.pytorch.org/docs/stable/generated/torch.ones.html)
`,
    },
  },

  torch_tril_notes: {
    en: {
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
    zh: {
      title: "torch.tril",
      md: `
**返回矩阵的下三角部分；把指定对角线以上的元素置零。**

## 函数签名

\`\`\`python
torch.tril(input, diagonal=0)
\`\`\`

## \`diagonal\` 参数

- 如果 \`diagonal = 0\`：保留主对角线及其以下的部分。
- 如果 \`diagonal > 0\`：正值表示额外包含主对角线上方相应数量的对角线。
- 如果 \`diagonal < 0\`：负值表示排除主对角线下方相应数量的对角线。

文档：[torch.tril ↗](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)
`,
    },
  },

  register_buffer_notes: {
    en: {
      title: "register_buffer",
      md: `
**A buffer is a tensor that belongs to the model but is not trained.** It moves with the model (\`.to()\`, \`.cuda()\`, \`.cpu()\`) and is saved and loaded with it ([\`state_dict\`](#info:state_dict_notes)), but it is not a parameter, so the optimizer never updates it.

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
    zh: {
      title: "register_buffer",
      md: `
**buffer 是属于模型、但不参与训练的 tensor。** 它会跟着模型移动（\`.to()\`、\`.cuda()\`、\`.cpu()\`），也会和模型一起保存、加载（[\`state_dict\`](#info:state_dict_notes)）；但它不是 parameter，optimizer 永远不会更新它。

## Parameter、普通属性、buffer 的区别

在 \`nn.Module\` 里存一个 tensor，有三种方式。以 causal mask 为例：

\`\`\`python
# 1. parameter
self.mask = nn.Parameter(torch.tril(torch.ones(1024, 1024)))

# 2. 普通属性
self.mask = torch.tril(torch.ones(1024, 1024))

# 3. buffer
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))
\`\`\`

三者在三件事上不同：

|  | 会被训练吗 | 跟着 \`.to("cuda")\` 移动吗 | 存进 \`state_dict\` 吗 |
| --- | :---: | :---: | :---: |
| Parameter | 会 | 会 | 会 |
| 普通属性 | 不会 | 不会 | 不会 |
| buffer | **不会** | **会** | **会** |

> **类比。** 模型是一个工具箱。Parameter 是要磨的刀，训练就是在磨它。buffer 是箱子里的尺子：不用磨，但搬家时跟着箱子一起走。普通属性是贴在旧房子墙上的便条：搬家时会被落下。

## 为什么 causal mask 用 buffer

- **不能是 Parameter：** mask 是固定的规则，"不能往后看"不该被学习。如果是 Parameter，optimizer 每一步都会更新它，对角线上方的 0 可能慢慢变成 0.03，模型就开始偷看了。
- **不能是普通属性：** \`model.to("cuda")\` 只会搬 Parameter 和 buffer。普通属性留在 CPU 上，forward 时 \`att\`（在 GPU 上）和 \`mask\`（在 CPU 上）一起运算，就会报 device 不匹配的错。
- **buffer 正好合适：** 不被训练，但会和模型一起搬到 GPU，一起保存和加载。

## \`register_buffer\` 的用法

\`\`\`python
Module.register_buffer(name, tensor, persistent=True)
\`\`\`

- **name：** buffer 会成为同名属性。执行 \`self.register_buffer("mask", ...)\` 之后，forward 里直接用 \`self.mask\`。
- **persistent：** 默认 \`True\`，buffer 会存进 \`state_dict\`。设为 \`False\` 时，它仍然跟着模型移动，但不会被保存；适合重新生成很便宜的东西。
- **列出 buffer：** buffer 出现在 \`model.buffers()\` 里，不在 \`model.parameters()\` 里。optimizer 是用 \`model.parameters()\` 建的，所以根本看不到它们。

"不被训练"指的是不被梯度更新。如果代码主动写入，buffer 的值还是可以变的：BatchNorm 把 \`running_mean\` 和 \`running_var\` 存成 buffer，并在 forward 时更新它们。causal mask 则从来不变。

**常见用途：** mask（比如 causal mask）、running statistics、固定的索引表，以及其他和模型绑定的常量。

## 在 GPT-2 里

\`\`\`python
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))
\`\`\`

这一行在 \`__init__\` 里把 $(1,\\ 1,\\ 1024,\\ 1024)$ 的下三角 mask 存一次。每次 forward 再切成当前长度，详见 [Causal Mask](#info:causal_mask_notes)。

> 🆕 后来的实现（比如新版 nanoGPT）大多直接调用 \`F.scaled_dot_product_attention(..., is_causal=True)\`，在 kernel 内部处理 causal mask，不再存 1024 × 1024 的 mask buffer。

文档：[register_buffer ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
`,
    },
  },

  state_dict_notes: {
    en: {
      title: "state_dict",
      md: `
**\`state_dict\` is a dictionary holding a model's parameters and persistent buffers: the key is a name, the value is the tensor.** It is the model's save file. It is a PyTorch concept: \`nn.Module\` and optimizers both have a \`state_dict()\` method.

## What it looks like

For the GPT2 code on this site, \`model.state_dict()\` looks roughly like this (shapes for real GPT-2):

\`\`\`text
"tok_emb.weight"                 → (50257, 768)
"pos_emb.weight"                 → (1024, 768)
"blocks.0.ln1.weight"            → (768,)              # γ
"blocks.0.ln1.bias"              → (768,)              # β
"blocks.0.attn.mask"             → (1, 1, 1024, 1024)  # buffer
"blocks.0.attn.c_attn.weight"    → (2304, 768)
"blocks.0.attn.c_attn.bias"      → (2304,)
...
"blocks.11.mlp. ..."
"ln_f.weight"                    → (768,)
"ln_f.bias"                      → (768,)
"lm_head.weight"                 → (50257, 768)
\`\`\`

Each key is the attribute path in the code: \`self.blocks[0].attn.c_attn.weight\` becomes \`"blocks.0.attn.c_attn.weight"\`.

| What | In \`state_dict\`? |
| --- | :---: |
| Parameters (W, b, γ, β, embedding tables) | yes |
| Persistent [buffers](#info:register_buffer_notes) (e.g. \`mask\`) | yes |
| Plain attributes | no |
| Activations computed during forward | no |
| Modules without tensors (e.g. \`nn.Dropout\`) | no entry |

> With [weight tying](#info:weight_tying_notes), \`tok_emb.weight\` and \`lm_head.weight\` both appear as keys, but they point to the **same** tensor.

### Does the mask need to be saved?

For the causal mask, **not really**:

- It is always rebuilt by \`torch.tril(torch.ones(1024, 1024))\` in \`__init__\`, with the same result every time.
- Loading starts with \`model = GPT2(...)\`, which already builds the mask; the saved copy just overwrites it with identical values.

It also takes space:

\`\`\`text
one mask:  1024 × 1024 × 4 bytes (float32) = 4 MB
12 blocks: 12 × 4 MB = 48 MB
\`\`\`

That is close to 10% of a ~500 MB file spent on 12 identical triangles.

**Why is it saved by default, then?** \`register_buffer\` is a general tool, and \`persistent=True\` is the right default for buffers that **cannot** be rebuilt, such as BatchNorm's \`running_mean\`, a statistic gathered during training. For the mask, a better choice is:

\`\`\`python
self.register_buffer("mask", torch.tril(...), persistent=False)
\`\`\`

It still moves to the GPU with the model, but is not written to the file. The code on this site uses the default, so its mask is saved; some GPT-2 implementations also skip these mask keys when loading.

## Saving and loading

A \`state_dict\` is **not a file**: it is a Python dictionary in memory. It becomes a file only when written to disk with \`torch.save\`:

\`\`\`python
sd = model.state_dict()          # step 1: a dict in memory; no file yet
torch.save(sd, "gpt2.pt")        # step 2: write the dict to the file gpt2.pt
\`\`\`

Without step 2, the dictionary is gone when the program exits. \`torch.save\` serializes it with Python's pickle into a zip-format file; the name and extension are up to you (\`.pt\` and \`.pth\` are common).

Loading goes the other way:

\`\`\`python
model = GPT2(...)                # build the same structure first (randomly initialized)
sd = torch.load("gpt2.pt")       # file → dict in memory
model.load_state_dict(sd)        # dict → values filled into the model
\`\`\`

\`state_dict\` stores **values only, not the structure**. To load, the code must first [build a model with the same structure](#hover:state_dict_build_notes); the values are then matched **by name**. If a name or a shape does not match, \`load_state_dict\` raises an error.

> **Analogy.** The model code is a building's **blueprint**; the \`state_dict\` is the **inventory** of what goes in each room. When moving, you take only the inventory: at the new place you build the same building from the blueprint, then put everything back room by room according to the inventory.

### One file or several

- GPT-2 small has 124M parameters; in float32 that is about 500 MB, so one file is enough.
- 🆕 Models of tens of GB usually **split** the dictionary into several shards, one file each, plus an index file recording which shard holds each key. A typical Hugging Face layout:

\`\`\`text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json      ← which file each key is in
\`\`\`

### 🆕 Why \`.safetensors\` is now common

A \`.pt\` file is stored with pickle, and unpickling **can execute arbitrary code**. Loading an untrusted \`.pt\` file downloaded from the internet is therefore a security risk (recent PyTorch versions default to \`torch.load(..., weights_only=True)\` to limit this).

\`.safetensors\` stores only names, shapes and values, so it cannot hide code, and it also loads faster. Hugging Face now uses it by default.

## Checkpoints during training

Training does not save just once at the end; it saves a **checkpoint** every so often:

\`\`\`python
for step in range(max_steps):
    ...  # one training step
    if step % 1000 == 0:
        torch.save({"model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step}, f"ckpt_{step}.pt")
\`\`\`

Two reasons:

- **Crashes.** Training can run for days. If a machine fails, training resumes from the latest checkpoint instead of starting over. For that the **optimizer's** state is needed too: \`optimizer.state_dict()\` holds its own state, e.g. Adam's two running statistics for every parameter.
- **The last step is not always the best.** Loss on a validation set is tracked during training. An earlier checkpoint is often better than the last one (later steps may start to overfit), and the best one is what gets released.

After training, for inference or release, usually **only the model's \`state_dict\`** is kept. Adam's optimizer state is about twice the size of the parameters and is not needed for inference.

## Outside PyTorch

The name \`state_dict\` is PyTorch's, but the idea (save parameters as name → array) is common:

- **Hugging Face:** the \`.safetensors\` file written by \`save_pretrained()\` is essentially a \`state_dict\`.
- **TensorFlow/Keras:** \`model.save_weights()\` and checkpoints.
- **JAX/Flax:** the parameters already are a nested dictionary (a pytree), which is saved directly.

Docs: [Saving and loading models ↗](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
`,
    },
    zh: {
      title: "state_dict",
      md: `
**\`state_dict\` 是一个字典，装着模型的所有参数和 persistent buffer：key 是名字，value 是 tensor 本身。** 它就是模型的"存档"。这是 PyTorch 的概念：\`nn.Module\` 和 optimizer 都有 \`state_dict()\` 方法。

## 长什么样

用本网站的 GPT2 代码举例，\`model.state_dict()\` 大致是这样（形状按真实 GPT-2）：

\`\`\`text
"tok_emb.weight"                 → (50257, 768)
"pos_emb.weight"                 → (1024, 768)
"blocks.0.ln1.weight"            → (768,)              # γ
"blocks.0.ln1.bias"              → (768,)              # β
"blocks.0.attn.mask"             → (1, 1, 1024, 1024)  # buffer
"blocks.0.attn.c_attn.weight"    → (2304, 768)
"blocks.0.attn.c_attn.bias"      → (2304,)
...
"blocks.11.mlp. ..."
"ln_f.weight"                    → (768,)
"ln_f.bias"                      → (768,)
"lm_head.weight"                 → (50257, 768)
\`\`\`

key 就是代码里属性的路径：\`self.blocks[0].attn.c_attn.weight\` 在字典里叫 \`"blocks.0.attn.c_attn.weight"\`。

| 内容 | 在 \`state_dict\` 里吗 |
| --- | :---: |
| Parameter（W、b、γ、β、embedding 表） | 在 |
| persistent 的 [buffer](#info:register_buffer_notes)（比如 \`mask\`） | 在 |
| 普通属性 | 不在 |
| forward 中间算出来的 activations | 不在 |
| 没有 tensor 的模块（比如 \`nn.Dropout\`） | 没有条目 |

> 因为 [weight tying](#info:weight_tying_notes)，\`tok_emb.weight\` 和 \`lm_head.weight\` 两个 key 都会出现，但它们指向**同一个** tensor。

### mask 有必要存吗？

对 causal mask 来说，**其实没必要**：

- 它每次都由 \`__init__\` 里的 \`torch.tril(torch.ones(1024, 1024))\` 重新生成，结果永远一样。
- 加载时要先 \`model = GPT2(...)\`，这一步已经把 mask 造好了；存档里那份读进来，只是覆盖成一模一样的值。

而且它还占地方：

\`\`\`text
一个 mask：1024 × 1024 × 4 字节（float32）= 4 MB
12 个 block：12 × 4 MB = 48 MB
\`\`\`

一个约 500 MB 的文件里，将近 10% 存的是 12 个一模一样的三角矩阵。

**那为什么默认会存？** \`register_buffer\` 是通用工具，默认 \`persistent=True\` 是为**没法重新生成**的 buffer 准备的，比如 BatchNorm 的 \`running_mean\`，它是训练中统计出来的值。对 mask 来说，更合适的写法是：

\`\`\`python
self.register_buffer("mask", torch.tril(...), persistent=False)
\`\`\`

这样它仍然跟着模型搬到 GPU，但不会写进文件。本网站的代码用的是默认值，所以 mask 会被存；有些 GPT-2 实现在加载时也会主动跳过这些 mask 的 key。

## 保存和加载

\`state_dict\` **不是文件**，它只是内存里的一个 Python 字典。要用 \`torch.save\` 写到磁盘上，才会变成文件：

\`\`\`python
sd = model.state_dict()          # 第 1 步：内存里得到一个字典，还没有文件
torch.save(sd, "gpt2.pt")        # 第 2 步：把字典写进文件 gpt2.pt
\`\`\`

只做第 1 步的话，程序一关，字典就没了。\`torch.save\` 用 Python 的 pickle 把字典打包成一个 zip 格式的文件，文件名和后缀自己定（常见 \`.pt\`、\`.pth\`）。

读的时候反过来：

\`\`\`python
model = GPT2(...)                # 先按同样结构建一个模型（随机初始化）
sd = torch.load("gpt2.pt")       # 文件 → 内存里的字典
model.load_state_dict(sd)        # 字典 → 填进模型
\`\`\`

\`state_dict\` **只存数值，不存结构**。加载时要先用代码[建出同样结构的模型](#hover:state_dict_build_notes)，再**按名字**一一对应地填进去。名字或形状对不上，\`load_state_dict\` 就会报错。

> **类比。** 模型代码像一栋楼的**设计图**，\`state_dict\` 像每个房间里摆什么的**家具清单**。搬家时只带清单：到了新地方先按设计图盖出同样的楼，再照清单把家具摆回每个房间。

### 一个文件还是多个

- GPT-2 small 有 124M 参数，用 float32 存大约 500 MB，一个文件就够了。
- 🆕 几十 GB 的大模型通常把字典**切成好几块**，每块一个文件，再配一个索引文件，记录每个 key 在哪一块。Hugging Face 上常见的样子：

\`\`\`text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json      ← 每个 key 在哪个文件里
\`\`\`

### 🆕 为什么现在多用 \`.safetensors\`

\`.pt\` 文件是用 pickle 存的，而 pickle 加载时**可以执行任意代码**。所以从网上下载一个陌生的 \`.pt\` 文件再 \`torch.load\`，有安全风险（新版 PyTorch 默认 \`torch.load(..., weights_only=True)\` 来限制这一点）。

\`.safetensors\` 只存名字、形状和数值，藏不了代码，加载也更快。Hugging Face 现在默认用这种格式。

## 训练中的 checkpoint

训练时不会只在最后存一次，而是隔一段时间存一个 **checkpoint**：

\`\`\`python
for step in range(max_steps):
    ...  # 训练一步
    if step % 1000 == 0:
        torch.save({"model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step}, f"ckpt_{step}.pt")
\`\`\`

原因有两个：

- **防止中途崩溃。** 训练可能要跑好几天。机器出问题时，可以从最近的 checkpoint 接着训练，不用从头来。这时 **optimizer** 的状态也必须存：\`optimizer.state_dict()\` 存的是 optimizer 自己的状态，比如 Adam 为每个参数记的两组统计量。
- **最后一步不一定最好。** 训练时会同时看验证集上的 loss。中间某个 checkpoint 常常比最后一步更好（后面可能开始过拟合），最终发布的是效果最好的那一份。

训练结束后，拿去推理或发布时，通常**只保留模型的 \`state_dict\`**。Adam 的 optimizer 状态大约是参数的两倍大，推理用不上。

## PyTorch 以外

\`state_dict\` 这个名字是 PyTorch 的，但"把参数存成 名字 → 数组"这个思路是通用的：

- **Hugging Face：** \`save_pretrained()\` 写出的 \`.safetensors\` 文件，本质上就是一个 \`state_dict\`。
- **TensorFlow/Keras：** \`model.save_weights()\` 和 checkpoint。
- **JAX/Flax：** 参数本来就是一个嵌套字典（pytree），直接存这个字典。

文档：[Saving and loading models ↗](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
`,
    },
  },

  state_dict_build_notes: {
    en: {
      title: "Building the same structure (hover)",
      md: `
**How is “the same structure” built?** The structure lives in the model class: the \`nn.xxx(...)\` calls in \`__init__\`. In this site's code (simplified):

\`\`\`python
class GPT2(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, max_toks, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_toks, n_embd)
        self.blocks  = nn.ModuleList([TransformerBlock(...) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
\`\`\`

This fixes three things: which layers exist, the shape of each, and the name of each. The attribute name \`tok_emb\` is what later becomes the key \`"tok_emb.weight"\`.

"Building the same structure" just means running this code with the same hyperparameters, e.g. \`model = GPT2(vocab_size=50304, n_layer=12, n_embd=768, n_head=12, max_toks=1024)\`. Afterwards every layer exists with the right shape, but holds **random** numbers: the model runs, but its output is gibberish. \`load_state_dict\` then overwrites those numbers. Roughly:

\`\`\`text
for each key in sd, e.g. "blocks.0.attn.c_attn.weight":
  1. find the tensor with the same name in the model
  2. check that the shapes match, e.g. both (2304, 768)
  3. copy the numbers in
\`\`\`

If the model is built with different hyperparameters, loading fails. Say training used \`n_layer=12\` but the model is built with \`n_layer=6\`:

- **Unexpected key(s):** the file has \`blocks.6.…\` to \`blocks.11.…\`, which the model does not have.
- **Missing key(s):** the reverse case, a name the model has but the file does not.
- **size mismatch:** the names match but a shape does not, e.g. if \`n_embd\` were 512.

So loading needs the **same code** and the **same hyperparameters**. That is why a Hugging Face model folder also contains a \`config.json\` that records hyperparameters such as \`n_layer\` and \`n_embd\`. 🆕 \`from_pretrained()\` reads \`config.json\`, builds the model, then loads the weights, all in one call.
`,
    },
    zh: {
      title: "怎么建出同样结构的模型",
      md: `
**怎么建出同样结构的模型？** 结构写在模型的 class 里，也就是 \`__init__\` 里那一串 \`nn.xxx(...)\`。本网站的代码（简化后）：

\`\`\`python
class GPT2(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, max_toks, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_toks, n_embd)
        self.blocks  = nn.ModuleList([TransformerBlock(...) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
\`\`\`

这里定了三件事：有哪几层、每层的形状、每层叫什么名字。属性名 \`tok_emb\` 后来就成了 key \`"tok_emb.weight"\`。

"建出同样结构的模型"就是用同样的超参数运行这段代码，比如 \`model = GPT2(vocab_size=50304, n_layer=12, n_embd=768, n_head=12, max_toks=1024)\`。执行完，所有层都造好了，形状也都对，但里面是**随机**的数字：模型能跑，只是输出全是乱码。\`load_state_dict\` 再把这些数字覆盖掉，大致是：

\`\`\`text
for sd 里的每个 key，比如 "blocks.0.attn.c_attn.weight"：
  1. 在 model 里找到同名的 tensor
  2. 检查形状是否一样，比如都是 (2304, 768)
  3. 一样就把数字拷进去
\`\`\`

如果建模型时用的超参数不一样，加载就会失败。比如训练时 \`n_layer=12\`，加载前却写成 \`n_layer=6\`：

- **Unexpected key(s)：** 文件里有 \`blocks.6.…\` 到 \`blocks.11.…\`，模型里没有这些名字。
- **Missing key(s)：** 反过来，模型里有某个名字而文件里没有。
- **size mismatch：** 名字对上了但形状不同，比如 \`n_embd\` 写成了 512。

所以加载时必须用**同一份代码**和**同样的超参数**。这也是为什么 Hugging Face 的模型目录里除了权重文件，还有一个 \`config.json\`，记着 \`n_layer\`、\`n_embd\` 这些超参数。🆕 \`from_pretrained()\` 会先读 \`config.json\` 把模型建出来，再加载权重，一步完成。
`,
    },
  },
};
