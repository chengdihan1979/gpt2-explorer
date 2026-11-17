// Notes: MLP / feed-forward and GELU.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  mlp_fc_in_notes: {
    en: {
      title: "MLP expansion: nn.Linear(n_embd, 4 * n_embd)",
      md: `
**\`fc_in\` is the first layer of the MLP. It turns each token's 768-dimensional vector into 3072 numbers, and each of those numbers measures how well the token matches one learned pattern.**

## Where it sits

\`\`\`python
self.fc_in = nn.Linear(n_embd, 4 * n_embd)    # 768 → 3072
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

The MLP's input is \`ln2(x)\`, the residual stream after LayerNorm, with shape $(B,\\ T,\\ C)$ where $C$ = \`n_embd\` = 768. The full MLP is:

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   ← this layer: widen 4×
→ GELU       (B, T, 3072)   non-linearity, shape unchanged
→ fc_out     (B, T, 768)    back to 768
→ dropout    (B, T, 768)
\`\`\`

\`nn.Linear\` only acts on the **last** dimension. Every token goes through the same weights, but separately: the MLP never mixes information between tokens. (Mixing between tokens is attention's job.)

## Each output is a pattern detector

[\`nn.Linear\` stores its weight as (out, in)](#hover:linear_weight_notes), so \`fc_in.weight\` has shape (3072, 768): 3072 rows, each with 768 numbers. For one token, let $h$ be its 768-dimensional input vector, i.e. \`ln2(x)[b, t, :]\`. Output number $j$ is

\`\`\`text
out[j] = row_j · h + bias[j]      (dot product, j = 0 … 3071)
\`\`\`

A dot product measures **how similar** two vectors are: large and positive when they point the same way, near 0 when unrelated, negative when they point opposite ways. So each row is a **pattern**, and \`out[j]\` is "how well this token matches pattern j". The rows are learned during training.

**A small example** with $C = 4$, bias 0, and input \`h = [1, 0, 2, -1]\`:

\`\`\`text
row 0 = [ 1, 0, 1, 0]   →  1×1 + 0×0 + 1×2 + 0×(-1) =  3    strong match
row 1 = [ 0, 1, 0, 1]   →  0×1 + 1×0 + 0×2 + 1×(-1) = -1    weak mismatch
row 2 = [-1, 0, 0, 1]   → -1×1 + 0×0 + 0×2 + 1×(-1) = -2    opposite
\`\`\`

In GPT-2 there are 3072 such detectors, so every token gets 3072 match scores.

## Why wider than 768

- The 768 dimensions are the [bandwidth of the residual stream](#hover:proj_residual_stream_notes): the shared "whiteboard" that every layer reads from and writes to.
- The model wants to recognize far more than 768 patterns: "this is a plural noun", "the previous word is *the*", "we are inside a code block", "this is the second half of a person's name", and so on.
- Widening to 3072 gives each layer 3072 detectors to check at the same time. Afterwards [\`fc_out\`](#info:mlp_fc_out_notes) compresses the results back to 768 dimensions and writes them onto the whiteboard.

\`\`\`text
768  ──fc_in──→  3072 match scores  ──GELU──→  ──fc_out──→  768
read whiteboard   wide "workspace"                          write back
\`\`\`

> The widening only helps because of the [GELU](#info:gelu) in the middle. Without it, \`fc_out(fc_in(h))\` would be two linear maps in a row, which equal a single 768 × 768 linear map, and the 3072 detectors would add nothing.

## Why exactly 4×

There is no derivation; it is an empirical choice. The original Transformer paper used 512 → 2048, which is 4×. A wider layer gives the model more capacity but costs more parameters and compute; 4× was a good trade-off in their experiments, and GPT-2 kept it.

> 🆕 LLaMA and many later models use **SwiGLU** instead of GELU. SwiGLU has 3 weight matrices instead of 2, so the width is reduced to about 8/3 × to keep the parameter count similar. For example, LLaMA-7B uses 4096 → 11008 (8/3 × 4096 ≈ 10923, rounded up to a multiple of 256).

## Parameters

For this layer: weight $3072 \\times 768$ plus bias $3072$:

\`\`\`text
768 × 3072 + 3072 = 2,362,368
\`\`\`

Compared with attention, the MLP has about **twice** as many parameters per block:

| Part | Layers | Parameters |
| --- | --- | --- |
| MLP | \`fc_in\`: 768 × 3072 + 3072 | 2,362,368 |
| | \`fc_out\`: 3072 × 768 + 768 | 2,360,064 |
| | **total** | **4,722,432** (≈ 4.72M) |
| Attention | \`c_attn\`: 768 × 2304 + 2304 | 1,771,776 |
| | \`proj\`: 768 × 768 + 768 | 590,592 |
| | **total** | **2,362,368** (≈ 2.36M) |

## Variants

- 🆕 Gated variants (GEGLU, SwiGLU) replace "Linear → GELU" with two parallel Linears whose outputs are multiplied; the idea is still widen → non-linearity → project back.
- GPT-2 runs attention first and the MLP second, each in its own residual branch. 🆕 Some later models (e.g. GPT-J, PaLM) run the attention and MLP branches in parallel on the same input.
`,
    },
    zh: {
      title: "MLP 扩展：nn.Linear(n_embd, 4 * n_embd)",
      md: `
**\`fc_in\` 是 MLP 的第一层。它把每个 token 的 768 维向量变成 3072 个数，每个数衡量这个 token 和某一个学到的模式有多匹配。**

## 位置

\`\`\`python
self.fc_in = nn.Linear(n_embd, 4 * n_embd)    # 768 → 3072
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

MLP 的输入是 \`ln2(x)\`，也就是经过 LayerNorm 的残差流，形状是 $(B,\\ T,\\ C)$，其中 $C$ = \`n_embd\` = 768。整个 MLP：

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   ← 这一层：放大 4 倍
→ GELU       (B, T, 3072)   非线性，形状不变
→ fc_out     (B, T, 768)    缩回 768
→ dropout    (B, T, 768)
\`\`\`

\`nn.Linear\` 只作用在**最后一维**上。每个 token 用同一套权重，但各算各的：MLP 从不在 token 之间交换信息（那是 attention 的工作）。

## 每个输出是一个模式检测器

[\`nn.Linear\` 的 weight 按 (out, in) 存](#hover:linear_weight_notes)，所以 \`fc_in.weight\` 的形状是 (3072, 768)：3072 行，每行 768 个数。对一个 token，记它的 768 维输入为 $h$，即 \`ln2(x)[b, t, :]\`。第 $j$ 个输出是：

\`\`\`text
out[j] = row_j · h + bias[j]      (点积，j = 0 … 3071)
\`\`\`

点积衡量两个向量**有多像**：方向一致时又大又正，不相关时接近 0，方向相反时为负。所以每一行是一个**模式（pattern）**，\`out[j]\` 就是"这个 token 有多符合模式 j"。这些行是训练中学出来的。

**小例子**：$C = 4$，bias 为 0，输入 \`h = [1, 0, 2, -1]\`：

\`\`\`text
row 0 = [ 1, 0, 1, 0]   →  1×1 + 0×0 + 1×2 + 0×(-1) =  3    很符合
row 1 = [ 0, 1, 0, 1]   →  0×1 + 1×0 + 0×2 + 1×(-1) = -1    不太符合
row 2 = [-1, 0, 0, 1]   → -1×1 + 0×0 + 0×2 + 1×(-1) = -2    相反
\`\`\`

GPT-2 里有 3072 个这样的检测器，所以每个 token 会得到 3072 个匹配分数。

## 为什么要比 768 宽

- 768 维是[残差流的"带宽"](#hover:proj_residual_stream_notes)，也就是所有层共用、都在上面读写的那块"白板"。
- 模型想识别的模式远不止 768 种，比如"这是复数名词"、"前一个词是 *the*"、"现在在一段代码里"、"这是某个人名的后半部分"等等。
- 放大到 3072，每一层就有 3072 个检测器可以同时检查。检查完，[\`fc_out\`](#info:mlp_fc_out_notes) 再把结果压回 768 维，写回白板。

\`\`\`text
768  ──fc_in──→  3072 match scores  ──GELU──→  ──fc_out──→  768
read whiteboard   wide "workspace"                          write back
\`\`\`

> 放大之所以有用，是因为中间有 [GELU](#info:gelu)。如果没有它，\`fc_out(fc_in(h))\` 就是连续两个线性变换，等于一个 768 × 768 的线性变换，那 3072 个检测器就白加了。

## 为什么正好是 4 倍

没有理论推导，是经验值。原始 Transformer 论文用的是 512 → 2048，即 4 倍。宽度越大，模型容量越大，但参数和计算量也越大；4 倍是他们实验中较好的折中，GPT-2 沿用了它。

> 🆕 LLaMA 和很多后来的模型用 **SwiGLU** 代替 GELU。SwiGLU 有 3 个权重矩阵而不是 2 个，所以把宽度缩到约 8/3 倍，让参数量差不多。例如 LLaMA-7B 是 4096 → 11008（8/3 × 4096 ≈ 10923，向上取到 256 的倍数）。

## 参数量

这一层：weight $3072 \\times 768$，加上 bias $3072$：

\`\`\`text
768 × 3072 + 3072 = 2,362,368
\`\`\`

和 attention 比，每个 block 里 MLP 的参数大约是它的 **2 倍**：

| 部分 | 层 | 参数量 |
| --- | --- | --- |
| MLP | \`fc_in\`: 768 × 3072 + 3072 | 2,362,368 |
| | \`fc_out\`: 3072 × 768 + 768 | 2,360,064 |
| | **合计** | **4,722,432**（≈ 4.72M） |
| Attention | \`c_attn\`: 768 × 2304 + 2304 | 1,771,776 |
| | \`proj\`: 768 × 768 + 768 | 590,592 |
| | **合计** | **2,362,368**（≈ 2.36M） |

## 变体

- 🆕 门控变体（GEGLU、SwiGLU）把 "Linear → GELU" 换成两个并列的 Linear，再把输出相乘；思路仍然是放大 → 非线性 → 投影回去。
- GPT-2 先做 attention、再做 MLP，各自在自己的残差分支里。🆕 一些后来的模型（例如 GPT-J、PaLM）让 attention 和 MLP 两个分支并行，吃同一个输入。
`,
    },
  },

  mlp_fc_out_notes: {
    en: {
      title: "MLP projection back: nn.Linear(4 * n_embd, n_embd)",
      md: `
**\`fc_out\` is the last layer of the MLP. It turns the 3072 numbers from GELU back into a 768-dimensional vector, the update that is added to the residual stream.**

## Where it sits

\`\`\`python
self.fc_out = nn.Linear(4 * n_embd, n_embd)    # 3072 → 768
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   widen: 3072 pattern scores
→ GELU       (B, T, 3072)   decide which patterns "fire"
→ fc_out     (B, T, 768)    ← this layer: back to 768
→ dropout    (B, T, 768)
\`\`\`

The result must be 768-dimensional because the block adds it to the residual stream, which is 768-dimensional: \`x = x + mlp(ln2(x))\`. Adding vectors only works when their shapes match.

## How nn.Linear computes

\`nn.Linear(in_features, out_features)\` computes

\`\`\`text
y = x @ weight.T + bias
\`\`\`

where \`weight\` is **stored** with shape (out, in). For \`fc_out = nn.Linear(3072, 768)\`, \`weight\` is (768, 3072), and the multiplication uses its transpose (3072, 768):

\`\`\`text
(B, T, 3072) @ (3072, 768) → (B, T, 768)
\`\`\`

More on \`nn.Linear\`: [why (out, in), why \`x @ W.T\` instead of \`W x\`, and what actually runs](#hover:linear_weight_notes).

## A worked example: 2 → 4 → 2

The real sizes 768 → 3072 → 768 cannot be computed by hand, so shrink them to **2 → 4 → 2**. It widens 2× instead of 4×, but the process is the same: first wider, then narrower. All biases are 0.

### Step 1: fc_in, 2 → 4

Input \`h = [1, 2]\`. \`fc_in.weight\` has shape (out, in) = (4, 2). Each of its 4 rows is a pattern detector, and each output is a dot product:

\`\`\`text
row 0 = [ 1,  1]  →  1×1 + 1×2    =  3
row 1 = [ 1, -1]  →  1×1 + (-1)×2 = -1
row 2 = [-1,  0]  → -1×1 + 0×2    = -1
row 3 = [ 0,  1]  →  0×1 + 1×2    =  2

scores = [3, -1, -1, 2]       ← 2 numbers became 4
\`\`\`

### Step 2: GELU, one number at a time

[GELU](#info:gelu) acts on each number separately, so the shape does not change. Roughly, it keeps positive numbers and pushes negative ones close to 0:

\`\`\`text
GELU(3)  ≈  3.00    pattern 0 fires
GELU(-1) ≈ -0.16    almost off
GELU(-1) ≈ -0.16    almost off
GELU(2)  ≈  1.95    pattern 3 fires

a = [3.00, -0.16, -0.16, 1.95]
\`\`\`

### Step 3: fc_out, 4 → 2

\`fc_out = nn.Linear(4, 2)\`. Its stored \`weight\` is (out, in) = (2, 4); the multiplication uses \`weight.T\`, which is (4, 2):

\`\`\`text
weight (2×4):              weight.T (4×2):
[ 1, 0, 1,  0 ]            [ 1,  0 ]   ← row 0
[ 0, 1, 1, -1 ]            [ 0,  1 ]   ← row 1
                           [ 1,  1 ]   ← row 2
                           [ 0, -1 ]   ← row 3
\`\`\`

\`a\` has shape [(4,)](#hover:vector_1d_notes), so \`a @ weight.T\` is (4,) @ (4, 2) → (2,). Multiplying a vector by a matrix from the left means: **use the vector's numbers as weights, and add up the matrix's rows**:

\`\`\`text
 3.00 × [1,  0] = [ 3.00,  0   ]
-0.16 × [0,  1] = [ 0,    -0.16]
-0.16 × [1,  1] = [-0.16, -0.16]
 1.95 × [0, -1] = [ 0,    -1.95]
                  ──────────────
out             = [ 2.84, -2.27]    ← 4 numbers back to 2
\`\`\`

Computing one output number at a time gives the same result. Each output number is a row of the stored \`weight\` dotted with \`a\`:

\`\`\`text
out[0] = [1, 0, 1,  0] · a = 3.00 − 0.16         =  2.84
out[1] = [0, 1, 1, -1] · a = −0.16 − 0.16 − 1.95 = −2.27
\`\`\`

Checked in PyTorch:

\`\`\`python
fc_out = nn.Linear(4, 2, bias=False)
fc_out.weight.shape        # torch.Size([2, 4])
# after setting weight = [[1, 0, 1, 0], [0, 1, 1, -1]]:
a = torch.tensor([3.00, -0.16, -0.16, 1.95])
fc_out(a)                  # tensor([ 2.8400, -2.2700])
a @ fc_out.weight.T        # tensor([ 2.8400, -2.2700])  same
\`\`\`

### The whole example

\`\`\`text
h = [1, 2]
  → fc_in  → [3, -1, -1, 2]               2 → 4: 4 detectors score the input
  → GELU   → [3.00, -0.16, -0.16, 1.95]   patterns 0 and 3 fire; 1 and 2 are nearly off
  → fc_out → [2.84, -2.27]                4 → 2: the firing patterns write their rows
\`\`\`

The output comes mostly from rows 0 and 3 of \`weight.T\`, because only patterns 0 and 3 fired.

## The MLP as "if … then write …" rules

The example suggests a way to read the whole MLP. Rule $j$ consists of:

- **row $j$ of \`fc_in.weight\`**: the pattern to check ("if the token looks like this …");
- **GELU of the score**: how strongly the rule fires (near 0 means it did not fire);
- **row $j$ of \`fc_out.weight.T\`** (the same numbers as column $j$ of the stored \`fc_out.weight\`): the 768-dimensional vector to write ("… then write this").

With the real sizes:

\`\`\`text
a: the 3072 numbers after GELU, a[j] = how much rule j fired
fc_out.weight.T: (3072, 768), row j = what rule j writes

out = a[0] × row 0 + a[1] × row 1 + … + a[3071] × row 3071 + bias
    = one 768-dimensional vector
\`\`\`

After GELU most \`a[j]\` are close to 0, so in practice only the few rules that fired write into the output.

So 3072 is the **number of rules per block**, and 768 is the **width of what gets written**. These are separate: many rules can write into the same directions, which is why the MLP can have far more rules than the residual stream has dimensions. Each of the 12 blocks has its own MLP with its own 3072 rules.

> 🆕 This view comes from the 2021 paper *Transformer Feed-Forward Layers Are Key-Value Memories* (Geva et al.): the rows of \`fc_in\` are keys and the rows of \`fc_out.weight.T\` are values. Later work such as ROME edits facts a model has memorized by changing MLP weights.

## Parameters

Weight 768 × 3072 plus bias 768:

\`\`\`text
3072 × 768 + 768 = 2,360,064
\`\`\`

Together with [\`fc_in\`](#info:mlp_fc_in_notes) (2,362,368), the MLP has 4,722,432 parameters per block.
`,
    },
    zh: {
      title: "MLP 投影回去：nn.Linear(4 * n_embd, n_embd)",
      md: `
**\`fc_out\` 是 MLP 的最后一层。它把 GELU 输出的 3072 个数变回一个 768 维向量，也就是要加回残差流的那个更新。**

## 位置

\`\`\`python
self.fc_out = nn.Linear(4 * n_embd, n_embd)    # 3072 → 768
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   放大：3072 个模式分数
→ GELU       (B, T, 3072)   决定哪些模式"触发"
→ fc_out     (B, T, 768)    ← 这一层：缩回 768
→ dropout    (B, T, 768)
\`\`\`

结果必须是 768 维，因为 block 要把它加到 768 维的残差流上：\`x = x + mlp(ln2(x))\`。两个向量形状一致才能相加。

## nn.Linear 怎么算

\`nn.Linear(in_features, out_features)\` 计算的是：

\`\`\`text
y = x @ weight.T + bias
\`\`\`

其中 \`weight\` **存储**的形状是 (out, in)。\`fc_out = nn.Linear(3072, 768)\` 的 \`weight\` 是 (768, 3072)，乘法时用它的转置 (3072, 768)：

\`\`\`text
(B, T, 3072) @ (3072, 768) → (B, T, 768)
\`\`\`

更多细节：[为什么存成 (out, in)、为什么写 \`x @ W.T\` 而不是 \`W x\`，以及实际是怎么算的](#hover:linear_weight_notes)

## 例子：2 → 4 → 2

真实尺寸 768 → 3072 → 768 没法手算，所以缩小成 **2 → 4 → 2**。它放大的是 2 倍而不是 4 倍，但过程一样：先变宽，再变窄。所有 bias 都设为 0。

### 第 1 步：fc_in，2 → 4

输入 \`h = [1, 2]\`。\`fc_in.weight\` 形状 (out, in) = (4, 2)。它的 4 行各是一个模式检测器，每个输出是一个点积：

\`\`\`text
row 0 = [ 1,  1]  →  1×1 + 1×2    =  3
row 1 = [ 1, -1]  →  1×1 + (-1)×2 = -1
row 2 = [-1,  0]  → -1×1 + 0×2    = -1
row 3 = [ 0,  1]  →  0×1 + 1×2    =  2

scores = [3, -1, -1, 2]       ← 2 个数变成了 4 个
\`\`\`

### 第 2 步：GELU，逐个数作用

[GELU](#info:gelu) 对每个数单独作用，所以形状不变。大致效果是：正数基本保留，负数被压到接近 0：

\`\`\`text
GELU(3)  ≈  3.00    模式 0 触发
GELU(-1) ≈ -0.16    几乎没触发
GELU(-1) ≈ -0.16    几乎没触发
GELU(2)  ≈  1.95    模式 3 触发

a = [3.00, -0.16, -0.16, 1.95]
\`\`\`

### 第 3 步：fc_out，4 → 2

\`fc_out = nn.Linear(4, 2)\`。它存储的 \`weight\` 是 (out, in) = (2, 4)；乘法时用 \`weight.T\`，形状是 (4, 2)：

\`\`\`text
weight (2×4):              weight.T (4×2):
[ 1, 0, 1,  0 ]            [ 1,  0 ]   ← row 0
[ 0, 1, 1, -1 ]            [ 0,  1 ]   ← row 1
                           [ 1,  1 ]   ← row 2
                           [ 0, -1 ]   ← row 3
\`\`\`

\`a\` 的形状是 [(4,)](#hover:vector_1d_notes)，所以 \`a @ weight.T\` 是 (4,) @ (4, 2) → (2,)。向量从左边乘矩阵的意思是：**用向量里的数当权重，把矩阵的各行加起来**：

\`\`\`text
 3.00 × [1,  0] = [ 3.00,  0   ]
-0.16 × [0,  1] = [ 0,    -0.16]
-0.16 × [1,  1] = [-0.16, -0.16]
 1.95 × [0, -1] = [ 0,    -1.95]
                  ──────────────
out             = [ 2.84, -2.27]    ← 4 个数变回 2 个
\`\`\`

一个输出一个输出地算，结果一样。每个输出数是存储的 \`weight\` 的一行和 \`a\` 的点积：

\`\`\`text
out[0] = [1, 0, 1,  0] · a = 3.00 − 0.16         =  2.84
out[1] = [0, 1, 1, -1] · a = −0.16 − 0.16 − 1.95 = −2.27
\`\`\`

在 PyTorch 里验证过：

\`\`\`python
fc_out = nn.Linear(4, 2, bias=False)
fc_out.weight.shape        # torch.Size([2, 4])
# 把 weight 设成 [[1, 0, 1, 0], [0, 1, 1, -1]] 之后：
a = torch.tensor([3.00, -0.16, -0.16, 1.95])
fc_out(a)                  # tensor([ 2.8400, -2.2700])
a @ fc_out.weight.T        # tensor([ 2.8400, -2.2700])  一样
\`\`\`

### 整个过程

\`\`\`text
h = [1, 2]
  → fc_in  → [3, -1, -1, 2]               2 → 4：4 个检测器打分
  → GELU   → [3.00, -0.16, -0.16, 1.95]   模式 0 和 3 触发，1 和 2 几乎关掉
  → fc_out → [2.84, -2.27]                4 → 2：触发的模式把各自的行写进输出
\`\`\`

输出主要来自 \`weight.T\` 的第 0 行和第 3 行，因为只有模式 0 和 3 触发了。

## 把 MLP 看成"如果……就写……"的规则

这个例子提供了理解整个 MLP 的一种方式。第 $j$ 条规则由三部分组成：

- **\`fc_in.weight\` 的第 $j$ 行**：要检查的模式（"如果 token 像这样……"）；
- **分数经过 GELU**：这条规则触发了多少（接近 0 就是没触发）；
- **\`fc_out.weight.T\` 的第 $j$ 行**（和存储的 \`fc_out.weight\` 第 $j$ 列是同一组数）：要写的 768 维向量（"……就写这个"）。

换成真实尺寸：

\`\`\`text
a: the 3072 numbers after GELU, a[j] = how much rule j fired
fc_out.weight.T: (3072, 768), row j = what rule j writes

out = a[0] × row 0 + a[1] × row 1 + … + a[3071] × row 3071 + bias
    = one 768-dimensional vector
\`\`\`

GELU 之后大多数 \`a[j]\` 接近 0，所以实际上只有少数触发了的规则在往输出里写东西。

所以 3072 是**每个 block 里规则的条数**，768 是**写出来的东西的宽度**。这两者是独立的：很多规则可以往同一些方向写，所以规则数可以远多于残差流的维数。12 个 block 各有自己的 MLP，各有自己的 3072 条规则。

> 🆕 这种看法来自 2021 年的论文 *Transformer Feed-Forward Layers Are Key-Value Memories*（Geva et al.）：\`fc_in\` 的行是 key，\`fc_out.weight.T\` 的行是 value。后来 ROME 等工作就是通过修改 MLP 权重来改模型记住的事实。

## 参数量

weight 768 × 3072，加上 bias 768：

\`\`\`text
3072 × 768 + 768 = 2,360,064
\`\`\`

加上 [\`fc_in\`](#info:mlp_fc_in_notes) 的 2,362,368，每个 block 的 MLP 一共有 4,722,432 个参数。
`,
    },
  },

  gelu: {
    en: {
      title: "GELU",
      md: `
**GELU (Gaussian Error Linear Unit) is the non-linearity between \`fc_in\` and \`fc_out\` in the MLP. It decides how much of each of the 3072 pattern scores is let through.**

\`\`\`python
self.act = nn.GELU(approximate="tanh")
\`\`\`

GELU acts on **each number separately**, so the shape does not change: $(B,\\ T,\\ 3072) \\to (B,\\ T,\\ 3072)$.

## Why the MLP needs a non-linearity

Without GELU, \`fc_out(fc_in(h))\` would be two linear maps in a row, and two linear maps combine into one (biases left out for simplicity):

\`\`\`text
fc_out(fc_in(h)) = W2 (W1 h) = (W2 W1) h      ← W2 W1 is just one 768 × 768 matrix
\`\`\`

Then the 3072 detectors would be wasted: a single 768 × 768 matrix could do exactly the same thing. GELU lets each detector "fire" or "not fire", which is what lets the MLP express [if … then write … rules](#info:mlp_fc_out_notes).

## The formula

\`\`\`text
GELU(x) = x × Φ(x)
\`\`\`

$\\Phi(x)$ is the [CDF of the standard normal distribution](#hover:cdf): if you draw a random number from a normal distribution with mean 0 and standard deviation 1, $\\Phi(x)$ is the probability that it is smaller than $x$.

- When $x$ is large, $\\Phi(x) \\approx 1$, so GELU(x) ≈ x: the number passes almost unchanged.
- When $x$ is very negative, $\\Phi(x) \\approx 0$, so GELU(x) ≈ 0: the number is almost switched off.
- Near 0, $\\Phi(x)$ is between 0 and 1, so the number passes partly.

Think of $\\Phi(x)$ as a "pass-through fraction": the larger $x$, the more of it gets through.

> GELU comes from the paper [Hendrycks & Gimpel (2016) ↗](https://arxiv.org/abs/1606.08415).

## Concrete values

Computed in PyTorch:

| x | −3 | −2 | −1 | −0.75 | −0.5 | 0 | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Φ(x) | 0.001 | 0.023 | 0.159 | 0.227 | 0.309 | 0.5 | 0.691 | 0.841 | 0.977 | 0.999 |
| GELU(x) | −0.004 | −0.046 | −0.159 | −0.170 | −0.154 | 0 | 0.346 | 0.841 | 1.954 | 2.996 |

Each GELU value is $x \\times \\Phi(x)$, e.g. GELU(2) = 2 × 0.977 = 1.954.

![GELU curve from −3.5 to 3.5 with the line y = x for comparison](/llm-explorer/notes/gelu.svg)

- **Positive inputs** pass almost unchanged: GELU(2) = 1.95.
- **Negative inputs** are pushed close to 0, but not exactly 0. The lowest point is at $x \\approx -0.75$, where GELU ≈ −0.17.
- The curve is **smooth**, with no corner.

This is where the values in the [MLP worked example](#info:mlp_fc_out_notes) come from: GELU(3) ≈ 3.00, GELU(−1) ≈ −0.16, GELU(2) ≈ 1.95.

## Tanh approximation (used in GPT-2)

Computing $\\Phi(x)$ exactly needs the erf function. GPT-2 used a faster approximation built from tanh:

\`\`\`text
GELU(x) ≈ 0.5 · x · (1 + tanh( √(2/π) · (x + 0.044715 · x³) ))
\`\`\`

The difference is tiny: at the points in the table above it is at most about 0.0004. The site's code uses \`approximate="tanh"\` so that its outputs match the original GPT-2 weights. The \`try/except\` around it is there because older PyTorch versions' \`nn.GELU\` has no \`approximate\` argument and only offers the exact version.

Why approximate at all? Historically, evaluating erf was slower than a few multiplications plus a tanh, especially on older GPU kernels and before [common fusions ↗](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/). See also this [GitHub issue ↗](https://github.com/pytorch/pytorch/issues/39853).

### Today

- **PyTorch:** \`torch.nn.GELU()\` / \`F.gelu\` support \`approximate='none'\` (exact) or \`approximate='tanh'\`. The exact version is generally fast enough; tanh can be slightly faster on some hardware. Some transformer libraries have a "FastGELU", which is just the tanh form.
- **TensorFlow:** \`tf.nn.gelu(x, approximate=False)\` computes the exact erf form; \`approximate=True\` uses the tanh form.

## Why GELU instead of ReLU?

### What ReLU is

\`\`\`text
ReLU(x) = max(0, x)      positive inputs pass unchanged, negative inputs become 0
\`\`\`

Compared with GELU (computed in PyTorch):

| x | −3 | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU(x) | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 | 2 |
| GELU(x) | −0.004 | −0.046 | −0.159 | −0.154 | 0 | 0.346 | 0.841 | 1.954 |

Overall they are similar: positive inputs mostly pass, negative inputs are mostly switched off. The difference is near 0: ReLU has a sharp **corner** at 0, GELU a **smooth** transition.

### The key difference: the gradient

During training, how a weight changes depends on the gradient: how much the output changes when the input changes a little. The gradients (derivatives) of the two:

| x | −5 | −3 | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU gradient | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| GELU gradient | −0.000 | −0.012 | −0.085 | −0.083 | 0.133 | 0.5 | 0.867 | 1.083 | 1.085 |

- **ReLU:** the gradient is **exactly 0** for every negative input, and jumps from 0 to 1 at 0. (PyTorch uses 0 at exactly x = 0.)
- **GELU:** the gradient changes smoothly. Between −3 and 0 it is still nonzero; only far out, around −5, does it approach 0.

### "Dying ReLU": what a zero gradient does

Detector $j$ in the MLP computes \`h[j] = row_j · x + bias[j]\` (see [MLP expansion](#info:mlp_fc_in_notes)). Suppose one training step updates too aggressively and pushes \`bias[j]\` to −5, while \`row_j · x\` normally stays between −2 and 2:

\`\`\`text
h[j] is always between -7 and -3     ← negative for every token
ReLU(h[j]) = 0                       ← the output is always 0
ReLU gradient = 0                    ← the gradient sent back is always 0
→ row_j and bias[j] never receive a gradient, so they never change again
\`\`\`

The detector is "dead": one of the 3072 is wasted, permanently.

With GELU, at h[j] = −3 the gradient is −0.012. Small, but not 0, so the weights can still be pulled back slowly.

To see these gradients at work, here is the whole backward pass through a tiny MLP, step by step: [worked example: backward through GELU vs. ReLU](#hover:gelu_backprop_notes).

### Two more properties of GELU

1. **Smooth:** no corner, so the gradient changes continuously and the optimizer does not hit a sudden jump near 0. This makes training a bit more stable.
2. **A little output for negative inputs:** down to −0.17. Small negative inputs are not completely erased, so a bit of "how negative" still reaches the next layer.

### How big is the difference in practice?

**Small.** The GELU paper and the BERT and GPT experiments found it usually a little better than ReLU (slightly lower loss), so BERT and the GPT series chose it. It is not that ReLU does not work; GELU is just somewhat better.

> 🆕 Most later models use **gated** activations instead, such as LLaMA's SwiGLU: two Linears, one computing a value and one computing a "gate", multiplied together. As mentioned in [MLP expansion](#info:mlp_fc_in_notes), this is why LLaMA's MLP is about 8/3 × wide instead of 4 ×. Some models use ReLU² (ReLU squared, from the Primer paper).
`,
    },
    zh: {
      title: "GELU",
      md: `
**GELU（Gaussian Error Linear Unit）是 MLP 里夹在 \`fc_in\` 和 \`fc_out\` 之间的非线性函数。它决定 3072 个模式分数中，每一个能"通过"多少。**

\`\`\`python
self.act = nn.GELU(approximate="tanh")
\`\`\`

GELU 对**每个数单独**作用，所以形状不变：$(B,\\ T,\\ 3072) \\to (B,\\ T,\\ 3072)$。

## 为什么 MLP 需要一个非线性函数

如果没有 GELU，\`fc_out(fc_in(h))\` 就是连续两次线性变换，而两次线性变换合起来仍是一次（为了简单，这里省略 bias）：

\`\`\`text
fc_out(fc_in(h)) = W2 (W1 h) = (W2 W1) h      ← W2 W1 只是一个 768 × 768 的矩阵
\`\`\`

那样 3072 个检测器就白放了：一个 768 × 768 的矩阵就能做到完全一样的事。GELU 让每个检测器能"触发"或"不触发"，MLP 才能表达["如果……就写……"的规则](#info:mlp_fc_out_notes)。

## 公式

\`\`\`text
GELU(x) = x × Φ(x)
\`\`\`

$\\Phi(x)$ 是[标准正态分布的 CDF（累积分布函数）](#hover:cdf)：从均值 0、标准差 1 的正态分布里随机抽一个数，它小于 $x$ 的概率就是 $\\Phi(x)$。

- $x$ 很大时，$\\Phi(x) \\approx 1$，所以 GELU(x) ≈ x：基本原样通过。
- $x$ 很负时，$\\Phi(x) \\approx 0$，所以 GELU(x) ≈ 0：基本被关掉。
- $x$ 在 0 附近时，$\\Phi(x)$ 在 0 和 1 之间：打个折再通过。

可以把 $\\Phi(x)$ 理解成一个"通过比例"：$x$ 越大，通过得越多。

> GELU 出自论文 [Hendrycks & Gimpel (2016) ↗](https://arxiv.org/abs/1606.08415)。

## 具体数值

用 PyTorch 算的：

| x | −3 | −2 | −1 | −0.75 | −0.5 | 0 | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Φ(x) | 0.001 | 0.023 | 0.159 | 0.227 | 0.309 | 0.5 | 0.691 | 0.841 | 0.977 | 0.999 |
| GELU(x) | −0.004 | −0.046 | −0.159 | −0.170 | −0.154 | 0 | 0.346 | 0.841 | 1.954 | 2.996 |

每个 GELU 值都是 $x \\times \\Phi(x)$，例如 GELU(2) = 2 × 0.977 = 1.954。

![GELU 曲线（−3.5 到 3.5），虚线是 y = x](/llm-explorer/notes/gelu.svg)

- **正数**基本原样通过：GELU(2) = 1.95。
- **负数**被压到接近 0，但不是正好 0。最低点在 $x \\approx -0.75$，值约 −0.17。
- 曲线是**平滑**的，没有拐角。

[MLP 例子](#info:mlp_fc_out_notes)里的数就是这么来的：GELU(3) ≈ 3.00，GELU(−1) ≈ −0.16，GELU(2) ≈ 1.95。

## tanh 近似（GPT-2 用的）

精确计算 $\\Phi(x)$ 需要 erf 函数。GPT-2 用了一个基于 tanh 的近似公式，算起来更快：

\`\`\`text
GELU(x) ≈ 0.5 · x · (1 + tanh( √(2/π) · (x + 0.044715 · x³) ))
\`\`\`

两者差别极小：在上表这些点上最多约 0.0004。本网站代码用 \`approximate="tanh"\`，是为了和原版 GPT-2 的权重完全对上。外面那层 \`try/except\`，是因为旧版 PyTorch 的 \`nn.GELU\` 没有 \`approximate\` 参数，只能用精确版。

为什么要近似？以前算 erf 比几次乘法加一个 tanh 慢，在老的 GPU kernel 上、以及[常见的算子融合 ↗](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)普及之前尤其明显。另见这个 [GitHub issue ↗](https://github.com/pytorch/pytorch/issues/39853)。

### 现在

- **PyTorch：** \`torch.nn.GELU()\` / \`F.gelu\` 支持 \`approximate='none'\`（精确）和 \`approximate='tanh'\`。精确版一般已经够快；tanh 版在某些硬件上略快一点。有些 transformer 库里的 "FastGELU" 就是 tanh 版。
- **TensorFlow：** \`tf.nn.gelu(x, approximate=False)\` 是精确的 erf 版；\`approximate=True\` 是 tanh 版。

## 为什么用 GELU 而不是 ReLU？

### ReLU 是什么

\`\`\`text
ReLU(x) = max(0, x)      正数原样通过，负数一律变成 0
\`\`\`

和 GELU 对比（PyTorch 算的）：

| x | −3 | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU(x) | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 | 2 |
| GELU(x) | −0.004 | −0.046 | −0.159 | −0.154 | 0 | 0.346 | 0.841 | 1.954 |

两者大体上很像：正数基本通过，负数基本关掉。区别在 0 附近：ReLU 在 0 处有一个**尖角**，GELU 是**平滑**过渡。

### 关键区别：梯度

训练时，权重怎么改取决于梯度，也就是"输入变一点，输出变多少"。两者的梯度（导数）：

| x | −5 | −3 | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU 梯度 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| GELU 梯度 | −0.000 | −0.012 | −0.085 | −0.083 | 0.133 | 0.5 | 0.867 | 1.083 | 1.085 |

- **ReLU**：所有负数输入的梯度都**正好是 0**，在 0 处从 0 突然跳到 1。（x 正好等于 0 时，PyTorch 取 0。）
- **GELU**：梯度平滑变化。在 −3 到 0 之间仍然不是 0，只有到 −5 这样很远的地方才接近 0。

### "dying ReLU"：梯度为 0 的后果

MLP 的第 $j$ 个检测器算的是 \`h[j] = row_j · x + bias[j]\`（见 [MLP 扩展](#info:mlp_fc_in_notes)）。假设训练中某一步更新得太猛，把 \`bias[j]\` 推到了 −5，而 \`row_j · x\` 平时只在 −2 到 2 之间：

\`\`\`text
h[j] is always between -7 and -3     ← 对所有 token 都是负数
ReLU(h[j]) = 0                       ← 输出永远是 0
ReLU gradient = 0                    ← 传回去的梯度永远是 0
→ row_j 和 bias[j] 永远收不到梯度，再也不会更新
\`\`\`

这个检测器就"死"了：3072 个里白白少了一个，而且救不回来。

换成 GELU，h[j] = −3 时梯度是 −0.012，虽然小，但不是 0，权重还能慢慢被拉回来。

想看这些梯度在训练中具体怎么起作用，这里用一个很小的 MLP 把整个 backward 一步一步算了一遍：[具体例子：穿过 GELU 和 ReLU 的 backward](#hover:gelu_backprop_notes)。

### GELU 的另外两个特点

1. **平滑**：没有尖角，梯度连续变化，优化器在 0 附近不会遇到突变，训练更稳定一些。
2. **负数区有一点输出**：最低到 −0.17。小的负输入不会被完全抹掉，"有多负"的信息还能传一点给后面的层。

### 实际差别有多大

**不大。** GELU 论文和 BERT、GPT 的实验显示，它通常比 ReLU 好一点点（loss 略低），所以 BERT 和 GPT 系列都选了它。不是 ReLU 不能用，只是 GELU 稍好一些。

> 🆕 后来的模型大多换成了**门控**激活函数，比如 LLaMA 的 SwiGLU：两个 Linear，一个算值，一个算"门"，两者相乘。[MLP 扩展](#info:mlp_fc_in_notes)里提过，这就是 LLaMA 的 MLP 宽度约为 8/3 倍而不是 4 倍的原因。也有模型用 ReLU²（ReLU 的平方，来自 Primer 论文）。
`,
    },
  },

  cdf: {
    en: {
      title: "CDF",
      md: `
**The standard normal CDF, step by step**

### 1. The standard normal distribution

\`torch.randn\` draws random numbers from the standard normal distribution. Ten draws:

\`\`\`text
-1.13, -1.15, -0.25, -0.43, 0.85, 0.69, -0.32, -2.12, 0.32, -1.26
\`\`\`

- **Mean 0:** the numbers are centered on 0, with as many positive as negative ones.
- **Standard deviation 1:** most numbers fall between −1 and 1, few go beyond ±2, almost none beyond ±3:

| Range | Share of numbers |
| --- | --- |
| between −1 and 1 | 68.3% |
| between ±1 and ±2 (both sides together) | 27.2% |
| between ±2 and ±3 (both sides together) | 4.3% |
| beyond ±3 | 0.3% |
| *cumulative:* between −2 and 2 | 95.4% (= 68.3% + 27.2%) |
| *cumulative:* between −3 and 3 | 99.7% (= 95.4% + 4.3%) |

"Standard" means exactly these two parameters: mean 0, standard deviation 1.

### 2. The PDF: where the numbers are dense

The PDF (probability density function), written φ(x), is the bell curve. The higher the curve at a point, the more likely a draw lands near it:

\`\`\`text
x        0      0.5    1      1.5    2
φ(x)    0.399  0.352  0.242  0.130  0.054     ← highest at 0, lower toward both sides
\`\`\`

![Bell curve of the standard normal PDF, with φ(0), φ(1) and φ(2) marked](/llm-explorer/notes/normal_pdf.svg)

**The formula of the PDF:**

$$
\\varphi(x) = \\frac{1}{\\sqrt{2\\pi}}\\, e^{-x^2/2}
$$

It has two parts.

*e^(−x²/2) gives the bell shape:*

\`\`\`text
x            0      1       2       3
x²/2         0      0.5     2       4.5
e^(−x²/2)    1      0.607   0.135   0.011
\`\`\`

- It is largest (1) at x = 0.
- The farther x is from 0, the larger x², the more negative the exponent, and the faster the value drops.
- Because of the square, x and −x give the same value, so the curve is symmetric.

*1/√(2π) ≈ 0.399 makes the total area 1:* probabilities must add up to 1, so the area under the curve must be 1. The area under e^(−x²/2) alone is √(2π) ≈ 2.507 (a famous result called the Gaussian integral). Multiplying by 1/√(2π) brings the area to exactly 1.

Multiplying the two parts gives the PDF values in the table above:

\`\`\`text
φ(0) = 0.399 × 1     = 0.399
φ(1) = 0.399 × 0.607 = 0.242
φ(2) = 0.399 × 0.135 = 0.054
\`\`\`

**The general normal distribution** with mean μ and standard deviation σ:

$$
\\varphi(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\, e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}
$$

- **x − μ** moves the center from 0 to μ.
- **Dividing by σ²**: the larger σ, the wider the curve.
- **The 1/σ in front** makes the wider curve correspondingly lower, so the area stays 1.

With μ = 0 and σ = 1 this is the standard normal formula again.

### 3. The CDF: the area to the left of x

The CDF (cumulative distribution function), written Φ(x), answers one question:

> If you draw one random number, what is the probability that it is **smaller than x**?

On the plot, it is the **area under the bell curve to the left of x**. The total area under the whole curve is 1.

Φ(1) is the shaded area:

![PDF with the area left of x = 1 shaded: Φ(1) ≈ 0.841](/llm-explorer/notes/normal_cdf_area.svg)

Drawing 100,000 numbers and counting the fraction below x matches Φ(x) closely:

| x | −2 | −1 | 0 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| Φ(x), from the formula | 0.023 | 0.159 | 0.5 | 0.841 | 0.977 |
| fraction of 100,000 draws | 0.023 | 0.161 | 0.500 | 0.842 | 0.978 |

How to read them:

- **Φ(0) = 0.5:** half the numbers are below 0, because the distribution is symmetric around 0.
- **Φ(1) = 0.841:** 84% are below 1, so only 16% are above 1.
- **Φ(−1) = 0.159:** only 16% are below −1.
- **Φ(2) = 0.977:** almost all (97.7%) are below 2.

Properties:

- **It goes from 0 to 1:** Φ ≈ 0 for very negative x, Φ ≈ 1 for large x.
- **It only increases:** moving x right can only add area.
- **Symmetry:** Φ(−x) = 1 − Φ(x). For example Φ(−1) = 1 − 0.841 = 0.159.
- **Its derivative is the PDF:** Φ grows fastest where the bell is tallest (around 0) and barely moves at the far ends.

Plotting Φ(x) itself gives an S-shaped curve that rises from 0 to 1, steepest at 0:

![S-shaped curve of the standard normal CDF Φ(x)](/llm-explorer/notes/normal_cdf_curve.svg)

Two handy facts (the same 100,000 draws confirm them):

\`\`\`text
between -1 and 1:  Φ(1) - Φ(-1) = 0.841 - 0.159 = 0.683   (draws: 0.681)
between -2 and 2:  Φ(2) - Φ(-2) = 0.977 - 0.023 = 0.954   (draws: 0.955)
\`\`\`

### 4. The connection to GELU

\`\`\`text
GELU(x) = x × Φ(x)
\`\`\`

The idea in the GELU paper resembles dropout: **keep the input x with probability Φ(x), otherwise set it to 0**. The larger x, the closer Φ(x) is to 1 and the more surely it is kept. GELU does not actually draw anything random; it takes the **average** outcome:

\`\`\`text
average output = Φ(x) × x  +  (1 − Φ(x)) × 0  =  x × Φ(x)
                  kept          set to 0
\`\`\`

For x = 1: 84.1% of the time the output is 1, 15.9% of the time it is 0, so the average is 0.841, exactly GELU(1).

Why the normal CDF? After LayerNorm, a neuron's inputs are roughly spread around 0 with a scale of about 1. Φ(x) then asks "how high does this input rank within a typical distribution?", and the higher it ranks, the more of it is kept.

### Is there a formula for the CDF?

It depends on what "formula" means.

**1. A definition: yes.** The CDF is the area under the PDF from −∞ to x:

$$
\\Phi(x) = \\int_{-\\infty}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
$$

This is exact, but it contains an integral: computing Φ(1) means actually finding an area.

**2. A simple closed form: no.** A closed form would write Φ(x) using only basic functions: + − × ÷, powers, roots, eˣ, log, sin and so on. It has been **proved** that the integral of e^(−t²/2) cannot be written this way. Nobody is still looking; it is impossible.

**3. So it got its own name: erf.** Since no basic formula exists, mathematicians defined the integral as a new function, the error function erf:

$$
\\operatorname{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-t^2}\\, dt, \\qquad \\Phi(x) = \\frac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)
$$

How this formula follows from the definitions: [step-by-step derivation](#hover:erf_derivation_notes).

This is like sin(x): it cannot be written with + − × ÷ either, but it has a name and every computer has an implementation. \`torch.erf\` and \`math.erf\` give exact values:

\`\`\`python
0.5 * (1 + math.erf(1 / math.sqrt(2)))    # 0.8413 = Φ(1)
\`\`\`

Internally, erf is computed with series or piecewise polynomial approximations, accurate to the limit of floating-point precision.

**4. Approximate formulas: many.** GPT-2's tanh version of GELU is really **a tanh approximation of Φ(x)**:

$$
\\Phi(x) \\approx \\frac{1}{2}\\left(1 + \\tanh\\!\\left(\\sqrt{\\tfrac{2}{\\pi}}\\,\\left(x + 0.044715\\, x^3\\right)\\right)\\right)
$$

Multiply both sides by x and you get GELU's tanh formula.

| Question | Answer |
| --- | --- |
| An exact definition? | Yes: the integral of the PDF |
| Only basic functions? | No, proved impossible |
| How is it computed? | With erf: Φ(x) = 0.5 × (1 + erf(x/√2)) |
| A form without erf? | Approximations, e.g. the tanh one in GELU |
`,
    },
    zh: {
      title: "CDF",
      md: `
**标准正态分布的 CDF，一步一步讲**

### 1. 标准正态分布

\`torch.randn\` 抽出来的随机数就服从标准正态分布。抽 10 个：

\`\`\`text
-1.13, -1.15, -0.25, -0.43, 0.85, 0.69, -0.32, -2.12, 0.32, -1.26
\`\`\`

- **均值 0**：围绕 0 分布，正数和负数一样多。
- **标准差 1**：大部分数落在 −1 到 1 之间，很少超过 ±2，几乎不会超过 ±3：

| 区间 | 落在其中的比例 |
| --- | --- |
| −1 到 1 之间 | 68.3% |
| ±1 到 ±2 之间（两边合计） | 27.2% |
| ±2 到 ±3 之间（两边合计） | 4.3% |
| 超过 ±3 | 0.3% |
| *累计：* −2 到 2 之间 | 95.4%（= 68.3% + 27.2%） |
| *累计：* −3 到 3 之间 | 99.7%（= 95.4% + 4.3%） |

"标准"就是指均值 0、标准差 1 这一组参数。

### 2. PDF：哪里的数多

PDF（概率密度函数，记作 φ(x)）就是那条钟形曲线。曲线越高的地方，抽到附近的数的可能性越大：

\`\`\`text
x        0      0.5    1      1.5    2
φ(x)    0.399  0.352  0.242  0.130  0.054     ← 0 处最高，越往两边越低
\`\`\`

![标准正态分布 PDF 的钟形曲线，标出了 φ(0)、φ(1)、φ(2)](/llm-explorer/notes/normal_pdf.svg)

**PDF 的公式：**

$$
\\varphi(x) = \\frac{1}{\\sqrt{2\\pi}}\\, e^{-x^2/2}
$$

分成两部分看。

*e^(−x²/2)：决定钟形*

\`\`\`text
x            0      1       2       3
x²/2         0      0.5     2       4.5
e^(−x²/2)    1      0.607   0.135   0.011
\`\`\`

- x = 0 时最大（等于 1）。
- x 离 0 越远，x² 越大，e 的负指数越大，值掉得越快。
- 因为是平方，x 和 −x 的值一样，所以曲线左右对称。

*1/√(2π) ≈ 0.399：让总面积等于 1*。概率加起来必须是 1，所以曲线下的总面积必须是 1。单独 e^(−x²/2) 下面的面积是 √(2π) ≈ 2.507（一个著名的积分结果，叫高斯积分）。乘以 1/√(2π)，面积就正好是 1。

两部分乘起来，就是上面 PDF 表格里的数：

\`\`\`text
φ(0) = 0.399 × 1     = 0.399
φ(1) = 0.399 × 0.607 = 0.242
φ(2) = 0.399 × 0.135 = 0.054
\`\`\`

**一般的正态分布**，均值为 μ、标准差为 σ：

$$
\\varphi(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\, e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}
$$

- **x − μ**：把中心从 0 移到 μ。
- **除以 σ²**：σ 越大，曲线越宽。
- **前面的 1/σ**：曲线变宽后相应地压矮，让面积仍然是 1。

取 μ = 0、σ = 1，就回到标准正态分布的公式。

### 3. CDF：x 左边的面积

CDF（累积分布函数，记作 Φ(x)）回答的问题是：

> 随机抽一个数，它**小于 x** 的概率是多少？

在图上，就是钟形曲线下方、**x 左边那部分的面积**。整条曲线下的总面积是 1。

Φ(1) 就是涂色部分的面积：

![PDF 曲线下 x = 1 左边的面积被涂色：Φ(1) ≈ 0.841](/llm-explorer/notes/normal_cdf_area.svg)

抽 10 万个数，数一数有多少比 x 小，结果和 Φ(x) 几乎一样：

| x | −2 | −1 | 0 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| Φ(x)，公式算的 | 0.023 | 0.159 | 0.5 | 0.841 | 0.977 |
| 10 万次抽样中的比例 | 0.023 | 0.161 | 0.500 | 0.842 | 0.978 |

怎么读这些数：

- **Φ(0) = 0.5**：一半的数小于 0，因为分布关于 0 对称。
- **Φ(1) = 0.841**：84% 的数小于 1，也就是只有 16% 大于 1。
- **Φ(−1) = 0.159**：只有 16% 的数小于 −1。
- **Φ(2) = 0.977**：几乎所有数（97.7%）都小于 2。

性质：

- **从 0 涨到 1**：x 很负时 Φ ≈ 0，x 很大时 Φ ≈ 1。
- **单调递增**：x 往右移，左边的面积只会变大。
- **对称**：Φ(−x) = 1 − Φ(x)。例如 Φ(−1) = 1 − 0.841 = 0.159。
- **导数就是 PDF**：钟形曲线最高的地方（0 附近），Φ 涨得最快；两端几乎不动。

把 Φ(x) 本身画出来，是一条从 0 升到 1 的 S 形曲线，在 0 处最陡：

![标准正态分布 CDF Φ(x) 的 S 形曲线](/llm-explorer/notes/normal_cdf_curve.svg)

两个常用结论（同样的 10 万次抽样也验证了）：

\`\`\`text
between -1 and 1:  Φ(1) - Φ(-1) = 0.841 - 0.159 = 0.683   (draws: 0.681)
between -2 and 2:  Φ(2) - Φ(-2) = 0.977 - 0.023 = 0.954   (draws: 0.955)
\`\`\`

### 4. 和 GELU 的关系

\`\`\`text
GELU(x) = x × Φ(x)
\`\`\`

GELU 论文里的想法和 dropout 很像：**以概率 Φ(x) 保留输入 x，否则置 0**。x 越大，Φ(x) 越接近 1，越肯定被保留。GELU 不真的去随机，而是直接取**平均结果**：

\`\`\`text
average output = Φ(x) × x  +  (1 − Φ(x)) × 0  =  x × Φ(x)
                  kept          set to 0
\`\`\`

例如 x = 1：84.1% 的情况输出 1，15.9% 的情况输出 0，平均是 0.841，正好是 GELU(1)。

为什么选正态分布的 CDF？经过 LayerNorm 之后，神经元的输入大致分布在 0 附近、尺度约为 1。Φ(x) 就是在问："这个输入在一个典型分布里排第几？" 排得越靠前，保留得越多。

### CDF 有没有公式？

要看"公式"指什么。

**1. 定义式：有。** CDF 就是 PDF 从 −∞ 到 x 的面积：

$$
\\Phi(x) = \\int_{-\\infty}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
$$

这是准确的公式，但它包含积分：要算 Φ(1)，就得真的去求一块面积。

**2. 简单的"闭式"公式：没有。** 闭式是指只用加减乘除、幂、根号、eˣ、log、sin 这类基本函数，把 Φ(x) 直接写出来。数学上已经**证明**，e^(−t²/2) 的积分做不到这一点。这不是还没人找到，而是不可能存在。

**3. 所以给它起了名字：erf。** 既然写不成基本函数，数学家就把这个积分定义成一个新函数，叫误差函数 erf：

$$
\\operatorname{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-t^2}\\, dt, \\qquad \\Phi(x) = \\frac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)
$$

这个公式是怎么从定义推出来的：[一步一步的推导](#hover:erf_derivation_notes)。

这就像 sin(x)：它其实也不能用加减乘除写出来，只是有了名字，计算机里有现成的实现。\`torch.erf\`、\`math.erf\` 直接就能给出精确值：

\`\`\`python
0.5 * (1 + math.erf(1 / math.sqrt(2)))    # 0.8413 = Φ(1)
\`\`\`

计算机内部用级数或分段多项式逼近来算 erf，精度可以达到浮点数的极限。

**4. 近似公式：有很多。** GPT-2 用的 GELU tanh 版本，其实就是**用 tanh 近似 Φ(x)**：

$$
\\Phi(x) \\approx \\frac{1}{2}\\left(1 + \\tanh\\!\\left(\\sqrt{\\tfrac{2}{\\pi}}\\,\\left(x + 0.044715\\, x^3\\right)\\right)\\right)
$$

两边各乘以 x，就得到了 GELU 的 tanh 近似公式。

| 问题 | 答案 |
| --- | --- |
| 有没有准确的定义？ | 有：PDF 的积分 |
| 能不能只用基本函数写出来？ | 不能，数学上已证明 |
| 实际怎么算？ | 用 erf：Φ(x) = 0.5 × (1 + erf(x/√2)) |
| 有没有不用 erf 的写法？ | 有近似公式，比如 GELU 里的 tanh 版 |
`,
    },
  },

  erf_derivation_notes: {
    en: {
      title: "Deriving Φ(x) = ½(1 + erf(x/√2)) (hover)",
      md: `
**Deriving Φ(x) = ½(1 + erf(x/√2))**

Start from the two definitions:

$$
\\Phi(x) = \\int_{-\\infty}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt, \\qquad \\operatorname{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-u^2}\\, du
$$

They differ in two ways: the exponent ($-t^2/2$ vs. $-u^2$) and the lower limit ($-\\infty$ vs. 0). Fix them one at a time.

*Step 1: split the integral at 0.*

$$
\\Phi(x) = \\int_{-\\infty}^{0} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt + \\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
$$

The first part is $\\tfrac{1}{2}$: the total area under the PDF is 1, and the curve is symmetric around 0, so exactly half lies left of 0. (This is $\\Phi(0) = 0.5$.)

*Step 2: substitute $t = \\sqrt{2}\\, u$ to turn the exponent into $-u^2$.* Three things change:

- exponent: $t^2/2 = (\\sqrt{2}\\,u)^2/2 = u^2$;
- differential: $dt = \\sqrt{2}\\, du$ (the derivative of $t = \\sqrt{2}\\,u$ is $\\sqrt{2}$);
- limits: $t = 0 \\Rightarrow u = 0$, and $t = x \\Rightarrow u = x/\\sqrt{2}$ (from $u = t/\\sqrt{2}$).

$$
\\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
= \\int_{0}^{x/\\sqrt{2}} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-u^2} \\sqrt{2}\\, du
= \\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
$$

using $\\dfrac{\\sqrt{2}}{\\sqrt{2\\pi}} = \\dfrac{\\sqrt{2}}{\\sqrt{2}\\,\\sqrt{\\pi}} = \\dfrac{1}{\\sqrt{\\pi}}$.

*Step 3: match erf.* erf has $\\tfrac{2}{\\sqrt{\\pi}}$ in front, and we have $\\tfrac{1}{\\sqrt{\\pi}}$, exactly half:

$$
\\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
= \\frac{1}{2} \\cdot \\underbrace{\\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du}_{\\operatorname{erf}(x/\\sqrt{2})}
= \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)
$$

*Step 4: add the two parts.*

$$
\\Phi(x) = \\frac{1}{2} + \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right) = \\frac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)
$$

*Check with x = 1:* erf(1/√2) = erf(0.7071) = 0.6827, so Φ(1) = 0.5 × (1 + 0.6827) = 0.8413 ✓.

- **Negative x works too.** The second integral then runs from 0 to the left and is negative; erf is negative for negative inputs too: erf(−z) = −erf(z). For example Φ(−1) = 0.5 × (1 − 0.6827) = 0.159.
- **erf(x/√2) has its own meaning:** it is the share of numbers between −x and x. The 0.6827 above is the 68.3% between −1 and 1 from the CDF popup. Reason: the second integral is the area from 0 to x, which equals ½ erf(x/√2); by symmetry the area from −x to x is twice that, erf(x/√2).
- **Why 2/√π in front of erf:** it makes erf(∞) = 1, i.e. 100% of the numbers lie between −∞ and ∞.
`,
    },
    zh: {
      title: "推导 Φ(x) = ½(1 + erf(x/√2)) (hover)",
      md: `
**推导 Φ(x) = ½(1 + erf(x/√2))**

从两个定义出发：

$$
\\Phi(x) = \\int_{-\\infty}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt, \\qquad \\operatorname{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-u^2}\\, du
$$

两者有两处不同：指数（$-t^2/2$ 和 $-u^2$）和积分下限（$-\\infty$ 和 0）。下面逐一解决。

*第 1 步：在 0 处把积分拆开。*

$$
\\Phi(x) = \\int_{-\\infty}^{0} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt + \\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
$$

第一段等于 $\\tfrac{1}{2}$：PDF 曲线下的总面积是 1，而曲线关于 0 对称，所以 0 左边正好占一半。（这就是 $\\Phi(0) = 0.5$。）

*第 2 步：换元 $t = \\sqrt{2}\\, u$，让指数变成 $-u^2$。* 三样东西跟着变：

- 指数：$t^2/2 = (\\sqrt{2}\\,u)^2/2 = u^2$；
- 微分：$dt = \\sqrt{2}\\, du$（$t = \\sqrt{2}\\,u$ 对 $u$ 求导是 $\\sqrt{2}$）；
- 上下限：$t = 0 \\Rightarrow u = 0$，$t = x \\Rightarrow u = x/\\sqrt{2}$（由 $u = t/\\sqrt{2}$ 得到）。

$$
\\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
= \\int_{0}^{x/\\sqrt{2}} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-u^2} \\sqrt{2}\\, du
= \\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
$$

这里用到 $\\dfrac{\\sqrt{2}}{\\sqrt{2\\pi}} = \\dfrac{\\sqrt{2}}{\\sqrt{2}\\,\\sqrt{\\pi}} = \\dfrac{1}{\\sqrt{\\pi}}$。

*第 3 步：凑出 erf。* erf 前面是 $\\tfrac{2}{\\sqrt{\\pi}}$，现在是 $\\tfrac{1}{\\sqrt{\\pi}}$，正好差一半：

$$
\\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
= \\frac{1}{2} \\cdot \\underbrace{\\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du}_{\\operatorname{erf}(x/\\sqrt{2})}
= \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)
$$

*第 4 步：两段相加。*

$$
\\Phi(x) = \\frac{1}{2} + \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right) = \\frac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)
$$

*用 x = 1 验证：* erf(1/√2) = erf(0.7071) = 0.6827，所以 Φ(1) = 0.5 × (1 + 0.6827) = 0.8413 ✓。

- **x 为负数时也成立。** 这时第二段积分是从 0 往左积，结果为负；erf 对负数也是负的：erf(−z) = −erf(z)。例如 Φ(−1) = 0.5 × (1 − 0.6827) = 0.159。
- **erf(x/√2) 本身也有含义：** 它是落在 −x 到 x 之间的比例。上面的 0.6827，就是 CDF 弹窗里 −1 到 1 之间的 68.3%。原因：第二段积分是 0 到 x 的面积，等于 ½ erf(x/√2)；由对称性，−x 到 x 的面积是它的 2 倍，即 erf(x/√2)。
- **erf 前面为什么是 2/√π：** 这个系数让 erf(∞) = 1，也就是"落在 −∞ 到 ∞ 之间的比例是 100%"。
`,
    },
  },

  gelu_backprop_notes: {
    en: {
      title: "Backward through GELU: a worked example (hover)",
      md: `
**Backward through GELU: a worked example**

How does the gradient actually travel back from the loss, through GELU, to the weights of fc_in? Here is a tiny MLP, 2 → 4 → 2, computed step by step. All numbers come from PyTorch autograd and were checked by hand. It uses the exact GELU (GPT-2 uses the tanh version, which differs only in the 4th decimal).

## The setup

- input \`x = [1, 2]\` (one token, 2 features instead of 768)
- fc_in: \`W1\` is (4, 2), 4 detectors instead of 3072; its rows are \`[1, 1]\`, \`[1, −1]\`, \`[−1, 0]\`, \`[0, 1]\`
- fc_out: \`W2\` is (2, 4): \`[[1, 0, 1, 0], [0, 1, 1, −1]]\`
- no biases, to keep the arithmetic short
- a target \`y = [2, −2]\` and a simple loss, the squared error: \`loss = ½ × Σ (out − y)²\`. (GPT-2 uses cross entropy instead; the backward mechanism is the same.)

## Forward

\`\`\`text
h    = W1 · x     = [3, -1, -1, 2]
a    = GELU(h)    = [2.996, -0.159, -0.159, 1.954]
out  = W2 · a     = [2.837, -2.272]
loss = ½ × (0.837² + 0.272²) = 0.387
\`\`\`

For example \`h[1] = [1, −1] · [1, 2] = 1 − 2 = −1\`, and \`out[0] = a[0] + a[2] = 2.996 − 0.159 = 2.837\`.

## Backward: start at the loss, walk back one layer at a time

The rule is the **chain rule**: each time you go back through a layer, multiply the gradient you are holding by that layer's local derivative. "\`g_v\`" below means "the gradient of the loss with respect to v": how much the loss changes when v increases by 1.

### Step 1: the gradient of out

The derivative of \`½ (out − y)²\` with respect to out is \`out − y\`:

\`\`\`text
g_out = [2.837 − 2, −2.272 + 2] = [0.837, −0.272]
\`\`\`

out[0] is 0.837 too high, so it should go down; out[1] is 0.272 too low, so it should go up.

### Step 2: back through fc_out (out = W2 · a)

This layer produces two things.

**(a) The gradient of its weights W2**, used later to update W2. \`W2[i][j]\` affects the loss only through the term \`W2[i][j] × a[j]\` inside \`out[i]\`, so:

\`\`\`text
g_W2[i][j] = g_out[i] × a[j]

g_W2 = [[ 2.509, -0.133, -0.133,  1.637],
        [-0.814,  0.043,  0.043, -0.531]]
\`\`\`

For example \`g_W2[0][0] = 0.837 × 2.996 = 2.509\`.

**(b) The gradient of a**, passed further back. Write fc_out out in full:

\`\`\`text
out[0] = a[0]        + a[2]
out[1] =        a[1] + a[2] − a[3]
\`\`\`

One \`a[j]\` can appear in **several** outputs. The coefficient in front of \`a[j]\` in \`out[i]\` is \`W2[i][j]\`; going down the outputs, those coefficients are column j of W2. If \`a[j]\` increases by a small amount, every out[i] it appears in moves, and each of those moves changes the loss. So the gradient is the **sum over all paths**:

\`\`\`text
g_a[j] = Σ_i  g_out[i] × W2[i][j]
\`\`\`

Plugging in the numbers for all four:

\`\`\`text
g_a[0] = 0.837×1 + (−0.272)×0    =  0.837
g_a[1] = 0.837×0 + (−0.272)×1    = −0.272
g_a[2] = 0.837×1 + (−0.272)×1    =  0.565
g_a[3] = 0.837×0 + (−0.272)×(−1) =  0.272
\`\`\`

\`a[3]\` is not in out[0] (coefficient 0), so that path contributes nothing; its coefficient in out[1] is −1, which flips −0.272 into +0.272. In matrix form this is \`g_a = W2ᵀ · g_out\`: transposing turns column j of W2 into row j.

### Step 3: back through GELU (the key step)

GELU acts on each number separately, so its backward is also one number at a time: multiply by the GELU gradient at that point (the values in the gradient table of the GELU note).

\`\`\`text
g_h[j] = g_a[j] × GELU'(h[j])
\`\`\`

| j | h[j] | GELU'(h[j]) | g_a[j] | g_h[j] |
| --- | --- | --- | --- | --- |
| 0 | 3 | 1.012 | 0.837 | 0.847 |
| 1 | −1 | −0.083 | −0.272 | 0.023 |
| 2 | −1 | −0.083 | 0.565 | −0.047 |
| 3 | 2 | 1.085 | 0.272 | 0.295 |

- Detectors 0 and 3 fired (h > 0): GELU' ≈ 1, so the gradient passes almost unchanged.
- Detectors 1 and 2 did not fire (h = −1): GELU' = −0.083, so the gradient shrinks to about 1/12, but it is **not 0**.

**With ReLU instead**, ReLU' at h = [3, −1, −1, 2] is [1, 0, 0, 1]:

\`\`\`text
g_h = [0.837, 0, 0, 0.272]
\`\`\`

Detectors 1 and 2 get exactly 0. This is where "dying ReLU" comes from: a detector that did not fire learns nothing from this step.

### Step 4: back through fc_in (h = W1 · x)

Same rule as step 2(a): \`W1[j][k]\` only acts through \`W1[j][k] × x[k]\` inside \`h[j]\`:

\`\`\`text
g_W1[j][k] = g_h[j] × x[k]          x = [1, 2]

g_W1 = [[ 0.847,  1.695],
        [ 0.023,  0.045],
        [-0.047, -0.094],
        [ 0.295,  0.590]]
\`\`\`

Row j of \`g_W1\` is \`g_h[j] × x\`. Rows 1 and 2 are small but nonzero, thanks to GELU; with ReLU they would be exactly 0, and those two detectors' weights would not move.

In the real model the backward continues: it also computes \`g_x = W1ᵀ · g_h\` and passes it on to ln2, the residual stream, and back to the embeddings. This small example stops here.

## Update the weights (learning rate 0.1)

Each weight takes a small step against its gradient: \`w ← w − 0.1 × g_w\`.

\`\`\`text
W1 row 0: [1,  1] → [1 − 0.0847, 1 − 0.1695] = [0.915,  0.831]
W1 row 1: [1, −1] → [1 − 0.0023, −1 − 0.0045] = [0.998, -1.005]
W2 changes the same way.
\`\`\`

Forward again with the new weights:

\`\`\`text
loss: 0.387 → 0.163     ✓ smaller
\`\`\`

## The pattern

Going back, every layer does two things:

1. **Gradient of its own weights:** (gradient arriving here) × (the input this layer saw in the forward pass).
2. **Gradient to pass back:** (gradient arriving here) × (this layer's local derivative).

| Layer | Forward | Gradient passed back | Gradient of its weights |
| --- | --- | --- | --- |
| loss | ½ (out − y)² | out − y | none |
| fc_out | W2 · a | W2ᵀ · g_out | g_out × a |
| GELU | GELU(h), per number | g_a × GELU'(h), per number | none (no weights) |
| fc_in | W1 · x | W1ᵀ · g_h | g_h × x |

The backward needs values from the forward pass: \`a\` for W2's gradient, \`h\` for GELU', \`x\` for W1's gradient. PyTorch therefore keeps these intermediate values in memory after the forward pass, which is why training uses much more memory than inference.
`,
    },
    zh: {
      title: "穿过 GELU 的 backward：具体例子 (hover)",
      md: `
**穿过 GELU 的 backward：具体例子**

梯度到底是怎么从 loss 出发，穿过 GELU，一直传到 fc_in 的权重上的？下面用一个很小的 MLP（2 → 4 → 2）一步一步算。所有数字都是 PyTorch 自动求导算出来的，也逐步手算核对过。这里用的是精确版 GELU；GPT-2 用的 tanh 版只在小数点后第 4 位有差别。

## 设定

- 输入 \`x = [1, 2]\`（一个 token，2 维，代替 768 维）
- fc_in：\`W1\` 形状是 (4, 2)，4 个检测器（代替 3072 个），4 行分别是 \`[1, 1]\`、\`[1, −1]\`、\`[−1, 0]\`、\`[0, 1]\`
- fc_out：\`W2\` 形状是 (2, 4)：\`[[1, 0, 1, 0], [0, 1, 1, −1]]\`
- 不加 bias，让计算短一点
- 目标 \`y = [2, −2]\`，loss 用简单的平方误差：\`loss = ½ × Σ (out − y)²\`。（GPT-2 最后用的是 cross entropy，但 backward 的机制完全一样。）

## Forward

\`\`\`text
h    = W1 · x     = [3, -1, -1, 2]
a    = GELU(h)    = [2.996, -0.159, -0.159, 1.954]
out  = W2 · a     = [2.837, -2.272]
loss = ½ × (0.837² + 0.272²) = 0.387
\`\`\`

比如 \`h[1] = [1, −1] · [1, 2] = 1 − 2 = −1\`，\`out[0] = a[0] + a[2] = 2.996 − 0.159 = 2.837\`。

## Backward：从 loss 开始，一层一层往回走

规则是**链式法则**：每往回穿过一层，就把手上的梯度乘以这一层的局部导数。下面的 "\`g_v\`" 表示"loss 对 v 的梯度"，也就是 v 增加 1 时 loss 大约变化多少。

### 第 1 步：out 的梯度

\`½ (out − y)²\` 对 out 求导是 \`out − y\`：

\`\`\`text
g_out = [2.837 − 2, −2.272 + 2] = [0.837, −0.272]
\`\`\`

out[0] 偏大了 0.837，应该往下调；out[1] 偏小了 0.272，应该往上调。

### 第 2 步：穿过 fc_out（out = W2 · a）

这一层要算两样东西。

**(a) 自己权重 W2 的梯度**，之后用来更新 W2。\`W2[i][j]\` 只通过 \`out[i]\` 里的 \`W2[i][j] × a[j]\` 这一项影响 loss，所以：

\`\`\`text
g_W2[i][j] = g_out[i] × a[j]

g_W2 = [[ 2.509, -0.133, -0.133,  1.637],
        [-0.814,  0.043,  0.043, -0.531]]
\`\`\`

比如 \`g_W2[0][0] = 0.837 × 2.996 = 2.509\`。

**(b) a 的梯度**，继续往前传。先把 fc_out 展开写：

\`\`\`text
out[0] = a[0]        + a[2]
out[1] =        a[1] + a[2] − a[3]
\`\`\`

一个 \`a[j]\` 可能出现在**好几个**输出里。\`a[j]\` 在 \`out[i]\` 里前面的系数是 \`W2[i][j]\`；把各个输出里的这些系数竖着排起来，正好是 W2 的第 j 列。\`a[j]\` 增加一点点，它出现的每个 out[i] 都会跟着变，每个变化又都会影响 loss。所以梯度是**所有路径的贡献加起来**：

\`\`\`text
g_a[j] = Σ_i  g_out[i] × W2[i][j]
\`\`\`

代入数字，四个全部算出来：

\`\`\`text
g_a[0] = 0.837×1 + (−0.272)×0    =  0.837
g_a[1] = 0.837×0 + (−0.272)×1    = −0.272
g_a[2] = 0.837×1 + (−0.272)×1    =  0.565
g_a[3] = 0.837×0 + (−0.272)×(−1) =  0.272
\`\`\`

\`a[3]\` 不在 out[0] 里（系数是 0），这条路径没有贡献；它在 out[1] 里的系数是 −1，把 −0.272 翻成了 +0.272。写成矩阵形式就是 \`g_a = W2ᵀ · g_out\`：转置以后，W2 的第 j 列变成了第 j 行。

### 第 3 步：穿过 GELU（重点）

GELU 对每个数单独作用，所以它的 backward 也是逐个数做：乘以这个点上 GELU 的梯度（就是 GELU notes 梯度表里的那些数）。

\`\`\`text
g_h[j] = g_a[j] × GELU'(h[j])
\`\`\`

| j | h[j] | GELU'(h[j]) | g_a[j] | g_h[j] |
| --- | --- | --- | --- | --- |
| 0 | 3 | 1.012 | 0.837 | 0.847 |
| 1 | −1 | −0.083 | −0.272 | 0.023 |
| 2 | −1 | −0.083 | 0.565 | −0.047 |
| 3 | 2 | 1.085 | 0.272 | 0.295 |

- 检测器 0 和 3 触发了（h > 0）：GELU' ≈ 1，梯度几乎原样通过。
- 检测器 1 和 2 没触发（h = −1）：GELU' = −0.083，梯度被压小到约 1/12，但**不是 0**。

**如果换成 ReLU**，h = [3, −1, −1, 2] 处的 ReLU' 是 [1, 0, 0, 1]：

\`\`\`text
g_h = [0.837, 0, 0, 0.272]
\`\`\`

检测器 1 和 2 收到的梯度正好是 0。这就是"dying ReLU"的来源：没触发的检测器在这一步什么也学不到。

### 第 4 步：穿过 fc_in（h = W1 · x）

和第 2 步 (a) 同样的规则：\`W1[j][k]\` 只通过 \`h[j]\` 里的 \`W1[j][k] × x[k]\` 这一项起作用：

\`\`\`text
g_W1[j][k] = g_h[j] × x[k]          x = [1, 2]

g_W1 = [[ 0.847,  1.695],
        [ 0.023,  0.045],
        [-0.047, -0.094],
        [ 0.295,  0.590]]
\`\`\`

\`g_W1\` 的第 j 行就是 \`g_h[j] × x\`。第 1、2 行很小但不是 0，这要归功于 GELU；换成 ReLU 就正好是 0，这两个检测器的权重不会动。

在真实模型里，backward 还会继续：算出 \`g_x = W1ᵀ · g_h\`，再往前传给 ln2、残差流，一直传到 embedding。这个小例子到这里就停了。

## 更新权重（学习率 0.1）

每个权重都朝梯度的反方向走一小步：\`w ← w − 0.1 × g_w\`。

\`\`\`text
W1 第 0 行：[1,  1] → [1 − 0.0847, 1 − 0.1695]  = [0.915,  0.831]
W1 第 1 行：[1, −1] → [1 − 0.0023, −1 − 0.0045] = [0.998, -1.005]
W2 也按同样方式更新。
\`\`\`

用新的权重再做一次 forward：

\`\`\`text
loss：0.387 → 0.163     ✓ 变小了
\`\`\`

## 规律总结

往回走时，每一层都做两件事：

1. **自己权重的梯度**：（传到这里的梯度）×（这一层 forward 时的输入）。
2. **往前传的梯度**：（传到这里的梯度）×（这一层的局部导数）。

| 层 | Forward | 往前传的梯度 | 自己权重的梯度 |
| --- | --- | --- | --- |
| loss | ½ (out − y)² | out − y | 无 |
| fc_out | W2 · a | W2ᵀ · g_out | g_out × a |
| GELU | GELU(h)，逐个数 | g_a × GELU'(h)，逐个数 | 无（没有权重） |
| fc_in | W1 · x | W1ᵀ · g_h | g_h × x |

backward 要用到 forward 时的中间值：算 W2 的梯度要用 \`a\`，算 GELU' 要用 \`h\`，算 W1 的梯度要用 \`x\`。所以 forward 之后 PyTorch 会把这些中间值留在内存里，这也是训练比推理占内存多得多的原因。
`,
    },
  },

  mlp_dropout_notes: {
    en: {
      title: "Residual dropout in MLP",
      md: `
**\`self.drop\` randomly zeros part of the MLP output during training, right before that output is added to the residual stream.** It is the last step of the MLP, and it is the same residual dropout as \`resid_drop\` at the end of attention.

## Where it sits

\`\`\`python
self.drop = nn.Dropout(dropout)                       # in __init__; this site uses dropout = 0.1
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))  # end of forward; output: (B, T, C)
\`\`\`

The returned value is what the block adds to the residual stream:

\`\`\`python
x = x + self.mlp(self.ln2(x))
\`\`\`

So the order is: [fc_out](#info:mlp_fc_out_notes) → \`drop\` → add to \`x\`.

Two different \`x\` appear here, which is easy to mix up:

- **\`x\` in the block** is the residual stream: each token's 768-number vector that started as token embedding + position embedding and has had every earlier sublayer's update added to it.
- **\`x\` inside \`MLP.forward\`** is just its argument, \`ln2(x)\`: a normalized copy of the residual stream. The MLP reads that copy, computes an update, and the block adds the update back.

## What it does

The same rule as every \`nn.Dropout\` (details in [Residual dropout in Causal Self-Attention](#info:resid_dropout_notes)):

- **Training** (\`model.train()\`): each number of the \`fc_out\` output is independently set to 0 with probability $p = 0.1$; every number that survives is multiplied by $1/(1-p) = 1/0.9 \\approx 1.111$, which keeps the average output unchanged.
- **Inference** (\`model.eval()\`): it does nothing; the output equals the input.

It acts on the \`fc_out\` output, shape $(B,\\ T,\\ C)$: the update the MLP is about to write into the residual stream. It does **not** act on the 3072 GELU outputs inside the MLP.

## A worked example

One token, first 4 of the 768 dimensions. The numbers are made up for illustration:

\`\`\`text
dimension:              0      1      2      3
x (residual stream):  [0.20, -0.50,  1.00,  0.30]
update = fc_out(...): [0.50, -0.20,  0.90,  0.10]
\`\`\`

**Without dropout** (inference), the whole update is added:

\`\`\`text
x + update          = [0.70, -0.70,  1.90,  0.40]
\`\`\`

**With dropout** (training), suppose this time dimension 2 is drawn to be dropped:

\`\`\`text
mask:                 [ 1,     1,     0,     1   ]
dropped, × 1/0.9:     [0.556, -0.222, 0,    0.111]   ← 0.50/0.9, -0.20/0.9, 0, 0.10/0.9
x + dropped update  = [0.756, -0.722, 1.00, 0.411]
\`\`\`

In dimension 2 the update is gone, but \`x\`'s original **1.00 is still there**. The effect is "this time the MLP wrote nothing into dimension 2", not "the information in dimension 2 is lost". The next training step draws a new mask.

## What it does to the backward pass

In the forward pass a dropped number is multiplied by 0 and a surviving one by 1.111, so in the backward pass the gradient is multiplied by exactly the same numbers (the chain rule; see the [worked backward example](#hover:gelu_backprop_notes)):

\`\`\`text
gradient arriving from later layers:  [g0,        g1,        g2, g3       ]
after passing back through dropout:   [1.111×g0,  1.111×g1,  0,  1.111×g3 ]
\`\`\`

Dimension 2 of the update had no effect on the loss this step, so its gradient is 0. Dimension 2 of the output is written by **row 2 of \`fc_out.weight\`** (and \`fc_out.bias[2]\`), so in this step that row gets no gradient from this token and does not change because of it. Next step a different set of rows is skipped.

## Why use it

### 1. Regularization: no single dimension can be relied on

Without dropout, the MLP could learn a shortcut, such as always writing one important signal into one particular dimension, with later layers learning to read only that dimension. With dropout, any dimension of the update may vanish in a given step, so the result has to come out right even with part of it missing. Information gets spread across several dimensions, which generalizes better to text the model has not seen.

This matters especially for the MLP: it holds about two thirds of each block's parameters (4.7M vs. 2.4M for attention), and more parameters mean more capacity to memorize the training data.

### 2. It does not damage the main path

Dropout is on the **branch** (\`mlp(...)\`), not on \`x\`. The residual stream passes through \`x + ...\` complete. If dropout were applied to \`x\` itself in every block, each of the 12 blocks would randomly erase part of everything the earlier layers had accumulated, and the losses would compound with depth.

## Choosing p

- GPT-2 uses $p = 0.1$.
- 🆕 Many recent LLMs (e.g. LLaMA) pretrain with dropout 0: the training data is huge and usually seen about once, so overfitting is not the main problem, and dropout would only slow learning down.

## References

1. **Original Transformer paper (where residual dropout is defined)**

   Vaswani et al., "Attention Is All You Need," §5.4 "Regularization": applies dropout to the output of each sublayer before adding it to the residual (and also to embeddings/attention probs).

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 specifics (configs & terminology)**

   Hugging Face docs show GPT-2's three dropouts: \`resid_pdrop\` (residual branches), \`attn_pdrop\` (attention probs), \`embd_pdrop\` (embeddings).

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

3. **General dropout theory (why it helps)**

   Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR 2014).

   [Dropout: A Simple Way to Prevent Neural Networks from Overfitting ↗](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

4. **Related/contrastive regularizer (dropping whole residual paths)**

   Huang et al., "Deep Networks with Stochastic Depth" (CVPR 2016): drops entire residual blocks during training ("drop-path"), a contrast with element-wise residual dropout.

   [Deep Networks with Stochastic Depth ↗](https://arxiv.org/abs/1603.09382)
`,
    },
    zh: {
      title: "MLP 里的 residual dropout",
      md: `
**\`self.drop\` 在训练时随机把 MLP 输出的一部分置成 0，然后这个输出才加到残差流上。** 它是 MLP 的最后一步，和 attention 末尾的 \`resid_drop\` 是同一种 residual dropout。

## 位置

\`\`\`python
self.drop = nn.Dropout(dropout)                       # 在 __init__ 里；本站用 dropout = 0.1
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))  # forward 的最后；输出 (B, T, C)
\`\`\`

返回值就是 block 加到残差流上的东西：

\`\`\`python
x = x + self.mlp(self.ln2(x))
\`\`\`

所以顺序是：[fc_out](#info:mlp_fc_out_notes) → \`drop\` → 加到 \`x\` 上。

这里出现了两个不同的 \`x\`，很容易混：

- **block 里的 \`x\`** 是残差流：每个 token 一个 768 维向量，起点是 token embedding + 位置 embedding，之后每个子层的更新量都加了上去。
- **\`MLP.forward\` 里的 \`x\`** 只是它的参数，也就是 \`ln2(x)\`：残差流做过 LayerNorm 的一个副本。MLP 读这个副本、算出更新量，再由 block 把更新量加回残差流。

## 它做什么

规则和所有 \`nn.Dropout\` 一样（细节见 [Residual dropout in Causal Self-Attention](#info:resid_dropout_notes)）：

- **训练**（\`model.train()\`）：\`fc_out\` 输出的每个数各自以 $p = 0.1$ 的概率被置成 0；留下来的数都乘以 $1/(1-p) = 1/0.9 \\approx 1.111$，这样平均值保持不变。
- **推理**（\`model.eval()\`）：什么都不做，输出等于输入。

它作用在 \`fc_out\` 的输出上，形状是 $(B,\\ T,\\ C)$，也就是 MLP 准备写进残差流的更新量。它**不**作用在 MLP 内部那 3072 个 GELU 输出上。

## 具体例子

一个 token，只看 768 维里的前 4 维。数字是为了演示编的：

\`\`\`text
dim                    0      1      2      3
x                    [0.20, -0.50,  1.00,  0.30]   ← 残差流
update = fc_out(...) [0.50, -0.20,  0.90,  0.10]   ← 更新量
\`\`\`

**不 dropout**（推理）时，整个更新量都加上去：

\`\`\`text
x + update           [0.70, -0.70,  1.90,  0.40]
\`\`\`

**dropout**（训练）时，假设这次抽中第 2 维被丢掉：

\`\`\`text
mask                 [ 1,     1,     0,     1   ]
update × mask/0.9    [0.556, -0.222, 0,    0.111]   ← 0.50/0.9, -0.20/0.9, 0, 0.10/0.9
new x                [0.756, -0.722, 1.00, 0.411]   ← x + 丢过的更新量
\`\`\`

第 2 维的更新量没了，但 \`x\` 原来的 **1.00 还在**。效果是"这一次 MLP 没往第 2 维写东西"，而不是"第 2 维的信息丢了"。下一步训练会重新抽一个 mask。

## 对 backward 的影响

forward 时被丢掉的数乘了 0，留下来的乘了 1.111，所以 backward 时梯度也乘以完全相同的数（链式法则，见[穿过 GELU 的 backward 例子](#hover:gelu_backprop_notes)）：

\`\`\`text
[g0,        g1,        g2, g3       ]   ← 从后面传回来的梯度
[1.111×g0,  1.111×g1,  0,  1.111×g3 ]   ← 穿过 dropout 之后
\`\`\`

这一步里，更新量的第 2 维对 loss 没有影响，所以它的梯度是 0。输出的第 2 维是由 **\`fc_out.weight\` 的第 2 行**（和 \`fc_out.bias[2]\`）写出来的，所以这一步里这一行从这个 token 收不到梯度，也不会因为它而改变。下一步换另一批行被跳过。

## 为什么要用

### 1. 正则化：不能依赖某一个维度

没有 dropout 时，MLP 可能学会走捷径，比如总把某个重要信号写在固定的一维上，后面的层也只学会读这一维。有了 dropout，更新量的任何一维在某一步都可能消失，所以必须在少了一部分的情况下照样得到对的结果。信息因此分散到多个维度上，对没见过的文本泛化得更好。

这对 MLP 尤其重要：它占了每个 block 大约三分之二的参数（4.7M，attention 是 2.4M），参数越多，越有能力死记训练数据。

### 2. 不伤害主路径

dropout 放在**分支**（\`mlp(...)\`）上，而不是 \`x\` 上。残差流穿过 \`x + ...\` 时始终是完整的。如果每个 block 都对 \`x\` 本身做 dropout，12 个 block 每一个都会随机抹掉前面所有层积累下来的一部分信息，层数越深损失越多。

## p 怎么选

- GPT-2 用 $p = 0.1$。
- 🆕 现在很多大模型（比如 LLaMA）预训练时把 dropout 设成 0：数据量非常大，每条数据基本只看一遍，过拟合不是主要问题，dropout 反而会拖慢学习。

## 参考资料

1. **Transformer 原论文（residual dropout 的出处）**

   Vaswani et al., "Attention Is All You Need," §5.4 "Regularization"：在每个子层的输出加回残差之前做 dropout（embedding 和 attention 概率上也有）。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 的具体配置和术语**

   Hugging Face 文档列出了 GPT-2 的三种 dropout：\`resid_pdrop\`（残差分支）、\`attn_pdrop\`（attention 概率）、\`embd_pdrop\`（embedding）。

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

3. **dropout 的一般理论（为什么有效）**

   Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR 2014)。

   [Dropout: A Simple Way to Prevent Neural Networks from Overfitting ↗](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

4. **对比：整条残差分支一起丢掉**

   Huang et al., "Deep Networks with Stochastic Depth" (CVPR 2016)：训练时随机丢掉整个残差 block（"drop-path"），可以和逐元素的 residual dropout 对比。

   [Deep Networks with Stochastic Depth ↗](https://arxiv.org/abs/1603.09382)
`,
    },
  },
};
