// Notes: Residual connections and initialization.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  residual_why_notes: {
    en: {
      title: "Why Residual Connections",
      md: `
**What would happen if a block replaced $x$ instead of adding to it? Comparing the two shows why GPT-2 needs residual connections.**

Count sublayers rather than blocks: each block has 2 (attention and MLP), so GPT-2 small has **24 sublayers** in total. $x_0$ is the input to the first block: token embedding + position embedding.

## 1. With residual: the original information stays intact

$$
\\begin{aligned}
x_1 &= x_0 + \\Delta_1 \\\\[4pt]
x_2 &= x_1 + \\Delta_2 = x_0 + \\Delta_1 + \\Delta_2 \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= x_0 + \\Delta_1 + \\Delta_2 + \\cdots + \\Delta_{24}
\\end{aligned}
$$

$x_0$ is still present, **unchanged**, in $x_{24}$. Each sublayer only has to learn *what to add*; none of them is responsible for preserving what was already there.

## 2. Without residual: the information is handed over 24 times

$$
\\begin{aligned}
x_1 &= f_1(x_0) \\\\[4pt]
x_2 &= f_2(x_1) = f_2(f_1(x_0)) \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= f_{24}(f_{23}(\\cdots f_2(f_1(x_0))\\cdots))
\\end{aligned}
$$

$x_0$ is now buried inside 24 nested functions. For the last layer to know "which token was here, and at which position", **every one** of the 24 sublayers must learn to pass that information through intact. If any one of them fails, it is lost.

> **Analogy.** Without residual, it is a game of telephone: 24 people pass a message along, each retelling it in their own words, and by the end it is distorted. With residual, the original document travels in a folder; each person only attaches a sticky note with additions. The last person sees the original plus every note.

## 3. Attention makes the problem worse

The attention output is a **weighted average of other positions' values**. Suppose position 5 attends mostly to positions 3 and 4; its attention output is then mostly their content.

- **Without residual:** that output *replaces* $x$, so position 5's own identity ("which token am I?") is washed out.
- **With residual:** the output is *added*, so position 5's own information is still there.

## 4. Gradients: the same story in reverse

During backprop, the gradient must travel from the last layer back to the first.

- **Without residual:** the gradient is multiplied by each of the 24 sublayers' Jacobians in turn. If each one scales it by a factor even slightly away from 1, the effect compounds:
  - $0.8^{24} \\approx 0.005$: the gradient almost vanishes, and the early layers barely learn.
  - $1.2^{24} \\approx 80$: the gradient explodes.
- **With residual:** each sublayer's Jacobian is $I + J_{\\text{sublayer}}$. The $I$ is the direct path from $x_0$, so the gradient can reach the first layers without being multiplied away.

See [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes) for the Jacobian picture in more detail.

## Bottom line

> Residual connections turn $x$ into an **add-only main road**, the residual stream. The original information is never overwritten, and gradients get a direct path back. This is a key reason 12-layer models, and today's models with 100+ layers, can be trained at all.
`,
    },
    zh: {
      title: "为什么要用残差连接",
      md: `
**如果一个 block 不是往 $x$ 上加东西，而是直接替换 $x$，会怎样？把两种做法对比一下，就能看出 GPT-2 为什么需要残差连接。**

按子层而不是按 block 来数：每个 block 有 2 个子层（attention 和 MLP），所以 GPT-2 small 一共有 **24 个子层**。$x_0$ 是第一个 block 的输入：token embedding + 位置 embedding。

## 1. 有残差：原来的信息完整保留

$$
\\begin{aligned}
x_1 &= x_0 + \\Delta_1 \\\\[4pt]
x_2 &= x_1 + \\Delta_2 = x_0 + \\Delta_1 + \\Delta_2 \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= x_0 + \\Delta_1 + \\Delta_2 + \\cdots + \\Delta_{24}
\\end{aligned}
$$

$x_0$ 仍然**原封不动**地在 $x_{24}$ 里。每个子层只需要学"该加什么"，没有哪个子层要负责把已有的信息保存下来。

## 2. 没有残差：信息要被转手 24 次

$$
\\begin{aligned}
x_1 &= f_1(x_0) \\\\[4pt]
x_2 &= f_2(x_1) = f_2(f_1(x_0)) \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= f_{24}(f_{23}(\\cdots f_2(f_1(x_0))\\cdots))
\\end{aligned}
$$

现在 $x_0$ 被埋在 24 层嵌套函数里面。最后一层要想知道"这里是哪个 token、在哪个位置"，24 个子层里的**每一个**都必须学会把这些信息完整地传下去。只要有一个没做到，信息就丢了。

> **类比。** 没有残差就像传话游戏：24 个人依次传一句话，每个人都用自己的话复述一遍，到最后已经走样了。有残差时，原始文件装在文件夹里一路传下去，每个人只在上面贴一张写着补充内容的便利贴。最后一个人能看到原文加上所有便利贴。

## 3. attention 让这个问题更严重

attention 的输出是**其他位置 value 的加权平均**。假设位置 5 主要关注位置 3 和 4，它的 attention 输出就主要是这两个位置的内容。

- **没有残差：** 这个输出会*替换*掉 $x$，位置 5 自己的身份（"我是哪个 token？"）就被冲掉了。
- **有残差：** 这个输出是*加上去*的，位置 5 自己的信息仍然在。

## 4. 梯度：同样的道理，方向相反

backprop 时，梯度要从最后一层一路传回第一层。

- **没有残差：** 梯度要依次乘以 24 个子层的 Jacobian。只要每次乘的因子稍微偏离 1，影响就会累积：
  - $0.8^{24} \\approx 0.005$：梯度几乎消失，前面几层几乎学不到东西。
  - $1.2^{24} \\approx 80$：梯度爆炸。
- **有残差：** 每个子层的 Jacobian 是 $I + J_{\\text{子层}}$。其中的 $I$ 就是从 $x_0$ 直接过来的路径，所以梯度能传到前面几层，不会被一路乘没。

Jacobian 角度的更多细节见 [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes)。

## 结论

> 残差连接把 $x$ 变成一条**只加不改的主干道**，也就是残差流。原来的信息永远不会被覆盖，梯度也有一条直达的回路。这正是 12 层的模型、以及今天 100 多层的模型能训练起来的关键原因之一。
`,
    },
  },

  attn_residual_notes: {
    en: {
      title: "Attention residual",
      md: `
**Code:** \`x = x + self.attn(self.ln1(x))\`

**This line is the first half of a Transformer block: normalize the residual stream, let attention compute an update for each token from the tokens before it, and add that update back.** The second half, \`x = x + self.mlp(self.ln2(x))\`, has the same shape (see [MLP residual](#info:mlp_residual_notes)).

## Where it sits

\`\`\`python
def forward(self, x):                    # x: (B, T, C), the residual stream
    x = x + self.attn(self.ln1(x))       # first half: attention  ← this line
    x = x + self.mlp(self.ln2(x))        # second half: MLP
    return x
\`\`\`

\`x\` here is the **residual stream**: each token's 768-number vector. In block 1 it is token embedding + position embedding; after that, every sublayer (12 blocks × 2 = 24 of them) adds its update to it. A useful picture is a notebook per token: the embedding writes the first draft, and every sublayer adds a note.

## The three steps

1. **\`self.ln1(x)\`, normalize first (pre-LN).** [LayerNorm](#info:layernorm) rescales each token's 768 numbers to roughly mean 0 and variance 1, so attention always sees inputs on a similar scale, which keeps the q·k scores and the softmax in a stable range. It produces a normalized **copy**; \`x\` itself is not changed.
2. **\`self.attn(...)\`, compute the update.** Causal self-attention lets each token take information from itself and the tokens before it (never after it, because of the causal mask), and returns an update of shape $(B, T, C)$. It ends with the output projection and [residual dropout](#info:resid_dropout_notes).
3. **\`x + ...\`, add it back.** The update is added number by number to the residual stream.

Shapes, which must match for the add:

\`\`\`text
x                 (B, T, C)
self.ln1(x)       (B, T, C)
self.attn(...)    (B, T, C)
x + ...           (B, T, C)
\`\`\`

The difference from the MLP half: **attention is the only place where tokens exchange information.** The MLP works on each token separately. So after this line, the token "sat" in "The cat sat" has information from "The" and "cat" in its \`x\`.

## Forward: a worked example

One token, first 4 dimensions, using the numbers from the [residual dropout note](#info:resid_dropout_notes) (training, dimension 1 dropped):

\`\`\`text
x (residual stream):      [0.20, -0.50,  1.00, 0.30]
attn(ln1(x)):             [1.30,  0,    -0.33, 0.89]   ← the update, after dropout
new x:                    [1.50, -0.50,  0.67, 1.19]
\`\`\`

The original information in \`x\` is kept; attention only adds a correction. In dimension 1 the update was dropped, and \`x\`'s −0.50 passes through unchanged.

## Backward: at the "+", the gradient is copied to both sides

Write the line as \`new_x = old_x + branch\`, where \`branch = attn(ln1(old_x))\`.

- If \`old_x\` increases by 0.001, \`new_x\` increases by 0.001.
- If \`branch\` increases by 0.001, \`new_x\` also increases by 0.001.

So the local derivative of "+" is **1 for both sides**, and the gradient arriving at \`new_x\` is passed to both sides unchanged. Suppose the gradient arriving from later layers is \`g = [0.3, −0.1, 0.2, 0.05]\`:

\`\`\`text
      ┌──→ straight to old_x:  [0.3, -0.1, 0.2, 0.05]   ← unchanged, × 1
g  ───┤
      └──→ into the branch:    [0.3, -0.1, 0.2, 0.05]   ← also unchanged; then back through
                                                           dropout, proj, attention, ln1
\`\`\`

The copy that enters the branch passes back through attention and ln1 and arrives at \`old_x\` as well; call it \`g_branch\`. So, by the "sum over paths" rule (see the [worked backward example](#hover:gelu_backprop_notes)):

\`\`\`text
gradient of old_x = g  (straight path)  +  g_branch  (path through attention)
\`\`\`

One difference from the MLP: attention mixed tokens in the forward pass, so in the backward pass the branch's gradient also flows back to the **earlier tokens** it read from. The gradient of token "cat"'s \`x\` includes a share from "sat"'s update, because "sat" took information from "cat".

## Why the residual matters: a gradient highway

GPT-2 has 24 lines of the form \`x = x + branch\`.

- **With residuals:** along the straight paths of all 24 "+", the gradient from the loss reaches the embeddings **unchanged**. Each branch adds its own contribution on top, but never blocks the straight path.
- **Without residuals** (\`x = attn(ln1(x))\`): the gradient would have to pass through all 24 sublayers one after another, multiplied by each one's local derivative. Suppose each sublayer happened to halve it on average: after 24 sublayers it would be 0.5²⁴ ≈ 0.00000006, and the first layers would learn almost nothing (**vanishing gradients**). If each doubled it, it would explode.

A related benefit at the start of training: the weights are small, so the update is close to 0 and \`new_x ≈ old_x\`. Every block starts close to "do nothing", and the deep stack trains like a shallow one until the blocks learn useful updates.

## Why LayerNorm goes inside the branch (pre-LN)

There are two ways to combine LayerNorm with the residual:

\`\`\`text
pre-LN  (GPT-2):                  x = x + attn(LN(x))     ← LN only on the branch
post-LN (original Transformer):   x = LN(x + attn(x))     ← LN after the add, on the main path
\`\`\`

- **Pre-LN:** the straight path \`x → x + ...\` contains only the "+", so the gradient along it is multiplied by exactly 1 in every block.
- **Post-LN:** the straight path goes through LN after every add, so the gradient along it is multiplied by LN's local derivative 24 times. These factors can compound and shrink or grow the gradient. Post-LN models are known to need a careful learning-rate warmup to train at all, and become hard to train when deep.

GPT-2 (2019) switched to pre-LN, and adds one extra LayerNorm, \`ln_f\`, after the last block, because with pre-LN the stream itself is never normalized on the main path. For the same argument written with Jacobians, see [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes).

> 🆕 Pre-LN is now the standard. Many recent models (e.g. LLaMA) keep the same layout but use RMSNorm instead of LayerNorm: it only rescales and does not subtract the mean, which is a little cheaper.

## Summary

| | Forward | Backward |
| --- | --- | --- |
| Straight path (\`x\`) | information passes through complete | gradient passes back unchanged |
| Branch (\`attn(ln1(x))\`) | adds an update read from earlier tokens | adds its gradient, also to earlier tokens |
`,
    },
    zh: {
      title: "Attention 残差",
      md: `
**代码：** \`x = x + self.attn(self.ln1(x))\`

**这一行是 Transformer block 的前半段：先把残差流归一化，让 attention 根据每个 token 和它前面的 token 算出一个更新量，再把更新量加回去。** 后半段 \`x = x + self.mlp(self.ln2(x))\` 结构相同（见 [MLP 残差](#info:mlp_residual_notes)）。

## 位置

\`\`\`python
def forward(self, x):                    # x: (B, T, C)，残差流
    x = x + self.attn(self.ln1(x))       # 前半段：attention  ← 这一行
    x = x + self.mlp(self.ln2(x))        # 后半段：MLP
    return x
\`\`\`

这里的 \`x\` 是**残差流**：每个 token 一个 768 维向量。在第 1 个 block 里，它就是 token embedding + 位置 embedding；之后每个子层（12 个 block × 2 = 24 个）都往上加自己的更新量。可以把它想成每个 token 的一本笔记：embedding 写下初稿，之后每个子层补充一点内容。

## 三个步骤

1. **\`self.ln1(x)\`，先归一化（pre-LN）。** [LayerNorm](#info:layernorm) 把每个 token 的 768 个数调整到均值约 0、方差约 1，让 attention 每次看到的输入尺度都差不多，q·k 分数和 softmax 也就保持在稳定的范围里。它产生的是一个归一化的**副本**，\`x\` 本身不变。
2. **\`self.attn(...)\`，算更新量。** 因果自注意力让每个 token 从自己和它前面的 token 那里取信息（因为 causal mask，取不到后面的），返回形状为 $(B, T, C)$ 的更新量。最后两步是输出投影和 [residual dropout](#info:resid_dropout_notes)。
3. **\`x + ...\`，加回去。** 更新量逐个数地加到残差流上。

形状必须一致才能相加：

\`\`\`text
x                 (B, T, C)
self.ln1(x)       (B, T, C)
self.attn(...)    (B, T, C)
x + ...           (B, T, C)
\`\`\`

和 MLP 那一半的区别：**attention 是 token 之间交换信息的唯一地方。** MLP 对每个 token 单独计算。所以这一行之后，"The cat sat" 里 "sat" 的 \`x\` 已经带上了从 "The" 和 "cat" 取来的信息。

## Forward：具体例子

一个 token，只看前 4 维，用 [residual dropout notes](#info:resid_dropout_notes) 里的数字（训练中，第 1 维被丢掉）：

\`\`\`text
x            = [0.20, -0.50,  1.00, 0.30]   ← 残差流
attn(ln1(x)) = [1.30,  0,    -0.33, 0.89]   ← 更新量，已经过 dropout
new x        = [1.50, -0.50,  0.67, 1.19]   ← 新的 x
\`\`\`

\`x\` 原来的信息都保留着，attention 只是在上面加了一个修正。第 1 维的更新量被丢掉了，\`x\` 的 −0.50 原样通过。

## Backward：梯度遇到"+"会复制成两份

把这一行写成 \`新x = 旧x + 分支\`，其中 \`分支 = attn(ln1(旧x))\`。

- \`旧x\` 增加 0.001，\`新x\` 就增加 0.001。
- \`分支\` 增加 0.001，\`新x\` 也增加 0.001。

所以"+"对两边的局部导数**都是 1**，传到 \`新x\` 的梯度会原样分给两边。假设从后面传回来的梯度是 \`g = [0.3, −0.1, 0.2, 0.05]\`：

\`\`\`text
      ┌──→ [0.3, -0.1, 0.2, 0.05]   直接给旧x：原样，乘 1
g  ───┤
      └──→ [0.3, -0.1, 0.2, 0.05]   给分支 attn(ln1)：也原样；之后再往里穿过
                                    dropout、proj、attention、ln1
\`\`\`

进入分支的那一份穿过 attention 和 ln1，最后也回到 \`旧x\` 身上，记作 \`g_分支\`。按照"所有路径的贡献加起来"（见[穿过 GELU 的 backward 例子](#hover:gelu_backprop_notes)）：

\`\`\`text
旧x 的梯度 = g（直接路径） + g_分支（穿过 attention 的路径）
\`\`\`

和 MLP 的一个区别：attention 在 forward 时混合了 token，所以 backward 时分支的梯度也会流回它读过的**前面的 token**。"cat" 的 \`x\` 的梯度里，有一部分来自 "sat" 的更新量，因为 "sat" 从 "cat" 那里取过信息。

## 为什么残差重要：梯度高速公路

GPT-2 一共有 24 个 \`x = x + 分支\`。

- **有残差：** 沿着这 24 个"+"的直接路径，从 loss 出发的梯度可以**原封不动**地一路传回 embedding。每个分支在上面加上自己的那份，但不会挡住直接路径。
- **没有残差**（\`x = attn(ln1(x))\`）：梯度必须依次穿过全部 24 个子层，每穿过一层就乘一次这一层的局部导数。假设每层平均把梯度缩小一半，24 层之后就是 0.5²⁴ ≈ 0.00000006，前面几层几乎学不到东西（**梯度消失**）。如果每层平均放大一倍，就会爆炸。

训练刚开始时还有一个相关的好处：权重很小，更新量接近 0，所以 \`新x ≈ 旧x\`。每个 block 一开始都接近"什么都不做"，整个深层网络像浅层网络一样容易训练，直到各个 block 学会有用的更新量。

## 为什么 LayerNorm 放在分支里（pre-LN）

LayerNorm 和残差有两种组合方式：

\`\`\`text
pre-LN  (GPT-2):                  x = x + attn(LN(x))     ← LN 只在分支上
post-LN (original Transformer):   x = LN(x + attn(x))     ← LN 在相加之后，位于主路径上
\`\`\`

- **pre-LN：** 直接路径 \`x → x + ...\` 上只有"+"，所以沿这条路径的梯度在每个 block 里都正好乘以 1。
- **post-LN：** 直接路径在每次相加之后都要穿过 LN，所以沿这条路径的梯度要乘 24 次 LN 的局部导数。这些因子会累积，让梯度变小或变大。post-LN 的模型通常需要仔细的学习率 warmup 才训练得起来，层数一深就很难训练。

GPT-2（2019）改用了 pre-LN，并在最后一个 block 之后多加了一个 LayerNorm \`ln_f\`，因为在 pre-LN 里，主路径上的残差流本身从来没有被归一化过。用 Jacobian 写出来的同一个论证，见 [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes)。

> 🆕 现在 pre-LN 已经是标准做法。很多新模型（比如 LLaMA）保持同样的结构，但把 LayerNorm 换成了 RMSNorm：只做缩放，不减均值，计算更省一点。

## 总结

| | Forward | Backward |
| --- | --- | --- |
| 直接路径（\`x\`） | 原来的信息完整通过 | 梯度原样传回去 |
| 分支（\`attn(ln1(x))\`） | 加上一个由本 token 和前面 token 算出的更新量 | 加上自己那份梯度，也会传给前面的 token |
`,
    },
  },

  mlp_residual_notes: {
    en: {
      title: "MLP residual",
      md: `
**Code:** \`x = x + self.mlp(self.ln2(x))\`

**This line is the second half of a Transformer block: normalize the residual stream, let the MLP compute an update for each token, and add that update back.** After it, the block is finished and \`x\` goes on to the next block.

## Where it sits

\`\`\`python
def forward(self, x):                    # x: (B, T, C), the residual stream
    x = x + self.attn(self.ln1(x))       # first half: attention
    x = x + self.mlp(self.ln2(x))        # second half: MLP  ← this line
    return x
\`\`\`

It has exactly the same shape as the [attention residual](#info:attn_residual_notes) line; only the sublayer differs.

\`x\` here is the **residual stream**: each token's 768-number vector. It starts as token embedding + position embedding, and every sublayer (12 blocks × 2 = 24 of them) adds its update to it. By the time this line runs in block 5, \`x\` already contains the embedding plus the updates of 4 full blocks and of block 5's attention.

## The three steps

1. **\`self.ln2(x)\`, normalize first (pre-LN).** [LayerNorm](#info:layernorm) rescales each token's 768 numbers to roughly mean 0 and variance 1, so the MLP always sees inputs on a similar scale. This produces a normalized **copy**; \`x\` itself is not changed.
2. **\`self.mlp(...)\`, compute the update.** For each token separately (no mixing between tokens): [fc_in](#info:mlp_fc_in_notes) 768 → 3072, [GELU](#info:gelu), [fc_out](#info:mlp_fc_out_notes) 3072 → 768, [dropout](#info:mlp_dropout_notes).
3. **\`x + ...\`, add it back.** The update is added number by number to the residual stream.

Shapes, which must match for the add:

\`\`\`text
x                 (B, T, C)
self.ln2(x)       (B, T, C)
self.mlp(...)     (B, T, C)
x + ...           (B, T, C)
\`\`\`

## Forward: a worked example

One token, first 4 dimensions, continuing the example from the [MLP dropout note](#info:mlp_dropout_notes) (training, dimension 2 dropped):

\`\`\`text
x (residual stream):  [0.20,  -0.50,  1.00, 0.30]
mlp(ln2(x)):          [0.556, -0.222, 0,    0.111]   ← the update, after dropout
new x:                [0.756, -0.722, 1.00, 0.411]
\`\`\`

The original information in \`x\` is kept; the MLP only adds a correction on top of it.

## Backward: at the "+", the gradient is copied to both sides

Write the line as \`new_x = old_x + branch\`, where \`branch = mlp(ln2(old_x))\`.

- If \`old_x\` increases by 0.001, \`new_x\` increases by 0.001.
- If \`branch\` increases by 0.001, \`new_x\` also increases by 0.001.

So the local derivative of "+" is **1 for both sides**, and the gradient arriving at \`new_x\` is passed to both sides unchanged. Suppose the gradient arriving from later layers is \`g = [0.3, −0.1, 0.2, 0.05]\`:

\`\`\`text
      ┌──→ straight to old_x:  [0.3, -0.1, 0.2, 0.05]   ← unchanged, × 1
g  ───┤
      └──→ into the branch:    [0.3, -0.1, 0.2, 0.05]   ← also unchanged; then back through
                                                           dropout, fc_out, GELU, fc_in, ln2
\`\`\`

The copy that enters the branch passes back through the MLP and ln2 (as in the [worked backward example](#hover:gelu_backprop_notes)) and finally also arrives at \`old_x\`; call it \`g_branch\`. So:

\`\`\`text
gradient of old_x = g  (straight path)  +  g_branch  (path through the MLP)
\`\`\`

This is the "sum over paths" rule: \`old_x\` affects the loss along two paths (directly into \`new_x\`, and through ln2 and the MLP), so the two gradients are added.

## Why this matters: a gradient highway

GPT-2 has 12 blocks, so 24 lines of the form \`x = x + branch\`.

- **With residuals:** along the straight paths of all 24 "+", the gradient from the loss reaches the embeddings **unchanged**. Each branch adds its own contribution on top, but never blocks the straight path.
- **Without residuals** (\`x = mlp(ln2(x))\`): the gradient would have to pass through all 24 sublayers one after another, multiplied by each one's local derivative. Suppose each sublayer happened to halve it on average: after 24 sublayers it would be 0.5²⁴ ≈ 0.00000006, and the first layers would learn almost nothing. This is the **vanishing gradient** problem. (If each one doubled it instead, it would explode.)

A related benefit shows up at the start of training: the weights are small, so the update \`mlp(ln2(x))\` is close to 0 and \`new_x ≈ old_x\`. Every block starts out close to "do nothing", and the deep stack behaves like a shallow one until the blocks learn useful updates. That makes deep networks much easier to start training.

For the same argument in Jacobian form, and why putting LayerNorm **before** the sublayer (pre-LN, as here) keeps the straight path clean while the original Transformer's post-LN does not, see [Prevents vanishing/exploding](#info:mlp_prevents_vanishing_exploding_notes).

## Summary

| | Forward | Backward |
| --- | --- | --- |
| Straight path (\`x\`) | the original information passes through complete | the gradient passes back unchanged |
| Branch (\`mlp(ln2(x))\`) | adds a per-token correction | adds its own gradient contribution |

The residual connection is what makes it possible to stack many blocks: forward, nothing is lost; backward, the gradient reaches every layer.
`,
    },
    zh: {
      title: "MLP 残差",
      md: `
**代码：** \`x = x + self.mlp(self.ln2(x))\`

**这一行是 Transformer block 的后半段：先把残差流归一化，让 MLP 给每个 token 算一个更新量，再把更新量加回去。** 这一行之后，block 就结束了，\`x\` 进入下一个 block。

## 位置

\`\`\`python
def forward(self, x):                    # x: (B, T, C)，残差流
    x = x + self.attn(self.ln1(x))       # 前半段：attention
    x = x + self.mlp(self.ln2(x))        # 后半段：MLP  ← 这一行
    return x
\`\`\`

它和 [attention 残差](#info:attn_residual_notes)那一行结构完全一样，只是子层不同。

这里的 \`x\` 是**残差流**：每个 token 一个 768 维向量。它的起点是 token embedding + 位置 embedding，之后每个子层（12 个 block × 2 = 24 个）都往上加自己的更新量。比如在第 5 个 block 运行这一行时，\`x\` 已经包含了 embedding、前 4 个 block 的全部更新量，以及第 5 个 block attention 的更新量。

## 三个步骤

1. **\`self.ln2(x)\`，先归一化（pre-LN）。** [LayerNorm](#info:layernorm) 把每个 token 的 768 个数调整到均值约 0、方差约 1，让 MLP 每次看到的输入尺度都差不多。它产生的是一个归一化的**副本**，\`x\` 本身不变。
2. **\`self.mlp(...)\`，算更新量。** 对每个 token 单独计算（token 之间不混合）：[fc_in](#info:mlp_fc_in_notes) 768 → 3072，[GELU](#info:gelu)，[fc_out](#info:mlp_fc_out_notes) 3072 → 768，[dropout](#info:mlp_dropout_notes)。
3. **\`x + ...\`，加回去。** 更新量逐个数地加到残差流上。

形状必须一致才能相加：

\`\`\`text
x                 (B, T, C)
self.ln2(x)       (B, T, C)
self.mlp(...)     (B, T, C)
x + ...           (B, T, C)
\`\`\`

## Forward：具体例子

一个 token，只看前 4 维，接着 [MLP dropout notes](#info:mlp_dropout_notes) 里的例子（训练中，第 2 维被丢掉）：

\`\`\`text
x            = [0.20,  -0.50,  1.00, 0.30]    ← 残差流
mlp(ln2(x))  = [0.556, -0.222, 0,    0.111]   ← 更新量，已经过 dropout
new x        = [0.756, -0.722, 1.00, 0.411]   ← 新的 x
\`\`\`

\`x\` 原来的信息都保留着，MLP 只是在上面加了一个修正。

## Backward：梯度遇到"+"会复制成两份

把这一行写成 \`新x = 旧x + 分支\`，其中 \`分支 = mlp(ln2(旧x))\`。

- \`旧x\` 增加 0.001，\`新x\` 就增加 0.001。
- \`分支\` 增加 0.001，\`新x\` 也增加 0.001。

所以"+"对两边的局部导数**都是 1**，传到 \`新x\` 的梯度会原样分给两边。假设从后面传回来的梯度是 \`g = [0.3, −0.1, 0.2, 0.05]\`：

\`\`\`text
      ┌──→ [0.3, -0.1, 0.2, 0.05]   直接给旧x：原样，乘 1
g  ───┤
      └──→ [0.3, -0.1, 0.2, 0.05]   给分支 mlp(ln2)：也原样；之后再往里穿过
                                    dropout、fc_out、GELU、fc_in、ln2
\`\`\`

进入分支的那一份穿过 MLP 和 ln2（过程和[穿过 GELU 的 backward 例子](#hover:gelu_backprop_notes)一样），最后也回到 \`旧x\` 身上，记作 \`g_分支\`。所以：

\`\`\`text
旧x 的梯度 = g（直接路径） + g_分支（穿过 MLP 的路径）
\`\`\`

这就是"所有路径的贡献加起来"：\`旧x\` 通过两条路径影响 loss（直接进入 \`新x\`，以及经过 ln2 和 MLP），所以两份梯度相加。

## 为什么重要：梯度高速公路

GPT-2 有 12 个 block，也就是 24 个 \`x = x + 分支\`。

- **有残差：** 沿着这 24 个"+"的直接路径，从 loss 出发的梯度可以**原封不动**地一路传回 embedding。每个分支在上面加上自己的那份，但不会挡住直接路径。
- **没有残差**（\`x = mlp(ln2(x))\`）：梯度必须依次穿过全部 24 个子层，每穿过一层就乘一次这一层的局部导数。假设每层平均把梯度缩小一半，24 层之后就是 0.5²⁴ ≈ 0.00000006，前面几层几乎学不到东西。这就是**梯度消失**。（如果每层平均放大一倍，就会梯度爆炸。）

训练刚开始时还有一个相关的好处：权重很小，更新量 \`mlp(ln2(x))\` 接近 0，所以 \`新x ≈ 旧x\`。每个 block 一开始都接近"什么都不做"，整个深层网络表现得像一个浅层网络，直到各个 block 学会有用的更新量。这让深层网络的训练更容易起步。

同样的论证用 Jacobian 的形式写出来，以及为什么把 LayerNorm 放在子层**之前**（pre-LN，就是这里的写法）能让直接路径保持干净，而原始 Transformer 的 post-LN 做不到，见 [Prevents vanishing/exploding](#info:mlp_prevents_vanishing_exploding_notes)。

## 总结

| | Forward | Backward |
| --- | --- | --- |
| 直接路径（\`x\`） | 原来的信息完整通过 | 梯度原样传回去 |
| 分支（\`mlp(ln2(x))\`） | 加上一个逐 token 的修正 | 加上自己那份梯度 |

残差连接是能把很多 block 堆起来的关键：forward 时信息不丢，backward 时梯度能到达每一层。
`,
    },
  },

  prevents_vanishing_exploding_notes: {
    en: {
      title: "Prevents vanishing/exploding",
      md: `
**Two concrete toy examples that show why pre-LN keeps gradients from vanishing/exploding in deep stacks.**

## Example 1: "safe at init" path (identity highway)

Setup (early training): attention weights are small, so the sublayer output is near zero:

$$
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0
$$

- **Pre-LN block:** $x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l$.

  Backward: $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\;\\Rightarrow\\;$ the gradient passes nearly unchanged.

- **Post-LN (contrast):** $x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l)$,

  so $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)$, and repeated products can shrink or grow gradients.

## Example 2: Jacobian picture with residual vs. no residual

Consider one block as a function $F(x) = x + f(\\mathrm{LN}(x))$.

### Jacobian (pre-LN)

$$
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x)
$$

If $\\|J_f\\|$ is small and $J_{\\mathrm{LN}}$ is bounded, then $J_F \\approx I + \\epsilon$ with small $\\epsilon$. Across $L$ layers, $\\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)$ stays near $I$ $\\Rightarrow$ no vanishing/exploding.

### No residual

$$
G(x) = f(\\mathrm{LN}(x)), \\qquad J_G = J_f\\,J_{\\mathrm{LN}}
$$

Across depth, $\\prod_{\\ell=1}^{L} J_G(x_\\ell)$ can vanish if $\\|J_G\\|<1$ or explode if $\\|J_G\\|>1$.

Here $\\|J_G\\|$ is the [spectral norm ↗](https://mathworld.wolfram.com/SpectralNorm.html).

## Why pre-LN helps more than post-LN

- **Pre-LN:** $x_{l+1}=x_l + f(\\mathrm{LN}(x_l))$ keeps an identity path in both forward and backward.
- **Post-LN:** $x_{l+1}=\\mathrm{LN}(x_l+f(x_l))$. Even the skip connection is passed through $J_{\\mathrm{LN}}$ every layer, so the identity gets scaled repeatedly, which can accumulate into vanishing/exploding.

## Bottom line

> Pre-LN residual attention provides an identity gradient highway and bounds the non-identity part, preventing vanishing/exploding in deep stacks.
    `,
    },
    zh: {
      title: "防止梯度消失/爆炸",
      md: `
**两个具体的玩具例子，说明为什么 pre-LN 能防止深层网络里的梯度消失和爆炸。**

## 例子 1：初始化时的"安全"路径（恒等高速公路）

设定（训练早期）：attention 的权重很小，所以子层的输出接近 0：

$$
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0
$$

- **Pre-LN block：** $x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l$。

  Backward：$\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\;\\Rightarrow\\;$ 梯度几乎原样通过。

- **Post-LN（对比）：** $x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l)$，

  所以 $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)$，反复相乘可能让梯度缩小或放大。

## 例子 2：有残差和没有残差时的 Jacobian

把一个 block 看成函数 $F(x) = x + f(\\mathrm{LN}(x))$。

### Jacobian（pre-LN）

$$
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x)
$$

如果 $\\|J_f\\|$ 很小、$J_{\\mathrm{LN}}$ 有界，那么 $J_F \\approx I + \\epsilon$，$\\epsilon$ 很小。经过 $L$ 层，$\\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)$ 仍然接近 $I$ $\\Rightarrow$ 不会消失也不会爆炸。

### 没有残差

$$
G(x) = f(\\mathrm{LN}(x)), \\qquad J_G = J_f\\,J_{\\mathrm{LN}}
$$

随着层数加深，如果 $\\|J_G\\|<1$，$\\prod_{\\ell=1}^{L} J_G(x_\\ell)$ 可能消失；如果 $\\|J_G\\|>1$，可能爆炸。

这里的 $\\|J_G\\|$ 是[谱范数 ↗](https://mathworld.wolfram.com/SpectralNorm.html)。

## 为什么 pre-LN 比 post-LN 更有帮助

- **Pre-LN：** $x_{l+1}=x_l + f(\\mathrm{LN}(x_l))$，forward 和 backward 都保留了一条恒等路径。
- **Post-LN：** $x_{l+1}=\\mathrm{LN}(x_l+f(x_l))$。连跳跃连接每一层都要穿过 $J_{\\mathrm{LN}}$，恒等路径被反复缩放，累积起来就可能导致梯度消失或爆炸。

## 结论

> Pre-LN 的残差 attention 提供了一条恒等的梯度高速公路，同时限制了非恒等部分的大小，从而防止深层网络里的梯度消失和爆炸。
`,
    },
  },

  mlp_prevents_vanishing_exploding_notes: {
    en: {
      title: "Prevents vanishing/exploding",
      md: `
**Two concrete toy examples that show why pre-LN keeps gradients from vanishing/exploding in deep stacks.**

## Example 1: "safe at init" path (identity highway)

Setup (early training): attention weights are small, so the sublayer output is near zero:

$$
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0
$$

- **Pre-LN block:** $x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l$.

  Backward: $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\;\\Rightarrow\\;$ the gradient passes nearly unchanged.

- **Post-LN (contrast):** $x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l)$,

  so $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)$, and repeated products can shrink or grow gradients.

## Example 2: Jacobian picture with residual vs. no residual

Consider one block as a function $F(x) = x + f(\\mathrm{LN}(x))$.

### Jacobian (pre-LN)

$$
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x)
$$

If $\\|J_f\\|$ is small and $J_{\\mathrm{LN}}$ is bounded, then $J_F \\approx I + \\epsilon$ with small $\\epsilon$. Across $L$ layers, $\\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)$ stays near $I$ $\\Rightarrow$ no vanishing/exploding.

### No residual

$$
G(x) = f(\\mathrm{LN}(x)), \\qquad J_G = J_f\\,J_{\\mathrm{LN}}
$$

Across depth, $\\prod_{\\ell=1}^{L} J_G(x_\\ell)$ can vanish if $\\|J_G\\|<1$ or explode if $\\|J_G\\|>1$.

Here $\\|J_G\\|$ is the [spectral norm ↗](https://mathworld.wolfram.com/SpectralNorm.html).

## Why pre-LN helps more than post-LN

- **Pre-LN:** $x_{l+1}=x_l + f(\\mathrm{LN}(x_l))$ keeps an identity path in both forward and backward.
- **Post-LN:** $x_{l+1}=\\mathrm{LN}(x_l+f(x_l))$. Even the skip connection is passed through $J_{\\mathrm{LN}}$ every layer, so the identity gets scaled repeatedly, which can accumulate into vanishing/exploding.

## Bottom line

> Pre-LN residual attention provides an identity gradient highway and bounds the non-identity part, preventing vanishing/exploding in deep stacks.
    `,
    },
    zh: {
      title: "防止梯度消失/爆炸",
      md: `
**两个具体的玩具例子，说明为什么 pre-LN 能防止深层网络里的梯度消失和爆炸。**

## 例子 1：初始化时的"安全"路径（恒等高速公路）

设定（训练早期）：attention 的权重很小，所以子层的输出接近 0：

$$
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0
$$

- **Pre-LN block：** $x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l$。

  Backward：$\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\;\\Rightarrow\\;$ 梯度几乎原样通过。

- **Post-LN（对比）：** $x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l)$，

  所以 $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)$，反复相乘可能让梯度缩小或放大。

## 例子 2：有残差和没有残差时的 Jacobian

把一个 block 看成函数 $F(x) = x + f(\\mathrm{LN}(x))$。

### Jacobian（pre-LN）

$$
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x)
$$

如果 $\\|J_f\\|$ 很小、$J_{\\mathrm{LN}}$ 有界，那么 $J_F \\approx I + \\epsilon$，$\\epsilon$ 很小。经过 $L$ 层，$\\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)$ 仍然接近 $I$ $\\Rightarrow$ 不会消失也不会爆炸。

### 没有残差

$$
G(x) = f(\\mathrm{LN}(x)), \\qquad J_G = J_f\\,J_{\\mathrm{LN}}
$$

随着层数加深，如果 $\\|J_G\\|<1$，$\\prod_{\\ell=1}^{L} J_G(x_\\ell)$ 可能消失；如果 $\\|J_G\\|>1$，可能爆炸。

这里的 $\\|J_G\\|$ 是[谱范数 ↗](https://mathworld.wolfram.com/SpectralNorm.html)。

## 为什么 pre-LN 比 post-LN 更有帮助

- **Pre-LN：** $x_{l+1}=x_l + f(\\mathrm{LN}(x_l))$，forward 和 backward 都保留了一条恒等路径。
- **Post-LN：** $x_{l+1}=\\mathrm{LN}(x_l+f(x_l))$。连跳跃连接每一层都要穿过 $J_{\\mathrm{LN}}$，恒等路径被反复缩放，累积起来就可能导致梯度消失或爆炸。

## 结论

> Pre-LN 的残差 attention 提供了一条恒等的梯度高速公路，同时限制了非恒等部分的大小，从而防止深层网络里的梯度消失和爆炸。
`,
    },
  },

  variance_init_notes: {
    en: {
      title: "Variance at initialization",
      md: `
**Several design choices in GPT-2, such as the $1/\\sqrt{hs}$ in attention and the scaling of residual-writing weights, exist to keep variances under control at initialization.** This note derives every variance involved, from LayerNorm's output to the attention scores and the residual stream. The numbers are for this site's code (PyTorch default initialization) and match measurements in PyTorch.

## 0. What variance is

The **mean** $E[X]$ of a random number $X$ is its average. The **variance** measures how far it typically is from the mean: the average squared distance.

\`\`\`text
Var(X) = E[(X − E[X])²]
       = E[X²] − (E[X])²        ← equivalent, often more convenient
\`\`\`

When the mean is 0, this is simply **Var(X) = E[X²]**, "the average of the square". That form is used throughout.

The **standard deviation** is √variance: the number's typical size. Variance 64 means standard deviation 8, i.e. values around ±8.

**What is random here:** at initialization the weights are drawn at random, so q and k are random too. All means and variances below are averages over that random initialization.

## 1. Four rules

**Rule 1: multiplying by a constant c multiplies the variance by c²**

\`\`\`text
Var(cX) = c² × Var(X)
e.g. if X has variance 64, then X / 8 has variance 64 / 8² = 1
\`\`\`

**Rule 2: for independent random numbers, variances add**

\`\`\`text
Var(X + Y) = Var(X) + Var(Y)      (X, Y independent)
\`\`\`

Variances add, not standard deviations, because random signs partly cancel. With two numbers that are each +1 or −1 with probability ½:

\`\`\`text
a     b     a + b
+1    +1     +2
+1    −1      0
−1    +1      0
−1    −1     −2

each case has probability 1/4

a alone:   E[a] = 0,  a² is always 1  →  Var(a) = 1   (same for b)

mean:      E[a + b] = (2 + 0 + 0 − 2) / 4 = 0

variance:  Var(a + b) = E[(a + b)²] − (E[a + b])²
                      = E[(a + b)²] − 0²             ← because E[a + b] = 0
                      = (2² + 0² + 0² + (−2)²) / 4
                      = (4 + 0 + 0 + 4) / 4
                      = 2                            ← exactly Var(a) + Var(b) = 1 + 1

std:       √2 ≈ 1.41                                 ← not 1 + 1 = 2
\`\`\`

So the sum of $n$ such terms is typically about $\\sqrt{n}$ in size, not $n$.

**Rule 3: product of two independent numbers, both with mean 0**

\`\`\`text
Var(X × Y) = Var(X) × Var(Y)
\`\`\`

Why: for independent $X$ and $Y$, the average of a product is the product of the averages: $E[XY] = E[X] \\times E[Y]$. The same holds for $X^2$ and $Y^2$, which are independent too. So:

\`\`\`text
mean:      E[XY] = E[X] × E[Y] = 0 × 0 = 0      ← both have mean 0

variance:  Var(XY) = E[(XY)²] − (E[XY])²        ← definition of variance
                   = E[X²Y²] − 0²               ← because E[XY] = 0
                   = E[X²] × E[Y²]              ← X² and Y² are independent
                   = Var(X) × Var(Y)            ← mean 0, so E[X²] = Var(X) and E[Y²] = Var(Y)
\`\`\`

**Rule 4: the uniform distribution U(−a, a) has variance a²/3**

$U(-a, a)$ means "any value between $-a$ and $a$, all equally likely". Its mean is 0 (it is symmetric), so its variance is the average of $x^2$ over that range:

$$
\\mathrm{Var} = \\frac{1}{2a}\\int_{-a}^{a} x^2\\,dx = \\frac{1}{2a} \\cdot \\frac{2a^3}{3} = \\frac{a^2}{3}
$$

## 2. The starting point: LayerNorm's output h

Attention's input is \`ln1(x)\`, with shape $(B,\\ T,\\ C) = (B,\\ T,\\ 768)$: one 768-dim vector per token. \`c_attn\` turns each token's vector into that token's q, k and v separately, so it is enough to look at **one token at a time**. Below, **h** means one token's vector, \`ln1(x)[b, t, :]\` (768 numbers), and $h[j]$ is its $j$-th component. At initialization ($\\gamma = 1$, $\\beta = 0$), [LayerNorm](#info:layernorm) makes each token's 768 components have mean 0 and variance 1. With mean 0, variance is the average of the squares, so:

\`\`\`text
(h[0]² + h[1]² + ... + h[767]²) / 768 = 1
i.e.  h[0]² + h[1]² + ... + h[767]² = 768
\`\`\`

This is exact (up to a tiny eps), enforced by LayerNorm for **every** token. A 4-dim example:

\`\`\`text
before LayerNorm:       [2, 4, 6, 8]
subtract the mean 5:    [−3, −1, 1, 3]
variance:               (9 + 1 + 1 + 9) / 4 = 5
divide by √5 ≈ 2.236:   h = [−1.34, −0.45, 0.45, 1.34]

average of h[j]²:       (1.8 + 0.2 + 0.2 + 1.8) / 4 = 1     ✓
\`\`\`

## 3. The variance of the weights

This site's code uses PyTorch's default initialization: the weight of \`nn.Linear(768, …)\` is drawn from $U(-1/\\sqrt{768},\\ 1/\\sqrt{768})$, where 768 is the layer's number of inputs (see "PyTorch default initialization" in [Attention linear](#info:attention_linear_notes) for where this range comes from). By rule 4, with $a = 1/\\sqrt{768}$:

\`\`\`text
Var(W) = (1/√768)² / 3 = 1 / (3 × 768) ≈ 0.000434
\`\`\`

The bias is drawn from the same range, so $\\mathrm{Var}(b) = 1/(3 \\times 768)$ as well. (For comparison, original GPT-2 uses $N(0,\\ 0.02)$: $\\mathrm{Var}(W) = 0.02^2 = 0.0004$.)

## 4. One component of q: variance about 1/3

Fix one token. Its h is now 768 fixed numbers; only the weights are random. Component $i$ of that token's q (one of its 768 components) is:

\`\`\`text
q[i] = W[i,0] h[0] + W[i,1] h[1] + ... + W[i,767] h[767] + b[i]
\`\`\`

**Mean 0:**

\`\`\`text
E[q[i]] = E[W[i,0] h[0] + ... + W[i,767] h[767] + b[i]]
        = E[W[i,0]] h[0] + ... + E[W[i,767]] h[767] + E[b[i]]   ← (1) and (2)
        = 0 × h[0] + ... + 0 × h[767] + 0                       ← (3)
        = 0
\`\`\`

The three facts used:

1. **The average of a sum is the sum of the averages:** $E[X + Y] = E[X] + E[Y]$. This always holds, independent or not.
2. **A constant factor comes out of the average:** $E[cX] = c \\times E[X]$. Here $h[j]$ is a constant, because the token is fixed.
3. **Each weight and the bias have mean 0:** they are drawn from $U(-a,\\ a)$, which is symmetric around 0, so $E[W] = 0$ and $E[b] = 0$.

Intuitively: each term $W[i,j]\\,h[j]$ is positive or negative with equal chance, whatever $h[j]$ is, so it averages to 0.

**Variance:** each term is a random $W$ times the constant $h[j]$, so by rule 1 its variance is $h[j]^2 \\times \\mathrm{Var}(W)$. The 768 terms use independent weights, so by rule 2 they add:

\`\`\`text
Var(q[i]) = Var(W) × (h[0]² + h[1]² + ... + h[767]²) + Var(b)
          = Var(W) × 768 + Var(b)          ← the sum of squares from section 2
          = 1/(3 × 768) × 768 + 0.0004
          ≈ 1/3 ≈ 0.334                    ← measured: 0.334 ✓
\`\`\`

k and v are the same, about 1/3 each; they use other, independently drawn parts of \`c_attn\`'s weight. Thanks to LayerNorm, this holds for **any** input token.

## 5. One term q[i] × k[i]: variance 1/9

From here on, q and k are **one head's 64-dim pieces** (see [Query View](#info:query_view_notes)): q comes from the query token, k from the key token, and $i$ runs from 0 to 63. Each component still has mean 0 and variance about 1/3, by section 4.

q and k use different, independently drawn weights, so at initialization they are independent. By rule 3:

\`\`\`text
Var(q[i] × k[i]) = Var(q[i]) × Var(k[i]) = 1/3 × 1/3 = 1/9
\`\`\`

## 6. The dot product q · k: a sum of hs terms

With head size $hs = 64$, the score between the two tokens is:

\`\`\`text
q · k = q[0]k[0] + q[1]k[1] + ... + q[63]k[63]
\`\`\`

Different $i$ use different rows of the weights, drawn independently, so the 64 terms are independent. Each has variance 1/9, and by rule 2:

\`\`\`text
Var(q · k) = 64 × 1/9 ≈ 7.1
std        = √7.1 ≈ 2.67                  ← measured: 2.67 ✓
\`\`\`

**The Transformer paper's assumption:** if each component of q and k had variance 1, each term would have variance 1, and the sum variance 64 (standard deviation 8). That is where "variance about 64" comes from.

Either way, **the dot product's variance is proportional to $hs$**: doubling the head size doubles it.

## 7. Dividing by √hs: the variance no longer depends on hs

### The problem: the variance grows with hs

Section 6 in general form: $q \\cdot k$ is a sum of $hs$ independent terms, and by rule 3 each term $q[i]\\,k[i]$ has variance $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i])$. By rule 2:

\`\`\`text
Var(q · k) = hs × Var(q[i]) × Var(k[i])
\`\`\`

With this site's initialization ($\\mathrm{Var}(q[i]) = \\mathrm{Var}(k[i]) = 1/3$, so each term has variance 1/9):

\`\`\`text
hs     Var(q · k)          std
32     32 × 1/9  = 3.56    1.89
64     64 × 1/9  = 7.11    2.67     ← GPT-2 small
128    128 × 1/9 = 14.22   3.77
\`\`\`

The bigger the head size, the bigger the scores, even though q and k themselves have not changed.

### The fix: divide by √hs

The code computes \`(q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)\`, i.e. every score is multiplied by $c = 1/\\sqrt{hs}$. By rule 1, the variance is multiplied by $c^2 = 1/hs$:

\`\`\`text
Var(q · k / √hs) = (1/√hs)² × Var(q · k)                  ← rule 1 with c = 1/√hs
                 = (1/hs) × hs × Var(q[i]) × Var(k[i])     ← Var(q · k) from above
                 = Var(q[i]) × Var(k[i])                   ← hs cancels
\`\`\`

The result no longer contains $hs$: for every head size the scores have the same variance.

\`\`\`text
hs     before dividing    after dividing by √hs
32     3.56               3.56 / 32  = 1/9
64     7.11               7.11 / 64  = 1/9      std ≈ 0.33   ← measured: 0.334 ✓
128    14.22              14.22 / 128 = 1/9
\`\`\`

Under the Transformer paper's assumption (variance 1 per component), the same steps give $1 \\times 1 = 1$: standard deviation 1 for every head size.

### Why √hs and not hs

Variance scales by the **square** of the factor (rule 1). To divide the variance by $hs$, the scores are divided by $\\sqrt{hs}$. Dividing the scores by $hs$ itself would divide the variance by $hs^2$, leaving $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i]) / hs$: now the scores would **shrink** as $hs$ grows, the opposite problem. Only $\\sqrt{hs}$ cancels $hs$ exactly.

### Why scores must not be too large

softmax is very sensitive to the **difference** between scores. Take two scores. With standard deviation 8 (the paper's assumption, before dividing), a difference of 16 is common:

\`\`\`text
scores [16, 0]  →  softmax ≈ [0.9999999, 0.0000001]
scores [2, 0]   →  softmax ≈ [0.88, 0.12]              ← the same scores divided by 8
\`\`\`

With $[16, 0]$ essentially all the weight goes to one token, and it barely moves when the scores change: for two scores, the output's sensitivity to their difference is $p(1-p)$, which is about $0.0000001$ here, versus about $0.105$ for $[2, 0]$. Tiny sensitivity means tiny gradients, so the model can hardly learn where to attend. See [Q @ K](#info:q_k_product_notes) for more.

## 8. The residual stream: 24 additions

The dot product sums 64 terms; the residual stream sums 24 layer outputs, under the same rule 2:

\`\`\`text
x_final = x_embedding + update_1 + update_2 + ... + update_24
\`\`\`

GPT-2 small has 12 blocks, each adding once from attention and once from the MLP: 24 additions. If each update has variance about $v$ and they are roughly independent:

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v      ← grows with depth
\`\`\`

**Scaling the residual-writing weights:** at initialization, the weights of [\`proj\`](#info:attention_output_linear_notes) (and the MLP's \`fc_out\`) are multiplied by $1/\\sqrt{24}$. An update is (approximately) linear in these weights: they are the last matrix the update passes through, so multiplying them by $c$ multiplies the update by $c$ (GPT-2 initializes biases to 0, so the bias does not get in the way). Each update therefore shrinks by $1/\\sqrt{24}$, and by rule 1 its variance becomes $v/24$:

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v/24
             = Var(x_embedding) + v           ← no longer grows with depth
\`\`\`

That is where the square root in $1/\\sqrt{N}$ comes from: variance scales by $c^2$, so to make $N$ variances add up to one, each update is scaled by $1/\\sqrt{N}$.

The GPT-2 paper and nanoGPT do this scaling; the code on this site does not (see [Attention output projection](#info:attention_output_linear_notes)).

## Summary

\`\`\`text
h = one token's LayerNorm out    average of h[j]² = 1 (exact)
W at initialization              Var(W) = 1/(3·768)
q[i] = 768 terms W·h             Var ≈ 768 × 1/(3·768) = 1/3
q[i]·k[i]                        Var = 1/3 × 1/3 = 1/9
q·k = 64 terms                   Var = 64/9 ≈ 7.1, std 2.67
divide by √64 = 8                Var = 1/9, std 0.33 (independent of hs)

residual stream = 24 updates     variance grows with depth
proj / fc_out weights × 1/√24    each update's variance ÷ 24; the total stops growing
\`\`\`

> All of this holds **only at initialization**, when the weights are random, symmetric and independent. Once training starts, the weights change and so do these variances. The point of these choices is to give training a starting point on a sensible scale.
`,
    },
    zh: {
      title: "初始化时的方差",
      md: `
**GPT-2 里有几处设计，比如 attention 里的 $1/\\sqrt{hs}$、对 residual-writing weights 的缩放，都是为了在初始化时把方差控制住。** 这篇笔记把用到的每一个方差都推导一遍：从 LayerNorm 的输出开始，一直到 attention 分数和残差流。下面的数字对应本网站的代码（PyTorch 默认初始化），并且都和在 PyTorch 里实测的结果一致。

## 0. 方差是什么

一个随机数 $X$ 的**均值** $E[X]$，就是它平均起来是多少。**方差**衡量它偏离均值的程度，是"离均值的距离的平方"的平均：

\`\`\`text
Var(X) = E[(X − E[X])²]
       = E[X²] − (E[X])²        ← equivalent, often more convenient
\`\`\`

均值是 0 时，这个式子简化成 **Var(X) = E[X²]**，也就是"平方的平均"。后面一直在用这个形式。

**标准差** = √方差，表示这个数"典型的大小"。比如方差是 64，标准差是 8，意思是这个数通常在 ±8 左右。

**"随机"指的是什么：** 初始化时权重是随机抽的，所以 q、k 也是随机的。下面说的均值和方差，都是对"随机初始化"取的平均。

## 1. 四条规则

**规则 1：乘以常数 c，方差乘以 c²**

\`\`\`text
Var(cX) = c² × Var(X)
e.g. if X has variance 64, then X / 8 has variance 64 / 8² = 1
\`\`\`

**规则 2：独立的随机数相加，方差直接相加**

\`\`\`text
Var(X + Y) = Var(X) + Var(Y)      (X, Y independent)
\`\`\`

是方差相加，不是标准差相加，因为正负号会部分抵消。用两个各自以 ½ 的概率取 +1 或 −1 的数来看：

\`\`\`text
a     b     a + b
+1    +1     +2
+1    −1      0
−1    +1      0
−1    −1     −2

each case has probability 1/4

a alone:   E[a] = 0,  a² is always 1  →  Var(a) = 1   (same for b)

mean:      E[a + b] = (2 + 0 + 0 − 2) / 4 = 0

variance:  Var(a + b) = E[(a + b)²] − (E[a + b])²
                      = E[(a + b)²] − 0²             ← because E[a + b] = 0
                      = (2² + 0² + 0² + (−2)²) / 4
                      = (4 + 0 + 0 + 4) / 4
                      = 2                            ← exactly Var(a) + Var(b) = 1 + 1

std:       √2 ≈ 1.41                                 ← not 1 + 1 = 2
\`\`\`

所以 $n$ 个这样的数相加，结果的典型大小是 $\\sqrt{n}$，而不是 $n$。

**规则 3：两个独立的数相乘，两个的均值都为 0**

\`\`\`text
Var(X × Y) = Var(X) × Var(Y)
\`\`\`

原因：$X$ 和 $Y$ 独立时，乘积的平均等于平均的乘积：$E[XY] = E[X] \\times E[Y]$。$X^2$ 和 $Y^2$ 也是独立的，同样成立。所以：

\`\`\`text
mean:      E[XY] = E[X] × E[Y] = 0 × 0 = 0      ← both have mean 0

variance:  Var(XY) = E[(XY)²] − (E[XY])²        ← definition of variance
                   = E[X²Y²] − 0²               ← because E[XY] = 0
                   = E[X²] × E[Y²]              ← X² and Y² are independent
                   = Var(X) × Var(Y)            ← mean 0, so E[X²] = Var(X) and E[Y²] = Var(Y)
\`\`\`

**规则 4：均匀分布 U(−a, a) 的方差是 a²/3**

$U(-a, a)$ 的意思是"$-a$ 到 $a$ 之间的任何值，可能性都一样"。它是对称的，均值为 0，所以方差就是 $x^2$ 在这个范围上的平均：

$$
\\mathrm{Var} = \\frac{1}{2a}\\int_{-a}^{a} x^2\\,dx = \\frac{1}{2a} \\cdot \\frac{2a^3}{3} = \\frac{a^2}{3}
$$

## 2. 起点：LayerNorm 的输出 h

attention 的输入是 \`ln1(x)\`，形状是 $(B,\\ T,\\ C) = (B,\\ T,\\ 768)$：每个 token 一个 768 维向量。\`c_attn\` 对每个 token 的向量分别算出这个 token 的 q、k、v，所以**一次只看一个 token** 就够了。下面的 **h** 指一个 token 的向量，也就是 \`ln1(x)[b, t, :]\`（768 个数），$h[j]$ 是它的第 $j$ 个分量。初始化时（$\\gamma = 1$，$\\beta = 0$），[LayerNorm](#info:layernorm) 让每个 token 的 768 个分量均值为 0、方差为 1。均值为 0 时，方差就是平方的平均，所以：

\`\`\`text
(h[0]² + h[1]² + ... + h[767]²) / 768 = 1
i.e.  h[0]² + h[1]² + ... + h[767]² = 768
\`\`\`

这是精确的（只差一个很小的 eps），并且 LayerNorm 对**每一个** token 都保证这一点。一个 4 维的例子：

\`\`\`text
before LayerNorm:       [2, 4, 6, 8]
subtract the mean 5:    [−3, −1, 1, 3]
variance:               (9 + 1 + 1 + 9) / 4 = 5
divide by √5 ≈ 2.236:   h = [−1.34, −0.45, 0.45, 1.34]

average of h[j]²:       (1.8 + 0.2 + 0.2 + 1.8) / 4 = 1     ✓
\`\`\`

## 3. 权重的方差

本网站的代码用 PyTorch 默认初始化：\`nn.Linear(768, …)\` 的权重从 $U(-1/\\sqrt{768},\\ 1/\\sqrt{768})$ 里抽，768 是这一层的输入维数（这个范围是怎么来的，见 [Attention linear](#info:attention_linear_notes) 里的 "PyTorch default initialization" 一节）。按规则 4，取 $a = 1/\\sqrt{768}$：

\`\`\`text
Var(W) = (1/√768)² / 3 = 1 / (3 × 768) ≈ 0.000434
\`\`\`

bias 也从同一个范围里抽，所以 $\\mathrm{Var}(b)$ 也是 $1/(3 \\times 768)$。（作为对照，GPT-2 原版用 $N(0,\\ 0.02)$：$\\mathrm{Var}(W) = 0.02^2 = 0.0004$。）

## 4. q 的一维：方差约为 1/3

固定一个 token。它的 h 就是确定的 768 个数，随机的只有权重。这个 token 的 q 的第 $i$ 个分量（768 个分量中的一个）是：

\`\`\`text
q[i] = W[i,0] h[0] + W[i,1] h[1] + ... + W[i,767] h[767] + b[i]
\`\`\`

**均值为 0：**

\`\`\`text
E[q[i]] = E[W[i,0] h[0] + ... + W[i,767] h[767] + b[i]]
        = E[W[i,0]] h[0] + ... + E[W[i,767]] h[767] + E[b[i]]   ← (1) and (2)
        = 0 × h[0] + ... + 0 × h[767] + 0                       ← (3)
        = 0
\`\`\`

用到的三个事实：

1. **和的平均等于平均的和：** $E[X + Y] = E[X] + E[Y]$。不管是否独立，这一条都成立。
2. **常数因子可以提到平均外面：** $E[cX] = c \\times E[X]$。这里 $h[j]$ 是常数，因为 token 是固定的。
3. **每个权重和 bias 的均值都是 0：** 它们从 $U(-a,\\ a)$ 里抽取，这个分布以 0 为中心对称，所以 $E[W] = 0$，$E[b] = 0$。

直观地说：不管 $h[j]$ 是多少，每一项 $W[i,j]\\,h[j]$ 取正和取负的机会相等，平均下来就是 0。

**方差：** 每一项是一个随机的 $W$ 乘以常数 $h[j]$，按规则 1，它的方差是 $h[j]^2 \\times \\mathrm{Var}(W)$。768 项用的是互相独立的权重，按规则 2 直接相加：

\`\`\`text
Var(q[i]) = Var(W) × (h[0]² + h[1]² + ... + h[767]²) + Var(b)
          = Var(W) × 768 + Var(b)          ← the sum of squares from section 2
          = 1/(3 × 768) × 768 + 0.0004
          ≈ 1/3 ≈ 0.334                    ← measured: 0.334 ✓
\`\`\`

k 和 v 也一样，都约为 1/3，它们用的是 \`c_attn\` 权重里另外几部分、各自独立抽取的值。因为有 LayerNorm，这对**任何**输入 token 都成立。

## 5. 一项 q[i] × k[i]：方差 1/9

从这里开始，q 和 k 指的是**一个 head 的 64 维切片**（见 [Query View](#info:query_view_notes)）：q 来自 query token，k 来自 key token，$i$ 从 0 到 63。按第 4 节，每个分量的均值仍然是 0，方差仍然约为 1/3。

q 和 k 用的是不同的、各自独立抽取的权重，所以初始化时两者独立。按规则 3：

\`\`\`text
Var(q[i] × k[i]) = Var(q[i]) × Var(k[i]) = 1/3 × 1/3 = 1/9
\`\`\`

## 6. 点积 q · k：hs 项相加

head size $hs = 64$，两个 token 之间的分数是：

\`\`\`text
q · k = q[0]k[0] + q[1]k[1] + ... + q[63]k[63]
\`\`\`

不同的 $i$ 用的是权重里不同的行，它们是各自独立抽取的，所以这 64 项互相独立。每一项的方差都是 1/9，按规则 2：

\`\`\`text
Var(q · k) = 64 × 1/9 ≈ 7.1
std        = √7.1 ≈ 2.67                  ← measured: 2.67 ✓
\`\`\`

**Transformer 原论文的假设：** 如果 q、k 每一维的方差都是 1，每一项的方差就是 1，加起来是 64（标准差 8）。"方差约为 64"就是这么来的。

不管是哪一种情况，**点积的方差都和 $hs$ 成正比**：head size 翻倍，方差也翻倍。

## 7. 除以 √hs：方差不再依赖 hs

### 问题：方差随 hs 增大

把第 6 节写成一般形式：$q \\cdot k$ 是 $hs$ 个独立项的和；按规则 3，每一项 $q[i]\\,k[i]$ 的方差是 $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i])$。按规则 2：

\`\`\`text
Var(q · k) = hs × Var(q[i]) × Var(k[i])
\`\`\`

在本网站的初始化下（$\\mathrm{Var}(q[i]) = \\mathrm{Var}(k[i]) = 1/3$，所以每一项的方差是 1/9）：

\`\`\`text
hs     Var(q · k)          std
32     32 × 1/9  = 3.56    1.89
64     64 × 1/9  = 7.11    2.67     ← GPT-2 small
128    128 × 1/9 = 14.22   3.77
\`\`\`

head size 越大，分数就越大，尽管 q 和 k 本身没有变。

### 解决办法：除以 √hs

代码里算的是 \`(q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)\`，也就是每个分数乘以 $c = 1/\\sqrt{hs}$。按规则 1，方差乘以 $c^2 = 1/hs$：

\`\`\`text
Var(q · k / √hs) = (1/√hs)² × Var(q · k)                  ← rule 1 with c = 1/√hs
                 = (1/hs) × hs × Var(q[i]) × Var(k[i])     ← Var(q · k) from above
                 = Var(q[i]) × Var(k[i])                   ← hs cancels
\`\`\`

结果里不再有 $hs$：不管 head size 是多少，分数的方差都一样。

\`\`\`text
hs     before dividing    after dividing by √hs
32     3.56               3.56 / 32  = 1/9
64     7.11               7.11 / 64  = 1/9      std ≈ 0.33   ← measured: 0.334 ✓
128    14.22              14.22 / 128 = 1/9
\`\`\`

在 Transformer 原论文的假设下（每个分量方差为 1），同样的步骤得到 $1 \\times 1 = 1$：不管 head size 是多少，标准差都是 1。

### 为什么是 √hs，而不是 hs

方差按因子的**平方**变化（规则 1）。要让方差除以 $hs$，分数就要除以 $\\sqrt{hs}$。如果分数直接除以 $hs$，方差会除以 $hs^2$，剩下 $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i]) / hs$：这时 $hs$ 越大，分数反而越**小**，变成了相反的问题。只有 $\\sqrt{hs}$ 能正好把 $hs$ 消掉。

### 为什么分数不能太大

softmax 对分数之间的**差**非常敏感。看两个分数的情况。标准差为 8 时（论文的假设，除之前），相差 16 很常见：

\`\`\`text
scores [16, 0]  →  softmax ≈ [0.9999999, 0.0000001]
scores [2, 0]   →  softmax ≈ [0.88, 0.12]              ← the same scores divided by 8
\`\`\`

$[16, 0]$ 时几乎全部权重都给了一个 token，而且分数变化时它几乎不动：两个分数的情况下，输出对分数差的敏感度是 $p(1-p)$，这里只有约 $0.0000001$，而 $[2, 0]$ 时约为 $0.105$。敏感度极小，梯度就极小，模型就很难学会该关注哪里。更多见 [Q @ K](#info:q_k_product_notes)。

## 8. 残差流：24 次相加

点积是 64 项相加，残差流是 24 层的输出相加，用的是同一条规则 2：

\`\`\`text
x_final = x_embedding + update_1 + update_2 + ... + update_24
\`\`\`

GPT-2 small 有 12 个 block，每个 block 里 attention 和 MLP 各加一次，一共 24 次。假设每次更新的方差约为 $v$，并且各次大致独立：

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v      ← grows with depth
\`\`\`

**缩放 residual-writing weights：** 初始化时把 [\`proj\`](#info:attention_output_linear_notes)（以及 MLP 的 \`fc_out\`）的权重乘以 $1/\\sqrt{24}$。更新（近似）和这些权重成线性关系：它们是更新经过的最后一个矩阵，所以权重乘以 $c$，更新也乘以 $c$（GPT-2 把 bias 初始化为 0，所以 bias 不影响这一点）。因此每次更新缩小为 $1/\\sqrt{24}$，按规则 1，方差变成 $v/24$：

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v/24
             = Var(x_embedding) + v           ← no longer grows with depth
\`\`\`

这就是 $1/\\sqrt{N}$ 里那个根号的来源：方差按 $c^2$ 变化，要让 $N$ 份方差加起来只相当于一份，每次更新就要乘以 $1/\\sqrt{N}$。

GPT-2 论文和 nanoGPT 做了这个缩放；本网站的代码没有做（见 [Attention output projection](#info:attention_output_linear_notes)）。

## 总结

\`\`\`text
h = one token's LayerNorm out    average of h[j]² = 1 (exact)
W at initialization              Var(W) = 1/(3·768)
q[i] = 768 terms W·h             Var ≈ 768 × 1/(3·768) = 1/3
q[i]·k[i]                        Var = 1/3 × 1/3 = 1/9
q·k = 64 terms                   Var = 64/9 ≈ 7.1, std 2.67
divide by √64 = 8                Var = 1/9, std 0.33 (independent of hs)

residual stream = 24 updates     variance grows with depth
proj / fc_out weights × 1/√24    each update's variance ÷ 24; the total stops growing
\`\`\`

> 这些都**只在初始化时**成立，因为那时权重是随机、对称、独立的。训练开始后权重会变，这些方差也会跟着变。这些设计的作用，是让训练从一个尺度合适的起点开始。
`,
    },
  },
};
