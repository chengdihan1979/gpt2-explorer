// Notes: Training loop and optimizers.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  gradient_notes: {
    en: {
      title: "Gradient",
      md: `
**Here's a tiny 2-D numeric example that shows how the gradient is multiplied by each layer's Jacobian (matrix) in backprop.**

## Setup

Let

$$
W_1 = \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix},\\quad W_2 = \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix},\\quad x = \\begin{bmatrix}1\\\\ -2\\end{bmatrix}
$$

## Forward pass

$$
z = W_1 x = \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix} \\begin{bmatrix}1\\\\ -2\\end{bmatrix} = \\begin{bmatrix}1.0 + 0.5(-2)\\\\ -0.2 + 1.2(-2)\\end{bmatrix} = \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix}
$$

$$
y = W_2 z = \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix} \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix} = \\begin{bmatrix}0.8\\cdot 0.0 + (-0.3)(-2.6)\\\\ 0.4\\cdot 0.0 + 1.1(-2.6)\\end{bmatrix} = \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}
$$

$$
L = \\tfrac{1}{2}\\|y\\|^2 = \\tfrac{1}{2}\\big(0.78^2 + (-2.86)^2\\big) = \\tfrac{1}{2}(0.6084 + 8.1796) = 4.394
$$

## Backprop (chain rule = Jacobian multiplications)

Since $L = \\tfrac{1}{2}\\|y\\|^2$,

$$
\\frac{\\partial L}{\\partial y} = y = \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}
$$

$$
\\frac{\\partial L}{\\partial x} = \\Big(\\frac{\\partial y}{\\partial x}\\Big)^{\\!\\top}\\frac{\\partial L}{\\partial y} = (W_2 W_1)^{\\top} y = W_1^{\\top} W_2^{\\top} y
$$

### Step 1: multiply by $W_2^\\top$

$$
W_2^{\\top} = \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix} \\\\[6pt]
g_1 = W_2^{\\top}\\frac{\\partial L}{\\partial y} = \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} = \\begin{bmatrix}0.8\\cdot 0.78 + 0.4(-2.86)\\\\ -0.3\\cdot 0.78 + 1.1(-2.86)\\end{bmatrix} = \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix}
$$

### Step 2: multiply by $W_1^\\top$

$$
W_1^{\\top} = \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix} \\\\[6pt]
\\frac{\\partial L}{\\partial x} = W_1^{\\top} g_1 = \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix} \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix} = \\begin{bmatrix}1.0(-0.52) + (-0.2)(-3.38)\\\\ 0.5(-0.52) + 1.2(-3.38)\\end{bmatrix} = \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}
$$

## Check (single combined Jacobian)

$$
W_2 W_1 = \\begin{bmatrix}0.86 & 0.04\\\\ 0.18 & 1.52\\end{bmatrix} \\\\[6pt]
(W_2 W_1)^{\\top} y = \\begin{bmatrix}0.86 & 0.18\\\\ 0.04 & 1.52\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} = \\begin{bmatrix}0.86\\cdot 0.78 + 0.18(-2.86)\\\\ 0.04\\cdot 0.78 + 1.52(-2.86)\\end{bmatrix} = \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}
$$

## What this shows

1. Backprop takes the upstream gradient and left-multiplies it by each layer's Jacobian transpose, in reverse order.
2. In deep stacks, this becomes a product of many matrices; that product can shrink or grow rapidly: vanishing/exploding gradients.
3. Residual/identity paths keep each factor near identity, keeping the overall product well-behaved.

> **Note:** for $y = Wx + b$, the Jacobian with respect to $x$ is exactly $J = \\partial y / \\partial x = W$. In backprop you then multiply the upstream gradient by $J^\\top = W^\\top$. Outside of this special case, no: the Jacobian is not just $W$.
`,
    },
    zh: {
      title: "梯度",
      md: `
**下面是一个很小的 2 维数值例子，展示 backprop 时梯度如何被每一层的 Jacobian（矩阵）相乘。**

## 设定

令

$$
W_1 = \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix},\\quad W_2 = \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix},\\quad x = \\begin{bmatrix}1\\\\ -2\\end{bmatrix}
$$

## 前向传播

$$
z = W_1 x = \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix} \\begin{bmatrix}1\\\\ -2\\end{bmatrix} = \\begin{bmatrix}1.0 + 0.5(-2)\\\\ -0.2 + 1.2(-2)\\end{bmatrix} = \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix}
$$

$$
y = W_2 z = \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix} \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix} = \\begin{bmatrix}0.8\\cdot 0.0 + (-0.3)(-2.6)\\\\ 0.4\\cdot 0.0 + 1.1(-2.6)\\end{bmatrix} = \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}
$$

$$
L = \\tfrac{1}{2}\\|y\\|^2 = \\tfrac{1}{2}\\big(0.78^2 + (-2.86)^2\\big) = \\tfrac{1}{2}(0.6084 + 8.1796) = 4.394
$$

## Backprop（链式法则 = Jacobian 相乘）

因为 $L = \\tfrac{1}{2}\\|y\\|^2$，

$$
\\frac{\\partial L}{\\partial y} = y = \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}
$$

$$
\\frac{\\partial L}{\\partial x} = \\Big(\\frac{\\partial y}{\\partial x}\\Big)^{\\!\\top}\\frac{\\partial L}{\\partial y} = (W_2 W_1)^{\\top} y = W_1^{\\top} W_2^{\\top} y
$$

### 第 1 步：乘以 $W_2^\\top$

$$
W_2^{\\top} = \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix} \\\\[6pt]
g_1 = W_2^{\\top}\\frac{\\partial L}{\\partial y} = \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} = \\begin{bmatrix}0.8\\cdot 0.78 + 0.4(-2.86)\\\\ -0.3\\cdot 0.78 + 1.1(-2.86)\\end{bmatrix} = \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix}
$$

### 第 2 步：乘以 $W_1^\\top$

$$
W_1^{\\top} = \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix} \\\\[6pt]
\\frac{\\partial L}{\\partial x} = W_1^{\\top} g_1 = \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix} \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix} = \\begin{bmatrix}1.0(-0.52) + (-0.2)(-3.38)\\\\ 0.5(-0.52) + 1.2(-3.38)\\end{bmatrix} = \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}
$$

## 验证（合成一个 Jacobian）

$$
W_2 W_1 = \\begin{bmatrix}0.86 & 0.04\\\\ 0.18 & 1.52\\end{bmatrix} \\\\[6pt]
(W_2 W_1)^{\\top} y = \\begin{bmatrix}0.86 & 0.18\\\\ 0.04 & 1.52\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} = \\begin{bmatrix}0.86\\cdot 0.78 + 0.18(-2.86)\\\\ 0.04\\cdot 0.78 + 1.52(-2.86)\\end{bmatrix} = \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}
$$

## 这说明了什么

1. Backprop 拿到上游梯度后，按相反的顺序，依次左乘每一层 Jacobian 的转置。
2. 在很深的网络里，这就变成了很多矩阵的连乘；这个乘积可能迅速变小或变大：也就是梯度消失/梯度爆炸。
3. 残差/恒等路径让每个因子都接近单位矩阵，从而让整个乘积保持稳定。

> **注意：** 对于 $y = Wx + b$，关于 $x$ 的 Jacobian 恰好就是 $J = \\partial y / \\partial x = W$。backprop 时再把上游梯度乘以 $J^\\top = W^\\top$。除了这种特殊情况，Jacobian 并不只是 $W$。
`,
    },
  },

  zero_grad_notes: {
    en: {
      title: "optimizer.zero_grad()",
      md: `
**Why every training step starts by clearing the gradients.**

## Where it sits in the training loop

Each step of the loop does five things:

\`\`\`python
for step in range(max_steps):
    x, y = dataloader.get_batch("train")    # 1. get a batch
    optimizer.zero_grad(set_to_none=True)   # 2. clear the previous step's gradients
    _, loss = model(x, y)                   # 3. forward pass, compute the loss
    loss.backward()                         # 4. backward pass, compute every parameter's gradient
    torch.nn.utils.clip_grad_norm_(...)     #    (clip the gradients)
    optimizer.step()                        # 5. update the parameters with the gradients
\`\`\`

This note is about step 2.

## The reason: gradients accumulate, they are not overwritten

\`backward()\` **adds** the new gradient to each parameter's \`.grad\`; it does not replace the old value.

Check with one parameter: $w = 2$ and $\\text{loss} = (3w)^2$, so $\\frac{d\\,\\text{loss}}{dw} = 18w = 36$.

\`\`\`text
backward three times without clearing:
step 0:  w.grad = 36
step 1:  w.grad = 72     ← 36 + 36
step 2:  w.grad = 108    ← 36 + 36 + 36

set w.grad back to None, then backward:
         w.grad = 36     ← correct
\`\`\`

Without clearing, the gradient used at step 100 would be the sum of the gradients from all 100 steps, and the parameters would be pushed further and further off.

## Why PyTorch accumulates

Accumulating is the behavior you want in two common cases:

1. **One parameter is used more than once.** [Weight tying](#hover:weight_tying_notes) is an example: \`tok_emb.weight\` is used at the input and again at the output, and the two gradients have to be added. Accumulation does exactly that.
2. **Gradient accumulation.** When a large batch does not fit on the GPU, split it into several micro-batches. Call \`backward()\` on each without clearing, then call \`step()\` once. The result is the same as training on the large batch. Karpathy's build-nanogpt reproduction of GPT-2 (124M) uses this to reach a batch of about 0.5M tokens.

\`\`\`python
optimizer.zero_grad(set_to_none=True)
for micro_step in range(grad_accum_steps):
    x, y = dataloader.get_batch("train")
    _, loss = model(x, y)
    loss = loss / grad_accum_steps   # so the sum equals the mean over the whole large batch
    loss.backward()                  # gradients add up in p.grad
optimizer.step()
\`\`\`

> The \`loss / grad_accum_steps\` matters: each micro-batch's loss is already a mean over its own tokens, so without the division the accumulated gradient would be \`grad_accum_steps\` times too large.

## What \`set_to_none=True\` does

There are two ways to clear:

\`\`\`text
set_to_none=False:  p.grad is filled with zeros (the tensor still takes memory)
set_to_none=True:   p.grad = None (the memory is freed)
\`\`\`

On the next \`backward()\`, if \`.grad\` is \`None\`, PyTorch writes the new gradient straight in, instead of reading a tensor of zeros and adding to it. This saves memory and is slightly faster. Since PyTorch 2.0, \`set_to_none=True\` is the default.

> One difference to be aware of: after clearing with \`None\`, a parameter that receives no gradient in the next step keeps \`.grad = None\`, not a tensor of zeros. Optimizers skip parameters whose \`.grad\` is \`None\`.
`,
    },
    zh: {
      title: "optimizer.zero_grad()",
      md: `
**为什么每一步训练都要先清空梯度。**

## 它在训练循环里的位置

训练循环每一步做 5 件事：

\`\`\`python
for step in range(max_steps):
    x, y = dataloader.get_batch("train")    # 1. 取一个 batch
    optimizer.zero_grad(set_to_none=True)   # 2. 清空上一步的梯度
    _, loss = model(x, y)                   # 3. forward，算出 loss
    loss.backward()                         # 4. backward，算出每个参数的梯度
    torch.nn.utils.clip_grad_norm_(...)     #    （裁剪梯度）
    optimizer.step()                        # 5. 用梯度更新参数
\`\`\`

这篇讲第 2 步。

## 原因：PyTorch 的梯度是累加的，不是覆盖的

\`backward()\` 算出梯度后，会**加到**参数的 \`.grad\` 上，而不是替换原来的值。

用一个参数验证：$w = 2$，$\\text{loss} = (3w)^2$，所以 $\\frac{d\\,\\text{loss}}{dw} = 18w = 36$。

\`\`\`text
不清零，连续 backward 三次：
step 0:  w.grad = 36
step 1:  w.grad = 72     ← 36 + 36
step 2:  w.grad = 108    ← 36 + 36 + 36

把 w.grad 设回 None 再 backward：
         w.grad = 36     ← 正确
\`\`\`

如果不清零，第 100 步用的梯度就是前 100 步梯度的总和，参数会被越推越远。

## 为什么 PyTorch 要设计成累加

在两种常见情况下，累加正是想要的行为：

1. **一个参数被用了好几次。** [weight tying](#hover:weight_tying_notes) 就是这样：\`tok_emb.weight\` 在输入端和输出端各用了一次，两份梯度必须加在一起。累加正好做到这一点。
2. **Gradient accumulation（梯度累积）。** GPU 放不下一个大 batch 时，可以把它拆成几个小 batch（micro-batch）。每个小 batch 都 \`backward()\` 一次但不清零，最后只 \`step()\` 一次，效果和用大 batch 训练一样。Karpathy 的 build-nanogpt 复现 GPT-2（124M）时，就是用这个方法凑出约 0.5M token 的 batch。

\`\`\`python
optimizer.zero_grad(set_to_none=True)
for micro_step in range(grad_accum_steps):
    x, y = dataloader.get_batch("train")
    _, loss = model(x, y)
    loss = loss / grad_accum_steps   # 让加起来的结果等于整个大 batch 的平均
    loss.backward()                  # 梯度在 p.grad 里累加
optimizer.step()
\`\`\`

> \`loss / grad_accum_steps\` 这一步不能省：每个小 batch 的 loss 已经是它自己那些 token 的平均，不除的话，累加出来的梯度会大 \`grad_accum_steps\` 倍。

## \`set_to_none=True\` 是什么

有两种清零方式：

\`\`\`text
set_to_none=False:  p.grad 填成全 0 的 tensor（还占着内存）
set_to_none=True:   p.grad = None（释放内存）
\`\`\`

下一次 \`backward()\` 时，如果 \`.grad\` 是 \`None\`，PyTorch 会直接把新梯度写进去，不用先读一遍全 0 的 tensor 再相加，所以更省内存，也稍微快一点。从 PyTorch 2.0 开始，\`set_to_none=True\` 已经是默认值。

> 有一点区别要注意：用 \`None\` 清零后，如果某个参数在下一步没有收到梯度，它的 \`.grad\` 会一直是 \`None\`，而不是全 0。optimizer 会跳过 \`.grad\` 为 \`None\` 的参数。
`,
    },
  },

  backward_notes: {
    en: {
      title: "loss.backward()",
      md: `
**What \`loss.backward()\` does: one backward pass that computes the gradient of every parameter.**

## \`backward()\` computes gradients; \`step()\` updates parameters

\`backward()\` does **not** change any weight. It only computes the gradients and stores them in each parameter's \`.grad\`. The next line, \`optimizer.step()\`, reads \`.grad\` and updates the parameters.

\`\`\`python
loss.backward()     # compute gradients, store them in p.grad; weights unchanged
optimizer.step()    # read p.grad, update the weights
\`\`\`

Check with one parameter: $w = 2$, $\\text{loss} = (3w)^2$, so the gradient is $18w = 36$. The optimizer is plain SGD with learning rate $\\eta = 0.01$:

\`\`\`text
before backward:  w = 2.0    w.grad = None
after backward:   w = 2.0    w.grad = 36      ← gradient computed, w unchanged
after step:       w = 1.64   w.grad = 36      ← now w changes: 2 − 0.01 × 36 = 1.64
\`\`\`

So \`.grad\` has to be stored: it is how \`backward()\` hands its result to \`step()\`.

### Why keep them as two separate steps

If \`backward()\` updated the parameters right away, none of these would be possible:

1. **Process the gradients before updating.** Between \`backward()\` and \`step()\` the code calls [\`clip_grad_norm_\`](#hover:clip_grad_norm_notes): it measures how large all the gradients are together and scales them down if they are too large. That needs every gradient to be computed first.
2. **Gradient accumulation.** Several micro-batches each call \`backward()\`, their gradients add up in \`.grad\`, and \`step()\` runs once. See [zero_grad](#hover:zero_grad_notes).
3. **Different update rules.** SGD does $w \\leftarrow w - \\eta \\cdot g$. AdamW, used for GPT-2, also uses running averages of past gradients and adapts the step size per parameter. \`backward()\` only provides the gradient; the optimizer you choose decides how to use it.

So the work in each training step is split like this:

\`\`\`text
zero_grad()  clear .grad
backward()   compute gradients, write them into .grad
(clip)       adjust .grad
step()       read .grad, update the parameters
\`\`\`

## One backward pass covers every parameter

In this code each training step calls \`loss.backward()\` once, and that one call fills the \`.grad\` of **all** ~124M parameters.

During the forward pass, PyTorch records every operation and links them into a **computation graph**, from the embeddings, through the 12 blocks, \`ln_f\` and \`lm_head\`, to the loss. \`backward()\` starts at the loss and walks this graph **once, in reverse**. At each layer it uses the chain rule to pass the gradient on, and computes that layer's parameter gradients along the way:

\`\`\`text
loss → lm_head → ln_f → block 12 → block 11 → ... → block 1 → pos_emb, tok_emb
\`\`\`

When the walk is done, every parameter that took part in the forward pass has its gradient. No parameter needs its own backward call.

> The one case with several \`backward()\` calls per step is gradient accumulation. Even then, each call computes all parameters' gradients; the calls just add up in \`.grad\` before the single \`step()\`.

## What each layer does: two things

During the backward pass, each layer receives one gradient from the layer after it: the gradient of the loss with respect to **this layer's output**. Call it $\\frac{\\partial L}{\\partial y}$, the upstream gradient. With it, the layer does two things:

\`\`\`text
1. Gradients of its own parameters:  dL/dW, dL/db   → stored in W.grad, b.grad
2. Gradient of its input:            dL/dx          → handed to the previous layer as its dL/dy
\`\`\`

Thing 2 is what lets the gradient travel back layer by layer: this layer's $\\frac{\\partial L}{\\partial x}$ is the previous layer's $\\frac{\\partial L}{\\partial y}$.

### Worked example: one Linear layer

\`nn.Linear(2, 3)\`: 2 inputs, 3 outputs, $y = x W^\\top + b$.

\`\`\`text
x = [1, 2]

W = [[1, 0],      b = [0, 0, 0]
     [0, 1],
     [1, 1]]

y = x @ W.T + b = [1, 2, 3]
\`\`\`

Suppose the layer after it hands back this upstream gradient:

\`\`\`text
dL/dy = [0.5, -1, 2]
\`\`\`

**Thing 1: the parameter gradients.** Each output is $y_i = x \\cdot W[i] + b_i$, so:

\`\`\`text
# row i = the i-th upstream number × the input
dL/dW[i] = dL/dy[i] × x

W.grad = [[ 0.5 × [1, 2]],   = [[ 0.5,  1],
          [-1   × [1, 2]],      [-1,   -2],
          [ 2   × [1, 2]]]      [ 2,    4]]

# b[i] is added straight to y[i], so its gradient passes through unchanged
b.grad = dL/dy = [0.5, -1, 2]
\`\`\`

**Thing 2: the input gradient.** $x$ feeds every $y_i$, so its gradient is the sum of the rows $W[i]$, weighted by $\\frac{\\partial L}{\\partial y_i}$:

\`\`\`text
dL/dx = 0.5 × W[0] + (−1) × W[1] + 2 × W[2]
      = 0.5 × [1, 0] − [0, 1] + 2 × [1, 1]
      = [2.5, 1]                 → passed to the previous layer
\`\`\`

> For a longer chain, where the gradient is multiplied by each layer's Jacobian in turn, see [this two-layer example](#hover:gradient_notes).

## Across the whole of GPT-2

\`\`\`text
loss
 ↓ dL/dlogits = (p − y) / (B*T)        ← derived in the cross-entropy note
lm_head:  compute W.grad, pass dL/dx to ln_f
 ↓
ln_f:     compute γ.grad and β.grad, pass dL/dx to block 12
 ↓
block 12 → block 11 → ... → block 1    every Linear and LayerNorm inside does the same two things
 ↓
tok_emb, pos_emb:  only thing 1 (they are the first layer; nothing comes before them)
\`\`\`

The starting gradient $\\frac{(p - y)}{B \\times T}$ is derived in [Cross Entropy](#hover:cross_entropy_notes).

Layers **without parameters**, such as GELU, softmax and dropout, only do thing 2: they transform the gradient and pass it on.

> The [weight tying](#hover:weight_tying_notes) example already used both things. Its "output side" gradient is \`lm_head\`'s thing 1 (parameter gradient); its "input side" gradient is \`lm_head\`'s thing 2: the $\\frac{\\partial L}{\\partial x}$ that flows back to the embedding row.
`,
    },
    zh: {
      title: "loss.backward()",
      md: `
**\`loss.backward()\` 做了什么：一次反向传播，算出所有参数的梯度。**

## \`backward()\` 只算梯度，\`step()\` 才更新参数

\`backward()\` **不会**改任何 weight。它只算梯度，把结果存进每个参数的 \`.grad\`。下一行 \`optimizer.step()\` 读出 \`.grad\`，才去更新参数。

\`\`\`python
loss.backward()     # 只算梯度，结果存进 p.grad；weight 不变
optimizer.step()    # 读出 p.grad，用它更新 weight
\`\`\`

用一个参数验证：$w = 2$，$\\text{loss} = (3w)^2$，梯度是 $18w = 36$。优化器用最简单的 SGD，学习率 $\\eta = 0.01$：

\`\`\`text
backward 之前:  w = 2.0    w.grad = None
backward 之后:  w = 2.0    w.grad = 36      ← 梯度算出来了，w 没动
step 之后:      w = 1.64   w.grad = 36      ← w 才变了：2 − 0.01 × 36 = 1.64
\`\`\`

所以 \`.grad\` 必须存下来：它就是 \`backward()\` 交给 \`step()\` 的结果。

### 为什么要分成两步

如果 \`backward()\` 算完梯度就顺手更新参数，下面这些事情都做不了：

1. **先处理梯度，再更新参数。** 代码里 \`backward()\` 和 \`step()\` 之间有一行 [\`clip_grad_norm_\`](#hover:clip_grad_norm_notes)：它先看所有梯度合起来有多大，太大就整体缩小。这一步要等全部梯度都算完才能做。
2. **Gradient accumulation。** 几个小 batch 各 \`backward()\` 一次，梯度先在 \`.grad\` 里累加，最后只 \`step()\` 一次。见 [zero_grad](#hover:zero_grad_notes)。
3. **更新规则不止一种。** SGD 是 $w \\leftarrow w - \\eta \\cdot g$；GPT-2 用的 AdamW 还会用到过去很多步梯度的平均值，再按每个参数的情况调整步长。\`backward()\` 只负责提供梯度，怎么用由你选的 optimizer 决定。

所以每一步训练的分工是：

\`\`\`text
zero_grad()  清空 .grad
backward()   算梯度，写进 .grad
(clip)       调整 .grad
step()       读 .grad，更新参数
\`\`\`

## 一次反向传播覆盖所有参数

这份代码里，每个训练 step 只调用一次 \`loss.backward()\`，这一次就把**全部**约 1.24 亿个参数的 \`.grad\` 都填好。

forward 时，PyTorch 会记下每一步运算，连成一张**计算图**：从 embedding 开始，经过 12 个 block、\`ln_f\`、\`lm_head\`，一直到 loss。\`backward()\` 从 loss 出发，沿着这张图**倒着走一遍**。每经过一层，就用链式法则把梯度往前传，同时把这一层参数的梯度算出来：

\`\`\`text
loss → lm_head → ln_f → block 12 → block 11 → ... → block 1 → pos_emb, tok_emb
\`\`\`

走完这一遍，每个参与了 forward 的参数都拿到了梯度，不需要每个参数单独 backward 一次。

> 唯一一个 step 里 backward 好几次的情况是 gradient accumulation。即便如此，每次 backward 也都是一次算出所有参数的梯度，只是这几份梯度在 \`.grad\` 里累加，最后只 \`step()\` 一次。

## 每一层做两件事

反向传播时，每一层都会从它后面那一层收到一份梯度：**loss 对这一层输出的梯度**，记作 $\\frac{\\partial L}{\\partial y}$（上游梯度）。拿到它之后，这一层做两件事：

\`\`\`text
1. 算自己参数的梯度:   dL/dW, dL/db   → 存进 W.grad, b.grad
2. 算对输入的梯度:     dL/dx          → 交给前一层，当作它的 dL/dy
\`\`\`

第 2 件事就是梯度能一层层往回传的原因：这一层的 $\\frac{\\partial L}{\\partial x}$，就是前一层的 $\\frac{\\partial L}{\\partial y}$。

### 用一个 Linear 层算一遍

\`nn.Linear(2, 3)\`：输入 2 维，输出 3 维，$y = x W^\\top + b$。

\`\`\`text
x = [1, 2]

W = [[1, 0],      b = [0, 0, 0]
     [0, 1],
     [1, 1]]

y = x @ W.T + b = [1, 2, 3]
\`\`\`

假设后面那一层传回来的上游梯度是：

\`\`\`text
dL/dy = [0.5, -1, 2]
\`\`\`

**第 1 件事：参数的梯度。** 每个输出 $y_i = x \\cdot W[i] + b_i$，所以：

\`\`\`text
dL/dW[i] = dL/dy[i] × x          （第 i 行 = 上游梯度的第 i 个数 × 输入）

W.grad = [[ 0.5 × [1, 2]],   = [[ 0.5,  1],
          [-1   × [1, 2]],      [-1,   -2],
          [ 2   × [1, 2]]]      [ 2,    4]]

b.grad = dL/dy = [0.5, -1, 2]    （b[i] 直接加到 y[i] 上，所以梯度原样传过来）
\`\`\`

**第 2 件事：输入的梯度。** $x$ 参与了每一个 $y_i$，所以它的梯度是每一行 $W[i]$ 按 $\\frac{\\partial L}{\\partial y_i}$ 加权求和：

\`\`\`text
dL/dx = 0.5 × W[0] + (−1) × W[1] + 2 × W[2]
      = 0.5 × [1, 0] − [0, 1] + 2 × [1, 1]
      = [2.5, 1]                 → 传给前一层
\`\`\`

> 多层串起来时，梯度会依次乘上每一层的 Jacobian，见[这个两层的例子](#hover:gradient_notes)。

## 放到整个 GPT-2 里

\`\`\`text
loss
 ↓ dL/dlogits = (p − y) / (B*T)        ← cross-entropy 那篇推过的
lm_head:  算 W.grad，把 dL/dx 传给 ln_f
 ↓
ln_f:     算 γ.grad 和 β.grad，把 dL/dx 传给 block 12
 ↓
block 12 → block 11 → ... → block 1    每个内部的 Linear、LayerNorm 都做同样的两件事
 ↓
tok_emb, pos_emb:  只做第 1 件事（它们是第一层，前面没有层了）
\`\`\`

起点的梯度 $\\frac{(p - y)}{B \\times T}$ 在 [Cross Entropy](#hover:cross_entropy_notes) 里推导过。

GELU、softmax、dropout 这类**没有参数**的层，只做第 2 件事：把梯度变换一下，继续往前传。

> 前面 [weight tying](#hover:weight_tying_notes) 的例子其实已经用到了这两件事。它的"输出端梯度"是 \`lm_head\` 的第 1 件事（参数梯度）；"输入端梯度"是 \`lm_head\` 的第 2 件事：算出 $\\frac{\\partial L}{\\partial x}$，传回 embedding 的那一行。
`,
    },
  },

  clip_grad_norm_notes: {
    en: {
      title: "clip_grad_norm_",
      md: `
**Gradient clipping: when the gradients are too large, scale them all down together.**

\`\`\`python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
\`\`\`

This line sits between \`backward()\` and \`step()\`. The gradients have been computed but not yet used to update the parameters, so this is the moment to check whether they are too large. (Why the two steps are separate: see [backward](#hover:backward_notes).)

## What it does

1. Treat the gradients of **all** parameters as one long vector (about 124M numbers for GPT-2) and compute its length, the norm: square every number, add them all up, take the square root.

   $$
   \\text{norm} = \\sqrt{\\sum_{\\text{all gradient entries } g} g^2}
   $$

2. If the norm is at most \`max_norm\` (here 1.0), do nothing.
3. If the norm is larger, multiply **every** gradient by the same factor $\\frac{\\text{max\\_norm}}{\\text{norm}}$, so the total length becomes exactly \`max_norm\`.

## Worked example

Suppose the model had only two parameters, \`a\` and \`b\`, with 2 numbers each:

\`\`\`text
a.grad = [3, 0]
b.grad = [0, 4]

step 1:  total norm = √(3² + 0² + 0² + 4²) = √25 = 5
step 3:  5 > 1.0, so the scale factor = 1.0 / 5 = 0.2

a.grad = [3, 0] × 0.2 = [0.6, 0]
b.grad = [0, 4] × 0.2 = [0, 0.8]

total norm after clipping = √(0.6² + 0.8²) = 1.0
\`\`\`

If the gradients are already small, nothing changes:

\`\`\`text
a.grad = [0.3, 0],  b.grad = [0, 0.4]
total norm = 0.5 ≤ 1.0   →  unchanged
\`\`\`

## Two key points

**1. It changes the length, not the direction.** Every gradient is multiplied by the same factor, so the ratios between parameters stay the same (\`a\` and \`b\` are still 3 : 4). The update points the same way as before; the step is just shorter.

**2. The norm is global, not per parameter.** If each parameter were clipped on its own, the ratio between \`a\` and \`b\` could change, and so would the direction of the update. Clipping by the global norm keeps the direction.

## Why it is needed

Most steps produce gradients of roughly similar size. Occasionally a "bad batch", such as an unusual stretch of data, produces a very large gradient. Without clipping, that one step would push the parameters far away and the loss would jump (a **loss spike**); in the worst case training never recovers.

Clipping puts a cap on how far any single step can move: a bad batch still affects that step, but only by a limited amount.

## The return value

\`clip_grad_norm_\` returns the total norm **before** clipping (5.0 in the example). Many training scripts print it:

\`\`\`python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"step {step} loss {loss.item():.4f} norm {norm:.4f}")
\`\`\`

This number is worth watching: a sudden jump in the norm is an early sign that training is going wrong.

## Details

- The trailing \`_\` in the name is a PyTorch convention for **in-place** operations: it modifies \`.grad\` directly instead of returning new gradients.
- The threshold 1.0 is the value used in the GPT-3 paper ("we clip the global norm of the gradient at 1.0"); most LLM training runs since then use 1.0 as well.
`,
    },
    zh: {
      title: "clip_grad_norm_",
      md: `
**梯度裁剪：梯度太大时，把所有梯度一起缩小。**

\`\`\`python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
\`\`\`

这一行夹在 \`backward()\` 和 \`step()\` 之间。梯度已经算好了，但还没拿去更新参数，正好在这时检查一下梯度是不是太大。（为什么要分成两步，见 [backward](#hover:backward_notes)。）

## 它做了什么

1. 把**所有参数**的梯度拼成一个大向量（GPT-2 里约有 1.24 亿个数），算出它的长度，也就是 norm：每个数平方、全部加起来、再开根号。

   $$
   \\text{norm} = \\sqrt{\\sum_{\\text{所有梯度元素 } g} g^2}
   $$

2. 如果 norm 不超过 \`max_norm\`（这里是 1.0），什么都不做。
3. 如果超过了，就把**所有**梯度乘以同一个系数 $\\frac{\\text{max\\_norm}}{\\text{norm}}$，让总长度正好变成 \`max_norm\`。

## 算一遍

假设模型只有两个参数 \`a\` 和 \`b\`，各 2 个数：

\`\`\`text
a.grad = [3, 0]
b.grad = [0, 4]

第 1 步：总 norm = √(3² + 0² + 0² + 4²) = √25 = 5
第 3 步：5 > 1.0，所以缩放系数 = 1.0 / 5 = 0.2

a.grad = [3, 0] × 0.2 = [0.6, 0]
b.grad = [0, 4] × 0.2 = [0, 0.8]

缩完之后的总 norm = √(0.6² + 0.8²) = 1.0
\`\`\`

如果梯度本来就小，就不变：

\`\`\`text
a.grad = [0.3, 0],  b.grad = [0, 0.4]
总 norm = 0.5 ≤ 1.0   →  不变
\`\`\`

## 两个关键点

**1. 只改长度，不改方向。** 所有梯度乘的是同一个系数，所以各个参数之间的比例不变（\`a\` 和 \`b\` 还是 3 : 4）。更新的方向和原来一样，只是步子变小了。

**2. 算的是全局 norm，不是每个参数各算各的。** 如果每个参数单独裁剪，\`a\` 和 \`b\` 之间的比例就可能被打乱，更新方向也就变了。用全局 norm 裁剪，方向保持不变。

## 为什么需要它

大多数 step 的梯度大小都差不多。但偶尔会遇到一个"坏 batch"，比如一段很少见的数据，算出来的梯度特别大。不裁剪的话，这一步会把参数推得很远，loss 突然冲高（叫 **loss spike**），严重时训练就再也恢复不过来。

裁剪相当于给每一步能走多远设了一个上限：坏 batch 还是会影响这一步，但影响有限。

## 返回值

\`clip_grad_norm_\` 返回**裁剪之前**的总 norm（上面例子里是 5.0）。很多训练脚本会把它打印出来：

\`\`\`python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"step {step} loss {loss.item():.4f} norm {norm:.4f}")
\`\`\`

训练时值得盯着这个数：如果 norm 突然变得很大，往往是训练要出问题的早期信号。

## 细节

- 名字结尾的 \`_\` 是 PyTorch 的惯例，表示**原地修改**（in-place）：它直接改 \`.grad\`，而不是返回新的梯度。
- 阈值 1.0 是 GPT-3 论文里用的值（"we clip the global norm of the gradient at 1.0"），之后大部分 LLM 训练也沿用 1.0。
`,
    },
  },

  optim_adamw_notes: {
    en: {
      title: "Optimizers: from SGD to AdamW",
      md: `
**An optimizer turns gradients into parameter updates.**

[\`loss.backward()\`](#hover:backward_notes) computes every parameter's gradient; \`optimizer.step()\` then decides how far, and in which direction, to move each parameter. Optimizers differ only in that rule. They are used in every kind of training, pretraining and finetuning alike.

This note follows the history: each optimizer fixes a problem of the one before. Click a name for the details and a worked example. The last section explains the AdamW settings in this site's code.

## Two problems with plain SGD

The simplest rule, [SGD](#info:sgd_notes), moves every parameter against its gradient $g$ by a fixed learning rate $\\eta$ (\`lr\` in PyTorch): $w \\leftarrow w - \\eta \\cdot g$. It has two problems, and the optimizers below were invented to fix them:

1. **Gradients are noisy.** Each step estimates the gradient from one small batch, so part of every step is noise and the path zigzags. Fix: **momentum**, an average of recent gradients, so the noise cancels and the consistent direction remains.
2. **One learning rate for all parameters.** Parameters can have gradients of very different sizes. With $\\eta = 0.01$, a parameter whose gradient is about 100 moves 1 per step (it may oscillate), while one whose gradient is about 0.01 moves 0.0001 (it barely learns). Fix: a **per-parameter step size** that shrinks for parameters with consistently large gradients and grows for those with small ones.

\`\`\`text
SGD
 ├─ problem 1 (noise)            → Momentum
 └─ problem 2 (one η for all)    → Adagrad → RMSProp
                                               ↓
                     Momentum + RMSProp  =  Adam
                                               ↓
             decoupled weight decay  =  AdamW   ← used in this code
\`\`\`

## Timeline

The eras below say what language models **used** at the time; several of these methods are much older.

### Before 2014: the building blocks

- **[SGD](#info:sgd_notes)**: the basic rule. Stochastic approximation goes back to Robbins & Monro (1951).
- **[SGD with momentum](#info:sgd_momentum_notes)**: adds a velocity to smooth the updates. Proposed by Polyak (1964); shown to matter for deep networks by Sutskever et al. (2013).
- **[Adagrad](#info:adagrad_notes)** (2011): the first popular per-parameter learning rate. It helped with sparse, high-variance gradients such as word embeddings, where a rare word's row gets a gradient only now and then.
- **[RMSProp](#info:rmsprop_notes)** (2012): fixes Adagrad's step size shrinking toward zero over time; widely used in early RNN language models.

### 2014–2017: Adam becomes the default

- **[Adam](#info:adam_notes)** (2014) becomes the default in NLP (RNN language models, seq2seq with attention). It combines momentum with RMSProp's per-parameter scaling, so it needs much less learning-rate tuning than SGD.

### 2017–2019: Transformers

- **Adam + warmup becomes standard:** a short linear warmup, then a decay (inverse-sqrt in the original Transformer, later usually cosine).
- Post-norm gives way to pre-norm; together with Adam's adaptivity and warmup, this makes deep stacks train stably.
- **Decoupled weight decay.** The AdamW paper (Loshchilov & Hutter, 2017) shows that weight decay should be applied directly to the weights, not added to the gradient as L2 regularization. BERT (2018) already used Adam with this kind of decoupled decay, and excluded LayerNorm and bias from it; this recipe spread with BERT.

### 2019–2021: scaling up (GPT-2/3, T5)

- **[AdamW](#info:adamw_notes)** becomes the default for large language models, used with warmup + cosine decay, global-norm [gradient clipping](#hover:clip_grad_norm_notes), and no decay on norm/bias weights.
- The GPT-2 paper gives no optimizer details. The [GPT-3](#info:gpt3_notes) paper does: Adam with $\\beta_1 = 0.9$, $\\beta_2 = 0.95$, gradient clipping at 1.0, weight decay 0.1, and warmup followed by cosine decay. This site's code uses the same values.
- **Adafactor** (memory-efficient and Adam-like, with factored second moments) is used for T5/mT5 when optimizer memory is tight.

### 2022–2023: AdamW holds

- Pretraining stays on AdamW (or Adafactor where memory is the limit).
- New optimizers such as Lion and Sophia show gains in some setups, but none broadly replaces AdamW.
- Fused AdamW kernels become standard for speed: PyTorch's built-in \`AdamW(..., fused=True)\` (since PyTorch 2.0), Apex, DeepSpeed.

### 2024–today

- AdamW is still the most common default for LLM pretraining and full-parameter finetuning; Adafactor remains an option when memory must be minimized.
- 🆕 **Muon** (Keller Jordan, late 2024, first used in the NanoGPT speedrun) updates each 2D weight matrix with an orthogonalized version of its momentum, so no direction in the matrix dominates the update. Moonshot scaled it up and trained Kimi K2 (2025, 1T total parameters) with **MuonClip**: Muon plus a "QK-clip" that rescales the query/key weights whenever attention logits grow too large. Muon is used only for the hidden weight matrices; embeddings, the output layer and norm weights still use AdamW. It is the best-known case so far of an optimizer replacing AdamW for pretraining at this scale.

## The AdamW in this code

\`\`\`python
# weight decay only on 2D weights (Linear, Embedding), not on biases or LayerNorm
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
\`\`\`

### Two parameter groups

- **\`decay\`**: parameters with 2 or more dimensions: every \`nn.Linear\` weight and the embedding tables. These hold almost all of the model's parameters, and weight decay keeps them from growing too large.
- **\`no_decay\`**: 1D parameters: every bias, and LayerNorm's $\\gamma$ and $\\beta$. They are a tiny share of the parameters, so decaying them does little against overfitting. Decaying $\\gamma$ would even work against LayerNorm: it pulls $\\gamma$ toward 0, which shrinks the normalized output that LayerNorm is meant to rescale.
- The tied weight (\`tok_emb.weight\` = \`lm_head.weight\`, see [weight tying](#hover:weight_tying_notes)) appears only once, because \`model.parameters()\` skips duplicates. It lands in \`decay\`.

This is the same split nanoGPT uses: \`p.dim() >= 2\` decides.

### The hyperparameters

- **\`lr=3e-4\`**: a common safe learning rate for Adam on small models. (The GPT-3 paper used 6e-4 for its 125M model, the size of GPT-2 small.)
- **\`betas=(0.9, 0.95)\`**: the averaging rates of Adam's two running averages. A rate $\\beta$ averages over roughly the last $\\frac{1}{1 - \\beta}$ steps:
  - $\\beta_1 = 0.9$: the momentum (average gradient) covers about the last 10 steps.
  - $\\beta_2 = 0.95$: the average squared gradient covers about the last 20 steps. PyTorch's default is 0.999, about 1000 steps. The shorter window reacts faster when gradient sizes change, which makes large-model training more stable; GPT-3 used 0.95.
- **\`weight_decay=0.1\`** (only in the \`decay\` group): each step shrinks the weights a little toward 0, in proportion to the learning rate. The value comes from GPT-3. See [AdamW](#info:adamw_notes) for how decoupled decay works.

### What this code leaves out

The learning rate stays at 3e-4 for the whole run. Real GPT-2/GPT-3 training uses a **schedule**: a linear warmup from near 0 over the first steps, then a cosine decay down to a small value. Warmup avoids large updates while Adam's running averages are still unreliable; the decay lets the model settle into a minimum at the end.

## References

1. **Decoupled Weight Decay Regularization** (AdamW). Loshchilov & Hutter, 2017. [arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
2. **Language Models are Few-Shot Learners** (GPT-3), Appendix B: training details. Brown et al., 2020. [arXiv:2005.14165 ↗](https://arxiv.org/abs/2005.14165)
3. **Muon: An optimizer for hidden layers in neural networks**. Keller Jordan, 2024. [kellerjordan.github.io ↗](https://kellerjordan.github.io/posts/muon/)
4. **Kimi K2: Open Agentic Intelligence** (MuonClip). Moonshot AI, 2025. [arXiv:2507.20534 ↗](https://arxiv.org/abs/2507.20534)
`,
    },
    zh: {
      title: "Optimizer：从 SGD 到 AdamW",
      md: `
**optimizer 负责把梯度变成参数的更新。**

[\`loss.backward()\`](#hover:backward_notes) 算出每个参数的梯度；接着 \`optimizer.step()\` 决定每个参数往哪个方向、走多远。不同的 optimizer，区别只在这个规则。预训练和微调都要用 optimizer。

这篇按历史顺序讲：每个 optimizer 都是为了解决上一个的某个问题。点名字可以看详细内容和数值例子。最后一节讲这个网站代码里 AdamW 的设置。

## 普通 SGD 的两个问题

最简单的规则是 [SGD](#info:sgd_notes)：每个参数都按一个固定的学习率，往梯度 $g$ 的反方向走一步，学习率记作 $\\eta$（PyTorch 里的参数名是 \`lr\`）：$w \\leftarrow w - \\eta \\cdot g$。它有两个问题，下面这些 optimizer 就是为了解决它们而出现的：

1. **梯度很"吵"。** 每一步只用一个小 batch 估算梯度，所以每一步的方向里都有一部分是噪声，走出来的路线是"之"字形的。解决办法：**momentum**，对最近几步的梯度求平均，噪声互相抵消，稳定的方向被保留下来。
2. **所有参数共用一个学习率。** 不同参数的梯度大小可能差很多。$\\eta = 0.01$ 时，梯度约为 100 的参数每步移动 1（可能来回震荡），梯度约为 0.01 的参数每步只移动 0.0001（几乎学不动）。解决办法：**给每个参数自己的步长**，梯度一直很大的参数步子自动缩小，梯度一直很小的步子自动放大。

\`\`\`text
SGD
 ├─ 问题 1（噪声）            → Momentum
 └─ 问题 2（共用学习率）      → Adagrad → RMSProp
                                            ↓
                  Momentum + RMSProp  =  Adam
                                            ↓
                 解耦的 weight decay  =  AdamW   ← 这份代码用的
\`\`\`

## 时间线

下面每个时期写的是语言模型当时**在用**什么；其中好几个方法提出得早得多。

### 2014 年以前：基础组件

- **[SGD](#info:sgd_notes)**：最基本的规则。随机逼近的思想可以追溯到 Robbins & Monro（1951）。
- **[SGD with momentum](#info:sgd_momentum_notes)**：加入速度（velocity），让更新更平滑。由 Polyak（1964）提出；Sutskever 等人（2013）证明了它对深度网络很重要。
- **[Adagrad](#info:adagrad_notes)**（2011）：第一个流行起来的"每个参数单独的学习率"方法。它对稀疏、高方差的梯度很有帮助，比如词向量：罕见词的那一行只是偶尔才拿到梯度。
- **[RMSProp](#info:rmsprop_notes)**（2012）：修正了 Adagrad 步长随时间缩到接近 0 的问题；在早期 RNN 语言模型中被广泛使用。

### 2014–2017：Adam 成为默认选择

- **[Adam](#info:adam_notes)**（2014）成为 NLP 的默认选择（RNN 语言模型、带 attention 的 seq2seq）。它把 momentum 和 RMSProp 的"按参数缩放"结合在一起，所以比 SGD 少得多地需要反复调学习率。

### 2017–2019：Transformer

- **Adam + warmup 成为标准**：先做一小段线性 warmup，再衰减（原始 Transformer 用 inverse-sqrt，后来通常用 cosine）。
- 从 post-norm 过渡到 pre-norm；再加上 Adam 的自适应性和 warmup，很深的层堆叠也能稳定训练。
- **解耦的 weight decay。** AdamW 论文（Loshchilov & Hutter，2017）指出：weight decay 应该直接作用在权重上，而不是作为 L2 正则项加到梯度里。BERT（2018）用的 Adam 已经是这种解耦的 decay，而且不对 LayerNorm 和 bias 做 decay；这套做法随着 BERT 流行开来。

### 2019–2021：规模扩大（GPT-2/3、T5）

- **[AdamW](#info:adamw_notes)** 成为大语言模型的默认选择，搭配 warmup + cosine 衰减、全局范数的[梯度裁剪](#hover:clip_grad_norm_notes)，并且对 norm/bias 的权重不做 decay。
- GPT-2 论文没有公布 optimizer 的细节。[GPT-3](#info:gpt3_notes) 论文公布了：Adam，$\\beta_1 = 0.9$、$\\beta_2 = 0.95$，梯度裁剪阈值 1.0，weight decay 0.1，先 warmup 再 cosine 衰减。这个网站的代码用的就是这些值。
- **Adafactor**（省内存、类似 Adam，对二阶矩做分解）在 optimizer 内存紧张时被 T5/mT5 采用。

### 2022–2023：AdamW 站稳

- 预训练仍然用 AdamW（或者在内存是瓶颈时用 Adafactor）。
- Lion、Sophia 等新 optimizer 在某些设置下表现更好，但都没能普遍取代 AdamW。
- 为了提速，融合的（fused）AdamW kernel 成为标配：PyTorch 自带的 \`AdamW(..., fused=True)\`（PyTorch 2.0 起）、Apex、DeepSpeed。

### 2024 至今

- AdamW 仍然是 LLM 预训练和全参数微调最常用的默认选择；必须把内存压到最小时，Adafactor 仍是一个选项。
- 🆕 **Muon**（Keller Jordan，2024 年底，最早用在 NanoGPT speedrun 里）对每个二维权重矩阵，用"正交化"之后的 momentum 来更新，让矩阵里没有哪个方向主导更新。Moonshot 把它扩展到大规模，用 **MuonClip** 训练了 Kimi K2（2025，总参数 1T）：MuonClip 是 Muon 加上 "QK-clip"，一旦 attention 的 logits 变得太大，就把 query/key 的权重按比例缩小。Muon 只用于中间层的权重矩阵；embedding、输出层和 norm 的权重仍然用 AdamW。这是目前在这种规模的预训练上用 optimizer 取代 AdamW 最知名的例子。

## 这份代码里的 AdamW

\`\`\`python
# weight decay only on 2D weights (Linear, Embedding), not on biases or LayerNorm
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
\`\`\`

### 两组参数

- **\`decay\`**：2 维及以上的参数，也就是所有 \`nn.Linear\` 的权重和 embedding 表。模型几乎所有的参数都在这里，weight decay 防止它们长得太大。
- **\`no_decay\`**：1 维的参数，也就是所有 bias，以及 LayerNorm 的 $\\gamma$ 和 $\\beta$。它们只占参数的很小一部分，对它们做 decay 对防止过拟合帮助不大。对 $\\gamma$ 做 decay 甚至会和 LayerNorm 的作用对着干：它把 $\\gamma$ 往 0 拉，而 $\\gamma$ 本来是负责把归一化后的输出重新放大的。
- 共享的权重（\`tok_emb.weight\` = \`lm_head.weight\`，见 [weight tying](#hover:weight_tying_notes)）只出现一次，因为 \`model.parameters()\` 会跳过重复的参数。它被分到 \`decay\` 组。

nanoGPT 也是这样分组的：用 \`p.dim() >= 2\` 来判断。

### 超参数

- **\`lr=3e-4\`**：小模型上 Adam 常用的一个稳妥的学习率。（GPT-3 论文里和 GPT-2 small 同样大小的 125M 模型用的是 6e-4。）
- **\`betas=(0.9, 0.95)\`**：Adam 两个滑动平均的平均速率。速率为 $\\beta$ 的滑动平均，大约相当于对最近 $\\frac{1}{1 - \\beta}$ 步求平均：
  - $\\beta_1 = 0.9$：momentum（梯度的平均）大约覆盖最近 10 步。
  - $\\beta_2 = 0.95$：梯度平方的平均大约覆盖最近 20 步。PyTorch 的默认值是 0.999，大约 1000 步。窗口短一些，梯度大小变化时反应更快，大模型训练更稳定；GPT-3 用的就是 0.95。
- **\`weight_decay=0.1\`**（只用在 \`decay\` 组）：每一步都把权重往 0 稍微缩一点，缩多少和学习率成正比。这个值来自 GPT-3。解耦的 decay 具体怎么算，见 [AdamW](#info:adamw_notes)。

### 这份代码省略了什么

学习率在整个训练过程中一直是 3e-4。真实的 GPT-2/GPT-3 训练会用**学习率调度**（schedule）：前面一段从接近 0 线性 warmup 上来，然后按 cosine 曲线衰减到一个很小的值。warmup 是为了在 Adam 的滑动平均还不可靠的时候避免大幅更新；衰减是为了让模型在最后稳定地收敛到一个最低点。

## 参考资料

1. **Decoupled Weight Decay Regularization**（AdamW）。Loshchilov & Hutter，2017。[arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
2. **Language Models are Few-Shot Learners**（GPT-3），附录 B：训练细节。Brown 等人，2020。[arXiv:2005.14165 ↗](https://arxiv.org/abs/2005.14165)
3. **Muon: An optimizer for hidden layers in neural networks**。Keller Jordan，2024。[kellerjordan.github.io ↗](https://kellerjordan.github.io/posts/muon/)
4. **Kimi K2: Open Agentic Intelligence**（MuonClip）。Moonshot AI，2025。[arXiv:2507.20534 ↗](https://arxiv.org/abs/2507.20534)
`,
    },
  },

  sgd_notes: {
    en: {
      title: "SGD (Stochastic Gradient Descent)",
      md: `
**The classic and simplest optimizer: it updates parameters in the negative gradient direction with a fixed learning rate.**

## What it is

- The **gradient** tells you which way the loss increases fastest.
- To go **downhill**, move **against** the gradient.

Say we have parameters $\\theta$ and a loss $L(\\theta)$ we want to minimize. The update rule is:

$$
\\theta \\leftarrow \\theta - \\eta \\nabla_{\\theta} L(\\theta)
$$

- $\\eta$ is the **learning rate** (step size). Too big → overshoot. Too small → slow.
- $\\nabla_{\\theta} L(\\theta)$ is the gradient of the loss with respect to the parameters $\\theta$.

> Think of it as walking downhill in fog. The gradient is your local downward slope; the learning rate is how big a step you take. Take a big step and you might miss the bottom. Take a small step and it might take a long time to get to the bottom.

## Tiny numeric example

### Setup

- Model: $\\hat{y} = wx + b$
- Loss: $L = \\frac{1}{2} (\\hat{y} - y)^2$, where $y$ is the target value
- Data point: $x = 2$, $y = 5$
- Init: $w = 1.0$, $b = 0.0$
- Learning rate: $\\eta = 0.1$

### Step 1

**Forward pass:**

- $\\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2$
- Error $e = \\hat{y} - y = 2 - 5 = -3$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5$

**Backprop:**

- $\\nabla_{\\hat{y}} L = \\hat{y} - y = e = -3$
- $\\nabla_w L = \\nabla_{\\hat{y}} L \\cdot \\nabla_w \\hat{y} = e \\cdot x = -3 \\cdot 2 = -6$
- $\\nabla_b L = \\nabla_{\\hat{y}} L \\cdot \\nabla_b \\hat{y} = e = -3$

**Optimizer update:**

- $w \\leftarrow w - \\eta \\nabla_w L = 1.0 - 0.1 \\cdot (-6) = 1.6$
- $b \\leftarrow b - \\eta \\nabla_b L = 0.0 - 0.1 \\cdot (-3) = 0.3$

### Step 2

**Forward pass:**

- $\\hat{y} = wx + b = 1.6 \\cdot 2 + 0.3 = 3.5$
- Error $e = \\hat{y} - y = 3.5 - 5 = -1.5$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-1.5)^2 = 1.125$

**Backprop:**

- $\\nabla_{\\hat{y}} L = e = -1.5$
- $\\nabla_w L = e \\cdot x = -1.5 \\cdot 2 = -3$
- $\\nabla_b L = e = -1.5$

**Optimizer update:**

- $w = 1.6 - 0.1 \\cdot (-3) = 1.9$
- $b = 0.3 - 0.1 \\cdot (-1.5) = 0.45$

### Step 3

**Forward pass:**

- $\\hat{y} = wx + b = 1.9 \\cdot 2 + 0.45 = 4.25$
- Error $e = \\hat{y} - y = 4.25 - 5 = -0.75$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-0.75)^2 = 0.28125$

**Backprop:**

- $\\nabla_{\\hat{y}} L = e = -0.75$
- $\\nabla_w L = e \\cdot x = -0.75 \\cdot 2 = -1.5$
- $\\nabla_b L = e = -0.75$

**Optimizer update:**

- $w = 1.9 - 0.1 \\cdot (-1.5) = 2.05$
- $b = 0.45 - 0.1 \\cdot (-0.75) = 0.525$

> You can see the loss decreasing at each step (4.5 → 1.125 → 0.28125). That's the optimizer using backprop's gradients to move $w, b$ toward better values.

## Two problems with plain SGD

SGD works, but it has two problems, and the optimizers that followed were invented to fix them:

1. **Gradients are noisy.** In real training each step estimates the gradient from one small batch (in this site's code, 4 × 32 = 128 tokens), so part of every step is noise, and the path zigzags instead of heading straight downhill. Fix: [momentum](#info:sgd_momentum_notes), which averages recent gradients so the noise cancels.
2. **One learning rate for all parameters.** Parameters can have gradients of very different sizes. Take two parameters whose gradients are about 100 and about 0.01, with $\\eta = 0.01$:

   \`\`\`text
   parameter A:  moves 0.01 × 100  = 1       per step   ← may be too large and oscillate
   parameter B:  moves 0.01 × 0.01 = 0.0001  per step   ← too small, barely learns
   \`\`\`

   Raise $\\eta$ and A diverges; lower it and B stops learning. No single learning rate suits both. Fix: give each parameter its own step size, as [Adagrad](#info:adagrad_notes) does.

> The example above has a single data point, so its gradient has no noise, and $w$ and $b$ have gradients of similar size. That is why plain SGD does so well there.
`,
    },
    zh: {
      title: "SGD（随机梯度下降）",
      md: `
**最经典、最简单的 optimizer：用固定的学习率，沿着负 gradient 方向更新参数。**

## 它是什么

- **gradient** 告诉你 loss 往哪个方向增长得最快。
- 要**下坡**，就**逆着** gradient 走。

假设我们有参数 $\\theta$，以及想要最小化的 loss $L(\\theta)$。更新规则是：

$$
\\theta \\leftarrow \\theta - \\eta \\nabla_{\\theta} L(\\theta)
$$

- $\\eta$ 是**学习率**（步长）。太大 → 冲过头。太小 → 太慢。
- $\\nabla_{\\theta} L(\\theta)$ 是 loss 对参数 $\\theta$ 的 gradient。

> 可以把它想象成在大雾中下山。gradient 是你脚下局部的下坡方向；学习率是你每一步迈多大。步子迈大了，可能会错过谷底；步子迈小了，可能要走很久才能到达谷底。

## 小数值例子

### 设置

- 模型：$\\hat{y} = wx + b$
- Loss：$L = \\frac{1}{2} (\\hat{y} - y)^2$，其中 $y$ 是目标值
- 数据点：$x = 2$，$y = 5$
- 初始值：$w = 1.0$，$b = 0.0$
- 学习率：$\\eta = 0.1$

### 第 1 步

**前向传播：**

- $\\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2$
- 误差 $e = \\hat{y} - y = 2 - 5 = -3$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5$

**反向传播：**

- $\\nabla_{\\hat{y}} L = \\hat{y} - y = e = -3$
- $\\nabla_w L = \\nabla_{\\hat{y}} L \\cdot \\nabla_w \\hat{y} = e \\cdot x = -3 \\cdot 2 = -6$
- $\\nabla_b L = \\nabla_{\\hat{y}} L \\cdot \\nabla_b \\hat{y} = e = -3$

**Optimizer 更新：**

- $w \\leftarrow w - \\eta \\nabla_w L = 1.0 - 0.1 \\cdot (-6) = 1.6$
- $b \\leftarrow b - \\eta \\nabla_b L = 0.0 - 0.1 \\cdot (-3) = 0.3$

### 第 2 步

**前向传播：**

- $\\hat{y} = wx + b = 1.6 \\cdot 2 + 0.3 = 3.5$
- 误差 $e = \\hat{y} - y = 3.5 - 5 = -1.5$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-1.5)^2 = 1.125$

**反向传播：**

- $\\nabla_{\\hat{y}} L = e = -1.5$
- $\\nabla_w L = e \\cdot x = -1.5 \\cdot 2 = -3$
- $\\nabla_b L = e = -1.5$

**Optimizer 更新：**

- $w = 1.6 - 0.1 \\cdot (-3) = 1.9$
- $b = 0.3 - 0.1 \\cdot (-1.5) = 0.45$

### 第 3 步

**前向传播：**

- $\\hat{y} = wx + b = 1.9 \\cdot 2 + 0.45 = 4.25$
- 误差 $e = \\hat{y} - y = 4.25 - 5 = -0.75$
- Loss $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-0.75)^2 = 0.28125$

**反向传播：**

- $\\nabla_{\\hat{y}} L = e = -0.75$
- $\\nabla_w L = e \\cdot x = -0.75 \\cdot 2 = -1.5$
- $\\nabla_b L = e = -0.75$

**Optimizer 更新：**

- $w = 1.9 - 0.1 \\cdot (-1.5) = 2.05$
- $b = 0.45 - 0.1 \\cdot (-0.75) = 0.525$

> 可以看到 loss 每一步都在下降（4.5 → 1.125 → 0.28125）。这就是 optimizer 利用反向传播得到的 gradient，把 $w, b$ 推向更好的值。

## 普通 SGD 的两个问题

SGD 能用，但它有两个问题，后来的 optimizer 就是为了解决它们而出现的：

1. **梯度很"吵"。** 真实训练中，每一步只用一个小 batch 来估算梯度（这个网站的代码里是 4 × 32 = 128 个 token），所以每一步的方向里都有一部分是噪声，走出来的路线是"之"字形，而不是直接往下坡走。解决办法：[momentum](#info:sgd_momentum_notes)，对最近几步的梯度求平均，让噪声互相抵消。
2. **所有参数共用一个学习率。** 不同参数的梯度大小可能差很多。假设两个参数的梯度分别约为 100 和 0.01，$\\eta = 0.01$：

   \`\`\`text
   参数 A:  每步移动 0.01 × 100  = 1        ← 可能太大，来回震荡
   参数 B:  每步移动 0.01 × 0.01 = 0.0001   ← 太小，几乎学不动
   \`\`\`

   把 $\\eta$ 调大，A 会发散；调小，B 就学不动。没有一个学习率能同时照顾两者。解决办法：给每个参数自己的步长，也就是 [Adagrad](#info:adagrad_notes) 的做法。

> 上面的例子只有一个数据点，所以梯度里没有噪声，而且 $w$ 和 $b$ 的梯度大小也差不多。这就是普通 SGD 在那个例子里表现很好的原因。
`,
    },
  },

  sgd_momentum_notes: {
    en: {
      title: "SGD with momentum",
      md: `
**Instead of stepping with the current gradient alone, keep a velocity: a running average of past gradients, and step with that.**

## What it is

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

- $g = \\nabla_{\\theta} L$ is the current gradient.
- $\\beta$ (usually 0.9) controls how much "memory" to keep.
- $\\eta$ is the learning rate.

Each step keeps 90% of the old velocity and adds 10% of the new gradient. Unrolled, $v$ is a weighted average of all past gradients: the newest has weight 0.1, the one before $0.1 \\times 0.9$, then $0.1 \\times 0.81$, and so on, fading by a factor 0.9 per step. This is an **exponential moving average** (EMA). With $\\beta = 0.9$ it covers roughly the last $\\frac{1}{1 - \\beta} = 10$ steps.

The name comes from physics: $v$ behaves like a ball with inertia. Pushed the same way step after step, it keeps rolling that way; pushed back and forth, it barely moves.

## Why it helps: noise cancels

Plain [SGD](#info:sgd_notes) steps with each batch's gradient, and part of every such gradient is noise. Momentum averages the noise away.

Take a parameter with two coordinates. The true downhill direction is $x$, but every batch adds a large noise in $y$ whose sign flips from step to step:

\`\`\`text
odd steps:   g = [1, +3]
even steps:  g = [1, −3]
\`\`\`

**SGD** steps with $g$ itself: each step moves 1 in $x$ and ±3 in $y$. The noise (3) is larger than the signal (1), so the path zigzags across far more than it moves forward.

**Momentum** ($\\beta = 0.9$) steps with $v$:

\`\`\`text
step  1:  v = [0.100,  0.300]
step  2:  v = [0.190, -0.030]
step  3:  v = [0.271,  0.273]
step  4:  v = [0.344, -0.054]
step 10:  v = [0.651, -0.103]
step 20:  v = [0.878, -0.139]
step 40:  v = [0.985, -0.156]
\`\`\`

- **$x$ (signal):** every gradient is 1, so their average is 1. $v_x$ climbs from 0 toward 1.
- **$y$ (noise):** +3 and −3 alternate, so they cancel in the average. $v_y$ ends up swinging only about ±0.16.

Where ±0.16 comes from: once things settle, $v_y$ alternates between $+a$ and $-a$. An odd step turns $-a$ into $+a$: $a = 0.9 \\cdot (-a) + 0.1 \\cdot 3$, so $1.9a = 0.3$ and $a = \\frac{0.3}{1.9} \\approx 0.16$.

The noise shrinks from 3 to about 0.16, roughly 1/19 of its size, while the signal is kept. The path becomes nearly straight. The same thing happens in a narrow valley (an "ill-conditioned" loss), where the gradient keeps flipping across the valley while pointing consistently along it.

## The cost: a slow start

SGD has no velocity: from step 1 it moves the full gradient. Momentum's $v$ starts at 0 and has to build up. Compare the $x$ direction above:

\`\`\`text
          SGD steps with    momentum steps with v
step  1:       1                 0.100
step  2:       1                 0.190
step  3:       1                 0.271
step 10:       1                 0.651
step 20:       1                 0.878
step 40:       1                 0.985   → approaches 1
\`\`\`

Each value is 0.9 × the previous one plus 0.1: $0.1$, then $0.1 \\cdot 0.9 + 0.1 = 0.19$, then $0.19 \\cdot 0.9 + 0.1 = 0.271$, and so on.

- **Early on**, momentum is slower than SGD: it takes about 10+ steps to warm up.
- **Once warmed up**, both move at the same speed along the signal: $\\eta \\times 1$ per step. In the EMA form, $v$ settles at the average gradient and never exceeds it, because each step adds only $(1 - \\beta) = 10\\%$ of the new gradient.

So momentum is not faster along the signal; its gain is in the noise direction (±3 per step for SGD vs about ±0.16). [Adam](#info:adam_notes) removes the slow start with a "bias correction".

## Full worked example: one data point, no noise

This example has a single data point, so the gradient has no noise to cancel. It shows only the cost (the slow start), step by step, so it is a useful contrast to the example above.

### Setup

We use the same setup as the numeric example in the [SGD notes](#info:sgd_notes).

- Model: $\\hat{y} = wx + b$
- Loss: $L = \\frac{1}{2} (\\hat{y} - y)^2$, where $y$ is the target value
- Data point: $x = 2$, $y = 5$
- Init: $w = 1.0$, $b = 0.0$, $v_w = v_b = 0$
- Hyperparameters: learning rate $\\eta = 0.1$, momentum $\\beta = 0.9$

Remember, the gradients of the loss with respect to $w$ and $b$ are:

$$
\\nabla_w L = (\\hat{y} - y) \\cdot x = e \\cdot x \\\\[4pt]
\\nabla_b L = \\hat{y} - y = e
$$

### Step 1

- Forward pass: $\\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2$
- Error: $e = \\hat{y} - y = 2 - 5 = -3$
- Loss: $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5$
- Gradients: $\\nabla_w L = e \\cdot x = -3 \\cdot 2 = -6$, $\\nabla_b L = e = -3$

**Velocity:**

$$
v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-6) = -0.6 \\\\[4pt]
v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-3) = -0.3
$$

**Optimizer update:**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.0 - 0.1 \\cdot (-0.6) = 1.06 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.0 - 0.1 \\cdot (-0.3) = 0.03
$$

### Step 2

- Forward pass: $\\hat{y} = wx + b = 1.06 \\cdot 2 + 0.03 = 2.15$
- Error: $e = \\hat{y} - y = 2.15 - 5 = -2.85$
- Loss: $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.85)^2 = 4.0613$
- Gradients: $\\nabla_w L = e \\cdot x = -2.85 \\cdot 2 = -5.7$, $\\nabla_b L = e = -2.85$

**Velocity:**

$$
v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot (-0.6) + (1 - 0.9) \\cdot (-5.7) = -0.54 + (-0.57) = -1.11 \\\\[4pt]
v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot (-0.3) + (1 - 0.9) \\cdot (-2.85) = -0.27 + (-0.285) = -0.555
$$

**Optimizer update:**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.06 - 0.1 \\cdot (-1.11) = 1.171 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.03 - 0.1 \\cdot (-0.555) = 0.0855
$$

### Step 3

- Forward pass: $\\hat{y} = wx + b = 1.171 \\cdot 2 + 0.0855 = 2.4275$
- Error: $e = \\hat{y} - y = 2.4275 - 5 = -2.5725$
- Loss: $L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.5725)^2 = 3.3089$
- Gradients: $\\nabla_w L = e \\cdot x = -2.5725 \\cdot 2 = -5.145$, $\\nabla_b L = e = -2.5725$

**Velocity:**

$$
\\begin{aligned}
v_w &= \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot (-1.11) + (1 - 0.9) \\cdot (-5.145) \\\\
&= -0.999 + (-0.5145) = -1.5135 \\\\[4pt]
v_b &= \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot (-0.555) + (1 - 0.9) \\cdot (-2.5725) \\\\
&= -0.4995 + (-0.25725) = -0.75675
\\end{aligned}
$$

**Optimizer update:**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.171 - 0.1 \\cdot (-1.5135) = 1.32235 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.0855 - 0.1 \\cdot (-0.75675) = 0.161175
$$

> You can see the loss decreasing at each step (4.5 → 4.0613 → 3.3089). That's the optimizer using backprop's gradients to move $w, b$ toward better values. Compare with SGD, where the loss decreases faster (4.5 → 1.125 → 0.28125) in the first 3 steps. The reason is the slow start explained above: in steps 1–3 the velocity is only 0.1×, 0.19× and 0.271× of a full gradient.

## Why momentum loses on this example

There is no noise here, so momentum pays the cost without getting the benefit. It is also hurt later: as $w$ and $b$ approach the target, the gradient shrinks, but $v$ still remembers the larger gradients of earlier steps. So momentum overshoots the target and oscillates around it. After 10 steps momentum's loss is about 0.44, versus about $4 \\times 10^{-6}$ for SGD.

Momentum pays off on harder problems: noisy gradients (as in the example above) and narrow valleys. Training a language model on small batches is exactly that kind of problem.

## Making momentum look better earlier

If we want momentum to look better earlier, we can:

1. Lower $\\beta$ (e.g., 0.8), so the velocity builds up faster.
2. Slightly raise $\\eta$ (e.g., 0.12) to make up for the smaller steps early on.

## PyTorch's form is different

This note uses the EMA form $v \\leftarrow \\beta v + (1 - \\beta) g$, which is also the form [Adam](#info:adam_notes) uses for its first moment. PyTorch's \`torch.optim.SGD(momentum=0.9)\` uses a different form, without the $(1 - \\beta)$:

$$
v \\leftarrow \\beta v + g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

With a constant gradient of 1, this $v$ settles at $\\frac{1}{1 - \\beta} = 10$ instead of 1. The two forms differ by exactly the factor $(1 - \\beta)$ at every step, so PyTorch's form with learning rate $\\eta$ takes the same steps as the EMA form with learning rate $\\frac{\\eta}{1 - \\beta} = 10\\eta$. Keep this in mind when copying a learning rate from one form to the other.
`,
    },
    zh: {
      title: "SGD with momentum（带动量的 SGD）",
      md: `
**不再只用当前的梯度迈步，而是维护一个速度（velocity）：过去梯度的滑动平均，用它来迈步。**

## 它是什么

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

- $g = \\nabla_{\\theta} L$ 是当前的梯度。
- $\\beta$（通常取 0.9）控制保留多少"记忆"。
- $\\eta$ 是学习率。

每一步，速度保留 90% 的旧值，再加上 10% 的新梯度。展开来看，$v$ 是所有过去梯度的加权平均：最新的梯度权重是 0.1，前一个是 $0.1 \\times 0.9$，再前一个是 $0.1 \\times 0.81$，每往前一步就再乘一次 0.9。这种平均叫**指数滑动平均**（EMA，exponential moving average）。$\\beta = 0.9$ 时，大约相当于对最近 $\\frac{1}{1 - \\beta} = 10$ 步求平均。

"动量"这个名字来自物理：$v$ 就像一个有惯性的球。一直朝同一个方向推它，它就一直朝那个方向滚；左右来回推，它基本不动。

## 为什么有用：噪声会互相抵消

普通的 [SGD](#info:sgd_notes) 用每个 batch 的梯度直接迈步，而每个这样的梯度里都有一部分是噪声。momentum 通过求平均把噪声去掉。

看一个两维参数的例子。真正的下坡方向是 $x$，但每个 batch 都在 $y$ 方向上带来很大的噪声，而且正负每一步都在变：

\`\`\`text
第奇数步:  g = [1, +3]
第偶数步:  g = [1, −3]
\`\`\`

**SGD** 直接用 $g$ 迈步：每一步在 $x$ 方向走 1，在 $y$ 方向走 ±3。噪声（3）比有用的信号（1）还大，所以路线来回横摆的幅度远大于往前走的距离，走成"之"字形。

**Momentum**（$\\beta = 0.9$）用 $v$ 迈步：

\`\`\`text
step  1:  v = [0.100,  0.300]
step  2:  v = [0.190, -0.030]
step  3:  v = [0.271,  0.273]
step  4:  v = [0.344, -0.054]
step 10:  v = [0.651, -0.103]
step 20:  v = [0.878, -0.139]
step 40:  v = [0.985, -0.156]
\`\`\`

- **$x$ 方向（信号）：** 每一步的梯度都是 1，平均下来还是 1。$v_x$ 从 0 慢慢涨到接近 1。
- **$y$ 方向（噪声）：** +3 和 −3 交替出现，平均时互相抵消。$v_y$ 最后只在 ±0.16 左右摆动。

±0.16 是怎么来的：稳定之后，$v_y$ 在 $+a$ 和 $-a$ 之间交替。奇数步把 $-a$ 变成 $+a$：$a = 0.9 \\cdot (-a) + 0.1 \\cdot 3$，所以 $1.9a = 0.3$，$a = \\frac{0.3}{1.9} \\approx 0.16$。

噪声从 3 降到约 0.16，大约缩小到原来的 1/19，而信号被保留了下来，走出来的路线几乎是笔直的。在狭窄的山谷里（"病态"的 loss，ill-conditioned）也是同样的道理：梯度在横跨山谷的方向上来回翻转，在沿着山谷的方向上始终一致。

## 代价：起步慢

SGD 没有速度：从第 1 步开始就按完整的梯度走。momentum 的 $v$ 从 0 开始，需要慢慢积累。对比上面 $x$ 方向：

\`\`\`text
          SGD 用的值     momentum 用的 v
step  1:     1              0.100
step  2:     1              0.190
step  3:     1              0.271
step 10:     1              0.651
step 20:     1              0.878
step 40:     1              0.985   → 最终接近 1
\`\`\`

每个值都是前一个值 × 0.9 再加 0.1：$0.1$，然后 $0.1 \\cdot 0.9 + 0.1 = 0.19$，然后 $0.19 \\cdot 0.9 + 0.1 = 0.271$，以此类推。

- **前期**，momentum 比 SGD 慢：要大约 10 步以上才"热身"完毕。
- **热身之后**，两者在信号方向上一样快：每步都走 $\\eta \\times 1$。在 EMA 形式下，$v$ 最终停在梯度的平均值上，不会超过它，因为每一步只加入 $(1 - \\beta) = 10\\%$ 的新梯度。

所以 momentum 在信号方向上并不比 SGD 快；它的好处在噪声方向上（SGD 每步摆动 ±3，momentum 只摆动约 ±0.16）。[Adam](#info:adam_notes) 用 "bias correction" 解决了起步慢的问题。

## 完整的逐步算例：一个数据点，没有噪声

这个例子只有一个数据点，所以梯度里没有可以抵消的噪声。它只展示了代价（起步慢），并且一步步算出来，正好和上面的例子形成对比。

### 设置

我们使用与 [SGD 笔记](#info:sgd_notes)中数值例子相同的设置。

- 模型：$\\hat{y} = wx + b$
- Loss：$L = \\frac{1}{2} (\\hat{y} - y)^2$，其中 $y$ 是目标值
- 数据点：$x = 2$，$y = 5$
- 初始值：$w = 1.0$，$b = 0.0$，$v_w = v_b = 0$
- 超参数：学习率 $\\eta = 0.1$，momentum $\\beta = 0.9$

回忆一下，loss 对 $w$ 和 $b$ 的 gradient 是：

$$
\\nabla_w L = (\\hat{y} - y) \\cdot x = e \\cdot x \\\\[4pt]
\\nabla_b L = \\hat{y} - y = e
$$

### 第 1 步

- 前向传播：$\\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2$
- 误差：$e = \\hat{y} - y = 2 - 5 = -3$
- Loss：$L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5$
- Gradient：$\\nabla_w L = e \\cdot x = -3 \\cdot 2 = -6$，$\\nabla_b L = e = -3$

**速度：**

$$
v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-6) = -0.6 \\\\[4pt]
v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-3) = -0.3
$$

**Optimizer 更新：**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.0 - 0.1 \\cdot (-0.6) = 1.06 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.0 - 0.1 \\cdot (-0.3) = 0.03
$$

### 第 2 步

- 前向传播：$\\hat{y} = wx + b = 1.06 \\cdot 2 + 0.03 = 2.15$
- 误差：$e = \\hat{y} - y = 2.15 - 5 = -2.85$
- Loss：$L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.85)^2 = 4.0613$
- Gradient：$\\nabla_w L = e \\cdot x = -2.85 \\cdot 2 = -5.7$，$\\nabla_b L = e = -2.85$

**速度：**

$$
v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot (-0.6) + (1 - 0.9) \\cdot (-5.7) = -0.54 + (-0.57) = -1.11 \\\\[4pt]
v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot (-0.3) + (1 - 0.9) \\cdot (-2.85) = -0.27 + (-0.285) = -0.555
$$

**Optimizer 更新：**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.06 - 0.1 \\cdot (-1.11) = 1.171 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.03 - 0.1 \\cdot (-0.555) = 0.0855
$$

### 第 3 步

- 前向传播：$\\hat{y} = wx + b = 1.171 \\cdot 2 + 0.0855 = 2.4275$
- 误差：$e = \\hat{y} - y = 2.4275 - 5 = -2.5725$
- Loss：$L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.5725)^2 = 3.3089$
- Gradient：$\\nabla_w L = e \\cdot x = -2.5725 \\cdot 2 = -5.145$，$\\nabla_b L = e = -2.5725$

**速度：**

$$
\\begin{aligned}
v_w &= \\beta \\cdot v_w + (1 - \\beta) \\cdot \\nabla_w L = 0.9 \\cdot (-1.11) + (1 - 0.9) \\cdot (-5.145) \\\\
&= -0.999 + (-0.5145) = -1.5135 \\\\[4pt]
v_b &= \\beta \\cdot v_b + (1 - \\beta) \\cdot \\nabla_b L = 0.9 \\cdot (-0.555) + (1 - 0.9) \\cdot (-2.5725) \\\\
&= -0.4995 + (-0.25725) = -0.75675
\\end{aligned}
$$

**Optimizer 更新：**

$$
w \\leftarrow w - \\eta \\cdot v_w = 1.171 - 0.1 \\cdot (-1.5135) = 1.32235 \\\\[4pt]
b \\leftarrow b - \\eta \\cdot v_b = 0.0855 - 0.1 \\cdot (-0.75675) = 0.161175
$$

> 可以看到 loss 每一步都在下降（4.5 → 4.0613 → 3.3089）。这就是 optimizer 利用反向传播得到的 gradient，把 $w, b$ 推向更好的值。和 SGD 对比一下：前 3 步里，SGD 的 loss 下降得更快（4.5 → 1.125 → 0.28125）。原因就是上面讲的起步慢：第 1–3 步的速度只有完整梯度的 0.1 倍、0.19 倍和 0.271 倍。

## 为什么 momentum 在这个例子里输了

这里没有噪声，所以 momentum 只付出了代价，没有得到好处。后期它还会吃亏：当 $w$ 和 $b$ 接近目标时，梯度变小了，但 $v$ 还记着前面那些更大的梯度，于是 momentum 会冲过目标，在它附近来回振荡。10 步之后，momentum 的 loss 约为 0.44，而 SGD 约为 $4 \\times 10^{-6}$。

momentum 在更难的问题上才有回报：梯度有噪声（就像上面那个例子），或者 loss 是狭窄的山谷。用小 batch 训练语言模型，正是这类问题。

## 让 momentum 更早显出优势

如果想让 momentum 更早表现得更好，我们可以：

1. 降低 $\\beta$（例如 0.8），让速度积累得更快。
2. 稍微提高 $\\eta$（例如 0.12），弥补早期步子偏小的问题。

## PyTorch 的写法不一样

这篇用的是 EMA 形式 $v \\leftarrow \\beta v + (1 - \\beta) g$，[Adam](#info:adam_notes) 的一阶矩也是这种形式。但 PyTorch 的 \`torch.optim.SGD(momentum=0.9)\` 用的是另一种写法，没有 $(1 - \\beta)$：

$$
v \\leftarrow \\beta v + g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

梯度一直是 1 时，这个 $v$ 最终停在 $\\frac{1}{1 - \\beta} = 10$，而不是 1。两种写法在每一步都正好差一个 $(1 - \\beta)$ 的系数，所以 PyTorch 写法用学习率 $\\eta$，和 EMA 写法用学习率 $\\frac{\\eta}{1 - \\beta} = 10\\eta$ 走的步子完全一样。在两种写法之间照搬学习率时要注意这一点。
`,
    },
  },

  adagrad_notes: {
    en: {
      title: "Adagrad",
      md: `
**Adagrad (Adaptive gradient)** gives each parameter its own learning rate, set automatically from that parameter's past gradients.

It fixes the second of the [two problems of SGD](#info:sgd_notes) (see the end of the SGD note): one learning rate for all parameters.

## Update rule

Adagrad keeps a running **sum of squared gradients** for each parameter $i$:

$$
s_i \\leftarrow s_i + g_i^2
$$

where $g_i$ is the gradient of parameter $i$ (a single number) and $s_i$ starts at 0. It then updates the parameter with the learning rate divided by $\\sqrt{s_i}$:

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

$\\epsilon$ is a tiny constant (e.g. $10^{-8}$) that only prevents division by zero.

Two quantities are worth keeping apart:

- **Effective learning rate** $\\eta_{\\text{eff},\\,i} = \\frac{\\eta}{\\sqrt{s_i} + \\epsilon}$: parameter $i$'s own learning rate. It depends on how much squared gradient the parameter has **accumulated so far**.
- **Actual step** $\\eta_{\\text{eff},\\,i} \\cdot g_i$: how far the parameter really moves. It is the effective learning rate times the current gradient.

### Vector form and the $\\odot$ symbol

Written for all parameters at once, as vectors, the accumulator is $s \\leftarrow s + g \\odot g$. The symbol $\\odot$ means **element-wise multiplication**: multiply the numbers in matching positions; the result is a vector of the same shape. For example, $(-3, 2) \\odot (-3, 2) = (9, 4)$. (Unlike the dot product $(-3, 2) \\cdot (-3, 2) = 9 + 4 = 13$, nothing is added up.) So $g \\odot g$ squares each parameter's gradient separately: each parameter accumulates only its own squared gradients, which is why each one gets its own learning rate. In PyTorch this is \`g * g\`.

## Effect 1: the step size does not depend on the gradient's scale

Take two parameters: A's gradient is always 100, B's is always 0.01, and $\\eta = 0.01$.

**SGD** moves each by $\\eta \\cdot g$:

\`\`\`text
A: 0.01 × 100  = 1      per step
B: 0.01 × 0.01 = 0.0001 per step      ← 10,000× apart
\`\`\`

**Adagrad, step 1:**

\`\`\`text
A: s = 100²  = 10000,   √s = 100   →  moves 0.01 × 100  / 100  = 0.01
B: s = 0.01² = 0.0001,  √s = 0.01  →  moves 0.01 × 0.01 / 0.01 = 0.01
\`\`\`

Both take the **same** step, 0.01. A's effective learning rate is much smaller than B's, but multiplied by its much larger gradient, the actual step comes out the same.

The reason: the numerator is $g$, and the denominator $\\sqrt{s}$ is proportional to the size of $g$. Scale the gradient by 10,000 and both scale together, so their ratio does not change. Each step moves about $\\eta$, in the direction given by the gradient's sign; how large the gradient is no longer matters.

At step $t$, with a constant gradient $g$:

$$
s = t\\, g^2, \\qquad \\sqrt{s} = \\sqrt{t}\\, |g|, \\qquad \\text{step} = \\frac{\\eta\\, g}{\\sqrt{t}\\, |g|} = \\pm\\frac{\\eta}{\\sqrt{t}}
$$

The same for A and B.

## Effect 2: a parameter that rarely gets a gradient takes larger steps

What matters is not only how **large** a parameter's gradients are, but how **often** it gets one. $s$ records the total accumulated so far, so a parameter that rarely gets a gradient keeps a small $s$.

Example: a common word's embedding row gets a gradient of 1 every step; a rare word's row gets a gradient of 1 only once every 100 steps. At step 10,000:

\`\`\`text
common word:  10,000 gradients so far,  s = 10,000,  √s = 100
              step now:  η × 1 / 100 = 0.01 η

rare word:    100 gradients so far,     s = 100,     √s = 10
              step when it appears:  η × 1 / 10 = 0.1 η
\`\`\`

The gradients are the same size (1), yet when the rare word appears it moves **10 times** as far as the common word. A rare word gets few chances to learn, so each chance counts for more; a common word, updated almost every step, takes small steps instead of being pushed around again and again. This is why Adagrad became popular for word vectors.

The same situation exists in GPT-2: the [token embedding](#info:embedding_tok) row of a rare token receives an input-side gradient only when that token is in the batch.

## Why the square root?

**It is what makes Effect 1 work.** The step is $\\frac{g}{\\sqrt{s}}$ times $\\eta$, and $\\sqrt{s}$ grows in proportion to the size of $g$, so the two cancel. Dividing by $s$ instead would give $\\frac{g}{t\\, g^2} = \\frac{1}{t\\, g}$: larger gradients would get *smaller* steps, the opposite of what we want.

**It gives a gentle decay.** When the gradients keep a similar size, $s_i$ grows about linearly with time, $s_i \\approx t\\, g_i^2$, so

$$
\\eta_{\\text{eff},\\,i} \\approx \\frac{\\eta}{|g_i| \\sqrt{t}}
$$

and the actual step shrinks like $\\frac{\\eta}{\\sqrt{t}}$. This decay is steady but not so fast that learning dies early. Dividing by $s_i$ would instead give a $\\frac{1}{t}$ decay, which becomes tiny too quickly.

> The $\\frac{1}{\\sqrt{t}}$ rate also matches classic results in online convex optimization, where it gives good regret/convergence bounds.

## Worked example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Accumulator:** $s \\leftarrow s + g \\odot g$, starting from $s_0 = (0, 0)$
- **Update:** $\\theta \\leftarrow \\theta - \\Delta$, with $\\Delta = \\eta_{\\text{eff}} \\cdot g$ and $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **Hyperparameters:** $\\eta = 1.0$, $\\epsilon = 10^{-8}$ (negligible here)
- **Init:** $\\theta_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)$
- $\\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3,\\; 1/2)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-1, 1) = (1, -1)$

The two coordinates have different effective learning rates, $1/3$ and $1/2$, yet the gradients were $-3$ and $2$ and **both coordinates moved exactly 1** $= \\eta$. This is Effect 1: on the first step $s = g^2$, so every coordinate moves $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$.

### Step 2

- $\\theta_1 = (1, -1)$
- $g_2 = (1 - 3,\\; -1 + 2) = (-2, 1)$
- $s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)$
- $\\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)$
- $\\theta_2 = \\theta_1 - \\Delta = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)$

Both effective learning rates shrank: $1/3 \\to 0.277$ and $1/2 \\to 0.447$.

### Step 3

- $\\theta_2 = (1.554, -1.447)$
- $g_3 = (1.554 - 3,\\; -1.447 + 2) = (-1.446, 0.553)$
- $s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)$
- $\\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)$
- $\\theta_3 = \\theta_2 - \\Delta = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)$

They shrank again: $0.277 \\to 0.257$ and $0.447 \\to 0.434$.

### What the example shows

- Each coordinate has its own effective learning rate, and it only shrinks as $s$ accumulates.
- $\\theta_1$ has the larger gradients, so it gets the **smaller** effective learning rate ($0.257$ vs $0.434$ by step 3). Its actual steps are not smaller, though: equal in step 1 (both 1), and after that even slightly larger ($0.554$ vs $0.447$, $0.372$ vs $0.24$), because its larger gradient makes up for the smaller learning rate. (Its gradients stay larger because it started farther from its target: 3 away vs 2.) Adagrad evens the coordinates out instead of letting the larger gradient dominate.

## The problem: the steps only shrink

$s$ is a running **sum**; it only grows, never shrinks. So the effective learning rate $\\frac{\\eta}{\\sqrt{s}}$ only gets smaller. With a gradient of constant size:

\`\`\`text
step      1:  moves η / 1
step    100:  moves η / 10
step 10,000:  moves η / 100
\`\`\`

Training a language model takes hundreds of thousands of steps; by then the steps are so small that learning stalls. And if a parameter's gradients were large early on and become small later, $s$ does not go down: the past keeps holding its steps down.

\`\`\`text
first 100 steps, gradient 10:   s = 100 × 10² = 10,000,  √s = 100
step 101, gradient 1:           moves η × 1 / 100 = 0.01 η    ← held down by the past
\`\`\`

## Summary

| Case | Actual step |
|---|---|
| Gradients always large vs always small | About the same size: Adagrad evens them out (Effect 1) |
| Gradients frequent vs rare | Larger for the rare one, when it gets a gradient (Effect 2) |
| Large gradients in the past, small now | Small now: held down by the past (the problem) |

[RMSProp](#info:rmsprop_notes) fixes the problem: instead of summing the whole history, it averages only the recent past.
`,
    },
    zh: {
      title: "Adagrad",
      md: `
**Adagrad（Adaptive gradient，自适应梯度）** 给每个参数分配自己的学习率，并根据这个参数过去的梯度自动设定。

它解决的是 [SGD 的两个问题](#info:sgd_notes)中的第二个（见 SGD note 的最后一节）：所有参数共用一个学习率。

## 更新规则

Adagrad 为每个参数 $i$ 维护一个**梯度平方的累加和**：

$$
s_i \\leftarrow s_i + g_i^2
$$

其中 $g_i$ 是参数 $i$ 的梯度（一个数），$s_i$ 从 0 开始。然后用学习率除以 $\\sqrt{s_i}$ 来更新参数：

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

$\\epsilon$ 是一个很小的常数（比如 $10^{-8}$），只是为了防止除以 0。

有两个量要分清楚：

- **有效学习率** $\\eta_{\\text{eff},\\,i} = \\frac{\\eta}{\\sqrt{s_i} + \\epsilon}$：参数 $i$ 自己的学习率。它取决于这个参数**到目前为止累计**了多少梯度平方。
- **实际步子** $\\eta_{\\text{eff},\\,i} \\cdot g_i$：参数真正移动的距离，等于有效学习率乘以当前的梯度。

### 向量写法和 $\\odot$ 符号

把所有参数写成向量、一起表示，累加器就是 $s \\leftarrow s + g \\odot g$。符号 $\\odot$ 表示**逐元素相乘**：两个形状相同的向量，对应位置的数各自相乘，结果还是同样形状的向量。比如 $(-3, 2) \\odot (-3, 2) = (9, 4)$。（它和点积不同：点积 $(-3, 2) \\cdot (-3, 2) = 9 + 4 = 13$ 要加起来，逐元素相乘不加。）所以 $g \\odot g$ 就是把每个参数的梯度各自平方：每个参数只累加自己的梯度平方，这正是每个参数能有自己学习率的原因。在 PyTorch 里就是 \`g * g\`。

## 效果 1：步长和梯度的大小无关

假设有两个参数：A 的梯度一直是 100，B 的梯度一直是 0.01，$\\eta = 0.01$。

**SGD** 每个参数移动 $\\eta \\cdot g$：

\`\`\`text
A: 0.01 × 100  = 1       每步
B: 0.01 × 0.01 = 0.0001  每步      ← 差了 10000 倍
\`\`\`

**Adagrad 第 1 步：**

\`\`\`text
A: s = 100²  = 10000,   √s = 100   →  移动 0.01 × 100  / 100  = 0.01
B: s = 0.01² = 0.0001,  √s = 0.01  →  移动 0.01 × 0.01 / 0.01 = 0.01
\`\`\`

两个参数走的步子**一样大**，都是 0.01。A 的有效学习率比 B 小得多，但乘上它大得多的梯度以后，实际步子是一样的。

原因是：分子是 $g$，分母 $\\sqrt{s}$ 和 $g$ 的大小成正比。梯度放大 10000 倍，分子和分母一起放大，比值不变。所以每一步大约走 $\\eta$ 那么远，方向由梯度的正负决定，梯度本身有多大已经不重要了。

梯度一直是 $g$ 时，第 $t$ 步：

$$
s = t\\, g^2, \\qquad \\sqrt{s} = \\sqrt{t}\\, |g|, \\qquad \\text{步长} = \\frac{\\eta\\, g}{\\sqrt{t}\\, |g|} = \\pm\\frac{\\eta}{\\sqrt{t}}
$$

A 和 B 完全一样。

## 效果 2：很少拿到梯度的参数，步子更大

重要的不只是参数的梯度有多**大**，还有它多**常**拿到梯度。$s$ 记录的是到目前为止累计的总量，所以很少拿到梯度的参数，$s$ 一直很小。

例子：常见词的 embedding 行每一步都拿到梯度 1；罕见词的那一行每 100 步才拿到一次梯度 1。到第 10000 步时：

\`\`\`text
常见词:  拿到过 10000 次梯度,  s = 10000,  √s = 100
         这一步走:  η × 1 / 100 = 0.01 η

罕见词:  只拿到过 100 次梯度,  s = 100,    √s = 10
         它出现的这一步走:  η × 1 / 10 = 0.1 η
\`\`\`

梯度大小一样（都是 1），但罕见词出现时，走的距离是常见词的 **10 倍**。罕见词学习的机会少，所以每次机会都多学一点；常见词几乎每一步都在更新，步子小一些，不会被反复推来推去。这就是 Adagrad 当年在词向量上特别成功的原因。

GPT-2 里也有同样的情况：罕见 token 的 [token embedding](#info:embedding_tok) 那一行，只有在这个 token 出现在 batch 里时，才会收到输入端的梯度。

## 为什么要开平方根？

**效果 1 就靠它。** 步长是 $\\eta$ 乘以 $\\frac{g}{\\sqrt{s}}$，而 $\\sqrt{s}$ 和 $g$ 的大小成正比，两者正好抵消。如果改成除以 $s$，就会得到 $\\frac{g}{t\\, g^2} = \\frac{1}{t\\, g}$：梯度越大，步子反而越**小**，和我们想要的正好相反。

**衰减比较温和。** 梯度大小保持差不多时，$s_i$ 大致随时间线性增长，$s_i \\approx t\\, g_i^2$，所以

$$
\\eta_{\\text{eff},\\,i} \\approx \\frac{\\eta}{|g_i| \\sqrt{t}}
$$

实际步子按 $\\frac{\\eta}{\\sqrt{t}}$ 缩小。这种衰减稳步进行，但又不会快到让学习过早停滞。如果除以 $s_i$，就会变成 $\\frac{1}{t}$ 的衰减，步子很快就变得极小。

> $\\frac{1}{\\sqrt{t}}$ 的速率也和经典的在线凸优化（online convex optimization）结果一致：在那里它能给出良好的 regret/收敛界。

## 数值例子

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$，在 $\\theta = (3, -2)$ 处最小
- **梯度：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **累加器：** $s \\leftarrow s + g \\odot g$，从 $s_0 = (0, 0)$ 开始
- **更新：** $\\theta \\leftarrow \\theta - \\Delta$，其中 $\\Delta = \\eta_{\\text{eff}} \\cdot g$，$\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **超参数：** $\\eta = 1.0$，$\\epsilon = 10^{-8}$（这里可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)$
- $\\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3,\\; 1/2)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-1, 1) = (1, -1)$

两个坐标的有效学习率不同，分别是 $1/3$ 和 $1/2$；梯度分别是 $-3$ 和 $2$，但**两个坐标都正好走了 1** $= \\eta$。这就是效果 1：第 1 步时 $s = g^2$，所以每个坐标都移动 $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$。

### 第 2 步

- $\\theta_1 = (1, -1)$
- $g_2 = (1 - 3,\\; -1 + 2) = (-2, 1)$
- $s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)$
- $\\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)$
- $\\theta_2 = \\theta_1 - \\Delta = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)$

两个有效学习率都变小了：$1/3 \\to 0.277$，$1/2 \\to 0.447$。

### 第 3 步

- $\\theta_2 = (1.554, -1.447)$
- $g_3 = (1.554 - 3,\\; -1.447 + 2) = (-1.446, 0.553)$
- $s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)$
- $\\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)$
- $\\theta_3 = \\theta_2 - \\Delta = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)$

又变小了：$0.277 \\to 0.257$，$0.447 \\to 0.434$。

### 这个例子说明了什么

- 每个坐标都有自己的有效学习率，而且随着 $s$ 的累积只会变小。
- $\\theta_1$ 的梯度更大，所以它的有效学习率**更小**（到第 3 步是 $0.257$ 对 $0.434$）。但它的实际步子并不更小：第 1 步两者相等（都是 1），之后甚至稍大一些（$0.554$ 对 $0.447$，$0.372$ 对 $0.24$），因为更大的梯度弥补了更小的学习率。（它的梯度一直更大，是因为它离目标更远：一开始差 3，而 $\\theta_2$ 只差 2。）Adagrad 把两个坐标拉平，而不是让梯度大的那个占主导。

## 它的问题：步长只会变小

$s$ 是**累加和**，只增不减。所以有效学习率 $\\frac{\\eta}{\\sqrt{s}}$ 只会越来越小。梯度大小不变时：

\`\`\`text
step      1:  移动 η / 1
step    100:  移动 η / 10
step 10,000:  移动 η / 100
\`\`\`

训练语言模型要跑几十万步，到后面步子小到几乎不动，训练就停滞了。而且，如果某个参数早期梯度很大、后来变小了，$s$ 也不会减少：过去的大梯度会一直压着它的步子。

\`\`\`text
前 100 步梯度都是 10:   s = 100 × 10² = 10000,  √s = 100
第 101 步梯度变成 1:    移动 η × 1 / 100 = 0.01 η    ← 被过去压住了
\`\`\`

## 总结

| 情况 | 实际步子 |
|---|---|
| 梯度一直大 vs 一直小 | 差不多一样大：Adagrad 把差距抹平了（效果 1） |
| 梯度频繁 vs 稀少 | 稀少的那个，一旦拿到梯度，步子更大（效果 2） |
| 过去梯度大、现在变小 | 现在的步子很小：被过去压住了（它的问题） |

[RMSProp](#info:rmsprop_notes) 解决了这个问题：不再累加全部历史，只对最近一段时间求平均。
`,
    },
  },

  rmsprop_notes: {
    en: {
      title: "RMSProp",
      md: `
**RMSProp** keeps Adagrad's per-parameter learning rate, but averages only the **recent** squared gradients instead of summing all of them.

It fixes [Adagrad](#info:adagrad_notes)'s problem: Adagrad's sum only grows, so its steps only shrink, and large gradients from the past keep holding the steps down. (Proposed by Geoffrey Hinton in a 2012 lecture, never formally published.)

## Update rule

Only the accumulator changes. For each parameter $i$:

$$
\\begin{aligned}
\\text{Adagrad:}\\quad & s_i \\leftarrow s_i + g_i^2 \\\\
\\text{RMSProp:}\\quad & s_i \\leftarrow \\beta\\, s_i + (1 - \\beta)\\, g_i^2
\\end{aligned}
$$

The update is the same as Adagrad's:

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

- $g_i$ is the current gradient of parameter $i$; $s_i$ starts at 0.
- $\\beta$ is the decay rate, usually 0.9.
- $\\eta$ is the learning rate; $\\epsilon$ is a tiny constant that prevents division by zero.
- In vector form the accumulator is $s \\leftarrow \\beta s + (1 - \\beta)(g \\odot g)$, where $\\odot$ is element-wise multiplication (explained in the [Adagrad](#info:adagrad_notes) note).

### What $s$ means now

The new accumulator is an **exponential moving average** (EMA), the same kind of average [momentum](#info:sgd_momentum_notes) uses: each step keeps 90% of the old $s$ and adds 10% of the new $g^2$, so older gradients fade by a factor 0.9 per step. With $\\beta = 0.9$, $s$ is roughly the average of the last $\\frac{1}{1 - \\beta} = 10$ squared gradients.

So $s$ changes meaning:

- **Adagrad:** $s$ is "how much in total so far", so it keeps growing.
- **RMSProp:** $s$ is "how large the squared gradients have been recently". If the gradient size stays the same, $s$ stays the same.

$\\sqrt{s}$ is then the **root mean square** (RMS) of the recent gradients; dividing the gradient by it gives the method its name.

## What it fixes

### Effect 1 is kept; the shrinking is gone

With a constant gradient $g$, $s$ settles at $g^2$, so each step moves $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$:

- As in Adagrad, the step size does not depend on how large the gradient is (Adagrad's Effect 1).
- Unlike Adagrad, the step **does not shrink over time**: step 10,000 moves as far as step 100.

### It follows changes in gradient size

Adagrad's "held down by the past" case: the gradient is 10 for the first 100 steps, then 1 from step 101 on. How far does each step move afterwards (in units of $\\eta$)?

\`\`\`text
steps after the switch    Adagrad    RMSProp
         1                 0.010      0.105
        10                 0.010      0.168
        20                 0.010      0.277
        30                 0.010      0.439
        40                 0.010      0.637
        60                 0.010      0.921   → approaches 1
\`\`\`

- **Adagrad:** $s$ still holds the 100 large gradients and never lets them go, so the step stays stuck at $0.01\\eta$.
- **RMSProp:** the old large gradients fade by a factor 0.9 per step. After a few dozen steps $s$ reflects "the gradient is now 1", and the step recovers to nearly $\\eta$.

In LLM training the size of the gradients does change between phases of training, so being able to follow it matters.

## An analogy with momentum

RMSProp relates to Adagrad the way SGD with momentum relates to SGD: both add an EMA, a short memory of recent steps.

- **Adagrad → RMSProp:** replace the forever-growing sum of squared gradients with an EMA of squared gradients.
- **SGD → SGD with momentum:** replace the current gradient with an EMA of gradients.

The differences:

- **What gets averaged:** RMSProp averages **squared** gradients (their size); momentum averages the gradients themselves (their direction).
- **Per-parameter learning rate:** RMSProp, like Adagrad, gives each parameter its own; momentum, like SGD, uses one learning rate for all.
- **Problem fixed:** RMSProp fixes Adagrad's steps shrinking toward zero; momentum fixes SGD's noisy, zigzag path.

[Adam](#info:adam_notes) combines the two: an EMA of the gradients (momentum) divided by the root of an EMA of the squared gradients (RMSProp).

## Worked example

The same problem as in the [Adagrad](#info:adagrad_notes) note, so the two can be compared.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Accumulator:** $s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g)$, starting from $s_0 = (0, 0)$
- **Update:** $\\theta \\leftarrow \\theta - \\Delta$, with $\\Delta = \\eta_{\\text{eff}} \\cdot g$ and $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta = 0.9$, $\\epsilon = 10^{-8}$ (negligible here)
- **Init:** $\\theta_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.9, 0.4)$
- $\\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)$

Both coordinates moved $0.3162$, the same distance even though the gradients were $-3$ and $2$ (Effect 1). But $0.3162$ is $3.16\\eta$, more than three times the "normal" step $\\eta = 0.1$. This is the start-up problem described below.

### Step 2

- $\\theta_1 = (0.3162, -0.3162)$
- $g_2 = (0.3162 - 3,\\; -0.3162 + 2) = (-2.6838, 1.6838)$
- $s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot \\big((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)\\big) = (1.5303, 0.6435)$
- $\\sqrt{s_2} = (1.2370, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1247)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1247) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)$
- $\\theta_2 = \\theta_1 - \\Delta = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)$

### Step 3

- $\\theta_2 = (0.5332, -0.5261)$
- $g_3 = (0.5332 - 3,\\; -0.5261 + 2) = (-2.4668, 1.4739)$
- $s_3 = 0.9 \\cdot (1.5303, 0.6435) + (1 - 0.9) \\cdot \\big((-2.4668, 1.4739) \\odot (-2.4668, 1.4739)\\big) = (1.9858, 0.7964)$
- $\\sqrt{s_3} = (1.4092, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4739) = (-0.1751, 0.1652)$
- $\\theta_3 = \\theta_2 - \\Delta = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)$

### What the example shows

- The steps shrink over steps 1–3 ($0.316 \\to 0.217 \\to 0.175$ for $\\theta_1$), but not because $s$ only grows, as in Adagrad. $s$ is still warming up from 0 (see below), and the gradients themselves get smaller as $\\theta$ approaches the minimum.
- As in Adagrad, $\\theta_1$ (larger gradients) gets the smaller effective learning rate, but its actual steps are not smaller.

## The new problem: steps too large at the start

$s$ starts at 0, and the first step adds only $(1 - \\beta) = 10\\%$ of $g^2$:

\`\`\`text
s = 0.1 × g²,   √s = 0.316 × |g|
step 1 moves:  η × g / (0.316 |g|) = ±3.16 η      ← more than 3× the normal step
\`\`\`

This is the same cause as momentum's slow start: an EMA that starts at 0 is too small for the first several steps. In momentum the too-small average is the step itself, so the steps start **too small**; here it sits in the **denominator**, so the steps start **too large**. It takes roughly 10+ steps for $s$ to warm up. [Adam](#info:adam_notes) fixes both with its "bias correction".

## In PyTorch

\`torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8)\`: the decay rate $\\beta$ is called \`alpha\`, and its default is **0.99** (an average over roughly the last 100 steps), not the 0.9 used in this note and in Hinton's lecture.
`,
    },
    zh: {
      title: "RMSProp",
      md: `
**RMSProp** 保留了 Adagrad"每个参数自己的学习率"，但只对**最近**的梯度平方求平均，而不是把所有历史都加起来。

它解决的是 [Adagrad](#info:adagrad_notes) 的问题：Adagrad 的累加和只增不减，所以步子只会变小，过去的大梯度还会一直压着步子。（这个方法由 Geoffrey Hinton 在 2012 年的一次课上提出，没有正式发表过论文。）

## 更新规则

只改了累加器这一处。对每个参数 $i$：

$$
\\begin{aligned}
\\text{Adagrad:}\\quad & s_i \\leftarrow s_i + g_i^2 \\\\
\\text{RMSProp:}\\quad & s_i \\leftarrow \\beta\\, s_i + (1 - \\beta)\\, g_i^2
\\end{aligned}
$$

更新式和 Adagrad 一样：

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

- $g_i$ 是参数 $i$ 当前的梯度；$s_i$ 从 0 开始。
- $\\beta$ 是衰减率，通常取 0.9。
- $\\eta$ 是学习率；$\\epsilon$ 是一个很小的常数，防止除以 0。
- 写成向量形式，累加器就是 $s \\leftarrow \\beta s + (1 - \\beta)(g \\odot g)$，其中 $\\odot$ 是逐元素相乘（见 [Adagrad](#info:adagrad_notes) note 里的解释）。

### 现在 $s$ 的含义

新的累加器是一个**指数滑动平均**（EMA），和 [momentum](#info:sgd_momentum_notes) 用的是同一种平均：每一步保留 90% 的旧 $s$，再加上 10% 的新 $g^2$，所以越早的梯度每过一步就再乘一次 0.9，逐渐被遗忘。$\\beta = 0.9$ 时，$s$ 大约是最近 $\\frac{1}{1 - \\beta} = 10$ 个梯度平方的平均。

所以 $s$ 的含义变了：

- **Adagrad：** $s$ 是"到目前为止一共累计了多少"，所以会一直变大。
- **RMSProp：** $s$ 是"最近梯度的平方一般有多大"。梯度大小不变时，$s$ 也不变。

$\\sqrt{s}$ 就是最近梯度的**均方根**（root mean square，RMS）；用它去除梯度，这就是这个方法名字的由来。

## 它修好了什么

### 效果 1 还在，"只会变小"没了

梯度一直是 $g$ 时，$s$ 会稳定在 $g^2$，所以每一步走 $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$：

- 和 Adagrad 一样，步子和梯度的大小无关（Adagrad 的效果 1）。
- 和 Adagrad 不同，步子**不会随时间缩小**：第 10000 步和第 100 步走得一样远。

### 能跟上梯度大小的变化

用 Adagrad 那个"被过去压住"的例子：前 100 步梯度是 10，从第 101 步起梯度变成 1。之后每一步走多远（单位是 $\\eta$）？

\`\`\`text
梯度变成 1 之后      Adagrad    RMSProp
第  1 步              0.010      0.105
第 10 步              0.010      0.168
第 20 步              0.010      0.277
第 30 步              0.010      0.439
第 40 步              0.010      0.637
第 60 步              0.010      0.921   → 接近 1
\`\`\`

- **Adagrad：** $s$ 里还存着前 100 步的大梯度，而且永远不会减少，所以步子一直卡在 $0.01\\eta$。
- **RMSProp：** 过去的大梯度每步乘 0.9 逐渐被遗忘。几十步之后，$s$ 就反映出"现在梯度是 1"，步子恢复到接近 $\\eta$。

训练 LLM 时，梯度的大小在训练的不同阶段确实会变，所以能跟上这种变化很重要。

## 和 momentum 的类比

RMSProp 和 Adagrad 的关系，就像 SGD with momentum 和 SGD 的关系：两者都是加了一个 EMA，也就是对最近几步的短期记忆。

- **Adagrad → RMSProp：** 把永远增长的梯度平方累加和，换成梯度平方的 EMA。
- **SGD → SGD with momentum：** 把当前梯度，换成梯度的 EMA。

区别在于：

- **对什么求平均：** RMSProp 平均的是梯度的**平方**（反映大小）；momentum 平均的是梯度本身（反映方向）。
- **每个参数自己的学习率：** RMSProp 和 Adagrad 一样，每个参数有自己的学习率；momentum 和 SGD 一样，所有参数共用一个学习率。
- **解决的问题：** RMSProp 解决 Adagrad 步子缩到接近 0 的问题；momentum 解决 SGD 路线嘈杂、走"之"字形的问题。

[Adam](#info:adam_notes) 把两者结合在一起：梯度的 EMA（momentum），除以梯度平方的 EMA 的平方根（RMSProp）。

## 数值例子

和 [Adagrad](#info:adagrad_notes) note 用同一个问题，方便对比。

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$，在 $\\theta = (3, -2)$ 处最小
- **梯度：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **累加器：** $s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g)$，从 $s_0 = (0, 0)$ 开始
- **更新：** $\\theta \\leftarrow \\theta - \\Delta$，其中 $\\Delta = \\eta_{\\text{eff}} \\cdot g$，$\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **超参数：** $\\eta = 0.1$，$\\beta = 0.9$，$\\epsilon = 10^{-8}$（这里可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.9, 0.4)$
- $\\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)$

两个坐标都走了 $0.3162$：梯度分别是 $-3$ 和 $2$，走的距离却一样（效果 1）。但 $0.3162$ 是 $3.16\\eta$，比"正常"的步长 $\\eta = 0.1$ 大了三倍多。这就是下面要讲的起步问题。

### 第 2 步

- $\\theta_1 = (0.3162, -0.3162)$
- $g_2 = (0.3162 - 3,\\; -0.3162 + 2) = (-2.6838, 1.6838)$
- $s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot \\big((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)\\big) = (1.5303, 0.6435)$
- $\\sqrt{s_2} = (1.2370, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1247)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1247) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)$
- $\\theta_2 = \\theta_1 - \\Delta = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)$

### 第 3 步

- $\\theta_2 = (0.5332, -0.5261)$
- $g_3 = (0.5332 - 3,\\; -0.5261 + 2) = (-2.4668, 1.4739)$
- $s_3 = 0.9 \\cdot (1.5303, 0.6435) + (1 - 0.9) \\cdot \\big((-2.4668, 1.4739) \\odot (-2.4668, 1.4739)\\big) = (1.9858, 0.7964)$
- $\\sqrt{s_3} = (1.4092, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4739) = (-0.1751, 0.1652)$
- $\\theta_3 = \\theta_2 - \\Delta = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)$

### 这个例子说明了什么

- 第 1–3 步的步子在变小（$\\theta_1$：$0.316 \\to 0.217 \\to 0.175$），但原因不是像 Adagrad 那样 $s$ 只增不减。一是 $s$ 还在从 0 开始"热身"（见下一节）；二是 $\\theta$ 越来越接近最低点，梯度本身也在变小。
- 和 Adagrad 一样，$\\theta_1$（梯度更大）的有效学习率更小，但它的实际步子并不更小。

## 新问题：刚开始时步子太大

$s$ 从 0 开始，第 1 步只加了 $(1 - \\beta) = 10\\%$ 的 $g^2$：

\`\`\`text
s = 0.1 × g²,   √s = 0.316 × |g|
第 1 步走:  η × g / (0.316 |g|) = ±3.16 η      ← 是正常步长的 3 倍多
\`\`\`

这和 momentum"起步慢"是同一个原因：从 0 开始的 EMA，在最初几步里都偏小。在 momentum 里，偏小的平均值就是步子本身，所以步子一开始**太小**；在这里，它在**分母**上，所以步子一开始**太大**。$s$ 大约要 10 步以上才"热身"完毕。[Adam](#info:adam_notes) 用 "bias correction" 同时修正了这两个问题。

## 在 PyTorch 里

\`torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8)\`：衰减率 $\\beta$ 在这里叫 \`alpha\`，默认值是 **0.99**（大约相当于最近 100 步的平均），而不是这篇 note 和 Hinton 课上用的 0.9。
`,
    },
  },

  adam_notes: {
    en: {
      title: "Adam",
      md: `
**Adam** combines the two ideas before it: [momentum](#info:sgd_momentum_notes) (an average of the gradients) and [RMSProp](#info:rmsprop_notes) (a per-parameter step size from an average of the squared gradients). It also fixes the start-up problem both of them have.

## Update rule

For each parameter, Adam keeps two EMAs (exponential moving averages), both starting at 0:

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2)\\, g_t^2 && \\text{(EMA of squared gradients: RMSProp)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_{t} &= \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(update)}
\\end{aligned}
$$

- $g_t$ is the gradient at step $t$; for vectors of parameters, $g_t^2$ means $g_t \\odot g_t$ (element-wise, see [Adagrad](#info:adagrad_notes)).
- $m_t$ tracks the **direction** of recent gradients; $v_t$ tracks their **size**.
- $\\beta_1$ (usually 0.9) averages $m$ over roughly the last $\\frac{1}{1 - \\beta_1} = 10$ steps; $\\beta_2$ (usually 0.999) averages $v$ over roughly the last 1000 steps.
- $\\hat{m}_t$ and $\\hat{v}_t$ are the bias-corrected versions, explained below.
- $\\eta$ is the learning rate; $\\epsilon$ is a tiny constant that prevents division by zero.

Leaving out the bias correction for a moment, the update is simply $\\eta \\cdot \\frac{m}{\\sqrt{v}}$: momentum's average gradient, divided by RMSProp's RMS of the gradient.

## What combining them adds: noisy gradients get smaller steps

Look at the ratio $\\frac{m}{\\sqrt{v}}$: the numerator is the **average** gradient, the denominator is the **root mean square** of the gradient.

**Consistent gradients**, e.g. always 1:

\`\`\`text
m ≈ 1,  √v ≈ 1   →  m/√v ≈ 1,   each step moves η
\`\`\`

**Noisy gradients**, e.g. 4, −2, 4, −2, … (the average is still 1, but with large noise):

\`\`\`text
m ≈ average            = 1
v ≈ average of squares = (16 + 4) / 2 = 10,   √v ≈ 3.16
m/√v ≈ 1 / 3.16 = 0.32,   each step moves only 0.32 η
\`\`\`

When the signs flip back and forth, they cancel in $m$ (the momentum part), but the squares in $v$ do not cancel. So the less consistent the gradients, the smaller $\\frac{m}{\\sqrt{v}}$, and the more cautiously Adam steps; the more consistent, the closer it gets to the full step $\\eta$.

Think of $\\frac{m}{\\sqrt{v}}$ as a signal-to-noise ratio: how trustworthy the direction is. Its size is usually at most about 1, so each Adam step is **at most about $\\eta$**, however large the gradient itself is. This keeps the scale independence from Adagrad and RMSProp.

## Bias correction

### The problem: both averages start at 0

$m$ and $v$ start at 0, so for the first several steps both are too small. This is the same start-up problem as before:

- In [momentum](#info:sgd_momentum_notes), the too-small average is the step itself, so the steps start **too small**.
- In [RMSProp](#info:rmsprop_notes), the too-small average is in the denominator, so the steps start **too large**.

### How large the bias is

Suppose the gradient is always $g$. Unroll $m$:

\`\`\`text
m1 = 0.1 g
m2 = 0.9 × 0.1 g + 0.1 g = 0.19 g
m3 = 0.9 × 0.19 g + 0.1 g = 0.271 g
...
m_t = (1 − 0.9^t) × g
\`\`\`

In general $m_t = (1 - \\beta_1^t)\\, g$, and in the same way $v_t = (1 - \\beta_2^t)\\, g^2$. The average is too small by exactly the factor $(1 - \\beta^t)$.

### The fix: divide it back out

$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
$$

\`\`\`text
step 1:  m̂ = 0.1 g / 0.1       = g      ✓
         v̂ = 0.001 g² / 0.001  = g²     ✓
\`\`\`

As $t$ grows, $\\beta^t$ goes to 0 and the divisor goes to 1, so the correction fades away by itself. With the correction, the very first step is exactly $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$.

> The paper's Section 3 derives this for gradients that change over time: [Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980).

### Without the correction

With a constant gradient, the uncorrected step $\\eta \\cdot \\frac{m_t}{\\sqrt{v_t}}$ is too large by the factor

$$
F_t = \\frac{1 - \\beta_1^t}{\\sqrt{1 - \\beta_2^t}}
$$

For $\\beta_1 = 0.9$, $\\beta_2 = 0.999$:

\`\`\`text
step     1:  F = 3.16       (0.1 / √0.001)
step     2:  F = 4.25
step     3:  F = 4.95
step     4:  F = 5.44
step    12:  F = 6.57       ← the peak
step   100:  F = 3.24
step  1000:  F = 1.26
step  3000:  F = 1.03
\`\`\`

$m$ (10-step average) warms up quickly, but $v$ (1000-step average) takes thousands of steps, and until then the steps stay too large, by up to 6.6×. At the start of training the parameters are random and the gradients are large; steps several times too large then can easily make training diverge.

## Worked example

The same problem as in the [Adagrad](#info:adagrad_notes) and [RMSProp](#info:rmsprop_notes) notes.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (negligible here)
- **Init:** $\\theta_0 = (0, 0)$, $m_0 = (0, 0)$, $v_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- **Bias correction:**

  $$
  \\begin{aligned}
  \\hat{m}_1 &= \\frac{m_1}{1 - \\beta_1^1} = \\frac{(-0.3, 0.2)}{1 - 0.9^1} = (-3, 2) \\\\
  \\hat{v}_1 &= \\frac{v_1}{1 - \\beta_2^1} = \\frac{(0.009, 0.004)}{1 - 0.999^1} = (9, 4)
  \\end{aligned}
  $$

- **Update:**

  $$
  \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon}\\, \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9, 4)}} \\cdot (-3, 2) = (0.1, -0.1)
  $$

After correction, $\\hat{m}_1$ and $\\hat{v}_1$ are exactly $g_1$ and $g_1^2$, so both coordinates move exactly $\\eta = 0.1$. Without the correction they would move $3.16 \\times 0.1 = 0.316$, the same too-large first step as in the RMSProp example.

### Step 2

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- **Bias correction:**

  $$
  \\begin{aligned}
  \\hat{m}_2 &= \\frac{m_2}{1 - \\beta_1^2} = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.94737, 1.94737) \\\\
  \\hat{v}_2 &= \\frac{v_2}{1 - \\beta_2^2} = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.70485, 3.80490)
  \\end{aligned}
  $$

- **Update:**

  $$
  \\begin{aligned}
  \\theta_2 &= \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon}\\, \\hat{m}_2 = (0.1, -0.1) - \\frac{0.1}{\\sqrt{(8.70485, 3.80490)}} \\cdot (-2.94737, 1.94737) \\\\
  &= (0.1998973, -0.1998335)
  \\end{aligned}
  $$

### Step 3

- $\\theta_2 = (0.1998973, -0.1998335)$
- $g_3 = (0.1998973 - 3,\\; -0.1998335 + 2) = (-2.80010, 1.80017)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80010, 1.80017) = (-0.78401, 0.513017)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot \\big((-2.80010, 1.80017) \\odot (-2.80010, 1.80017)\\big) = (0.0252242, 0.010839)$
- **Bias correction:**

  $$
  \\begin{aligned}
  \\hat{m}_3 &= \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78401, 0.513017)}{1 - 0.9^3} = (-2.89303, 1.89305) \\\\
  \\hat{v}_3 &= \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252242, 0.010839)}{1 - 0.999^3} = (8.41647, 3.61661)
  \\end{aligned}
  $$

- **Update:**

  $$
  \\begin{aligned}
  \\theta_3 &= \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon}\\, \\hat{m}_3 = (0.1998973, -0.1998335) - \\frac{0.1}{\\sqrt{(8.41647, 3.61661)}} \\cdot (-2.89303, 1.89305) \\\\
  &= (0.2996185, -0.2993766)
  \\end{aligned}
  $$

### What the example shows

- **Every step moves about $\\eta = 0.1$** in both coordinates: $\\theta_1$ goes $0 \\to 0.1 \\to 0.2 \\to 0.3$, and $\\theta_2$ goes $0 \\to -0.1 \\to -0.2 \\to -0.3$. The gradients are $-3$ and $2$, yet the steps are the same: the step size does not depend on how large the gradient is.
- **The gradients here are perfectly consistent** (no noise), so $\\frac{\\hat{m}}{\\sqrt{\\hat{v}}} \\approx \\pm 1$ and Adam takes full steps. With noisy gradients the steps would be smaller.
- **Bias correction** makes step 1 exactly $\\eta$ instead of $3.16\\eta$.

## In PyTorch

\`torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)\`: \`betas\` is $(\\beta_1, \\beta_2)$. Bias correction is always applied.

This site's GPT-2 code uses [AdamW](#info:adamw_notes) (Adam with decoupled weight decay) and sets $\\beta_2 = 0.95$ instead of 0.999, following GPT-3: a shorter average for $v$ reacts faster when gradient sizes change. See the [optimizer overview](#info:optim_adamw_notes).

## References

1. **Adam: A Method for Stochastic Optimization**

   Kingma & Ba, 2014. The original paper.

   [arXiv:1412.6980 ↗](https://arxiv.org/pdf/1412.6980)

2. **StackExchange discussion**

   Why bias correction matters in Adam.

   [StackExchange Discussion ↗](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
`,
    },
    zh: {
      title: "Adam",
      md: `
**Adam** 把前面两个想法结合在一起：[momentum](#info:sgd_momentum_notes)（梯度的平均）和 [RMSProp](#info:rmsprop_notes)（用梯度平方的平均给每个参数定步长）。它还修正了这两者共有的"起步问题"。

## 更新规则

对每个参数，Adam 维护两个 EMA（指数滑动平均），都从 0 开始：

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2)\\, g_t^2 && \\text{(EMA of squared gradients: RMSProp)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_{t} &= \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(update)}
\\end{aligned}
$$

- $g_t$ 是第 $t$ 步的梯度；参数是向量时，$g_t^2$ 指 $g_t \\odot g_t$（逐元素相乘，见 [Adagrad](#info:adagrad_notes)）。
- $m_t$ 记录最近梯度的**方向**；$v_t$ 记录最近梯度的**大小**。
- $\\beta_1$（通常 0.9）让 $m$ 大约覆盖最近 $\\frac{1}{1 - \\beta_1} = 10$ 步；$\\beta_2$（通常 0.999）让 $v$ 大约覆盖最近 1000 步。
- $\\hat{m}_t$ 和 $\\hat{v}_t$ 是做了 bias correction 之后的版本，下面会解释。
- $\\eta$ 是学习率；$\\epsilon$ 是一个很小的常数，防止除以 0。

先不管 bias correction，更新量就是 $\\eta \\cdot \\frac{m}{\\sqrt{v}}$：momentum 的平均梯度，除以 RMSProp 的梯度均方根。

## 合在一起多了什么：梯度越乱，步子越小

看比值 $\\frac{m}{\\sqrt{v}}$：分子是梯度的**平均值**，分母是梯度的**均方根**。

**梯度方向一致**，比如一直是 1：

\`\`\`text
m ≈ 1,  √v ≈ 1   →  m/√v ≈ 1,   每步走 η
\`\`\`

**梯度方向乱跳**，比如 4、−2、4、−2……（平均值还是 1，但噪声很大）：

\`\`\`text
m ≈ 平均值       = 1
v ≈ 平方的平均值 = (16 + 4) / 2 = 10,   √v ≈ 3.16
m/√v ≈ 1 / 3.16 = 0.32,   每步只走 0.32 η
\`\`\`

正负来回跳时，它们在 $m$ 里互相抵消（这是 momentum 的作用），但 $v$ 里的平方不会抵消。所以梯度越不一致，$\\frac{m}{\\sqrt{v}}$ 越小，Adam 走得越谨慎；梯度越一致，就越接近全速 $\\eta$。

可以把 $\\frac{m}{\\sqrt{v}}$ 理解成一个"信噪比"：这个方向有多可信。它的绝对值通常最多约为 1，所以 Adam 每一步的步长**最多大约是 $\\eta$**，不管梯度本身有多大。这也保留了 Adagrad 和 RMSProp"步长和梯度大小无关"的性质。

## Bias correction

### 问题：两个平均值都从 0 开始

$m$ 和 $v$ 都从 0 开始，所以最初几步两者都偏小。这就是前面遇到过的起步问题：

- 在 [momentum](#info:sgd_momentum_notes) 里，偏小的平均值就是步子本身，所以步子一开始**太小**。
- 在 [RMSProp](#info:rmsprop_notes) 里，偏小的平均值在分母上，所以步子一开始**太大**。

### 偏小多少

假设梯度一直是 $g$，把 $m$ 展开：

\`\`\`text
m1 = 0.1 g
m2 = 0.9 × 0.1 g + 0.1 g = 0.19 g
m3 = 0.9 × 0.19 g + 0.1 g = 0.271 g
...
m_t = (1 − 0.9^t) × g
\`\`\`

一般地，$m_t = (1 - \\beta_1^t)\\, g$；同理 $v_t = (1 - \\beta_2^t)\\, g^2$。平均值正好偏小了 $(1 - \\beta^t)$ 这个比例。

### 修正：除回去

$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
$$

\`\`\`text
第 1 步:  m̂ = 0.1 g / 0.1       = g      ✓
          v̂ = 0.001 g² / 0.001  = g²     ✓
\`\`\`

随着 $t$ 变大，$\\beta^t$ 趋近 0，除数趋近 1，修正就自动消失了。有了修正，第 1 步正好走 $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$。

> 论文第 3 节对随时间变化的梯度做了推导：[Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980)。

### 如果不修正

梯度不变时，不修正的步长 $\\eta \\cdot \\frac{m_t}{\\sqrt{v_t}}$ 会被放大

$$
F_t = \\frac{1 - \\beta_1^t}{\\sqrt{1 - \\beta_2^t}}
$$

倍。$\\beta_1 = 0.9$、$\\beta_2 = 0.999$ 时：

\`\`\`text
step     1:  F = 3.16       (0.1 / √0.001)
step     2:  F = 4.25
step     3:  F = 4.95
step     4:  F = 5.44
step    12:  F = 6.57       ← 最大
step   100:  F = 3.24
step  1000:  F = 1.26
step  3000:  F = 1.03
\`\`\`

$m$（10 步平均）热身很快，但 $v$（1000 步平均）要几千步才热身完毕，在此之前步子一直偏大，最多大到 6.6 倍。训练刚开始时参数是随机的，梯度本来就大，这时步子再放大好几倍，很容易让训练发散。

## 数值例子

和 [Adagrad](#info:adagrad_notes)、[RMSProp](#info:rmsprop_notes) note 用同一个问题。

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$，在 $\\theta = (3, -2)$ 处最小
- **梯度：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **更新：** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **超参数：** $\\eta = 0.1$，$\\beta_1 = 0.9$，$\\beta_2 = 0.999$，$\\epsilon = 10^{-8}$（这里可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$，$m_0 = (0, 0)$，$v_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- **Bias correction：**

  $$
  \\begin{aligned}
  \\hat{m}_1 &= \\frac{m_1}{1 - \\beta_1^1} = \\frac{(-0.3, 0.2)}{1 - 0.9^1} = (-3, 2) \\\\
  \\hat{v}_1 &= \\frac{v_1}{1 - \\beta_2^1} = \\frac{(0.009, 0.004)}{1 - 0.999^1} = (9, 4)
  \\end{aligned}
  $$

- **更新：**

  $$
  \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon}\\, \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9, 4)}} \\cdot (-3, 2) = (0.1, -0.1)
  $$

修正之后，$\\hat{m}_1$ 和 $\\hat{v}_1$ 正好是 $g_1$ 和 $g_1^2$，所以两个坐标都正好走了 $\\eta = 0.1$。如果不修正，它们会走 $3.16 \\times 0.1 = 0.316$，和 RMSProp 例子里第 1 步一样偏大。

### 第 2 步

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- **Bias correction：**

  $$
  \\begin{aligned}
  \\hat{m}_2 &= \\frac{m_2}{1 - \\beta_1^2} = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.94737, 1.94737) \\\\
  \\hat{v}_2 &= \\frac{v_2}{1 - \\beta_2^2} = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.70485, 3.80490)
  \\end{aligned}
  $$

- **更新：**

  $$
  \\begin{aligned}
  \\theta_2 &= \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon}\\, \\hat{m}_2 = (0.1, -0.1) - \\frac{0.1}{\\sqrt{(8.70485, 3.80490)}} \\cdot (-2.94737, 1.94737) \\\\
  &= (0.1998973, -0.1998335)
  \\end{aligned}
  $$

### 第 3 步

- $\\theta_2 = (0.1998973, -0.1998335)$
- $g_3 = (0.1998973 - 3,\\; -0.1998335 + 2) = (-2.80010, 1.80017)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80010, 1.80017) = (-0.78401, 0.513017)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot \\big((-2.80010, 1.80017) \\odot (-2.80010, 1.80017)\\big) = (0.0252242, 0.010839)$
- **Bias correction：**

  $$
  \\begin{aligned}
  \\hat{m}_3 &= \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78401, 0.513017)}{1 - 0.9^3} = (-2.89303, 1.89305) \\\\
  \\hat{v}_3 &= \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252242, 0.010839)}{1 - 0.999^3} = (8.41647, 3.61661)
  \\end{aligned}
  $$

- **更新：**

  $$
  \\begin{aligned}
  \\theta_3 &= \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon}\\, \\hat{m}_3 = (0.1998973, -0.1998335) - \\frac{0.1}{\\sqrt{(8.41647, 3.61661)}} \\cdot (-2.89303, 1.89305) \\\\
  &= (0.2996185, -0.2993766)
  \\end{aligned}
  $$

### 这个例子说明了什么

- **每一步两个坐标都走了约 $\\eta = 0.1$**：$\\theta_1$ 是 $0 \\to 0.1 \\to 0.2 \\to 0.3$，$\\theta_2$ 是 $0 \\to -0.1 \\to -0.2 \\to -0.3$。梯度分别是 $-3$ 和 $2$，步子却一样：步长和梯度大小无关。
- **这里的梯度完全一致**（没有噪声），所以 $\\frac{\\hat{m}}{\\sqrt{\\hat{v}}} \\approx \\pm 1$，Adam 每步都走全速。如果梯度有噪声，步子会更小。
- **bias correction** 让第 1 步正好是 $\\eta$，而不是 $3.16\\eta$。

## 在 PyTorch 里

\`torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)\`：\`betas\` 就是 $(\\beta_1, \\beta_2)$。bias correction 总是会做。

这个网站的 GPT-2 代码用的是 [AdamW](#info:adamw_notes)（带解耦 weight decay 的 Adam），并且按照 GPT-3 的做法把 $\\beta_2$ 设成 0.95 而不是 0.999：$v$ 的平均窗口短一些，梯度大小变化时反应更快。见 [optimizer 总览](#info:optim_adamw_notes)。

## 参考资料

1. **Adam: A Method for Stochastic Optimization**

   Kingma & Ba，2014。原始论文。

   [arXiv:1412.6980 ↗](https://arxiv.org/pdf/1412.6980)

2. **StackExchange 讨论**

   为什么 Adam 需要 bias correction。

   [StackExchange Discussion ↗](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
`,
    },
  },

  adamw_notes: {
    en: {
      title: "AdamW",
      md: `
**AdamW** is [Adam](#info:adam_notes) plus **weight decay**, applied directly to the weights instead of being added to the gradient. The "W" stands for this decoupled **W**eight decay. It is the optimizer this site's GPT-2 code uses.

## What weight decay is

Before each update, shrink every weight a little:

$$
\\theta \\leftarrow (1 - \\eta \\lambda)\\, \\theta
$$

- $\\lambda$ is the weight decay strength (0.1 in this site's code); $\\eta$ is the learning rate.
- With $\\eta = 0.1$ and $\\lambda = 0.1$, $1 - \\eta\\lambda = 0.99$: every step first multiplies each weight by 0.99.

**Why do this?** A weight that the loss does not need gets no consistent push from the gradient, so the decay slowly pulls it toward 0. A weight that does help lower the loss keeps getting pushed back by its gradient and stays large. The model is therefore discouraged from relying on a few very large weights to memorize the training data. This is a form of **regularization**.

## In SGD, L2 regularization is the same thing

Another common way to keep weights small is **L2 regularization**: add a penalty to the loss,

$$
L_{\\text{total}} = L + \\frac{\\lambda}{2}\\, \\theta^2
$$

The penalty's gradient is $\\lambda \\theta$, so the SGD update becomes

$$
\\theta \\leftarrow \\theta - \\eta\\, (g + \\lambda \\theta) = (1 - \\eta \\lambda)\\, \\theta - \\eta\\, g
$$

That is "shrink by $(1 - \\eta\\lambda)$, then take the normal step": exactly weight decay. In SGD the two are the same, which is why the names were long used interchangeably.

## In Adam, L2 goes wrong

With Adam the two are **no longer the same**. The L2 way (what \`torch.optim.Adam(weight_decay=...)\` does) adds $\\lambda\\theta$ to the gradient and hands the result to Adam:

\`\`\`text
g' = g + λθ
m and v are computed from g', and the step is η · m̂ / √v̂
\`\`\`

Adam cannot tell which part of $g'$ comes from the loss and which part is decay. It divides **all** of it by $\\sqrt{\\hat{v}}$, the root mean square of the past values of $g'$ (over roughly the last $\\frac{1}{1 - \\beta_2} = 1000$ steps, or all steps so far if there have been fewer; see [Adam](#info:adam_notes)). So the decay part is scaled too:

- **Parameters with large gradients:** $\\sqrt{\\hat{v}}$ is large, the decay is divided down to almost nothing, and the weight barely decays.
- **Parameters with small gradients:** $\\sqrt{\\hat{v}}$ is small, the decay is magnified, and the weight decays very fast.

### Example: one $\\lambda$, a 100× difference

Two parameters A and B, both at $\\theta = 1$, with $\\eta = 0.01$ and $\\lambda = 0.1$, so the decay adds $\\lambda\\theta = 0.1$ to each gradient. A's own gradient alternates between $+10$ and $-10$; B's alternates between $+0.01$ and $-0.01$. Both average to 0: the loss does not need either weight, which is exactly when decay should pull them toward 0.

**Step 1: what is $\\sqrt{\\hat{v}}$?** It is the root mean square of the past $g' = g + 0.1$:

\`\`\`text
A:  g' = 10.1, −9.9, 10.1, …    squares 102.01, 98.01   →  mean 100.01,  √v̂ ≈ 10
B:  g' = 0.11, 0.09, 0.11, …    squares 0.0121, 0.0081  →  mean 0.0101,  √v̂ ≈ 0.1
\`\`\`

A's $g'$ is dominated by its own gradient, so the extra 0.1 hardly shows. B's $g'$ is almost entirely the decay term, with its own gradient as a small wobble.

**Step 2: how much does the decay part move $\\theta$?** The step is $\\eta \\cdot \\hat{m} / \\sqrt{\\hat{v}}$, and $\\hat{m}$ averages $g'$, so the decay term contributes $\\eta \\lambda \\theta / \\sqrt{\\hat{v}}$:

\`\`\`text
A:  0.01 × 0.1 / 10   = 0.0001   →  shrinks 0.01% per step
B:  0.01 × 0.1 / 0.1  = 0.01     →  shrinks 1% per step
\`\`\`

**Step 3: compare with what was intended.** $\\lambda = 0.1$ was meant to shrink each weight by $\\eta\\lambda = 0.001$, i.e. 0.1% per step:

| | Intended (AdamW) | Adam + L2 |
|---|---|---|
| A (large gradients) | 0.1% | 0.01%, 10× too weak |
| B (small gradients) | 0.1% | 1%, 10× too strong |

Same $\\lambda$, but a **100×** difference in actual decay, caused only by dividing the decay by $\\sqrt{\\hat{v}}$.

**Over 200 steps** (run in PyTorch):

\`\`\`text
                 A (gradient ±10)   B (gradient ±0.01)
no decay              0.982              0.982
Adam + L2             0.963              0.018     ← A barely decays, B is nearly 0
AdamW                 0.804              0.804     ← both decay the same
\`\`\`

- **A:** 0.0001 less per step, about 0.02 over 200 steps: 0.982 → 0.963.
- **B:** about 0.01 less per step, so it reaches nearly 0 within about 100 steps. (As $\\theta$ shrinks, the decay term and $\\sqrt{\\hat{v}}$ shrink with it, so it does not go exactly to 0.)
- **AdamW:** each step multiplies by $0.999$, and $0.999^{200} \\approx 0.819$, so $0.982 \\times 0.819 \\approx 0.804$ for both.

## Update rule

AdamW computes $m_t$, $\\hat{m}_t$, $v_t$, $\\hat{v}_t$ exactly as [Adam](#info:adam_notes) does, from the plain gradient $g_t$ (no $\\lambda\\theta$ added). Only the last line changes:

$$
\\begin{aligned}
\\text{Adam:}\\quad & \\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t \\\\
\\text{AdamW:}\\quad & \\theta_t = (1 - \\eta \\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t
\\end{aligned}
$$

The decay $(1 - \\eta\\lambda)$ multiplies the weight directly and never passes through $\\sqrt{\\hat{v}}$, so every parameter shrinks by the same fraction, whatever its gradients look like.

## Worked example

The same problem as in the [Adam](#info:adam_notes) note, so the two can be compared.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update:** $\\theta_t = (1 - \\eta\\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (negligible here), $\\lambda = 0.1$, so the decay factor is $1 - \\eta\\lambda = 0.99$
- **Init:** $\\theta_0 = (0, 0)$, $m_0 = (0, 0)$, $v_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + 0.1 \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + 0.001 \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- $\\hat{m}_1 = \\frac{(-0.3, 0.2)}{1 - 0.9} = (-3, 2)$, $\\quad \\hat{v}_1 = \\frac{(0.009, 0.004)}{1 - 0.999} = (9, 4)$, $\\quad \\sqrt{\\hat{v}_1} = (3, 2)$
- **Decay:** $0.99 \\cdot (0, 0) = (0, 0)$
- **Adam step:** $0.1 \\cdot \\frac{(-3, 2)}{(3, 2)} = (-0.1, 0.1)$
- $\\theta_1 = (0, 0) - (-0.1, 0.1) = (0.1, -0.1)$

The decay does nothing yet: $0.99 \\times 0 = 0$. Step 1 is identical to Adam's.

### Step 2

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + 0.1 \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + 0.001 \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- $\\hat{m}_2 = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.9474, 1.9474)$, $\\quad \\hat{v}_2 = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.7049, 3.8049)$, $\\quad \\sqrt{\\hat{v}_2} = (2.9504, 1.9506)$
- **Decay:** $0.99 \\cdot (0.1, -0.1) = (0.099, -0.099)$
- **Adam step:** $0.1 \\cdot \\frac{(-2.9474, 1.9474)}{(2.9504, 1.9506)} = (-0.0999, 0.0998)$
- $\\theta_2 = (0.099, -0.099) - (-0.0999, 0.0998) = (0.1989, -0.1988)$

### Step 3

- $\\theta_2 = (0.1989, -0.1988)$
- $g_3 = (0.1989 - 3,\\; -0.1988 + 2) = (-2.8011, 1.8012)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + 0.1 \\cdot (-2.8011, 1.8012) = (-0.7841, 0.5131)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + 0.001 \\cdot \\big((-2.8011, 1.8012) \\odot (-2.8011, 1.8012)\\big) = (0.025230, 0.010843)$
- $\\hat{m}_3 = \\frac{(-0.7841, 0.5131)}{1 - 0.9^3} = (-2.8934, 1.8934)$, $\\quad \\hat{v}_3 = \\frac{(0.025230, 0.010843)}{1 - 0.999^3} = (8.4183, 3.6178)$, $\\quad \\sqrt{\\hat{v}_3} = (2.9014, 1.9021)$
- **Decay:** $0.99 \\cdot (0.1989, -0.1988) = (0.1969, -0.1968)$
- **Adam step:** $0.1 \\cdot \\frac{(-2.8934, 1.8934)}{(2.9014, 1.9021)} = (-0.0997, 0.0995)$
- $\\theta_3 = (0.1969, -0.1968) - (-0.0997, 0.0995) = (0.2966, -0.2964)$

### What the example shows

Compare $\\theta_1$ with plain Adam on the same problem:

\`\`\`text
          Adam      AdamW
step 1    0.1000    0.1000
step 2    0.1999    0.1989    ← 0.001 smaller
step 3    0.2996    0.2966    ← 0.003 smaller
\`\`\`

- The difference is the decay: each step removes 1% of the current weight ($0.99 \\times 0.1 = 0.099$, i.e. 0.001 at step 2), and the effect adds up over steps.
- The Adam step itself stays about $\\eta = 0.1$ per step, just as in Adam, because the gradients that feed $m$ and $v$ do not include the decay.
- Here decay pulls $\\theta$ toward 0, away from the minimum at $(3, -2)$. In this toy problem every weight is needed, so decay only gets in the way; its benefit appears in a real model, where many weights are not needed.

## In PyTorch and in this site's GPT-2 code

\`torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)\`. Each step first multiplies every weight by $1 - \\eta\\lambda$ (\`1 - lr * weight_decay\`), then takes the Adam step.

### How strong the decay actually is

Note the $\\eta$ in $\\eta\\lambda$: \`weight_decay=0.1\` does **not** mean "shrink 10% per step". The code uses \`lr=3e-4\` and \`weight_decay=0.1\`:

\`\`\`text
η·λ = 0.0003 × 0.1 = 0.00003
each step:  θ ← 0.99997 · θ          ← shrinks only 0.003% per step
\`\`\`

To see what decay can do on its own, take a weight with **no gradient at all**: then $m = 0$, the Adam step is 0, and only the decay acts. Starting from 1, after $n$ steps it is $0.99997^n$:

\`\`\`text
after   1,000 steps:   0.970     ← this site's code runs max_steps = 1000
after  10,000 steps:   0.741
after  50,000 steps:   0.223
after 100,000 steps:   0.050
\`\`\`

A quick estimate: $0.99997^n \\approx e^{-0.00003\\, n}$. For example, at 10,000 steps $0.00003 \\times 10{,}000 = 0.3$ and $e^{-0.3} \\approx 0.741$. After about $1 / 0.00003 \\approx 33{,}000$ steps the weight is down to $1/e \\approx 37\\%$.

So weight decay is a slow, gentle pull. It does not erase useful weights, whose gradients push far harder than 0.003% per step; it acts on weights that go unneeded for tens of thousands of steps. With only 1000 steps, this site's training gets at most a 3% shrink from it; real GPT runs take hundreds of thousands of steps.

Because the decay is multiplied by $\\eta$, it also follows the learning rate: with an LR schedule (GPT-3 used warmup followed by cosine decay), the decay weakens as $\\eta$ goes down. This site's code has no LR schedule (see the [optimizer overview](#info:optim_adamw_notes)).

### Which parameters are decayed

The code splits the parameters into two groups by number of dimensions:

\`\`\`python
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
\`\`\`

For layer 0, for example:

\`\`\`text
decayed (2D):
  tok_emb.weight           (50304, 768)
  pos_emb.weight           (1024, 768)
  attn.c_attn.weight       (2304, 768)
  attn.proj.weight         (768, 768)
  mlp.fc_in.weight         (3072, 768)
  mlp.fc_out.weight        (768, 3072)

not decayed (1D):
  ln1.weight (γ), ln1.bias (β)     (768,)
  ln2.weight (γ), ln2.bias (β)     (768,)
  attn.c_attn.bias                 (2304,)
  attn.proj.bias                   (768,)
  mlp.fc_in.bias                   (3072,)
  mlp.fc_out.bias                  (768,)
\`\`\`

Over the whole model:

\`\`\`text
decayed:      50 tensors,  124,354,560 parameters   (99.9%)
not decayed:  98 tensors,      121,344 parameters   (0.1%)
\`\`\`

(\`lm_head.weight\` is the same tensor as \`tok_emb.weight\` because of [weight tying](#info:weight_tying_notes), so it is counted once.)

- **LayerNorm $\\gamma$ and $\\beta$:** in PyTorch's \`nn.LayerNorm\`, \`.weight\` is $\\gamma$ (initialized to 1) and \`.bias\` is $\\beta$ (initialized to 0), in $y = \\gamma \\cdot \\hat{x} + \\beta$. The natural value of $\\gamma$ is 1, not 0, so pulling it toward 0 would just shrink LayerNorm's output; that damages the model rather than simplifying it. $\\beta$ only shifts the output, so decaying it serves no regularization purpose either.
- **Biases:** a bias is added, not multiplied with the input. Memorizing the data with large weights happens in the matrices that multiply the input, not in biases. And biases are only 0.1% of the parameters, so decaying them would bring almost no benefit.

Decay is aimed at the large matrices that multiply the input (Linear and Embedding weights), which is where the model's capacity lies. \`p.dim() >= 2\` separates exactly those from the biases and LayerNorm parameters. This recipe comes from BERT and is used by GPT-3, nanoGPT, and others.

## References

1. **Decoupled Weight Decay Regularization** (AdamW). Loshchilov & Hutter, 2017. [arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
`,
    },
    zh: {
      title: "AdamW",
      md: `
**AdamW** 就是 [Adam](#info:adam_notes) 加上 **weight decay（权重衰减）**，而且 decay 直接作用在权重上，不加进梯度里。名字里的 "W" 指的就是这种 decoupled（解耦的）**W**eight decay。这个网站的 GPT-2 代码用的就是它。

## Weight decay 是什么

每一步更新之前，先把每个权重缩小一点点：

$$
\\theta \\leftarrow (1 - \\eta \\lambda)\\, \\theta
$$

- $\\lambda$ 是衰减强度（这个网站的代码里是 0.1）；$\\eta$ 是学习率。
- $\\eta = 0.1$、$\\lambda = 0.1$ 时，$1 - \\eta\\lambda = 0.99$：每一步先把每个权重乘以 0.99。

**为什么要这样做？** 如果 loss 不需要某个权重，梯度就不会持续推它，decay 会慢慢把它拉向 0。如果某个权重对降低 loss 有用，梯度会一直把它推回去，它就能保持比较大的值。这样模型就不容易靠少数几个很大的权重去死记训练数据。这是一种**正则化（regularization）**。

## 在 SGD 里，L2 正则化就是 weight decay

另一种让权重保持较小的常见做法是 **L2 正则化**：在 loss 上加一个惩罚项，

$$
L_{\\text{total}} = L + \\frac{\\lambda}{2}\\, \\theta^2
$$

惩罚项的梯度是 $\\lambda \\theta$，所以 SGD 的更新变成

$$
\\theta \\leftarrow \\theta - \\eta\\, (g + \\lambda \\theta) = (1 - \\eta \\lambda)\\, \\theta - \\eta\\, g
$$

也就是"先缩小 $(1 - \\eta\\lambda)$ 倍，再走正常的一步"，和 weight decay 完全一样。在 SGD 里两者是同一件事，所以以前这两个名字一直混着用。

## 在 Adam 里，L2 出了问题

换成 Adam 以后，两者就**不一样**了。L2 的写法（\`torch.optim.Adam(weight_decay=...)\` 就是这样做的）是把 $\\lambda\\theta$ 加进梯度，再交给 Adam：

\`\`\`text
g' = g + λθ
m 和 v 都用 g' 来算，最后走 η · m̂ / √v̂
\`\`\`

Adam 分不清 $g'$ 里哪部分来自 loss、哪部分是 decay，它把 $g'$ **整个**都除以 $\\sqrt{\\hat{v}}$。$\\sqrt{\\hat{v}}$ 是以前若干个 $g'$ 的均方根：大约是最近 $\\frac{1}{1 - \\beta_2} = 1000$ 个，不足 1000 步时就是到目前为止所有的（见 [Adam](#info:adam_notes)）。所以 decay 那部分也被缩放了：

- **梯度大的参数：** $\\sqrt{\\hat{v}}$ 很大，decay 被除得几乎没了，**几乎不衰减**。
- **梯度小的参数：** $\\sqrt{\\hat{v}}$ 很小，decay 反而被放大，**衰减得很猛**。

### 例子：同一个 $\\lambda$，相差 100 倍

两个参数 A 和 B，此刻都是 $\\theta = 1$；$\\eta = 0.01$，$\\lambda = 0.1$，所以 decay 给两者的梯度都加上 $\\lambda\\theta = 0.1$。A 自己的梯度在 $+10$ 和 $-10$ 之间来回跳；B 的在 $+0.01$ 和 $-0.01$ 之间来回跳。两者平均都是 0，也就是 loss 并不需要这两个权重，这正是 decay 应该把它们拉向 0 的情况。

**第一步：$\\sqrt{\\hat{v}}$ 是多少？** 它是以前若干个 $g' = g + 0.1$ 的均方根：

\`\`\`text
A:  g' = 10.1, −9.9, 10.1, …    平方 102.01, 98.01   →  平均 100.01,  √v̂ ≈ 10
B:  g' = 0.11, 0.09, 0.11, …    平方 0.0121, 0.0081  →  平均 0.0101,  √v̂ ≈ 0.1
\`\`\`

A 的 $g'$ 主要由它自己的梯度决定，多出来的 0.1 几乎看不出来。B 的 $g'$ 几乎全是 decay 那一项，它自己的梯度只是小小的抖动。

**第二步：decay 那部分让 $\\theta$ 变了多少？** Adam 的一步是 $\\eta \\cdot \\hat{m} / \\sqrt{\\hat{v}}$，而 $\\hat{m}$ 是 $g'$ 的平均，所以 decay 那一项的贡献是 $\\eta \\lambda \\theta / \\sqrt{\\hat{v}}$：

\`\`\`text
A:  0.01 × 0.1 / 10   = 0.0001   →  每步缩小 0.01%
B:  0.01 × 0.1 / 0.1  = 0.01     →  每步缩小 1%
\`\`\`

**第三步：和本来想要的比。** 设定 $\\lambda = 0.1$，本意是每步缩小 $\\eta\\lambda = 0.001$，也就是 0.1%：

| | 本意（AdamW） | Adam + L2 |
|---|---|---|
| A（梯度大） | 0.1% | 0.01%，弱了 10 倍 |
| B（梯度小） | 0.1% | 1%，强了 10 倍 |

用的是同一个 $\\lambda$，实际衰减却差了 **100 倍**。原因只有一个：decay 也被除以了 $\\sqrt{\\hat{v}}$。

**跑 200 步**（PyTorch 实测）：

\`\`\`text
                 A (梯度 ±10)    B (梯度 ±0.01)
不加 decay          0.982            0.982
Adam + L2           0.963            0.018     ← A 几乎没衰减，B 几乎到 0
AdamW               0.804            0.804     ← 两个衰减得一样
\`\`\`

- **A：** 每步少 0.0001，200 步共少约 0.02：0.982 → 0.963。
- **B：** 每步少约 0.01，大约 100 步就到了 0 附近。（$\\theta$ 变小以后，decay 那一项和 $\\sqrt{\\hat{v}}$ 也跟着变小，所以不会正好到 0。）
- **AdamW：** 每步乘 $0.999$，$0.999^{200} \\approx 0.819$，所以两者都是 $0.982 \\times 0.819 \\approx 0.804$。

## 更新规则

AdamW 算 $m_t$、$\\hat{m}_t$、$v_t$、$\\hat{v}_t$ 的方法和 [Adam](#info:adam_notes) 完全一样，用的是原本的梯度 $g_t$（不加 $\\lambda\\theta$）。只有最后一行不同：

$$
\\begin{aligned}
\\text{Adam:}\\quad & \\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t \\\\
\\text{AdamW:}\\quad & \\theta_t = (1 - \\eta \\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t
\\end{aligned}
$$

decay 的 $(1 - \\eta\\lambda)$ 直接乘在权重上，不经过 $\\sqrt{\\hat{v}}$，所以不管梯度是什么样，每个参数都按同样的比例缩小。

## 数值例子

和 [Adam](#info:adam_notes) note 用同一个问题，方便对比。

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$，在 $\\theta = (3, -2)$ 处最小
- **梯度：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **更新：** $\\theta_t = (1 - \\eta\\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **超参数：** $\\eta = 0.1$，$\\beta_1 = 0.9$，$\\beta_2 = 0.999$，$\\epsilon = 10^{-8}$（这里可以忽略），$\\lambda = 0.1$，所以衰减系数是 $1 - \\eta\\lambda = 0.99$
- **初始值：** $\\theta_0 = (0, 0)$，$m_0 = (0, 0)$，$v_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + 0.1 \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + 0.001 \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- $\\hat{m}_1 = \\frac{(-0.3, 0.2)}{1 - 0.9} = (-3, 2)$，$\\quad \\hat{v}_1 = \\frac{(0.009, 0.004)}{1 - 0.999} = (9, 4)$，$\\quad \\sqrt{\\hat{v}_1} = (3, 2)$
- **Decay：** $0.99 \\cdot (0, 0) = (0, 0)$
- **Adam 的一步：** $0.1 \\cdot \\frac{(-3, 2)}{(3, 2)} = (-0.1, 0.1)$
- $\\theta_1 = (0, 0) - (-0.1, 0.1) = (0.1, -0.1)$

decay 这时还没起作用：$0.99 \\times 0 = 0$。第 1 步和 Adam 完全一样。

### 第 2 步

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + 0.1 \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + 0.001 \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- $\\hat{m}_2 = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.9474, 1.9474)$，$\\quad \\hat{v}_2 = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.7049, 3.8049)$，$\\quad \\sqrt{\\hat{v}_2} = (2.9504, 1.9506)$
- **Decay：** $0.99 \\cdot (0.1, -0.1) = (0.099, -0.099)$
- **Adam 的一步：** $0.1 \\cdot \\frac{(-2.9474, 1.9474)}{(2.9504, 1.9506)} = (-0.0999, 0.0998)$
- $\\theta_2 = (0.099, -0.099) - (-0.0999, 0.0998) = (0.1989, -0.1988)$

### 第 3 步

- $\\theta_2 = (0.1989, -0.1988)$
- $g_3 = (0.1989 - 3,\\; -0.1988 + 2) = (-2.8011, 1.8012)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + 0.1 \\cdot (-2.8011, 1.8012) = (-0.7841, 0.5131)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + 0.001 \\cdot \\big((-2.8011, 1.8012) \\odot (-2.8011, 1.8012)\\big) = (0.025230, 0.010843)$
- $\\hat{m}_3 = \\frac{(-0.7841, 0.5131)}{1 - 0.9^3} = (-2.8934, 1.8934)$，$\\quad \\hat{v}_3 = \\frac{(0.025230, 0.010843)}{1 - 0.999^3} = (8.4183, 3.6178)$，$\\quad \\sqrt{\\hat{v}_3} = (2.9014, 1.9021)$
- **Decay：** $0.99 \\cdot (0.1989, -0.1988) = (0.1969, -0.1968)$
- **Adam 的一步：** $0.1 \\cdot \\frac{(-2.8934, 1.8934)}{(2.9014, 1.9021)} = (-0.0997, 0.0995)$
- $\\theta_3 = (0.1969, -0.1968) - (-0.0997, 0.0995) = (0.2966, -0.2964)$

### 这个例子说明了什么

把 $\\theta_1$ 和同一个问题上的 Adam 对比：

\`\`\`text
          Adam      AdamW
第 1 步   0.1000    0.1000
第 2 步   0.1999    0.1989    ← 小了 0.001
第 3 步   0.2996    0.2966    ← 小了 0.003
\`\`\`

- 差别就是 decay：每一步拿掉当前权重的 1%（$0.99 \\times 0.1 = 0.099$，第 2 步少了 0.001），这个效果会一步步累积。
- Adam 那一步本身还是每步约 $\\eta = 0.1$，和 Adam 一样，因为喂给 $m$ 和 $v$ 的梯度里不含 decay。
- 这里 decay 把 $\\theta$ 往 0 拉，也就是离最低点 $(3, -2)$ 更远了。在这个玩具问题里，每个权重都是需要的，所以 decay 只会拖后腿；它的好处要在真实模型里才体现出来，因为那里有很多权重是不需要的。

## 在 PyTorch 和这个网站的 GPT-2 代码里

\`torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)\`。每一步先把每个权重乘以 $1 - \\eta\\lambda$（\`1 - lr * weight_decay\`），再走 Adam 的一步。

### decay 实际有多强

注意 $\\eta\\lambda$ 里的 $\\eta$：\`weight_decay=0.1\` 的意思**不是**"每步缩小 10%"。代码里 \`lr=3e-4\`，\`weight_decay=0.1\`：

\`\`\`text
η·λ = 0.0003 × 0.1 = 0.00003
每步:  θ ← 0.99997 · θ          ← 每步只缩小 0.003%
\`\`\`

为了单独看 decay 的力量，假设某个权重**完全没有梯度**：这时 $m = 0$，Adam 那一步是 0，只剩 decay 在起作用。从 1 开始，$n$ 步之后就是 $0.99997^n$：

\`\`\`text
  1,000 步后:   0.970     ← 这个网站的代码 max_steps = 1000
 10,000 步后:   0.741
 50,000 步后:   0.223
100,000 步后:   0.050
\`\`\`

快速估算：$0.99997^n \\approx e^{-0.00003\\, n}$。比如 10,000 步：$0.00003 \\times 10{,}000 = 0.3$，而 $e^{-0.3} \\approx 0.741$。大约 $1 / 0.00003 \\approx 33{,}000$ 步后，权重缩小到 $1/e \\approx 37\\%$。

所以 weight decay 是一个很慢、很温和的力。它不会抹掉有用的权重，因为这些权重的梯度推力远远大于每步 0.003% 的拉力；它真正起作用的，是那些几万步都没被需要的权重。这个网站的训练只有 1000 步，decay 最多让权重缩小 3%；真正的 GPT 训练要跑几十万步。

decay 乘了 $\\eta$，所以它也会跟着学习率变：如果用 LR schedule（GPT-3 用了先 warmup 再 cosine 下降），$\\eta$ 变小时 decay 也一起变弱。这个网站的代码没有 LR schedule（见 [optimizer 总览](#info:optim_adamw_notes)）。

### 哪些参数做 decay

代码按维度把参数分成两组：

\`\`\`python
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
\`\`\`

以第 0 层为例：

\`\`\`text
做 decay (2D):
  tok_emb.weight           (50304, 768)
  pos_emb.weight           (1024, 768)
  attn.c_attn.weight       (2304, 768)
  attn.proj.weight         (768, 768)
  mlp.fc_in.weight         (3072, 768)
  mlp.fc_out.weight        (768, 3072)

不做 decay (1D):
  ln1.weight (γ), ln1.bias (β)     (768,)
  ln2.weight (γ), ln2.bias (β)     (768,)
  attn.c_attn.bias                 (2304,)
  attn.proj.bias                   (768,)
  mlp.fc_in.bias                   (3072,)
  mlp.fc_out.bias                  (768,)
\`\`\`

整个模型统计下来：

\`\`\`text
做 decay:     50 个 tensor,  124,354,560 个参数   (99.9%)
不做 decay:   98 个 tensor,      121,344 个参数   (0.1%)
\`\`\`

（因为 [weight tying](#info:weight_tying_notes)，\`lm_head.weight\` 和 \`tok_emb.weight\` 是同一个 tensor，只算一次。）

- **LayerNorm 的 $\\gamma$ 和 $\\beta$：** 在 PyTorch 的 \`nn.LayerNorm\` 里，\`.weight\` 就是 $\\gamma$（初始化为 1），\`.bias\` 就是 $\\beta$（初始化为 0），对应 $y = \\gamma \\cdot \\hat{x} + \\beta$。$\\gamma$ 的"中性"值是 1，不是 0，把它往 0 拉只会把 LayerNorm 的输出越压越小，这是在破坏模型，而不是让它更简单。$\\beta$ 只做平移，往 0 拉也没有正则化的意义。
- **Bias：** bias 是加上去的，不和输入相乘。模型"靠很大的权重死记数据"，靠的是那些和输入相乘的矩阵，不是 bias。而且 bias 总共只占 0.1%，对它们做 decay 几乎得不到好处。

decay 的目标是那些**和输入相乘的大矩阵**（Linear 和 Embedding 的 weight），它们才是模型容量的来源。\`p.dim() >= 2\` 正好把它们和 bias、LayerNorm 的参数分开。这个做法来自 BERT，GPT-3、nanoGPT 等也都这样做。

## 参考资料

1. **Decoupled Weight Decay Regularization**（AdamW）。Loshchilov & Hutter，2017。[arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
`,
    },
  },
};
