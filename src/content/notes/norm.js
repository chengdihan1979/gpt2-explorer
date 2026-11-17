// Notes: LayerNorm, pre-norm vs post-norm, RMSNorm.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  layernorm: {
    en: {
      title: "LayerNorm",
      md: `
**\`nn.LayerNorm\`** applies layer normalization over the last dimension of the input (the embedding dimension $n\\_embd$).

## The formula

Given any input vector $h \\in \\mathbb{R}^{d}$, LayerNorm computes

$$
\\text{LN}(h) = \\gamma \\odot \\frac{h - \\mu(h)}{\\sqrt{\\sigma^2(h) + \\epsilon}} + \\beta
$$

- $\\mu, \\sigma^2$ are the mean and variance over the last (feature) dimension.
- $\\gamma, \\beta \\in \\mathbb{R}^{d}$ are a learned scale and shift. [Shared by all tokens? One per block?](#hover:layernorm_params_notes)
- $\\epsilon$ is a small constant for numerical safety ($10^{-5}$ by default in PyTorch).

> It applies to **each token individually**, across its embedding dimensions. GPT-2 uses it in three places: \`ln1\` and \`ln2\` inside every block, and the **final layer norm** \`ln_f\` before the output projection (\`lm_head\`).

### Example

$\\gamma$ and $\\beta$ are ignored for simplicity. For a token embedding vector

$$
x = [5.0,\\; 4.5,\\; 6.2,\\; 4.8],
$$

LayerNorm computes the mean and standard deviation over the 4 dimensions:

$$
\\mu = 5.125, \\quad \\sigma \\approx 0.646
$$

and normalizes each element:

$$
\\hat{x}_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}
$$

So the normalized output is approximately

$$
\\hat{x} = [-0.19,\\; -0.97,\\; 1.66,\\; -0.50].
$$

## Why it's used in GPT-style models

Each sublayer reads from the residual stream, which only ever accumulates: $x_{24} = x_0 + \\Delta_1 + \\cdots + \\Delta_{24}$ over the 24 sublayers ([why residual](#hover:residual_why_notes)). Every token's vector has the same 768 dimensions, but its **magnitude** (norm) differs:

- the embedding rows already differ in size (in real GPT-2, the norm of position 0's row is 9.88; position 1023's is 0.12);
- each sublayer adds a different amount to each token;
- these differences build up with depth.

Why does that matter? Attention and the MLP apply the **same fixed weights** to every token in a layer. Without normalization, the size of what they receive would keep changing: from token to token within a layer, and from layer to layer as the stream grows. And size alone changes what they compute: a larger input can push attention's softmax into saturation and make GELU behave differently, even if its direction (its "meaning") is the same.

[Why? Examples: attention, GELU, and a moving target](#hover:layernorm_scale_notes)

### 1. Stabilizes training by controlling magnitude

After $x \\rightarrow \\hat{x} = \\frac{x - \\mu}{\\sigma}$, the softmax in attention sees $\\frac{QK^\\top}{\\sqrt{d}}$ on a stable scale, reducing exploding logits and saturated softmax. The MLP nonlinearity also operates in a stable range.

**Quick numbers.** $Q$ and $K$ are linear in $x$, so doubling $x$ doubles both and multiplies every dot product by 4:

- scores $[2, 1, 0]$ → softmax $\\approx [0.665, 0.245, 0.090]$: attention is spread out;
- scores $[8, 4, 0]$ → softmax $\\approx [0.982, 0.018, 0.0003]$: almost all weight on one position, and gradients through the softmax nearly vanish.

[Why? A tiny numeric example](#hover:stabilize_training_notes)

### 2. Improves gradient flow

The Jacobian around normalized activations is better conditioned (roughly scaled by $\\frac{1}{\\sigma}$), which curbs exploding/vanishing gradients and reduces sensitivity to the learning-rate choice.

[Why? A tiny numeric example](#hover:improves_gradient_flow_notes)

### 3. Provides scale invariance

Rescaling upstream weights changes $\\mu$ and $\\sigma$ in the same proportion, so the normalized activations barely change. The model learns "directional" features, while LayerNorm's $\\gamma$ and $\\beta$ reintroduce any useful scale/shift.

**Quick numeric example.** Let $x = [2, -1, 1]$ and scale it by $c = 10$ to $x' = [20, -10, 10]$.

- **Without LN:** $x'$ is 10× larger → Q/K dot products and logits become ~10× larger → likely softmax saturation.
- **With LN:** $\\hat{x}' = \\hat{x}$. The forward behavior after LN is the same, and backprop through LN also accounts for $\\frac{1}{\\sigma}$, so gradients aren't inflated by the scale.

## Summary: three angles on stability

The three points are related but not the same; each tackles stability from a different angle.

1. **Forward magnitude control (avoids softmax saturation).** LN standardizes each token's vector so dot products/logits don't blow up ⇒ softmax stays out of saturation ⇒ its Jacobian $\\mathrm{diag}(p) - pp^\\top$ doesn't collapse to zero, so gradients keep flowing through it. It targets the numerical size of forward activations after LN.

2. **Gradient flow conditioning (LN's Jacobian).** In backprop, LN applies a specific linear map. With $g$ the gradient arriving at the normalized output $\\hat{y}$:

   $$
   \\nabla_{x} L = \\frac{1}{\\sigma} \\left( g - \\mu(g)\\mathbf{1} - \\hat{y}\\,\\mu(g \\odot \\hat{y}) \\right)
   $$

   The $\\frac{1}{\\sigma}$ scales the gradient and the two subtracted terms project out components, directly shaping gradients and reducing exploding/vanishing. It targets the local conditioning of gradients at LN during backprop, independent of upstream rescaling. Intuitively, whatever gradient arrives at LN, it is reshaped into a well-conditioned, scale-controlled signal before being sent further back.

3. **Scale invariance (decouples direction from scale).** Upstream scaling doesn't affect LN's output, since $\\mu$ and $\\sigma$ scale accordingly. It targets invariance to arbitrary rescaling from earlier layers.

## Pre-norm vs post-norm

GPT-2 uses **pre-norm**, unlike the **post-norm** of the original Transformer paper. The difference is only where LayerNorm sits relative to the residual add.

**Post-norm (original paper):** LN normalizes after the residual addition.

$$
\\text{SA: } x_1 = \\text{LN}(x + \\text{MHA}(x)) \\\\[4pt]
\\text{FFN: } y = \\text{LN}(x_1 + \\text{FFN}(x_1))
$$

**Pre-norm (GPT-2, and widely used now for deep stacks):** LN comes before each sublayer.

$$
\\text{SA: } x_1 = x + \\text{MHA}(\\text{LN}(x)) \\\\[4pt]
\\text{FFN: } y = x_1 + \\text{FFN}(\\text{LN}(x_1))
$$

In pre-norm, LN is applied to a **copy** of the stream. In \`x = x + self.attn(self.ln1(x))\`, the normalized \`self.ln1(x)\` goes into attention and is then discarded; the update is added back to the **un-normalized** $x$. Across all 12 blocks the stream itself is never normalized, only added to. That is why GPT-2 adds the final \`ln_f\`: it normalizes the stream once, right before \`lm_head\`. In post-norm, by contrast, $x$ itself is replaced by $\\text{LN}(\\dots)$ after every sublayer.

> The main reason pre-norm is widely used is that the residual gives gradients an identity path: $\\frac{\\partial y}{\\partial x} = I + \\dots$ This makes deep stacks train stably.

See a [tiny numeric example](#hover:post_norm_gradient_example_notes) that illustrates the issue with post-norm.

## Recent developments: RMSNorm

Among recent open-weight LLMs, **RMSNorm** is the default more often than LayerNorm. Examples:

- **Llama 2:** the paper says "we … apply pre-normalization using RMSNorm" (§2.2, Training Details). [Llama 2 paper ↗](https://ar5iv.labs.arxiv.org/html/2307.09288)
- **Llama 3:** Meta's official model code defines and uses \`RMSNorm\`. [meta-llama/llama3 model.py ↗](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- **Mistral 7B:** Hugging Face's Mistral model code defines and uses \`MistralRMSNorm\`. [modeling_mistral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- **Mixtral:** Hugging Face's Mixtral implementation (derived from the Mistral codebase) uses RMSNorm. [modeling_mixtral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)
- **Qwen:** model cards list the architecture as "transformers with RoPE, SwiGLU, RMSNorm …" (e.g. Qwen2.5-1.5B). [Qwen2.5-1.5B ↗](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Gemma:** Hugging Face's Gemma docs describe the model as using GeGLU activations and RMSNorm. [Gemma docs ↗](https://huggingface.co/docs/transformers/en/model_doc/gemma)

See a [tiny numeric example](#hover:rmsnorm_gradient_example_notes) of how RMSNorm works.

## References

1. **Layer Normalization (2016)**

   Ba, Kiros, Hinton. Defines LN, contrasts it with BatchNorm, and motivates per-example normalization.

   [Layer Normalization ↗](https://arxiv.org/pdf/1607.06450)

2. **GPT-2 technical report: where LN sits in GPT-2 (pre-LN)**

   Radford et al. note that LN is moved to the input of each sub-block and that a final LN is added.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

3. **PyTorch API: the exact math and arguments used in practice**

   \`torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine, …)\`. Useful for implementation details and the $\\epsilon$ you'll see in configs.

   [torch.nn.LayerNorm ↗](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

4. **Pre-LN vs post-LN analysis (why GPT-2 uses pre-LN)**

   Xiong et al., "On Layer Normalization in the Transformer Architecture." Theoretical and empirical comparison; shows pre-LN stabilizes training (often without warm-up).

   [On Layer Normalization in the Transformer Architecture ↗](https://arxiv.org/pdf/2002.04745)

5. **RMSNorm**

   Zhang & Sennrich introduce RMSNorm, a lighter alternative (no centering) with similar quality and speed gains. A useful contrast to LN.

   [RMSNorm ↗](https://arxiv.org/pdf/1910.07467)
`,
    },
    zh: {
      title: "LayerNorm",
      md: `
**\`nn.LayerNorm\`** 在输入的最后一个维度（embedding 维度 $n\\_embd$）上做 layer normalization。

## 公式

对任意输入向量 $h \\in \\mathbb{R}^{d}$，LayerNorm 计算

$$
\\text{LN}(h) = \\gamma \\odot \\frac{h - \\mu(h)}{\\sqrt{\\sigma^2(h) + \\epsilon}} + \\beta
$$

- $\\mu, \\sigma^2$ 是在最后一个（特征）维度上算的均值和方差。
- $\\gamma, \\beta \\in \\mathbb{R}^{d}$ 是可学习的缩放和平移。[所有 token 共用吗？每个 block 一套吗？](#hover:layernorm_params_notes)
- $\\epsilon$ 是一个很小的常数，防止除以 0（PyTorch 默认 $10^{-5}$）。

> 它对**每个 token 单独**计算，范围是这个 token 的所有 embedding 维度。GPT-2 在三个地方用它：每个 block 里的 \`ln1\` 和 \`ln2\`，以及输出投影（\`lm_head\`）之前的**最后一个 layer norm** \`ln_f\`。

### 例子

为了简单，先忽略 $\\gamma$ 和 $\\beta$。对一个 token 的 embedding 向量

$$
x = [5.0,\\; 4.5,\\; 6.2,\\; 4.8],
$$

LayerNorm 在这 4 个维度上算均值和标准差：

$$
\\mu = 5.125, \\quad \\sigma \\approx 0.646
$$

再对每个元素做归一化：

$$
\\hat{x}_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}
$$

得到的输出约为

$$
\\hat{x} = [-0.19,\\; -0.97,\\; 1.66,\\; -0.50].
$$

## 为什么 GPT 类模型要用它

每个子层都从残差流读取输入，而残差流只会不断累加：经过 24 个子层后，$x_{24} = x_0 + \\Delta_1 + \\cdots + \\Delta_{24}$（[为什么用残差](#hover:residual_why_notes)）。每个 token 的向量都是 768 维，但**大小**（范数）各不相同：

- embedding 的各行本来就大小不一（真实 GPT-2 里，位置 0 那一行的范数是 9.88，位置 1023 是 0.12）；
- 每个子层给每个 token 加的量不一样；
- 这些差别随着层数加深不断累积。

这有什么影响？在同一层里，attention 和 MLP 对每个 token 用的是**同一套固定权重**。如果不做归一化，它们收到的输入大小会一直变化：同一层里 token 和 token 之间不同，随着残差流变大，层与层之间也不同。而光是大小变化就会改变计算结果：即使方向（也就是"含义"）不变，更大的输入也可能让 attention 的 softmax 饱和，让 GELU 的表现不同。

[为什么？例子：attention、GELU 和不断移动的目标](#hover:layernorm_scale_notes)

### 1. 控制大小，让训练稳定

经过 $x \\rightarrow \\hat{x} = \\frac{x - \\mu}{\\sigma}$ 之后，attention 里 softmax 看到的 $\\frac{QK^\\top}{\\sqrt{d}}$ 保持在稳定的尺度上，减少 logits 爆炸和 softmax 饱和。MLP 的非线性函数也在稳定的范围里工作。

**简单算一下。** $Q$ 和 $K$ 都是 $x$ 的线性函数，所以 $x$ 翻倍，两者都翻倍，每个点积变成 4 倍：

- 分数 $[2, 1, 0]$ → softmax $\\approx [0.665, 0.245, 0.090]$：注意力比较分散；
- 分数 $[8, 4, 0]$ → softmax $\\approx [0.982, 0.018, 0.0003]$：几乎全部权重都在一个位置上，穿过 softmax 的梯度也几乎消失。

[为什么？一个很小的数值例子](#hover:stabilize_training_notes)

### 2. 改善梯度流动

在归一化后的激活值附近，Jacobian 的条件更好（大致按 $\\frac{1}{\\sigma}$ 缩放），这能抑制梯度爆炸和消失，也让训练对学习率的选择不那么敏感。

[为什么？一个很小的数值例子](#hover:improves_gradient_flow_notes)

### 3. 尺度不变性

把上游的权重整体放大或缩小，$\\mu$ 和 $\\sigma$ 会按同样比例变化，所以归一化后的结果几乎不变。模型学到的是"方向"上的特征，需要的尺度和偏移再由 LayerNorm 的 $\\gamma$ 和 $\\beta$ 补回来。

**简单数值例子。** 设 $x = [2, -1, 1]$，乘以 $c = 10$ 得到 $x' = [20, -10, 10]$。

- **没有 LN：** $x'$ 大了 10 倍 → Q/K 点积和 logits 也大了约 10 倍 → softmax 很可能饱和。
- **有 LN：** $\\hat{x}' = \\hat{x}$。LN 之后的 forward 完全一样；backward 穿过 LN 时也会乘以 $\\frac{1}{\\sigma}$，所以梯度不会因为尺度变大而被放大。

## 总结：从三个角度看稳定性

这三点相关但不相同，分别从不同角度解决稳定性问题。

1. **控制 forward 的大小（避免 softmax 饱和）。** LN 把每个 token 的向量标准化，点积和 logits 就不会变得太大 ⇒ softmax 不会饱和 ⇒ 它的 Jacobian $\\mathrm{diag}(p) - pp^\\top$ 不会塌缩成 0，梯度能继续穿过它。它针对的是 LN 之后 forward 激活值的数值大小。

2. **调节梯度流动（LN 的 Jacobian）。** backward 时，LN 对梯度做一个特定的线性变换。设 $g$ 是传到归一化输出 $\\hat{y}$ 上的梯度：

   $$
   \\nabla_{x} L = \\frac{1}{\\sigma} \\left( g - \\mu(g)\\mathbf{1} - \\hat{y}\\,\\mu(g \\odot \\hat{y}) \\right)
   $$

   $\\frac{1}{\\sigma}$ 对梯度做缩放，减去的两项则去掉了某些分量，直接塑造梯度，减少爆炸和消失。它针对的是 backward 时梯度在 LN 处的局部条件，和上游怎么缩放无关。直观地说：不管传到 LN 的梯度是什么样，在继续往回传之前，都会被整理成一个条件良好、尺度受控的信号。

3. **尺度不变性（把方向和大小分开）。** 上游的缩放不影响 LN 的输出，因为 $\\mu$ 和 $\\sigma$ 会跟着按比例变化。它针对的是前面的层任意缩放时的不变性。

## Pre-norm 和 post-norm

GPT-2 用的是 **pre-norm**，和 Transformer 原论文的 **post-norm** 不同。两者唯一的区别是 LayerNorm 相对于残差相加的位置。

**Post-norm（原论文）：** 残差相加之后再做 LN。

$$
\\text{SA: } x_1 = \\text{LN}(x + \\text{MHA}(x)) \\\\[4pt]
\\text{FFN: } y = \\text{LN}(x_1 + \\text{FFN}(x_1))
$$

**Pre-norm（GPT-2，现在的深层模型也普遍这么用）：** LN 放在每个子层之前。

$$
\\text{SA: } x_1 = x + \\text{MHA}(\\text{LN}(x)) \\\\[4pt]
\\text{FFN: } y = x_1 + \\text{FFN}(\\text{LN}(x_1))
$$

在 pre-norm 里，LN 作用在残差流的一个**副本**上。在 \`x = x + self.attn(self.ln1(x))\` 中，归一化后的 \`self.ln1(x)\` 进入 attention 之后就不再使用；更新量加回的是**没有归一化的** $x$。整整 12 个 block 里，残差流本身从来没被归一化过，只是不断被加上更新量。这就是 GPT-2 要额外加一个 \`ln_f\` 的原因：在 \`lm_head\` 之前，对残差流做一次归一化。相比之下，在 post-norm 里，每个子层之后 $x$ 本身都会被替换成 $\\text{LN}(\\dots)$。

> pre-norm 被广泛使用的主要原因是：残差给了梯度一条恒等路径，$\\frac{\\partial y}{\\partial x} = I + \\dots$，所以深层网络能稳定训练。

看一个[很小的数值例子](#hover:post_norm_gradient_example_notes)，说明 post-norm 的问题。

## 新进展：RMSNorm

在近来的开源权重大模型里，**RMSNorm** 比 LayerNorm 更常作为默认选择。例如：

- **Llama 2：** 论文写道 "we … apply pre-normalization using RMSNorm"（§2.2，Training Details）。[Llama 2 论文 ↗](https://ar5iv.labs.arxiv.org/html/2307.09288)
- **Llama 3：** Meta 官方的模型代码定义并使用了 \`RMSNorm\`。[meta-llama/llama3 model.py ↗](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- **Mistral 7B：** Hugging Face 的 Mistral 模型代码定义并使用了 \`MistralRMSNorm\`。[modeling_mistral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- **Mixtral：** Hugging Face 的 Mixtral 实现（来自 Mistral 的代码）使用 RMSNorm。[modeling_mixtral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)
- **Qwen：** 模型卡上写的架构是 "transformers with RoPE, SwiGLU, RMSNorm …"（例如 Qwen2.5-1.5B）。[Qwen2.5-1.5B ↗](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Gemma：** Hugging Face 的 Gemma 文档说明它使用 GeGLU 激活和 RMSNorm。[Gemma 文档 ↗](https://huggingface.co/docs/transformers/en/model_doc/gemma)

看一个[很小的数值例子](#hover:rmsnorm_gradient_example_notes)，了解 RMSNorm 怎么算。

## 参考资料

1. **Layer Normalization（2016）**

   Ba、Kiros、Hinton。定义了 LN，和 BatchNorm 做了对比，并说明了为什么要对每个样本单独归一化。

   [Layer Normalization ↗](https://arxiv.org/pdf/1607.06450)

2. **GPT-2 技术报告：LN 在 GPT-2 里的位置（pre-LN）**

   Radford 等人说明，LN 被移到了每个子模块的输入处，并且额外加了一个最后的 LN。

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

3. **PyTorch API：实际使用的公式和参数**

   \`torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine, …)\`。可以查实现细节，以及配置里常见的 $\\epsilon$。

   [torch.nn.LayerNorm ↗](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

4. **Pre-LN 和 post-LN 的分析（GPT-2 为什么用 pre-LN）**

   Xiong 等人，"On Layer Normalization in the Transformer Architecture"。从理论和实验两方面做了对比，说明 pre-LN 让训练更稳定（常常不需要 warm-up）。

   [On Layer Normalization in the Transformer Architecture ↗](https://arxiv.org/pdf/2002.04745)

5. **RMSNorm**

   Zhang 和 Sennrich 提出了 RMSNorm，一个更轻量的替代方案（不减均值），效果相近，速度更快。适合和 LN 对比。

   [RMSNorm ↗](https://arxiv.org/pdf/1910.07467)
`,
    },
  },

  layernorm_params_notes: {
    en: {
      title: "LayerNorm Parameters",
      md: `
**Which LayerNorm values are shared, and which are not? $\\gamma$ and $\\beta$ are learned per LayerNorm and per dimension, and shared by every token.**

## Shared by all tokens within one LayerNorm

Take block 3's \`ln1\`. Every token that passes through it, from every sequence in the batch and at every position, uses **the same** $\\gamma$ and $\\beta$.

## Separate for each LayerNorm

GPT-2 small has 25 LayerNorms, and each has its own $\\gamma$ and $\\beta$. Nothing is shared between them:

- block 1: \`ln1\` has one set, \`ln2\` another;
- block 2: \`ln1\` has one set, \`ln2\` another;
- …
- block 12: \`ln1\` has one set, \`ln2\` another;
- \`ln_f\`: one more set.

## One value per dimension, not per token

$\\gamma$ and $\\beta$ each have 768 entries, one per embedding dimension:

$$
\\gamma = [\\gamma_1, \\gamma_2, \\dots, \\gamma_{768}], \\qquad \\beta = [\\beta_1, \\beta_2, \\dots, \\beta_{768}]
$$

Dimension $i$ of **every** token is scaled by the same $\\gamma_i$ and shifted by the same $\\beta_i$, but different dimensions can have different values. A 3-dimensional example with $\\gamma = [2, 1, 0.5]$ and $\\beta = [0, 1, 0]$:

- token A, normalized to $[1.22,\\; 0,\\; -1.22]$ → $[2.45,\\; 1,\\; -0.61]$;
- token B, normalized to $[-1.22,\\; 1.22,\\; 0]$ → $[-2.45,\\; 2.22,\\; 0]$.

Both tokens use the same $\\gamma$ and $\\beta$; within each token, every dimension gets its own scale and shift.

## Contrast with $\\mu$ and $\\sigma$

- **$\\mu$ and $\\sigma$** are not parameters. They are computed on the fly, **per token**, so every token has its own.
- **$\\gamma$ and $\\beta$** are learned parameters, **shared** by all tokens, with one value per dimension.

$\\gamma$ starts at all ones and $\\beta$ at all zeros. Training updates them like any other weight, and after training they are fixed.

## Parameter count

$$
25 \\text{ LayerNorms} \\times (768 + 768) = 38{,}400 \\text{ parameters}
$$

That is about 0.03% of GPT-2 small's roughly 124M parameters.
`,
    },
    zh: {
      title: "LayerNorm 的参数",
      md: `
**LayerNorm 里哪些值是共用的，哪些不是？$\\gamma$ 和 $\\beta$ 是每个 LayerNorm、每个维度各学一个，所有 token 共用。**

## 同一个 LayerNorm 里，所有 token 共用

以第 3 个 block 的 \`ln1\` 为例。穿过它的每个 token，不管来自 batch 里哪个序列、在哪个位置，用的都是**同一套** $\\gamma$ 和 $\\beta$。

## 每个 LayerNorm 各有一套

GPT-2 small 一共有 25 个 LayerNorm，每个都有自己的 $\\gamma$ 和 $\\beta$，彼此不共享：

- block 1：\`ln1\` 一套，\`ln2\` 另一套；
- block 2：\`ln1\` 一套，\`ln2\` 另一套；
- ……
- block 12：\`ln1\` 一套，\`ln2\` 另一套；
- \`ln_f\`：再一套。

## 每个维度一个值，而不是每个 token 一个

$\\gamma$ 和 $\\beta$ 各有 768 个数，每个 embedding 维度一个：

$$
\\gamma = [\\gamma_1, \\gamma_2, \\dots, \\gamma_{768}], \\qquad \\beta = [\\beta_1, \\beta_2, \\dots, \\beta_{768}]
$$

**每个** token 的第 $i$ 维都用同一个 $\\gamma_i$ 缩放、同一个 $\\beta_i$ 平移，但不同维度的值可以不一样。用一个 3 维的例子，$\\gamma = [2, 1, 0.5]$，$\\beta = [0, 1, 0]$：

- token A 归一化后是 $[1.22,\\; 0,\\; -1.22]$ → $[2.45,\\; 1,\\; -0.61]$；
- token B 归一化后是 $[-1.22,\\; 1.22,\\; 0]$ → $[-2.45,\\; 2.22,\\; 0]$。

两个 token 用的是同一套 $\\gamma$ 和 $\\beta$；在每个 token 内部，每个维度有自己的缩放和平移。

## 和 $\\mu$、$\\sigma$ 对比

- **$\\mu$ 和 $\\sigma$** 不是参数。它们是**每个 token** 当场算出来的，所以每个 token 都有自己的一对。
- **$\\gamma$ 和 $\\beta$** 是学出来的参数，所有 token **共用**，每个维度一个值。

$\\gamma$ 初始化为全 1，$\\beta$ 初始化为全 0。训练时它们和其他权重一样被更新，训练结束后就固定下来。

## 参数量

$$
25 \\text{ 个 LayerNorm} \\times (768 + 768) = 38{,}400 \\text{ 个参数}
$$

约占 GPT-2 small 大约 1.24 亿参数的 0.03%。
`,
    },
  },

  layernorm_scale_notes: {
    en: {
      title: "Why Input Scale Matters",
      md: `
**Why does it matter if the input's scale varies? Because attention and the MLP use fixed weights, and their behavior depends on the size of the input, not just its direction.**

Within one layer, every token goes through the **same** $W_q$, $W_k$, $W_v$ and the same MLP weights. Those weights are fixed, but without LayerNorm the size of their input would not be. Below are three places where that hurts.

## 1. Between tokens in the same layer: attention

Take two tokens that point in the **same direction** (think "same meaning") but differ in size:

- token A has norm 1;
- token B $= 10 \\times$ A, so it has norm 10.

$q$ and $k$ are linear in $x$, so scaling the inputs by 10 scales $q$ by 10 and $k$ by 10, and every dot product $q \\cdot k$ by **100**. Suppose a set of small inputs gives these scores:

- scores $[0.2,\\; 0.1,\\; 0]$ → softmax $\\approx [0.367,\\; 0.332,\\; 0.301]$: attention is spread over all three positions.

The same inputs, 10× larger:

- scores $[20,\\; 10,\\; 0]$ → softmax $\\approx [0.99995,\\; 0.00005,\\; 0.000000002]$: all attention goes to one position, and the softmax is saturated, so its gradients nearly vanish.

> Same direction, same weights, completely different attention, and the only difference is the size. One set of weights cannot be right for both norm 1 and norm 10.

## 2. Between tokens in the same layer: the MLP

The MLP's nonlinearity, GELU, treats small and large inputs very differently:

- input $[1,\\; -1]$ → GELU → $[0.84,\\; -0.16]$: the negative coordinate still passes a little signal;
- input $[10,\\; -10]$ → GELU → $[10.0,\\; 0.00]$: the negative coordinate is cut off entirely.

Scaling the input by 10 doesn't just scale the output by 10; it changes the output's **shape**. So the size of the input changes what computation the MLP actually performs.

## 3. Between layers: a moving target

The residual stream keeps accumulating, so early layers tend to see small inputs and later layers larger ones. Without normalization:

- each layer's weights would have to be tuned to its own typical input size;
- during training, earlier layers keep changing, so the input size that later layers see keeps changing too. Every layer would be chasing a **moving target**.

## An analogy

> Attention and the MLP are like a translator whose headphones have a fixed volume. Some speakers whisper, others shout. A whisper is too quiet to make out; a shout is so loud that only the loudest word gets through. Before understanding *what* is said, the translator has to deal with *how loud* it is. LayerNorm is an **automatic volume control** in front of the headphones: whisper or shout, it always arrives at the same level, so the translator only has to focus on the content, the vector's direction and shape.

## Bottom line

> With LayerNorm, every sublayer's input has mean 0 and standard deviation 1, whatever the token or the layer. The fixed weights only need to learn **what** the input says, not cope with **how big** it is.
`,
    },
    zh: {
      title: "为什么输入的大小很重要",
      md: `
**输入的大小不一样，为什么会有问题？因为 attention 和 MLP 用的是固定的权重，而它们的行为不只取决于输入的方向，也取决于输入的大小。**

在同一层里，每个 token 都穿过**同一套** $W_q$、$W_k$、$W_v$ 和同一套 MLP 权重。权重是固定的，但如果没有 LayerNorm，它们收到的输入大小就不固定。下面是三个会出问题的地方。

## 1. 同一层里 token 和 token 之间：attention

取两个**方向相同**（可以理解成"含义相同"）但大小不同的 token：

- token A 的范数是 1；
- token B $= 10 \\times$ A，范数是 10。

$q$ 和 $k$ 都是 $x$ 的线性函数，所以输入放大 10 倍，$q$ 放大 10 倍，$k$ 也放大 10 倍，每个点积 $q \\cdot k$ 就放大 **100** 倍。假设一组小输入得到这样的分数：

- 分数 $[0.2,\\; 0.1,\\; 0]$ → softmax $\\approx [0.367,\\; 0.332,\\; 0.301]$：注意力分散在三个位置上。

同样的输入放大 10 倍：

- 分数 $[20,\\; 10,\\; 0]$ → softmax $\\approx [0.99995,\\; 0.00005,\\; 0.000000002]$：注意力全部集中到一个位置，softmax 饱和，梯度几乎消失。

> 方向相同、权重相同，注意力却完全不同，唯一的区别只是大小。一套权重不可能同时适合范数为 1 和范数为 10 的输入。

## 2. 同一层里 token 和 token 之间：MLP

MLP 的非线性函数 GELU 对小输入和大输入的处理很不一样：

- 输入 $[1,\\; -1]$ → GELU → $[0.84,\\; -0.16]$：负的那一维还能传过去一点信号；
- 输入 $[10,\\; -10]$ → GELU → $[10.0,\\; 0.00]$：负的那一维被完全切掉。

输入放大 10 倍，输出并不是简单地也放大 10 倍，而是连输出的**形状**都变了。所以输入的大小会改变 MLP 实际做的计算。

## 3. 层与层之间：不断移动的目标

残差流不断累加，所以前面的层看到的输入往往比较小，后面的层看到的比较大。如果不做归一化：

- 每一层的权重都得针对自己的典型输入大小来调；
- 训练过程中前面的层一直在变，后面的层看到的输入大小也跟着一直变。每一层都在追一个**不断移动的目标**。

## 类比

> attention 和 MLP 就像一个戴着固定音量耳机的翻译。有的说话人小声嘀咕，有的大喊大叫。嘀咕太轻，听不清；喊叫太响，只有最响的那个词能听到。翻译在理解"说了*什么*"之前，得先应付"声音*多大*"。LayerNorm 就是装在耳机前面的**自动音量控制**：不管是嘀咕还是喊叫，传到耳机里都是同样的音量，翻译只需要专注于内容，也就是向量的方向和形状。

## 结论

> 有了 LayerNorm，不管是哪个 token、哪一层，每个子层的输入都是均值 0、标准差 1。固定的权重只需要学输入说的是**什么**，而不用应付输入有**多大**。
`,
    },
  },

  final_ln_notes: {
    en: {
      title: "Final LayerNorm (ln_f)",
      md: `
**After the 12 blocks and before \`lm_head\`, GPT-2 normalizes the residual stream once more:**

\`\`\`python
for blk in self.blocks:
    x = blk(x)
x = self.ln_f(x)          # ← this line
logits = self.lm_head(x)
\`\`\`

### Why it is needed

GPT-2 is **pre-LN**. \`ln1\` and \`ln2\` normalize only a **copy** of the stream that goes into attention or the MLP. The residual stream $x$ itself is never normalized: it just keeps receiving 24 updates (12 attention + 12 MLP). So by the end, the size of $x$ differs from token to token, and is usually large.

\`lm_head\` takes the dot product of $x$ with each word's vector to get the logits (in GPT-2 those vectors are the rows of the token embedding, see [weight tying](#info:weight_tying_notes)). A dot product grows in proportion to the size of $x$. So if $x$ went in directly, **the bigger $x$ is, the sharper the softmax**.

### Worked example

To keep it small: 4 dimensions instead of 768, and a vocabulary of 3 words.

\`\`\`text
word vectors E:  word 0 = [ 0.5, 0,   0.5, 0  ]
                 word 1 = [ 0,   0.5, 0,   0.5]
                 word 2 = [-0.5, 0,   0.5, 0  ]
\`\`\`

Two tokens whose $x$ points in the **same direction**, but token B is 8× larger (say, it collected bigger updates in the stream).

**Without \`ln_f\`:**

\`\`\`text
           x                   logits = x · E        softmax
token A:   [1, -0.5, 1.5, 0]   [1.25, -0.25, 0.25]   [0.63,   0.14, 0.23  ]
token B:   [8, -4,   12,  0]   [10,   -2,    2   ]   [0.9997, 0.00, 0.0003]
\`\`\`

Each logit is a dot product, e.g. for token A and word 0: 1×0.5 + (−0.5)×0 + 1.5×0.5 + 0×0 = 1.25. Token B's logits are exactly 8× token A's.

Same direction, same "meaning", yet B is almost 100% sure of word 0. Its confidence comes only from $x$ happening to be large.

**With \`ln_f\`** (γ = 1, β = 0, ε ignored), step by step for token A:

1. Mean: (1 − 0.5 + 1.5 + 0) / 4 = 0.5
2. Subtract the mean: [0.5, −1, 1, −0.5]
3. Variance: (0.25 + 1 + 1 + 0.25) / 4 = 0.625, so std = √0.625 ≈ 0.791
4. Divide by std: [0.632, −1.265, 1.265, −0.632]

For token B, the mean is 4 and the std is 8 × 0.791. Both scale by 8, so the 8 cancels and B gives **the same vector**:

\`\`\`text
ln_f(A) = ln_f(B) = [0.632, -1.265, 1.265, -0.632]
logits  = [0.949, -0.949, 0.316]   →   softmax = [0.59, 0.09, 0.32]
\`\`\`

Now the prediction depends only on the **direction** of $x$, not on how much the stream happened to accumulate. (Numbers checked in PyTorch.)

### What if the model wants to be more confident?

\`ln_f\` has its own learnable γ and β (768 each), the last of GPT-2's 25 LayerNorms (2 per block × 12, plus \`ln_f\`). During training γ learns a suitable overall scale, which sets how sharp the logits are. The difference: this scale is **learned and shared by all tokens**, instead of being set by how large each token's stream happened to get.

### Why the original Transformer has no \`ln_f\`

The original Transformer is post-LN: after every sublayer, $x$ itself is replaced by $\\text{LN}(x + \\text{sublayer}(x))$, so the stream leaving the last layer is already normalized. GPT-2 moved to pre-LN, so it has to add this one normalization at the end. The GPT-2 paper: "an additional layer normalization was added after the final self-attention block". See [LayerNorm](#info:layernorm) for pre-LN vs post-LN.

> 🆕 Modern models such as LLaMA keep this final norm but use RMSNorm instead of LayerNorm (no mean subtraction, no β).
`,
    },
    zh: {
      title: "最后的 LayerNorm（ln_f）",
      md: `
**12 个 block 跑完之后、送进 \`lm_head\` 之前，GPT-2 会再对残差流做一次归一化：**

\`\`\`python
for blk in self.blocks:
    x = blk(x)
x = self.ln_f(x)          # ← 这一行
logits = self.lm_head(x)
\`\`\`

### 为什么需要它

GPT-2 用的是 **pre-LN**。\`ln1\`、\`ln2\` 只归一化残差流的一个**副本**，这个副本送进 attention 或 MLP。残差流 $x$ 本身从来没被归一化过，只是一路被加上 24 个更新量（12 个 attention + 12 个 MLP）。所以到最后，$x$ 的大小因 token 而异，而且通常比较大。

\`lm_head\` 拿 $x$ 和词表里每个词的向量做点积，得到 logits（GPT-2 里这些向量就是 token embedding 的各行，见 [weight tying](#info:weight_tying_notes)）。点积和 $x$ 的大小成正比，所以如果直接把 $x$ 送进去，**$x$ 越大，softmax 就越尖**。

### 具体例子

为了简单：用 4 维代替 768 维，词表只有 3 个词。

\`\`\`text
word vectors E:  word 0 = [ 0.5, 0,   0.5, 0  ]
                 word 1 = [ 0,   0.5, 0,   0.5]
                 word 2 = [-0.5, 0,   0.5, 0  ]
\`\`\`

两个 token 的 $x$ **方向相同**，但 token B 大了 8 倍（比如它在残差流里被加了更大的更新量）。

**没有 \`ln_f\`：**

\`\`\`text
           x                   logits = x · E        softmax
token A:   [1, -0.5, 1.5, 0]   [1.25, -0.25, 0.25]   [0.63,   0.14, 0.23  ]
token B:   [8, -4,   12,  0]   [10,   -2,    2   ]   [0.9997, 0.00, 0.0003]
\`\`\`

每个 logit 都是一个点积，比如 token A 和词 0：1×0.5 + (−0.5)×0 + 1.5×0.5 + 0×0 = 1.25。token B 的 logits 正好是 token A 的 8 倍。

方向相同，"意思"一样，B 却几乎 100% 确定是词 0。它的把握只来自 $x$ 碰巧比较大。

**加上 \`ln_f\`**（γ = 1，β = 0，忽略 ε），以 token A 为例一步步算：

1. 均值：(1 − 0.5 + 1.5 + 0) / 4 = 0.5
2. 减去均值：[0.5, −1, 1, −0.5]
3. 方差：(0.25 + 1 + 1 + 0.25) / 4 = 0.625，所以标准差 = √0.625 ≈ 0.791
4. 除以标准差：[0.632, −1.265, 1.265, −0.632]

token B 的均值是 4，标准差是 8 × 0.791。分子分母都放大了 8 倍，8 被约掉，所以 B 得到**同一个向量**：

\`\`\`text
ln_f(A) = ln_f(B) = [0.632, -1.265, 1.265, -0.632]
logits  = [0.949, -0.949, 0.316]   →   softmax = [0.59, 0.09, 0.32]
\`\`\`

现在预测只取决于 $x$ 的**方向**，和残差流碰巧积累了多大无关。（数字已用 PyTorch 验证。）

### 那模型想"更有把握"怎么办

\`ln_f\` 有自己可学习的 γ 和 β（各 768 个），是 GPT-2 25 个 LayerNorm 里的最后一个（每个 block 2 个 × 12，再加 \`ln_f\`）。训练中 γ 会学出一个合适的整体尺度，决定 logits 有多尖。区别在于：这个尺度是**学出来的、所有 token 共用**，而不是由每个 token 的残差流碰巧有多大决定的。

### 为什么原始 Transformer 没有 \`ln_f\`

原始 Transformer 用 post-LN：每个子层之后，$x$ 本身会被替换成 $\\text{LN}(x + \\text{sublayer}(x))$，所以最后一层输出的残差流已经归一化过了。GPT-2 改成 pre-LN，就得在最后补这一次归一化。GPT-2 论文原话："an additional layer normalization was added after the final self-attention block"。pre-LN 和 post-LN 的对比见 [LayerNorm](#info:layernorm)。

> 🆕 LLaMA 等现代模型保留了这个最后的 norm，但把 LayerNorm 换成了 RMSNorm（不减均值，没有 β）。
`,
    },
  },

  stabilize_training_notes: {
    en: {
      title: "Stabilize Training Example",
      md: `
**A tiny numeric example of why LayerNorm stabilizes training by controlling magnitude, so attention logits don't blow up and softmax doesn't saturate.**

## Setup

Two tokens, $d = 3$. Use identity projections, so $Q = X$ and $K = X$, and scale scores by $1/\\sqrt{d}$.

Raw embeddings:

$$
e_1 = \\begin{bmatrix}10 & -5 & 7\\end{bmatrix}, \\quad e_2 = \\begin{bmatrix}-8 & 12 & -4\\end{bmatrix}
$$

## 1. Without LayerNorm

Scores for token 1 attending to $\\{1, 2\\}$:

$$
\\begin{aligned}
s_{11} &= \\frac{e_1 \\cdot e_1}{\\sqrt{3}} = \\frac{10 \\cdot 10 + (-5) \\cdot (-5) + 7 \\cdot 7}{\\sqrt{3}} = \\frac{100 + 25 + 49}{\\sqrt{3}} = \\frac{174}{\\sqrt{3}} \\approx 100.46 \\\\[4pt]
s_{12} &= \\frac{e_1 \\cdot e_2}{\\sqrt{3}} = \\frac{10 \\cdot (-8) + (-5) \\cdot 12 + 7 \\cdot (-4)}{\\sqrt{3}} = \\frac{-80 - 60 - 28}{\\sqrt{3}} = \\frac{-168}{\\sqrt{3}} \\approx -96.99
\\end{aligned}
$$

Softmax over $[100.46, -96.99]$ $\\Rightarrow \\approx [1.00, 0.00]$: **totally saturated**, which leads to vanishing gradients.

[Why? A step-by-step explanation](#info:stabilize_training_no_layernorm_backprop_notes)

## 2. With LayerNorm

Normalize each token (numbers rounded):

- For $e_1$: mean $= 4$, std $\\approx 6.4807$, so $\\hat{e}_1 = \\begin{bmatrix}0.9258, -1.3887, 0.4629\\end{bmatrix}$.
- For $e_2$: mean $= 0$, std $\\approx 8.6410$, so $\\hat{e}_2 = \\begin{bmatrix}-0.9258, 1.3887, -0.4629\\end{bmatrix}$.

Scores for token 1 attending to $\\{1, 2\\}$:

$$
\\begin{aligned}
s_{11} &= \\frac{\\hat{e}_1 \\cdot \\hat{e}_1}{\\sqrt{3}} = \\frac{0.9258 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.4629 \\cdot 0.4629}{\\sqrt{3}} \\approx 1.732 \\\\[4pt]
s_{12} &= \\frac{\\hat{e}_1 \\cdot \\hat{e}_2}{\\sqrt{3}} = \\frac{0.9258 \\cdot (-0.9258) + (-1.3887) \\cdot 1.3887 + 0.4629 \\cdot (-0.4629)}{\\sqrt{3}} \\approx -1.732
\\end{aligned}
$$

Softmax over $[1.732, -1.732]$ $\\Rightarrow \\approx [0.9696, 0.0304]$: **not saturated**, so it doesn't lead to vanishing gradients.

## Bottom line

> LN standardizes activations, so during the forward pass downstream dot products/logits stay in a reasonable numeric range → softmax avoids saturation → probabilities remain sensitive to changes.
`,
    },
    zh: {
      title: "稳定训练的例子",
      md: `
**一个很小的数值例子：LayerNorm 通过控制大小来稳定训练，让 attention 的 logits 不会过大，softmax 不会饱和。**

## 设定

两个 token，$d = 3$。投影用恒等映射，所以 $Q = X$、$K = X$，分数乘以 $1/\\sqrt{d}$。

原始 embedding：

$$
e_1 = \\begin{bmatrix}10 & -5 & 7\\end{bmatrix}, \\quad e_2 = \\begin{bmatrix}-8 & 12 & -4\\end{bmatrix}
$$

## 1. 没有 LayerNorm

token 1 对 $\\{1, 2\\}$ 的注意力分数：

$$
\\begin{aligned}
s_{11} &= \\frac{e_1 \\cdot e_1}{\\sqrt{3}} = \\frac{10 \\cdot 10 + (-5) \\cdot (-5) + 7 \\cdot 7}{\\sqrt{3}} = \\frac{100 + 25 + 49}{\\sqrt{3}} = \\frac{174}{\\sqrt{3}} \\approx 100.46 \\\\[4pt]
s_{12} &= \\frac{e_1 \\cdot e_2}{\\sqrt{3}} = \\frac{10 \\cdot (-8) + (-5) \\cdot 12 + 7 \\cdot (-4)}{\\sqrt{3}} = \\frac{-80 - 60 - 28}{\\sqrt{3}} = \\frac{-168}{\\sqrt{3}} \\approx -96.99
\\end{aligned}
$$

对 $[100.46, -96.99]$ 做 softmax $\\Rightarrow \\approx [1.00, 0.00]$：**完全饱和**，会导致梯度消失。

[为什么？一步一步的解释](#info:stabilize_training_no_layernorm_backprop_notes)

## 2. 有 LayerNorm

对每个 token 做归一化（数字已四舍五入）：

- $e_1$：均值 $= 4$，标准差 $\\approx 6.4807$，所以 $\\hat{e}_1 = \\begin{bmatrix}0.9258, -1.3887, 0.4629\\end{bmatrix}$。
- $e_2$：均值 $= 0$，标准差 $\\approx 8.6410$，所以 $\\hat{e}_2 = \\begin{bmatrix}-0.9258, 1.3887, -0.4629\\end{bmatrix}$。

token 1 对 $\\{1, 2\\}$ 的注意力分数：

$$
\\begin{aligned}
s_{11} &= \\frac{\\hat{e}_1 \\cdot \\hat{e}_1}{\\sqrt{3}} = \\frac{0.9258 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.4629 \\cdot 0.4629}{\\sqrt{3}} \\approx 1.732 \\\\[4pt]
s_{12} &= \\frac{\\hat{e}_1 \\cdot \\hat{e}_2}{\\sqrt{3}} = \\frac{0.9258 \\cdot (-0.9258) + (-1.3887) \\cdot 1.3887 + 0.4629 \\cdot (-0.4629)}{\\sqrt{3}} \\approx -1.732
\\end{aligned}
$$

对 $[1.732, -1.732]$ 做 softmax $\\Rightarrow \\approx [0.9696, 0.0304]$：**没有饱和**，不会导致梯度消失。

## 结论

> LN 把激活值标准化，所以 forward 时后面的点积和 logits 都保持在合理的数值范围内 → softmax 不会饱和 → 概率对输入的变化仍然敏感。
`,
    },
  },

  stabilize_training_no_layernorm_backprop_notes: {
    en: {
      title: "Why saturated softmax leads to vanishing gradients",
      md: `
**Why a saturated softmax (the no-LayerNorm case) starves earlier layers of gradient, step by step.**

## 1. Softmax from logits to probabilities

The logits from the example: $z = [100.46, -96.99]$.

After softmax: $p \\approx [1.00, 0.00]$, up to an $\\epsilon$ of about $10^{-86}$.

## 2. Cross-entropy loss and its gradient w.r.t. the logits

For a one-hot target $y$ (say the true class is 1: $y = [1, 0]$), the cross-entropy loss is

$$
L = -\\sum_i y_i \\log p_i = -\\log p_1
$$

The gradient of the loss w.r.t. the logits is

$$
\\frac{\\partial L}{\\partial z_i} = p_i - y_i
$$

Read the mathematical proof [here](#info:gradient_loss_to_logits_proof_notes).

Plugging in the values:

$$
\\begin{aligned}
p_1 &\\approx 1 - 10^{-86}, \\; y_1 = 1 &&\\Rightarrow \\frac{\\partial L}{\\partial z_1} \\approx (1 - 10^{-86}) - 1 = -10^{-86} \\\\[4pt]
p_2 &\\approx 10^{-86}, \\; y_2 = 0 &&\\Rightarrow \\frac{\\partial L}{\\partial z_2} \\approx 10^{-86} - 0 = 10^{-86}
\\end{aligned}
$$

$$
\\Rightarrow \\nabla_z L = [-10^{-86}, 10^{-86}]
$$

This is **essentially zero in magnitude**.

> Intuition: when softmax is saturated and confidently correct, the loss surface is almost flat, so the gradient is near 0.

## 3. Backprop to earlier layers becomes tiny

Suppose the logits came from a linear layer $z = Wh + b$, where $h$ is the previous layer's activations. Then

$$
\\frac{\\partial L}{\\partial W} = (\\nabla_z L)\\,h^\\top, \\qquad
\\frac{\\partial L}{\\partial h} = W^\\top (\\nabla_z L), \\qquad
\\frac{\\partial L}{\\partial b} = \\nabla_z L
$$

Because $\\nabla_z L$ has entries on the order of $10^{-86}$, all the weight gradients and the gradient flowing back into $h$ become $10^{-86}$ in magnitude. That means earlier layers receive essentially no learning signal.

## 4. Some thoughts

- Early in training, we don't want logits so huge that softmax saturates, because then we stop getting useful gradients to shape the representations.
- If the model is **confident but wrong** (e.g. $p \\approx [0, 1]$ while $y = [1, 0]$), then $\\nabla_z L \\approx [-1, 1]$: the network does get a strong corrective signal. Vanishing happens in the "confidently correct" case.
- [Label smoothing](#info:label_smoothing_notes) keeps gradients from collapsing completely even when predictions are confident.
`,
    },
    zh: {
      title: "为什么饱和的 softmax 会导致梯度消失",
      md: `
**一步一步说明：softmax 饱和时（也就是没有 LayerNorm 的情况），前面的层为什么几乎收不到梯度。**

## 1. softmax：从 logits 到概率

例子里的 logits：$z = [100.46, -96.99]$。

softmax 之后：$p \\approx [1.00, 0.00]$，误差 $\\epsilon$ 大约是 $10^{-86}$。

## 2. Cross-entropy loss 和它对 logits 的梯度

对于 one-hot 目标 $y$（假设正确类别是 1：$y = [1, 0]$），cross-entropy loss 是

$$
L = -\\sum_i y_i \\log p_i = -\\log p_1
$$

loss 对 logits 的梯度是

$$
\\frac{\\partial L}{\\partial z_i} = p_i - y_i
$$

数学证明见[这里](#info:gradient_loss_to_logits_proof_notes)。

代入数值：

$$
\\begin{aligned}
p_1 &\\approx 1 - 10^{-86}, \\; y_1 = 1 &&\\Rightarrow \\frac{\\partial L}{\\partial z_1} \\approx (1 - 10^{-86}) - 1 = -10^{-86} \\\\[4pt]
p_2 &\\approx 10^{-86}, \\; y_2 = 0 &&\\Rightarrow \\frac{\\partial L}{\\partial z_2} \\approx 10^{-86} - 0 = 10^{-86}
\\end{aligned}
$$

$$
\\Rightarrow \\nabla_z L = [-10^{-86}, 10^{-86}]
$$

它的大小**基本上就是 0**。

> 直觉：softmax 饱和、而且很有把握地预测对了时，loss 的曲面几乎是平的，所以梯度接近 0。

## 3. 传回前面层的梯度变得极小

假设 logits 来自一个线性层 $z = Wh + b$，$h$ 是前一层的激活值。那么

$$
\\frac{\\partial L}{\\partial W} = (\\nabla_z L)\\,h^\\top, \\qquad
\\frac{\\partial L}{\\partial h} = W^\\top (\\nabla_z L), \\qquad
\\frac{\\partial L}{\\partial b} = \\nabla_z L
$$

因为 $\\nabla_z L$ 的每个分量都在 $10^{-86}$ 这个量级，所有权重的梯度、以及流回 $h$ 的梯度，大小也都是 $10^{-86}$ 量级。这意味着前面的层基本收不到任何学习信号。

## 4. 几点思考

- 训练早期，我们不希望 logits 大到让 softmax 饱和，否则就得不到有用的梯度来塑造表示。
- 如果模型**很有把握但预测错了**（比如 $p \\approx [0, 1]$ 而 $y = [1, 0]$），那么 $\\nabla_z L \\approx [-1, 1]$：网络会收到很强的纠正信号。梯度消失发生在"很有把握且预测对了"的情况。
- [Label smoothing](#info:label_smoothing_notes) 能让梯度在预测很有把握时也不至于完全消失。
`,
    },
  },

  improves_gradient_flow_notes: {
    en: {
      title: "Improves gradient flow",
      md: `
**A tiny numeric example comparing the gradient w.r.t. the pre-LN activations with and without LN.**

## Setup

A single token vector of size $n = 3$:

$$
x = \\begin{bmatrix}10, -5, 7\\end{bmatrix}
$$

Mean and standard deviation:

$$
\\mu = \\frac{10 + (-5) + 7}{3} = 4, \\quad \\sigma = \\sqrt{\\frac{(10-4)^2 + (-5-4)^2 + (7-4)^2}{3}} = 6.4807
$$

To keep the example simple, let a simple linear head follow, with $W = \\mathrm{diag}(3, 1, 0.5)$ and $b = 0$, and use a squared loss to a zero target:

$$
L = \\frac{1}{2} \\|Wy + b - 0\\|^2
$$

where $y$ is either the identity output (without LN) or the LN output.

## 1. Without LN

There is no LN, so it's an identity pass: $y = x$.

- **Forward:** $Wy = \\mathrm{diag}(3, 1, 0.5) \\cdot x = \\begin{bmatrix}30, -5, 3.5\\end{bmatrix}$
- **Gradient w.r.t. $y$:** $\\nabla_y L = W^\\top (Wy) = W^2 y$
- Since $W$ is diagonal, $W^2 = \\mathrm{diag}(9, 1, 0.25)$.
- **Gradient w.r.t. the input $x$** (here $y = x$): $\\nabla_x L = W^2 x = \\begin{bmatrix}9 \\cdot 10, 1 \\cdot (-5), 0.25 \\cdot 7\\end{bmatrix} = \\begin{bmatrix}90, -5, 1.75\\end{bmatrix}$

This gradient is large and very uneven, dominated by the first coordinate. If upstream layers rescale $x$ by some factor $c$, the gradient scales by $c$ as well (the learning rate becomes sensitive to scale).

## 2. With LN (pre-LN, GPT-2 style)

First compute the normalized vector:

$$
\\hat{y} = \\mathrm{LN}(x) = \\frac{x - \\mu}{\\sigma} \\approx \\begin{bmatrix}0.9258, -1.3887, 0.4629\\end{bmatrix}
$$

Forward:

$$
W\\hat{y} = \\mathrm{diag}(3, 1, 0.5)\\,\\hat{y}
$$

Gradient w.r.t. $\\hat{y}$:

$$
g = \\nabla_{\\hat{y}} L = W^\\top (W\\hat{y}) = W^2\\hat{y} = \\mathrm{diag}(9, 1, 0.25)\\,\\hat{y} = \\begin{bmatrix}8.3324, -1.3887, 0.1157\\end{bmatrix}
$$

Now backprop through the LN. For LN over a vector (affine removed for simplicity), the closed form is

$$
\\nabla_{x} L = \\frac{1}{\\sigma} \\left(g - \\mu(g)\\mathbf{1} - \\hat{y}\\,\\mu(g \\odot \\hat{y})\\right)
$$

where $\\mathbf{1}$ is a vector of all ones, $\\mu(g)$ is the mean of $g$, and $\\odot$ is the elementwise product.

Compute the two mean terms:

$$
\\begin{aligned}
\\mu(g) &= \\frac{8.3324 + (-1.3887) + 0.1157}{3} \\approx 2.3531 \\\\[4pt]
\\mu(g \\odot \\hat{y}) &= \\frac{8.3324 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.1157 \\cdot 0.4629}{3} = 3.2321
\\end{aligned}
$$

Assemble:

$$
\\begin{aligned}
g - \\mu(g)\\mathbf{1} &\\approx \\begin{bmatrix}5.9793, -3.7419, -2.2374\\end{bmatrix} \\\\[4pt]
\\hat{y}\\,\\mu(g \\odot \\hat{y}) &\\approx \\begin{bmatrix}2.9924, -4.4886, 1.4962\\end{bmatrix}
\\end{aligned}
$$

Subtract and scale by $\\frac{1}{\\sigma}$:

$$
\\nabla_{x} L = \\frac{1}{6.4807} \\begin{bmatrix}2.9869, 0.7467, -3.7336\\end{bmatrix} \\approx \\begin{bmatrix}0.461, 0.115, -0.576\\end{bmatrix}
$$

> Now the gradient is much more even: no single dimension dominates.
`,
    },
    zh: {
      title: "改善梯度流动",
      md: `
**一个很小的数值例子：比较有 LN 和没有 LN 时，loss 对 LN 之前激活值的梯度。**

## 设定

一个 token 向量，大小 $n = 3$：

$$
x = \\begin{bmatrix}10, -5, 7\\end{bmatrix}
$$

均值和标准差：

$$
\\mu = \\frac{10 + (-5) + 7}{3} = 4, \\quad \\sigma = \\sqrt{\\frac{(10-4)^2 + (-5-4)^2 + (7-4)^2}{3}} = 6.4807
$$

为了让例子简单，后面接一个简单的线性层，$W = \\mathrm{diag}(3, 1, 0.5)$，$b = 0$，loss 用到目标 0 的平方误差：

$$
L = \\frac{1}{2} \\|Wy + b - 0\\|^2
$$

其中 $y$ 是恒等输出（没有 LN 时）或 LN 的输出。

## 1. 没有 LN

没有 LN，相当于恒等传递：$y = x$。

- **Forward：** $Wy = \\mathrm{diag}(3, 1, 0.5) \\cdot x = \\begin{bmatrix}30, -5, 3.5\\end{bmatrix}$
- **对 $y$ 的梯度：** $\\nabla_y L = W^\\top (Wy) = W^2 y$
- 因为 $W$ 是对角矩阵，$W^2 = \\mathrm{diag}(9, 1, 0.25)$。
- **对输入 $x$ 的梯度**（这里 $y = x$）：$\\nabla_x L = W^2 x = \\begin{bmatrix}9 \\cdot 10, 1 \\cdot (-5), 0.25 \\cdot 7\\end{bmatrix} = \\begin{bmatrix}90, -5, 1.75\\end{bmatrix}$

这个梯度很大，而且非常不均匀，被第一个分量主导。如果上游的层把 $x$ 放大 $c$ 倍，梯度也会跟着放大 $c$ 倍（学习率就会对尺度很敏感）。

## 2. 有 LN（pre-LN，GPT-2 的方式）

先算归一化后的向量：

$$
\\hat{y} = \\mathrm{LN}(x) = \\frac{x - \\mu}{\\sigma} \\approx \\begin{bmatrix}0.9258, -1.3887, 0.4629\\end{bmatrix}
$$

Forward：

$$
W\\hat{y} = \\mathrm{diag}(3, 1, 0.5)\\,\\hat{y}
$$

对 $\\hat{y}$ 的梯度：

$$
g = \\nabla_{\\hat{y}} L = W^\\top (W\\hat{y}) = W^2\\hat{y} = \\mathrm{diag}(9, 1, 0.25)\\,\\hat{y} = \\begin{bmatrix}8.3324, -1.3887, 0.1157\\end{bmatrix}
$$

接着 backprop 穿过 LN。对一个向量做 LN（为了简单去掉了仿射部分），闭式解是

$$
\\nabla_{x} L = \\frac{1}{\\sigma} \\left(g - \\mu(g)\\mathbf{1} - \\hat{y}\\,\\mu(g \\odot \\hat{y})\\right)
$$

其中 $\\mathbf{1}$ 是全 1 向量，$\\mu(g)$ 是 $g$ 的均值，$\\odot$ 是逐元素相乘。

先算两个均值项：

$$
\\begin{aligned}
\\mu(g) &= \\frac{8.3324 + (-1.3887) + 0.1157}{3} \\approx 2.3531 \\\\[4pt]
\\mu(g \\odot \\hat{y}) &= \\frac{8.3324 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.1157 \\cdot 0.4629}{3} = 3.2321
\\end{aligned}
$$

组合起来：

$$
\\begin{aligned}
g - \\mu(g)\\mathbf{1} &\\approx \\begin{bmatrix}5.9793, -3.7419, -2.2374\\end{bmatrix} \\\\[4pt]
\\hat{y}\\,\\mu(g \\odot \\hat{y}) &\\approx \\begin{bmatrix}2.9924, -4.4886, 1.4962\\end{bmatrix}
\\end{aligned}
$$

相减，再乘以 $\\frac{1}{\\sigma}$：

$$
\\nabla_{x} L = \\frac{1}{6.4807} \\begin{bmatrix}2.9869, 0.7467, -3.7336\\end{bmatrix} \\approx \\begin{bmatrix}0.461, 0.115, -0.576\\end{bmatrix}
$$

> 现在梯度均匀多了，没有哪一个维度占主导。
`,
    },
  },

  post_norm_gradient_example_notes: {
    en: {
      title: "Post-norm gradient example",
      md: `
**A single residual block in post-norm form, to illustrate the gradient issue with post-norm.**

$$
\\text{(post-norm)} \\quad z = x + f(x), \\quad y = \\mathrm{LN}(z)
$$

## Setup

For simplicity, assume:

- Hidden size $m = 2$, so LN normalizes across 2 features.
- Input $x = [1, -1]$ (mean $= 0$, variance $= 1$).
- A simple residual branch $f(x) = ax$ with $a = 2$ (amplifying the residual path by a factor of 2).

## Forward pass

- Residual add: $z = x + f(x) = x + 2x = 3x = [3, -3]$
- LayerNorm over features:
  - mean $\\mu = (3 + (-3)) / 2 = 0$
  - variance $v = ((3 - 0)^2 + (-3 - 0)^2) / 2 = 9$, std $\\sigma = 3$
  - $y = \\mathrm{LN}(z) = (z - \\mu) / \\sigma = [3, -3]/3 = [1, -1]$

The forward pass is fine.

## Backward pass (the problem!)

For LayerNorm (per-feature LN with $m$ features), the Jacobian at this point has the following closed form ([proof](#info:ln_jacobian_closed_form_proof_notes)):

$$
J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
$$

where $\\mathbf{I}$ is the $m \\times m$ identity matrix and $\\mathbf{1}$ is the all-ones vector in $\\mathbb{R}^m$.

Plug in the values $m = 2$, $\\sigma = 3$, $y = [1, -1]$:

$$
\\begin{aligned}
\\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top &= \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top = \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix} \\\\[4pt]
\\frac{1}{m} yy^\\top &= \\frac{1}{2} yy^\\top = \\frac{1}{2} \\begin{bmatrix}1 & -1\\\\-1 & 1\\end{bmatrix} = \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix} \\\\[4pt]
\\mathbf{I} - \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{2} yy^\\top &= \\mathbf{I} - \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix} - \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix} = \\begin{bmatrix}0 & 0\\\\0 & 0\\end{bmatrix}
\\end{aligned}
$$

Thus $J = \\frac{1}{\\sigma} \\times 0 = 0$.

## Interpretation

At this configuration, LayerNorm's Jacobian is exactly zero. That means for any upstream gradient $\\nabla_y L$, the gradient propagated back to $z$ is

$$
\\nabla_z L = J^\\top \\nabla_y L = 0
$$

So the gradient dies completely before it even reaches the residual add.

Also note the $\\frac{1}{\\sigma}$ in the Jacobian: with more layers, many such Jacobians chain up, so even when they are not exactly zero, the repeated products of $\\frac{1}{\\sigma}$ can still shrink the gradient to near zero.

> **Caveat about this toy setup:** with $m = 2$ the LayerNorm output is always $\\pm[1, -1]$ (constant up to sign), so its Jacobian is zero for *every* input, not only here. The exact zero is an artifact of choosing $m = 2$. At realistic widths the Jacobian is not zero; the real issue is that the $\\frac{1}{\\sigma}$ factor and the projection are applied to the whole stream at every layer, so their product across depth can shrink (or grow) the gradient.

## Contrast with pre-norm

In pre-norm, you'd compute $y = x + f(\\mathrm{LN}(x))$. The sublayer always sees unit-scale inputs, and the LN Jacobian is applied before the residual add. So there's always a "gradient highway": even if the residual branch's Jacobian goes to zero, the highway still carries gradient through.
`,
    },
    zh: {
      title: "Post-norm 梯度的例子",
      md: `
**用一个 post-norm 形式的残差块，说明 post-norm 的梯度问题。**

$$
\\text{(post-norm)} \\quad z = x + f(x), \\quad y = \\mathrm{LN}(z)
$$

## 设定

为了简单，假设：

- 隐藏维度 $m = 2$，所以 LN 在 2 个特征上做归一化。
- 输入 $x = [1, -1]$（均值 $= 0$，方差 $= 1$）。
- 残差分支很简单：$f(x) = ax$，$a = 2$（把残差分支放大 2 倍）。

## Forward

- 残差相加：$z = x + f(x) = x + 2x = 3x = [3, -3]$
- 在特征维度上做 LayerNorm：
  - 均值 $\\mu = (3 + (-3)) / 2 = 0$
  - 方差 $v = ((3 - 0)^2 + (-3 - 0)^2) / 2 = 9$，标准差 $\\sigma = 3$
  - $y = \\mathrm{LN}(z) = (z - \\mu) / \\sigma = [3, -3]/3 = [1, -1]$

forward 没有问题。

## Backward（问题在这里！）

对 LayerNorm（在 $m$ 个特征上做 LN），这个点上的 Jacobian 有下面的闭式解（[证明](#info:ln_jacobian_closed_form_proof_notes)）：

$$
J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
$$

其中 $\\mathbf{I}$ 是 $m \\times m$ 单位矩阵，$\\mathbf{1}$ 是 $\\mathbb{R}^m$ 里的全 1 向量。

代入 $m = 2$、$\\sigma = 3$、$y = [1, -1]$：

$$
\\begin{aligned}
\\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top &= \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top = \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix} \\\\[4pt]
\\frac{1}{m} yy^\\top &= \\frac{1}{2} yy^\\top = \\frac{1}{2} \\begin{bmatrix}1 & -1\\\\-1 & 1\\end{bmatrix} = \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix} \\\\[4pt]
\\mathbf{I} - \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{2} yy^\\top &= \\mathbf{I} - \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix} - \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix} = \\begin{bmatrix}0 & 0\\\\0 & 0\\end{bmatrix}
\\end{aligned}
$$

所以 $J = \\frac{1}{\\sigma} \\times 0 = 0$。

## 解读

在这个配置下，LayerNorm 的 Jacobian 正好是 0。这意味着不管上游传来的梯度 $\\nabla_y L$ 是什么，传回 $z$ 的梯度都是

$$
\\nabla_z L = J^\\top \\nabla_y L = 0
$$

所以梯度还没到残差相加那里就已经完全消失了。

还要注意 Jacobian 里的 $\\frac{1}{\\sigma}$：层数一多，这样的 Jacobian 会一个接一个连乘，所以即使它们不正好是 0，反复乘 $\\frac{1}{\\sigma}$ 仍然可能把梯度压到接近 0。

> **关于这个玩具设定的说明：** 当 $m = 2$ 时，LayerNorm 的输出永远是 $\\pm[1, -1]$（只差一个符号），所以它的 Jacobian 对*任何*输入都是 0，不只是这里。正好等于 0 是选了 $m = 2$ 造成的。在实际的宽度下 Jacobian 不是 0；真正的问题是，每一层都要对整个残差流施加 $\\frac{1}{\\sigma}$ 因子和投影，它们在深度上的乘积可能把梯度缩小（或放大）。

## 和 pre-norm 对比

在 pre-norm 里，计算的是 $y = x + f(\\mathrm{LN}(x))$。子层看到的输入总是单位尺度的，而 LN 的 Jacobian 只作用在残差相加之前的分支上。所以始终有一条"梯度高速公路"：即使残差分支的 Jacobian 变成 0，高速公路仍然能把梯度传过去。
`,
    },
  },

  post_norm_scaling_example_notes: {
    en: {
      title: "Post-norm scaling example",
      md: `
**In this practice, we'll first see an example where, under post-norm without scaling, the gradient vanishes with depth. Then we'll add scaling and compare.**

## Setup (post-norm)

To keep the example simple, we'll use a 2-D LayerNorm and a super simple residual branch that simply **adds a fixed vector $u$** at each layer, so we can see the growth cleanly.

For each layer $l$:

$$
z_l = x_{l-1} + \\alpha u
$$

where

- Feature dimension $m = 2$.
- Start with $x_0 = [0, 0]$.
- Fixed residual vector $u = [1, 0]$: a fixed unit step on the first feature, just to keep the arithmetic simple.
- LayerNorm uses $\\gamma = 1$, $\\beta = 0$: no shift/scale.
- Let $\\sigma(z) = \\sqrt{\\frac{(z_1 - \\mu)^2 + (z_2 - \\mu)^2}{2}}$ with $\\mu = \\frac{z_1 + z_2}{2}$: the standard deviation of $z$, as LayerNorm computes it.
- An informative assumption: the per-layer gradient attenuation through LN is $\\approx 1/\\sigma(z_l)$ (LN's Jacobian operator norm scales up to $\\approx 1/\\sigma(z_l)$). [Why](#info:ln_gradient_attenuation_upper_bound_proof_notes) This is an upper bound: with $m = 2$ the actual LN Jacobian is exactly zero (see the [post-norm gradient example](#info:post_norm_gradient_example_notes)), so the products below track the bound, not the true gradient.

With this setup, we'll compare **no scaling** ($\\alpha = 1$) vs **depth scaling for $N = 4$ layers** ($\\alpha = 1/\\sqrt{N} = 0.5$).

## Case 1: no scaling, $\\alpha = 1$, $N = 4$

**Layer 1**

$$
\\begin{aligned}
z_1 &= x_0 + \\alpha u = [0, 0] + 1 \\times [1, 0] = [1, 0] \\\\[4pt]
\\mu_1 &= (1 + 0)/2 = 0.5 \\\\[4pt]
\\sigma_1 &= \\sqrt{\\left((1-0.5)^2 + (0-0.5)^2\\right)/2} = 0.5 \\\\[4pt]
x_1 &= (z_1 - \\mu_1)/\\sigma_1 = [1, -1]
\\end{aligned}
$$

**Layer 2**

$$
\\begin{aligned}
z_2 &= x_1 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_2 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_2 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_2 &= (z_2 - \\mu_2)/\\sigma_2 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

**Layer 3**

$$
\\begin{aligned}
z_3 &= x_2 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_3 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_3 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_3 &= (z_3 - \\mu_3)/\\sigma_3 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

**Layer 4**

$$
\\begin{aligned}
z_4 &= x_3 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_4 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_4 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_4 &= (z_4 - \\mu_4)/\\sigma_4 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

Accumulate the per-layer factors:

$$
\\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\approx 0.5926
$$

Gradients get shrunk by about 40% by the four LNs, and would keep shrinking with more layers.

## Case 2: depth scaling, $\\alpha = 0.5$, $N = 4$

**Layer 1**

$$
\\begin{aligned}
z_1 &= x_0 + \\alpha u = [0, 0] + 0.5 \\times [1, 0] = [0.5, 0] \\\\[4pt]
\\mu_1 &= (0.5 + 0)/2 = 0.25 \\\\[4pt]
\\sigma_1 &= \\sqrt{\\left((0.5-0.25)^2 + (0-0.25)^2\\right)/2} = 0.25 \\\\[4pt]
x_1 &= (z_1 - \\mu_1)/\\sigma_1 = [0.25, -0.25]/0.25 = [1, -1]
\\end{aligned}
$$

**Layer 2**

$$
\\begin{aligned}
z_2 &= x_1 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_2 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_2 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_2 &= (z_2 - \\mu_2)/\\sigma_2 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

**Layer 3**

$$
\\begin{aligned}
z_3 &= x_2 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_3 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_3 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_3 &= (z_3 - \\mu_3)/\\sigma_3 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

**Layer 4**

$$
\\begin{aligned}
z_4 &= x_3 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_4 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_4 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_4 &= (z_4 - \\mu_4)/\\sigma_4 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

Accumulate the per-layer factors:

$$
\\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\approx 2.05
$$

> With scaling, $\\sigma_l$ stays smaller (1.25 instead of 1.5), so the product of bounds is larger (2.05 vs 0.59). The per-layer factor $1/1.25 = 0.8$ is still below 1, though, so with more layers the product would still shrink, just more slowly; the 2.05 comes mostly from layer 1 ($1/0.25 = 4$).
`,
    },
    zh: {
      title: "Post-norm 缩放示例",
      md: `
**这个练习里，我们先看一个例子：在不做缩放的 post-norm 下，梯度随深度消失。然后加上缩放再做对比。**

## 设定（post-norm）

为了让例子简单，我们用一个 2 维的 LayerNorm，以及一个极简的残差分支：它在每一层只是**加上一个固定向量 $u$**，这样能清楚地看到增长。

对每一层 $l$：

$$
z_l = x_{l-1} + \\alpha u
$$

其中

- 特征维度 $m = 2$。
- 从 $x_0 = [0, 0]$ 开始。
- 固定残差向量 $u = [1, 0]$：在第一个特征上走固定的一个单位步长，只是为了让算术简单。
- LayerNorm 用 $\\gamma = 1$、$\\beta = 0$：不做平移/缩放。
- 令 $\\sigma(z) = \\sqrt{\\frac{(z_1 - \\mu)^2 + (z_2 - \\mu)^2}{2}}$，其中 $\\mu = \\frac{z_1 + z_2}{2}$：即 LayerNorm 计算的 $z$ 的标准差。
- 一个有启发性的假设：每一层经过 LN 时梯度的衰减大约是 $\\approx 1/\\sigma(z_l)$（LN 的 Jacobian 算子范数最多约为 $\\approx 1/\\sigma(z_l)$）。[原因](#info:ln_gradient_attenuation_upper_bound_proof_notes) 这是一个上界：当 $m = 2$ 时，实际的 LN Jacobian 恰好为零（见 [post-norm 梯度的例子](#info:post_norm_gradient_example_notes)），所以下面的乘积跟踪的是这个上界，而不是真实的梯度。

在这个设定下，我们比较**不缩放**（$\\alpha = 1$）和**针对 $N = 4$ 层的深度缩放**（$\\alpha = 1/\\sqrt{N} = 0.5$）。

## 情况 1：不缩放，$\\alpha = 1$，$N = 4$

**第 1 层**

$$
\\begin{aligned}
z_1 &= x_0 + \\alpha u = [0, 0] + 1 \\times [1, 0] = [1, 0] \\\\[4pt]
\\mu_1 &= (1 + 0)/2 = 0.5 \\\\[4pt]
\\sigma_1 &= \\sqrt{\\left((1-0.5)^2 + (0-0.5)^2\\right)/2} = 0.5 \\\\[4pt]
x_1 &= (z_1 - \\mu_1)/\\sigma_1 = [1, -1]
\\end{aligned}
$$

**第 2 层**

$$
\\begin{aligned}
z_2 &= x_1 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_2 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_2 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_2 &= (z_2 - \\mu_2)/\\sigma_2 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

**第 3 层**

$$
\\begin{aligned}
z_3 &= x_2 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_3 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_3 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_3 &= (z_3 - \\mu_3)/\\sigma_3 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

**第 4 层**

$$
\\begin{aligned}
z_4 &= x_3 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1] \\\\[4pt]
\\mu_4 &= (2 + (-1))/2 = 0.5 \\\\[4pt]
\\sigma_4 &= \\sqrt{\\left((2-0.5)^2 + (-1-0.5)^2\\right)/2} = 1.5 \\\\[4pt]
x_4 &= (z_4 - \\mu_4)/\\sigma_4 = [1.5, -1.5]/1.5 = [1, -1]
\\end{aligned}
$$

把每层的因子乘起来：

$$
\\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\approx 0.5926
$$

经过四个 LN，梯度被缩小了约 40%，层数更多的话还会继续缩小。

## 情况 2：深度缩放，$\\alpha = 0.5$，$N = 4$

**第 1 层**

$$
\\begin{aligned}
z_1 &= x_0 + \\alpha u = [0, 0] + 0.5 \\times [1, 0] = [0.5, 0] \\\\[4pt]
\\mu_1 &= (0.5 + 0)/2 = 0.25 \\\\[4pt]
\\sigma_1 &= \\sqrt{\\left((0.5-0.25)^2 + (0-0.25)^2\\right)/2} = 0.25 \\\\[4pt]
x_1 &= (z_1 - \\mu_1)/\\sigma_1 = [0.25, -0.25]/0.25 = [1, -1]
\\end{aligned}
$$

**第 2 层**

$$
\\begin{aligned}
z_2 &= x_1 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_2 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_2 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_2 &= (z_2 - \\mu_2)/\\sigma_2 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

**第 3 层**

$$
\\begin{aligned}
z_3 &= x_2 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_3 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_3 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_3 &= (z_3 - \\mu_3)/\\sigma_3 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

**第 4 层**

$$
\\begin{aligned}
z_4 &= x_3 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1] \\\\[4pt]
\\mu_4 &= (1.5 + (-1))/2 = 0.25 \\\\[4pt]
\\sigma_4 &= \\sqrt{\\left((1.5-0.25)^2 + (-1-0.25)^2\\right)/2} = 1.25 \\\\[4pt]
x_4 &= (z_4 - \\mu_4)/\\sigma_4 = [1.25, -1.25]/1.25 = [1, -1]
\\end{aligned}
$$

把每层的因子乘起来：

$$
\\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\approx 2.05
$$

> 加了缩放后，$\\sigma_l$ 保持得更小（1.25 而不是 1.5），所以上界的乘积更大（2.05 对 0.59）。不过每层的因子 $1/1.25 = 0.8$ 仍然小于 1，所以层数更多时乘积还是会缩小，只是缩小得更慢；2.05 主要来自第 1 层（$1/0.25 = 4$）。
`,
    },
  },

  rmsnorm_gradient_example_notes: {
    en: {
      title: "RMSNorm gradient example",
      md: `
**A tiny numeric example to illustrate how RMSNorm works.**

## The formula

$$
\\mathrm{RMSNorm}(x) = \\gamma \\cdot \\frac{x}{\\sqrt{\\frac{1}{d} \\sum_{i}x_i^2 + \\epsilon}}
$$

- $\\gamma$ is a learnable scaling factor; we'll talk about what it's used for later.
- $\\epsilon$ is a small constant to avoid division by zero.

## Setup

$$
x = \\begin{bmatrix}2, -1, 3, 0\\end{bmatrix}, \\quad \\gamma = [1.0, 0.5, 2.0, 1.5]
$$

So in this example $d = 4$. We'll also ignore the $\\epsilon$ term for simplicity.

### Step 1: compute the RMS

$$
\\mathrm{RMS}(x) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2} = \\sqrt{\\frac{1}{4} (2^2 + (-1)^2 + 3^2 + 0^2)} = \\sqrt{\\frac{1}{4} (4 + 1 + 9 + 0)} = \\sqrt{\\frac{14}{4}} = \\sqrt{3.5} \\approx 1.8708
$$

### Step 2: normalize by the RMS

$$
\\hat{x} = \\frac{x}{\\mathrm{RMS}(x)} = \\begin{bmatrix}2/1.8708, -1/1.8708, 3/1.8708, 0/1.8708\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.5345, 1.6036, 0.0000\\end{bmatrix}
$$

### Step 3: apply per-feature scaling

$$
y = \\begin{bmatrix}1.0690 \\cdot 1.0, -0.5345 \\cdot 0.5, 1.6036 \\cdot 2.0, 0.0000 \\cdot 1.5\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.2673, 3.2071, 0.0000\\end{bmatrix}
$$

This is the RMSNorm output.

Compared with LayerNorm, RMSNorm skips two things: it does not subtract the mean (the $x$ above is divided directly, with no centering), and it has no shift $\\beta$.

## Why do we need the scale factor $\\gamma$?

Pure normalization fixes the RMS of the output to $\\approx 1$, which is too restrictive and not ideal. $\\gamma$ lets the model learn and reshape that constraint per feature: different features can be scaled up or down as training discovers which directions are useful.
`,
    },
    zh: {
      title: "RMSNorm 的例子",
      md: `
**一个很小的数值例子，说明 RMSNorm 怎么算。**

## 公式

$$
\\mathrm{RMSNorm}(x) = \\gamma \\cdot \\frac{x}{\\sqrt{\\frac{1}{d} \\sum_{i}x_i^2 + \\epsilon}}
$$

- $\\gamma$ 是可学习的缩放因子，它的用处后面再讲。
- $\\epsilon$ 是一个很小的常数，防止除以 0。

## 设定

$$
x = \\begin{bmatrix}2, -1, 3, 0\\end{bmatrix}, \\quad \\gamma = [1.0, 0.5, 2.0, 1.5]
$$

所以这个例子里 $d = 4$。为了简单，也忽略 $\\epsilon$。

### 第 1 步：算 RMS

$$
\\mathrm{RMS}(x) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2} = \\sqrt{\\frac{1}{4} (2^2 + (-1)^2 + 3^2 + 0^2)} = \\sqrt{\\frac{1}{4} (4 + 1 + 9 + 0)} = \\sqrt{\\frac{14}{4}} = \\sqrt{3.5} \\approx 1.8708
$$

### 第 2 步：除以 RMS

$$
\\hat{x} = \\frac{x}{\\mathrm{RMS}(x)} = \\begin{bmatrix}2/1.8708, -1/1.8708, 3/1.8708, 0/1.8708\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.5345, 1.6036, 0.0000\\end{bmatrix}
$$

### 第 3 步：逐维度缩放

$$
y = \\begin{bmatrix}1.0690 \\cdot 1.0, -0.5345 \\cdot 0.5, 1.6036 \\cdot 2.0, 0.0000 \\cdot 1.5\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.2673, 3.2071, 0.0000\\end{bmatrix}
$$

这就是 RMSNorm 的输出。

和 LayerNorm 相比，RMSNorm 少了两步：不减均值（上面的 $x$ 没有先减去均值），也没有平移项 $\\beta$。

## 为什么需要缩放因子 $\\gamma$？

只做归一化，会把输出的 RMS 固定在 $\\approx 1$，这个限制太死，并不理想。$\\gamma$ 让模型可以按维度学习、调整这个限制：训练中发现哪些方向有用，就可以把对应的维度放大或缩小。
`,
    },
  },

  ln_jacobian_closed_form_proof_notes: {
    en: {
      title: "LayerNorm Jacobian closed form proof",
      md: `
**Let's prove the closed form of the Jacobian of LayerNorm:**

$$
J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right), \\quad \\text{with } y = \\frac{z - \\mu}{\\sigma}
$$

## Setup

For a vector $z \\in \\mathbb{R}^m$:

- $\\mu = \\frac{1}{m} \\mathbf{1}^\\top z$ (feature mean)
- $\\mathrm{var} = \\frac{1}{m} \\sum_{i} (z_i - \\mu)^2 = \\frac{1}{m} z^\\top z - \\mu^2$
- $\\sigma = \\sqrt{\\mathrm{var}}$ (feature standard deviation; ignore $\\epsilon$ for simplicity)
- $y = \\mathrm{LN}(z) = \\frac{z - \\mu}{\\sigma}$

We want the Jacobian $J = \\frac{\\partial y}{\\partial z}$.

## Some useful derivatives

### 1. Mean

$$
\\frac{\\partial \\mu}{\\partial z} = \\frac{1}{m} \\mathbf{1}
$$

### 2. Std

$$
\\begin{aligned}
\\frac{\\partial\\, \\mathrm{var}}{\\partial z} &= \\frac{2}{m} z - 2\\mu\\frac{\\partial \\mu}{\\partial z} = \\frac{2}{m} (z - \\mu \\mathbf{1}) \\\\[4pt]
\\frac{\\partial \\sigma}{\\partial z} &= \\frac{\\partial \\sigma}{\\partial\\, \\mathrm{var}} \\times \\frac{\\partial\\, \\mathrm{var}}{\\partial z} = \\frac{1}{2\\sigma} \\times \\frac{2}{m} (z - \\mu \\mathbf{1}) = \\frac{1}{m} \\times \\frac{z - \\mu \\mathbf{1}}{\\sigma} = \\frac{1}{m} \\times y
\\end{aligned}
$$

### 3. Numerator $z - \\mu \\mathbf{1}$

$$
\\frac{\\partial (z - \\mu \\mathbf{1})}{\\partial z} = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top
$$

## Chain rule for $y = \\frac{z - \\mu \\mathbf{1}}{\\sigma}$

$$
\\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\times \\frac{\\partial (z - \\mu \\mathbf{1})}{\\partial z} + (z - \\mu \\mathbf{1}) \\times \\frac{\\partial (1/\\sigma)}{\\partial z}
$$

Since $\\frac{\\partial (1/\\sigma)}{\\partial z} = -\\frac{1}{\\sigma^2} \\times \\frac{\\partial \\sigma}{\\partial z}$, we have

$$
\\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{z - \\mu \\mathbf{1}}{\\sigma^2}\\left(\\frac{1}{m} y\\right)^\\top
$$

Now plug in $z = y\\sigma + \\mu \\mathbf{1}$:

$$
\\begin{aligned}
\\frac{\\partial y}{\\partial z} &= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{(y\\sigma + \\mu \\mathbf{1}) - \\mu \\mathbf{1}}{\\sigma^2}\\left(\\frac{1}{m} y\\right)^\\top \\\\[4pt]
&= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{y}{\\sigma}\\left(\\frac{1}{m} y\\right)^\\top \\\\[4pt]
&= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
\\end{aligned}
$$

> That's exactly the formula we wanted to prove.
`,
    },
    zh: {
      title: "LayerNorm Jacobian 闭式解的证明",
      md: `
**下面证明 LayerNorm 的 Jacobian 的闭式解：**

$$
J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right), \\quad \\text{其中 } y = \\frac{z - \\mu}{\\sigma}
$$

## 设定

对向量 $z \\in \\mathbb{R}^m$：

- $\\mu = \\frac{1}{m} \\mathbf{1}^\\top z$（特征的均值）
- $\\mathrm{var} = \\frac{1}{m} \\sum_{i} (z_i - \\mu)^2 = \\frac{1}{m} z^\\top z - \\mu^2$
- $\\sigma = \\sqrt{\\mathrm{var}}$（特征的标准差；为了简单忽略 $\\epsilon$）
- $y = \\mathrm{LN}(z) = \\frac{z - \\mu}{\\sigma}$

要求的是 Jacobian $J = \\frac{\\partial y}{\\partial z}$。

## 几个有用的导数

### 1. 均值

$$
\\frac{\\partial \\mu}{\\partial z} = \\frac{1}{m} \\mathbf{1}
$$

### 2. 标准差

$$
\\begin{aligned}
\\frac{\\partial\\, \\mathrm{var}}{\\partial z} &= \\frac{2}{m} z - 2\\mu\\frac{\\partial \\mu}{\\partial z} = \\frac{2}{m} (z - \\mu \\mathbf{1}) \\\\[4pt]
\\frac{\\partial \\sigma}{\\partial z} &= \\frac{\\partial \\sigma}{\\partial\\, \\mathrm{var}} \\times \\frac{\\partial\\, \\mathrm{var}}{\\partial z} = \\frac{1}{2\\sigma} \\times \\frac{2}{m} (z - \\mu \\mathbf{1}) = \\frac{1}{m} \\times \\frac{z - \\mu \\mathbf{1}}{\\sigma} = \\frac{1}{m} \\times y
\\end{aligned}
$$

### 3. 分子 $z - \\mu \\mathbf{1}$

$$
\\frac{\\partial (z - \\mu \\mathbf{1})}{\\partial z} = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top
$$

## 对 $y = \\frac{z - \\mu \\mathbf{1}}{\\sigma}$ 用链式法则

$$
\\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\times \\frac{\\partial (z - \\mu \\mathbf{1})}{\\partial z} + (z - \\mu \\mathbf{1}) \\times \\frac{\\partial (1/\\sigma)}{\\partial z}
$$

因为 $\\frac{\\partial (1/\\sigma)}{\\partial z} = -\\frac{1}{\\sigma^2} \\times \\frac{\\partial \\sigma}{\\partial z}$，所以

$$
\\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{z - \\mu \\mathbf{1}}{\\sigma^2}\\left(\\frac{1}{m} y\\right)^\\top
$$

再代入 $z = y\\sigma + \\mu \\mathbf{1}$：

$$
\\begin{aligned}
\\frac{\\partial y}{\\partial z} &= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{(y\\sigma + \\mu \\mathbf{1}) - \\mu \\mathbf{1}}{\\sigma^2}\\left(\\frac{1}{m} y\\right)^\\top \\\\[4pt]
&= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top\\right) - \\frac{y}{\\sigma}\\left(\\frac{1}{m} y\\right)^\\top \\\\[4pt]
&= \\frac{1}{\\sigma}\\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
\\end{aligned}
$$

> 这正是要证明的公式。
`,
    },
  },

  ln_gradient_attenuation_upper_bound_proof_notes: {
    en: {
      title: "Proof of gradient attenuation upper bound",
      md: `
**Let's prove the upper bound of the gradient attenuation of LayerNorm.**

## The Jacobian

As proved [here](#info:ln_jacobian_closed_form_proof_notes), LayerNorm's Jacobian at a vector $z \\in \\mathbb{R}^m$ can be written as

$$
J = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
$$

with

$$
y = \\frac{z - \\mu}{\\sigma}, \\quad \\mu = \\frac{1}{m} \\mathbf{1}^\\top z, \\quad \\sigma = \\sqrt{\\frac{1}{m} \\sum_{i=1}^m (z_i - \\mu)^2}
$$

- The $1/\\sigma$ in front scales the Jacobian.
- The matrix in parentheses,

  $$
  P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
  $$

  is an **orthogonal projection** onto the subspace orthogonal to both the all-ones vector $\\mathbf{1}$ and the vector $y$.

An orthogonal projection matrix has the property that its 2-norm (a.k.a. the spectral/operator norm) is $\\leq 1$, so $\\|P\\|_2 \\leq 1$. (See the related [math](#info:orthogonal_projection_notes) here.)

## The bound

Hence, the 2-norm of the Jacobian is bounded by

$$
\\|J\\|_2 = \\frac{1}{\\sigma} \\|P\\|_2 \\leq \\frac{1}{\\sigma}
$$

Here we use one important property of the 2-norm of a matrix: it is the largest factor by which the matrix can stretch a vector's Euclidean length, over any direction.

> Thus, $\\|J\\|_2$ directly upper-bounds the **per-layer gradient attenuation/amplification**.
`,
    },
    zh: {
      title: "梯度衰减上界的证明",
      md: `
**我们来证明 LayerNorm 梯度衰减的上界。**

## Jacobian

如[这里](#info:ln_jacobian_closed_form_proof_notes)所证，LayerNorm 在向量 $z \\in \\mathbb{R}^m$ 处的 Jacobian 可以写成

$$
J = \\frac{1}{\\sigma} \\left(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)
$$

其中

$$
y = \\frac{z - \\mu}{\\sigma}, \\quad \\mu = \\frac{1}{m} \\mathbf{1}^\\top z, \\quad \\sigma = \\sqrt{\\frac{1}{m} \\sum_{i=1}^m (z_i - \\mu)^2}
$$

- 前面的 $1/\\sigma$ 对 Jacobian 做缩放。
- 括号里的矩阵

  $$
  P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
  $$

  是一个**正交投影**，投影到同时与全 1 向量 $\\mathbf{1}$ 和向量 $y$ 正交的子空间上。

正交投影矩阵有一个性质：它的 2-范数（也叫谱范数/算子范数）$\\leq 1$，所以 $\\|P\\|_2 \\leq 1$。（相关的[数学推导](#info:orthogonal_projection_notes)见这里。）

## 上界

因此，Jacobian 的 2-范数有上界

$$
\\|J\\|_2 = \\frac{1}{\\sigma} \\|P\\|_2 \\leq \\frac{1}{\\sigma}
$$

这里用到了矩阵 2-范数的一个重要性质：它是在所有方向中，矩阵能把一个向量的欧几里得长度拉伸的最大倍数。

> 因此，$\\|J\\|_2$ 直接给出了**每一层梯度衰减/放大**的上界。
`,
    },
  },

  orthogonal_projection_notes: {
    en: {
      title: "Orthogonal projection",
      md: `
The matrix we are interested in is

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

- $\\mathbf{I}$ is the identity matrix of size $m \\times m$.
- $\\mathbf{1}$ is the all-ones vector of size $m$.
- $y$ is the LayerNorm output (zero mean, and $\\frac{1}{m} y^\\top y = 1$).

> **Theorem:** If $P$ is [symmetric](#hover:orthogonal_projection_symmetric_notes) ($P^\\top = P$) and [idempotent](#hover:orthogonal_projection_idempotent_notes) ($P^2 = P$), then $P$ is an orthogonal projector onto its range $Range(P)$. [Proof](#hover:orthogonal_projection_proof_notes)

## A. $P$ is the orthogonal projector onto the subspace orthogonal to both $\\mathbf{1}$ and $y$

Under the LayerNorm conditions, we have

$$
\\mathbf{1}^\\top y = 0, \\quad y^\\top y = m
$$

Let

$$
A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top, \\quad B = \\frac{1}{m} yy^\\top
$$

Define $S$ as the subspace that is orthogonal to both $\\mathbf{1}$ and $y$:

$$
S = \\{v \\in \\mathbb{R}^m : \\mathbf{1}^\\top v = 0, \\ y^\\top v = 0\\}
$$

We'll prove the following facts.

### 1. $P$ projects any $z$ into $S$

Take any vector $z$ and set $w = Pz$. We'll show $w$ is orthogonal to both $\\mathbf{1}$ and $y$.

**Orthogonal to $\\mathbf{1}$:**

$$
\\mathbf{1}^\\top w = \\mathbf{1}^\\top \\left(I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)z = \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top \\mathbf{1}}_{=m} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top y}_{=0} y^\\top z = 0
$$

**Orthogonal to $y$:**

$$
y^\\top w = y^\\top z - \\frac{1}{m} \\underbrace{y^\\top \\mathbf{1}}_{=0} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{y^\\top y}_{=m} y^\\top z = 0
$$

Then $w = Pz \\in S$. So $Range(P) \\subseteq S$.

### 2. $Pz = z$ for any $z \\in S$

Take any $z \\in S$. Then $\\mathbf{1}^\\top z = 0$ and $y^\\top z = 0$, so

$$
\\begin{aligned}
Pz &= \\left(I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)z \\\\
&= z - \\frac{1}{m} \\mathbf{1}(\\mathbf{1}^\\top z) - \\frac{1}{m} y(y^\\top z) \\\\
&= z - \\frac{1}{m} \\mathbf{1} \\cdot 0 - \\frac{1}{m} y \\cdot 0 \\\\
&= z
\\end{aligned}
$$

### 3. $P$ is the orthogonal projector onto $S$

From 1 and 2:

- We showed $Range(P) \\subseteq S$ (from 1) and $Pz = z$ for any $z \\in S$ (from 2). Hence $Range(P) = S$ and $P$ acts as the identity on $S$.
- We also showed $P$ is symmetric and idempotent at the beginning of this note.

> According to the theorem, $P$ is the orthogonal projector onto its range $Range(P)$, and $Range(P) = S$, so **$P$ is the orthogonal projector onto $S$**.

## B. The 2-norm of any orthogonal projection matrix $P \\neq 0$ is 1

We'll prove that for an orthogonal projection matrix $P$ (i.e., $P^\\top = P$ and $P^2 = P$), the 2-norm satisfies

$$
\\|P\\|_2 \\leq 1
$$

and in fact $\\|P\\|_2 = 1$ unless $P = 0$.

Here is a quick proof.

### Geometric (Pythagoras) proof

Any vector $x$ decomposes uniquely as $x = u + s$ with $u \\in range(P)$ and $s \\in ker(P)$.

Since $Px = u$,

$$
\\|Px\\|_2^2 = \\|u\\|_2^2 \\leq \\|u\\|_2^2 + \\|s\\|_2^2 = \\|x\\|_2^2 \\Rightarrow \\|Px\\|_2 \\leq \\|x\\|_2 \\Rightarrow \\frac{\\|Px\\|_2}{\\|x\\|_2} \\leq 1
$$

The definition of the 2-norm is $\\|P\\|_2 = \\max_{x \\neq 0} \\frac{\\|Px\\|_2}{\\|x\\|_2}$, so $\\|P\\|_2 \\leq 1$.

> In fact, if $P \\neq 0$, pick a nonzero $x \\in range(P)$. Then $Px = x$, so $\\|Px\\|_2 = \\|x\\|_2$, and we have $\\|P\\|_2 = 1$.
`,
    },
    zh: {
      title: "正交投影",
      md: `
我们关心的矩阵是

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

- $\\mathbf{I}$ 是 $m \\times m$ 的单位矩阵。
- $\\mathbf{1}$ 是长度为 $m$ 的全 1 向量。
- $y$ 是 LayerNorm 的输出（均值为零，且 $\\frac{1}{m} y^\\top y = 1$）。

> **定理：** 如果 $P$ 是[对称的](#hover:orthogonal_projection_symmetric_notes)（$P^\\top = P$）且是[幂等的](#hover:orthogonal_projection_idempotent_notes)（$P^2 = P$），那么 $P$ 是到其值域 $Range(P)$ 上的正交投影。[证明](#hover:orthogonal_projection_proof_notes)

## A. $P$ 是到同时与 $\\mathbf{1}$ 和 $y$ 正交的子空间上的正交投影

在 LayerNorm 的条件下，我们有

$$
\\mathbf{1}^\\top y = 0, \\quad y^\\top y = m
$$

令

$$
A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top, \\quad B = \\frac{1}{m} yy^\\top
$$

定义 $S$ 为同时与 $\\mathbf{1}$ 和 $y$ 正交的子空间：

$$
S = \\{v \\in \\mathbb{R}^m : \\mathbf{1}^\\top v = 0, \\ y^\\top v = 0\\}
$$

我们来证明下面几个事实。

### 1. $P$ 把任意 $z$ 投影到 $S$ 中

任取一个向量 $z$，令 $w = Pz$。我们证明 $w$ 同时与 $\\mathbf{1}$ 和 $y$ 正交。

**与 $\\mathbf{1}$ 正交：**

$$
\\mathbf{1}^\\top w = \\mathbf{1}^\\top \\left(I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)z = \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top \\mathbf{1}}_{=m} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top y}_{=0} y^\\top z = 0
$$

**与 $y$ 正交：**

$$
y^\\top w = y^\\top z - \\frac{1}{m} \\underbrace{y^\\top \\mathbf{1}}_{=0} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{y^\\top y}_{=m} y^\\top z = 0
$$

于是 $w = Pz \\in S$。所以 $Range(P) \\subseteq S$。

### 2. 对任意 $z \\in S$，$Pz = z$

任取 $z \\in S$。那么 $\\mathbf{1}^\\top z = 0$ 且 $y^\\top z = 0$，所以

$$
\\begin{aligned}
Pz &= \\left(I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top\\right)z \\\\
&= z - \\frac{1}{m} \\mathbf{1}(\\mathbf{1}^\\top z) - \\frac{1}{m} y(y^\\top z) \\\\
&= z - \\frac{1}{m} \\mathbf{1} \\cdot 0 - \\frac{1}{m} y \\cdot 0 \\\\
&= z
\\end{aligned}
$$

### 3. $P$ 是到 $S$ 上的正交投影

由 1 和 2：

- 我们证明了 $Range(P) \\subseteq S$（由 1）以及对任意 $z \\in S$ 有 $Pz = z$（由 2）。因此 $Range(P) = S$，并且 $P$ 在 $S$ 上就是恒等映射。
- 本笔记开头我们也证明了 $P$ 是对称且幂等的。

> 根据定理，$P$ 是到其值域 $Range(P)$ 上的正交投影，而 $Range(P) = S$，所以 **$P$ 是到 $S$ 上的正交投影**。

## B. 任何正交投影矩阵 $P \\neq 0$ 的 2-范数都是 1

我们来证明：对于正交投影矩阵 $P$（即 $P^\\top = P$ 且 $P^2 = P$），它的 2-范数满足

$$
\\|P\\|_2 \\leq 1
$$

并且事实上，除非 $P = 0$，否则 $\\|P\\|_2 = 1$。

下面是一个简短的证明。

### 几何（勾股定理）证明

任意向量 $x$ 都可以唯一分解为 $x = u + s$，其中 $u \\in range(P)$，$s \\in ker(P)$。

由于 $Px = u$，

$$
\\|Px\\|_2^2 = \\|u\\|_2^2 \\leq \\|u\\|_2^2 + \\|s\\|_2^2 = \\|x\\|_2^2 \\Rightarrow \\|Px\\|_2 \\leq \\|x\\|_2 \\Rightarrow \\frac{\\|Px\\|_2}{\\|x\\|_2} \\leq 1
$$

2-范数的定义是 $\\|P\\|_2 = \\max_{x \\neq 0} \\frac{\\|Px\\|_2}{\\|x\\|_2}$，所以 $\\|P\\|_2 \\leq 1$。

> 事实上，如果 $P \\neq 0$，取一个非零的 $x \\in range(P)$。那么 $Px = x$，所以 $\\|Px\\|_2 = \\|x\\|_2$，于是 $\\|P\\|_2 = 1$。
`,
    },
  },

  orthogonal_projection_idempotent_notes: {
    en: {
      title: "Idempotent projection (hover)",
      md: `
**Prove $P$ is idempotent: $P^2 = P$.**

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

under the LayerNorm conditions

$$
\\mathbf{1}^\\top y = 0, \\quad y^\\top y = m
$$

Let

$$
A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top, \\quad B = \\frac{1}{m} yy^\\top
$$

We'll prove the following facts.

### 1. $A$ and $B$ are idempotent

$$
A^2 = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}\\mathbf{1}(m)\\mathbf{1}^\\top = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top = A
$$

Similarly, using $y^\\top y = m$, we have

$$
B^2 = \\frac{1}{m^2}y(y^\\top y)y^\\top = \\frac{1}{m^2}y(m)y^\\top = \\frac{1}{m}yy^\\top = B
$$

### 2. $A$ and $B$ are orthogonal

Using $\\mathbf{1}^\\top y = 0$:

$$
AB = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top y)y^\\top = \\frac{1}{m^2}\\mathbf{1}(0)y^\\top = 0 \\\\[4pt]
BA = \\frac{1}{m^2}y(y^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}y(0)\\mathbf{1}^\\top = 0
$$

### 3. Expand $P^2$ and simplify

$$
\\begin{aligned}
P^2 &= (I - A - B)^2 \\\\
&= I - 2A - 2B + A^2 + B^2 + AB + BA \\\\
&= I - 2A - 2B + A + B + 0 + 0 \\quad \\text{(by facts 1 and 2)} \\\\
&= I - A - B \\\\
&= P
\\end{aligned}
$$

> Hence $P^2 = P$.
`,
    },
    zh: {
      title: "投影的幂等性（悬浮提示）",
      md: `
**证明 $P$ 是幂等的：$P^2 = P$。**

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

在 LayerNorm 的条件下

$$
\\mathbf{1}^\\top y = 0, \\quad y^\\top y = m
$$

令

$$
A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top, \\quad B = \\frac{1}{m} yy^\\top
$$

我们来证明下面几个事实。

### 1. $A$ 和 $B$ 都是幂等的

$$
A^2 = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}\\mathbf{1}(m)\\mathbf{1}^\\top = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top = A
$$

类似地，利用 $y^\\top y = m$，有

$$
B^2 = \\frac{1}{m^2}y(y^\\top y)y^\\top = \\frac{1}{m^2}y(m)y^\\top = \\frac{1}{m}yy^\\top = B
$$

### 2. $A$ 和 $B$ 是正交的

利用 $\\mathbf{1}^\\top y = 0$：

$$
AB = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top y)y^\\top = \\frac{1}{m^2}\\mathbf{1}(0)y^\\top = 0 \\\\[4pt]
BA = \\frac{1}{m^2}y(y^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}y(0)\\mathbf{1}^\\top = 0
$$

### 3. 展开 $P^2$ 并化简

$$
\\begin{aligned}
P^2 &= (I - A - B)^2 \\\\
&= I - 2A - 2B + A^2 + B^2 + AB + BA \\\\
&= I - 2A - 2B + A + B + 0 + 0 \\quad \\text{(by facts 1 and 2)} \\\\
&= I - A - B \\\\
&= P
\\end{aligned}
$$

> 因此 $P^2 = P$。
`,
    },
  },

  orthogonal_projection_symmetric_notes: {
    en: {
      title: "Symmetric projection (hover)",
      md: `
**Prove $P$ is symmetric: $P^\\top = P$.**

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

Take the transpose term by term, using $(A+B)^\\top = A^\\top + B^\\top$, $(AB)^\\top = B^\\top A^\\top$ and $(cA)^\\top = cA^\\top$ for a scalar $c$.

1. $I^\\top = I$ (the identity is symmetric).
2. $(\\mathbf{1}\\mathbf{1}^\\top)^\\top = (\\mathbf{1}^\\top)^\\top \\mathbf{1}^\\top = \\mathbf{1}\\mathbf{1}^\\top$.
3. $(yy^\\top)^\\top = (y^\\top)^\\top y^\\top = yy^\\top$.

Therefore,

$$
P^\\top = I^\\top - \\frac{1}{m} (\\mathbf{1}\\mathbf{1}^\\top)^\\top - \\frac{1}{m} (yy^\\top)^\\top = I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top = P
$$
`,
    },
    zh: {
      title: "投影的对称性（悬浮提示）",
      md: `
**证明 $P$ 是对称的：$P^\\top = P$。**

$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
$$

逐项取转置，利用 $(A+B)^\\top = A^\\top + B^\\top$、$(AB)^\\top = B^\\top A^\\top$，以及对标量 $c$ 有 $(cA)^\\top = cA^\\top$。

1. $I^\\top = I$（单位矩阵是对称的）。
2. $(\\mathbf{1}\\mathbf{1}^\\top)^\\top = (\\mathbf{1}^\\top)^\\top \\mathbf{1}^\\top = \\mathbf{1}\\mathbf{1}^\\top$。
3. $(yy^\\top)^\\top = (y^\\top)^\\top y^\\top = yy^\\top$。

因此，

$$
P^\\top = I^\\top - \\frac{1}{m} (\\mathbf{1}\\mathbf{1}^\\top)^\\top - \\frac{1}{m} (yy^\\top)^\\top = I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top = P
$$
`,
    },
  },

  orthogonal_projection_proof_notes: {
    en: {
      title: "Symmetric + idempotent ⇒ orthogonal projector",
      md: `
**Let's prove the theorem that a symmetric idempotent matrix is the orthogonal projector onto its range.**

- **Idempotent** ($P^2 = P$) alone $\\Rightarrow$ $P$ is a projection (onto $range(P)$ along $ker(P)$), but not necessarily orthogonal.
- **Idempotent + symmetric** ($P^\\top = P$) $\\Rightarrow$ $P$ is the orthogonal projector onto its column space $S = Range(P)$.

## Proof

### 1. Idempotent $\\Rightarrow$ projection

For any $x$:

- $Px \\in range(P)$.
- $x - Px \\in ker(P)$, because $P(x - Px) = Px - P^2x = Px - Px = 0$.

So every $x$ decomposes uniquely as $x = Px + (x - Px)$, with the first part in $range(P)$ and the second in $ker(P)$. That's exactly a projection.

### 2. Add symmetry $\\Rightarrow$ orthogonal projector

Let $S = range(P)$.

Choose any vector $z \\in ker(P)$, so $Pz = 0$.

Choose any vector $y = Pw \\in S$. Then

$$
y^\\top z = w^\\top P^\\top z = w^\\top P z = 0
$$

So $z$ and $y$ are orthogonal. Because $z$ is chosen arbitrarily from $ker(P)$ and $y$ is chosen arbitrarily from $S$:

> $ker(P) \\perp S$, i.e., $P$ is the orthogonal projector onto $S$.
`,
    },
    zh: {
      title: "对称 + 幂等 ⇒ 正交投影",
      md: `
**我们来证明这条定理：对称且幂等的矩阵，就是到其值域上的正交投影。**

- 仅有**幂等**（$P^2 = P$）$\\Rightarrow$ $P$ 是一个投影（沿 $ker(P)$ 投影到 $range(P)$ 上），但不一定是正交的。
- **幂等 + 对称**（$P^\\top = P$）$\\Rightarrow$ $P$ 是到其列空间 $S = Range(P)$ 上的正交投影。

## 证明

### 1. 幂等 $\\Rightarrow$ 投影

对任意 $x$：

- $Px \\in range(P)$。
- $x - Px \\in ker(P)$，因为 $P(x - Px) = Px - P^2x = Px - Px = 0$。

所以每个 $x$ 都可以唯一分解为 $x = Px + (x - Px)$，第一部分在 $range(P)$ 中，第二部分在 $ker(P)$ 中。这正是一个投影。

### 2. 再加上对称性 $\\Rightarrow$ 正交投影

令 $S = range(P)$。

任取一个向量 $z \\in ker(P)$，即 $Pz = 0$。

任取一个向量 $y = Pw \\in S$。那么

$$
y^\\top z = w^\\top P^\\top z = w^\\top P z = 0
$$

所以 $z$ 和 $y$ 正交。由于 $z$ 是从 $ker(P)$ 中任意选取的，$y$ 是从 $S$ 中任意选取的：

> $ker(P) \\perp S$，即 $P$ 是到 $S$ 上的正交投影。
`,
    },
  },
};
