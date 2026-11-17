// Notes: Scaling after GPT-2 (GPT-3, Chinchilla): same architecture, bigger models and more data.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  gpt3_notes: {
    en: {
      title: "GPT-3: scaling up and in-context learning",
      md: `
**GPT-3** (OpenAI, 2020) kept GPT-2's architecture and made it more than 100× larger. Its contribution was not a new layer but a new way to use the model: a large enough model can pick up a new task from a few examples written into the prompt, with no training. This is called **in-context learning**.

## Architecture: GPT-2, scaled up

| | GPT-2 small | GPT-2 XL | GPT-3 |
|---|---|---|---|
| Layers $L$ | 12 | 48 | 96 |
| Width $d_{\\text{model}}$ | 768 | 1600 | 12288 |
| Attention heads | 12 | 25 | 96 |
| Context length | 1024 | 1024 | **2048** |
| Vocabulary | 50257 | 50257 | 50257 (same BPE) |
| Parameters | 124M | 1.5B | **175B** |

### Checking the parameter count

Ignoring the small biases and LayerNorm parameters, each [transformer block](#info:highlevel_transformer_introduction_notes) has about $12 d^2$ parameters:

- **Attention:** $W_q, W_k, W_v$ and the output projection, four $d \\times d$ matrices: $4d^2$.
- **MLP:** $d \\to 4d$ and $4d \\to d$: $4d^2 + 4d^2 = 8d^2$.

For GPT-3:

\`\`\`text
blocks:      12 × 96 × 12288²            ≈ 173.9B
embeddings:  (50257 + 2048) × 12288      ≈   0.6B
total                                    ≈ 174.6B    ← the paper's "175B"
\`\`\`

So GPT-3's block is the same as the GPT-2 code on this site: pre-LN, GELU, learned position embeddings, [weight tying](#info:weight_tying_notes). Only the sizes change.

### What did change

1. **Context length** doubled from 1024 to 2048 tokens.
2. **Alternating attention patterns:** layers alternate between normal (dense) attention and "locally banded sparse" attention that only looks at nearby tokens, taken from the Sparse Transformer. This did not catch on. The kind of local attention that later became common is Mistral's sliding window.

### Training settings

These are the settings the [optimizer notes](#info:optim_adamw_notes) refer to: Adam with $\\beta_1 = 0.9$, $\\beta_2 = 0.95$, gradient [clipping](#info:clip_grad_norm_notes) at 1.0, [weight decay](#info:adamw_notes) 0.1, and a learning rate schedule (linear warmup over the first 375M tokens, then cosine decay to 10% of the peak). GPT-3 was trained on 300B tokens.

## In-context learning

### Before: fine-tuning

In the GPT-1 and BERT era, using a pretrained model for a new task meant **fine-tuning**:

\`\`\`text
pretrained model
  → collect thousands of labeled examples ("English → French")
  → keep training with gradient descent, updating the weights
  → a model specialized for translation
\`\`\`

Every new task needed new labeled data and another training run.

### GPT-3: put the examples in the prompt

The weights are **not changed at all**. The task description and a few examples go into the input text, and the model continues it. An example from the GPT-3 paper:

\`\`\`text
Translate English to French:          ← task description
sea otter => loutre de mer            ← example 1
peppermint => menthe poivrée          ← example 2
plush giraffe => girafe peluche       ← example 3
cheese =>                             ← the model continues: fromage
\`\`\`

By the number of examples in the prompt:

- **Zero-shot:** only the task description, no examples.
- **One-shot:** one example.
- **Few-shot:** several examples, typically 10 to 100, limited by the 2048-token context.

### Why this works

For the model this is **still next-token prediction**, exactly what the GPT-2 code on this site trains for. The training data contains many lists of "A => B" pairs. To predict the next token well, the model has to notice the pattern in the earlier lines and apply it to the last one. No gradient is computed: the "learning" happens entirely in the forward pass, where [attention](#info:causal_analogy_notes) reads the earlier examples.

### It gets stronger with scale

The paper's key finding: **the larger the model, the more it gains from examples in the prompt.** Adding examples barely helps small models, but helps the 175B model a lot. For example, on two-digit addition the 175B model is 100% accurate few-shot, and about 80% on three-digit addition.

This is why making models larger became the main direction. The next question, how to split a compute budget between a larger model and more data, is answered by [Chinchilla](#info:chinchilla_notes).

🆕 **What followed:** "prompting" comes from here. GPT-3 itself does not follow instructions well; you have to lay out a pattern for it to continue, as above. InstructGPT (2022) used supervised fine-tuning (SFT) plus RLHF so the model directly understands "Translate this sentence into French."

## References

1. **Language Models are Few-Shot Learners** (GPT-3). Brown et al., 2020. [arXiv:2005.14165 ↗](https://arxiv.org/abs/2005.14165)
`,
    },
    zh: {
      title: "GPT-3：放大规模与 in-context learning",
      md: `
**GPT-3**（OpenAI，2020）保留了 GPT-2 的架构，只是把它放大了 100 多倍。它的贡献不是新的层，而是一种新的用法：模型足够大时，只要在 prompt 里写几个例子，它就能学会一个新任务，完全不用训练。这叫 **in-context learning（上下文学习）**。

## 架构：放大的 GPT-2

| | GPT-2 small | GPT-2 XL | GPT-3 |
|---|---|---|---|
| 层数 $L$ | 12 | 48 | 96 |
| 宽度 $d_{\\text{model}}$ | 768 | 1600 | 12288 |
| 注意力头数 | 12 | 25 | 96 |
| 上下文长度 | 1024 | 1024 | **2048** |
| 词表 | 50257 | 50257 | 50257（同一个 BPE） |
| 参数量 | 124M | 1.5B | **175B** |

### 验证参数量

忽略很少的 bias 和 LayerNorm 参数，每个 [transformer block](#info:highlevel_transformer_introduction_notes) 大约有 $12 d^2$ 个参数：

- **Attention：** $W_q$、$W_k$、$W_v$ 和输出投影，4 个 $d \\times d$ 矩阵：$4d^2$。
- **MLP：** $d \\to 4d$ 和 $4d \\to d$：$4d^2 + 4d^2 = 8d^2$。

代入 GPT-3：

\`\`\`text
block:       12 × 96 × 12288²            ≈ 173.9B
embedding:   (50257 + 2048) × 12288      ≈   0.6B
合计                                      ≈ 174.6B    ← 论文里的 "175B"
\`\`\`

所以 GPT-3 的 block 和这个网站的 GPT-2 代码一样：pre-LN、GELU、学到的位置 embedding、[weight tying](#info:weight_tying_notes)。只是尺寸变了。

### 真正改了什么

1. **上下文长度**从 1024 加倍到 2048 个 token。
2. **交替使用两种注意力**：一层是正常的全注意力（dense），下一层是只看附近 token 的 "locally banded sparse" 注意力，来自 Sparse Transformer。这个做法后来没有流行。后来流行的局部注意力是 Mistral 的 sliding window（滑动窗口）。

### 训练设置

[optimizer 的几篇 note](#info:optim_adamw_notes) 里提到的就是这些设置：Adam，$\\beta_1 = 0.9$、$\\beta_2 = 0.95$，梯度[裁剪](#info:clip_grad_norm_notes)阈值 1.0，[weight decay](#info:adamw_notes) 0.1，以及学习率调度（前 3.75 亿个 token 线性 warmup，然后 cosine 衰减到峰值的 10%）。GPT-3 一共训练了 3000 亿（300B）个 token。

## In-context learning

### 以前：fine-tuning

GPT-1 和 BERT 那个时代，要让预训练模型做一个新任务，就要 **fine-tuning（微调）**：

\`\`\`text
预训练好的模型
  → 收集几千条标注数据（"英文 → 法文"）
  → 用梯度下降继续训练，更新权重
  → 得到一个专门做翻译的模型
\`\`\`

每换一个任务，都要重新收集标注数据、重新训练一次。

### GPT-3：把例子写进 prompt

**权重完全不动**。把任务说明和几个例子写进输入文本，让模型接着往下写。下面是 GPT-3 论文里的例子：

\`\`\`text
Translate English to French:          ← 任务说明
sea otter => loutre de mer            ← 例子 1
peppermint => menthe poivrée          ← 例子 2
plush giraffe => girafe peluche       ← 例子 3
cheese =>                             ← 模型接着生成: fromage
\`\`\`

按 prompt 里给几个例子，分成三种：

- **Zero-shot：** 只有任务说明，不给例子。
- **One-shot：** 给 1 个例子。
- **Few-shot：** 给好几个例子，通常 10 到 100 个，受 2048 个 token 的上下文长度限制。

### 为什么这样能行

对模型来说，这**仍然只是 next-token prediction**，和这个网站的 GPT-2 代码训练的任务完全一样。训练数据里有大量"A => B"这样的对照列表。要准确预测下一个 token，模型就得看出前面几行的规律，再把它用到最后一行上。整个过程没有计算任何梯度："学习"完全发生在 forward 里，是 [attention](#info:causal_analogy_notes) 读到了前面的例子。

### 规模越大，能力越强

论文最关键的发现是：**模型越大，从 prompt 里的例子中学到的越多。** 给小模型加例子几乎没用，给 175B 的模型加例子，效果提升很明显。比如两位数加法，175B 模型在 few-shot 下准确率是 100%，三位数加法也有大约 80%。

这就是"把模型做大"成为主流方向的原因。接下来的问题是：算力预算该怎么在"更大的模型"和"更多的数据"之间分配？这由 [Chinchilla](#info:chinchilla_notes) 回答。

🆕 **之后的影响：** 今天的"写 prompt"就是从这里来的。GPT-3 本身不太会听指令，你得像上面那样摆好格式，让它接着往下写。InstructGPT（2022）用监督微调（SFT）加 RLHF，让模型能直接听懂"请把这句话翻译成法文"。

## 参考资料

1. **Language Models are Few-Shot Learners**（GPT-3）。Brown 等人，2020。[arXiv:2005.14165 ↗](https://arxiv.org/abs/2005.14165)
`,
    },
  },

  chinchilla_notes: {
    en: {
      title: "Chinchilla: how to split compute between parameters and data",
      md: `
**Chinchilla** (DeepMind, 2022) answered a practical question: with a fixed amount of compute, how large should the model be, and how many tokens should it be trained on? The answer: grow the two **together**, about 20 training tokens per parameter. By that rule [GPT-3](#info:gpt3_notes) was much too large for the data it saw.

## How much compute training takes

A standard estimate of the total training compute (in floating-point operations, FLOPs):

$$
C \\approx 6 \\cdot N \\cdot D
$$

- $N$ is the number of parameters; $D$ is the number of training tokens.
- **Forward pass: 2 operations per parameter per token.** Each weight is used once per token, in one multiply and one add, as in a [Linear](#info:linear) layer's $W x$.
- **Backward pass: about 4 operations per parameter per token**, roughly twice the forward pass: [backward](#info:backward_notes) computes both the gradient for the weights and the gradient for the layer's input.
- Together: $2 + 4 = 6$.

So **for a fixed compute budget $C$, $N$ and $D$ trade off against each other**: a bigger model can be trained on fewer tokens.

## The Chinchilla result: scale $N$ and $D$ equally

The authors trained over 400 models (70M to 16B parameters, 5B to 500B tokens) and, for each compute budget, found which split gives the lowest loss. The result:

$$
D_{\\text{optimal}} \\approx 20 \\times N
$$

$N$ and $D$ should grow at the same rate: with 4× the compute, use 2× the parameters and 2× the tokens.

Before this, the common guide was an earlier scaling-law paper (Kaplan et al., 2020), which concluded that most extra compute should go into a larger model and relatively little into more data. GPT-3 followed that advice.

## GPT-3 was under-trained

\`\`\`text
              parameters N   tokens D    tokens per parameter
GPT-3         175B           300B        1.7      ← far below 20
Gopher        280B           300B        1.1
Chinchilla    70B            1.4T        20.0
\`\`\`

- **GPT-3:** $C = 6 \\times 175\\text{B} \\times 300\\text{B} \\approx 3.2 \\times 10^{23}$ FLOPs. Splitting that budget 20:1 gives $C = 6 \\times N \\times 20N$, so $N = \\sqrt{C / 120} \\approx 51\\text{B}$ parameters and $D \\approx 1\\text{T}$ tokens: a model less than a third the size, trained on more than 3× the data.
- **Chinchilla vs Gopher:** DeepMind's own Gopher (280B) and Chinchilla (70B) used similar compute ($5.0 \\times 10^{23}$ vs $5.9 \\times 10^{23}$ FLOPs). Chinchilla, a quarter of Gopher's size, did better on almost every benchmark, and is also 4× cheaper to run.

## 🆕 After Chinchilla: training past the "optimum"

20:1 is the split that minimizes **training** compute. But a model is trained once and then used for inference many, many times, and inference cost depends only on $N$. So later models deliberately train smaller models on far more than 20 tokens per parameter:

\`\`\`text
LLaMA-1 7B      1T tokens      about 150 tokens per parameter
LLaMA-3 8B      15T tokens     about 1900 tokens per parameter
\`\`\`

The loss keeps going down with more data, just more slowly than at the 20:1 point. Spending extra compute on training buys a small, strong model that is cheap to run. This is the starting point of LLaMA.

## This site's training run, for scale

The GPT-2 code on this site trains for 1000 steps on batches of $4 \\times 32$ tokens: $1000 \\times 4 \\times 32 = 128{,}000$ tokens for 124M parameters, about 0.001 tokens per parameter. It demonstrates the training loop; it cannot actually train GPT-2.

## References

1. **Training Compute-Optimal Large Language Models** (Chinchilla). Hoffmann et al., 2022. [arXiv:2203.15556 ↗](https://arxiv.org/abs/2203.15556)
2. **Scaling Laws for Neural Language Models.** Kaplan et al., 2020. [arXiv:2001.08361 ↗](https://arxiv.org/abs/2001.08361)
`,
    },
    zh: {
      title: "Chinchilla：算力该怎么分给参数和数据",
      md: `
**Chinchilla**（DeepMind，2022）回答了一个很实际的问题：算力固定时，模型该做多大、该用多少 token 来训练？答案是：两者**一起**增长，每个参数大约对应 20 个训练 token。按这条规则，[GPT-3](#info:gpt3_notes) 相对它见过的数据来说，模型做得太大了。

## 训练要花多少算力

训练总计算量（以浮点运算次数 FLOPs 计）有一个常用的估算公式：

$$
C \\approx 6 \\cdot N \\cdot D
$$

- $N$ 是参数量；$D$ 是训练用的 token 数。
- **Forward：每个参数、每个 token 做 2 次运算。** 每个权重对每个 token 用一次：一次乘法加一次加法，就像 [Linear](#info:linear) 层里的 $W x$。
- **Backward：每个参数、每个 token 大约做 4 次运算**，约为 forward 的 2 倍，因为 [backward](#info:backward_notes) 既要算权重的梯度，也要算这一层输入的梯度。
- 合计：$2 + 4 = 6$。

所以**算力预算 $C$ 固定时，$N$ 和 $D$ 此消彼长**：模型越大，能训练的 token 就越少。

## Chinchilla 的结论：$N$ 和 $D$ 同比例增长

作者训练了 400 多个模型（7000 万到 160 亿参数，50 亿到 5000 亿 token），对每个算力预算找出 loss 最低的分配方式。结论是：

$$
D_{\\text{optimal}} \\approx 20 \\times N
$$

$N$ 和 $D$ 应该以相同的速度增长：算力增加 4 倍，参数和 token 就各增加 2 倍。

在这之前，大家参考的是更早的一篇 scaling law 论文（Kaplan 等人，2020）。它的结论是：多出来的算力大部分应该用来把模型做大，数据只需要增加一点。GPT-3 就是照这个思路做的。

## GPT-3 训练不足

\`\`\`text
              参数 N      token 数 D    每个参数对应的 token
GPT-3         175B        300B          1.7      ← 远少于 20
Gopher        280B        300B          1.1
Chinchilla    70B         1.4T          20.0
\`\`\`

- **GPT-3：** $C = 6 \\times 175\\text{B} \\times 300\\text{B} \\approx 3.2 \\times 10^{23}$ FLOPs。把这个预算按 20:1 分配，就是 $C = 6 \\times N \\times 20N$，所以 $N = \\sqrt{C / 120} \\approx 51\\text{B}$ 个参数、$D \\approx 1\\text{T}$ 个 token：模型不到原来的三分之一，数据是原来的 3 倍多。
- **Chinchilla 对 Gopher：** DeepMind 自己的 Gopher（280B）和 Chinchilla（70B）用的算力差不多（$5.0 \\times 10^{23}$ 对 $5.9 \\times 10^{23}$ FLOPs）。Chinchilla 只有 Gopher 的四分之一大，却在几乎所有评测上都更好，推理也便宜 4 倍。

## 🆕 Chinchilla 之后：训练得比"最优"还久

20:1 是让**训练**算力最省的分配。但模型只训练一次，之后要用来推理很多很多次，而推理成本只取决于 $N$。所以后来的模型故意用远超每个参数 20 个 token 的数据，去训练更小的模型：

\`\`\`text
LLaMA-1 7B      1T token      每个参数约 150 个 token
LLaMA-3 8B      15T token     每个参数约 1900 个 token
\`\`\`

数据越多，loss 还会继续下降，只是比在 20:1 附近降得慢。训练时多花一些算力，换来一个小而强、推理便宜的模型。这就是 LLaMA 的出发点。

## 对照一下这个网站的训练

这个网站的 GPT-2 代码训练 1000 步，每个 batch $4 \\times 32$ 个 token：一共 $1000 \\times 4 \\times 32 = 128{,}000$ 个 token，对应 1.24 亿个参数，每个参数大约只有 0.001 个 token。它只是演示训练流程，并不能真的训练出 GPT-2。

## 参考资料

1. **Training Compute-Optimal Large Language Models**（Chinchilla）。Hoffmann 等人，2022。[arXiv:2203.15556 ↗](https://arxiv.org/abs/2203.15556)
2. **Scaling Laws for Neural Language Models.** Kaplan 等人，2020。[arXiv:2001.08361 ↗](https://arxiv.org/abs/2001.08361)
`,
    },
  },
};
