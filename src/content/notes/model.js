// Notes: GPT-2 overview and the transformer block.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  gpt2_notes: {
    en: {
      title: "GPT-2: overview",
      md: `
**GPT-2** (OpenAI, 2019) is a decoder-only Transformer trained for one task: predict the next token. This site's code builds its smallest version, 124M parameters. This note is the overview: what GPT-2 is, how data flows through it, and what changed from GPT-1. Each step links to its detailed note.

## What GPT-2 is

- **Decoder-only:** a stack of identical Transformer blocks with [causal self-attention](#info:causal_mask_notes): each token can only see itself and the tokens before it. There is no encoder.
- **One training objective:** next-token prediction with [cross-entropy](#info:cross_entropy_notes) loss. No labels are needed; the next token in the text is the target.
- **Training data: WebText,** about 8 million web pages (40 GB of text), collected from outbound Reddit links with at least 3 karma, as a rough filter for quality.
- **The paper's point:** a language model trained this way, with no task-specific training, can already do some tasks zero-shot (e.g. answer questions, summarize, translate a little), and gets better at them as it grows. [GPT-3](#info:gpt3_notes) took this much further.

## Four sizes

| | Layers | Width $d_{\\text{model}}$ | Heads | Parameters |
|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 | 124M |
| GPT-2 medium | 24 | 1024 | 16 | 355M |
| GPT-2 large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

All four share the context length (1024 tokens), the vocabulary (50257 tokens) and the head size ($d_{\\text{model}} / \\text{heads} = 64$). The paper listed the two smallest as 117M and 345M; counted exactly, they are 124M and 355M.

This site's code is **GPT-2 small**: \`n_layer=12, n_embd=768, n_head=12\`.

## How data flows through the model

Shapes use $B$ = batch size, $T$ = sequence length, $C = 768$, $V$ = vocabulary size (see [dimension changes](#info:dimension_change_notes) for every step).

1. **[Input text](#info:input_data_notes)** → **[tokenizer](#info:tokenizer_notes)** (byte-level BPE): text becomes token ids, shape $(B, T)$.
2. **[Token embedding](#info:embedding_tok)** + **[position embedding](#info:embedding_pos)**: each id becomes a 768-dim vector, plus a learned vector for its position: $(B, T, C)$. Then [dropout](#info:emb_dropout_notes).
3. **12 [transformer blocks](#info:highlevel_transformer_introduction_notes)**, each $(B, T, C) \\to (B, T, C)$:
   - [LayerNorm](#info:layernorm) → [causal self-attention](#info:attention_linear_notes) → add back ([residual](#info:attn_residual_notes))
   - [LayerNorm](#info:layernorm) → [MLP](#info:mlp_fc_in_notes) (Linear → [GELU](#info:gelu) → Linear) → add back ([residual](#info:mlp_residual_notes))
4. **[Final LayerNorm](#info:final_ln_notes)** (\`ln_f\`): $(B, T, C)$.
5. **[LM head](#info:lm_head_notes)**: a Linear layer to vocabulary scores (logits), $(B, T, V)$. Its weight is the token embedding matrix ([weight tying](#info:weight_tying_notes)).
6. **[Cross-entropy loss](#info:cross_entropy_notes)** against the next tokens, averaged over all $B \\times T$ positions. Training then runs [backward](#info:backward_notes) and an [optimizer](#info:optim_adamw_notes) step.

## What changed from GPT-1

GPT-1 (2018) had the same overall design: 12 layers, width 768, 117M parameters. GPT-2 changed a few things:

- **Pre-LN instead of post-LN:** LayerNorm moved to the **input** of each sublayer, \`x + attn(ln(x))\`, instead of after the residual add. This keeps a clean residual path and makes deep stacks easier to train (see [why residual connections](#info:residual_why_notes) and [improved gradient flow](#info:improves_gradient_flow_notes)).
- **A final LayerNorm** (\`ln_f\`) after the last block, needed because with pre-LN the residual stream is never normalized at the end otherwise.
- **Scaled initialization:** the weights of the layers that write into the residual stream are scaled by $1/\\sqrt{N}$, where $N$ is the number of residual layers, so the stream's variance does not grow with depth (see [variance at initialization](#info:variance_init_notes)).
- **Longer context:** 512 → 1024 tokens.
- **Bigger vocabulary:** byte-level BPE with 50257 tokens (GPT-1 used about 40,000), so any text can be encoded without an "unknown" token.
- **More data and larger models:** WebText instead of BooksCorpus, up to 1.5B parameters.

## How this site's code differs from the original

- **\`vocab_size=50304\`** instead of 50257: padded up to a multiple of 64, which makes the matrix multiplications faster on GPUs. The 47 extra tokens never appear in the data.
- **Initialization:** the code uses PyTorch's default initialization, not GPT-2's $1/\\sqrt{N}$ scaling.
- **Training:** 1000 steps on a small \`input.txt\`, with a constant learning rate. It demonstrates the training loop; it does not reproduce GPT-2 (see [Chinchilla](#info:chinchilla_notes) for the scale real training needs).

## What came after GPT-2

The next models first kept this architecture and changed the scale:

- [GPT-3](#info:gpt3_notes) (2020): the same block, 175B parameters; large models can learn a task from examples in the prompt (in-context learning).
- [Chinchilla](#info:chinchilla_notes) (2022): for a fixed compute budget, grow parameters and training tokens together, about 20 tokens per parameter.

## References

1. **Language Models are Unsupervised Multitask Learners** (GPT-2). Radford et al., 2019. [PDF ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. **Improving Language Understanding by Generative Pre-Training** (GPT-1). Radford et al., 2018. [PDF ↗](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
`,
    },
    zh: {
      title: "GPT-2：总览",
      md: `
**GPT-2**（OpenAI，2019）是一个 decoder-only 的 Transformer，只为一个任务训练：预测下一个 token。这个网站的代码实现的是它最小的版本，1.24 亿（124M）参数。这篇是总览：GPT-2 是什么、数据怎么在模型里流动、和 GPT-1 相比改了什么。每一步都链到对应的详细 note。

## GPT-2 是什么

- **Decoder-only：** 由一摞结构相同的 Transformer block 组成，用的是 [causal self-attention](#info:causal_mask_notes)：每个 token 只能看到自己和它前面的 token。没有 encoder。
- **只有一个训练目标：** 用 [cross-entropy](#info:cross_entropy_notes) loss 做 next-token prediction。不需要标注：文本里的下一个 token 就是目标。
- **训练数据 WebText：** 大约 800 万个网页（40 GB 文本），来自 Reddit 上获得至少 3 个 karma 的外链，用来粗略地筛选质量。
- **论文的观点：** 这样训练出来的语言模型，不做任何针对任务的训练，就已经能 zero-shot 完成一些任务（比如回答问题、做摘要、少量翻译），而且模型越大做得越好。[GPT-3](#info:gpt3_notes) 把这一点推进了一大步。

## 四种尺寸

| | 层数 | 宽度 $d_{\\text{model}}$ | 头数 | 参数量 |
|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 | 124M |
| GPT-2 medium | 24 | 1024 | 16 | 355M |
| GPT-2 large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

四种尺寸的上下文长度（1024 个 token）、词表（50257 个 token）和每个头的大小（$d_{\\text{model}} / \\text{头数} = 64$）都一样。论文里把最小的两个写成 117M 和 345M；按实际参数精确地数，是 124M 和 355M。

这个网站的代码是 **GPT-2 small**：\`n_layer=12, n_embd=768, n_head=12\`。

## 数据在模型里怎么流动

形状里 $B$ = batch size，$T$ = 序列长度，$C = 768$，$V$ = 词表大小（每一步的形状见 [dimension changes](#info:dimension_change_notes)）。

1. **[输入文本](#info:input_data_notes)** → **[tokenizer](#info:tokenizer_notes)**（字节级 BPE）：文本变成 token id，形状 $(B, T)$。
2. **[Token embedding](#info:embedding_tok)** + **[位置 embedding](#info:embedding_pos)**：每个 id 变成一个 768 维向量，再加上它所在位置的一个学到的向量：$(B, T, C)$。然后做 [dropout](#info:emb_dropout_notes)。
3. **12 个 [transformer block](#info:highlevel_transformer_introduction_notes)**，每个都是 $(B, T, C) \\to (B, T, C)$：
   - [LayerNorm](#info:layernorm) → [causal self-attention](#info:attention_linear_notes) → 加回去（[残差](#info:attn_residual_notes)）
   - [LayerNorm](#info:layernorm) → [MLP](#info:mlp_fc_in_notes)（Linear → [GELU](#info:gelu) → Linear）→ 加回去（[残差](#info:mlp_residual_notes)）
4. **[最后的 LayerNorm](#info:final_ln_notes)**（\`ln_f\`）：$(B, T, C)$。
5. **[LM head](#info:lm_head_notes)**：一个 Linear 层，输出每个词的分数（logits），$(B, T, V)$。它的权重就是 token embedding 矩阵（[weight tying](#info:weight_tying_notes)）。
6. **[Cross-entropy loss](#info:cross_entropy_notes)**：和下一个 token 比较，在全部 $B \\times T$ 个位置上取平均。然后训练做 [backward](#info:backward_notes)，再让 [optimizer](#info:optim_adamw_notes) 更新一步。

## 和 GPT-1 相比改了什么

GPT-1（2018）的整体设计一样：12 层、宽度 768、1.17 亿参数。GPT-2 改了这几处：

- **Pre-LN 代替 post-LN：** LayerNorm 移到每个子层的**输入**处，\`x + attn(ln(x))\`，而不是放在残差相加之后。这样残差路径保持干净，深层网络更容易训练（见 [为什么要残差连接](#info:residual_why_notes) 和 [改善梯度流动](#info:improves_gradient_flow_notes)）。
- **最后加一个 LayerNorm**（\`ln_f\`），放在最后一个 block 之后。用了 pre-LN 以后，残差流在最后不会被归一化，所以需要它。
- **缩放初始化：** 往残差流里写的那些层，权重乘以 $1/\\sqrt{N}$，$N$ 是残差层的个数，这样残差流的方差不会随深度增长（见 [初始化时的方差](#info:variance_init_notes)）。
- **上下文更长：** 512 → 1024 个 token。
- **词表更大：** 字节级 BPE，50257 个 token（GPT-1 大约 40,000 个），任何文本都能编码，不会出现"未知词"。
- **数据更多、模型更大：** 用 WebText 代替 BooksCorpus，最大到 15 亿参数。

## 这个网站的代码和原版有什么不同

- **\`vocab_size=50304\`** 而不是 50257：补到 64 的整数倍，GPU 上的矩阵乘法会更快。多出来的 47 个 token 在数据里从来不会出现。
- **初始化：** 代码用的是 PyTorch 默认的初始化，没有 GPT-2 的 $1/\\sqrt{N}$ 缩放。
- **训练：** 在一个很小的 \`input.txt\` 上跑 1000 步，学习率不变。它是用来演示训练流程的，并不能复现 GPT-2（真实训练需要多大规模，见 [Chinchilla](#info:chinchilla_notes)）。

## GPT-2 之后

接下来的模型一开始保留了这个架构，改变的是规模：

- [GPT-3](#info:gpt3_notes)（2020）：同样的 block，1750 亿参数；大模型能从 prompt 里的例子学会一个任务（in-context learning）。
- [Chinchilla](#info:chinchilla_notes)（2022）：算力固定时，参数和训练 token 应该一起增长，每个参数大约对应 20 个 token。

## 参考资料

1. **Language Models are Unsupervised Multitask Learners**（GPT-2）。Radford 等人，2019。[PDF ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. **Improving Language Understanding by Generative Pre-Training**（GPT-1）。Radford 等人，2018。[PDF ↗](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
`,
    },
  },

  highlevel_transformer_introduction_notes: {
    en: {
      title: "High-level Transformer",
      md: `
**A high-level view of a GPT-2 transformer block: first the structure in code, then an analogy for intuition.**

## Part 1: Block structure

### Where the blocks sit in the model

After the embedding step, $x$ has shape $(B, T, 768)$. It then passes through **12 transformer blocks**, one after another. The main path of \`GPT2.forward\`:

\`\`\`python
x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]   # token + position embedding
x = self.emb_drop(x)                                     # dropout (training only)
for blk in self.blocks:                                  # 12 blocks, run in order
    x = blk(x)
x = self.ln_f(x)                                         # final LayerNorm
logits = self.lm_head(x)                                 # output projection
\`\`\`

- **The 12 blocks share the same structure, but each has its own weights.** Nothing is shared between blocks.
- **Each block maps $(B, T, 768)$ to $(B, T, 768)$.** Because the shape never changes, blocks can be stacked one after another.

### What happens inside one block

\`\`\`python
class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # sublayer 1: attention
        x = x + self.mlp(self.ln2(x))    # sublayer 2: MLP
        return x
\`\`\`

Just two lines, one per sublayer:

- **Attention:** each position **looks at other positions** (only earlier ones, because of the causal mask) and gathers useful information from them.
- **MLP:** each position **processes its own vector independently**, without looking at any other position.

So a block is "communicate, then think", repeated 12 times.

> **Information moves between positions only inside attention.** The MLP always works on one position at a time.

### Two structural details

**1. Residual connections: \`x = x + ...\`**

The code is not \`x = attn(...)\` but \`x = x + attn(...)\`. A sublayer doesn't replace $x$; it **adds a small update on top of it**.

Think of $x$ as a main road that runs through all 12 layers, usually called the **residual stream**. Each sublayer reads from the stream, computes an update, and adds it back.

[Why residual connections matter: with vs. without](#info:residual_why_notes)

**2. LayerNorm before each sublayer: \`attn(self.ln1(x))\`**

The input to each sublayer first goes through a [LayerNorm](#info:layernorm), which keeps its values in a stable range. This placement is called **Pre-LN**.

> The GPT-2 paper calls this out as a change from the original Transformer. The original put LayerNorm **after** each sublayer (Post-LN). GPT-2 moves it to the **input** of each sublayer and adds one more LayerNorm after the last block: \`ln_f\` above.

## Part 2: The analogy

### Tokens = people in a meeting

Each token is like a person holding their own notes (its current representation).

### Attention = the conversation

Everyone briefly scans what the others (those earlier in the sequence) are saying and decides whose points matter for them. Each person builds a short "summary I need" by selectively listening.

> **Why it matters:** this is how long-range connections form. Facts from far back in the text can instantly influence you.

### MLP (feed-forward) = private thinking

After listening, each person goes off for a moment of inner reasoning: "Given what I heard, how should I update my notes?"

> **Why it matters:** it lets each token transform and rephrase ideas, adding nonlinearity and capacity.

### Two LayerNorms = the organizer's ground rules (applied twice)

Before the conversation and before the private thinking, the organizer reminds everyone to keep comments calm and on-topic: no one's voice too loud, none too quiet.

> **Why it matters:** it keeps both phases stable and comparable, so the block behaves predictably.

### Residual connections = a running draft

After each phase (conversation, then private thinking), you add your new insights to your previous notes rather than replacing them.

> **Why it matters:** you never lose what you already knew; updates are incremental and safe.

### Dropout = occasional silence to build resilience

Sometimes random remarks are withheld during practice meetings.

> **Why it matters:** the group learns to make good decisions even if a few voices are missing, preventing overreliance on any single cue.

### Putting it together

Each block is a two-step cycle:

1. **Communicate** (attention)
2. **Compute** (MLP)

The organizer (LayerNorm) keeps order, a running draft (residual) preserves history, and practice with partial attendance (dropout) ensures robustness. Stacking many such blocks lets tokens repeatedly share, rethink, and refine.
`,
    },
    zh: {
      title: "Transformer 概览",
      md: `
**从高层看一个 GPT-2 transformer block：先看代码里的结构，再用一个类比建立直觉。**

## 第一部分：Block 的结构

### Block 在模型里的位置

经过 embedding 之后，$x$ 的形状是 $(B, T, 768)$。接着它依次穿过 **12 个 transformer block**。\`GPT2.forward\` 的主路径：

\`\`\`python
x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]   # token + position embedding
x = self.emb_drop(x)                                     # dropout（只在训练时）
for blk in self.blocks:                                  # 12 个 block，按顺序执行
    x = blk(x)
x = self.ln_f(x)                                         # 最后的 LayerNorm
logits = self.lm_head(x)                                 # 输出投影
\`\`\`

- **12 个 block 结构相同，但各有各的权重。** block 之间不共享任何参数。
- **每个 block 都把 $(B, T, 768)$ 映射成 $(B, T, 768)$。** 形状始终不变，所以 block 可以一个接一个地叠起来。

### 一个 block 里面做了什么

\`\`\`python
class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 子层 1：attention
        x = x + self.mlp(self.ln2(x))    # 子层 2：MLP
        return x
\`\`\`

只有两行，每个子层一行：

- **Attention：** 每个位置**去看其他位置**（因为 causal mask，只能看前面的），从它们那里收集有用的信息。
- **MLP：** 每个位置**独立处理自己的向量**，不看任何其他位置。

所以一个 block 就是"先交流，再思考"，重复 12 次。

> **位置之间的信息只在 attention 里流动。** MLP 永远一次只处理一个位置。

### 两个结构细节

**1. 残差连接：\`x = x + ...\`**

代码写的不是 \`x = attn(...)\`，而是 \`x = x + attn(...)\`。子层不会替换 $x$，而是**在它上面加一个小的更新量**。

可以把 $x$ 想成一条贯穿全部 12 层的主干道，通常叫作 **residual stream（残差流）**。每个子层从残差流里读取，算出一个更新量，再加回去。

[为什么残差连接很重要：有和没有的对比](#info:residual_why_notes)

**2. 每个子层之前的 LayerNorm：\`attn(self.ln1(x))\`**

每个子层的输入都先经过一个 [LayerNorm](#info:layernorm)，把数值保持在稳定的范围内。这种放法叫作 **Pre-LN**。

> GPT-2 论文专门指出，这是相对原始 Transformer 的一处改动。原始 Transformer 把 LayerNorm 放在每个子层**之后**（Post-LN）。GPT-2 把它挪到每个子层的**输入**处，并在最后一个 block 之后多加了一个 LayerNorm：也就是上面的 \`ln_f\`。

## 第二部分：类比

### Token = 会议里的人

每个 token 就像一个拿着自己笔记（它当前的表示）的人。

### Attention = 交谈

每个人快速扫一眼其他人（序列中排在前面的人）在说什么，判断谁的观点对自己重要。每个人通过有选择地听，整理出一份"我需要的摘要"。

> **为什么重要：** 长距离的联系就是这样建立的。文本里很靠前的事实，可以立刻影响到你。

### MLP（前馈网络）= 独自思考

听完之后，每个人各自花点时间想一想："根据我听到的，我该怎么更新我的笔记？"

> **为什么重要：** 它让每个 token 能转换、重新表述想法，带来非线性和更大的容量。

### 两个 LayerNorm = 主持人的规矩（提醒两次）

在交谈之前和独自思考之前，主持人都会提醒大家发言保持冷静、不要跑题：谁的声音都不能太大，也不能太小。

> **为什么重要：** 它让两个阶段都保持稳定、可比较，整个 block 的行为就可以预期。

### 残差连接 = 不断修订的草稿

每个阶段（先交谈，再独自思考）之后，你把新想法**补充**到原来的笔记上，而不是把笔记换掉。

> **为什么重要：** 已经知道的东西永远不会丢；更新是渐进的、安全的。

### Dropout = 偶尔有人不说话，以此锻炼韧性

在练习会上，有时会随机屏蔽一些发言。

> **为什么重要：** 即使少了几个人的声音，大家也能学会做出好的判断，不会过度依赖某一条线索。

### 合起来看

每个 block 是一个两步循环：

1. **交流**（attention）
2. **计算**（MLP）

主持人（LayerNorm）维持秩序，不断修订的草稿（残差）保留历史，缺席练习（dropout）保证稳健。把许多这样的 block 叠起来，token 就能一遍遍地分享、反思、完善。
`,
    },
  },

  dimension_change_notes: {
    en: {
      title: "Dimension changes during forward",
      md: `
**How the tensor shapes change during the forward pass.**

Notation:

- $B$ = batch size
- $T$ = sequence length
- $C$ = $n\\_embd$
- $V$ = \`vocab_size\`

## Inputs and helpers

\`\`\`text
idx                          (B, T)     Long tensor of token IDs
pos = arange(0, T)           (T)
\`\`\`

## Embeddings and sum

\`\`\`text
tok_emb(idx)                 (B, T, C)
pos_emb(pos)                 (T, C)
pos_emb(pos)[None, :, :]     (1, T, C)  broadcasts in the sum
x = tok_emb(idx) + pos_emb(pos)[None, :, :]
                             (B, T, C)
\`\`\`

## Dropout and blocks

\`\`\`text
x = emb_drop(x)              (B, T, C)  same shape; values randomly zeroed in train mode
for blk in blocks: x = blk(x)
                             (B, T, C)
\`\`\`

## Final norm and head

\`\`\`text
x = ln_f(x)                  (B, T, C)
logits = lm_head(x)          (B, T, V)
\`\`\`

## Targets and loss (if provided)

\`\`\`text
targets                              (B, T)
logits.view(-1, logits.size(-1))     (B*T, V)
targets.view(-1)                     (B*T)
loss = F.cross_entropy((B*T, V), (B*T))
                                     scalar ()  a single number
\`\`\`

## Return

- \`logits\`: $(B, T, V)$
- \`loss\`: a scalar, or \`None\`
`,
    },
    zh: {
      title: "前向传播中的维度变化",
      md: `
**前向传播过程中张量形状如何变化。**

记号：

- $B$ = batch size
- $T$ = 序列长度
- $C$ = $n\\_embd$
- $V$ = \`vocab_size\`

## 输入和辅助量

\`\`\`text
idx                          (B, T)     Long tensor of token IDs
pos = arange(0, T)           (T)
\`\`\`

## Embedding 和求和

\`\`\`text
tok_emb(idx)                 (B, T, C)
pos_emb(pos)                 (T, C)
pos_emb(pos)[None, :, :]     (1, T, C)  broadcasts in the sum
x = tok_emb(idx) + pos_emb(pos)[None, :, :]
                             (B, T, C)
\`\`\`

## Dropout 和 block

\`\`\`text
x = emb_drop(x)              (B, T, C)  same shape; values randomly zeroed in train mode
for blk in blocks: x = blk(x)
                             (B, T, C)
\`\`\`

## 最后的 norm 和 head

\`\`\`text
x = ln_f(x)                  (B, T, C)
logits = lm_head(x)          (B, T, V)
\`\`\`

## Targets 和 loss（如果提供了）

\`\`\`text
targets                              (B, T)
logits.view(-1, logits.size(-1))     (B*T, V)
targets.view(-1)                     (B*T)
loss = F.cross_entropy((B*T, V), (B*T))
                                     scalar ()  a single number
\`\`\`

## 返回值

- \`logits\`：$(B, T, V)$
- \`loss\`：一个标量，或 \`None\`
`,
    },
  },
};
