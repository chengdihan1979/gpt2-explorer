// Notes: GPT-2 overview and inference.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  gpt2_notes: {
    en: {
      title: "GPT-2",
      md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  `,
    },
    zh: {
      title: "GPT-2",
      md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  `,
    },
  },

  gpt2_inference_notes: {
    en: {
      title: "GPT-2 inference",
      md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  
    `,
    },
    zh: {
      title: "GPT-2 推理",
      md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  
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
