// Chinese versions of the notes, keyed like INFO. A note missing here falls back to English.
// Each entry: { title, md }. Keep technical terms (and code) in English.
export const INFO_ZH = {
  register_buffer_notes: {
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
  state_dict_notes: {
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

  state_dict_build_notes: {
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

  softmax_notes: {
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

  attn_dropout_notes: {
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

  att_v_product_notes: {
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

  matmul_example_notes: {
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

  reorder_merge_heads_notes: {
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

  contiguous_notes: {
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
};
