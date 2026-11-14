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

  attention_output_linear_notes: {
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

  proj_residual_stream_notes: {
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

  variance_init_notes: {
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
  resid_dropout_notes: {
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

MLP 分支在同样的位置也有 residual dropout：\`x = x + mlp(ln2(x))\`，其中 MLP 的输出在相加之前也要经过 dropout。见 [Residual Dropout in MLP](#info:mlp_dropout_notes)。

## p 怎么选

- GPT-2 用 $p = 0.1$。
- 🆕 很多近期的 LLM（例如 LLaMA）预训练时这里的 dropout 也设成 0：训练数据量巨大，而且通常每条只看一遍左右，过拟合不是主要问题。
`,
  },
  mlp_fc_in_notes: {
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
  mlp_fc_out_notes: {
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
  linear_weight_notes: {
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
  vector_1d_notes: {
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
  gelu: {
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

![GELU 曲线（−3.5 到 3.5），虚线是 y = x](/gpt2-explorer/notes/gelu.svg)

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

ReLU 的定义是：

\`\`\`text
ReLU(x) = max(0, x)
\`\`\`

它的梯度：

- $x > 0$ 时：1（梯度原样传过去）；
- $x < 0$ 时：0（梯度断掉）。

这让优化很简单，也避免了早期深度网络里 sigmoid、tanh 带来的梯度消失。

但它也带来了 **"dying ReLU"** 问题：如果一个神经元的权重让它的输入太经常是负数，它的梯度就一直是 0，于是这个神经元"死掉"，再也不会更新。

> GELU 不会直接丢掉负数，而是用正态分布的 CDF 把它们平滑地缩小。
`,
  },
  cdf: {
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

![标准正态分布 PDF 的钟形曲线，标出了 φ(0)、φ(1)、φ(2)](/gpt2-explorer/notes/normal_pdf.svg)

**PDF 的公式：**

\`\`\`text
φ(x) = 1/√(2π) × e^(−x²/2)
\`\`\`

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

\`\`\`text
φ(x) = 1/(σ√(2π)) × e^(−(x−μ)²/(2σ²))
\`\`\`

- **x − μ**：把中心从 0 移到 μ。
- **除以 σ²**：σ 越大，曲线越宽。
- **前面的 1/σ**：曲线变宽后相应地压矮，让面积仍然是 1。

取 μ = 0、σ = 1，就回到标准正态分布的公式。

### 3. CDF：x 左边的面积

CDF（累积分布函数，记作 Φ(x)）回答的问题是：

> 随机抽一个数，它**小于 x** 的概率是多少？

在图上，就是钟形曲线下方、**x 左边那部分的面积**。整条曲线下的总面积是 1。

Φ(1) 就是涂色部分的面积：

![PDF 曲线下 x = 1 左边的面积被涂色：Φ(1) ≈ 0.841](/gpt2-explorer/notes/normal_cdf_area.svg)

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

![标准正态分布 CDF Φ(x) 的 S 形曲线](/gpt2-explorer/notes/normal_cdf_curve.svg)

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

\`\`\`text
Φ(x) = ∫ from −∞ to x of  1/√(2π) × e^(−t²/2) dt
\`\`\`

这是准确的公式，但它包含积分：要算 Φ(1)，就得真的去求一块面积。

**2. 简单的"闭式"公式：没有。** 闭式是指只用加减乘除、幂、根号、eˣ、log、sin 这类基本函数，把 Φ(x) 直接写出来。数学上已经**证明**，e^(−t²/2) 的积分做不到这一点。这不是还没人找到，而是不可能存在。

**3. 所以给它起了名字：erf。** 既然写不成基本函数，数学家就把这个积分定义成一个新函数，叫误差函数 erf：

\`\`\`text
erf(z) = 2/√π × ∫ from 0 to z of  e^(−t²) dt

Φ(x) = 0.5 × (1 + erf(x / √2))
\`\`\`

这就像 sin(x)：它其实也不能用加减乘除写出来，只是有了名字，计算机里有现成的实现。\`torch.erf\`、\`math.erf\` 直接就能给出精确值：

\`\`\`python
0.5 * (1 + math.erf(1 / math.sqrt(2)))    # 0.8413 = Φ(1)
\`\`\`

计算机内部用级数或分段多项式逼近来算 erf，精度可以达到浮点数的极限。

**4. 近似公式：有很多。** GPT-2 用的 GELU tanh 版本，其实就是**用 tanh 近似 Φ(x)**：

\`\`\`text
Φ(x) ≈ 0.5 × (1 + tanh( √(2/π) × (x + 0.044715 × x³) ))
\`\`\`

两边各乘以 x，就得到了 GELU 的 tanh 近似公式。

| 问题 | 答案 |
| --- | --- |
| 有没有准确的定义？ | 有：PDF 的积分 |
| 能不能只用基本函数写出来？ | 不能，数学上已证明 |
| 实际怎么算？ | 用 erf：Φ(x) = 0.5 × (1 + erf(x/√2)) |
| 有没有不用 erf 的写法？ | 有近似公式，比如 GELU 里的 tanh 版 |
`,
  },
};
