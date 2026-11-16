// Chinese versions of the notes, keyed like INFO. A note missing here falls back to English.
// Each entry: { title, md }. Keep technical terms (and code) in English.
const prevents_vanishing_exploding_md_zh = `
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
`;

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

MLP 分支在同样的位置也有 residual dropout：\`x = x + mlp(ln2(x))\`，其中 MLP 的输出在相加之前也要经过 dropout。见 [MLP 里的 residual dropout](#info:mlp_dropout_notes)。

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
  erf_derivation_notes: {
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
  gelu_backprop_notes: {
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
  mlp_dropout_notes: {
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
  mlp_residual_notes: {
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
  attn_residual_notes: {
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
  layernorm: {
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
  layernorm_params_notes: {
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
  residual_why_notes: {
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
  layernorm_scale_notes: {
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
  stabilize_training_notes: {
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
  improves_gradient_flow_notes: {
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
  post_norm_gradient_example_notes: {
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
  rmsnorm_gradient_example_notes: {
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
  stabilize_training_no_layernorm_backprop_notes: {
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
  ln_jacobian_closed_form_proof_notes: {
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
  prevents_vanishing_exploding_notes: {
    title: "防止梯度消失/爆炸",
    md: prevents_vanishing_exploding_md_zh,
  },
  mlp_prevents_vanishing_exploding_notes: {
    title: "防止梯度消失/爆炸",
    md: prevents_vanishing_exploding_md_zh,
  },
  gradient_loss_to_logits_proof_notes: {
    title: "loss 对 logits 的梯度",
    md: `
**证明：softmax cross-entropy 对 logits 的梯度是 $p - y$。**

## 设定

- Logits：$z \\in \\mathbb{R}^C$
- Softmax：$p_i = \\dfrac{e^{z_i}}{\\sum_{k=1}^C e^{z_k}}$
- 目标：$y$（通常是 one-hot，但也可以是任意概率分布，比如 label smoothing 时）
- Cross-entropy loss：$L = -\\sum_{i=1}^C y_i \\log p_i$

## 证明

利用恒等式

$$
\\log p_i = z_i - \\log \\sum_{k=1}^C e^{z_k}
$$

（对 softmax 的定义两边取 log 就得到它。）代入 loss：

$$
L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i \\left(z_i - \\log \\sum_{k=1}^C e^{z_k}\\right) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
$$

因为 $y$ 是概率分布，$\\sum_{i=1}^C y_i = 1$，所以

$$
L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
$$

对 $z_j$ 求导（第一项里只有 $i = j$ 那一项含 $z_j$；第二项用 $\\log$ 的求导法则，$\\log u$ 的导数是 $u'/u$）：

$$
\\frac{\\partial L}{\\partial z_j} = -y_j + \\frac{e^{z_j}}{\\sum_{k=1}^C e^{z_k}} = p_j - y_j
$$
`,
  },
  label_smoothing_notes: {
    title: "Label Smoothing",
    md: `
**Label smoothing 把 one-hot 目标换成一个"软"目标，这样即使模型非常有把握，梯度 $\\nabla_z L = p - y$ 也永远不会正好变成 0。**

## 为什么 one-hot 目标会让梯度塌缩

用标准的（one-hot）cross-entropy 时，如果模型预测对了而且已经饱和：

$$
p \\approx [1, 0, \\ldots] \\text{ 且 } y = [1, 0, \\ldots] \\Rightarrow p - y \\approx [0, 0, \\ldots]
$$

梯度接近 0（塌缩）。

## 平滑后的目标

用 label smoothing 系数 $\\epsilon$，把正确类别 $c$ 和其他 $C - 1$ 个类别的目标设成

$$
y_c^{LS} = 1 - \\epsilon, \\qquad y_{j \\neq c}^{LS} = \\frac{\\epsilon}{C-1}
$$

这样即使 $p \\approx [1, 0, \\ldots]$，也有

$$
\\nabla_z L = p - y^{LS} \\approx \\left[\\epsilon, -\\frac{\\epsilon}{C-1}, \\ldots\\right]
$$

它不是 0，所以 logits 仍然能得到有用的更新。

> 直观地说，label smoothing 会把最高的 logit 往下压一点，把其他 logits 往上推一点，从而减少过度自信。这样模型已经很有把握时，梯度也不会完全停掉，训练能继续进行，也减少了脆弱的过拟合。

## 小例子（$C = 3$，$\\epsilon = 0.1$）

- 平滑后的目标：$y^{LS} = [0.9, 0.05, 0.05]$。
- 如果 $p = [0.999, 0.0005, 0.0005]$（过度自信），那么

$$
\\nabla_z L = [0.999 - 0.9,\\; 0.0005 - 0.05,\\; 0.0005 - 0.05] = [0.099, -0.0495, -0.0495]
$$

不是 0，所以权重仍然会被更新。

## 参考资料

1. **原始论文**

   Rethinking the Inception Architecture for Computer Vision（Inception-v3）。提出了 label smoothing 作为一种正则化方法，并说明了动机。

   [Rethinking the Inception Architecture for Computer Vision ↗](https://arxiv.org/pdf/1512.00567)

2. **Transformer 训练细节**

   Attention Is All You Need。明确写了训练时使用 $\\epsilon = 0.1$ 的 label smoothing，并说明了它的效果。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

3. **理论和实验研究**

   When Does Label Smoothing Help?（Müller、Kornblith、Hinton，NeurIPS 2019）。研究了它对泛化和校准的影响，以及为什么它会损害知识蒸馏。

   [When Does Label Smoothing Help? ↗](https://arxiv.org/pdf/1906.02629)

4. **相关的正则化方法（confidence penalty）以及它和 LS 的关系**

   Pereyra 等人，ICLR 2017 workshop。惩罚低熵（过于自信）的输出；通过 KL 散度的方向说明了它和 label smoothing 的关系。

   [Regularizing Neural Networks by Penalizing Confident Output Distributions ↗](https://arxiv.org/pdf/1701.06548)

5. **校准（calibration）的背景**

   Guo 等人，On Calibration of Modern Neural Networks（ICML 2017）。说明了现代神经网络往往过度自信（给出的置信度高于实际准确率），并提出用 temperature scaling 修正。这是理解为什么要减少过度自信（label smoothing 也在做这件事）的背景资料。

   [On Calibration of Modern Neural Networks ↗](https://arxiv.org/pdf/1706.04599)
`,
  },
  final_ln_notes: {
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
  lm_head_notes: {
    title: "lm_head",
    md: `
**\`lm_head\` 在每个位置，给词表里的每个词打一个分（logit）。**

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)   # 768 → 50304
logits = self.lm_head(x)                                   # x 来自 ln_f
\`\`\`

## 权重矩阵：每个词一行

\`lm_head.weight\` 的形状是 **(50304, 768)**：词表里每个词一行，每行是一个 768 维向量。因为 [weight tying](#hover:weight_tying_notes)，它和 token embedding 表 \`tok_emb.weight\` 是**同一个**矩阵。

## 一个 logit = 一次点积

对每个位置，\`lm_head\` 把 $x$（\`ln_f\` 的输出）和每一行做点积：

\`\`\`text
logits[word i] = x · W[i]  =  Σ_j  x[j] × W[i][j]      (i = 0 … 50303, j = 0 … 767)
\`\`\`

一个词的向量和 $x$ 方向越接近，点积越大，分数就越高。所以 \`lm_head\` 本质上是在问：**$x$ 最像哪个词？**

## 形状

\`\`\`text
x:       (B, T, 768)
W:       (50304, 768)
logits:  (B, T, 50304)      # 每个位置，对每个词都有一个分数
\`\`\`

**每个位置**都有自己的 50304 个分数。位置 t 的 logits 用来预测第 t+1 个 token。

- **训练时**，这 T 组 logits 全部进入 [cross-entropy](#hover:cross_entropy_notes) loss，各自和真实的下一个 token 比较。整个 batch 所有位置一次批量矩阵乘法算完。
- **生成时**，只用最后一个位置的 logits 来选下一个 token。

> [为什么和词向量做点积就能预测下一个 token？更深入的解释](#hover:lm_head_insights_notes)

## 细节

- **参数量：** 768 × 50304 = 38,633,472，约 3860 万，大约占 GPT-2 small（约 1.24 亿）的三成。因为 weight tying，这些参数只存一份。
- **为什么是 50304 而不是 GPT-2 原本的 50257？** 这是 nanoGPT 的做法：把词表补到 64 的倍数（50304 = 64 × 786），GPU 矩阵乘法会更快。多出来的 47 个"词"永远不会出现在训练数据里，训练会把它们的 logits 压低。
- **为什么 \`bias=False\`？**
  1. *Weight tying。* 输出权重就是输入的 embedding 表（\`self.lm_head.weight = self.tok_emb.weight\`）。embedding 没有 bias，也不需要单独的输出 bias。
  2. *收益很小。* bias 会多出 \`vocab_size\` 个参数，几乎没有好处，所以 logits 就是纯点积。
`,
  },
  weight_tying_notes: {
    title: "Weight Tying（权重共享）",
    md: `
**Weight tying（输入和输出共用同一张 embedding 表）**

Weight tying 让输出 softmax 的权重等于输入的 token embedding 权重：

\`\`\`python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# tie lm_head weight to tok_emb    (把 lm_head 的权重绑到 tok_emb 上)
self.lm_head.weight = self.tok_emb.weight
\`\`\`

> 这个想法来自论文 [Using the Output Embedding to Improve Language Models ↗](https://arxiv.org/abs/1608.05859)

## 它做了什么

最后一个 block 之后，每个位置都有一个最终向量 $h \\in \\mathbb{R}^{n\\_embd}$。\`lm_head\` 拿 $h$ 和词表里每个词的那一行做点积，给每个词打一个分（logit）：

$$
\\text{logit}_i = h \\cdot W[i]
$$

$h$ 和第 $i$ 行的方向越接近，词 $i$ 的分数就越高。共享之后，第 $i$ 行同时扮演两个角色：

- **输入端：** 读入词 $i$ 时，用来表示它的向量。
- **输出端：** 预测下一个词是不是 $i$ 时，模型拿来比对的模板。

> 要预测 \`Ġcat\`，模型就得把 $h$ 推向 \`Ġcat\` 自己的 embedding。

## 为什么这么做

- **参数更少：** 两个矩阵 \`V*d\` + \`d*V\` 变成一个 \`V*d\`，大约少了一半。对 GPT-2 small 来说省了 $50{,}257 \\times 768 \\approx 38.6\\text{M}$ 个参数：模型约 124M，而不是约 163M。
- **泛化更好：** 起到有益的正则化作用，通常能降低 perplexity。

## 梯度怎么流

这行赋值并没有复制任何东西：两个层指向的是**同一个 \`Parameter\` 对象**，所以这个共享矩阵收到的是两处用法梯度的**和**。

用一个极简的例子看清楚每一份梯度从哪来。去掉所有 block，让 embedding 直接接 \`lm_head\`；词表只有 3 个词，\`n_embd\` = 4：

\`\`\`text
E = [[ 0.5, 0,   0.5, 0  ],    # word 0
     [ 0,   0.5, 0,   0.5],    # word 1
     [-0.5, 0,   0.5, 0  ]]    # word 2

输入 token = word 0   →  x = E[0] = [0.5, 0, 0.5, 0]     ← 输入端用了 E
logits = x @ E.T = [0.5, 0, 0]                           ← 输出端又用了 E
p = softmax(logits) = [0.452, 0.274, 0.274]
正确的下一个 token = word 2，loss = 1.294
\`\`\`

loss 对 logits 的梯度是 $p - y$（[为什么？](#hover:cross_entropy_notes)）：

\`\`\`text
p − y = [0.452, 0.274, 0.274] − [0, 0, 1] = [0.452, 0.274, -0.726]
\`\`\`

### 输出端：每一行都有梯度

因为 $\\text{logit}_i = x \\cdot E[i]$，第 $i$ 行收到的梯度是 $(p_i - y_i)\\, x$：

\`\`\`text
row 0:   0.452 × [0.5, 0, 0.5, 0] = [ 0.226, 0,  0.226, 0]
row 1:   0.274 × [0.5, 0, 0.5, 0] = [ 0.137, 0,  0.137, 0]
row 2:  -0.726 × [0.5, 0, 0.5, 0] = [-0.363, 0, -0.363, 0]
\`\`\`

更新时减去梯度。row 2（正确答案）的梯度是负的，所以它**朝 $x$ 的方向移动**；row 0、row 1 是错误答案，**被推离 $x$**。

### 输入端：只有被查的那一行

$x$ 是查表查出的 row 0，所以这份梯度只加到 row 0。$x$ 影响了每个 logit，所以它的梯度是所有词向量按 $p_i - y_i$ 加权求和：

\`\`\`text
dL/dx = 0.452 × E[0] + 0.274 × E[1] − 0.726 × E[2]
      = [0.589, 0.137, -0.137, 0.137]       → 只加到 row 0
row 1、row 2：0
\`\`\`

### 两份相加

\`\`\`text
         输出端                   输入端                           合计 (E.grad)
row 0:  [ 0.226, 0, 0.226, 0] + [0.589, 0.137, -0.137, 0.137] = [ 0.815, 0.137, 0.089, 0.137]
row 1:  [ 0.137, 0, 0.137, 0] + 0                              = [ 0.137, 0,     0.137, 0    ]
row 2:  [-0.363, 0,-0.363, 0] + 0                              = [-0.363, 0,    -0.363, 0    ]
\`\`\`

在 PyTorch 里验证过：分别只让一端用 $E$ 算出梯度，两份加起来，正好等于共享版本的 \`E.grad\`。

### 对训练意味着什么

- **输入端：** 只有在 batch 里出现过的 token，对应的行才有梯度。
- **输出端：** softmax 每一步都给词表的**每一行**梯度（这个网站的代码里是 50,304 行）。错误词被推开的力度和它的概率 $p_i$ 成正比。

所以共享之后，即使是罕见词的 embedding，也能通过输出端一直得到训练。

## 补充

- 很多 GPT-2 实现在共享权重时，\`lm_head\` 都用 \`bias=False\`。
- 共享要求形状完全相同：\`lm_head.weight.shape == tok_emb.weight.shape == (vocab_size, n_embd)\`。
- 不需要转置形状就能对上，是因为 \`nn.Linear(n_embd, vocab_size)\` 把权重存成 \`(out, in)\` = \`(vocab_size, n_embd)\`，和 \`nn.Embedding(vocab_size, n_embd)\` 的布局一样。
- 原版 GPT-2 也共享权重：OpenAI 的代码直接用 token embedding 表 \`wte\` 算 logits：\`logits = tf.matmul(h_flat, wte, transpose_b=True)\`。

## 🆕 近期发展

共享是一种取舍。输入端的那一行应该表达"这个词是什么"；输出端的那一行应该表达"什么样的上下文后面会接这个词"。强迫一个向量同时做两件事，是一种约束。

- **大模型通常不共享：** Llama 3.1 8B 和 70B，Qwen2.5 7B 和 72B。在这个规模下，embedding 表只占参数的一小部分，省下的不多。
- **小模型经常共享：** Llama 3.2 1B 和 3B，Qwen2.5 0.5B 和 3B，Gemma 2。这些模型里 embedding 表占参数的比例很大。

> 在 Hugging Face 的模型 config 里，对应的是 \`tie_word_embeddings\` 这个开关。

## 参考资料

1. **Using the Output Embedding to Improve Language Models**

   Press & Wolf，2016。证明了共享输入和输出 embedding 能减少参数并降低 perplexity。

   [arXiv:1608.05859 ↗](https://arxiv.org/abs/1608.05859)

2. **Tying Word Vectors and Word Classifiers**

   Inan、Khosravi & Socher，2016。从语言建模的一个 loss 框架出发，独立提出了同样的想法。

   [arXiv:1611.01462 ↗](https://arxiv.org/abs/1611.01462)

3. **GPT-2 参考模型代码**

   OpenAI 原版的 model.py，里面用 \`wte\` 计算 logits。

   [openai/gpt-2 model.py ↗](https://github.com/openai/gpt-2/blob/master/src/model.py)
`,
  },
  lm_head_insights_notes: {
    title: "lm_head 深入解释",
    md: `
**为什么和输出 embedding 做点积，就能预测下一个 token？**

## 设定

- $x \\in \\mathbb{R}^{n\\_embd}$ 是某个位置上 token 的最终 hidden state。
- $e_v \\in \\mathbb{R}^{n\\_embd}$ 是词表中 token $v$ 的输出权重向量（因为 weight tying，它同时也是 $v$ 的输入 embedding）。

首先要理解训练达成了什么：

1. **hidden state $x$** 汇总了当前位置及之前所有 token 的上下文，同时也编码了下一个 token 应该具备的特征。

   例如输入是 "The dogs that barked loudly"。训练之后，第 5 个位置的 hidden state 从位置 1 到 5 的 token 中学到了"主语是复数"这个上下文，同时也编码了"下一个 token 很可能是动词"这个特征。

2. **$e_v$ 是 token $v$ 的一个全局分类器权重向量：** 应该选 $v$ 的上下文，会和这个方向对齐。它与上下文无关（不会随时变化），是从整个语料中学到的 token $v$ 的一般含义。

3. **cross-entropy loss 同时塑造两边。** 它推动：
   - hidden state $x$ 去对齐正确 token 的 $e_v$；
   - $e_v$ 朝着"最能把 $v$ 是正确答案的上下文和其他上下文区分开"的方向移动。久而久之，$v$ 是正确答案的那些上下文，就会落在 $e_v$ 附近。

记住这几点，我们继续。

## 点积形式

LM head 用 $x$ 和 $e_v$ 的点积，给 token $v$ 一个 logit $\\ell_v$：

$$
\\ell_v = x \\cdot e_v
$$

## 余弦形式

根据 $\\mathbb{R}^{d}$ 中余弦的定义，还可以写成：

$$
\\ell_v = x \\cdot e_v = \\lVert x \\rVert \\, \\lVert e_v \\rVert \\, \\cos\\theta(x, e_v)
$$

### 三个因子

所以每个 token 的 logit $\\ell_v$ 可以拆成三部分：

1. $x$ 的大小；
2. $e_v$ 的大小；
3. $e_v$ 的方向，它决定了 $x$ 和 $e_v$ 之间随上下文变化的余弦相似度。

## 最后一个 LayerNorm 的作用

有了最后的 layer norm（\`ln_f\`），$x$ 的大小几乎是常数，只在很小的范围内变化。于是另外两个因子，$e_v$ 的大小和余弦相似度，成了决定 logit 的主要因素。

> 粗略地说，可以把因子 3（**$e_v$ 的方向**）看作捕捉上下文，把因子 2（**$e_v$ 的大小**）看作捕捉这个 token 的出现频率。

- **$e_v$ 的方向：** $x$ 的方向越接近 $e_v$，第三个因子越大，对 logit 的贡献也越大。从语义上说，这意味着 token $v$ 和 hidden state $x$ 所包含的上下文更相符，所以会以较高的概率被预测出来。
- **$e_v$ 的大小：** 大语言模型（LLM）的 token embedding 矩阵与 token 频率相关：常见 token 的 embedding 范数一般更大，因此这些 token 的预测概率也更高。

## 为什么常见 token 的范数更大？

下面是一个直观的解释。

训练时我们最小化 loss $\\mathcal{L}$，参数按下式更新：

$$
\\Delta\\theta = -\\eta \\, \\nabla_\\theta \\mathcal{L}
$$

和某个 token $v$ 绑定的参数（它的向量 $e_v$，如果有 bias 还包括 $b_v$），只在这个 token 参与 loss 的那些步里得到梯度。

### LM head 向量的梯度（无 bias 的 head）

$$
\\ell = x E^\\top, \\quad p = \\mathrm{softmax}(\\ell), \\quad \\frac{\\partial \\mathcal{L}}{\\partial e_v} = \\big(p_v(x) - \\mathbf{1}\\{y = v\\}\\big) \\, x
$$

在整个数据集上：

$$
\\nabla_{e_v} \\mathcal{L} = \\sum_{t} \\big(p_v(x_t) - \\mathbf{1}\\{y_t = v\\}\\big) \\, x_t
$$

### 常见 token "得到更多梯度"

1. 如果 token $v$ 作为目标出现了 $C_v$ 次，就有 $C_v$ 个强项（$-\\mathbf{1}\\{y_t = v\\}$ 那部分）把 $e_v$ 拉向 $v$ 之前的 hidden state $x$。
2. 在其他位置，$e_v$ 通过 $p_v(x_t)$ 参与，当 $v$ 不是目标时，产生较小的"推开"项。
3. 经过大量样本，第一个效果是让 $e_v$ 的方向和 hidden state $x$ 对齐。
4. 方向基本对齐之后，增大 $e_v$ 的大小就成了一种高效的办法，能在许多次出现中一起抬高 $v$ 的 logit，相当于一个全局增益或先验。常见 token 提供的这类更新更多，所以范数往往更大。

> **经验法则：** 出现次数越多 $\\Rightarrow$ 梯度贡献越多（越大）$\\Rightarrow$ 这个 token 的参数被调整得越充分。

## 最后

- **token embedding $e_v$**（来自 \`wte.weight\`）与上下文无关。它捕捉的是 token 类型层面的倾向：从这个 token 在整个语料中出现的方式里，学到的大致、平均的语义和句法（比如 "dogs" 是复数、偏名词、常出现在 "are/were" 这类动词附近）。可以把它看作一个基准原型。
- **它在某句话里的具体含义**，也就是"上下文中的 token"，存在 hidden state $x_t$ 里，这是 token 经过位置编码 + self-attention + MLP 之后得到的。这就是 LM head 用来选下一个 token 的**上下文化表示**。

> 所以 embedding 承载的是**一般含义**，而不是**随时变化的上下文**。上下文由 transformer 注入，体现在 $x_t$ 中，而不是 $e_v$ 中。
`,
  },
  cross_entropy_notes: {
    title: "Cross Entropy（交叉熵）",
    md: `
**Cross-entropy loss** 衡量模型给正确答案分了多少概率。GPT-2 在每个位置上，对真实的下一个 token 算一次，再把所有位置平均成一个数。

文档：[CrossEntropy ↗](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

## 贯穿全文的例子

取一个只有 3 个词的词表。某个位置上模型输出的 logits 如下，正确的下一个词是 word 2：

\`\`\`text
logits z = [0.5, 0, 0]
正确答案 t = 2
\`\`\`

## 第 1 步：logits → 概率（softmax）

softmax 对每个 logit 取 exp，再除以总和，得到的结果都是正数、加起来等于 1：

\`\`\`text
exp(0.5) = 1.6487
exp(0)   = 1
exp(0)   = 1
总和 S   = 1.6487 + 1 + 1 = 3.6487

p = [1.6487 / S, 1 / S, 1 / S]
  = [0.452, 0.274, 0.274]
\`\`\`

## 第 2 步：loss = −log(正确答案的概率)

$$
\\text{loss} = -\\log p_t
$$

公式里只出现正确答案的概率。这里 $t = 2$：

\`\`\`text
p[2] = 0.27407
loss = −ln(0.27407) = 1.2944
\`\`\`

这里的 log 是自然对数（底数 $e$），PyTorch 用的也是它。

错误的词也会间接起作用：softmax 要除以总和 $S$，所以错误词的 logit 越高，$p_t$ 就越小，loss 就越大。

### 用 one-hot 写同一个公式

把正确答案写成 one-hot 向量 $y$：正确的词是 1，其他都是 0。这里 $y = [0, 0, 1]$。于是

$$
\\text{loss} = -\\sum_i y_i \\log p_i
$$

$y_i = 0$ 的项全部消失，只剩 $-\\log p_t$，就是上面的公式。

> 实际中（比如 GPT-2），target 传的是类别索引（这里是 2），不是 one-hot 向量。结果一样，但省内存、省计算得多。

## 这些数字是什么意思

| 正确答案的概率 | loss |
|---|---|
| 1.0（完全确定且答对） | 0 |
| 0.5 | 0.693 |
| 1/3（3 个词里均匀乱猜） | 1.099 |
| 0.274（这个例子） | 1.294 |
| 0.01 | 4.605 |

这个例子的 loss（1.294）比均匀乱猜（1.099）还高：模型把最大的 logit 给了 word 0，而不是正确答案 word 2。

在 $V$ 个词里均匀乱猜时 $p_t = 1/V$，loss 就是 $\\ln V$。对 GPT-2 来说，$\\ln 50{,}304 \\approx 10.83$。如果模型初始的 logits 都很小（比如 GPT-2 用 $\\mathcal{N}(0, 0.02)$ 初始化），训练刚开始时 loss 就在这个值附近。

## 直接从 logits 算

把 softmax 代进 loss：

$$
\\text{loss} = -\\log \\frac{e^{z_t}}{S} = \\log S - z_t, \\qquad S = \\sum_j e^{z_j}
$$

代入例子：$\\ln 3.6487 - 0 = 1.2944$，结果一样。

PyTorch 的 \`F.cross_entropy\` 接收原始 logits，按这种形式（log-sum-exp）计算，不会先算出 $p$ 再取 log。很小的概率在浮点数里可能被舍入成 0，而 $\\log 0 = -\\infty$；log-sum-exp 的写法避开了这个问题。所以模型传给 \`F.cross_entropy\` 的是 logits，不是概率。

## 对 logits 的梯度：$p - y$

反向传播就是从这个梯度开始的，所以值得推一遍。从 $\\text{loss} = \\log S - z_t$ 出发，对某一个 logit $z_i$ 分别求两项的导数。

**第一项 $\\log S$。** 用链式法则：$\\log$ 的导数是 $1/S$；$S$ 里只有 $e^{z_i}$ 这一项含 $z_i$：

$$
\\frac{\\partial \\log S}{\\partial z_i} = \\frac{1}{S} \\cdot e^{z_i} = p_i
$$

结果正好是 softmax 的定义。

**第二项 $-z_t$。** 它只和正确答案的 logit 有关：

$$
\\frac{\\partial (-z_t)}{\\partial z_i} = \\begin{cases} -1 & i = t \\\\ 0 & i \\ne t \\end{cases} \\;=\\; -y_i
$$

**两项相加：**

$$
\\frac{\\partial\\, \\text{loss}}{\\partial z_i} = p_i - y_i
$$

代入例子：

\`\`\`text
p     = [0.452, 0.274,  0.274]
y     = [0,     0,      1    ]
p − y = [0.452, 0.274, -0.726]
\`\`\`

在 PyTorch 里用两种方法核对：一是 autograd，二是把每个 logit 加 0.001，看 loss 变化了多少。

\`\`\`text
autograd z.grad                  = [ 0.4519,  0.2741, -0.7259]
logit 0 加 0.001：变化 / 0.001   =   0.4518
logit 1 加 0.001：变化 / 0.001   =   0.2741
logit 2 加 0.001：变化 / 0.001   =  -0.7260
\`\`\`

这个梯度的作用（更新时减去梯度）：

- **错误的词**梯度是 $+p_i$，所以 logit **下降**。错误词占的概率越多，被压得越狠。
- **正确的词**梯度是 $p_t - 1$，总是负数，所以 logit **上升**。$p_t$ 离 1 越远，往上推得越狠。
- 当 $p = y$（正确答案概率为 1）时，梯度为 0，不再更新。

梯度这么简单，是因为 softmax 里的 $\\exp$ 和 cross-entropy 里的 $\\log$ 求导时互相抵消了。这也是 PyTorch 把 softmax 和 cross-entropy 合成一个函数的原因之一。

## 在 GPT-2 里：从 $(B, T, V)$ 的 logits 到一个数

模型用一行代码算 loss：

\`\`\`python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
\`\`\`

### targets 就是输入往后错一位

每个位置的正确答案是紧跟在它后面的那个 token。\`get_batch\` 就是这样构造 \`y\` 的：

\`\`\`text
一段 token：  [464, 3290, 3332, 319, 262]     （编号是随便写的）

idx (x)     = [464, 3290, 3332, 319]     喂给模型
targets (y) = [3290, 3332, 319, 262]     每个位置的下一个 token
\`\`\`

所以形状是：

- **Logits**（softmax 之前的原始分数）：$(B, T, V)$
- **Targets**（真实的 token 编号，不是 one-hot）：$(B, T)$

其中 $B$ 是 batch size，$T$ 是序列长度，$V$ 是词表大小。

### 展平：\`.view(-1, V)\` 和 \`.view(-1)\`

\`F.cross_entropy\` 要求：

- **Input：** 2 维张量 $(N, C)$：$N$ 个样本，每个样本对 $C$ 个类别打分。
- **Target：** 1 维张量 $(N,)$，里面是整数类别索引。

它不认识 $B$ 和 $T$，所以要把这两个维度合成一个：

- \`logits.view(-1, V)\`：$(B, T, V) \\to (B \\times T, V)$。每一行是一个位置 $(b, t)$：模型在这里对整个词表的打分。
- \`targets.view(-1)\`：$(B, T) \\to (B \\times T,)$。每一项是这一行对应的正确下一个 token 的编号。

> batch 里的每个位置都变成一道独立的分类题，一共 $B \\times T$ 道。

### 对所有位置取平均

$B \\times T$ 行各算出一个 loss，$-\\log p_t$。默认（\`reduction='mean'\`）把它们平均成一个数。用 $B = 2$、$T = 4$、$V = 5$ 和随机 logits 在 PyTorch 里验证：

\`\`\`text
logits.view(-1, V).shape = (8, 5)
targets.view(-1).shape   = (8,)

8 个位置各自的 loss：
[1.916, 1.822, 2.035, 3.344, 2.511, 1.021, 2.102, 0.378]

平均                     = 1.8913
F.cross_entropy 的结果   = 1.8913   ✓
\`\`\`

因为 loss 是平均值，每个位置的 logits 梯度也要除以 $B \\times T$：

$$
\\frac{\\partial\\, \\text{loss}}{\\partial\\, \\text{logits}[b, t]} = \\frac{p - y}{B \\times T}
$$

这样梯度的大小不会随 batch 变大而变大，换 batch size 时 learning rate 也不用跟着改。

> 一个长度为 $T$ 的序列，一次 forward 就得到 $T$ 道"预测下一个词"的题：每个 token 都是训练信号。之所以能这样做，靠的是 [causal mask](#info:causal_mask_notes)：位置 $t$ 看不到它后面的 token，也就偷看不到自己的答案。

## 参考资料

1. **PyTorch 文档**

   CrossEntropyLoss（多分类 softmax CE；targets 是类别索引，不是 one-hot）。

   [PyTorch Docs ↗](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

2. **CS231n Softmax Classifier**

   讲解 softmax + cross-entropy 的推导及其梯度。

   [CS231n Softmax Classifier ↗](https://cs231n.github.io/linear-classify/)
`,
  },
  highlevel_transformer_introduction_notes: {
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
  query_view_notes: {
    title: "Query View（拆成多个 head）",
    md: `
## Query projection

\`q\` 是合并投影的第一部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 q 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
q of one token (768 dims):            # 一个 token 的 q（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 q 拆成 3 个 head，每个 2 维。

\`\`\`text
q = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

k 和 v 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's q, head 1's q]
token 1: [head 0's q, head 1's q]
token 2: [head 0's q, head 1's q]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's q, token 1's q, token 2's q]
head 1: [token 0's q, token 1's q, token 2's q]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
  },
  key_view_notes: {
    title: "Key View（拆成多个 head）",
    md: `
## Key projection

\`k\` 是合并投影的第二部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 k 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
k of one token (768 dims):            # 一个 token 的 k（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 k 拆成 3 个 head，每个 2 维。

\`\`\`text
k = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

q 和 v 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's k, head 1's k]
token 1: [head 0's k, head 1's k]
token 2: [head 0's k, head 1's k]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's k, token 1's k, token 2's k]
head 1: [token 0's k, token 1's k, token 2's k]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
  },
  value_view_notes: {
    title: "Value View（拆成多个 head）",
    md: `
## Value projection

\`v\` 是合并投影的第三部分：\`qkv = self.c_attn(x)\` 把每个 token 从 $C$ 维映射到 $3C$ 维，\`q, k, v = qkv.split(C, dim=2)\` 再把它切成三份。每一份都等价于一个独立的线性层，作用在每个 token 上：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, C)
$$

## 把通道拆成多个 head

\`.view(B, T, n_head, hs)\` 把最后一维 $C$ 拆成 $n\\_head \\times hs$，其中 $hs = \\left\\lfloor \\dfrac{C}{n\\_head} \\right\\rfloor$：

$$
(B, T, C) \\;\\rightarrow\\; (B, T, n\\_head, hs)
$$

在 GPT-2 small 里，$n\\_head = 12$，$hs = 768 / 12 = 64$，所以每个 token 的 768 维 v 被**按顺序**切成 12 段，每段 64 维：

\`\`\`text
v of one token (768 dims):            # 一个 token 的 v（768 维）
[ dims 0–63 | dims 64–127 | dims 128–191 | ... | dims 704–767 ]
   head 0       head 1         head 2               head 11
\`\`\`

一个更小的例子：把 6 维的 v 拆成 3 个 head，每个 2 维。

\`\`\`text
v = [1, 2, 3, 4, 5, 6]

after view(3, 2):
[[1, 2],    <- head 0
 [3, 4],    <- head 1
 [5, 6]]    <- head 2
\`\`\`

> **数值没有变。** \`view\` 只改变同一批数字的索引方式：不做计算，也不复制。768 个数还是那 768 个数，只是从一长行变成了 12 行、每行 64 个。

q 和 k 也按同样的方式拆分，所以每个 token 最后都有 12 个小 q、12 个小 k 和 12 个小 v，每个 64 维。head $i$ 用的是它们各自的第 $i$ 段。

## 把 head 这一维挪到 batch 旁边

\`.transpose(1, 2)\` 交换 $T$ 和 $n\\_head$ 两个轴：

$$
(B, T, n\\_head, hs) \\;\\rightarrow\\; (B, n\\_head, T, hs)
$$

在 GPT-2 small 里就是 $(B, T, 12, 64) \\rightarrow (B, 12, T, 64)$。

### 同样的数据，换一种分组

例子：$T = 3$ 个 token，2 个 head（每一项是一个 64 维向量）：

\`\`\`text
Before (B, T, n_head, hs): grouped by token     # 之前：按 token 分组
token 0: [head 0's v, head 1's v]
token 1: [head 0's v, head 1's v]
token 2: [head 0's v, head 1's v]

After (B, n_head, T, hs): grouped by head       # 之后：按 head 分组
head 0: [token 0's v, token 1's v, token 2's v]
head 1: [token 0's v, token 1's v, token 2's v]
\`\`\`

数字没有变，变的只是分组方式：从"每个 token 有哪些 head"变成"每个 head 有哪些 token"。

### 为什么要按 head 分组？

Attention 是在**每个 head 内部**、**跨 token** 进行的：在 head 0 里，每个 token 的 q 要和每个 token 的 k 比较。transpose 之后，最后两维是 $(T, hs)$，也就是一个 $T \\times 64$ 的矩阵，装着一个 head 里所有 token 的向量。所以下一步 \`q @ k.transpose(-2, -1)\` 就是在这两维上做普通的矩阵乘法。PyTorch 把前面的 $(B, n\\_head)$ 当作 batch 维，对每个序列、每个 head 并行地做这次乘法：

\`\`\`text
(B, n_head, T, hs)
 └──┬───┘  └─┬─┘
 batch dims  one matrix per head     # batch 维 | 每个 head 一个矩阵
\`\`\`
`,
  },
  q_k_product_notes: {
    title: "Q @ K 乘积",
    md: `
## Scaled dot-product attention 详解

\`q\` 和 \`k\` 的形状是 $(B, n\\_head, T, hs)$（每个 head 的 query 和 key）。

- \`k.transpose(-2, -1)\` $\\rightarrow (B, n\\_head, hs, T)$ 交换 key 的最后两个轴，这样每个 query 都能和所有时间步的 key 做点积。
- \`@\` 是批量矩阵乘法：

  $$
  (B, n\\_head, T, hs)\\ @\\ (B, n\\_head, hs, T) \\;\\rightarrow\\; (B, n\\_head, T, T)
  $$

  结果：每个 head 得到一个 $T \\times T$ 的原始 attention 分数矩阵（每一行对应一个 query 位置，每一列对应一个 key 位置）。
- \`* (1.0 / hs**0.5)\` 乘以 $1/\\sqrt{hs}$（$hs$ 是 head size）。这就是 "Scaled Dot-Product Attention" 里的缩放。

## $T \\times T$ 矩阵的含义

第 $(i, j)$ 项是 token $i$ 的 q 和 token $j$ 的 k 的点积，再乘以 $1/\\sqrt{hs}$（原因见下文）：一个数，表示位置 $i$ 对位置 $j$ 有多感兴趣：

$$
\\mathrm{att}[i][j] = \\frac{q_i \\cdot k_j}{\\sqrt{hs}}
$$

这个计算在每个序列、每个 head 内部分别进行。

以 "the river bank"（$T = 3$）在某一个 head 里为例（数字是编的，只为展示结构）：

\`\`\`text
             k_the   k_river   k_bank
q_the     [   1.0     0.2       0.5  ]   # the 给每个词的分数
q_river   [   0.3     1.2       0.4  ]   # river 给每个词的分数
q_bank    [   0.1     2.5       0.8  ]   # bank 给每个词的分数：river 最高
\`\`\`

- **每一行：** 一个 token 作为提问者，给所有 token 打分。
- **每一列：** 一个 token 作为被查找的对象，被所有 token 打分。

> **这是位置之间第一次发生交互。** \`c_attn\`、\`split\`、\`view\` 和 \`transpose\` 都是对每个 token 单独操作的。到了 \`q @ k.transpose(-2, -1)\`，token $i$ 的 q 才第一次遇到**其他** token 的 k。

有两点要注意：

- 这个矩阵仍然是**完整的** $T \\times T$，包括往后看的分数（比如 the → bank）。下一步的 [causal mask](#info:causal_mask_notes) 会把这些去掉。
- 每个 head 都有自己的 $T \\times T$ 矩阵，所以每个序列有 12 个。

> 🆕 现在大多数 LLM 不再把 position embedding 加到 $x$ 上，而是在这一步之前按位置旋转 q 和 k，让每个分数只取决于 $i$ 和 $j$ 之间的距离：[RoPE](#info:rope_notes)。

## 为什么要乘 $1/\\sqrt{hs}$

在 GPT-2 small 里 $hs = 64$，所以这一步就是把每个分数除以 $\\sqrt{64} = 8$。

### 问题：点积随维度增大

一个分数是 $hs$ 个乘积之和：

$$
q \\cdot k = q_1 k_1 + q_2 k_2 + \\cdots + q_{hs} k_{hs}
$$

假设 $q$ 和 $k$ 的各分量相互独立，均值为 0、方差为 1（大致是 LayerNorm 之后的尺度）。那么每一项 $q_i k_i$ 的方差是 1，$hs$ 项加起来，方差就是 $hs$：

$$
\\operatorname{Var}(q \\cdot k) = hs, \\qquad \\operatorname{std}(q \\cdot k) = \\sqrt{hs}
$$

$hs = 64$ 时标准差是 8，所以 $\\pm 8$、$\\pm 16$ 这样的分数很常见。维度越多，分数越大。

> 在本站代码的初始化状态下，$q$ 和 $k$ 各分量的方差其实约为 1/3，而不是 1（LayerNorm 的输出方差是 1，但 q 和 k 还要经过一个线性层）。结论不变：方差仍然和 $hs$ 成正比。每个方差的逐步推导见 [初始化时的方差](#info:variance_init_notes)。

### 为什么分数大了不好：softmax 饱和

- 分数 $[8,\\; 0,\\; -8]$ → softmax $\\approx [0.9997,\\; 0.0003,\\; 0.0000]$：几乎所有权重都在一个 token 上；
- 分数 $[1,\\; 0,\\; -1]$ → softmax $\\approx [0.665,\\; 0.245,\\; 0.090]$：权重分散开。

饱和的 softmax 会带来两个问题：

- 它几乎是 one-hot，一个位置只能看到一个 token；
- 它的梯度接近 0，模型几乎没法调整注意力该放在哪里。

### 解决办法：除以 $\\sqrt{hs}$

除以 $\\sqrt{hs}$ 把标准差拉回到 1：

$$
\\operatorname{std}\\!\\left(\\frac{q \\cdot k}{\\sqrt{hs}}\\right) = 1
$$

上面的 $[8, 0, -8]$ 除以 8，正好变成 $[1, 0, -1]$。

> 不管 head size 多大，分数的标准差都保持在 1 左右，softmax 一开始就不会饱和。这就是 **Scaled Dot-Product Attention** 里 "scaled" 的含义。Transformer 论文给出的理由相同："for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients"（[Vaswani et al., 2017 ↗](https://arxiv.org/abs/1706.03762)，§3.2.1）。

### 和 LayerNorm 的关系

- **LayerNorm** 控制**输入** $x$ 的尺度。
- **$1/\\sqrt{hs}$** 控制**求和结果** $q \\cdot k$ 的尺度。即使输入已经归一化，这个和也会随项数增多而变大。

两者配合使用，谁也替代不了谁。

这一步之后，先加 causal mask，再在最后一维上做 softmax，得到 attention 概率。
`,
  },
  attention_linear_notes: {
    title: "Attention 的 Linear（c_attn）",
    md: `
## 它是什么

\`self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)\` 是一个合并的投影，用一次矩阵乘法同时算出 **Q、K、V**。它等价于把三个 \`Linear(n_embd, n_embd)\` 摞在一起：

$$
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}
$$

## 参数形状

- **Weight：** $(3 \\times n\\_embd,\\ n\\_embd)$
- **Bias：** $(3 \\times n\\_embd)$

从数学上看，$W$ 把 $768$ 维映射到 $2304$ 维，所以有 $768 \\times 2304$ 个数。PyTorch 的 \`nn.Linear\` 把它存成 **(out, in)** $= (2304, 768)$，计算的是 $x\\,W^\\top + b$，所以上面的 weight 形状看起来是转置过的。

## 每个 block 一套，所有 token 共用

- **每个 block 各自一套。** 12 个 block 各有自己的 \`c_attn\`，各有自己的 $W$ 和 $b$，各自独立初始化、独立训练。block 之间不共享任何东西，所以每一层都学自己计算 q、k、v 的方式。
- **同一个 block 内部共用。** 在一个 block 里，每个 token，无论来自 batch 里哪个序列、处在哪个位置，都乘以**同一个** $W$，加上同一个 $b$。\`c_attn\` 没有位置的概念：它对每个位置用同一条规则，所以两个位置的 q、k、v 之所以不同，只是因为它们的输入向量不同。比如 "the cat and the dog" 里出现了两次 "the"：
  - **在第 1 个 block 里**，两者的 token embedding 相同，所以输入的唯一区别是 position embedding。
  - **在后面的 block 里**，每个 "the" 还通过 attention 收集到了不同的上下文：第一个只看得到自己，第二个看得到 "the cat and the"。它们的输入差得越来越远，差别主要来自上下文，而不只是位置。

## 它对张量做了什么

1. 输入 $x \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$。
2. \`qkv = c_attn(x)\` 得到 $qkv \\in \\mathbb{R}^{B \\times T \\times 3n\\_embd}$。每个位置的 768 维向量都用同一个 $W$ **单独**投影：位置之间还没有任何信息流动。信息流动要到 $q K^\\top$ 才开始。
3. \`q, k, v = qkv.split(n_embd, 2)\` 得到 $q, k, v \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$。
4. 把 q、k、v 拆成多个 head，$hs = n\\_embd / n\\_head$：\`view(B, T, n_head, hs).transpose(1, 2)\` 得到 $(B,\\ n\\_head,\\ T,\\ hs)$。

## 为什么把 Q/K/V 合并？

kernel 启动次数更少，缓存利用更好：一次大的 GEMM（矩阵乘法）代替三个独立的 Linear。结果和三个独立的 Linear 完全相同。

## 参数量

- **Weights：** $3 \\times n\\_embd^2$
- **Bias：** $3 \\times n\\_embd$

> 例子（$n\\_embd = 768$）：weights $3 \\cdot 768^2 = 1{,}769{,}472$，bias $3 \\cdot 768 = 2{,}304$。

## PyTorch 的默认初始化

PyTorch 的 \`nn.Linear\` 默认用 **Kaiming uniform** 初始化 weight，bias 则用同样边界的普通均匀分布。

### Weights

PyTorch 用 Kaiming（He）uniform 初始化 \`Linear.weight\`：

\`\`\`python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
\`\`\`

传入 \`a=math.sqrt(5)\` 是 PyTorch 的一个技巧，让得到的均匀分布范围变成 $\\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}}, \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)$，在数值上和直接用这个边界的普通均匀分布一样。这是为了在重构成 \`kaiming_uniform_\` 时保持向后兼容（[论坛帖子 ↗](https://discuss.pytorch.org/t/nn-linear-default-weight-initialisation-assumes-leaky-relu-activation/46712/2)）。

$$
W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right), \\quad \\text{fan\\_in} = n\\_embd
$$

$n\\_embd = 768$ 时，边界约为 $\\pm 0.0361$。

### Biases

bias 不用 Kaiming，而是直接从同样边界的普通均匀分布里抽：

\`\`\`python
fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
bound = 1 / sqrt(fan_in)
init.uniform_(self.bias, -bound, bound)
\`\`\`

$$
\\Rightarrow b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)
$$

> 所以：weight 用的是 Kaiming uniform，但参数设置成和普通均匀分布同样的范围；bias 直接用这个普通均匀分布。

**注意：** 很多 Transformer/GPT-2 实现会覆盖这个默认值，对 Linear 层和 embedding 用正态分布 $\\mathcal{N}(0, 0.02^2)$ 初始化。原版 GPT-2 代码（OpenAI 的 TensorFlow \`model.py\`）对 \`c_attn\` 正是这样做的：weight $\\sim \\mathcal{N}(0, 0.02^2)$，bias **全为 0**。本站代码没有覆盖任何初始化，所以用的是上面的 PyTorch 默认值。

## 为什么要随机，为什么要小？

- **为什么不全设为 0？** 如果 $W = 0$，每个位置得到的 q、k、v 都一样，$W$ 的每个元素收到的梯度也一样，于是它们永远保持相同，没法学到不同的东西。随机值打破了这种对称性。
- **为什么要小？** weight 小，q 和 k 就小，分数 $q \\cdot k$ 一开始接近 0，softmax 一开始接近均匀分布。这样它就不会饱和，从第一步起梯度就能流动。

q、k、v 最终代表的一切含义，都是从这些随机起点学出来的。

## 参考资料

1. **Kaiming 初始化**

   提出了针对整流激活函数（ReLU/LeakyReLU）的初始化方法，并推导出保持方差的方案。

   [He et al., 2015 ↗](https://arxiv.org/pdf/1502.01852.pdf)

2. **PyTorch 默认值**

   PyTorch 的初始化代码，展示了确切的行为。

   [PyTorch init code ↗](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50)

3. **GPT-2 / HF Transformers 的做法（覆盖 PyTorch 默认值）**

   GPT-2（HF Transformers）：权重矩阵从（截断的）Normal(0, 0.02) 初始化，由 config 里的 \`initializer_range=0.02\` 控制。用 Transformers 实例化 GPT-2 时，这是广泛使用的默认值。

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

4. **通用的 Transformer 初始化策略**

   通用的 Transformer 初始化说明（HF 等）里也描述了同样的策略："weights ~ Normal(0, initializer_range) (often 0.02), biases 0; LayerNorm γ=1, β=0."

   [HF Transformers Modeling Utils ↗](https://huggingface.co/transformers/v3.1.0/internal/modeling_utils.html)
`,
  },
  causal_mask_notes: {
    title: "Causal Mask（因果掩码）",
    md: `
**这一行对 attention 分数加上因果（只能往回看）掩码，让每个 token 只能关注它自己和前面的 token。**

\`\`\`python
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
\`\`\`

- \`att\`：$(B,\\ n\\_head,\\ T,\\ T)$，来自[上一步](#info:q_k_product_notes)的缩放分数 $q\\,k^\\top/\\sqrt{hs}$
- \`self.mask\`：$(1,\\ 1,\\ 1024,\\ 1024)$，对角线及其下方是 1，上方是 0

## 为什么需要掩码

$q\\,k^\\top$ 得到的分数矩阵是**完整的** $T \\times T$：每个位置都给所有其他位置打了分，**包括它后面的位置**。以 4 个 token \`I love black cats\` 为例（行 = query，"谁在看"；列 = key，"被看的是谁"）：

\`\`\`text
            I     love   black   cats
I        [  ✓      ✓       ✓      ✓  ]
love     [  ✓      ✓       ✓      ✓  ]
black    [  ✓      ✓       ✓      ✓  ]
cats     [  ✓      ✓       ✓      ✓  ]
\`\`\`

看 \`love\` 这一行。它的输出要用来预测下一个 token \`black\`，可它已经给 \`black\` 打过分了。这就是**偷看答案**。

- **训练时：** 整个序列并行处理，每个位置都预测它的下一个 token。如果 \`love\` 能看到 \`black\`，预测 \`black\` 就只是照抄，模型学不到任何有用的东西。
- **推理时：** 文本从左到右生成，后面的 token **还不存在**。训练时把未来遮住，训练才和模型的实际使用方式一致。

解决办法：在 softmax 之前，把对角线以上的部分全部挡住，让每一行只看到**它自己和更早的位置**。

## 一步步看

### 1. \`self.mask\` 是什么

它在 \`__init__\` 里只创建一次：

\`\`\`python
self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))
\`\`\`

- [torch.ones](#info:torch_ones_notes)(1024, 1024) 全是 1；[torch.tril](#info:torch_tril_notes) 保留**下三角**（对角线及其下方），把上方全部设为 0。以 4×4 为例：

  \`\`\`text
  torch.tril(torch.ones(4, 4))
  [[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]
  \`\`\`

  第 $t$ 行在第 $0 \\dots t$ 列是 1：位置 $t$ 可以看它自己和它前面的所有位置。
- [unsqueeze(0)](#info:unsqueeze_notes)\`.unsqueeze(0)\` 在前面加两个维度，得到 $(1,\\ 1,\\ 1024,\\ 1024)$，这样它就能在 batch 和 head 两个维度上**广播**（原理见下面第 4 步末尾的 **广播**）。
- 它是一个 [buffer](#info:register_buffer_notes)：会跟着模块一起移到 GPU/CPU，也会一起保存，但它不是参数，从不参与训练。

### 2. 切到当前长度

buffer 按最大长度 1024 创建，但当前序列只有 $T$ 个 token：

\`\`\`text
self.mask[:, :, :T, :T]   →   (1, 1, T, T)
\`\`\`

这一步取出左上角的 $T \\times T$。下三角矩阵的左上角本身还是下三角矩阵，所以 $T = 4$ 时，它正好就是上面那个 $4 \\times 4$ 矩阵。

### 3. 变成"哪些连接被禁止"

\`== 0\` 把 0/1 矩阵变成布尔值：元素为 0 的地方是 **True**，也就是要挡住的未来位置。

\`\`\`text
mask == 0
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
\`\`\`

### 4. \`masked_fill\`：在 True 的地方写入 \`-inf\`

\`att.masked_fill(bool_mask, -inf)\` 在掩码为 True 的位置把 \`att\` 写成 $-\\infty$，其他分数保持不变。以 \`I love black cats\` 在某一个 head 里的分数为例，前后对比：

\`\`\`text
before                                    after
          I     love  black  cats                   I     love  black  cats
I      [ 2.1,   0.3,  1.1,   0.6 ]      I      [ 2.1,   -inf, -inf,  -inf ]
love   [ 0.5,   1.3,  1.7,   2.5 ]      love   [ 0.5,   1.3,  -inf,  -inf ]
black  [ 0.2,   0.9,  1.7,   1.4 ]      black  [ 0.2,   0.9,  1.7,   -inf ]
cats   [ 0.1,   0.4,  2.5,   0.8 ]      cats   [ 0.1,   0.4,  2.5,   0.8  ]
\`\`\`

现在 \`I\` 只看得到自己，\`love\` 看得到 \`I\` 和自己，\`cats\` 看得到全部四个。一般地，对 query 位置 $t$：

$$
\\text{att}_{t,j} = -\\infty \\quad \\text{for } j > t
$$

**广播。** 掩码是 $(1,\\ 1,\\ T,\\ T)$，而 \`att\` 是 $(B,\\ n\\_head,\\ T,\\ T)$。大小为 1 的维度会被拉伸，所以 batch 里每个序列、每个 head 用的都是**同一个**三角形。这是对的："不能往后看"只取决于位置，和内容、和是哪个 head 都无关。

### 5. 为什么填 \`-inf\` 而不是 \`0\`？

下一行是 \`att = F.softmax(att, dim=-1)\`。softmax 对每个分数取 $e$ 的幂，再除以这一行的总和：

$$
\\text{softmax}(s)_i = \\frac{e^{s_i}}{\\sum_j e^{s_j}}
$$

拿第 4 步里 \`love\` 这一行来看。

**填 \`-inf\`：未来位置的权重正好是 0。** 因为 $e^{-\\infty} = 0$：

\`\`\`text
scores:   [ 0.5,   1.3,   -inf,  -inf ]
e^s:      [ 1.65,  3.67,  0,     0    ]    sum 5.32
softmax:  [ 0.31,  0.69,  0,     0    ]
\`\`\`

未来位置得不到任何权重，允许的位置会**自动重新归一化**：加起来仍然是 1。

**填 \`0\` 不行。** 分数为 0 并不表示"忽略"，它只是一个普通分数，而 $e^0 = 1$：

\`\`\`text
scores:   [ 0.5,   1.3,   0,     0     ]
softmax:  [ 0.225, 0.501, 0.137, 0.137 ]
\`\`\`

两个未来位置仍然分到了 27% 的注意力：模型还在偷看。

**为什么不在 softmax 之后把权重清零？** 那样这一行加起来就不再是 1 了。用没加掩码的 \`love\` 行 \`[0.5, 1.3, 1.7, 2.5]\`：

\`\`\`text
softmax (no mask):  [ 0.072, 0.16, 0.238, 0.53 ]
zero the future:    [ 0.072, 0.16, 0,     0    ]    sum only 0.23
\`\`\`

还得再除一次才能把总和修正回 1。在 softmax **之前**填 \`-inf\`，一步就同时做到了"未来 = 0"和"每行加起来 = 1"。

> 一句话：$e^{-\\infty} = 0$，所以 softmax 把分数为 \`-inf\` 的位置当作不存在。

## 补充

- 这个掩码只负责**因果性**，不处理 padding。如果一个 batch 里有填充的 token，通常还会再加一个 padding mask（又一次 \`masked_fill\`），让 padding 位置也变成 $-\\infty$。
- 有些代码用一个很大的负数（比如 $-10^9$）代替 $-\\infty$；softmax 之后它的权重同样几乎是 0。
`,
  },
  tokenizer_notes: {
    title: "Tokenizer（GPT-2 的 BPE）",
    md: `
这行代码返回 GPT-2 的 **byte-level BPE tokenizer**（词表大小 **50257**，包含特殊 token \`<|endoftext|>\`，id 为 **50256**）。

它包含 byte↔[unicode 映射表](#info:gpt2_byte_unicode_map_notes)、[GPT-2 merge ranks](#info:gpt2_merge_ranks_notes)、[regex pre-tokenizer](#info:regex_pre_tokenizer_notes)，以及特殊 token。

## Tokenization 是怎么进行的

### 1 · Pre-tokenize（预切分）

用 GPT-2 的正则表达式把原始文本切成若干片段（把连续的字母、数字、标点、前导空格等分开）。

**例子**（每个方框是一个片段；⎵ 表示一个空格字符）

- \`Hello, world!\` → \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!⟧\`
- \`2025-09-19\` → \`⟦2025⟧⟦-⟧⟦09⟧⟦-⟧⟦19⟧\`

[深入了解 regex pre-tokenizer →](#info:regex_pre_tokenizer_notes)

### 2 · 字节编码

把每个片段编码成 UTF-8 字节（大小写和空白都保留）。再通过 byte↔unicode 表，把每个字节映射成一个安全的 Unicode 符号，这样每个字节都能被表示出来。

**例 1**（第 1 步得到的片段）

- \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!⟧\` → \`⟦Hello⟧⟦,⟧⟦Ġworld⟧⟦!⟧\`
- \`⎵world\` 内部：字节 \`[0x20, 0x77, 0x6F, 0x72, 0x6C, 0x64]\` → \`Ġ w o r l d\`
- 只有空格字节 \`0x20\` 变了（变成 \`Ġ\`）；ASCII 字母和标点都映射成它们自己。

**例 2**（一个非 ASCII 字符）：片段 \`⎵café\`（来自文本 \`I love café\`）

- 字节：\`[0x20, 0x63, 0x61, 0x66, 0xC3, 0xA9]\`（空格和重音符号都保留）。
- Unicode：\`U+0120 Ġ\` · \`U+0063 c\` · \`U+0061 a\` · \`U+0066 f\` · \`U+00C3 Ã\` · \`U+00A9 ©\`
- 得到字符串 \`ĠcafÃ©\`。

> **顺序很重要：** 正则要先在原始文本上运行，因为替身字符 \`Ġ\` 算作字母，\`HelloĠworld\` 就不会在空格处被切开了。

[深入了解 byte-unicode 映射表 →](#info:gpt2_byte_unicode_map_notes)

### 3 · BPE merges（合并）

对每个片段，按照 GPT-2 的 [merge ranks](#info:gpt2_merge_ranks_notes) 反复合并符号对，直到不能再合并为止，得到子词单元。

### 4 · 查 ID

在词表（\`encoder.json\`）中查每个最终的子词，得到它的 token id。

- \`⟦Hello⟧⟦,⟧⟦Ġworld⟧⟦!⟧\` → \`[15496, 11, 995, 0]\`
- 这里每个片段本身已经是一个完整的 token，所以不需要再拆分；\`!\` 的 id 是 0，因为它是第一个可打印字节。

> **注：** 因为是 byte-level 的，任何输入字符串都能被 tokenize，不会出现"未知"token，原始的空格和大小写也都保留下来。

## 最近的发展：GPT-2 之后 tokenizer 有哪些变化

BPE 至今仍是主流做法。变化最大的是词表大小：

- GPT-2：约 50K 个 token
- GPT-4（\`cl100k_base\`）：约 100K
- GPT-4o（\`o200k_base\`）：约 200K
- Llama 3：约 128K

词表越大，同样的文本切出的 token 越少，所以生成更快，而且对中文等非英语语言处理得更好。
代价是 embedding 表和输出层更大：两者的形状都是 $(V, n\\_embd)$，所以 $V$ 每增加一个 token，它们就各多一行，而向量大小 $n\\_embd$ 不变。

### 数字是怎么切分的

GPT-2 的正则会把一整串连续数字放在一起（\` ?\\p{N}+\`），然后 BPE 在其内部合并。tokenizer 是确定性的（同一个数字总是以同样的方式切分），但切分方式和数位毫无关系：

- \`1234\` → \`⟦12⟧⟦34⟧\`
- \`12345\` → \`⟦123⟧⟦45⟧\`
- \`1234567\` → \`⟦123⟧⟦45⟧⟦67⟧\`
- \`3517+4825=8342\` → \`⟦35⟧⟦17⟧⟦+⟧⟦48⟧⟦25⟧⟦=⟧⟦8⟧⟦342⟧\`

同一个 token 可以代表不同的数位。在 \`4825\` → \`⟦48⟧⟦25⟧\` 中，\`25\` 里的 \`2\` 是十位；在 \`2517\` → \`⟦25⟧⟦17⟧\` 中，它是千位。要把两个数相加，模型得先根据 token 的长度和个数推断出每个数字处在哪一位；写答案时，还必须输出 BPE 恰好会产生的那种切分（\`⟦8⟧⟦342⟧\`，而不是 \`⟦83⟧⟦42⟧\`）。这并不会让算术变得不可能，但会让它更难学。

较新的 tokenizer 改了数字规则，主要有两种做法：

- **每块最多 3 位数字**（\`\\p{N}{1,3}\`）：GPT-4（\`cl100k_base\`）、GPT-4o（\`o200k_base\`）、Llama 3、DeepSeek-V3。\`1234567\` → \`⟦123⟧⟦456⟧⟦7⟧\`。每块有 1–3 位数字，而且前导空格不再粘在数字上。不过分组仍然是从左往右进行的，所以块的边界并不像 \`1,234,567\` 那样和千位对齐。
- **每位数字一个 token**（\`\\p{N}\`）：Qwen2.5、Qwen3；LLaMA 1/2 用的是带数字拆分的 SentencePiece。\`1234567\` → \`⟦1⟧⟦2⟧⟦3⟧⟦4⟧⟦5⟧⟦6⟧⟦7⟧\`。数位整齐对齐，进位在每个数字里都以同样的方式进行，代价是序列更长。

> 并没有唯一的赢家：Meta 从逐位切分（LLaMA 1/2）改成了 3 位一组（Llama 3），当时它换成了基于 tiktoken 的 128K 词表。闭源模型（Claude、Gemini 等）不公开它们的 tokenizer。如今的模型做大数算术时，也更多依赖逐步推理或计算器/代码工具，而不只是靠 tokenization。

## 参考资料

1. **一手资料（架构 + 输入表示）**

   GPT-2 技术报告。详细说明了 byte-level BPE、按字节处理的动机，以及扩大后的词表（50,257）。见 §2.2 "Input Representation" 和 Table 2。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

2. **GPT-2 BPE tokenizer 的原始源码**

   OpenAI 的 encoder.py，约 120 行。上面所有步骤都在里面：\`bytes_to_unicode()\`（byte↔unicode 映射表）、正则模式 \`self.pat\`（pre-tokenize）、\`bpe()\`（按 rank 合并），以及 \`encode()\` / \`decode()\`。它加载的词表（\`encoder.json\`）和合并列表（\`vocab.bpe\`）由 \`download_model.py\` 下载。

   [openai/gpt-2 encoder.py ↗](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

3. **Tokenizer 实现**

   OpenAI 的 tiktoken。清晰实用的参考：GPT-2 的 byte-level BPE（无损、可逆），以及 bytes ↔ tokens 映射是如何实现的。

   [tiktoken ↗](https://github.com/openai/tiktoken)

4. **BPE 的奠基论文（为什么用子词/字节级）**

   Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units."。它不是专门讲 GPT-2 的，但它是子词/BPE 词表背后的经典参考，GPT-2 的 embedding 就建立在这种词表之上。

   [Neural Machine Translation of Rare Words with Subword Units ↗](https://aclanthology.org/P16-1162/)

5. **Let's build the GPT Tokenizer**

   Karpathy 讲 GPT Tokenizer 的视频。关于 GPT-2 byte-level BPE 的实用讲解（从 1430 秒开始）。

   [BPE - Let's build the GPT Tokenizer ↗](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s)

6. **动手/教学用的 BPE 参考**

   Karpathy 的 minBPE（最简的 byte-level BPE）

   [minbpe ↗](https://github.com/karpathy/minbpe)

7. **GPT-2 embedding 的实证分析**

   对 GPT-2 embedding 权重的可解释性深入研究（分布特性、与位置的关系）。对 wte 内部的进阶探索。

   [An exploration of GPT-2's embedding weights ↗](https://www.alignmentforum.org/posts/BMghmAxYxeSdAteDc/an-exploration-of-gpt-2-s-embedding-weights)

`,
  },
  gpt2_byte_unicode_map_notes: {
    title: "GPT-2 byte-unicode 映射表",
    md: `
## 它是什么

一个固定的一一映射：把 256 个可能的字节（0–255）和 256 个可打印的 Unicode 码点对应起来。

- **byte → unicode（"byte encoder"）：** 把每个原始字节变成一个可见的 Unicode 字符。
- **unicode → byte（"byte decoder"）：** 上面的精确逆映射，所以完全可逆。

## 为什么需要它

经典 BPE 是在"字符"上操作的。而 GPT-2 是 byte-level 的：它要无损地处理任意字节序列（emoji、重音字符、类似二进制的字节等）。但 BPE 代码处理的是字符串，而且它的合并列表（\`vocab.bpe\`）把每个符号对写成用空格隔开的两个符号，例如 \`Ġ t\`。如果符号里出现原始的空格、换行或控制字节，这种格式就会被破坏。这个映射表这样解决问题：

1. 给每个字节一个安全、可见的替身，
2. 避免和空白/控制字符冲突，
3. 让 BPE 在可打印的字符串上运行，同时保持完全可逆。

> regex pre-tokenizer 在更早的阶段、在原始文本上运行，所以它永远看不到这些替身字符。

## 它是怎么构建的（概念上）

- 本来就"好看、可打印"的字节（例如 ASCII 和许多 Latin-1 字符）映射成它们自己（同一个码点）。
- 其余字节（例如空格、制表符、换行等）映射到从 U+0100 开始的**互不相同的可打印码点**（256 + n，按字节顺序），保证它们可见且不是空白。

**例子**

- 字节 \`0x41\`（\`A\`）→ 映射成标准的 \`A\`（同一个字符）
- 字节 \`0x20\`（空格）→ 替身 \`Ġ\`（U+0120），这就是为什么片段 \`⟦⎵world⟧\` 显示为 \`⟦Ġworld⟧\`。
- 字节 \`0xC3 0xA9\`（é）→ \`Ã©\`（两个字节都是可打印的 Latin-1，所以各自映射成自己：两个字符，每个字节一个）
- 字节 \`0xE2 0x80 0x94\`（破折号 —）→ \`âĢĶ\`（\`0xE2\` 映射成自己，即 \`â\`；\`0x80\` → \`Ģ\`，\`0x94\` → \`Ķ\`）

这样得到两张表：

- \`byte_encoder: {0..255 → one-char unicode strings}\`
- \`byte_decoder: {that unicode char → original byte}\`

**哪些字节映射成自己？** 在 GPT-2 的映射表里：**33–126**（除空格外的可打印 ASCII）和 **161–172、174–255**（可打印的 Latin-1）。

这些范围之外的字节（例如控制字符 0–31、空格 32、127–160 和 173）会得到专门的替身（\`U+0100\`、\`U+0101\`、…、\`U+0143\`）：一共 68 个字节。空格是其中第 33 个（n = 32），所以它变成 U+0120 \`Ġ\`。

**Emoji 例子**（UTF-8：\`F0 9F 98 8A\`）

- \`0xF0\`（= 240）在 174–255 范围内 ⇒ 映射成自己（在 Latin-1 中显示为 \`ð\`）。
- \`0x9F\`（159）、\`0x98\`（152）、\`0x8A\`（138）不在可打印集合里 ⇒ 替身是 \`Ł\`、\`ĺ\`、\`Ĭ\`（Latin Extended-A 字符）。
- 所以 😊 变成字符串 \`ðŁĺĬ\`：一个映射成自己的字符加三个替身，每个字节一个可见字符，完全可逆。

## 它是怎么用的（编码路径）

1. 用 regex pre-tokenizer 把原始文本切成块。
2. 把每块编码成 UTF-8 字节。
3. 把每个字节 \`b\` 映射成它的替身 Unicode 字符 \`u = byte_encoder[b]\`。
   - 例子：块 \`⟦⎵Hello⟧\`（带前导空格）→ 字节 \`[0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F]\` → 映射后的字符串 \`ĠHello\`（空格变成了 \`Ġ\`；ASCII 字母映射成自己）。
   - 非 ASCII 字符（例如 é → 字节 \`C3 A9\`）会变成两个映射后的字符，每个字节一个。
4. 在每个映射后的块上运行 BPE 合并，然后查出每个结果 token 的 id。

## 解码（精确的逆过程）

1. 在词表里查出每个 token id 对应的字符串 → 把它们拼接成一个由映射字符组成的字符串。
2. 通过 \`byte_decoder\` 把每个字符映射回一个字节。
3. 对这些字节做 UTF-8 解码 → 得到原始文本。

> 这保证了无损往返：\`decode(encode(s)) == s\`

`,
  },
  regex_pre_tokenizer_notes: {
    title: "Regex pre-tokenizer（正则预切分）",
    md: `
## 它是什么

一个基于正则表达式的切分器，在 BPE 合并之前把输入文本切成若干块。之后 BPE 合并只在每块内部进行，永远不会跨越块的边界。

## 为什么需要它

- **主要原因（GPT-2 论文，§2.2）** — 没有它的话，BPE 会跨字符类型合并，把词表浪费在 \`dog.\`、\`dog!\`、\`dog?\` 这样的变体上。先切分可以让字母、数字和标点各自待在不同的块里。
- **速度和确定性** — BPE 在短块上运行，候选的合并更少。
- **语义** — 把前导空格和后面的 token 放在一起（所以会得到 \`Ġworld\` 这样的 token），把字母归为一组、数字归为一组、标点/符号归为一组。
- **可逆性** — 和 byte-level 编码配合得很好，保证无损往返。

## GPT-2 用的大致模式（概念上）

在下面的模式里，⎵ 表示一个字面上的空格字符。

- **缩写：** \`'s, 't, 're, 've, 'm, 'll, 'd\`（只匹配小写：\`CAN'T\` → \`⟦CAN⟧⟦'⟧⟦T⟧\`）
- **可选的前导空格 + 字母：** \`⎵?\\p{L}+\`
- **可选的前导空格 + 数字：** \`⎵?\\p{N}+\`
- **可选的前导空格 + 其他符号：** \`⎵?[^\\s\\p{L}\\p{N}]+\`
- **空白：** \`\\s+(?!\\S)\` 取走一串空格，但把最后一个留给下一个词；\`\\s+\` 接住剩下的。
  - \`Hello⎵⎵⎵world\` → \`⟦Hello⟧⟦⎵⎵⟧⟦⎵world⟧\`

## 它如何影响 BPE

在原始文本上运行 regex pre-tokenizer → 把每块转成 UTF-8 字节并映射成替身字符（空格 → \`Ġ\`）→ 在每块内部，反复合并 rank 最低的相邻符号对。

> 正则必须在字节映射之前运行：\`Ġ\` 算作字母，所以 \`HelloĠworld\` 不会被切开。

合并不会跨越块的边界。这就是为什么 \`⟦⎵world⟧\`（带着前导空格）是一个单独的块，内部可以合并，但它不会和前面的 token 合并。

**特殊 token：** 原始的 \`encoder.py\` 没有特殊处理：文本 \`<|endoftext|>\` 和其他文本一样被切分，变成 7 个普通 token。\`tiktoken\` 只有在你把它放进 \`allowed_special\` 时，才会把它映射成单个 id 50256（默认会报错）。

## 例子

每个方框是一个块；⎵ 表示一个空格字符。

### 字母归为一组

- \`Hello world\` → \`⟦Hello⟧⟦⎵world⟧\` — 字母待在一起；前导空格粘在下一块上。
- \`AB12\` → \`⟦AB⟧⟦12⟧\` — 字母是一块，数字是另一块。

### 数字归为一组

- \`Room 1234\` → \`⟦Room⟧⟦⎵1234⟧\`
- \`1,234.50\` → \`⟦1⟧⟦,⟧⟦234⟧⟦.⟧⟦50⟧\` — 数字归为一组，但逗号和句点是单独的符号块。

### 标点/符号归为一组

- \`Hello, world!!\` → \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!!⟧\` — 逗号，以及连续的两个感叹号，都是符号块。
- \`Good-morning :)⎵\` → \`⟦Good⟧⟦-⟧⟦morning⟧⟦⎵:)⟧⟦⎵⟧\` — 笑脸和它的前导空格组成一个符号块；末尾的空格自成一个空白块。

### 缩写被特殊处理

- \`can't\` → \`⟦can⟧⟦'t⟧\` — pre-tokenizer 对 \`'s, 't, 're, 've, 'm, 'll, 'd\` 有专门的规则。

### 非 ASCII 符号

- \`Hi — ok ★\` → \`⟦Hi⟧⟦⎵—⟧⟦⎵ok⟧⟦⎵★⟧\` — 破折号和星号是非 ASCII 符号；每个都和它的前导空格一起成为单独的块。

## 核心思想

> 它先用正则把原始文本切成块：字母放一起、数字放一起、符号放一起，任何前导空格都跟着下一块。然后 BPE 合并只在每块内部进行——不跨越块的边界。

## 参考资料

1. **GPT-2 参考编码器（包含确切的正则）**

   OpenAI 原始的 GPT-2 encoder.py，定义了预切分模式和 byte-unicode 映射表。

   [openai/gpt-2 encoder.py ↗](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

2. **GPT-2 技术报告（输入表示）**

   Radford et al.，§2.2 "Input Representation" — 解释了为什么要阻止 BPE 跨字符类别合并。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

3. **tiktoken**

   OpenAI 的生产级 tokenizer；它的 "gpt2" 编码使用同样的模式。

   [tiktoken ↗](https://github.com/openai/tiktoken)

`,
  },
  lossless_round_trip_notes: {
    title: "无损往返（Lossless round-trip）",
    md: `
**"无损往返"的意思是：把文本编码成 token ID，再把这些 ID 解码回文本，得到的字节和最初的完全一样：没有任何改动或丢失。**

形式化地写：

$$
\\mathtt{decode}(\\mathtt{encode}(s)) = s
$$

## 为什么 GPT-2（byte-level BPE）满足这一点

1. 它直接在原始字节上操作（不转小写、不做规范化、不去掉空格）。
2. 任何字节序列都能被表示。
3. 解码恰好是编码的逆过程。
`,
  },
  gpt2_merge_ranks_notes: {
    title: "GPT-2 merge ranks（合并优先级）",
    md: `
## Byte-Pair Encoding（BPE）回顾

从一串基本符号开始（对 GPT-2 来说是字节）。BPE 模型有一个从大语料中学到的符号对合并列表，是一条一条学出来的：最常见的符号对最先合并（rank 0），然后是更新后语料中最常见的符号对（rank 1），依此类推。GPT-2 学了 50,000 条合并规则；它的 rank 0 是 \`Ġ t\`。

## 为什么 rank 很重要

编码时，tokenizer 反复查看相邻的符号对，在列表中出现的符号对里，合并 rank 数字最小（优先级最高）的那一对。更新序列后重复这个过程，直到没有可合并的符号对为止。最后合并出的块就是你的 token。

## \`tiktoken\` 存的是什么

GPT-2 的 tokenizer 自带一个映射，实际上就表示了这些 rank。在实践中：

- rank 越低 ⇒ 合并规则越早学到（越常见）⇒ 比 rank 更高的规则先应用。
- 这让 tokenization 是确定性的，并且和 GPT-2 训练时一致。

## 和 token ID 的关系

每条合并规则都会产生一个新的词表条目。GPT-2 的 50,257 个 id 是这样排布的：

- **id 0–255：** 256 个单字节（可打印字节排在前面，所以 \`!\` = 0）；
- **id 256–50255：** 50,000 条合并，按 rank 顺序排列，所以 rank 为 $r$ 的合并结果的 id 是 $256 + r$；
- **id 50256：** 特殊 token \`<|endoftext|>\`。

\`tiktoken\` 把这些存成一张 token rank 表，其中 token 的 rank 就等于它的 id，合并时优先选择合并结果 rank 最低的那一对。这和上面的顺序是一样的。

## 小型玩具例子（虚构的）

文本：\`banana\` → 字节 → 初始符号：\`⟦b⟧⟦a⟧⟦n⟧⟦a⟧⟦n⟧⟦a⟧\`

假设合并规则（带 rank）如下：

- rank 0：\`(a, n) → an\`
- rank 1：\`(an, a) → ana\`
- rank 2：\`(b, an) → ban\`

编码过程：

1. Rank 0 \`(a, n)\` 是现有符号对中 rank 最低的；把它出现的每一处都合并 → \`⟦b⟧⟦an⟧⟦an⟧⟦a⟧\`
2. 现在的符号对：\`(b, an)\` rank 2，\`(an, an)\` 没有规则，\`(an, a)\` rank 1 ⇒ \`⟦b⟧⟦an⟧⟦ana⟧\`
3. 现在的符号对：\`(b, an)\` rank 2，\`(an, ana)\` 没有规则 ⇒ \`⟦ban⟧⟦ana⟧\`
4. 没有可用的合并了 → 最终 token：\`⟦ban⟧⟦ana⟧\`

> 真实的 GPT-2 tokenizer 恰好给出同样的切分：\`banana\` → \`[3820, 2271]\`。

## 空格的处理

GPT-2 是 byte-level 的。空格是字节的一部分，很多 token 把"空格+单词"编码成一个 token（在可视化中通常显示为带前导 \`Ġ\`）。这些 token 来自特定的合并规则及其 rank。

`,
  },
  input_data_notes: {
    title: "输入数据（文本）",
    md: `
## 它是什么

原始文本字符串（文档、prompt），在进入模型之前会先被 tokenize。

## 常见的预处理

- 如有需要，规范化空白/换行。
- 确定最大序列长度（$T$）和切分策略（例如滑动窗口、按段落分块）。
- 划分成 train/val/test，避免数据泄漏。

## Batch 的形状

- tokenize 之后，输入是 **token ID**：$(B, T)$。
- 可选的 **attention mask**：$(B, T)$，用来标记 padding（GPT-2 只做 causal 训练时一般不用，但在一般场景下有用）。
`,
  },
  gpt2_notes: {
    title: "GPT-2",
    md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  `,
  },
  gpt2_inference_notes: {
    title: "GPT-2 推理",
    md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  
`,
  },
  dimension_change_notes: {
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
  linear: {
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
  embedding_tok: {
    title: "Token Embedding",
    md: `
**把 token ID 映射成稠密向量。**

## 形状

- **权重矩阵：** $W \\in \\mathbb{R}^{\\text{vocab\\_size} \\times n\\_embd}$，词表里每个 token 一行。
- **输入 ID：** $\\text{input\\_ids} \\in \\mathbb{Z}^{B \\times T}$
- **Embedding 查表：** $\\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$

### 参数量

$$
\\text{params} = \\text{vocab\\_size} \\times n\\_embd
$$

> GPT-2 small：$50{,}257 \\times 768 \\approx 3.86 \\times 10^7$

## 训练

- $W$ 通过反向传播学习。
- 只有 batch 中出现的 token 对应的行会被更新。
- GPT-2 把 $W$ 和输出 softmax 的权重（\`lm_head\`）绑在一起。[什么是 weight tying？](#info:weight_tying_notes)

## 为什么用 embedding，而不是 one-hot 编码？

one-hot 的 token 向量长度是 $\\text{vocab\\_size}$，而且极其稀疏。\`nn.Embedding\` 等价于用 one-hot 向量乘以 $W$，但它：

- 避免创建巨大的稀疏向量，
- 学到稠密的、带语义的表示，相关的 token 在 embedding 空间里会靠得很近。

## 参考资料

1. **PyTorch 文档**

   \`nn.Embedding\` 把 embedding 定义为一个查找表，它 "stores embeddings of a fixed dictionary and size"，并且 "retrieved using indices."

   [nn.Embedding ↗](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

2. **GPT-2 技术报告（Input Representation）**

   把 GPT-2 的输入描述为 token embedding + position embedding（求和），说明 token id 在进入网络堆叠之前，先被转换成 embedding。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

`,
  },
  broadcasting: {
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
  emb_dropout_notes: {
    title: "Embedding Dropout",
    md: `
**在 token embedding 和位置 embedding 相加之后，GPT-2 会先过一个 dropout 层**，再把这个表示送进第一个 Transformer block。

## 为什么在这里用 dropout？

- 在训练时随机把 embedding 向量的一部分置零，防止过拟合。
- 迫使模型不过分依赖某些特定的 embedding 维度。
- 在输入阶段就提供正则化，和 Transformer block 内部的 dropout（例如 attention 层和前馈层之后的 dropout）互相补充。

## 例子

假设 token embedding 和位置 embedding 相加之后，某个 token 得到这个 6 维向量：

$$
h = [0.8, -0.5, 1.2, 0.0, -0.3, 0.9]
$$

这就是第一个 Transformer block 的原始输入。

### 应用 dropout（比如 $p = 0.5$）

Dropout 以概率 $p$ 随机"遮掉"一些维度。保留下来的维度会乘以 $1/(1-p)$ 重新缩放，以保持期望不变。

Dropout mask 示例：

$$
m = [1, 0, 1, 0, 1, 1]
$$

应用 mask 并缩放（缩放因子 = 2，因为 $1/(1-0.5) = 2$）：

$$
h' = h \\odot m \\times 2 = [1.6, 0.0, 2.4, 0.0, -0.6, 1.8]
$$

所以第二维和第四维被"丢掉"（置零），其余的都翻倍。

## 为什么这有帮助

- **提高泛化能力、防止过拟合：** 不让模型把自己和特定维度"焊死"，帮助它学到更稳健的模式。如果模型学会了过分依赖，比如说，第 3 维，dropout 会迫使它也去用其他维度，因为第 3 维在训练中可能会消失。
- **对 embedding 做正则化：** token embedding 不会编码脆弱的、只靠单一维度的"捷径"；它们需要把有用的信号更均匀地分散开。

## 参考资料

1. **原始 Transformer 论文（dropout 在这里被定义）**

   Vaswani et al. 明确写道："we apply dropout to the sums of the embeddings and the positional encodings."。GPT-2 沿用了这个做法（使用学习得到的绝对位置）。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)
`,
  },
  torch_ones_notes: {
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
  torch_tril_notes: {
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
  unsqueeze_notes: {
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
  causal_analogy_notes: {
    title: "因果类比",
    md: `
[因果类比](#info:causal_analogy_notes) · [Attention 推理示例](#info:attn_inference_example_notes)

**下面用一个类比来说明 GPT-2 的多头 self-attention 块是怎么工作的，帮你建立直觉。**

## 场景

一排人，序列里每个位置 $t$ 站一个人。每个人手里拿着笔记 $x_t$（对应位置 $t$ 上那个 token 的 embedding）。

## 1. 用笔记做三张卡片：\`c_attn(x) → q, k, v\`

- **Query（$q$）：** 我现在在找什么（由 $W_q$ 生成）。
- **Key（$k$）：** 我能提供什么，也就是别人能用什么来匹配到我（由 $W_k$ 生成）；它只是用来匹配的索引。
- **Value（$v$）：** 如果你来听我，我会分享什么（由 $W_v$ 生成）；这是真正被复制/混合进别人表示里的内容。

### 为什么要把 Query 和 Key 分开？

考虑一个更简单的设计：不要 $W_q$ 和 $W_k$，直接用 $x$ 打分：$\\text{score}(i \\to j) = x_i \\cdot x_j$。这样有两个问题。

**1. Attention 会是对称的。** 点积是对称的，$x_{\\text{bank}} \\cdot x_{\\text{river}} = x_{\\text{river}} \\cdot x_{\\text{bank}}$，所以 bank 对 river 的关注程度必须和 river 对 bank 的完全一样。但语言里的关系常常是单向的：bank 需要 river 来判断自己是哪个意思，而 river 本身意思就很清楚。把 q 和 k 分开后，两个方向的分数用的是不同的向量，可以不一样。一个 2 维的例子：

- $q_{\\text{bank}} = [0, 2]$，$k_{\\text{bank}} = [1, 0]$
- $q_{\\text{river}} = [0, 0]$，$k_{\\text{river}} = [0, 2]$
- bank → river：$q_{\\text{bank}} \\cdot k_{\\text{river}} = 0 \\cdot 0 + 2 \\cdot 2 = 4$（强关注）
- river → bank：$q_{\\text{river}} \\cdot k_{\\text{bank}} = 0 \\cdot 1 + 0 \\cdot 0 = 0$（不关注）

（在 causal mask 下，river 本来就看不到后面的 bank；这里要说明的是，任意两个词之间的分数可以不对称。）

**2. 每个词会主要关注它自己。** 如果直接用 $x$，一个词对自己的分数是 $x \\cdot x = \\|x\\|^2$，通常比它对其他任何词的分数都大。每个位置都会偏向自己，或者偏向和自己**相似**的词。但 bank 要找的不是另一个 bank；它需要 river 或 money 这类**不同**的、能补充信息的词。分开的 $W_q$ 和 $W_k$ 从 $x$ 里挑出不同的特征，于是模型可以学到"我的 q 去匹配那些 k 能补充我的词"。

> **简单说：** 把 q 和 k 分开，分数就可以**不对称**（A 关注 B 不代表 B 也关注 A），模型就能学会去找那些**和自己不同、但对自己有用**的词。

### 为什么要把 Key 和 Value 分开？

Key 用来寻址（相似度搜索），Value 用来承载内容。你通过 $q \\cdot k$ 选择*听谁的*，再通过 $v$ 从这些人那里复制有用的内容。

能不能用一个向量同时干两件事：既负责被找到，又负责提供内容？不行：*让你被找到的东西*和*你交出去的东西*是两回事。看这句：

> The cat sat on the mat because it was tired.

"it" 所在的位置需要弄清楚 "it" 指的是谁：

- **it 的 q：** 我是一个代词，在找前面一个单数、有生命的名词。
- **cat 的 k：** 我是单数、有生命的名词。→ 匹配得很好。
- **mat 的 k：** 我是单数、无生命的名词。→ 匹配得比较弱。
- **cat 的 v：** "cat" 的含义（一种动物，毛茸茸，会累……）。→ 匹配上之后，传给 "it" 的就是这个。

cat 的 k 和 v 装的是**不同的东西**：

- **k** 装的是让 cat **被找到**的特征：单数、有生命、名词。这些是用来匹配的标签。
- **v** 装的是找它的人真正**需要**的东西："cat" 的含义，这样 "it" 才知道累的是一只猫。

如果 $k = v$，"it" 拿到的就只有"单数、有生命、名词"这些标签，而不是 "cat" 本身。用下面图书馆的说法：你用索引卡来搜索，但带回家的是书的正文。

> **简单说：** q 是我在找什么，k 是别人凭什么能找到我，v 是找到我之后你能拿到什么。三者都来自同一个 $x$，但各有自己的 $W$，挑出各自需要的特征。q 和 k 只决定**关注谁**；v 才是真正**被搬运**的信息。

### 图书馆类比

- **Query：** 你的检索请求，比如"我想找关于河流地理的书"。
- **Key：** 每本书的索引卡，写着书名和关键词，用来和你的请求做**匹配**。
- **Value：** 书的正文。一本书匹配上之后，你**带走**的就是它。

匹配用的是 q 和 k；带走的是 v。

### 例子："I went to the river bank"

"bank" 有歧义（河岸还是银行），而它的 token embedding 无论哪种意思都一样。看 "bank" 所在的位置：

- **bank 的 q：** 我是个有歧义的词；我在找能告诉我自己是哪个意思的词。
- **river 的 k：** 我是一个地理/自然类名词。→ 和 bank 的 q 匹配得很好。
- **went 的 k：** 我是一个表示移动的动词。→ 匹配得不好。
- **river 的 v：** 关于河流、水、地理的信息。→ 因为 river 匹配上了，传给 bank 的就是这个。

后面的步骤这样使用它们：

- 第 3 步用 **$q \\cdot k$** 给相关性打分：bank 的 q 和前面每个词的 k 做点积；
- 第 7 步以这些分数为权重，把各个 **v** 混合起来，交给 bank。

> 两点提醒。上面的描述（"我在找……"、"我是地理类名词"）只是帮助理解的比喻：真实的 q、k、v 各自只是 768 个数字，它们编码什么是训练中学出来的，不是人为设计的。而且**每个 token 都有这三样**：bank 用自己的 q 去询问前面的词，而它自己的 k 和 v 则被后面的词匹配和读取。

## 2. 分成若干委员会（heads）：reshape 成 $(B, n\\_head, T, hs)$

每个委员会（head）用自己的 $W_q, W_k, W_v$ 并行工作，各自关注不同的视角（语法、长距离线索等）。

每个张量是怎么拆分成 head 的：[Query View](#info:query_view_notes) · [Key View](#info:key_view_notes) · [Value View](#info:value_view_notes)

### 为什么用 12 个 head 而不是一个？

**一个 head 只能朝一个方向看。** 在单个 head 里，每个位置只得到**一组** attention 权重（softmax 之后和为 1），并把各个 v 混合成**一个**结果。但一个词常常同时需要好几种信息：

> The tired dog that chased the cat was sleeping

"was" 所在的位置想知道：

- **主语是谁？** dog，要跳过 "that chased the cat"；这决定了用 *was* 而不是 *were*。
- **主语处于什么状态？** tired。
- **前一个词是什么？** cat，有助于理解局部结构。

只有一个 head 时，这三件事要争**同一组**权重：给 dog 的权重多了，给 tired 的就少了。混合出来的结果是所有东西的模糊平均，三件事哪件都做不好。

**12 个 head = 12 种独立的 attention 模式。** 每个 head 有自己 64 维的 q、k、v，独立地做 attention：

- head 3 找主语 → 几乎全部权重给 dog；
- head 7 看前一个词 → 几乎全部权重给 cat；
- head 9 找修饰语 → 大部分权重给 tired；
- ……

每个 head 都可以**专注于一件事**，互不干扰。把 12 个结果拼接起来，"was" 就同时拿到了所有这些信息。

> **类比：** 一个人带着一个问题读这句话。十二个人读同一句话，每人带着不同的问题（找主语、看前一个词、找修饰语……），最后把答案汇总起来。

两点说明：

1. **不需要额外代价。** $12 \\times 64 = 768$，所以参数量和计算量与一个 768 维的 head 大致相同。同样的预算被分成十二份，换来十二种不同的视角。
2. **分工是学出来的，不是指定的。** 上面"head 3 找主语"只是举例。每个 head 最终做什么由训练决定，有些 head 很难解释。但确实存在分工明确的 head：在真实的 GPT-2 small 里，第 0 层的 head 7 主要关注最近的几个 token，去掉位置信息后它就不再起作用。

## 3. 给"听谁的"打分：\`att_logits = q @ k^T / sqrt(hs)\`

每个人拿自己的 query 和所有人的 key 比较，得到匹配分数；除以 $\\sqrt{hs}$ 让数值保持平稳，免得 softmax 爆掉。

$T \\times T$ 分数矩阵长什么样：[Q @ K 乘积](#info:q_k_product_notes)

## 4. 因果规则（不能往后看）：把未来的位置 mask 成 $-\\infty$

这保证了语言模型的分解形式

$$
p(x_{1:T}) = \\prod_{t=1}^{T} p(x_t \\mid x_{\\le t-1})
$$

为什么需要它：

- **训练时不泄漏：** 虽然为了速度我们并行计算所有位置，mask 保证位置 $t$ 永远不会用到 $t+1, \\dots$。
- **和生成方式一致：** 推理时我们从左到右生成；训练时保持因果性，学到的行为就和模型的实际使用方式一致。
- **防止作弊：** 没有 mask，模型就能偷看答案（未来的 token）来作弊。

## 5. 把分数变成"倾听权重"：\`att = softmax(att_logits, dim=-1)\`

现在每个人在允许看的更早位置上都有一个概率分布：*我给每个来源多少注意力？* 每行和为 1，得到可以解释的混合权重。

## 6. 在部分连接断开的情况下练习：\`attn_drop(att)\`

训练时随机丢掉一些连接，增强鲁棒性，避免过度依赖某一条路径。

## 7. 用 Value 拼出你的个性化总结：\`y = att @ v\`

用你的 attention 权重把别人的 Value 向量混合起来（例如 $0.6\\,v_3 + 0.3\\,v_7 + \\cdots$）。

> 记住：Key 帮你选择；Value 是你拿走的东西。

## 8. 把各委员会的结果汇总并统一格式：合并 heads，然后 \`proj(y)\`

把各个 head 的输出重新拼回每个位置一个向量，投影到模型宽度，干净地送进残差流。

## 全局视角

> 在每个位置 $t$：通过 Query-Key 选出更早的位置（寻址），通过 Value 复制有用的内容，结合多种视角（heads），并且整个过程遵守与从左到右生成相一致的因果规则。
`,
  },
  attn_inference_example_notes: {
    title: "Attention 推理示例",
    md: `
[因果类比](#info:causal_analogy_notes) · [Attention 推理示例](#info:attn_inference_example_notes)

**这个例子展示单个 attention head 如何根据句子上下文推断出正确的动词形式（\`were\` 还是 \`was\`）。**

输入句子：**The dogs that barked loudly were tired.**

我们关注的是：根据前缀 \`The dogs that barked loudly\` 预测 token \`were\`（位置 6）。

## 发生了什么（单个 head，简化版）

目标：在 \`were\` 和 \`was\` 之间做选择。

为此，我们需要知道从句真正的主语，才能推断出正确的动词形式。在这个例子里，真正的主语是 \`dogs\`（复数），尽管中间隔着一个从句 "that barked loudly"。

### 说明

1. **模型其实并不知道这是一个动词位置。** 真实模型里没有内置词性标签或语法规则。模型是在训练中学会识别"下一个很可能是动词"的上下文。模型有一个隐藏状态 $x_t$，它是下面这些东西的函数：

   - 之前所有的 token 和这个 token 本身
   - 位置 embedding $p_t$
   - 很多层 self-attention 和前馈网络

   隐藏状态 $x_t$ 编码了诸如"这里通常是复数主语后面跟动词的位置……"这样的上下文特征，但它也可能编码别的东西：

   - **接下来可能是标点。** 这种情况下，模型可能预测一个逗号："The dogs that barked loudly, …"。
   - **其他的续写方式。** 这种情况下，模型可能预测别的动词，比如 \`seemed\` 或 \`are\`："The dogs that barked loudly are tired."

   但在这个玩具例子里，我们只关注模型为什么预测 \`were\` 而不是 \`was\`。

2. **假设：所有参数都已训练好。** 这是一个推理的例子，所以投影矩阵 $\\mathbf{W}_q, \\mathbf{W}_k, \\mathbf{W}_v$（以及输出投影 $\\mathbf{W}_o$）、MLP、LayerNorm、embedding 和 LM head 都是固定的。例子里的数字是示意性的，但它们反映了训练好的参数在推理时通常会产生的效果。

## 步骤 A：Value 携带有用的特征

假设这个 head 的 Value 向量包含：

- **复数性**（单数 $-1 \\,\\dots\\, +1$ 复数）
- **主语性**（这个 token 适不适合做主语中心词？）

> 这是一个**简化**。在真实模型里，Value 向量是**一个维度为 head size 的、学出来的高维表示**（例如 GPT-2 small 有 768 维、12 个 head，所以 head size 是 64）。所以在真实模型里它并不真的是两个数字的特征；我们只是把这个想法投影到二维，来说明这个 head 可能编码了什么。

玩具 value：

- \`The\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`dogs\`：$\\mathbf{V} \\approx \\begin{bmatrix}+1.0 \\\\ +0.9\\end{bmatrix}$ ← 复数名词，很可能是主语
- \`that\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.1\\end{bmatrix}$
- \`barked\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`loudly\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$

## 步骤 B：Query 和 Key 给"听谁的"打分

这个 head 把 Query $\\mathbf{q}_5$ 和每个 Key $\\mathbf{k}_j$（$j = 1 \\ldots 5$）做比较。编造的 attention logits（已经乘过 $1/\\sqrt{hs}$）：

- 对 \`The\`：$-0.5$
- 对 \`dogs\`：$+2.57$ ← 强匹配
- 对 \`that\`：$+0.3$
- 对 \`barked\`：$-0.2$
- 对 \`loudly\`：$-0.1$

> 这些数值是编出来的，用来展示规律（例如主语得到最高分）。在真实模型里，它们来自与训练好的 $\\mathbf{W}_q, \\mathbf{W}_k$ 计算出的点积，再加上 mask。

然后用 softmax 得到 attention 权重（和为 1）：

- $w(\\mathtt{The}) = 0.04$
- $w(\\mathtt{dogs}) = 0.78$
- $w(\\mathtt{that}) = 0.08$
- $w(\\mathtt{barked}) = 0.05$
- $w(\\mathtt{loudly}) = 0.05$

## 步骤 C：用这些权重混合 Value（\`att @ v\`）

加权平均得到这个 head 在位置 5 的输出：

$$
\\mathbf{y}_5
= 0.04\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.78\\cdot\\begin{bmatrix}+1.0, +0.9\\end{bmatrix}
+ 0.08\\cdot\\begin{bmatrix}0, 0.1\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
+ 0.05\\cdot\\begin{bmatrix}0, 0\\end{bmatrix}
\\\\[4pt]
\\mathbf{y}_5 \\approx \\begin{bmatrix}+0.78, +0.71\\end{bmatrix}
$$

> **解读：** 位置 5 现在带有很强的"复数 + 主语"信号。

## 步骤 D：为什么这会把 logits 往正确的方向推

把 LM head 看成很多行（每个词表项一行），每行和最终的隐藏向量做点积。只看这两个动词对应的行：

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4, +0.5\\end{bmatrix}$（偏好复数和主语）
- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1, +0.2\\end{bmatrix}$（惩罚复数）

这个 head 的贡献带来的点积：

$$
\\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot 0.78 + 0.5\\cdot 0.71 \\approx +1.45 \\\\[4pt]
\\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot 0.78 + 0.2\\cdot 0.71 \\approx -0.72
$$

和其他 head/层的结果合起来，再在整个词表上做 softmax 之后，\`were\` 的概率远高于 \`was\`。

## 为什么 \`dogs\` 在这里影响力很大

- **句法角色：** 它是包含这个动词的从句的真正主语，即使隔着 "that barked loudly"。
- **训练压力：** 在数以百万计的样本上，梯度推动参数，让动词的 Query 和支配它的主语的 Key 高度对齐。这使得主语上的 attention 权重很高。
- **Value 携带的内容：** 主语的 Value 恰好带有帮助选出正确动词形式的特征（复数性、主语性）。如果模型预测错了形式，梯度会增大正确主语的 $q \\cdot k$，并调整它的 $\\mathbf{V}$，让它更有信息量。

## 如果换成单数 \`dog\` 呢？

把 \`dogs\`（复数）换成 \`dog\`（单数）后，负责主谓一致的同一个 attention head 现在会把单数的证据往前传，于是 LM head 给 \`was\` 的分数会高于 \`were\`。

下面是同样的玩具推演，只是换成了单数。前缀：\`The dog that barked loudly\`。

### A) Value 携带特征

玩具特征：[复数性, 主语性]。

- \`The\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`dog\`：$\\mathbf{V} \\approx \\begin{bmatrix}-1.0 & +0.9\\end{bmatrix}$ ← 单数名词，很可能是主语
- \`that\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.1\\end{bmatrix}$
- \`barked\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`loudly\`：$\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$

### B) Query 和 Key 决定听谁的

位置 5 的 query（用于预测下一个 token）和位置 1 到 5 的 key 做比较。和前面一样，它和主语强匹配：

- 对 \`The\`：$-0.5$
- 对 \`dog\`：$+2.57$ ← 强匹配
- 对 \`that\`：$+0.3$
- 对 \`barked\`：$-0.2$
- 对 \`loudly\`：$-0.1$

Softmax → attention 权重：

$$
w(\\mathtt{The})=0.04,\\ \\ w(\\mathtt{dog})=0.78,\\ \\ w(\\mathtt{that})=0.08,\\ \\ w(\\mathtt{barked})=0.05,\\ \\ w(\\mathtt{loudly})=0.05
$$

### C) 用这些权重混合 value（\`att @ v\`）

$$
\\mathbf{y} \\approx 0.78\\cdot\\begin{bmatrix}-1.0, +0.9\\end{bmatrix} + 0.08\\cdot\\begin{bmatrix}0.0, 0.1\\end{bmatrix} \\approx \\boxed{\\begin{bmatrix}-0.78, +0.71\\end{bmatrix}}
$$

> **解读：** 位置 5 上有很强的单数信号，外加主语信号。

### D) LM head 的行（玩具）和得到的 logits

把每个词表 token 看成一行，和隐藏向量做点积。

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4 & +0.5\\end{bmatrix}$（偏好复数和主语）

  $$
  \\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot(-0.78) + 0.5\\cdot 0.71 \\approx -0.74
  $$

- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1 & +0.2\\end{bmatrix}$（惩罚复数，对主语无所谓）

  $$
  \\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot(-0.78) + 0.2\\cdot 0.71 \\approx +1.00
  $$

> 所以 $\\operatorname{logit}(\\mathtt{was}) > \\operatorname{logit}(\\mathtt{were})$。把所有 head/层合起来、在整个词表上做 softmax 之后，在单数主语的上下文里模型会明显偏向 \`was\`。
`,
  },
  gradient_notes: {
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
  positional_embeddings_example_notes: {
    title: "位置 Embedding 示例",
    md: `
**一个很小的 2 维例子：没有位置信息时，交换两个 token 只会让 attention 分数换个排列；有了位置信息，分数本身就会改变。**

## 设定（没有位置信息）

令 token embedding 为 2 维向量：

$$
e(A) = \\begin{bmatrix}1, 0\\end{bmatrix}, \\quad e(B) = \\begin{bmatrix}0, 1\\end{bmatrix}
$$

使用恒等投影，所以

$$
Q = X, \\quad K = X, \\quad S = QK^\\top = XX^\\top
$$

对于序列 $[A, B]$：

$$
X = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}, \\qquad S = XX^\\top = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}
$$

序列 $[B, A]$ 只是把 $X$ 的行换了个顺序。如[位置 embedding 笔记](#info:embedding_pos)所示，分数变成 $S' = PSP^\\top$。

> 这些数字和 $S$ 里的一样，只是行/列换了位置。Self-attention 分不清 $AB$ 和 $BA$：它看到的只是一个置换。

## 加上位置 embedding

令位置向量为

$$
p_0 = \\begin{bmatrix}1, 1\\end{bmatrix}, \\quad p_1 = \\begin{bmatrix}-1, 1\\end{bmatrix}
$$

输入改为 $h_t = e(\\text{token}_t) + p_t$。

### 序列 $[A, B]$（$A$ 在位置 0，$B$ 在位置 1）

$$
h_0 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}2, 1\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}-1, 2\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix} \\\\[4pt]
S = hh^\\top = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix}\\begin{bmatrix}2 & -1\\\\1 & 2\\end{bmatrix} = \\begin{bmatrix}5 & 0\\\\0 & 5\\end{bmatrix}
$$

### 序列 $[B, A]$（$B$ 在位置 0，$A$ 在位置 1）

$$
h_0 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}1, 2\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}0, 1\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix} \\\\[4pt]
S' = hh^\\top = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix}\\begin{bmatrix}1 & 0\\\\2 & 1\\end{bmatrix} = \\begin{bmatrix}5 & 2\\\\2 & 1\\end{bmatrix}
$$

> 现在 $S'$ 不再只是 $S$ 的行/列置换；点积的数值本身变了。没有位置信息时，打乱顺序只会置换 attention 矩阵；有了位置信息，数字本身会变，于是模型对顺序变得敏感。
`,
  },
  post_norm_scaling_example_notes: {
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
  embedding_pos: {
    title: "位置嵌入（Positional Embedding）",
    md: `
## 它是什么

Transformer 需要顺序信息，因为 self-attention 本身没有"序列"的概念。位置嵌入根据每个 token 所在的位置，给它一个独一无二的向量。

> GPT-2 的位置嵌入是查找表里的**可学习参数**，和其他权重一样通过反向传播训练。它们不是原始 Transformer 论文里的[固定正弦函数](#info:sinusoidal_pos_notes)。

位置嵌入权重矩阵的形状是：

$$
W_{pos} \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
$$

- $\\text{max\\_toks}$ 是最大序列长度（GPT-2 是 1024）。
- 每一行 $p_j$ 对应序列中的位置 $j$。

## 为什么需要位置嵌入？

直观上，它们让模型对序列中的顺序和位置有感知。但具体是怎么做到的？

### 没有位置信息时

假设我们只有 token embedding $X \\in \\mathbb{R}^{T \\times n\\_embd}$（为简单起见忽略 batch 维度）。Self-attention 计算：

$$
Q = X W_q, \\quad K = X W_k, \\quad \\text{scores} = Q K^\\top
$$

现在用一个置换矩阵 $P$ 打乱序列。新的输入是 $X' = P X$，于是

$$
Q' = X' W_q = P X W_q = P Q \\\\[2pt]
K' = X' W_k = P X W_k = P K
$$

分数变成：

$$
\\text{scores}' = Q' K'^\\top = (P Q)(P K)^\\top = P \\, (Q K^\\top) \\, P^\\top
$$

这些数和之前完全一样，只是行和列的顺序变了。

> 没有位置信息（也没有 mask）时，self-attention 是**置换等变的（permutation-equivariant）**：打乱输入，只会以同样的方式打乱 attention 矩阵。模型能"看到"有哪些 token，却看不到它们的顺序。

### 有位置信息时

加上位置嵌入 $p_t$ 就打破了这种对称性。现在交换两个 token，也会交换它们各自拿到的 $p_t$，所以 $Q$ 和 $K$ 会变。点积里包含了依赖位置的项，因此 $QK^\\top$ 的数值本身会变，而不只是行/列重新排序。模型就是这样对顺序变得敏感的。

还是不太明白？看这个[小例子](#info:positional_embeddings_example_notes)。

### 那 causal mask 呢？

上面的论证假设每个 token 都能 attend 到所有其他 token。GPT-2 用的是 [causal mask](#info:causal_analogy_notes)：位置 $t$ 只能 attend 到位置 $0, \\dots, t$。mask 本身就以两种方式携带了顺序信息：

- **每个位置能看到哪些 token 会变。** 把 \`A B C\` 打乱成 \`C A B\`：\`B\` 原来看到的是 $\\{A, B\\}$，现在看到的是 $\\{C, A, B\\}$，所以它的输出变了。结果不再只是重新排序，因此 causal attention 本来就对顺序敏感。
- **每个位置能看到的 token 数量不同。** 位置 $t$ 总是看到 $t + 1$ 个 token。即使在 \`a a a a\` 中，每个位置看到的 token 种类都一样，数量仍然不同（比如对可见 token 取平均，每个 token 的权重是 $\\frac{1}{t+1}$）。这让模型可以推断出自己的绝对位置。

Haviv et al. (2022) 训练了**完全没有位置编码**的 causal 语言模型，发现它们仍能和标准模型相媲美。探测（probing）显示，这些模型学到了一种隐式的绝对位置感；作者推测，是 causal mask 让每个 token 能推断出自己前面有多少个 token。

> 所以，"模型看不到顺序"这句话只对没有 mask 的 attention 成立。GPT-2 仍然加上了显式的位置嵌入，今天的 LLM 也仍然使用显式的位置信息（通常是 RoPE）。

## 运行时的用法

1. **Token ID → token embedding**

   $$
   \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
   $$

2. **位置 $0, 1, \\dots, T-1$ → 位置嵌入**

   $$
   \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
   $$

3. **通过广播把两者相加**（[什么是广播？](#info:broadcasting)）

   $$
   x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
   $$

### 例子：$n\\_embd = 4$

一个只有 3 个 token 的玩具词表，最多 3 个位置，向量是 4 维的。这些数字是编出来的，方便计算；在真实模型里它们从随机值开始，通过学习得到。

\`\`\`text
tok_emb (3 x 4)                      pos_emb (3 x 4)
id 0  dog    [1.0, 0.0, 0.5, 0.0]    pos 0  [0.1, 0.1, 0.0, 0.0]
id 1  bites  [0.0, 1.0, 0.0, 0.5]    pos 1  [0.0, 0.1, 0.1, 0.0]
id 2  man    [0.5, 0.0, 1.0, 0.0]    pos 2  [0.0, 0.0, 0.1, 0.1]
\`\`\`

**句子 1：** \`dog bites man\`，ids = [0, 1, 2]

\`\`\`text
pos 0  dog    [1.0, 0.0, 0.5, 0.0] + p0 = [1.1, 0.1, 0.5, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  man    [0.5, 0.0, 1.0, 0.0] + p2 = [0.5, 0.0, 1.1, 0.1]
\`\`\`

**句子 2：** \`man bites dog\`，ids = [2, 1, 0]

\`\`\`text
pos 0  man    [0.5, 0.0, 1.0, 0.0] + p0 = [0.6, 0.1, 1.0, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  dog    [1.0, 0.0, 0.5, 0.0] + p2 = [1.0, 0.0, 0.6, 0.1]
\`\`\`

结果说明了什么：

- **同一个 token，不同位置 → 不同的向量。** 句子 1 中位置 0 的 \`dog\` 是 [1.1, 0.1, 0.5, 0.0]；句子 2 中位置 2 的 \`dog\` 是 [1.0, 0.0, 0.6, 0.1]。两者相似（都是 \`dog\`），但不完全相同。
- **同一个 token，同一位置 → 完全相同的向量。** \`bites\` 在两个句子里都在位置 1，两次得到的都是 [0.0, 1.1, 0.1, 0.5]。
- **如果没有位置嵌入**，两个句子会得到同样的三个向量 {dog, bites, man}，只是顺序不同。现在 "dog bites man" 和 "man bites dog" 的输入真正不同了。

形状（两个句子放在一个 batch 里，$B = 2$）：

\`\`\`text
idx           = [[0, 1, 2],
                 [2, 1, 0]]       (2, 3)
tok_emb(idx)                      (2, 3, 4)
pos_emb(pos)   pos = [0, 1, 2]    (3, 4)
  [None, :, :]                    (1, 3, 4)   broadcast to every sequence in the batch
x = sum                           (2, 3, 4)
\`\`\`

> 两个句子加上的是**同样的** $p_0, p_1, p_2$：位置向量只取决于位置，与句子无关。

## 参数量

$$
\\text{params} = \\text{max\\_toks} \\times n\\_embd
$$

> GPT-2 small：$1024 \\times 768 \\approx 7.86 \\times 10^5$，比 token embedding 表（约 38M 参数）小得多。

## 形状一览

- **权重：** $[\\text{max\\_toks}, n\\_embd]$
- **位置：** $[T]$
- **位置嵌入：** $[T, n\\_embd]$
- **最终的和：** $[B, T, n\\_embd]$

## 近期发展

GPT-2 所用的可学习绝对位置嵌入，在最新的 state-of-the-art LLM 中已经不常用了。这些模型大多转向了相对位置方法，比如 [RoPE（旋转位置编码）↗](https://arxiv.org/pdf/2104.09864) 或 [ALiBi（带线性偏置的 attention）↗](https://arxiv.org/pdf/2108.12409)。

RoPE 是如何一步步工作的：[RoPE](#info:rope_notes)

## 参考文献

1. **Transformer 的位置编码**

   Vaswani et al., "Attention Is All You Need," §3.5。提出用固定正弦函数把位置信息加到 token embedding 上。

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 使用可学习的绝对位置嵌入**

   Hugging Face 的 GPT-2 文档明确指出，GPT-2 使用绝对位置嵌入（而不是正弦编码），所以建议在右侧填充（pad）输入。

   [GPT-2 配置 ↗](https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2)

3. **GPT-2 技术报告（输入表示）**

   Radford et al., "Language Models are Unsupervised Multitask Learners." 描述了输入是 token embedding 与位置嵌入之和，然后再送入层堆叠。

   [GPT-2 技术报告 ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

4. **相对位置编码**

   Shaw et al., "Self-Attention with Relative Position Representations." 提出相对位置嵌入，作为绝对位置嵌入的替代方案。

   [Self-Attention with Relative Position Representations ↗](https://arxiv.org/pdf/1803.02155)

5. **RoPE（旋转位置编码）**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding." 通过旋转 query 和 key，在 self-attention 中编码相对位置。

   [RoPE: Rotary Position Embedding ↗](https://arxiv.org/pdf/2104.09864)

6. **ALiBi（带线性偏置的 attention）**

   Press et al.。给 attention 分数加上与距离成正比的偏置；使模型能外推到更长的长度。

   [ALiBi: Attention with Linear Biases ↗](https://arxiv.org/pdf/2108.12409)

7. **没有位置编码的 causal LM**

   Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information," Findings of EMNLP 2022。表明没有位置编码的 causal LM 仍然有竞争力，并且学到了隐式的位置感，很可能是通过 causal mask。

   [arXiv:2203.16634 ↗](https://arxiv.org/abs/2203.16634)

  `,
  },

  sinusoidal_pos_notes: {
    title: "正弦位置编码（Sinusoidal Positional Encoding）",
    md: `
**固定的 sin/cos 位置向量，由公式算出来，而不是学出来的。**

原始 Transformer（Vaswani et al., 2017）把这些向量加到 token embedding 上，方式和 GPT-2 加它的[可学习位置嵌入](#info:embedding_pos)一样。它们没有任何部分是训练出来的。

## 公式

把 $d$ 个维度两两分成一对。第 $i$ 对有自己的频率 $\\omega_i$：

$$
PE(pos, 2i) = \\sin(\\omega_i \\cdot pos), \\qquad PE(pos, 2i+1) = \\cos(\\omega_i \\cdot pos), \\qquad \\omega_i = \\frac{1}{10000^{2i/d}}
$$

- 当 $i = 0$ 时，$\\omega = 1$：转得最快的一对。
- 随着 $i$ 增大，$\\omega$ 变小，这一对转得越来越慢。最慢的一对波长约为 $10000 \\cdot 2\\pi$。

## 直觉：一个有很多指针的时钟

每一对 $(\\sin, \\cos)$ 就是圆上的一根**指针**。位置每前进一步，指针 $i$ 就转过 $\\omega_i$ 弧度。

- **快指针**（像秒针）在相邻位置之间变化很大，所以能区分相邻的位置。
- **慢指针**（像时针）只有跨越很多位置才会明显移动，所以能区分近和远。

把所有指针一起读出来，就能确定位置，就像秒针、分针、时针合在一起给出精确时间一样。

## 例子：$d = 4$

两根指针：$\\omega_0 = 1$（快）和 $\\omega_1 = 1/10000^{2/4} = 1/100$（慢）。

\`\`\`text
         fast: sin,  cos      slow: sin,  cos
pos 0  [  0.000,  1.000,         0.000, 1.000 ]
pos 1  [  0.841,  0.540,         0.010, 1.000 ]
pos 2  [  0.909, -0.416,         0.020, 1.000 ]
pos 3  [  0.141, -0.990,         0.030, 1.000 ]
\`\`\`

- 快指针每一步都变化很大，所以相邻位置很容易区分。
- 慢指针几乎不动；要上百步才会明显转动。它大致告诉你处在序列的哪一段。

## 关键性质：平移 $k$ 是一个固定的旋转

对每一对：

$$
\\begin{bmatrix} \\sin(\\omega(p+k)) \\\\ \\cos(\\omega(p+k)) \\end{bmatrix}
=
\\begin{bmatrix} \\cos \\omega k & \\sin \\omega k \\\\ -\\sin \\omega k & \\cos \\omega k \\end{bmatrix}
\\begin{bmatrix} \\sin(\\omega p) \\\\ \\cos(\\omega p) \\end{bmatrix}
$$

这个矩阵只依赖 $k$，不依赖 $p$。所以"往回看 $k$ 个位置"在**每个位置上都是同一个线性映射**。论文给出的设计理由正是这一点：$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，这应该让 attention 很容易处理相对位置。

> GPT-2 学到的表最终也有类似的形态：相邻位置彼此接近，向量以多种速度画出平滑的环。区别在于 GPT-2 必须从数据中学出这一点，而正弦编码是直接内置的。

## 和 GPT-2 的可学习嵌入对比

- **参数：** 正弦编码没有参数；GPT-2 的表有 $1024 \\times 768$ 个。
- **超出训练长度：** 公式可以给任意位置算出向量；GPT-2 的表在 1023 之后就没有行了。
- **结构：** 正弦编码是设计出来的（多种速度的指针）；GPT-2 的是学出来的，最后也形成了类似的平滑环。
- **效果：** Vaswani et al. 也试过可学习位置嵌入，发现两者 "produced nearly identical results."
- **外推：** 他们选择正弦函数，是因为它 "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." 实际上效果很差：对一个没见过的位置有向量，并不意味着模型知道怎么用它。ALiBi 论文直接测量了这一点。

## 近期发展

**RoPE**（Rotary Position Embedding，旋转位置编码）是今天大多数 LLM 使用的方法，它用的是同一组频率 $\\omega_i$。区别在于 RoPE 不往输入里加位置向量，而是在每一个 attention 层内部，把 $q$ 和 $k$ 按与各自位置成正比的角度**旋转**，使得 $q \\cdot k$ 只依赖两个位置之间的**相对距离**。

RoPE 是如何一步步工作的：[RoPE](#info:rope_notes)

## 参考文献

1. **Attention Is All You Need, §3.5**

   Vaswani et al., 2017。定义了正弦编码，与可学习嵌入做了比较，并解释了这一选择。

   [arXiv:1706.03762 ↗](https://arxiv.org/abs/1706.03762)

2. **ALiBi（带线性偏置的 attention）**

   Press et al.。测试了正弦、旋转等位置方法外推到比训练时更长的序列时效果如何。

   [arXiv:2108.12409 ↗](https://arxiv.org/abs/2108.12409)

3. **RoPE**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding."

   [arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)

    `,
  },

  rope_notes: {
    title: "RoPE（旋转位置编码）",
    md: `
**RoPE（Rotary Position Embedding，旋转位置编码）是今天大多数 LLM 编码位置的方式。GPT-2 没有用它（GPT-2 用的是[可学习位置嵌入](#info:embedding_pos)），但放在 $q \\cdot k$ 这一步来理解最容易。**

## 1. 位置信息放在哪里

**GPT-2：在一开始加到 $x$ 上，只加一次。**

\`\`\`text
x = token embedding + position embedding    <- position mixed into x before block 1 (在 block 1 之前把位置混进 x)
q = x · W_qᵀ,   k = x · W_kᵀ
score = q · k
\`\`\`

**RoPE：$x$ 保持不动；$q$ 和 $k$ 在算分数之前被旋转。**

\`\`\`text
x = token embedding                          <- no position embedding (没有位置嵌入)
q = x · W_qᵀ,   k = x · W_kᵀ
q' = rotate q by an angle set by its position
k' = rotate k by an angle set by its position
score = q' · k'
\`\`\`

三点不同：

- RoPE **不往 $x$ 里加任何东西**：残差流里没有位置向量。
- 旋转发生在**每一个 attention 层内部**，就在 $q \\cdot k$ 之前，而不是只在输入处做一次。
- 只旋转 **q 和 k**，从不旋转 **v**。位置影响的是*该关注谁*，而不是*取走什么内容*。

## 2. 为什么旋转能编码位置：二维图景

把 $q$ 和 $k$ 想成平面上的两根箭头。它们的点积取决于两者的长度和**它们之间的夹角**。

现在假设位置 $m$ 上的向量被旋转 $m \\times \\theta$，其中 $\\theta = 10°$：

- 位置 5 的 q 转 50°，位置 3 的 k 转 30°：两者之间的夹角变化了 $50° - 30° = 20°$。
- 把两者都移到很远，q 在 105、k 在 103：q 转 1050°，k 转 1030°，变化仍然是 $20°$。

> **旋转之后，分数只取决于两个位置之间的距离**（$5 - 3 = 2$），而与它们具体在哪里无关。"前一个词"或"往前数两个词"在位置 5 和位置 1000 上用的是同一条打分规则。

## 3. 为什么"只依赖相对距离"是个好性质

**语言的大多数规则都是相对的。**

- 形容词通常位于它所修饰名词的**前一个位置**：\`black cat\`。
- \`it\` 通常指代**稍早一点**出现的东西。
- 动词通常出现在主语**之后几个位置**。

这些规则说的都是"往前 $n$ 个位置"，从不说"第 37 个位置"。同一条规则在文本开头和第 800 个 token 处都成立。

**如果用绝对位置，同一条规则得学很多遍。** 假设 \`black cat\` 出现了两次：

\`\`\`text
positions 3, 4      :  black cat
positions 503, 504  :  black cat
\`\`\`

- **只看相对距离（RoPE）：** 两对的距离都是 1，所以分数中的位置部分自动相同。学一次，处处适用。
- **绝对位置：** 模型看到的是"3 和 4"以及"503 和 504"，两对毫不相关的数字。它们其实是同一种关系，这一点必须从数据中学出来。

这就像教"看你前面一个座位"——在每个座位上都适用；对比分别训练"从 3 号座位看 2 号座位"、"从 503 号座位看 502 号座位"，等等。

> **内容依然重要。** "只依赖相对距离"说的是分数中的**位置**部分。q 和 k 仍然是从 token 算出来的，所以同样是距离 1，\`black → cat\` 可能得分很高，而 \`the → of\` 得分很低。RoPE 保证的是：同样的内容、同样的距离，无论出现在哪里，都得到同样的分数。

## 4. 从二维到 64 维：两两成对旋转

每个 head 的 q 和 k 有 $hs = 64$ 维。RoPE 把它们分成 **32 对**，把每一对当作一个二维向量，在各自的平面里旋转：

\`\`\`text
q (64 dims) = [ (q0, q1), (q2, q3), (q4, q5), ..., (q62, q63) ]
                 pair 0    pair 1    pair 2         pair 31
\`\`\`

在位置 $m$，第 $i$ 对被旋转角度 $m\\,\\theta_i$，用的就是普通的二维旋转：

$$
(a,\\, b) \\;\\rightarrow\\; \\bigl(a\\cos(m\\theta_i) - b\\sin(m\\theta_i),\\;\\; a\\sin(m\\theta_i) + b\\cos(m\\theta_i)\\bigr)
$$

每一对都有**自己的速度**：

$$
\\theta_i = \\frac{1}{10000^{2i/64}}, \\qquad i = 0, 1, \\dots, 31
$$

- **第 0 对：** 每个位置 $\\theta = 1$ 弧度，约 57.3°。每 $2\\pi \\approx 6.3$ 个位置转一整圈。
- **第 16 对：** $\\theta = 0.01$ 弧度，约 0.57°。每 ≈ 628 个位置转一整圈。
- **第 31 对：** $\\theta \\approx 0.00013$ 弧度。每 ≈ 47,000 个位置转一整圈。

> **弧度。** 这里的角度用弧度表示，和 \`torch.sin\` / \`torch.cos\` 一样。一整圈是 $360° = 2\\pi \\approx 6.283$ 弧度，所以 1 弧度 $= 180°/\\pi \\approx 57.3°$：即弧长等于半径时所对的角。

**为什么要把快的对和慢的对混在一起？**

- **快的对**在相邻位置之间转得多，所以能区分"相邻"和"隔两个"。但它们转几个位置就绕回来了，所以单靠它们无法区分远距离。
- **慢的对**在相邻位置之间几乎不变，但在长距离上角度会拉开，所以能区分"相隔 100"和"相隔 1000"。

两者合在一起，近和远都能覆盖，就像时钟的秒针、分针和时针。

### q 和 k 按同一规则旋转

两者使用**同一条**规则、**同样的** 32 个速度 $\\theta_i$。只有角度不同，因为角度由各自向量的位置决定：

- 位置 $m$ 的 q：第 $i$ 对转 $m\\,\\theta_i$；
- 位置 $n$ 的 k：第 $i$ 对转 $n\\,\\theta_i$。

同一个 token 自己的 q 和 k 位于同一个位置，所以转过的角度相同。

**为什么规则必须共享。** 在二维例子中（$\\theta = 10°$），位置 5 的 q 和位置 3 的 k 分别转 50° 和 30°，差是 $(5 - 3) \\times 10° = 20°$。两个角度都是"位置 × 同一个 $\\theta$"，所以绝对位置抵消掉，只剩下距离。

如果 q 用 10°，而 k 用 7°：

- 位置 5 的 q 和位置 3 的 k：$50° - 21° = 29°$；
- 位置 105 的 q 和位置 103 的 k：$1050° - 721° = 329°$。

距离相同，结果却不同：分数就不再只依赖相对距离了。

> "同一规则"指的是同一种*旋转*，而不是同一个*向量*。q 和 k 仍然是不同的向量，由 \`c_attn\` 用不同的权重算出；RoPE 只是在之后按各自的位置分别旋转它们。

## 5. 与正弦编码的关系

原始 Transformer 的[正弦编码](#info:sinusoidal_pos_notes)是：

$$
PE(pos, 2i) = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\qquad PE(pos, 2i+1) = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

频率**完全一样**，都是 $1/10000^{2i/d}$，也都是 sin/cos 成对出现。区别在于用法：

- **正弦编码：** sin/cos 值被**加**到 $x$ 上，只在输入处加一次，然后混进残差流。
- **RoPE：** sin/cos 值用来**旋转**（相乘）$q$ 和 $k$，在每一个 attention 层都做，并且不碰其他任何东西。

正弦编码那篇笔记说明了：平移 $k$ 个位置就是一个旋转。RoPE 把这个旋转**直接作用在 q 和 k 上**，所以 $q \\cdot k$ 自动只依赖相对距离。

> 🆕 **base（底数）。** 公式中的 10000 叫作 base（在 Hugging Face 配置里是 \`rope_theta\`）。Llama 2 用 10000；Llama 3 把它提高到 500000，让最慢的几对转得更慢，以支持更长的上下文。

## 6. 超出训练长度

当序列比模型训练时的长度更长时，会发生什么？

### GPT-2：表里没有对应的行

GPT-2 从一张查找表里取位置，每个位置对应一行学出来的向量：

\`\`\`text
pos_emb = nn.Embedding(max_toks, n_embd)    # real GPT-2: 1024 rows × 768
pos_emb(pos)                                 # pos = 0, 1, ..., T-1
\`\`\`

- 位置 0 到 1023 各有自己训练好的一行。
- 位置 1024 没有对应的行，所以查表会报索引错误（index error）。

所以 GPT-2 一次**最多只能看到 1024 个 token**。而且由于每一行都是单独学出来的，加一行新的也没用：它从未被训练过，模型不会知道它是什么意思。

### GPT-2 怎样生成超过 1024 个 token：滑动窗口

生成时，上下文还包括模型目前已经生成的 token，所以 prompt + 生成的 token 很快就会达到上限。因此生成代码在每一步都把输入裁剪成**最近的** 1024 个 token（nanoGPT 的 \`generate\` 会裁剪到 \`block_size\`）：

\`\`\`text
idx_cond = idx[:, -1024:]     # keep only the last 1024 tokens (只保留最后 1024 个 token)
logits = model(idx_cond)
\`\`\`

这**不是**把文本切成互不相干的块。分块会把文本切成不重叠的片段：

\`\`\`text
[0 ~ 1023] [1024 ~ 2047] [2048 ~ 2999]
\`\`\`

这样 token 1024 甚至看不到紧挨着它的 token 1023。滑动窗口则是每一步向前移动一个 token：

\`\`\`text
generating token 3000: sees [1976 ~ 2999]
generating token 3001: sees [1977 ~ 3000]
generating token 3002: sees [1978 ~ 3001]
\`\`\`

相邻的窗口共享 1023 个 token，所以附近的上下文总是连续的：每个 token 都能看到它前面的 1023 个 token。

丢掉的是**往前超过 1023 个 token** 的所有内容。生成 token 3000 时，token 0 到 1975 直接不可见，而不是被模糊地记住。如果某个名字在 token 100 处被提到，GPT-2 在 token 3000 处就用不上它。GPT-2 的短期记忆大约只有 1024 个 token。

> 🆕 这个限制是后来的模型大幅推高上下文长度的一个主要原因；如今很多模型支持 128K 甚至更多 token。

### RoPE：任何位置都有对应的角度

没有表，只有一条规则："位置 $m$ 的第 $i$ 对转 $m\\,\\theta_i$"。位置 5000 可以直接代入，所以不会出错。再结合相对距离性质（第 3 节），两个相隔 3 的 token，无论在位置 10 还是位置 5000，分数中的位置部分都一样。

**但"能算"不等于"好用"。** 一个在长度 4096 上训练的模型，在长度 32,000 上运行时通常会明显变差。原因在于最慢的那几对：

- 训练时它们只转过很小的角度。第 31 对在 4096 个位置内大约只转 0.55 弧度，不到一整圈的十分之一。
- 再往后，它们会达到**训练中从未见过的角度**，模型不知道该如何解读。

> 🆕 因此长上下文模型会加上额外的技巧，比如更大的 base（Llama 3 用 500000），或者对位置做插值/重新缩放（Position Interpolation、YaRN）。每种方法的思路都是：让远处的角度保持在训练时见过的范围内。

简而言之：GPT-2 一次永远看不到超过 1024 个 token，滑动窗口只是通过丢弃较早的文本来掩盖这一点；RoPE 模型可以接受更长的输入，但需要做些调整才能在那里保持准确。

## 参考文献

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)**

   Su et al. 提出了 RoPE，并证明旋转后的 $q \\cdot k$ 只依赖相对位置。

   [RoFormer ↗](https://arxiv.org/abs/2104.09864)

2. **Extending Context Window of Large Language Models via Positional Interpolation (2023)**

   Chen et al. 把位置按比例缩小，让更长的输入落在 RoPE 角度已训练过的范围内。

   [Position Interpolation ↗](https://arxiv.org/abs/2306.15595)

3. **YaRN: Efficient Context Window Extension of Large Language Models (2023)**

   Peng et al. 对快的对和慢的对用不同方式重新缩放 RoPE 频率，以扩展上下文长度。

   [YaRN ↗](https://arxiv.org/abs/2309.00071)
`,
  },

  ln_gradient_attenuation_upper_bound_proof_notes: {
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
`
  },

  orthogonal_projection_notes: {
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

  orthogonal_projection_idempotent_notes: {
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

  orthogonal_projection_symmetric_notes: {
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

  orthogonal_projection_proof_notes: {
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
  optim_adamw_notes: {
    title: "torch.optim.AdamW",
    md: `
## 它是什么

AdamW 是当前 LLM 预训练中占主导地位的 optimizer。optimizer 是预训练中负责做决策的算法。每一步，optimizer 都在回答这个问题："给定当前的参数和 gradient，我该如何更新参数来降低 loss？"所以：

- **gradient** 告诉我们 loss 对每个参数的局部斜率。
- **optimizer** 给出把 gradient 变成一次更新的算法。

## optimizer 的历史：从 SGD 到 AdamW

从我自己的学习经验来看，不了解 optimizer 的发展历史，就很难真正理解 AdamW。所以下面是一条以语言模型为中心的 optimizer 时间线，附上关键思想，比如谁用了哪个 optimizer、为什么会被替换。点击每个 optimizer 下面的链接可以查看详细内容。

### 2010–2013：稀疏特征时代（词向量、经典 NLP）

- **[SGD](#info:sgd_notes)**：简单、快速；最早的 optimizer。
- **[SGD with momentum](#info:sgd_momentum_notes)**：加入速度（velocity），让更新更平滑。
- **[Adagrad](#info:adagrad_notes)**：在稀疏、高方差的 gradient 上很流行；对早期的 embedding 和 logistic/softmax 模型很有帮助。它在训练中为每个参数单独调整学习率，并且随时间不断调整。
- **[RMSProp](#info:rmsprop_notes)**：修正了 Adagrad 步长逐渐消失的问题；在早期 RNN 语言模型中被广泛使用。

### 2014–2017：RNN → 早期 seq2seq

- **[Adam](#info:adam_notes)**：成为 NLP 的默认选择（RNN 语言模型、带 attention 的 seq2seq）。原因：它把 momentum 和 RMSProp 式的方差归一化结合在一起，所以比 SGD 少得多地需要反复调学习率（LR）。

### 2017–2019：Transformer 起飞

- **Adam + warmup 成为标准**：先做一小段线性 warmup，再衰减（通常是 inverse-sqrt 或 cosine）。
- 从 post-norm 过渡到 pre-norm；Adam 的自适应性加上 warmup，让很深的层堆叠也能稳定训练。
- weight decay 开始变得重要。BERT 让 Adam 搭配 weight decay 流行起来；不久之后，社区统一采用了解耦的 weight decay（AdamW），因为它更可预测、泛化更好。

### 2019–2021：规模扩大（T5、GPT-2/3、mT5）

- **[AdamW](#info:adamw_notes)** 成为大语言模型的首选（bf16、warmup + cosine、梯度裁剪、对 norm/bias 不做 decay）。
- **Adafactor**（省内存、类似 Adam）在 optimizer 状态内存紧张时，被 T5/mT5 这类模型广泛采用（对二阶矩做分解）。

### 2022–2023：进一步扩大规模

- 预训练仍然用 AdamW（或者在内存是瓶颈时用 Adafactor）。
- 微调：Lion、Adan、AdaBelief、Sophia 等的实验在某些设置下表现不错，但没有一个能在基础模型预训练上普遍取代 AdamW。
- 系统层面普遍采用融合的 AdamW kernel（Apex、DeepSpeed、xFormers）来提升吞吐量。

### 2024 至今

- AdamW 仍然是 LLM 预训练和大多数全参数微调的主流默认选择。
- 当必须把 optimizer 内存压到最小时，Adafactor 仍是一个实用的替代方案。
- 研究型 optimizer 不断出现，但生产环境的语言模型仍然坚持上面这对稳定的组合，并搭配：pre-norm/RMSNorm、bf16、线性 warmup（约占总步数的 0.5–3%）、cosine 衰减、全局范数裁剪，以及对 norm/bias 不做 decay。
`,
  },

  sgd_notes: {
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
`,
  },

  sgd_momentum_notes: {
    title: "SGD with momentum（带动量的 SGD）",
    md: `
**不再只用当前的 gradient 迈步，而是维护一个速度（velocity）：过去 gradient 的指数移动平均（EMA）。**

## 它是什么

速度能平滑噪声，并在持续下坡的方向上加速前进。

我们使用常见的 EMA 形式：

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

- $g = \\nabla_{\\theta} L$ 是当前的 gradient。
- $\\beta$（例如 0.9）控制保留多少"记忆"。
- $\\eta$ 是学习率。

## 小数值例子

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

> 可以看到 loss 每一步都在下降（4.5 → 4.0613 → 3.3089）。这就是 optimizer 利用反向传播得到的 gradient，把 $w, b$ 推向更好的值。和 SGD 对比一下：在这个例子的前 3 步里，SGD 的 loss 下降得更快（4.5 → 1.125 → 0.28125）。原因见下文。

## 为什么 momentum 一开始会落后

速度需要时间积累。在 $\\beta = 0.9$ 时，速度从 0 开始，在最初几步里只跟上当前 gradient 的一小部分。这让最初的几步比普通 SGD 更小。

比如在我们的小数值例子里：

- **第 1 步**保留 90% 的过去速度（目前是 $0$，因为 $v_w = 0, v_b = 0$），再加上 10% 的当前 gradient，所以第 1 步是原始 gradient 的 0.1 倍。
- **第 2 步**保留 90% 的过去速度（第 1 步的 0.1 倍原始 gradient），再加上 10% 的当前 gradient，所以第 2 步大约是原始 gradient 的 0.19 倍（$0.1 \\cdot 0.9 + 0.1 = 0.19$）。
- **第 3 步**保留 90% 的过去速度（第 1–2 步累积的 0.19 倍原始 gradient），再加上 10% 的当前 gradient，所以第 3 步大约是原始 gradient 的 0.271 倍（$0.19 \\cdot 0.9 + 0.1 = 0.271$）。

## momentum 何时、为何在后期胜出

一旦 gradient 的方向大致保持一致，速度就会不断累积，于是步子会比 SGD 更大、更平滑。这在更难的问题上会带来回报，比如有噪声或病态（ill-conditioned）的 loss。但在这个只有一个样本的问题上并不会：SGD 本身已经收敛得非常快，而 momentum 会冲过头并来回振荡。10 步之后，momentum 的 loss 约为 0.44，而 SGD 约为 $4 \\times 10^{-6}$。

## 让 momentum 更早显出优势

如果想让 momentum 更早表现得更好，我们可以：

1. 降低 $\\beta$（例如 0.8），让速度积累得更快。
2. 稍微提高 $\\eta$（例如 0.12），弥补早期步子偏小的问题。
`,
  },

  adagrad_notes: {
    title: "Adagrad",
    md: `
**Adagrad（Adaptive gradient，自适应梯度）** 是一种 optimizer，它给每个参数分配自己的学习率，并根据该参数过去的 gradient 自动调整。

> 过去 gradient 大/频繁的参数，之后的步子会更小；gradient 小/稀少的参数，之后的步子会更大。

## 更新规则

Adagrad 为每个参数维护一个**平方 gradient 的累加和**：

$$
s_i \\leftarrow s_i + g_i \\odot g_i
$$

其中 $g_i$ 是第 $i$ 个参数的 gradient，$s_i$ 是第 $i$ 个参数的平方 gradient 累加和。

然后 Adagrad 用学习率除以"平方 gradient 累加和的平方根加上一个小常数 $\\epsilon$"，得到第 $i$ 个参数的学习率：

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

这样，过去 gradient 大/频繁的参数，之后的步子会更小；gradient 小/稀少的参数，之后的步子会更大。

## 为什么要开平方根？

因为 $s_i$ 是累加和，对于平稳的 gradient，它大致随时间线性增长（$s_i \\propto t$）。在分母中使用 $\\sqrt{s_i}$，有效学习率就会按如下方式衰减：

$$
\\eta_{\\text{effective},\\,i} = \\frac{\\eta}{\\sqrt{s_i}} \\approx \\frac{\\eta}{\\sqrt{t}}.
$$

$\\frac{1}{\\sqrt{t}}$ 的衰减很温和：它让步子随时间稳步缩小，但又不会快到让学习过早停滞。如果直接除以 $s_i$，就会得到 $\\frac{1}{t}$ 的衰减，步子很快就变得极小，可能让训练停滞不前。

> 这也和经典的在线凸优化（online convex optimization）结果相关：在那里，$\\frac{1}{\\sqrt{t}}$ 能给出良好的 regret/收敛界。

## 小数值例子

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradient：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Adagrad 累加器：** $s \\leftarrow s + g \\odot g$，从 $s_0 = (0, 0)$ 开始
- **更新规则：** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon}\\, g$
- **超参数：** $\\eta = 1.0$，$\\epsilon = 10^{-8}$（非常小，数值上可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$

下面我们定义 $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s_t} + \\epsilon}$ 为时刻 $t$ 的有效学习率。

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)$
- $\\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3,\\; 1/2)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)$
- $\\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-1, 1) = (1, -1)$

坐标 1 和坐标 2 现在有了不同的有效学习率：$1/3$ 对 $1/2$。

### 第 2 步

- $\\theta_1 = (1, -1)$
- $g_2 = (1 - 3,\\; -1 + 2) = (-2, 1)$
- $s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)$
- $\\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)$
- $\\theta_2 = \\theta_1 - \\triangledown = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)$

坐标 1 的步长从 $1/3$ 缩小到 $0.277$；坐标 2 的步长从 $1/2$ 缩小到 $0.447$。

### 第 3 步

- $\\theta_2 = (1.554, -1.447)$
- $g_3 = (1.554 - 3,\\; -1.447 + 2) = (-1.446, 0.553)$
- $s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)$
- $\\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)$
- $\\theta_3 = \\theta_2 - \\triangledown = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)$

坐标 1 的步长从 $0.277$ 缩小到 $0.257$；坐标 2 的步长从 $0.447$ 缩小到 $0.434$。

## 要点

这个例子说明，每个坐标 $i$ 都有自己的学习率 $\\eta_{\\text{eff},\\,i}$，它取决于第 $i$ 个坐标过去的 gradient。随着 $s_i$ 不断累积过去的平方 gradient，这些学习率会随时间缩小。

> gradient 更大/更频繁的坐标（这里是 $\\theta_1$）缩小得更快，所以步子更小；gradient 小/稀少的坐标（这里是 $\\theta_2$）缩小得更慢，所以步子更大。
`,
  },

  rmsprop_notes: {
    title: "RMSProp",
    md: `
**RMSProp** 为每个参数维护平方 gradient 的指数移动平均（EMA），并用这个平均值的 RMS（平方根）来缩放步长。

## 它做了什么

$$
\\begin{aligned}
s_t &= \\beta s_{t-1} + (1 - \\beta) (g_t \\odot g_t) \\\\
\\theta_{t+1} &= \\theta_t - \\frac{\\eta}{\\sqrt{s_t} + \\epsilon}\\, g_t
\\end{aligned}
$$

- $\\beta$ 是衰减率（例如 0.9）。
- $s_t$ 是平方 gradient 的 EMA。
- $g_t$ 是时刻 $t$ 的 gradient，$\\theta_t$ 是时刻 $t$ 的参数。
- $\\eta$ 是学习率。
- $\\epsilon$ 是一个小常数，用来避免除以零。

## RMSProp 修正了什么（对比 Adagrad）

Adagrad 会永远累加平方 gradient，所以每个参数的步长单调缩小（$\\sim 1/\\sqrt{t}$），在长时间训练中可能变得过小。

RMSProp 用 EMA 取代了不断增长的累加和。这样可以防止学习率下降得太快，让算法在训练深度神经网络时更有效。

### 和 momentum 的类比

我们可以像看待 SGD+momentum 与 SGD 的关系那样，看待 RMSProp 与 Adagrad 的关系：

- **Adagrad → RMSProp：** 对平方 gradient 这个统计量使用 EMA（记忆），而不是不断增长的累加和。
- **SGD → SGD+momentum：** 对 gradient 本身加上 EMA（记忆）。

但两者仍有一些关键区别：

- **对什么做 EMA：** RMSProp = 平方 gradient。SGD+momentum = gradient。
- **逐参数学习率：** RMSProp（和 Adagrad 一样）保留逐参数的自适应步长；SGD+momentum（和 SGD 一样）使用一个全局学习率（没有逐参数）。
- **修正的失效模式：** Adagrad 的学习率在长时间训练中会衰减到接近零；RMSProp 避免了这一点。SGD 噪声大、走"之"字形；SGD+momentum 让它更平滑并加速。

## 小数值例子

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradient：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **RMSProp 累加器：** $s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g)$，从 $s_0 = (0, 0)$ 开始
- **更新规则：** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon}\\, g$
- **超参数：** $\\eta = 0.1$，$\\beta = 0.9$，$\\epsilon = 10^{-8}$（非常小，数值上可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.9, 0.4)$
- $\\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)$
- $\\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)$

### 第 2 步

- $\\theta_1 = (0.3162, -0.3162)$
- $g_2 = (0.3162 - 3,\\; -0.3162 + 2) = (-2.6838, 1.6838)$
- $s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot \\big((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)\\big) = (1.5303, 0.6435)$
- $\\sqrt{s_2} = (1.2370, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1247)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1247) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)$
- $\\theta_2 = \\theta_1 - \\triangledown = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)$

### 第 3 步

- $\\theta_2 = (0.5332, -0.5261)$
- $g_3 = (0.5332 - 3,\\; -0.5261 + 2) = (-2.4668, 1.4739)$
- $s_3 = 0.9 \\cdot (1.5303, 0.6435) + (1 - 0.9) \\cdot \\big((-2.4668, 1.4739) \\odot (-2.4668, 1.4739)\\big) = (1.9858, 0.7964)$
- $\\sqrt{s_3} = (1.4092, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121)$
- 更新 $\\triangledown = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4739) = (-0.1751, 0.1652)$
- $\\theta_3 = \\theta_2 - \\triangledown = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)$
`,
  },

  adam_notes: {
    title: "Adam",
    md: `
**Adam** 结合了 RMSProp 和 SGD+momentum 的优点。

## 它是什么

- **RMSProp：** 追踪平方 gradient（它刻画了 gradient 的尺度），用来归一化步长。
- **SGD+momentum：** 追踪 gradient 本身（它刻画了 gradient 的方向），用来平滑更新。

Adam 把这两个想法合并到一条更新规则里：

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) (g_t \\odot g_t) && \\text{(EMA of squared gradients: RMS scaling)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_{t+1} &= \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(adaptive step per parameter)}
\\end{aligned}
$$

- $\\beta_1$ 和 $\\beta_2$ 分别是 momentum 系数和 RMS 缩放系数。
- $\\epsilon$ 是一个小常数，用来避免除以零。

> 偏差修正（bias correction）解决的是 EMA（指数移动平均）有偏的问题。我们会在笔记最后讨论它。

## 小数值例子

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradient：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **更新规则：** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **超参数：** $\\eta = 0.1$，$\\beta_1 = 0.9$，$\\beta_2 = 0.999$，$\\epsilon = 10^{-8}$（非常小，数值上可以忽略）
- **初始值：** $\\theta_0 = (0, 0)$，$m_0 = (0, 0)$，$v_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- **偏差修正：**

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

### 第 2 步

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- **偏差修正：**

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
- **偏差修正：**

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

这个例子展示了 Adam 如何结合 RMSProp 和 SGD+momentum 的优点，得到一个更有效的 optimizer：

- **Momentum：** $\\hat{m}_t$ 是原始 gradient 的平滑版本，减少了噪声和"之"字形震荡。
- **逐参数自适应：** $\\hat{v}_t$ 在不同坐标上取值不同，所以每个参数都有自己的有效步长 $\\frac{\\eta}{\\sqrt{\\hat{v}_{t,i}} + \\epsilon}$。

## 偏差修正（Bias correction）

Adam 维护以下两者的 **EMA（指数移动平均）**：

- gradient，$m_t$（momentum），以及
- 平方 gradient，$v_t$（RMS 缩放）。

两个 EMA 都从零开始（$m_0 = 0$，$v_0 = 0$）。这会让估计值偏向零，而且在最初几步偏差很大，尤其是在常用的 $\\beta_1 = 0.9$、$\\beta_2 = 0.999$ 下。

例如，对于一个恒定的 gradient $g$，第一步是：

- $m_1 = \\beta_1 m_0 + (1 - \\beta_1)\\, g$（$m_1$ 从 $0.1g$ 开始），
- $v_1 = \\beta_2 v_0 + (1 - \\beta_2)\\, g^2$（$v_1$ 从 $0.001g^2$ 开始）。

所以如果没有偏差修正，第 1 步的分子只有预期值的 $0.1\\times$，分母只有预期值的 $\\sqrt{0.001} \\approx 0.0316\\times$。原始（未修正）的步长大小约为

$$
\\eta \\cdot \\frac{m_1}{\\sqrt{v_1}} = \\eta \\cdot \\frac{0.1g}{0.0316g} = \\eta \\cdot 3.16 = \\eta \\cdot F_1
$$

因此，在常用设置 $\\beta_1 = 0.9$、$\\beta_2 = 0.999$、$m_0 = 0$、$v_0 = 0$ 下：

- $F_1 = 3.16$：第一步大了 3.16 倍。
- 用类似的推理，可以得到 $F_2 = 4.25$、$F_3 = 4.95$、$F_4 = 5.44$。
- 到 $t = 12$ 时达到最高点 $F_{12} = 6.568$，之后逐渐下降到 1。

另一方面，有了偏差修正，$\\hat{m}_1 = g_1$，$\\hat{v}_1 = g_1^2$，所以 $\\frac{\\hat{m}_1}{\\sqrt{\\hat{v}_1}}$ 这一项对 $\\beta$ 的选择以及初始值 $m_0$、$v_0$ 都不那么敏感了。

### 修正方法

Adam 通过除以 $1 - \\beta^t$ 来修正：

$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
$$

> 完整证明请参阅原论文第 3 节：[Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980)。

## 参考资料

1. **Adam: A Method for Stochastic Optimization**

   Adam 的原始论文。

   [Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980)

2. **StackExchange 讨论**

   关于 Adam 中偏差修正的讨论。

   [StackExchange 讨论 ↗](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
`,
  },

  adamw_notes: {
    title: "AdamW",
    md: `
**AdamW** = Adam（momentum + RMS 缩放 + 偏差修正）再加上单独施加的 **weight decay**。

## 更新规则

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) (g_t \\odot g_t) && \\text{(EMA of squared gradients: RMS scaling)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_t &= (1 - \\eta \\lambda)\\, \\theta_t && \\text{(decoupled weight decay)} \\\\
\\theta_{t+1} &= \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(adaptive step per parameter)}
\\end{aligned}
$$

- $\\beta_1$ 和 $\\beta_2$ 分别是 momentum 系数和 RMS 缩放系数。
- $\\epsilon$ 是一个小常数，用来避免除以零。
- $\\lambda$ 是 weight decay 的强度（例如 0.1）。

## 小数值例子

### 设置

- **参数：** $\\theta = (\\theta_1, \\theta_2)$
- **Loss：** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradient：** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **更新规则：** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **超参数：** $\\eta = 0.1$，$\\beta_1 = 0.9$，$\\beta_2 = 0.999$，$\\epsilon = 10^{-8}$（非常小，数值上可以忽略），$\\lambda = 0.1$
- **初始值：** $\\theta_0 = (0, 0)$，$m_0 = (0, 0)$，$v_0 = (0, 0)$

### 第 1 步

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- **偏差修正：**

  $$
  \\begin{aligned}
  \\hat{m}_1 &= \\frac{m_1}{1 - \\beta_1^1} = \\frac{(-0.3, 0.2)}{1 - 0.9^1} = (-3, 2) \\\\
  \\hat{v}_1 &= \\frac{v_1}{1 - \\beta_2^1} = \\frac{(0.009, 0.004)}{1 - 0.999^1} = (9, 4)
  \\end{aligned}
  $$

- **衰减：** $\\theta_0 = (1 - \\eta \\lambda)\\, \\theta_0 = (1 - 0.1 \\cdot 0.1) \\cdot (0, 0) = (0, 0)$
- **更新：**

  $$
  \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon}\\, \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9, 4)}} \\cdot (-3, 2) = (0.1, -0.1)
  $$

### 第 2 步

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- **偏差修正：**

  $$
  \\begin{aligned}
  \\hat{m}_2 &= \\frac{m_2}{1 - \\beta_1^2} = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.94737, 1.94737) \\\\
  \\hat{v}_2 &= \\frac{v_2}{1 - \\beta_2^2} = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.70485, 3.80490)
  \\end{aligned}
  $$

- **衰减：** $\\theta_1 = (1 - \\eta \\lambda)\\, \\theta_1 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1, -0.1) = (0.099, -0.099)$
- **更新：**

  $$
  \\begin{aligned}
  \\theta_2 &= \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon}\\, \\hat{m}_2 = (0.099, -0.099) - \\frac{0.1}{\\sqrt{(8.70485, 3.80490)}} \\cdot (-2.94737, 1.94737) \\\\
  &= (0.1988973, -0.1988335)
  \\end{aligned}
  $$

### 第 3 步

- $\\theta_2 = (0.1988973, -0.1988335)$
- $g_3 = (0.1988973 - 3,\\; -0.1988335 + 2) = (-2.80110, 1.80117)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80110, 1.80117) = (-0.78411, 0.513117)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot \\big((-2.80110, 1.80117) \\odot (-2.80110, 1.80117)\\big) = (0.0252298, 0.0108426)$
- **偏差修正：**

  $$
  \\begin{aligned}
  \\hat{m}_3 &= \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78411, 0.513117)}{1 - 0.9^3} = (-2.893395, 1.89342) \\\\
  \\hat{v}_3 &= \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252298, 0.0108426)}{1 - 0.999^3} = (8.418349, 3.617817)
  \\end{aligned}
  $$

- **衰减：** $\\theta_2 = (1 - \\eta \\lambda)\\, \\theta_2 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1988973, -0.1988335) = (0.196908, -0.196845)$
- **更新：**

  $$
  \\begin{aligned}
  \\theta_3 &= \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon}\\, \\hat{m}_3 = (0.196908, -0.196845) - \\frac{0.1}{\\sqrt{(8.418349, 3.617817)}} \\cdot (-2.893395, 1.89342) \\\\
  &= (0.2966312, -0.2963911)
  \\end{aligned}
  $$
`,
  },
  zero_grad_notes: {
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
  backward_notes: {
    title: "loss.backward()",
    md: `
**\`loss.backward()\` 做了什么：一次反向传播，算出所有参数的梯度。**

## \`backward()\` 只算梯度，\`step()\` 才更新参数

\`backward()\` **不会**改任何 weight。它只算梯度，把结果存进每个参数的 \`.grad\`。下一行 \`optimizer.step()\` 读出 \`.grad\`，才去更新参数。

\`\`\`python
loss.backward()     # 只算梯度，结果存进 p.grad；weight 不变
optimizer.step()    # 读出 p.grad，用它更新 weight
\`\`\`

用一个参数验证：$w = 2$，$\\text{loss} = (3w)^2$，梯度是 $18w = 36$。优化器用最简单的 SGD，\`lr = 0.01\`：

\`\`\`text
backward 之前:  w = 2.0    w.grad = None
backward 之后:  w = 2.0    w.grad = 36      ← 梯度算出来了，w 没动
step 之后:      w = 1.64   w.grad = 36      ← w 才变了：2 − 0.01 × 36 = 1.64
\`\`\`

所以 \`.grad\` 必须存下来：它就是 \`backward()\` 交给 \`step()\` 的结果。

### 为什么要分成两步

如果 \`backward()\` 算完梯度就顺手更新参数，下面这些事情都做不了：

1. **先处理梯度，再更新参数。** 代码里 \`backward()\` 和 \`step()\` 之间有一行 \`clip_grad_norm_\`：它先看所有梯度合起来有多大，太大就整体缩小。这一步要等全部梯度都算完才能做。
2. **Gradient accumulation。** 几个小 batch 各 \`backward()\` 一次，梯度先在 \`.grad\` 里累加，最后只 \`step()\` 一次。见 [zero_grad](#hover:zero_grad_notes)。
3. **更新规则不止一种。** SGD 是 \`w = w − lr × grad\`；GPT-2 用的 AdamW 还会用到过去很多步梯度的平均值，再按每个参数的情况调整步长。\`backward()\` 只负责提供梯度，怎么用由你选的 optimizer 决定。

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
};
