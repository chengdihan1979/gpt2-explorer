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
};
