// Notes: Input text and the BPE tokenizer.
// Each note has an English and a Chinese version side by side.
// Links between notes use "#info:<key>" (opens in the drawer) or "#hover:<key>" (popup).

export default {
  input_data_notes: {
    en: {
      title: "Input Data (Text)",
      md: `
## What it is

Raw text strings (documents, prompts) that will be tokenized before entering the model.

## Common preprocessing

- Normalize whitespace/newlines if needed.
- Decide on the max sequence length ($T$) and the segmentation strategy (e.g., sliding windows, paragraph chunks).
- Split into train/val/test to avoid leakage.

## Batching shapes

- After tokenization, inputs are **token IDs**: $(B, T)$.
- Optionally an **attention mask**: $(B, T)$ to mark padding (not typically used in GPT-2 causal-only training, but useful in general).
`,
    },
    zh: {
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
  },

  tokenizer_notes: {
    en: {
      title: "Tokenizer (BPE for GPT-2)",
      md: `
This line of code returns GPT-2's **byte-level BPE tokenizer** (vocab size **50257**, includes the special token \`<|endoftext|>\` with id **50256**).

It contains the byte↔[unicode map](#info:gpt2_byte_unicode_map_notes), [GPT-2 merge ranks](#info:gpt2_merge_ranks_notes), [regex pre-tokenizer](#info:regex_pre_tokenizer_notes), and special tokens.

## How tokenization works

### 1 · Pre-tokenize

Split the raw text into pieces with GPT-2's regex (separates runs of letters, digits, punctuation, leading spaces, etc.).

**Example** (each box is one piece; ⎵ marks a space character)

- \`Hello, world!\` → \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!⟧\`
- \`2025-09-19\` → \`⟦2025⟧⟦-⟧⟦09⟧⟦-⟧⟦19⟧\`

[Deep dive into regex pre-tokenizer →](#info:regex_pre_tokenizer_notes)

### 2 · Byte encoding

Encode each piece as UTF-8 bytes (case and whitespace are preserved). Map each byte to a safe Unicode symbol via the byte↔unicode table so every byte is representable.

**Example 1** (the pieces from step 1)

- \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!⟧\` → \`⟦Hello⟧⟦,⟧⟦Ġworld⟧⟦!⟧\`
- Inside \`⎵world\`: bytes \`[0x20, 0x77, 0x6F, 0x72, 0x6C, 0x64]\` → \`Ġ w o r l d\`
- Only the space byte \`0x20\` changes (to \`Ġ\`); ASCII letters and punctuation map to themselves.

**Example 2** (a non-ASCII character): the piece \`⎵café\` (from the text \`I love café\`)

- Bytes: \`[0x20, 0x63, 0x61, 0x66, 0xC3, 0xA9]\` (space and accents preserved).
- Unicode: \`U+0120 Ġ\` · \`U+0063 c\` · \`U+0061 a\` · \`U+0066 f\` · \`U+00C3 Ã\` · \`U+00A9 ©\`
- The string \`ĠcafÃ©\`.

> **Order matters:** the regex runs on the raw text first, because the stand-in \`Ġ\` counts as a letter and \`HelloĠworld\` would no longer split at the space.

[Deep dive into byte-unicode map →](#info:gpt2_byte_unicode_map_notes)

### 3 · BPE merges

For each piece, iteratively merge symbol pairs using GPT-2's [merge ranks](#info:gpt2_merge_ranks_notes) until no more merges apply, yielding subword units.

### 4 · Look up IDs

Each final subword is looked up in the vocabulary (\`encoder.json\`) to get its token id.

- \`⟦Hello⟧⟦,⟧⟦Ġworld⟧⟦!⟧\` → \`[15496, 11, 995, 0]\`
- Each of these pieces is already a whole token, so no splitting happens; \`!\` is id 0 because it is the first printable byte.

> **Notes:** Because it's byte-level, every input string is tokenizable without "unknown" tokens, and the original spacing/casing is maintained.

## Recent developments: how tokenizers have changed since GPT-2

BPE is still the mainstream approach today. What has changed most is the vocabulary size:

- GPT-2: ~50K tokens
- GPT-4 (\`cl100k_base\`): ~100K
- GPT-4o (\`o200k_base\`): ~200K
- Llama 3: ~128K

With a larger vocabulary, the same text splits into fewer tokens, so generation is faster, and non-English languages such as Chinese are handled better.
The cost is a larger embedding table and output layer: both have shape $(V, n\\_embd)$, so they gain one row per token as $V$ grows, while the vector size $n\\_embd$ stays the same.

### How numbers are split

GPT-2's regex keeps a whole run of digits together (\` ?\\p{N}+\`), then BPE merges inside it. The tokenizer is deterministic (the same number always splits the same way), but the split has nothing to do with place value:

- \`1234\` → \`⟦12⟧⟦34⟧\`
- \`12345\` → \`⟦123⟧⟦45⟧\`
- \`1234567\` → \`⟦123⟧⟦45⟧⟦67⟧\`
- \`3517+4825=8342\` → \`⟦35⟧⟦17⟧⟦+⟧⟦48⟧⟦25⟧⟦=⟧⟦8⟧⟦342⟧\`

The same token can stand for different place values. In \`4825\` → \`⟦48⟧⟦25⟧\`, the \`2\` in \`25\` is the tens digit; in \`2517\` → \`⟦25⟧⟦17⟧\` it is the thousands digit. To add two numbers, the model must first work out which digit sits in which place from the token lengths and count, and when it writes the answer it must also emit the exact split BPE would produce (\`⟦8⟧⟦342⟧\`, not \`⟦83⟧⟦42⟧\`). This does not make arithmetic impossible, but it makes it harder to learn.

Newer tokenizers changed the digit rule. There are two main approaches:

- **At most 3 digits per chunk** (\`\\p{N}{1,3}\`): GPT-4 (\`cl100k_base\`), GPT-4o (\`o200k_base\`), Llama 3, DeepSeek-V3. \`1234567\` → \`⟦123⟧⟦456⟧⟦7⟧\`. Every chunk has 1–3 digits, and a leading space no longer sticks to the digits. Grouping still runs left to right, though, so chunk boundaries do not line up with thousands the way \`1,234,567\` does.
- **One digit per token** (\`\\p{N}\`): Qwen2.5, Qwen3; LLaMA 1/2 used SentencePiece with digit splitting. \`1234567\` → \`⟦1⟧⟦2⟧⟦3⟧⟦4⟧⟦5⟧⟦6⟧⟦7⟧\`. Place values line up cleanly and carrying works the same in every number, at the cost of longer sequences.

> There is no single winner: Meta moved from per-digit splitting (LLaMA 1/2) to 3-digit groups (Llama 3) when it switched to a tiktoken-based 128K vocabulary. Closed models (Claude, Gemini, etc.) do not publish their tokenizers. Today's models also lean on step-by-step reasoning or calculator/code tools for large arithmetic, not on tokenization alone.

## References

1. **Primary source (architecture + input representation)**

   GPT-2 technical report. It spells out byte-level BPE, the motivation for byte-level processing, and the expanded vocab (50,257). See §2.2 "Input Representation" and Table 2.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

2. **Original GPT-2 BPE tokenizer source code**

   OpenAI's encoder.py, about 120 lines. It holds all the steps above: \`bytes_to_unicode()\` (the byte↔unicode map), the regex pattern \`self.pat\` (pre-tokenize), \`bpe()\` (merges by rank), and \`encode()\` / \`decode()\`. The vocab (\`encoder.json\`) and merge list (\`vocab.bpe\`) it loads are fetched by \`download_model.py\`.

   [openai/gpt-2 encoder.py ↗](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

3. **Tokenizer implementation**

   OpenAI's tiktoken. Clear, practical reference for GPT-2's byte-level BPE (lossless, reversible), and how bytes ↔ tokens map is implemented.

   [tiktoken ↗](https://github.com/openai/tiktoken)

4. **Foundational BPE paper (why subword/byte-level)**

   Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units." Not GPT-2-specific, but it's the classic reference behind subword/BPE vocabularies GPT-2 builds embeddings over.

   [Neural Machine Translation of Rare Words with Subword Units ↗](https://aclanthology.org/P16-1162/)

5. **Let's build the GPT Tokenizer**

   Karpathy's video on GPT Tokenizer. Practical guides on GPT-2's byte-level BPE (from 1430s).

   [BPE - Let's build the GPT Tokenizer ↗](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s)

6. **Hands-on/educational BPE references**

   Karpathy's minBPE (minimal byte-level BPE)

   [minbpe ↗](https://github.com/karpathy/minbpe)

7. **Empirical analysis of GPT-2 embeddings**

   Interpretability deep-dive into GPT-2's embedding weights (distributional properties, relation to positions). Advanced exploration inside wte.

   [An exploration of GPT-2's embedding weights ↗](https://www.alignmentforum.org/posts/BMghmAxYxeSdAteDc/an-exploration-of-gpt-2-s-embedding-weights)

    `,
    },
    zh: {
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
  },

  gpt2_byte_unicode_map_notes: {
    en: {
      title: "GPT-2 byte-unicode map",
      md: `
## What it is

A fixed, 1-to-1 mapping between the 256 possible bytes (0–255) and 256 printable Unicode code points.

- **byte → unicode ("byte encoder"):** turns each raw byte into a single visible Unicode character.
- **unicode → byte ("byte decoder"):** the exact inverse, so it's perfectly reversible.

## Why it exists

Classical BPE operates on "characters." GPT-2, however, is byte-level: it wants to handle any byte sequence (emoji, accents, binary-ish bytes, etc.) losslessly. But the BPE code works on strings, and its merge list (\`vocab.bpe\`) writes each pair as two symbols separated by a space, e.g. \`Ġ t\`. A raw space, newline or control byte inside a symbol would break that format. The map solves this by:

1. Giving every byte a safe, visible stand-in,
2. Avoiding collisions with whitespace/control characters,
3. Letting BPE run on a printable string while staying exactly reversible.

> The regex pre-tokenizer runs earlier, on the raw text, so it never sees these stand-ins.

## How it's constructed (conceptually)

- Bytes that are already "nice printable" (e.g., ASCII and many Latin-1 characters) are mapped to themselves (same code point).
- The remaining bytes (e.g., space, tabs, newlines, etc.) are mapped to **distinct printable code points** starting at U+0100 (256 + n, in byte order), ensuring they're visible and not whitespace.

**Example**

- Byte \`0x41\` (\`A\`) → is mapped to standard \`A\` (same character)
- Byte \`0x20\` (space) → stand-in \`Ġ\` (U+0120), which is why the piece \`⟦⎵world⟧\` is shown as \`⟦Ġworld⟧\`.
- Bytes \`0xC3 0xA9\` (é) → \`Ã©\` (both bytes are printable Latin-1, so each maps to itself: two chars, one per byte)
- Bytes \`0xE2 0x80 0x94\` (em dash —) → \`âĢĶ\` (\`0xE2\` maps to itself as \`â\`; \`0x80\` → \`Ģ\`, \`0x94\` → \`Ķ\`)

This yields two tables:

- \`byte_encoder: {0..255 → one-char unicode strings}\`
- \`byte_decoder: {that unicode char → original byte}\`

**Which bytes map to themselves?** In GPT-2's map: **33–126** (printable ASCII excluding space) and **161–172, 174–255** (printable Latin-1).

Bytes outside those ranges (e.g., control chars 0–31, space 32, 127–160, and 173) get special stand-ins (\`U+0100\`, \`U+0101\`, …, \`U+0143\`): 68 bytes in total. Space is the 33rd of them (n = 32), so it becomes U+0120 \`Ġ\`.

**Emoji example** (UTF-8: \`F0 9F 98 8A\`)

- \`0xF0\` (= 240) is in the range 174–255 ⇒ maps to itself (prints as \`ð\` in Latin-1).
- \`0x9F\` (159), \`0x98\` (152), \`0x8A\` (138) are not in the printable sets ⇒ stand-ins \`Ł\`, \`ĺ\`, \`Ĭ\` (Latin Extended-A characters).
- So 😊 becomes the string \`ðŁĺĬ\`: one self-mapped character plus three stand-ins, one visible char per byte, fully reversible.

## How it's used (encode path)

1. Split the raw text into chunks with the regex pre-tokenizer.
2. Encode each chunk as UTF-8 bytes.
3. Map each byte \`b\` to its stand-in Unicode char \`u = byte_encoder[b]\`.
   - Example: the chunk \`⟦⎵Hello⟧\` (leading space) → bytes \`[0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F]\` → mapped string \`ĠHello\` (space became \`Ġ\`; ASCII letters map to themselves).
   - Non-ASCII (e.g., é → bytes \`C3 A9\`) becomes two mapped chars, one for each byte.
4. Run BPE merges on each mapped chunk, then look up each resulting token's id.

## Decoding (exact inverse)

1. Look up each token id's string in the vocabulary → concatenate them into one string of mapped chars.
2. Map each char back to a byte via \`byte_decoder\`.
3. UTF-8 decode the bytes → original text.

> This guarantees lossless round-trip: \`decode(encode(s)) == s\`

    `,
    },
    zh: {
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
  },

  regex_pre_tokenizer_notes: {
    en: {
      title: "Regex pre-tokenizer",
      md: `
## What it is

A regular expression based splitter that chops the input text into chunks before BPE merging happens. BPE merges are then applied within each chunk only, never across chunk boundaries.

## Why it exists

- **Main reason (GPT-2 paper, §2.2)** — without it, BPE would merge across character types and waste vocabulary on variants like \`dog.\`, \`dog!\`, \`dog?\`. Splitting first keeps letters, digits and punctuation in separate chunks.
- **Speed & determinism** — BPE runs on short chunks, so there are fewer candidate merges.
- **Semantics** — keeps leading spaces with the following token (so you get tokens like \`Ġworld\`), groups letters together, groups numbers together, and groups punctuation/symbols together.
- **Reversibility** — pairs neatly with byte-level encoding for lossless round-trip.

## Rough pattern GPT-2 uses (conceptually)

In these patterns, ⎵ stands for a literal space character.

- **contractions:** \`'s, 't, 're, 've, 'm, 'll, 'd\` (lowercase only: \`CAN'T\` → \`⟦CAN⟧⟦'⟧⟦T⟧\`)
- **optional leading space + letters:** \`⎵?\\p{L}+\`
- **optional leading space + digits:** \`⎵?\\p{N}+\`
- **optional leading space + other symbols:** \`⎵?[^\\s\\p{L}\\p{N}]+\`
- **whitespace:** \`\\s+(?!\\S)\` takes a run of spaces but leaves the last one for the next word; \`\\s+\` catches the rest.
  - \`Hello⎵⎵⎵world\` → \`⟦Hello⟧⟦⎵⎵⟧⟦⎵world⟧\`

## How it affects BPE

Apply the regex pre-tokenizer to the raw text → convert each chunk to UTF-8 bytes and map them to stand-in characters (space → \`Ġ\`) → in each chunk, repeatedly merge the lowest-rank adjacent pair.

> The regex must run before the byte mapping: \`Ġ\` counts as a letter, so \`HelloĠworld\` would not split.

Merges do not cross chunk boundaries. That's why \`⟦⎵world⟧\` (with its leading space) is a single chunk where merges can happen internally, but it won't merge with the preceding token.

**Special tokens:** the original \`encoder.py\` has no special handling: the text \`<|endoftext|>\` is split like any other text into 7 ordinary tokens. \`tiktoken\` maps it to the single id 50256 only if you pass it in \`allowed_special\` (by default it raises an error).

## Examples

Each box is one chunk; ⎵ marks a space character.

### Letters grouped

- \`Hello world\` → \`⟦Hello⟧⟦⎵world⟧\` — letters stay together; the leading space sticks to the next chunk.
- \`AB12\` → \`⟦AB⟧⟦12⟧\` — letters as one chunk, digits as another.

### Numbers grouped

- \`Room 1234\` → \`⟦Room⟧⟦⎵1234⟧\`
- \`1,234.50\` → \`⟦1⟧⟦,⟧⟦234⟧⟦.⟧⟦50⟧\` — digits group, but the comma and the period are separate symbol chunks.

### Punctuation / symbols grouped

- \`Hello, world!!\` → \`⟦Hello⟧⟦,⟧⟦⎵world⟧⟦!!⟧\` — the comma, and the run of two exclamation marks, are symbol chunks.
- \`Good-morning :)⎵\` → \`⟦Good⟧⟦-⟧⟦morning⟧⟦⎵:)⟧⟦⎵⟧\` — the smiley and its leading space form one symbol chunk; the trailing space becomes its own whitespace chunk.

### Contractions handled specially

- \`can't\` → \`⟦can⟧⟦'t⟧\` — the pre-tokenizer has dedicated rules for \`'s, 't, 're, 've, 'm, 'll, 'd\`.

### Non-ASCII symbols

- \`Hi — ok ★\` → \`⟦Hi⟧⟦⎵—⟧⟦⎵ok⟧⟦⎵★⟧\` — the em dash and the star are non-ASCII symbols; each becomes its own chunk with its leading space.

## Key idea

> It first splits the raw text into chunks using a regex: letters together, digits together, and symbols together, keeping any leading space with the next chunk. Then BPE merges run inside each chunk only—they don't cross chunk boundaries.

## References

1. **GPT-2 reference encoder (contains the exact regex)**

   OpenAI's original GPT-2 encoder.py, which defines the pre-tokenization pattern and the byte-unicode map.

   [openai/gpt-2 encoder.py ↗](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

2. **GPT-2 technical report (input representation)**

   Radford et al., §2.2 "Input Representation" — explains why BPE is prevented from merging across character categories.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

3. **tiktoken**

   OpenAI's production tokenizer; its "gpt2" encoding uses the same pattern.

   [tiktoken ↗](https://github.com/openai/tiktoken)

`,
    },
    zh: {
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
  },

  lossless_round_trip_notes: {
    en: {
      title: "Lossless round-trip",
      md: `
**"Lossless round-trip" means that if you encode text into token IDs and then decode those IDs back to text, you get exactly the same bytes you started with: nothing is altered or lost.**

Formally:

$$
\\mathtt{decode}(\\mathtt{encode}(s)) = s
$$

## Why it holds for GPT-2 (byte-level BPE)

1. It operates on raw bytes (no lowercasing, normalization, or space stripping).
2. Every byte sequence is representable.
3. Decode exactly inverts encode.
`,
    },
    zh: {
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
  },

  gpt2_merge_ranks_notes: {
    en: {
      title: "GPT-2 merge ranks",
      md: `
## Byte-Pair Encoding (BPE) refresher

Start from a sequence of basic symbols (for GPT-2, bytes). A BPE model has a list of symbol-pair merges learned from a large corpus, one at a time: the most frequent pair is merged first (rank 0), then the most frequent pair in the updated corpus (rank 1), and so on. GPT-2 learned 50,000 merges; its rank 0 is \`Ġ t\`.

## Why ranks matter

During encoding, the tokenizer repeatedly looks at adjacent pairs and merges the pair with the lowest rank number (highest priority) that appears in the list. It updates the sequence and repeats until no mergeable pairs remain. The final merged chunks are your tokens.

## What \`tiktoken\` stores

GPT-2's tokenizer ships with a mapping that effectively represents these ranks. In practice:

- Lower rank ⇒ earlier (more frequent) merge rule ⇒ applied before higher-rank rules.
- This makes tokenization deterministic and matches GPT-2's training.

## Relation to token IDs

Every merge creates one new vocabulary entry. GPT-2's 50,257 ids are laid out as:

- **ids 0–255:** the 256 single bytes (printable bytes first, so \`!\` = 0);
- **ids 256–50255:** the 50,000 merges, in rank order, so the result of merge rank $r$ has id $256 + r$;
- **id 50256:** the special token \`<|endoftext|>\`.

\`tiktoken\` stores these as a single table of token ranks, where a token's rank equals its id, and merges prefer the pair whose merged result has the lowest rank. That is the same order as above.

## Tiny toy example (made up)

Text: \`banana\` → bytes → initial symbols: \`⟦b⟧⟦a⟧⟦n⟧⟦a⟧⟦n⟧⟦a⟧\`

Suppose merges (with ranks):

- rank 0: \`(a, n) → an\`
- rank 1: \`(an, a) → ana\`
- rank 2: \`(b, an) → ban\`

Encoding proceeds:

1. Rank 0 \`(a, n)\` is the lowest-rank pair present; merge every occurrence → \`⟦b⟧⟦an⟧⟦an⟧⟦a⟧\`
2. Pairs now: \`(b, an)\` rank 2, \`(an, an)\` no rule, \`(an, a)\` rank 1 ⇒ \`⟦b⟧⟦an⟧⟦ana⟧\`
3. Pairs now: \`(b, an)\` rank 2, \`(an, ana)\` no rule ⇒ \`⟦ban⟧⟦ana⟧\`
4. No applicable merges remain → final tokens: \`⟦ban⟧⟦ana⟧\`

> The real GPT-2 tokenizer happens to give the same split: \`banana\` → \`[3820, 2271]\`.

## Space handling

GPT-2 is byte-level. Space is part of the bytes and many tokens encode "space+word" as a single token (often shown with a leading \`Ġ\` in visualizations). Those arise from specific merges and their ranks.

    `,
    },
    zh: {
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
  },
};
