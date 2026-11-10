import { ORIGINAL_INFO } from "../GPT2DecoderExplorer";

const prevents_vanishing_exploding_md = `
**Two concrete toy examples that show why pre-LN keeps gradients from vanishing/exploding in deep stacks.**

## Example 1: "safe at init" path (identity highway)

Setup (early training): attention weights are small, so the sublayer output is near zero:

$$
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0
$$

- **Pre-LN block:** $x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l$.

  Backward: $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\;\\Rightarrow\\;$ the gradient passes nearly unchanged.

- **Post-LN (contrast):** $x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l)$,

  so $\\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)$, and repeated products can shrink or grow gradients.

## Example 2: Jacobian picture with residual vs. no residual

Consider one block as a function $F(x) = x + f(\\mathrm{LN}(x))$.

### Jacobian (pre-LN)

$$
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x)
$$

If $\\|J_f\\|$ is small and $J_{\\mathrm{LN}}$ is bounded, then $J_F \\approx I + \\epsilon$ with small $\\epsilon$. Across $L$ layers, $\\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)$ stays near $I$ $\\Rightarrow$ no vanishing/exploding.

### No residual

$$
G(x) = f(\\mathrm{LN}(x)), \\qquad J_G = J_f\\,J_{\\mathrm{LN}}
$$

Across depth, $\\prod_{\\ell=1}^{L} J_G(x_\\ell)$ can vanish if $\\|J_G\\|<1$ or explode if $\\|J_G\\|>1$.

Here $\\|J_G\\|$ is the [spectral norm ↗](https://mathworld.wolfram.com/SpectralNorm.html).

## Why pre-LN helps more than post-LN

- **Pre-LN:** $x_{l+1}=x_l + f(\\mathrm{LN}(x_l))$ keeps an identity path in both forward and backward.
- **Post-LN:** $x_{l+1}=\\mathrm{LN}(x_l+f(x_l))$. Even the skip connection is passed through $J_{\\mathrm{LN}}$ every layer, so the identity gets scaled repeatedly, which can accumulate into vanishing/exploding.

## Bottom line

> Pre-LN residual attention provides an identity gradient highway and bounds the non-identity part, preventing vanishing/exploding in deep stacks.
    `;

export const INFO = {
  ...(ORIGINAL_INFO || {}),
  input_data_notes: {
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
  
  tokenizer_notes: {
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

  gpt2_byte_unicode_map_notes: {
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

  regex_pre_tokenizer_notes: {
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

  lossless_round_trip_notes: {
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

  mlp_fc_in_notes: {
    title: "MLP expansion: nn.Linear(n_embd, 4 * n_embd)",
    md: `
## What this layer does

- Projects each token's hidden vector from size $n\\_embd$ up to a larger **feed-forward** size $4 \\times n\\_embd$.
- This is the first linear layer in the Transformer block's MLP (a.k.a. FFN):

  $$
  h' = \\phi(\\underbrace{W_1 h + b_1}_{\\text{this layer}}), \\quad y = W_2 h' + b_2
  $$

## Why 4x? (a common design in Transformers)

- The MLP is the main place where non-linearity and per-token mixing capacity live.
- Using a wider hidden dimension increases expressive power at modest depth.
- Empirically, 4x is a strong trade-off between quality and compute/memory for GPT-2 style models. Larger ratios (e.g., 8x) can improve quality but cost more; smaller ones (2x) may underfit.

## Shapes

- **Input per token:** $h \\in \\mathbb{R}^{n\\_embd}$
- **After this layer:** $h_1 = W_1 h + b_1 \\in \\mathbb{R}^{4 n\\_embd}$
- **After activation (GELU):** same shape, $(4 \\times n\\_embd)$
- **Second linear maps back:** $(4 \\times n\\_embd) \\to (n\\_embd)$

## Parameter count (this layer only)

- **Weights:** $4 \\times n\\_embd \\times n\\_embd = 4 \\times n\\_embd^2$
- **Bias:** $4 \\times n\\_embd$
- **Total:** $\\sim 4 \\times n\\_embd^2 + 4 \\times n\\_embd$

## Alternatives / notes

- Some variants use ratios other than 4, gated linear units (GEGLU/SwiGLU), or different activations; the idea is similar: widen, apply a nonlinearity, project back.
- In Pre-LN GPT-2 blocks, this comes after LayerNorm (\`ln2\`) and runs after the attention sublayer, inside its own residual branch. (Running attention and MLP in parallel is a later variant, e.g. GPT-J / PaLM.)
`,
  },
  mlp_fc_out_notes: {
    title: "MLP projection back: nn.Linear(4 * n_embd, n_embd)",
    md: `
## What this layer does

- Projects features back from $4 \\times n\\_embd$ to $n\\_embd$ so the residual add aligns with the block input.
- FFN pattern recap:

  $$
  h' = \\phi(W_1 h + b_1), \\quad y = \\underbrace{W_2 h' + b_2}_{\\text{this layer}}
  $$

## Why project back?

Residual connections need matching dimensions; compressing to $n\\_embd$ keeps compute stable across layers.

## Shapes

- **Input:** $h' \\in \\mathbb{R}^{4 n\\_embd}$
- **Output:** $y \\in \\mathbb{R}^{n\\_embd}$

## Parameter count (this layer only)

- **Weights:** $n\\_embd \\times (4 \\times n\\_embd) = 4 \\times n\\_embd^2$
- **Bias:** $n\\_embd$
- **Total:** $\\sim 4 \\times n\\_embd^2 + n\\_embd$
`,
  },

  attn_residual_notes: {
    title: "Residual connection x = x + self.attn(self.ln1(x))",
    md: `
**This line is the pre-norm residual attention block. It does three things in one.**

## 1. Normalize before attention (pre-LN)

\`self.ln1(x)\` applies LayerNorm to each token's features so the attention sees a well-conditioned input (zero-mean, unit-variance per token). This stabilizes training.

## 2. Compute a context-dependent update

\`self.attn(...)\` (masked multi-head self-attention) looks at other tokens and returns an update with the same shape as $x$, $(B, T, C)$. You can think of it as a learned correction $\\Delta x$ telling each token what to borrow from its context.

## 3. Residual (skip) connection

\`x = x + (...)\` adds that update onto the original stream. The residual path:

1. gives a **gradient highway**: there's an identity path, so gradients flow even if the attention sublayer is poorly scaled early in training;
2. lets the block learn a **small perturbation around identity**: if the attention output is near zero, the block behaves like a no-op, which makes deep stacks much easier to optimize.

## Why GPT-2 uses pre-LN

GPT-2 uses pre-LN, $x + \\mathrm{Attn}(\\mathrm{LN}(x))$, instead of post-LN, $\\mathrm{LN}(x + \\mathrm{Attn}(x))$:

- Pre-LN keeps the residual path's [gradient](#info:gradient_notes) close to 1, which [prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes) and enables training many layers more reliably.
- Attention also receives a normalized input, which helps its softmax and projections operate in a stable range.

## Shape check (so the add works)

\`\`\`text
x                 (B, T, C)
self.ln1(x)       (B, T, C)
self.attn(...)    (B, T, C)
sum (elementwise) (B, T, C)
\`\`\`

> **Big picture:** the line implements "take your current token representations, normalize them, compute a context-aware adjustment via attention, and add that adjustment back": a stabilized, residual refinement step.
`,
  },

  prevents_vanishing_exploding_notes: {
    title: "Prevents vanishing/exploding",
    md: prevents_vanishing_exploding_md,
  },

  mlp_prevents_vanishing_exploding_notes: {
    title: "Prevents vanishing/exploding",
    md: prevents_vanishing_exploding_md,
  },

  gradient_notes: {
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
`
  },

  gpt2_merge_ranks_notes: {
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

  attention_linear_notes: {
    title: "Attention linear",
    md: `
## What it is

\`self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)\` is a single fused projection that computes **Q, K, V** in one matmul. It is equivalent to stacking three \`Linear(n_embd, n_embd)\` layers:

$$
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}
$$

## Parameter shapes

- **Weight:** $(3 \\times n\\_embd,\\ n\\_embd)$
- **Bias:** $(3 \\times n\\_embd)$

Mathematically, $W$ maps $768 \\to 2304$, so it holds $768 \\times 2304$ numbers. PyTorch's \`nn.Linear\` stores it as **(out, in)** $= (2304, 768)$ and computes $x\\,W^\\top + b$, which is why the weight shape above looks transposed.

## One set per block, shared by all tokens

- **Separate per block.** Each of the 12 blocks has its own \`c_attn\`, with its own $W$ and $b$, initialized independently and trained independently. Nothing is shared between blocks, so each layer learns its own way of computing q, k and v.
- **Shared within a block.** Inside one block, every token, from every sequence in the batch and at every position, is multiplied by the **same** $W$ and shifted by the same $b$. \`c_attn\` has no notion of position: it applies the same rule to every position, so two positions' q, k and v differ only because their input vectors differ. For example, take the two occurrences of "the" in "the cat and the dog":
  - **In block 1**, they have the same token embedding, so the only difference in their inputs is their position embeddings.
  - **In later blocks**, each "the" has also gathered different context through attention: the first one sees only itself, the second sees "the cat and the". Their inputs drift further apart, and the difference now comes mostly from context, not just position.

## What it does to the tensor

1. Input $x \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$.
2. \`qkv = c_attn(x)\` gives $qkv \\in \\mathbb{R}^{B \\times T \\times 3n\\_embd}$. Each position's 768-dim vector is projected **on its own**, with the same $W$: no information moves between positions yet. That starts only at $q K^\\top$.
3. \`q, k, v = qkv.split(n_embd, 2)\` gives $q, k, v \\in \\mathbb{R}^{B \\times T \\times n\\_embd}$.
4. Split q, k and v into heads with $hs = n\\_embd / n\\_head$: \`view(B, T, n_head, hs).transpose(1, 2)\` gives $(B,\\ n\\_head,\\ T,\\ hs)$.

## Why fuse Q/K/V?

Fewer kernel launches and better cache use: one big GEMM instead of three separate linears. The results are identical to three separate linears.

## Parameter count

- **Weights:** $3 \\times n\\_embd^2$
- **Bias:** $3 \\times n\\_embd$

> Example ($n\\_embd = 768$): weights $3 \\cdot 768^2 = 1{,}769{,}472$, bias $3 \\cdot 768 = 2{,}304$.

## PyTorch default initialization

By default, PyTorch's \`nn.Linear\` initializes weights with **Kaiming uniform** and biases with a plain uniform with the same bound.

### Weights

PyTorch initializes \`Linear.weight\` with Kaiming (He) uniform:

\`\`\`python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
\`\`\`

Passing \`a=math.sqrt(5)\` is a PyTorch trick so that the resulting uniform range becomes $\\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}}, \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)$, numerically the same range you'd get from a plain uniform with that bound. This was done for backward compatibility when they refactored to use \`kaiming_uniform_\` ([forum post ↗](https://discuss.pytorch.org/t/nn-linear-default-weight-initialisation-assumes-leaky-relu-activation/46712/2)).

$$
W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right), \\quad \\text{fan\\_in} = n\\_embd
$$

For $n\\_embd = 768$ the bound is $\\approx \\pm 0.0361$.

### Biases

The bias is not Kaiming; it's sampled explicitly from a plain uniform with the same bound:

\`\`\`python
fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
bound = 1 / sqrt(fan_in)
init.uniform_(self.bias, -bound, bound)
\`\`\`

$$
\\Rightarrow b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right)
$$

> So: weights use Kaiming uniform configured to yield the same range as a simple uniform, while biases use that simple uniform directly.

**Note:** many Transformer/GPT-2 implementations override this and use a normal init $\\mathcal{N}(0, 0.02^2)$ for linear layers and embeddings. The original GPT-2 code (OpenAI's TensorFlow \`model.py\`) does exactly that for \`c_attn\`: weights $\\sim \\mathcal{N}(0, 0.02^2)$ and biases **all zero**. The site's code doesn't override anything, so it uses the PyTorch defaults above.

## Why random, and why small?

- **Why not all zeros?** If $W = 0$, every position gets the same q, k, v, and every entry of $W$ receives the same gradient, so the entries stay identical forever and can never learn different things. Random values break this symmetry.
- **Why small?** Small weights give small q and k, so the scores $q \\cdot k$ start near 0 and the softmax starts close to uniform. That keeps it out of saturation, so gradients flow from the first step.

Everything q, k and v come to mean is learned from these random starting points.

## References

1. **Kaiming initialization**

   Introduces the initialization for rectifiers (ReLU/LeakyReLU) and derives the variance-preserving scheme.

   [He et al., 2015 ↗](https://arxiv.org/pdf/1502.01852.pdf)

2. **PyTorch defaults**

   The PyTorch init code that shows the exact behavior.

   [PyTorch init code ↗](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50)

3. **GPT-2 / HF Transformers practice (overriding PyTorch defaults)**

   GPT-2 (HF Transformers): weight matrices are initialized from a (truncated) Normal(0, 0.02), controlled by \`initializer_range=0.02\` in the config. This is the widely used default when you instantiate GPT-2 from Transformers.

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/main/model_doc/gpt2)

4. **General Transformer init policy**

   The same policy is described in general Transformer init notes (HF and others): "weights ~ Normal(0, initializer_range) (often 0.02), biases 0; LayerNorm γ=1, β=0."

   [HF Transformers Modeling Utils ↗](https://huggingface.co/transformers/v3.1.0/internal/modeling_utils.html)
`,
  },

  attention_output_linear_notes: {
    title: "Attention output projection",
    md: `
**The output projection lets information from different heads interact before it enters the residual stream.**

\`\`\`python
self.proj = nn.Linear(n_embd, n_embd, bias=True)    # C → C
y = self.resid_drop(self.proj(y))                   # (B, T, C)
\`\`\`

## From side by side to mixed

[Merging the heads](#info:reorder_merge_heads_notes) only places the heads' results **side by side**. Each head occupies its own slice, and the heads have not interacted yet:

\`\`\`text
black: [1.33, 1.47 | 0.50, -0.20]
         head 0       head 1
\`\`\`

\`proj\` is a $C \\to C$ linear layer: **every** output dimension is a weighted sum of **all** $C$ input dimensions, so every output dimension can use information from every head.

Continuing the $C = 4$ example, suppose the weight row for output dimension 0 is \`[0.5, 0, 1, 0]\` and the bias is 0:

\`\`\`text
out[0] = 0.5 × 1.33 + 0 × 1.47 + 1 × 0.50 + 0 × (-0.20)
       = 0.665 + 0.50
       = 1.17
         ↑ from head 0   ↑ from head 1
\`\`\`

This single output number combines both heads. The other 3 output dimensions each use their own weight row to combine them in other ways.

Intuitively: if head 0 found "the adjective before me" and head 1 found "the subject of the sentence", \`proj\` combines the two clues into one useful update.

## Why it is needed

1. **Mixing the heads.** Without \`proj\`, head 0's result would always stay in dimensions 0–63, head 1's in 64–127, and so on: the heads' findings could not be combined before the next block.
2. **Writing into the residual stream.** The attention output is added back onto \`x\` (\`x = x + attn(...)\`). \`proj\` turns the concatenated result into [a form suited to that addition](#hover:proj_residual_stream_notes).

At a high level:

- **Q/K do routing:** which tokens attend to which other tokens, within each head.
- **V carries content.**
- **The projection chooses where and how to write that content into the residual space** (a learned basis change).

> It lets the model reweight heads, rotate features, and gate what survives to the next layer.

## Shapes and parameters

\`\`\`text
input y:  (B, T, C)
output:   (B, T, C)     same shape
\`\`\`

- **Weight:** $(n\\_embd,\\ n\\_embd)$
- **Bias:** $(n\\_embd)$
- **Count:** $768 \\times 768 + 768 = 590{,}592$ parameters in GPT-2 small.

More strictly, the input dimension is $n\\_head \\cdot hs$ (the merged heads), so the \`nn.Linear\` weight, stored as (out, in), is $(n\\_embd,\\ n\\_head \\cdot hs)$. In GPT-2, $n\\_head \\cdot hs = n\\_embd$ (e.g. $12 \\times 64 = 768$), so the two are the same.

Like \`c_attn\`, \`proj\` applies the same transformation to each token independently; it does not mix information **between tokens**. That only happens in [\`att @ v\`](#info:att_v_product_notes).

## PyTorch default initialization

See the "PyTorch default initialization" section in the [Attention linear notes](#info:attention_linear_notes) for the details.

## Residual-writing weights

\`proj\` is the last matrix before the attention output is added back to the residual stream:

\`\`\`text
x = x + attn(ln1(x))
        └─ last step inside attn:  y = self.resid_drop(self.proj(y))
                                                       ↑
                              last matrix with weights before the add
\`\`\`

(\`resid_drop\` has no parameters.) So \`proj\`'s weight decides what the attention writes into the residual stream, and how much. Weights in this position are called **residual-writing weights**. The MLP has one in the same position, its final [\`fc_out\`](#info:mlp_fc_out_notes), so each block has 2, and GPT-2 small (12 blocks) has 24.

Why this matters: at initialization, some implementations scale the residual-writing weights by $\\frac{1}{\\sqrt{N}}$, where $N$ is the number of residual layers ($N = 2 \\times n\\_layer = 24$ in GPT-2 small, so the same factor is also written $\\frac{1}{\\sqrt{2\\, n\\_layer}}$). This keeps the sum of $N$ residual additions from blowing up in variance with depth (derivation: [Variance at initialization](#info:variance_init_notes), section 8). Matrices that do not write to the stream, such as \`c_attn\`, are not scaled.

Whether to use it varies between implementations. The GPT-2 paper does it, and 🆕 nanoGPT initializes these weights with standard deviation \`0.02 / sqrt(2 * n_layer)\`. The code on this site does not; it uses PyTorch's default initialization.

### Why pre-norm stacks like GPT-2 need it

In a pre-norm block, $x \\leftarrow x + f(\\mathrm{LN}(x))$: LayerNorm normalizes only the sublayer's **input**. The residual stream itself is never normalized, so every block adds to it and its variance grows with the number of residual additions. Scaling the residual-writing weights by $\\frac{1}{\\sqrt{N}}$ keeps that sum in range. This is what the GPT-2 paper does: it scales the weights of residual layers at initialization by $\\frac{1}{\\sqrt{N}}$ to account for the accumulation on the residual path with depth.

In a post-norm block, $x \\leftarrow \\mathrm{LN}(x + f(x))$, the stream is renormalized after every addition, so it cannot accumulate in the same way.

> See a tiny numeric [example](#info:post_norm_scaling_example_notes) of residual scaling in a post-norm stack.
`,
  },

  proj_residual_stream_notes: {
    title: "Writing into the residual stream (hover)",
    md: `
**What does "a form suited to the residual stream" mean?** The concatenated vector and the residual stream \`x\` are both 768-dimensional, but the same position does not mean the same thing in the two. \`proj\` translates the former into the latter's "coordinates" before they are added.

**The residual stream is a shared whiteboard**

\`x\` starts at the embedding and passes through all 12 blocks, and every block adds something to it:

\`\`\`text
x = x + attn(ln1(x))
x = x + mlp(ln2(x))
\`\`\`

So the 768 dimensions of \`x\` are **shared** by every layer. Think of a whiteboard with 768 columns: what each column (more precisely, each direction) stands for is a convention the whole model forms together during training. Every layer reads from it and writes to it.

**The concatenated vector uses each head's own coordinates**

After merging the heads, dimensions 0–63 of \`y\` are head 0's output, 64–127 are head 1's, and so on. These numbers were computed in each head's own value space; their meaning is specific to that head and has nothing to do with the convention for whiteboard columns 0–63.

Adding \`y\` to \`x\` directly, without \`proj\`, would mean:

\`\`\`text
x[0:64]    += head 0's output
x[64:128]  += head 1's output
...
\`\`\`

Head 0 could then only ever change whiteboard columns 0–63, head 1 only 64–127, and each head's output coordinates would have to line up exactly with what those columns mean. That is far too rigid.

**What proj does**

\`proj\` is a $768 \\times 768$ matrix. It lets each head's result be written **anywhere** on the whiteboard, in any amount:

\`\`\`text
what head 0 found  ──proj──→  the columns that need it
what head 1 found  ──proj──→  other columns, or the same ones as head 0
\`\`\`

As an analogy: each head writes a note in its own shorthand, and \`proj\` is the translator who rewrites the notes in the whiteboard's common format and puts them in the right columns. Later layers only understand the whiteboard's format, not each head's shorthand.

> "What each column stands for" is only a way to build intuition. In practice these meanings are learned, usually do not match concepts a person can read directly, and are often spread across many dimensions rather than one meaning per dimension.
`,
  },

  variance_init_notes: {
    title: "Variance at initialization",
    md: `
**Several design choices in GPT-2, such as the $1/\\sqrt{hs}$ in attention and the scaling of residual-writing weights, exist to keep variances under control at initialization.** This note derives every variance involved, from LayerNorm's output to the attention scores and the residual stream. The numbers are for this site's code (PyTorch default initialization) and match measurements in PyTorch.

## 0. What variance is

The **mean** $E[X]$ of a random number $X$ is its average. The **variance** measures how far it typically is from the mean: the average squared distance.

\`\`\`text
Var(X) = E[(X − E[X])²]
       = E[X²] − (E[X])²        ← equivalent, often more convenient
\`\`\`

When the mean is 0, this is simply **Var(X) = E[X²]**, "the average of the square". That form is used throughout.

The **standard deviation** is √variance: the number's typical size. Variance 64 means standard deviation 8, i.e. values around ±8.

**What is random here:** at initialization the weights are drawn at random, so q and k are random too. All means and variances below are averages over that random initialization.

## 1. Four rules

**Rule 1: multiplying by a constant c multiplies the variance by c²**

\`\`\`text
Var(cX) = c² × Var(X)
e.g. if X has variance 64, then X / 8 has variance 64 / 8² = 1
\`\`\`

**Rule 2: for independent random numbers, variances add**

\`\`\`text
Var(X + Y) = Var(X) + Var(Y)      (X, Y independent)
\`\`\`

Variances add, not standard deviations, because random signs partly cancel. With two numbers that are each +1 or −1 with probability ½:

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

So the sum of $n$ such terms is typically about $\\sqrt{n}$ in size, not $n$.

**Rule 3: product of two independent numbers, both with mean 0**

\`\`\`text
Var(X × Y) = Var(X) × Var(Y)
\`\`\`

Why: for independent $X$ and $Y$, the average of a product is the product of the averages: $E[XY] = E[X] \\times E[Y]$. The same holds for $X^2$ and $Y^2$, which are independent too. So:

\`\`\`text
mean:      E[XY] = E[X] × E[Y] = 0 × 0 = 0      ← both have mean 0

variance:  Var(XY) = E[(XY)²] − (E[XY])²        ← definition of variance
                   = E[X²Y²] − 0²               ← because E[XY] = 0
                   = E[X²] × E[Y²]              ← X² and Y² are independent
                   = Var(X) × Var(Y)            ← mean 0, so E[X²] = Var(X) and E[Y²] = Var(Y)
\`\`\`

**Rule 4: the uniform distribution U(−a, a) has variance a²/3**

$U(-a, a)$ means "any value between $-a$ and $a$, all equally likely". Its mean is 0 (it is symmetric), so its variance is the average of $x^2$ over that range:

$$
\\mathrm{Var} = \\frac{1}{2a}\\int_{-a}^{a} x^2\\,dx = \\frac{1}{2a} \\cdot \\frac{2a^3}{3} = \\frac{a^2}{3}
$$

## 2. The starting point: LayerNorm's output h

Attention's input is \`ln1(x)\`, with shape $(B,\\ T,\\ C) = (B,\\ T,\\ 768)$: one 768-dim vector per token. \`c_attn\` turns each token's vector into that token's q, k and v separately, so it is enough to look at **one token at a time**. Below, **h** means one token's vector, \`ln1(x)[b, t, :]\` (768 numbers), and $h[j]$ is its $j$-th component. At initialization ($\\gamma = 1$, $\\beta = 0$), [LayerNorm](#info:layernorm) makes each token's 768 components have mean 0 and variance 1. With mean 0, variance is the average of the squares, so:

\`\`\`text
(h[0]² + h[1]² + ... + h[767]²) / 768 = 1
i.e.  h[0]² + h[1]² + ... + h[767]² = 768
\`\`\`

This is exact (up to a tiny eps), enforced by LayerNorm for **every** token. A 4-dim example:

\`\`\`text
before LayerNorm:       [2, 4, 6, 8]
subtract the mean 5:    [−3, −1, 1, 3]
variance:               (9 + 1 + 1 + 9) / 4 = 5
divide by √5 ≈ 2.236:   h = [−1.34, −0.45, 0.45, 1.34]

average of h[j]²:       (1.8 + 0.2 + 0.2 + 1.8) / 4 = 1     ✓
\`\`\`

## 3. The variance of the weights

This site's code uses PyTorch's default initialization: the weight of \`nn.Linear(768, …)\` is drawn from $U(-1/\\sqrt{768},\\ 1/\\sqrt{768})$, where 768 is the layer's number of inputs (see "PyTorch default initialization" in [Attention linear](#info:attention_linear_notes) for where this range comes from). By rule 4, with $a = 1/\\sqrt{768}$:

\`\`\`text
Var(W) = (1/√768)² / 3 = 1 / (3 × 768) ≈ 0.000434
\`\`\`

The bias is drawn from the same range, so $\\mathrm{Var}(b) = 1/(3 \\times 768)$ as well. (For comparison, original GPT-2 uses $N(0,\\ 0.02)$: $\\mathrm{Var}(W) = 0.02^2 = 0.0004$.)

## 4. One component of q: variance about 1/3

Fix one token. Its h is now 768 fixed numbers; only the weights are random. Component $i$ of that token's q (one of its 768 components) is:

\`\`\`text
q[i] = W[i,0] h[0] + W[i,1] h[1] + ... + W[i,767] h[767] + b[i]
\`\`\`

**Mean 0:**

\`\`\`text
E[q[i]] = E[W[i,0] h[0] + ... + W[i,767] h[767] + b[i]]
        = E[W[i,0]] h[0] + ... + E[W[i,767]] h[767] + E[b[i]]   ← (1) and (2)
        = 0 × h[0] + ... + 0 × h[767] + 0                       ← (3)
        = 0
\`\`\`

The three facts used:

1. **The average of a sum is the sum of the averages:** $E[X + Y] = E[X] + E[Y]$. This always holds, independent or not.
2. **A constant factor comes out of the average:** $E[cX] = c \\times E[X]$. Here $h[j]$ is a constant, because the token is fixed.
3. **Each weight and the bias have mean 0:** they are drawn from $U(-a,\\ a)$, which is symmetric around 0, so $E[W] = 0$ and $E[b] = 0$.

Intuitively: each term $W[i,j]\\,h[j]$ is positive or negative with equal chance, whatever $h[j]$ is, so it averages to 0.

**Variance:** each term is a random $W$ times the constant $h[j]$, so by rule 1 its variance is $h[j]^2 \\times \\mathrm{Var}(W)$. The 768 terms use independent weights, so by rule 2 they add:

\`\`\`text
Var(q[i]) = Var(W) × (h[0]² + h[1]² + ... + h[767]²) + Var(b)
          = Var(W) × 768 + Var(b)          ← the sum of squares from section 2
          = 1/(3 × 768) × 768 + 0.0004
          ≈ 1/3 ≈ 0.334                    ← measured: 0.334 ✓
\`\`\`

k and v are the same, about 1/3 each; they use other, independently drawn parts of \`c_attn\`'s weight. Thanks to LayerNorm, this holds for **any** input token.

## 5. One term q[i] × k[i]: variance 1/9

From here on, q and k are **one head's 64-dim pieces** (see [Query View](#info:query_view_notes)): q comes from the query token, k from the key token, and $i$ runs from 0 to 63. Each component still has mean 0 and variance about 1/3, by section 4.

q and k use different, independently drawn weights, so at initialization they are independent. By rule 3:

\`\`\`text
Var(q[i] × k[i]) = Var(q[i]) × Var(k[i]) = 1/3 × 1/3 = 1/9
\`\`\`

## 6. The dot product q · k: a sum of hs terms

With head size $hs = 64$, the score between the two tokens is:

\`\`\`text
q · k = q[0]k[0] + q[1]k[1] + ... + q[63]k[63]
\`\`\`

Different $i$ use different rows of the weights, drawn independently, so the 64 terms are independent. Each has variance 1/9, and by rule 2:

\`\`\`text
Var(q · k) = 64 × 1/9 ≈ 7.1
std        = √7.1 ≈ 2.67                  ← measured: 2.67 ✓
\`\`\`

**The Transformer paper's assumption:** if each component of q and k had variance 1, each term would have variance 1, and the sum variance 64 (standard deviation 8). That is where "variance about 64" comes from.

Either way, **the dot product's variance is proportional to $hs$**: doubling the head size doubles it.

## 7. Dividing by √hs: the variance no longer depends on hs

### The problem: the variance grows with hs

Section 6 in general form: $q \\cdot k$ is a sum of $hs$ independent terms, and by rule 3 each term $q[i]\\,k[i]$ has variance $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i])$. By rule 2:

\`\`\`text
Var(q · k) = hs × Var(q[i]) × Var(k[i])
\`\`\`

With this site's initialization ($\\mathrm{Var}(q[i]) = \\mathrm{Var}(k[i]) = 1/3$, so each term has variance 1/9):

\`\`\`text
hs     Var(q · k)          std
32     32 × 1/9  = 3.56    1.89
64     64 × 1/9  = 7.11    2.67     ← GPT-2 small
128    128 × 1/9 = 14.22   3.77
\`\`\`

The bigger the head size, the bigger the scores, even though q and k themselves have not changed.

### The fix: divide by √hs

The code computes \`(q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)\`, i.e. every score is multiplied by $c = 1/\\sqrt{hs}$. By rule 1, the variance is multiplied by $c^2 = 1/hs$:

\`\`\`text
Var(q · k / √hs) = (1/√hs)² × Var(q · k)                  ← rule 1 with c = 1/√hs
                 = (1/hs) × hs × Var(q[i]) × Var(k[i])     ← Var(q · k) from above
                 = Var(q[i]) × Var(k[i])                   ← hs cancels
\`\`\`

The result no longer contains $hs$: for every head size the scores have the same variance.

\`\`\`text
hs     before dividing    after dividing by √hs
32     3.56               3.56 / 32  = 1/9
64     7.11               7.11 / 64  = 1/9      std ≈ 0.33   ← measured: 0.334 ✓
128    14.22              14.22 / 128 = 1/9
\`\`\`

Under the Transformer paper's assumption (variance 1 per component), the same steps give $1 \\times 1 = 1$: standard deviation 1 for every head size.

### Why √hs and not hs

Variance scales by the **square** of the factor (rule 1). To divide the variance by $hs$, the scores are divided by $\\sqrt{hs}$. Dividing the scores by $hs$ itself would divide the variance by $hs^2$, leaving $\\mathrm{Var}(q[i]) \\times \\mathrm{Var}(k[i]) / hs$: now the scores would **shrink** as $hs$ grows, the opposite problem. Only $\\sqrt{hs}$ cancels $hs$ exactly.

### Why scores must not be too large

softmax is very sensitive to the **difference** between scores. Take two scores. With standard deviation 8 (the paper's assumption, before dividing), a difference of 16 is common:

\`\`\`text
scores [16, 0]  →  softmax ≈ [0.9999999, 0.0000001]
scores [2, 0]   →  softmax ≈ [0.88, 0.12]              ← the same scores divided by 8
\`\`\`

With $[16, 0]$ essentially all the weight goes to one token, and it barely moves when the scores change: for two scores, the output's sensitivity to their difference is $p(1-p)$, which is about $0.0000001$ here, versus about $0.105$ for $[2, 0]$. Tiny sensitivity means tiny gradients, so the model can hardly learn where to attend. See [Q @ K](#info:q_k_product_notes) for more.

## 8. The residual stream: 24 additions

The dot product sums 64 terms; the residual stream sums 24 layer outputs, under the same rule 2:

\`\`\`text
x_final = x_embedding + update_1 + update_2 + ... + update_24
\`\`\`

GPT-2 small has 12 blocks, each adding once from attention and once from the MLP: 24 additions. If each update has variance about $v$ and they are roughly independent:

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v      ← grows with depth
\`\`\`

**Scaling the residual-writing weights:** at initialization, the weights of [\`proj\`](#info:attention_output_linear_notes) (and the MLP's \`fc_out\`) are multiplied by $1/\\sqrt{24}$. An update is (approximately) linear in these weights: they are the last matrix the update passes through, so multiplying them by $c$ multiplies the update by $c$ (GPT-2 initializes biases to 0, so the bias does not get in the way). Each update therefore shrinks by $1/\\sqrt{24}$, and by rule 1 its variance becomes $v/24$:

\`\`\`text
Var(x_final) ≈ Var(x_embedding) + 24 × v/24
             = Var(x_embedding) + v           ← no longer grows with depth
\`\`\`

That is where the square root in $1/\\sqrt{N}$ comes from: variance scales by $c^2$, so to make $N$ variances add up to one, each update is scaled by $1/\\sqrt{N}$.

The GPT-2 paper and nanoGPT do this scaling; the code on this site does not (see [Attention output projection](#info:attention_output_linear_notes)).

## Summary

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

> All of this holds **only at initialization**, when the weights are random, symmetric and independent. Once training starts, the weights change and so do these variances. The point of these choices is to give training a starting point on a sensible scale.
`,
  },

  attn_dropout_notes: {
    title: "Attention dropout",
    md: `
**Attention dropout randomly zeroes some attention weights during training, after softmax and before the weighted sum with $V$.**

$$
A = \\mathrm{softmax}(S + \\text{mask}), \\quad \\tilde A = \\mathrm{Dropout}_p(A), \\quad Y = \\tilde A\\,V
$$

In code: \`att = self.attn_drop(att)\`, where \`att\` is already masked and softmaxed, and \`self.attn_drop = nn.Dropout(dropout)\` with \`dropout=0.1\` on this site. The shape does not change: $(B,\\ n\\_head,\\ T,\\ T)$ in and out.

## What it does to one row

Take the \`black\` row from the [softmax](#info:softmax_notes) example, and suppose that this time the black→black weight is the one dropped:

\`\`\`text
                    I      love   black  cats
softmax:         [ 0.13,  0.27,  0.60,  0 ]
drop one (→ 0):  [ 0.13,  0.27,  0,     0 ]
rest × 1/0.9:    [ 0.14,  0.30,  0,     0 ]
\`\`\`

For this training step, \`black\` cannot see itself and has to take its information from \`I\` and \`love\`. The next step drops a different random set.

## Why multiply by 1/(1 − p)

### On average, nothing changes

Each weight is kept with probability $1 - p = 0.9$ and then multiplied by $1/0.9 \\approx 1.11$, or zeroed with probability $0.1$. For the 0.60 weight over 10 steps, about 9 keep it and 1 drops it:

\`\`\`text
kept (9 times):    0.60 × 1.11 = 0.667
dropped (1 time):  0
average = (9 × 0.667 + 1 × 0) / 10 = 0.60
\`\`\`

On average it is exactly the original weight. In general:

$$
\\tilde A = \\frac{A \\odot M}{1 - p}, \\qquad M_{ij} \\sim \\mathrm{Bernoulli}(1 - p), \\qquad \\mathbb{E}[\\tilde A] = A
$$

So a single row's sum is no longer 1 after dropout: 0.44 in the example above (the largest weight was dropped), 1.11 if nothing is dropped, 0.97 if only the 0.13 is dropped. On average, it is 1.

### Why that matters: inference has no dropout

At inference dropout is off, and every row sums to exactly 1. Without the $1/(1-p)$ factor, rows during training would average only 0.9, so \`att @ v\` would come out about 10% smaller in training than at inference. The model would adapt to the smaller values, then see larger ones at inference. Scaling by $1/(1-p)$ restores what was dropped, so the average matches inference.

### Why not renormalize the row?

Dividing the row by its sum (0.44 above) would also make it sum to 1, but it is not needed:

- The average already matches, which is what matters.
- Dropout is meant to be noise ("a bit less information, at random"); each sample does not need to sum to exactly 1.
- A fixed factor is simple; renormalizing needs an extra sum and division per row.

So \`nn.Dropout\` uses "zero, then multiply by $1/(1-p)$" everywhere it is used (this is called **inverted dropout**).

## Why it helps

The goal is to **prevent over-reliance on any single attention edge**. Dropout does not boost small weights; it randomly cuts edges:

- Cutting a small edge (say 0.07) changes little.
- Cutting a large edge (say 0.60) removes that information for this step, and the loss gets worse.

To keep the loss low when any edge might be cut, the model learns to:

1. **Not keep important information in one place only.** E.g. "\`black\` is an adjective" should also reach later tokens through other paths.
2. **Not bet everything on one token**, so it can still decide reasonably when that token is missing.

At inference, dropout is off and the weights are whatever the model learned: if \`black\` really should mostly attend to itself, it still does.

> **Analogy.** A student always copies homework from the same classmate. The teacher sends that classmate home at random, so the student has to understand the problems or check several sources. At the exam the classmate is there, but the student no longer depends on them alone.

**Example.** Suppose a head usually puts most attention on "dogs" to decide that the verb should be "were". With attention dropout, that strongest edge is sometimes zeroed during training. The model must then use other cues (plurality elsewhere, other heads), which builds redundancy and avoids brittle shortcuts.

## When it is active

- **Training only:** enabled in \`model.train()\`.
- **Disabled at inference:** \`model.eval()\`.

## Not only in attention

GPT-2 uses the same \`nn.Dropout\` in several places: [embedding dropout](#info:emb_dropout_notes), [residual dropout](#info:resid_dropout_notes) and [MLP dropout](#info:mlp_dropout_notes). They all serve as **regularization**: random noise during training, so the model does not memorize the training data (overfit) and generalizes to text it has not seen.

## Choosing p

- Typical $p \\approx 0.1$; GPT-2 uses 0.1.
- Smaller datasets may benefit from a larger $p$.
- 🆕 Many recent LLMs (e.g. LLaMA) pretrain with dropout 0: the data is huge and usually seen about once, so overfitting is not the main problem.
`,
  },

  resid_dropout_notes: {
    title: "Residual dropout in Causal Self-Attention",
    md: `
**Causal residual dropout (a.k.a. dropout on the attention branch's residual update)**

See the [MLP residual-dropout notes](#info:mlp_dropout_notes) for what dropout is, why residual dropout is useful, train vs. eval behavior, and an example. Everything below focuses on what's specific to causal self-attention.

## What it does

Causal residual dropout is applied to the mixed feature vector after the heads are concatenated and projected by \`W_O\`, right before the residual add. At this point, all attention heads have been combined into a single update vector for each position.

## Why it is useful

Heads can become spiky (e.g., lock onto one token path). Even if the attention pattern is spiky, the written features still pass through \`W_O\` before being added to the original stream. Randomly muting parts of that write:

- discourages dependence on a single head/channel's features;
- encourages redundant cues across heads.

> Think of \`Q/K/V\` as the "read side" of the attention mechanism. Then \`W_O\` is the "write side". Attention dropout perturbs the read-side pattern, while residual dropout regularizes the learned write directions, encouraging more robust, distributed writes and reducing reliance on any single head/channel even if the read pattern is spiky.
`,
  },

  mlp_residual_notes: {
    title: "MLP residual",
    md: `
**Code:** \`x = x + self.mlp(self.ln2(x))\`

## 1. Normalize before the MLP (pre-LN)

\`self.ln2(x)\` applies LayerNorm per token (across embedding dimensions) so the MLP sees a well-conditioned input (roughly zero-mean, unit-variance per token). This stabilizes training.

## 2. Position-wise MLP transforms features

The feed-forward sublayer is applied independently at each time step (no token mixing). Typically:

\`\`\`text
x → Linear(n_embd → 4×n_embd) → GELU → Linear(4×n_embd → n_embd) → (Dropout)
\`\`\`

This expand → nonlinearity → project back pattern builds rich, non-linear feature interactions per token.

## 3. Residual (skip) connection

\`x + (...)\` adds the MLP's output back to the original stream. Benefits:

1. **Gradient highway:** there is an identity path, helping [prevent vanishing/exploding](#info:mlp_prevents_vanishing_exploding_notes) gradients early in training.
2. **Small refinements:** if the MLP output is near zero, the block behaves like a near-identity map.

## Shapes (so the add is valid)

\`\`\`text
x                 (B, T, C)
self.ln2(x)       (B, T, C)
self.mlp(...)     (B, T, C)
elementwise sum   (B, T, C)
\`\`\`

> **Big picture:** normalize token features → compute a per-token non-linear update with the MLP → add that update back. This is the pre-norm residual MLP block that improves stability and optimization in deep Transformers.
`,
  },

  highlevel_transformer_introduction_notes: {
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

  residual_why_notes: {
    title: "Why Residual Connections",
    md: `
**What would happen if a block replaced $x$ instead of adding to it? Comparing the two shows why GPT-2 needs residual connections.**

Count sublayers rather than blocks: each block has 2 (attention and MLP), so GPT-2 small has **24 sublayers** in total. $x_0$ is the input to the first block: token embedding + position embedding.

## 1. With residual: the original information stays intact

$$
\\begin{aligned}
x_1 &= x_0 + \\Delta_1 \\\\[4pt]
x_2 &= x_1 + \\Delta_2 = x_0 + \\Delta_1 + \\Delta_2 \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= x_0 + \\Delta_1 + \\Delta_2 + \\cdots + \\Delta_{24}
\\end{aligned}
$$

$x_0$ is still present, **unchanged**, in $x_{24}$. Each sublayer only has to learn *what to add*; none of them is responsible for preserving what was already there.

## 2. Without residual: the information is handed over 24 times

$$
\\begin{aligned}
x_1 &= f_1(x_0) \\\\[4pt]
x_2 &= f_2(x_1) = f_2(f_1(x_0)) \\\\[4pt]
&\\;\\;\\vdots \\\\[4pt]
x_{24} &= f_{24}(f_{23}(\\cdots f_2(f_1(x_0))\\cdots))
\\end{aligned}
$$

$x_0$ is now buried inside 24 nested functions. For the last layer to know "which token was here, and at which position", **every one** of the 24 sublayers must learn to pass that information through intact. If any one of them fails, it is lost.

> **Analogy.** Without residual, it is a game of telephone: 24 people pass a message along, each retelling it in their own words, and by the end it is distorted. With residual, the original document travels in a folder; each person only attaches a sticky note with additions. The last person sees the original plus every note.

## 3. Attention makes the problem worse

The attention output is a **weighted average of other positions' values**. Suppose position 5 attends mostly to positions 3 and 4; its attention output is then mostly their content.

- **Without residual:** that output *replaces* $x$, so position 5's own identity ("which token am I?") is washed out.
- **With residual:** the output is *added*, so position 5's own information is still there.

## 4. Gradients: the same story in reverse

During backprop, the gradient must travel from the last layer back to the first.

- **Without residual:** the gradient is multiplied by each of the 24 sublayers' Jacobians in turn. If each one scales it by a factor even slightly away from 1, the effect compounds:
  - $0.8^{24} \\approx 0.005$: the gradient almost vanishes, and the early layers barely learn.
  - $1.2^{24} \\approx 80$: the gradient explodes.
- **With residual:** each sublayer's Jacobian is $I + J_{\\text{sublayer}}$. The $I$ is the direct path from $x_0$, so the gradient can reach the first layers without being multiplied away.

See [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes) for the Jacobian picture in more detail.

## Bottom line

> Residual connections turn $x$ into an **add-only main road**, the residual stream. The original information is never overwritten, and gradients get a direct path back. This is a key reason 12-layer models, and today's models with 100+ layers, can be trained at all.
`,
  },

  causal_analogy_notes: {
    title: "Causal Analogy",
    md: `
[Causal Analogy](#info:causal_analogy_notes) · [Attention Inference Example](#info:attn_inference_example_notes)

**Below is an analogy that shows how the GPT-2 multi-head self-attention block works, to give you an intuitive understanding.**

## Setting

A row of people, one per position $t$ in the sequence. Each person holds notes $x_t$ (corresponding to the embedding of the token at position $t$).

## 1. Make three cards from your notes: \`c_attn(x) → q, k, v\`

- **Query ($q$):** what I'm looking for now (built by $W_q$).
- **Key ($k$):** what I can offer, i.e. what I can be matched on (built by $W_k$); an index used for matching only.
- **Value ($v$):** what I will share if you listen (built by $W_v$); the actual content that gets copied/mixed into someone else's representation.

### Why separate Query and Key?

Consider a simpler design with no $W_q$ or $W_k$, scoring directly with $x$: $\\text{score}(i \\to j) = x_i \\cdot x_j$. This has two problems.

**1. Attention would be symmetric.** The dot product is symmetric, $x_{\\text{bank}} \\cdot x_{\\text{river}} = x_{\\text{river}} \\cdot x_{\\text{bank}}$, so bank would have to attend to river exactly as much as river attends to bank. But relations in language are often one-way: bank needs river to tell which sense it is, while river is clear on its own. With separate q and k, the two scores use different vectors and can differ. A 2-D example:

- $q_{\\text{bank}} = [0, 2]$, $k_{\\text{bank}} = [1, 0]$
- $q_{\\text{river}} = [0, 0]$, $k_{\\text{river}} = [0, 2]$
- bank → river: $q_{\\text{bank}} \\cdot k_{\\text{river}} = 0 \\cdot 0 + 2 \\cdot 2 = 4$ (strong attention)
- river → bank: $q_{\\text{river}} \\cdot k_{\\text{bank}} = 0 \\cdot 1 + 0 \\cdot 0 = 0$ (none)

(Under the causal mask river can't see the later word bank anyway; the point is that scores between any two words can be asymmetric.)

**2. Each word would attend mostly to itself.** With raw $x$, a word's score for itself is $x \\cdot x = \\|x\\|^2$, which is usually larger than its score for any other word. Every position would lean toward itself, or toward words **similar** to it. But bank isn't looking for another bank; it needs river or money, words that are **different** and add information. Separate $W_q$ and $W_k$ pick out different features of $x$, so the model can learn "my q matches words whose k complements me".

> **In short:** separating q and k lets scores be **asymmetric** (A attending to B doesn't mean B attends to A), so the model can learn to find words that are **different from itself but useful to it**.

### Why separate Key vs Value?

Keys are for addressing (similarity search), Values are for content. You choose *who* to listen to via $q \\cdot k$ and copy what matters from those people via $v$.

Could one vector do both jobs, getting found and delivering content? No: *what gets you found* and *what you hand over* are different things. Take:

> The cat sat on the mat because it was tired.

The position of "it" needs to work out who "it" refers to:

- **it's q:** I'm a pronoun, looking for an earlier singular, animate noun.
- **cat's k:** I'm a singular, animate noun. → A good match.
- **mat's k:** I'm a singular, inanimate noun. → A weaker match.
- **cat's v:** the meaning of "cat" (an animal, furry, can get tired…). → Once matched, this is what is passed to "it".

cat's k and v carry **different things**:

- **k** holds the features that get cat **found**: singular, animate, a noun. These are matching labels.
- **v** holds what a finder actually **needs**: the meaning of "cat", so "it" learns that the one who is tired is a cat.

If $k = v$, "it" would receive only the labels "singular, animate, noun", not "cat" itself. In library terms (below): you search with the index card, but you take home the book's text.

> **In short:** q is what I'm looking for, k is what I can be found by, v is what you get once you find me. All three come from the same $x$, but each has its own $W$ that picks out the features it needs. q and k only decide **who to attend to**; v is the information that actually **gets moved**.

### A library analogy

- **Query:** your search request, e.g. "I want books on river geography".
- **Key:** each book's index card, with its title and keywords, used to **match** against your request.
- **Value:** the book's actual text. Once a book matches, this is what you **take away**.

Matching uses q and k; what you take away is v.

### Example: "I went to the river bank"

"bank" is ambiguous (riverbank or financial bank), and its token embedding is the same either way. Look at the position of "bank":

- **bank's q:** I'm an ambiguous word; I'm looking for words that tell me which sense I am.
- **river's k:** I'm a geography/nature noun. → A good match for bank's q.
- **went's k:** I'm a verb of motion. → A poor match.
- **river's v:** information about rivers, water, geography. → Because river matched, this is what gets passed to bank.

Later steps use them like this:

- step 3 scores relevance with **$q \\cdot k$**: bank's q is dotted with every earlier word's k;
- step 7 mixes the **v**'s with those scores as weights and hands the result to bank.

> Two caveats. The descriptions above ("I'm looking for…", "I'm a geography noun") are only an aid to intuition: real q, k and v are just 768 numbers each, and what they encode is learned during training, not designed by hand. And **every token has all three**: bank uses its q to ask earlier words, while its own k and v are matched and read by the words that come after it.

## 2. Split into committees (heads): reshape to $(B, n\\_head, T, hs)$

Each committee (head) works in parallel with its own $W_q, W_k, W_v$ and looks at a different perspective (syntax, long-range cues, etc.).

How each tensor is split into heads: [Query View](#info:query_view_notes) · [Key View](#info:key_view_notes) · [Value View](#info:value_view_notes)

### Why 12 heads instead of one?

**One head can only look one way.** In a single head, each position gets **one** set of attention weights (summing to 1 after softmax) and mixes the v's into **one** result. But a word often needs several kinds of information at once:

> The tired dog that chased the cat was sleeping

The position of "was" wants to know:

- **Who is the subject?** dog, skipping over "that chased the cat"; this decides *was* rather than *were*.
- **What state is the subject in?** tired.
- **What is the previous word?** cat, which helps with local structure.

With one head, these three compete for the **same** weights: more weight on dog means less on tired. The mixed result is a blur of everything, and none of the three jobs is done well.

**12 heads = 12 independent attention patterns.** Each head has its own 64-dim q, k and v and does attention on its own:

- head 3 finds the subject → almost all weight on dog;
- head 7 looks at the previous word → almost all weight on cat;
- head 9 finds modifiers → most weight on tired;
- …

Each head can **focus on one thing** without interfering with the others. Concatenating the 12 results gives "was" all of this information at once.

> **Analogy:** one person reads the sentence with one question in mind. Twelve people read the same sentence, each with a different question (find the subject, check the previous word, find modifiers…), and then pool their answers.

Two caveats:

1. **It costs nothing extra.** $12 \\times 64 = 768$, so parameters and compute are about the same as one 768-dim head. The same budget is split twelve ways to get twelve different views.
2. **The division of labor is learned, not assigned.** "Head 3 finds the subject" above is only an illustration. What each head ends up doing is decided by training, and some heads are hard to interpret. But clearly specialized heads do exist: in real GPT-2 small, head 7 of layer 0 mostly attends to the last few tokens, and it stops working when position information is removed.

## 3. Score who to listen to: \`att_logits = q @ k^T / sqrt(hs)\`

Each person compares their query with everyone's keys to get compatibility scores; dividing by $\\sqrt{hs}$ keeps them numerically calm so softmax doesn't explode.

What the $T \\times T$ score matrix looks like: [Q @ K Product](#info:q_k_product_notes)

## 4. Causal rule (no looking ahead): mask future positions to $-\\infty$

This enforces the LM factorization

$$
p(x_{1:T}) = \\prod_{t=1}^{T} p(x_t \\mid x_{\\le t-1})
$$

Why it exists:

- **No leakage during training:** even though we compute all positions in parallel for speed, the mask guarantees position $t$ never uses $t+1, \\dots$.
- **Matches generation:** at inference we generate left to right; causality during training makes the learned behavior consistent with how we use the model.
- **Prevents cheating:** without masking, the model could peek at the answer (future tokens) and cheat.

## 5. Turn scores into listening weights: \`att = softmax(att_logits, dim=-1)\`

Now each person has a probability distribution over the allowed earlier positions: *how much attention do I give to each source?* Rows sum to 1, giving interpretable mixing weights.

## 6. Practice with partial connections: \`attn_drop(att)\`

Randomly drop some links during training to build robustness and avoid over-reliance on one path.

## 7. Build your personalized summary from Values: \`y = att @ v\`

Mix others' Value vectors using your attention weights (e.g., $0.6\\,v_3 + 0.3\\,v_7 + \\cdots$).

> Remember: Keys help you choose; Values are what you take.

## 8. Reassemble committees and standardize: merge heads, then \`proj(y)\`

Stitch the head outputs back into one vector per position, project to model width, and feed the residual stream cleanly.

## Big picture

> At each position $t$: select earlier positions via Query-Key (addressing), copy useful content via Values, combine multiple perspectives (heads), and do it under a causal rule that matches left-to-right generation.
`,
  },

  attn_inference_example_notes: {
    title: "Attention Inference Example",
    md: `
[Causal Analogy](#info:causal_analogy_notes) · [Attention Inference Example](#info:attn_inference_example_notes)

**This example shows how a single attention head can infer the correct verb form (\`were\` vs \`was\`) from the context of the sentence.**

Input sentence: **The dogs that barked loudly were tired.**

We focus on predicting the token \`were\` (position 6) from the prefix \`The dogs that barked loudly\`.

## What happens (one head, simplified)

Goal: decide between \`were\` and \`was\`.

To do this, we need to know the true subject of the clause so we can infer the correct verb form. In this example, the true subject is \`dogs\` (plural), even though there's an intervening clause "that barked loudly."

### Notes

1. **The model doesn't actually know it's a verb position.** In real models there are no POS tags or grammar rules baked in. Instead, the model learns from training to recognize contexts where a verb is likely next. The model has a hidden state $x_t$ that is a function of:

   - all previous tokens and the token itself
   - the positional embedding $p_t$
   - many layers of self-attention and feed-forward networks

   The hidden state $x_t$ encodes contextual features like "we're in a place where a verb typically follows a plural subject...", but it could also encode other things:

   - **Likely punctuation next.** In this case, the model might predict a comma instead: "The dogs that barked loudly, …".
   - **Alternate continuations.** In this case, the model might predict a different verb such as \`seemed\` or \`are\`: "The dogs that barked loudly are tired."

   But in this toy example, we focus on why the model predicts \`were\` instead of \`was\`.

2. **Assumption: all parameters are already trained.** This is an inference example, so the projections $\\mathbf{W}_q, \\mathbf{W}_k, \\mathbf{W}_v$ (and the output projection $\\mathbf{W}_o$), the MLPs, LayerNorms, embeddings, and the LM head are all fixed. The example's numbers are illustrative, but they reflect what trained parameters typically make happen at inference.

## Step A: Values carry useful features

Suppose this head's Value vectors include:

- **pluralness** (singular $-1 \\,\\dots\\, +1$ plural)
- **subjectness** (is this token a good subject head?)

> This is a **simplification**. In real models, the Value vectors are **a high-dimensional learned representation of size head size** (e.g. GPT-2 small has 768 dimensions and 12 heads, so the head size is 64). So in real models it isn't literally a 2-number feature; we just projected the idea into two dimensions to illustrate what the head might encode.

Toy values:

- \`The\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`dogs\`: $\\mathbf{V} \\approx \\begin{bmatrix}+1.0 \\\\ +0.9\\end{bmatrix}$ ← plural noun, likely subject
- \`that\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.1\\end{bmatrix}$
- \`barked\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$
- \`loudly\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 \\\\ 0.0\\end{bmatrix}$

## Step B: Queries and Keys score who to listen to

The head compares the Query $\\mathbf{q}_5$ to each Key $\\mathbf{k}_j$ ($j = 1 \\ldots 5$). Made-up attention logits (after the $1/\\sqrt{hs}$ scale):

- to \`The\`: $-0.5$
- to \`dogs\`: $+2.57$ ← strong match
- to \`that\`: $+0.3$
- to \`barked\`: $-0.2$
- to \`loudly\`: $-0.1$

> The numerical values are made up to show the pattern (e.g., the subject getting the highest score). In the real model they come from the dot products with the trained $\\mathbf{W}_q, \\mathbf{W}_k$, plus masking.

Then we apply softmax to get the attention weights (they sum to 1):

- $w(\\mathtt{The}) = 0.04$
- $w(\\mathtt{dogs}) = 0.78$
- $w(\\mathtt{that}) = 0.08$
- $w(\\mathtt{barked}) = 0.05$
- $w(\\mathtt{loudly}) = 0.05$

## Step C: Mix Values with those weights (\`att @ v\`)

The weighted average produces the head's output at position 5:

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

> **Interpretation:** position 5 now carries a strong plural + subject cue.

## Step D: Why this flips the logits the right way

Think of the LM head as rows (one per vocab item) that dot with the final hidden vector. The rows for just the two verbs:

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4, +0.5\\end{bmatrix}$ (likes plural & subject)
- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1, +0.2\\end{bmatrix}$ (penalizes plural)

Dot products from this head's contribution:

$$
\\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot 0.78 + 0.5\\cdot 0.71 \\approx +1.45 \\\\[4pt]
\\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot 0.78 + 0.2\\cdot 0.71 \\approx -0.72
$$

After combining with other heads/layers and softmax over the vocab, \`were\` gets much higher probability than \`was\`.

## Why \`dogs\` has high influence here

- **Syntactic role:** it's the true subject of the clause containing the verb, even across "that barked loudly."
- **Training pressure:** across millions of examples, gradients push the parameters so that a verb's Query aligns strongly with the Key of its governing subject. That makes the attention weight on the subject high.
- **Value payload:** the subject's Value carries exactly the features (plurality, subjectness) that help pick the correct verb form. If the model predicts the wrong form, gradients increase $q \\cdot k$ for the right subject and shape its $\\mathbf{V}$ to be more informative.

## What if we flip to singular \`dog\`?

With \`dog\` (singular) instead of \`dogs\` (plural), the same attention head that tracks subject–verb agreement will now pass singular evidence forward, so the LM head will score \`was\` higher than \`were\`.

Here's the same toy walk-through, just flipped to singular. Prefix: \`The dog that barked loudly\`.

### A) Values carry features

Toy features: [pluralness, subjectness].

- \`The\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`dog\`: $\\mathbf{V} \\approx \\begin{bmatrix}-1.0 & +0.9\\end{bmatrix}$ ← singular noun, likely subject
- \`that\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.1\\end{bmatrix}$
- \`barked\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$
- \`loudly\`: $\\mathbf{V} \\approx \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}$

### B) Queries and Keys decide who to listen to

Position 5's query (for the next-token slot) compares against the keys of positions 1 to 5. As before, it strongly matches the subject:

- to \`The\`: $-0.5$
- to \`dog\`: $+2.57$ ← strong match
- to \`that\`: $+0.3$
- to \`barked\`: $-0.2$
- to \`loudly\`: $-0.1$

Softmax → attention weights:

$$
w(\\mathtt{The})=0.04,\\ \\ w(\\mathtt{dog})=0.78,\\ \\ w(\\mathtt{that})=0.08,\\ \\ w(\\mathtt{barked})=0.05,\\ \\ w(\\mathtt{loudly})=0.05
$$

### C) Mix values with those weights (\`att @ v\`)

$$
\\mathbf{y} \\approx 0.78\\cdot\\begin{bmatrix}-1.0, +0.9\\end{bmatrix} + 0.08\\cdot\\begin{bmatrix}0.0, 0.1\\end{bmatrix} \\approx \\boxed{\\begin{bmatrix}-0.78, +0.71\\end{bmatrix}}
$$

> **Interpretation:** strong singular signal plus subject cue at position 5.

### D) LM head rows (toy) and resulting logits

Think of each vocab token as a row that dots with the hidden vector.

- $\\operatorname{row}(\\mathtt{were}) \\approx \\begin{bmatrix}+1.4 & +0.5\\end{bmatrix}$ (likes plural & subject)

  $$
  \\operatorname{logit}(\\mathtt{were}) \\approx 1.4\\cdot(-0.78) + 0.5\\cdot 0.71 \\approx -0.74
  $$

- $\\operatorname{row}(\\mathtt{was}) \\approx \\begin{bmatrix}-1.1 & +0.2\\end{bmatrix}$ (penalizes plural, okay with subject)

  $$
  \\operatorname{logit}(\\mathtt{was}) \\approx (-1.1)\\cdot(-0.78) + 0.2\\cdot 0.71 \\approx +1.00
  $$

> So $\\operatorname{logit}(\\mathtt{was}) > \\operatorname{logit}(\\mathtt{were})$. After combining all heads/layers and softmax over the full vocab, the model strongly prefers \`was\` in the singular-subject context.
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
    title: "GPT-2 inference",
    md: `
  $$
  \\href{#info:gpt2_inference_notes}{\\text{GPT2 Inference Example}}
  $$
  
    `,
  },

  embedding_pos: {
    title: "Positional Embedding",
    md: `
## What it is

Transformers need order information because self-attention has no sense of sequence. Positional embeddings give each token a unique vector based on its position.

> GPT-2's positional embeddings are **learnable parameters** in a lookup table, trained by backpropagation exactly like any other weight. They are not the [fixed sinusoids](#info:sinusoidal_pos_notes) of the original Transformer paper.

The shape of the positional embedding weight matrix is:

$$
W_{pos} \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
$$

- $\\text{max\\_toks}$ is the maximum sequence length (1024 for GPT-2).
- Each row $p_j$ corresponds to position $j$ in the sequence.

## Why do we need positional embeddings?

Intuitively, they give the model a sense of order and position in the sequence. But how exactly does that work?

### Without position information

Suppose we only have the token embeddings $X \\in \\mathbb{R}^{T \\times n\\_embd}$ (ignoring the batch dimension for simplicity). Self-attention computes:

$$
Q = X W_q, \\quad K = X W_k, \\quad \\text{scores} = Q K^\\top
$$

Now shuffle the sequence with a permutation matrix $P$. The new input is $X' = P X$, so

$$
Q' = X' W_q = P X W_q = P Q \\\\[2pt]
K' = X' W_k = P X W_k = P K
$$

and the scores become:

$$
\\text{scores}' = Q' K'^\\top = (P Q)(P K)^\\top = P \\, (Q K^\\top) \\, P^\\top
$$

These are the same numbers as before, with rows and columns reordered.

> Without position information (and without a mask), self-attention is **permutation-equivariant**: shuffling the inputs just shuffles the attention matrix the same way. The model can "see" which tokens are present, but not their order.

### With position information

Adding positional embeddings $p_t$ breaks that symmetry. Swapping two tokens now also swaps which $p_t$ each one gets, so $Q$ and $K$ change. The dot products include position-dependent terms, so $QK^\\top$ changes in value, not just by a row/column reordering. That's how the model becomes sensitive to order.

Still confused? See the [tiny example](#info:positional_embeddings_example_notes).

### What about the causal mask?

The argument above assumes every token can attend to every other token. GPT-2 uses a [causal mask](#info:causal_analogy_notes): position $t$ only attends to positions $0, \\dots, t$. The mask by itself carries order information in two ways:

- **Which tokens each position sees changes.** Shuffle \`A B C\` into \`C A B\`: \`B\` used to see $\\{A, B\\}$, now it sees $\\{C, A, B\\}$, so its output changes. The result is no longer just a reordering, so causal attention is already sensitive to order.
- **How many tokens each position sees differs.** Position $t$ always sees $t + 1$ tokens. Even in \`a a a a\`, where every position sees the same kind of token, the count still differs (e.g. averaging over the visible tokens gives each one a weight of $\\frac{1}{t+1}$). This lets the model infer its absolute position.

Haviv et al. (2022) trained causal language models with **no positional encoding at all** and found they were still competitive with standard models. Probing showed these models learn an implicit sense of absolute position; the authors conjecture that the causal mask lets each token infer how many tokens come before it.

> So "the model cannot see order" holds for attention without a mask. GPT-2 still adds explicit positional embeddings, and today's LLMs still use explicit position information (usually RoPE).

## Runtime usage

1. **Token IDs → token embeddings**

   $$
   \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
   $$

2. **Positions $0, 1, \\dots, T-1$ → positional embeddings**

   $$
   \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
   $$

3. **Add them together via broadcasting** ([What is broadcasting?](#info:broadcasting))

   $$
   x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
   $$

### Example with $n\\_embd = 4$

A toy vocabulary of 3 tokens, at most 3 positions, 4-dimensional vectors. The numbers are made up to keep the arithmetic easy; in a real model they start random and are learned.

\`\`\`text
tok_emb (3 x 4)                      pos_emb (3 x 4)
id 0  dog    [1.0, 0.0, 0.5, 0.0]    pos 0  [0.1, 0.1, 0.0, 0.0]
id 1  bites  [0.0, 1.0, 0.0, 0.5]    pos 1  [0.0, 0.1, 0.1, 0.0]
id 2  man    [0.5, 0.0, 1.0, 0.0]    pos 2  [0.0, 0.0, 0.1, 0.1]
\`\`\`

**Sentence 1:** \`dog bites man\`, ids = [0, 1, 2]

\`\`\`text
pos 0  dog    [1.0, 0.0, 0.5, 0.0] + p0 = [1.1, 0.1, 0.5, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  man    [0.5, 0.0, 1.0, 0.0] + p2 = [0.5, 0.0, 1.1, 0.1]
\`\`\`

**Sentence 2:** \`man bites dog\`, ids = [2, 1, 0]

\`\`\`text
pos 0  man    [0.5, 0.0, 1.0, 0.0] + p0 = [0.6, 0.1, 1.0, 0.0]
pos 1  bites  [0.0, 1.0, 0.0, 0.5] + p1 = [0.0, 1.1, 0.1, 0.5]
pos 2  dog    [1.0, 0.0, 0.5, 0.0] + p2 = [1.0, 0.0, 0.6, 0.1]
\`\`\`

What the result shows:

- **Same token, different position → different vector.** \`dog\` at position 0 in sentence 1 is [1.1, 0.1, 0.5, 0.0]; at position 2 in sentence 2 it is [1.0, 0.0, 0.6, 0.1]. Similar (both are \`dog\`), but not identical.
- **Same token, same position → identical vector.** \`bites\` is at position 1 in both sentences and gets [0.0, 1.1, 0.1, 0.5] both times.
- **Without positional embeddings**, both sentences would give the same three vectors {dog, bites, man}, only in a different order. Now the inputs for "dog bites man" and "man bites dog" really differ.

Shapes, with both sentences in one batch ($B = 2$):

\`\`\`text
idx           = [[0, 1, 2],
                 [2, 1, 0]]       (2, 3)
tok_emb(idx)                      (2, 3, 4)
pos_emb(pos)   pos = [0, 1, 2]    (3, 4)
  [None, :, :]                    (1, 3, 4)   broadcast to every sequence in the batch
x = sum                           (2, 3, 4)
\`\`\`

> Both sentences add the **same** $p_0, p_1, p_2$: a positional vector depends only on the position, not on the sentence.

## Parameters

$$
\\text{params} = \\text{max\\_toks} \\times n\\_embd
$$

> GPT-2 small: $1024 \\times 768 \\approx 7.86 \\times 10^5$, much smaller than the token embedding table (~38M params).

## Shapes at a glance

- **Weight:** $[\\text{max\\_toks}, n\\_embd]$
- **Positions:** $[T]$
- **Pos embeddings:** $[T, n\\_embd]$
- **Final sum:** $[B, T, n\\_embd]$

## Recent developments

Learned absolute positional embeddings, as used in GPT-2, are not widely used in the newest state-of-the-art LLMs. These have largely moved to relative positional methods such as [RoPE (Rotary Position Embeddings) ↗](https://arxiv.org/pdf/2104.09864) or [ALiBi (Attention with Linear Biases) ↗](https://arxiv.org/pdf/2108.12409).

How RoPE works, step by step: [RoPE](#info:rope_notes)

## References

1. **Transformer positional encodings**

   Vaswani et al., "Attention Is All You Need," §3.5. Introduces adding position information to token embeddings via fixed sinusoids.

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

2. **GPT-2 uses learned absolute positional embeddings**

   The Hugging Face GPT-2 docs explicitly note that GPT-2 uses absolute position embeddings (not sinusoidal), hence their advice to pad inputs on the right.

   [GPT-2 Configs ↗](https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2)

3. **GPT-2 technical report (input representation)**

   Radford et al., "Language Models are Unsupervised Multitask Learners." Describes the inputs as token + position embeddings summed before the stack.

   [GPT-2 Technical Report ↗](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

4. **Relative position encodings**

   Shaw et al., "Self-Attention with Relative Position Representations." Introduces relative positional embeddings as an alternative to absolute ones.

   [Self-Attention with Relative Position Representations ↗](https://arxiv.org/pdf/1803.02155)

5. **RoPE (Rotary Position Embeddings)**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding." Encodes relative positions in self-attention by rotating queries and keys.

   [RoPE: Rotary Position Embedding ↗](https://arxiv.org/pdf/2104.09864)

6. **ALiBi (Attention with Linear Biases)**

   Press et al. Adds distance-proportional biases to attention scores; enables length extrapolation.

   [ALiBi: Attention with Linear Biases ↗](https://arxiv.org/pdf/2108.12409)

7. **Causal LMs without positional encodings**

   Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information," Findings of EMNLP 2022. Shows that causal LMs with no positional encoding remain competitive and learn an implicit sense of position, likely through the causal mask.

   [arXiv:2203.16634 ↗](https://arxiv.org/abs/2203.16634)

  `,
  },

  sinusoidal_pos_notes: {
    title: "Sinusoidal Positional Encoding",
    md: `
**Fixed sin/cos position vectors, computed by a formula instead of learned.**

The original Transformer (Vaswani et al., 2017) adds these vectors to the token embeddings, the same way GPT-2 adds its [learned positional embeddings](#info:embedding_pos). Nothing about them is trained.

## The formula

Group the $d$ dimensions into pairs. Pair $i$ gets its own frequency $\\omega_i$:

$$
PE(pos, 2i) = \\sin(\\omega_i \\cdot pos), \\qquad PE(pos, 2i+1) = \\cos(\\omega_i \\cdot pos), \\qquad \\omega_i = \\frac{1}{10000^{2i/d}}
$$

- At $i = 0$, $\\omega = 1$: the fastest pair.
- As $i$ grows, $\\omega$ shrinks and the pair turns more slowly. The slowest pair has a wavelength of about $10000 \\cdot 2\\pi$.

## Intuition: a clock with many hands

Each pair $(\\sin, \\cos)$ is one **hand** on a circle. Each step in position turns hand $i$ by $\\omega_i$ radians.

- **Fast hands** (like a second hand) change a lot from one position to the next, so they tell neighbors apart.
- **Slow hands** (like an hour hand) only move noticeably over many positions, so they tell near from far.

Reading all the hands together pins down the position, just as the second, minute and hour hands together give the exact time.

## Example with $d = 4$

Two hands: $\\omega_0 = 1$ (fast) and $\\omega_1 = 1/10000^{2/4} = 1/100$ (slow).

\`\`\`text
         fast: sin,  cos      slow: sin,  cos
pos 0  [  0.000,  1.000,         0.000, 1.000 ]
pos 1  [  0.841,  0.540,         0.010, 1.000 ]
pos 2  [  0.909, -0.416,         0.020, 1.000 ]
pos 3  [  0.141, -0.990,         0.030, 1.000 ]
\`\`\`

- The fast hand changes a lot at every step, so neighboring positions are easy to tell apart.
- The slow hand barely moves; it takes on the order of a hundred steps to turn noticeably. It says roughly which stretch of the sequence you are in.

## Key property: shifting by $k$ is a fixed rotation

For each pair:

$$
\\begin{bmatrix} \\sin(\\omega(p+k)) \\\\ \\cos(\\omega(p+k)) \\end{bmatrix}
=
\\begin{bmatrix} \\cos \\omega k & \\sin \\omega k \\\\ -\\sin \\omega k & \\cos \\omega k \\end{bmatrix}
\\begin{bmatrix} \\sin(\\omega p) \\\\ \\cos(\\omega p) \\end{bmatrix}
$$

The matrix depends only on $k$, not on $p$. So "look $k$ positions back" is the **same linear map at every position**. The paper gives this as its reason for the design: $PE_{pos+k}$ is a linear function of $PE_{pos}$, which should make it easy for attention to work with relative positions.

> GPT-2's learned table ends up with a similar shape: neighboring positions are close, and the vectors trace smooth loops at several speeds. The difference is that GPT-2 has to learn this from data, while sinusoidal encoding builds it in.

## Compared with GPT-2's learned embeddings

- **Parameters:** sinusoidal has none; GPT-2's table has $1024 \\times 768$.
- **Beyond the training length:** the formula gives a vector for any position; GPT-2's table has no row past 1023.
- **Structure:** sinusoidal is designed (hands at many speeds); GPT-2's is learned and ends up with similar smooth loops.
- **Results:** Vaswani et al. also tried learned positional embeddings and found the two "produced nearly identical results."
- **Extrapolation:** they chose sinusoids because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." In practice this works poorly: having a vector for an unseen position does not mean the model knows how to use it. The ALiBi paper measured this directly.

## Recent developments

**RoPE** (Rotary Position Embedding), used by most LLMs today, uses the same set of frequencies $\\omega_i$. The difference is that RoPE does not add a position vector to the input. Instead, inside every attention layer it **rotates** $q$ and $k$ by angles proportional to their positions, so $q \\cdot k$ depends only on the **relative distance** between the two positions.

How RoPE works, step by step: [RoPE](#info:rope_notes)

## References

1. **Attention Is All You Need, §3.5**

   Vaswani et al., 2017. Defines the sinusoidal encoding, compares it with learned embeddings, and explains the choice.

   [arXiv:1706.03762 ↗](https://arxiv.org/abs/1706.03762)

2. **ALiBi (Attention with Linear Biases)**

   Press et al. Tests how well sinusoidal, rotary and other position methods extrapolate to longer sequences than seen in training.

   [arXiv:2108.12409 ↗](https://arxiv.org/abs/2108.12409)

3. **RoPE**

   Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding."

   [arXiv:2104.09864 ↗](https://arxiv.org/abs/2104.09864)

    `,
  },

  rope_notes: {
    title: "RoPE (Rotary Position Embedding)",
    md: `
**RoPE (Rotary Position Embedding) is how most LLMs today encode position. GPT-2 does not use it (it uses a [learned position embedding](#info:embedding_pos)), but it is easiest to understand right at the $q \\cdot k$ step.**

## 1. Where the position information goes

**GPT-2: added to $x$ once, at the start.**

\`\`\`text
x = token embedding + position embedding    <- position mixed into x before block 1
q = x · W_qᵀ,   k = x · W_kᵀ
score = q · k
\`\`\`

**RoPE: $x$ is left alone; $q$ and $k$ are rotated just before scoring.**

\`\`\`text
x = token embedding                          <- no position embedding
q = x · W_qᵀ,   k = x · W_kᵀ
q' = rotate q by an angle set by its position
k' = rotate k by an angle set by its position
score = q' · k'
\`\`\`

Three differences:

- RoPE **adds nothing to $x$**: the residual stream carries no position vector.
- The rotation happens **inside every attention layer**, right before $q \\cdot k$, not once at the input.
- Only **q and k** are rotated, never **v**. Position affects *who to attend to*, not *what content is taken*.

## 2. Why rotation encodes position: the 2-D picture

Think of $q$ and $k$ as two arrows in a plane. Their dot product depends on their lengths and the **angle between them**.

Now say the vector at position $m$ is rotated by $m \\times \\theta$, with $\\theta = 10°$:

- q at position 5 turns 50°, k at position 3 turns 30°: the angle between them changes by $50° - 30° = 20°$.
- Shift both far along, q at 105 and k at 103: q turns 1050°, k turns 1030°, and the change is still $20°$.

> **After rotation, the score depends only on the distance between the two positions** ($5 - 3 = 2$), not on where they are. "The previous word" or "two words back" is scored by the same rule at position 5 and at position 1000.

## 3. Why "only relative distance" is a good property

**Most rules of language are relative.**

- An adjective usually sits **one position before** its noun: \`black cat\`.
- \`it\` usually refers to something **a little earlier**.
- A verb often comes **a few positions after** its subject.

These rules say "$n$ positions back", never "position 37". The same rule holds at the start of a text and at token 800.

**With absolute positions, the same rule has to be learned many times.** Suppose \`black cat\` appears twice:

\`\`\`text
positions 3, 4      :  black cat
positions 503, 504  :  black cat
\`\`\`

- **Relative distance only (RoPE):** both pairs are 1 apart, so the position part of the score is automatically the same. Learned once, it works everywhere.
- **Absolute positions:** the model sees "3 and 4" and "503 and 504", two unrelated pairs of numbers. That they are the same relation must be learned from data.

It is like teaching "look one seat ahead", which works in every seat, versus separately training "from seat 3, look at seat 2", "from seat 503, look at seat 502", and so on.

> **Content still matters.** "Only relative distance" applies to the **position** part of the score. q and k are still computed from the tokens, so at the same distance 1, \`black → cat\` may score high and \`the → of\` low. What RoPE guarantees is that the same content at the same distance gets the same score wherever it appears.

## 4. From 2-D to 64-D: rotate in pairs

Each head's q and k have $hs = 64$ dims. RoPE groups them into **32 pairs** and treats each pair as a 2-D vector that rotates in its own plane:

\`\`\`text
q (64 dims) = [ (q0, q1), (q2, q3), (q4, q5), ..., (q62, q63) ]
                 pair 0    pair 1    pair 2         pair 31
\`\`\`

At position $m$, pair $i$ is rotated by the angle $m\\,\\theta_i$, using the ordinary 2-D rotation:

$$
(a,\\, b) \\;\\rightarrow\\; \\bigl(a\\cos(m\\theta_i) - b\\sin(m\\theta_i),\\;\\; a\\sin(m\\theta_i) + b\\cos(m\\theta_i)\\bigr)
$$

Each pair has its **own speed**:

$$
\\theta_i = \\frac{1}{10000^{2i/64}}, \\qquad i = 0, 1, \\dots, 31
$$

- **Pair 0:** $\\theta = 1$ radian per position, about 57.3°. One full turn every $2\\pi \\approx 6.3$ positions.
- **Pair 16:** $\\theta = 0.01$ radian, about 0.57°. One full turn every ≈ 628 positions.
- **Pair 31:** $\\theta \\approx 0.00013$ radian. One full turn every ≈ 47,000 positions.

> **Radians.** Angles here are in radians, as in \`torch.sin\` / \`torch.cos\`. A full turn is $360° = 2\\pi \\approx 6.283$ radians, so 1 radian $= 180°/\\pi \\approx 57.3°$: the angle subtended by an arc as long as the radius.

**Why mix fast and slow pairs?**

- **Fast pairs** turn a lot between neighbors, so they can tell "adjacent" from "two apart". But they wrap around after a few positions, so on their own they can't tell far distances apart.
- **Slow pairs** barely change between neighbors, but their angles spread out over long distances, so they can tell "100 apart" from "1000 apart".

Together they cover both near and far, like the second, minute and hour hands of a clock.

### q and k are rotated by the same rule

Both use the **same** rule with the **same** 32 speeds $\\theta_i$. Only the angle differs, because it is set by each vector's own position:

- q at position $m$: pair $i$ turns $m\\,\\theta_i$;
- k at position $n$: pair $i$ turns $n\\,\\theta_i$.

A token's own q and k sit at the same position, so they turn by the same angle.

**Why the rule must be shared.** In the 2-D example ($\\theta = 10°$), q at 5 and k at 3 turn 50° and 30°, and the difference is $(5 - 3) \\times 10° = 20°$. Both angles are "position × the same $\\theta$", so the absolute positions cancel and only the distance is left.

If q used 10° and k used 7° instead:

- q at 5 and k at 3: $50° - 21° = 29°$;
- q at 105 and k at 103: $1050° - 721° = 329°$.

Same distance, different result: the score would no longer depend only on relative distance.

> "Same rule" means the same *rotation*, not the same *vector*. q and k are still different vectors, made by \`c_attn\` with different weights; RoPE only rotates each of them according to its position afterwards.

## 5. Relation to sinusoidal encoding

The [sinusoidal encoding](#info:sinusoidal_pos_notes) of the original Transformer is:

$$
PE(pos, 2i) = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\qquad PE(pos, 2i+1) = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

The frequencies are **exactly the same**, $1/10000^{2i/d}$, also in sin/cos pairs. The difference is how they are used:

- **Sinusoidal:** the sin/cos values are **added** to $x$, once at the input, and get mixed into the residual stream.
- **RoPE:** the sin/cos values **rotate** (multiply) $q$ and $k$, in every attention layer, and touch nothing else.

The sinusoidal note shows that shifting by $k$ positions is a rotation. RoPE applies that rotation **directly to q and k**, so $q \\cdot k$ automatically depends only on relative distance.

> 🆕 **The base.** The 10000 in the formula is called the base (\`rope_theta\` in Hugging Face configs). Llama 2 uses 10000; Llama 3 raises it to 500000, which slows the slowest pairs further to support longer contexts.

## 6. Beyond the training length

What happens when the sequence is longer than the model was trained on?

### GPT-2: the table has no row for it

GPT-2 gets position from a lookup table, one learned row per position:

\`\`\`text
pos_emb = nn.Embedding(max_toks, n_embd)    # real GPT-2: 1024 rows × 768
pos_emb(pos)                                 # pos = 0, 1, ..., T-1
\`\`\`

- Positions 0 to 1023 each have their own trained row.
- Position 1024 has no row, so the lookup fails with an index error.

So GPT-2 can see **at most 1024 tokens** at once. And since each row is learned separately, adding a new row would not help: it was never trained, so the model would not know what it means.

### How GPT-2 generates past 1024 tokens: a sliding window

During generation the context also includes the tokens the model has generated so far, so prompt + generated tokens quickly reach the limit. Generation code therefore crops the input at every step to the **most recent** 1024 tokens (nanoGPT's \`generate\` crops to \`block_size\`):

\`\`\`text
idx_cond = idx[:, -1024:]     # keep only the last 1024 tokens
logits = model(idx_cond)
\`\`\`

This is **not** splitting the text into separate chunks. Chunking would cut it into non-overlapping pieces:

\`\`\`text
[0 ~ 1023] [1024 ~ 2047] [2048 ~ 2999]
\`\`\`

and token 1024 could not even see token 1023, right next to it. The sliding window instead moves forward by one token per step:

\`\`\`text
generating token 3000: sees [1976 ~ 2999]
generating token 3001: sees [1977 ~ 3000]
generating token 3002: sees [1978 ~ 3001]
\`\`\`

Consecutive windows share 1023 tokens, so nearby context is always continuous: every token sees the 1023 tokens before it.

What is lost is everything **more than 1023 tokens back**. When generating token 3000, tokens 0 to 1975 are simply invisible, not remembered vaguely. If a name was mentioned at token 100, GPT-2 cannot use it at token 3000. GPT-2 has a short-term memory of about 1024 tokens.

> 🆕 This limit is a major reason later models pushed context length much further; many current models support 128K tokens or more.

### RoPE: any position has an angle

There is no table, only the rule "pair $i$ at position $m$ turns $m\\,\\theta_i$". Position 5000 can be plugged in directly, so nothing breaks. Combined with the relative-distance property (section 3), two tokens 3 apart get the same position part of the score at position 10 and at position 5000.

**But "computable" is not "works well".** A model trained on length 4096 and run on length 32,000 usually gets noticeably worse. The cause is the slowest pairs:

- During training they only ever turned by small angles. Pair 31 turns about 0.55 radian over 4096 positions, less than a tenth of a full turn.
- Further out, they reach angles **never seen in training**, and the model does not know how to read them.

> 🆕 Long-context models therefore add extra techniques, such as a larger base (Llama 3 uses 500000) or interpolating/rescaling positions (Position Interpolation, YaRN). The idea in each case is to keep far-away angles within the range seen during training.

In short: GPT-2 can never see more than 1024 tokens at once, and a sliding window only hides that by dropping older text; a RoPE model can take longer input, but needs adjustments to stay accurate there.

## References

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)**

   Su et al. introduce RoPE and show that the rotated $q \\cdot k$ depends only on relative position.

   [RoFormer ↗](https://arxiv.org/abs/2104.09864)

2. **Extending Context Window of Large Language Models via Positional Interpolation (2023)**

   Chen et al. scale positions down so that longer inputs fall within the trained range of RoPE angles.

   [Position Interpolation ↗](https://arxiv.org/abs/2306.15595)

3. **YaRN: Efficient Context Window Extension of Large Language Models (2023)**

   Peng et al. rescale RoPE frequencies differently for fast and slow pairs to extend context length.

   [YaRN ↗](https://arxiv.org/abs/2309.00071)
`,
  },

  positional_embeddings_example_notes: {
    title: "Positional Embeddings Example",
    md: `
**A tiny 2-D example: without positions, swapping two tokens only permutes the attention scores; with positions, the scores themselves change.**

## Setup (no positions)

Let the token embeddings be 2-D vectors:

$$
e(A) = \\begin{bmatrix}1, 0\\end{bmatrix}, \\quad e(B) = \\begin{bmatrix}0, 1\\end{bmatrix}
$$

Use identity projections, so

$$
Q = X, \\quad K = X, \\quad S = QK^\\top = XX^\\top
$$

For the sequence $[A, B]$:

$$
X = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}, \\qquad S = XX^\\top = \\begin{bmatrix}1 & 0\\\\0 & 1\\end{bmatrix}
$$

The sequence $[B, A]$ just permutes the rows of $X$. As shown in the [positional embedding notes](#info:embedding_pos), the scores become $S' = PSP^\\top$.

> These are the same numbers as $S$, just with rows/columns swapped. Self-attention can't tell $AB$ from $BA$: it only sees a permutation.

## Add positional embeddings

Let the position vectors be

$$
p_0 = \\begin{bmatrix}1, 1\\end{bmatrix}, \\quad p_1 = \\begin{bmatrix}-1, 1\\end{bmatrix}
$$

We feed $h_t = e(\\text{token}_t) + p_t$.

### Sequence $[A, B]$ ($A$ at position 0, $B$ at position 1)

$$
h_0 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}2, 1\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}-1, 2\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix} \\\\[4pt]
S = hh^\\top = \\begin{bmatrix}2 & 1\\\\-1 & 2\\end{bmatrix}\\begin{bmatrix}2 & -1\\\\1 & 2\\end{bmatrix} = \\begin{bmatrix}5 & 0\\\\0 & 5\\end{bmatrix}
$$

### Sequence $[B, A]$ ($B$ at position 0, $A$ at position 1)

$$
h_0 = \\begin{bmatrix}0, 1\\end{bmatrix} + \\begin{bmatrix}1, 1\\end{bmatrix} = \\begin{bmatrix}1, 2\\end{bmatrix} \\\\[4pt]
h_1 = \\begin{bmatrix}1, 0\\end{bmatrix} + \\begin{bmatrix}-1, 1\\end{bmatrix} = \\begin{bmatrix}0, 1\\end{bmatrix} \\\\[4pt]
h = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix} \\\\[4pt]
S' = hh^\\top = \\begin{bmatrix}1 & 2\\\\0 & 1\\end{bmatrix}\\begin{bmatrix}1 & 0\\\\2 & 1\\end{bmatrix} = \\begin{bmatrix}5 & 2\\\\2 & 1\\end{bmatrix}
$$

> Now $S'$ is not just a row/column permutation of $S$; the dot products changed in value. Without positions, shuffling only permutes the attention matrix; with positions, the numbers themselves change, so the model becomes order-sensitive.
`,
  },

  layernorm: {
    title: "LayerNorm",
    md: `
**\`nn.LayerNorm\`** applies layer normalization over the last dimension of the input (the embedding dimension $n\\_embd$).

## The formula

Given any input vector $h \\in \\mathbb{R}^{d}$, LayerNorm computes

$$
\\text{LN}(h) = \\gamma \\odot \\frac{h - \\mu(h)}{\\sqrt{\\sigma^2(h) + \\epsilon}} + \\beta
$$

- $\\mu, \\sigma^2$ are the mean and variance over the last (feature) dimension.
- $\\gamma, \\beta \\in \\mathbb{R}^{d}$ are a learned scale and shift. [Shared by all tokens? One per block?](#info:layernorm_params_notes)
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

Each sublayer reads from the residual stream, which only ever accumulates: $x_{24} = x_0 + \\Delta_1 + \\cdots + \\Delta_{24}$ over the 24 sublayers ([why residual](#info:residual_why_notes)). Every token's vector has the same 768 dimensions, but its **magnitude** (norm) differs:

- the embedding rows already differ in size (in real GPT-2, the norm of position 0's row is 9.88; position 1023's is 0.12);
- each sublayer adds a different amount to each token;
- these differences build up with depth.

Why does that matter? Attention and the MLP apply the **same fixed weights** to every token in a layer. Without normalization, the size of what they receive would keep changing: from token to token within a layer, and from layer to layer as the stream grows. And size alone changes what they compute: a larger input can push attention's softmax into saturation and make GELU behave differently, even if its direction (its "meaning") is the same.

[Why? Examples: attention, GELU, and a moving target](#info:layernorm_scale_notes)

### 1. Stabilizes training by controlling magnitude

After $x \\rightarrow \\hat{x} = \\frac{x - \\mu}{\\sigma}$, the softmax in attention sees $\\frac{QK^\\top}{\\sqrt{d}}$ on a stable scale, reducing exploding logits and saturated softmax. The MLP nonlinearity also operates in a stable range.

**Quick numbers.** $Q$ and $K$ are linear in $x$, so doubling $x$ doubles both and multiplies every dot product by 4:

- scores $[2, 1, 0]$ → softmax $\\approx [0.665, 0.245, 0.090]$: attention is spread out;
- scores $[8, 4, 0]$ → softmax $\\approx [0.982, 0.018, 0.0003]$: almost all weight on one position, and gradients through the softmax nearly vanish.

[Why? A tiny numeric example](#info:stabilize_training_notes)

### 2. Improves gradient flow

The Jacobian around normalized activations is better conditioned (roughly scaled by $\\frac{1}{\\sigma}$), which curbs exploding/vanishing gradients and reduces sensitivity to the learning-rate choice.

[Why? A tiny numeric example](#info:improves_gradient_flow_notes)

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

See a [tiny numeric example](#info:post_norm_gradient_example_notes) that illustrates the issue with post-norm.

## Recent developments: RMSNorm

Among recent open-weight LLMs, **RMSNorm** is the default more often than LayerNorm. Examples:

- **Llama 2:** the paper says "we … apply pre-normalization using RMSNorm" (§2.2, Training Details). [Llama 2 paper ↗](https://ar5iv.labs.arxiv.org/html/2307.09288)
- **Llama 3:** Meta's official model code defines and uses \`RMSNorm\`. [meta-llama/llama3 model.py ↗](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- **Mistral 7B:** Hugging Face's Mistral model code defines and uses \`MistralRMSNorm\`. [modeling_mistral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- **Mixtral:** Hugging Face's Mixtral implementation (derived from the Mistral codebase) uses RMSNorm. [modeling_mixtral.py ↗](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)
- **Qwen:** model cards list the architecture as "transformers with RoPE, SwiGLU, RMSNorm …" (e.g. Qwen2.5-1.5B). [Qwen2.5-1.5B ↗](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Gemma:** Hugging Face's Gemma docs describe the model as using GeGLU activations and RMSNorm. [Gemma docs ↗](https://huggingface.co/docs/transformers/en/model_doc/gemma)

See a [tiny numeric example](#info:rmsnorm_gradient_example_notes) of how RMSNorm works.

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

  layernorm_params_notes: {
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

  layernorm_scale_notes: {
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

  stabilize_training_notes: {
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

  stabilize_training_no_layernorm_backprop_notes: {
    title: "Why satuated softmax leads to vanishing gradients",
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

  label_smoothing_notes: {
    title: "Label Smoothing",
    md: `
**Label smoothing replaces the one-hot target with a soft target, so the gradient $\\nabla_z L = p - y$ never becomes exactly zero, even when the model is extremely confident.**

## Why one-hot targets collapse

With standard (one-hot) cross-entropy, if the model is right and saturated:

$$
p \\approx [1, 0, \\ldots] \\text{ and } y = [1, 0, \\ldots] \\Rightarrow p - y \\approx [0, 0, \\ldots]
$$

That is a near-zero gradient (collapse).

## The smoothed target

With label smoothing $\\epsilon$, set the target for the correct class $c$ and the other $C - 1$ classes to

$$
y_c^{LS} = 1 - \\epsilon, \\qquad y_{j \\neq c}^{LS} = \\frac{\\epsilon}{C-1}
$$

Then, even if $p \\approx [1, 0, \\ldots]$, we get

$$
\\nabla_z L = p - y^{LS} \\approx \\left[\\epsilon, -\\frac{\\epsilon}{C-1}, \\ldots\\right]
$$

which is non-zero, so the logits still receive useful updates.

> Intuitively, label smoothing reduces overconfidence by nudging the top logit down a bit and pushing up the others slightly. This prevents complete gradient shutdown when the model is already confident, keeping training dynamics active and reducing brittle overfitting.

## Tiny example ($C = 3$, $\\epsilon = 0.1$)

- Smoothed target: $y^{LS} = [0.9, 0.05, 0.05]$.
- If $p = [0.999, 0.0005, 0.0005]$ (overconfident), then

$$
\\nabla_z L = [0.999 - 0.9,\\; 0.0005 - 0.05,\\; 0.0005 - 0.05] = [0.099, -0.0495, -0.0495]
$$

This is non-zero, so the weights still get updated.

## References

1. **Original paper**

   Rethinking the Inception Architecture for Computer Vision (Inception-v3). Introduces and motivates label smoothing as a regularizer.

   [Rethinking the Inception Architecture for Computer Vision ↗](https://arxiv.org/pdf/1512.00567)

2. **Transformer training detail**

   Attention Is All You Need. Explicitly states that they use label smoothing with $\\epsilon = 0.1$ and notes its effects.

   [Attention Is All You Need ↗](https://arxiv.org/pdf/1706.03762v2)

3. **Theory & empirical study**

   When Does Label Smoothing Help? (Müller, Kornblith, Hinton, NeurIPS 2019). Effects on generalization, calibration, and why it can hurt distillation.

   [When Does Label Smoothing Help? ↗](https://arxiv.org/pdf/1906.02629)

4. **Related regularizer (confidence penalty) and its connection to LS**

   Pereyra et al., ICLR 2017 workshop. Penalizes low-entropy outputs; shows the relation to label smoothing via the direction of the KL divergence.

   [Regularizing Neural Networks by Penalizing Confident Output Distributions ↗](https://arxiv.org/pdf/1701.06548)

5. **Calibration context**

   Guo et al., On Calibration of Modern Neural Networks (ICML 2017). Why model confidence is not calibrated and how label smoothing helps.

   [On Calibration of Modern Neural Networks ↗](https://arxiv.org/pdf/1706.04599)
`,
  },
  
  gradient_loss_to_logits_proof_notes: {
    title: "Gradient of loss to logits",
    md: `
**Proof that the gradient of softmax cross-entropy w.r.t. the logits is $p - y$.**

## Setup

- Logits: $z \\in \\mathbb{R}^C$
- Softmax: $p_i = \\dfrac{e^{z_i}}{\\sum_{k=1}^C e^{z_k}}$
- Target: $y$ (usually one-hot, but it can be any distribution, e.g. for label smoothing)
- Cross-entropy loss: $L = -\\sum_{i=1}^C y_i \\log p_i$

## Proof

Use the identity

$$
\\log p_i = z_i - \\log \\sum_{k=1}^C e^{z_k}
$$

Plug it into the loss:

$$
L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i \\left(z_i - \\log \\sum_{k=1}^C e^{z_k}\\right) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
$$

Because $y$ is a probability distribution, $\\sum_{i=1}^C y_i = 1$, so

$$
L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
$$

Differentiate w.r.t. $z_j$:

$$
\\frac{\\partial L}{\\partial z_j} = -y_j + \\frac{e^{z_j}}{\\sum_{k=1}^C e^{z_k}} = p_j - y_j
$$
`,
  },

  improves_gradient_flow_notes: {
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

  post_norm_gradient_example_notes: {
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

  rmsnorm_gradient_example_notes: {
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
x = \\frac{x}{\\mathrm{RMS}(x)} = \\begin{bmatrix}2/1.8708, -1/1.8708, 3/1.8708, 0/1.8708\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.5345, 1.6036, 0.0000\\end{bmatrix}
$$

### Step 3: apply per-feature scaling

$$
y = \\begin{bmatrix}1.0690 \\cdot 1.0, -0.5345 \\cdot 0.5, 1.6036 \\cdot 2.0, 0.0000 \\cdot 1.5\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.2673, 3.2071, 0.0000\\end{bmatrix}
$$

This is the RMSNorm output.

## Why do we need the scale factor $\\gamma$?

Pure normalization fixes the RMS of the output to $\\approx 1$, which is too restrictive and not ideal. $\\gamma$ lets the model learn and reshape that constraint per feature: different features can be scaled up or down as training discovers which directions are useful.
`,
  },

  post_norm_scaling_example_notes: {
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

  ln_jacobian_closed_form_proof_notes: {
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

  ln_gradient_attenuation_upper_bound_proof_notes: {
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
`
  },

  orthogonal_projection_notes: {
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

  orthogonal_projection_idempotent_notes: {
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

  orthogonal_projection_symmetric_notes: {
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
  orthogonal_projection_proof_notes: {
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

  optim_adamw_notes: {
    title: "torch.optim.AdamW",
    md: `
## What it is

AdamW is the dominant optimizer in current LLM pretraining. An optimizer is the decision-making algorithm in pretraining. At each step the optimizer answers the question: "Given the current parameters and the gradient, how should I update the parameters to reduce loss?" So:

- The **gradient** tells us the local slope of the loss with respect to each parameter.
- The **optimizer** gives us the algorithm that turns the gradient into an update.

## History of optimizers: from SGD to AdamW

From my learning experience, I find it's difficult to understand AdamW without understanding the history of optimizers. So below is a language-model-focused optimizer timeline with the key ideas, like who used each optimizer and why it changed. Click the links under each optimizer to see the details.

### 2010–2013: Sparse features era (word vectors, classic NLP)

- **[SGD](#info:sgd_notes)**: simple and fast; the original optimizer.
- **[SGD with momentum](#info:sgd_momentum_notes)**: adds velocity to smooth updates.
- **[Adagrad](#info:adagrad_notes)**: popular for sparse, high-variance gradients; helped early embeddings and logistic/softmax models. It adjusts the learning rate for each parameter during training and keeps adjusting it over time.
- **[RMSProp](#info:rmsprop_notes)**: fixes Adagrad's vanishing step size; widely used in early RNN LMs.

### 2014–2017: RNNs → early seq2seq

- **[Adam](#info:adam_notes)**: becomes the default in NLP (RNN LMs, seq2seq with attention). Why: it combines momentum with RMSProp-style variance normalization, so it needs much less LR fiddling than SGD.

### 2017–2019: Transformer takeoff

- **Adam + warmup standardizes**: a small linear warmup, then decay (often inverse-sqrt or cosine).
- Post-norm → pre-norm transitions; Adam's adaptivity plus warmup stabilize deep stacks.
- Weight decay starts to matter. BERT popularizes weight decay with Adam; the community converges on decoupled weight decay (AdamW) soon after, for predictability and better generalization.

### 2019–2021: Scaling up (T5, GPT-2/3, mT5)

- **[AdamW](#info:adamw_notes)** becomes the go-to for big LMs (bf16, warmup + cosine, grad clipping, no decay on norm/bias).
- **Adafactor** (memory-efficient, Adam-like) gains traction for T5/mT5-style models when optimizer-state RAM is tight (factored second moments).

### 2022–2023: Scaling up further

- Pretraining is still AdamW (or Adafactor where memory is the limiter).
- Finetuning: experiments with Lion, Adan, AdaBelief, Sophia, etc. show promise on some setups, but none broadly displaces AdamW for foundation-model pretraining.
- Systems stacks adopt fused AdamW kernels (Apex, DeepSpeed, xFormers) for throughput.

### 2024–today

- AdamW remains the mainstream default for LLM pretraining and most full-parameter finetunes.
- Adafactor remains a practical alternative when optimizer memory must be minimized.
- Research optimizers appear regularly, but production LMs stick to the stable duo above, paired with: pre-norm/RMSNorm, bf16, linear warmup (≈0.5–3% of steps), cosine decay, global-norm clipping, and no decay on norm/bias.
`,
  },

  sgd_notes: {
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
`,
  },

  sgd_momentum_notes: {
    title: "SGD with momentum",
    md: `
**Instead of stepping only with the current gradient, we keep a velocity: an exponential moving average (EMA) of past gradients.**

## What it is

The velocity smooths noise and accelerates motion in directions that stay consistently downhill.

We use the common EMA form:

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

- $g = \\nabla_{\\theta} L$ is the current gradient.
- $\\beta$ (e.g., 0.9) controls how much "memory" to keep.
- $\\eta$ is the learning rate.

## Tiny numeric example

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

> You can see the loss decreasing at each step (4.5 → 4.0613 → 3.3089). That's the optimizer using backprop's gradients to move $w, b$ toward better values. Compare with SGD, where the loss decreases faster (4.5 → 1.125 → 0.28125) in the first 3 steps of the example. The reason is below.

## Why momentum lags at first

The velocity needs to build up. With $\\beta = 0.9$, the velocity starts at 0, and in early steps it only tracks the current gradient by a fraction. That makes the initial steps smaller than plain SGD.

For example, in our tiny numeric example:

- **Step 1** keeps 90% of the past velocity ($0$ for now, because $v_w = 0, v_b = 0$) and adds 10% of the current gradient, so step 1 is 0.1× the raw gradient.
- **Step 2** keeps 90% of the past velocity (0.1× the raw gradient from step 1) and adds 10% of the current gradient, so step 2 is roughly 0.19× the raw gradient ($0.1 \\cdot 0.9 + 0.1 = 0.19$).
- **Step 3** keeps 90% of the past velocity (0.19× the raw gradient from steps 1–2) and adds 10% of the current gradient, so step 3 is roughly 0.271× the raw gradient ($0.19 \\cdot 0.9 + 0.1 = 0.271$).

## When and why momentum wins later

Once gradients stay roughly aligned, the velocity accumulates, so steps become larger and smoother than SGD. That pays off on harder problems, such as noisy or ill-conditioned losses. On this one-example problem it does not: SGD already converges very fast, while momentum overshoots and oscillates. After 10 steps momentum's loss is about 0.44, versus about $4 \\times 10^{-6}$ for SGD.

## Making momentum look better earlier

If we want momentum to look better earlier, we can:

1. Lower $\\beta$ (e.g., 0.8), so the velocity builds up faster.
2. Slightly raise $\\eta$ (e.g., 0.12) to make up for the smaller steps early on.
`,
  },

  adagrad_notes: {
    title: "Adagrad",
    md: `
**Adagrad (Adaptive gradient)** is an optimizer that gives each parameter its own learning rate and automatically adjusts it based on that parameter's past gradients.

> Parameters with large/frequent past gradients get smaller future steps; parameters with small/rare gradients get larger future steps.

## Update rule

Adagrad keeps a running **sum of squared gradients** for each parameter:

$$
s_i \\leftarrow s_i + g_i \\odot g_i
$$

where $g_i$ is the gradient of the $i$-th parameter and $s_i$ is the running sum of squared gradients for the $i$-th parameter.

Adagrad then divides the learning rate by the square root of the sum of squared gradients plus a small constant $\\epsilon$ to get the learning rate for the $i$-th parameter:

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

This way, parameters with large/frequent past gradients get smaller future steps, and parameters with small/rare gradients get larger future steps.

## Why the square root?

Because $s_i$ is a cumulative sum, it grows about linearly over time for stationary gradients ($s_i \\propto t$). Using $\\sqrt{s_i}$ in the denominator makes the effective learning rate decay like

$$
\\eta_{\\text{effective},\\,i} = \\frac{\\eta}{\\sqrt{s_i}} \\approx \\frac{\\eta}{\\sqrt{t}}.
$$

The $\\frac{1}{\\sqrt{t}}$ decay is gentle: it shrinks steps over time steadily, but not so fast that learning dies early. If we divided by $s_i$, we'd get $\\frac{1}{t}$ decay, which becomes tiny too quickly and can stall progress.

> This is also tied to classic online convex optimization results, where $\\frac{1}{\\sqrt{t}}$ yields good regret/convergence bounds.

## Tiny numeric example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Adagrad accumulator:** $s \\leftarrow s + g \\odot g$, starting from $s_0 = (0, 0)$
- **Update rule:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon}\\, g$
- **Hyperparameters:** $\\eta = 1.0$, $\\epsilon = 10^{-8}$ (tiny, so numerically negligible)
- **Init:** $\\theta_0 = (0, 0)$

Below we define $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s_t} + \\epsilon}$ as the effective learning rate at time $t$.

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)$
- $\\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3,\\; 1/2)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)$
- $\\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-1, 1) = (1, -1)$

Coordinate 1 and coordinate 2 now have different effective learning rates: $1/3$ vs $1/2$.

### Step 2

- $\\theta_1 = (1, -1)$
- $g_2 = (1 - 3,\\; -1 + 2) = (-2, 1)$
- $s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)$
- $\\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)$
- $\\theta_2 = \\theta_1 - \\triangledown = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)$

Coordinate 1's step shrank from $1/3$ to $0.277$; coordinate 2's step shrank from $1/2$ to $0.447$.

### Step 3

- $\\theta_2 = (1.554, -1.447)$
- $g_3 = (1.554 - 3,\\; -1.447 + 2) = (-1.446, 0.553)$
- $s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)$
- $\\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)$
- $\\theta_3 = \\theta_2 - \\triangledown = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)$

Coordinate 1's step shrank from $0.277$ to $0.257$; coordinate 2's step shrank from $0.447$ to $0.434$.

## Takeaways

This example shows that each coordinate $i$ gets its own learning rate $\\eta_{\\text{eff},\\,i}$, which depends on the past gradients of the $i$-th coordinate. These learning rates shrink over time as $s_i$ accumulates past squared gradients.

> Coordinates with larger/frequent gradients (here $\\theta_1$) get faster shrinkage, so they get smaller steps, while coordinates with small/rare gradients (here $\\theta_2$) get slower shrinkage, so they get larger steps.
`,
  },

  rmsprop_notes: {
    title: "RMSProp",
    md: `
**RMSProp** keeps an exponential moving average (EMA) of each parameter's squared gradient and scales the step by the RMS (square root) of that average.

## What it does

$$
\\begin{aligned}
s_t &= \\beta s_{t-1} + (1 - \\beta) (g_t \\odot g_t) \\\\
\\theta_{t+1} &= \\theta_t - \\frac{\\eta}{\\sqrt{s_t} + \\epsilon}\\, g_t
\\end{aligned}
$$

- $\\beta$ is the decay rate (e.g., 0.9).
- $s_t$ is the EMA of the squared gradient.
- $g_t$ is the gradient at time $t$, and $\\theta_t$ is the parameter at time $t$.
- $\\eta$ is the learning rate.
- $\\epsilon$ is a small constant to avoid division by zero.

## What RMSProp fixes (vs. Adagrad)

Adagrad sums squared gradients forever, so per-parameter step sizes shrink monotonically ($\\sim 1/\\sqrt{t}$) and can get too small in long runs.

RMSProp replaces the forever-growing sum with an EMA. This prevents the learning rate from decreasing too quickly, making the algorithm more effective in training deep neural networks.

### An analogy with momentum

We can think of RMSProp vs. Adagrad the way we think of SGD+momentum vs. SGD:

- **Adagrad → RMSProp:** use an EMA (memory) of the squared gradient statistic instead of the forever-growing sum.
- **SGD → SGD+momentum:** add an EMA (memory) of the gradient itself.

But there are still some key differences:

- **What gets the EMA:** RMSProp = squared gradients. SGD+momentum = gradients.
- **Per-parameter LR:** RMSProp (like Adagrad) keeps per-parameter adaptive step sizes; SGD+momentum (like SGD) uses one global LR (no per-parameter).
- **Failure mode fixed:** Adagrad's LR decays to near zero on long runs; RMSProp avoids that. SGD is noisy/zig-zaggy; SGD+momentum smooths and accelerates it.

## Tiny numeric example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **RMSProp accumulator:** $s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g)$, starting from $s_0 = (0, 0)$
- **Update rule:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon}\\, g$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta = 0.9$, $\\epsilon = 10^{-8}$ (tiny, so numerically negligible)
- **Init:** $\\theta_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.9, 0.4)$
- $\\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)$
- $\\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)$

### Step 2

- $\\theta_1 = (0.3162, -0.3162)$
- $g_2 = (0.3162 - 3,\\; -0.3162 + 2) = (-2.6838, 1.6838)$
- $s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot \\big((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)\\big) = (1.5303, 0.6435)$
- $\\sqrt{s_2} = (1.2370, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1247)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1247) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)$
- $\\theta_2 = \\theta_1 - \\triangledown = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)$

### Step 3

- $\\theta_2 = (0.5332, -0.5261)$
- $g_3 = (0.5332 - 3,\\; -0.5261 + 2) = (-2.4668, 1.4739)$
- $s_3 = 0.9 \\cdot (1.5303, 0.6435) + (1 - 0.9) \\cdot \\big((-2.4668, 1.4739) \\odot (-2.4668, 1.4739)\\big) = (1.9858, 0.7964)$
- $\\sqrt{s_3} = (1.4092, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121)$
- Update $\\triangledown = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4739) = (-0.1751, 0.1652)$
- $\\theta_3 = \\theta_2 - \\triangledown = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)$
`,
  },

  adam_notes: {
    title: "Adam",
    md: `
**Adam** combines the best ideas of RMSProp and SGD+momentum.

## What it is

- **RMSProp:** keeps track of the squared gradient (which captures the scale of the gradient) to normalize the step size.
- **SGD+momentum:** keeps track of the gradient itself (which captures the direction of the gradient) to smooth the updates.

Adam combines these two ideas in a single update rule:

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) (g_t \\odot g_t) && \\text{(EMA of squared gradients: RMS scaling)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_{t+1} &= \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(adaptive step per parameter)}
\\end{aligned}
$$

- $\\beta_1$ and $\\beta_2$ are the momentum and RMS scaling coefficients, respectively.
- $\\epsilon$ is a small constant to avoid division by zero.

> Bias correction fixes the biased EMA (exponential moving average) problem. We'll discuss it at the end of the notes.

## Tiny numeric example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update rule:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (tiny, so numerically negligible)
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

This example shows how Adam combines the best ideas of RMSProp and SGD+momentum to get a more effective optimizer:

- **Momentum:** $\\hat{m}_t$ is a smoothed version of the raw gradient, reducing noise and zigzagging.
- **Per-parameter adaptivity:** $\\hat{v}_t$ differs across coordinates, so each parameter gets its own effective step size $\\frac{\\eta}{\\sqrt{\\hat{v}_{t,i}} + \\epsilon}$.

## Bias correction

Adam keeps an **EMA (exponential moving average)** of:

- the gradient, $m_t$ (momentum), and
- the squared gradient, $v_t$ (RMS scaling).

Both EMAs start at zero ($m_0 = 0$, $v_0 = 0$). This biases the estimates toward zero, and the bias is large in the early steps, especially with the common $\\beta_1 = 0.9$, $\\beta_2 = 0.999$.

For example, for a constant gradient $g$, the first step would be:

- $m_1 = \\beta_1 m_0 + (1 - \\beta_1)\\, g$ ($m_1$ starts at $0.1g$),
- $v_1 = \\beta_2 v_0 + (1 - \\beta_2)\\, g^2$ ($v_1$ starts at $0.001g^2$).

So without bias correction, step 1 would use $0.1\\times$ the intended numerator and $\\sqrt{0.001} \\approx 0.0316\\times$ the intended denominator. The raw (uncorrected) step magnitude scales like

$$
\\eta \\cdot \\frac{m_1}{\\sqrt{v_1}} = \\eta \\cdot \\frac{0.1g}{0.0316g} = \\eta \\cdot 3.16 = \\eta \\cdot F_1
$$

So with the common setup $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $m_0 = 0$, $v_0 = 0$:

- $F_1 = 3.16$: the first step is too large by a factor of 3.16.
- Using similar reasoning, we get $F_2 = 4.25$, $F_3 = 4.95$, $F_4 = 5.44$.
- At $t = 12$, we reach a high of $F_{12} = 6.568$, and from there it descends to 1.

On the other hand, with bias correction, $\\hat{m}_1 = g_1$ and $\\hat{v}_1 = g_1^2$, so the $\\frac{\\hat{m}_1}{\\sqrt{\\hat{v}_1}}$ term becomes less sensitive to the choice of $\\beta$ and to the initial values $m_0$ and $v_0$.

### The fix

Adam fixes this by dividing by $1 - \\beta^t$:

$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
$$

> For the complete proof, refer to Section 3 of the original paper, [Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980).

## References

1. **Adam: A Method for Stochastic Optimization**

   Original paper of Adam.

   [Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980)

2. **StackExchange discussion**

   Discussion about the bias correction in Adam.

   [StackExchange Discussion ↗](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
`,
  },

  adamw_notes: {
    title: "AdamW",
    md: `
**AdamW** = Adam (momentum + RMS scaling + bias correction) plus **weight decay** applied separately.

## Update rule

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

- $\\beta_1$ and $\\beta_2$ are the momentum and RMS scaling coefficients, respectively.
- $\\epsilon$ is a small constant to avoid division by zero.
- $\\lambda$ is the weight decay strength (e.g., 0.1).

## Tiny numeric example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update rule:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (tiny, so numerically negligible), $\\lambda = 0.1$
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

- **Decay:** $\\theta_0 = (1 - \\eta \\lambda)\\, \\theta_0 = (1 - 0.1 \\cdot 0.1) \\cdot (0, 0) = (0, 0)$
- **Update:**

  $$
  \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon}\\, \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9, 4)}} \\cdot (-3, 2) = (0.1, -0.1)
  $$

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

- **Decay:** $\\theta_1 = (1 - \\eta \\lambda)\\, \\theta_1 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1, -0.1) = (0.099, -0.099)$
- **Update:**

  $$
  \\begin{aligned}
  \\theta_2 &= \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon}\\, \\hat{m}_2 = (0.099, -0.099) - \\frac{0.1}{\\sqrt{(8.70485, 3.80490)}} \\cdot (-2.94737, 1.94737) \\\\
  &= (0.1988973, -0.1988335)
  \\end{aligned}
  $$

### Step 3

- $\\theta_2 = (0.1988973, -0.1988335)$
- $g_3 = (0.1988973 - 3,\\; -0.1988335 + 2) = (-2.80110, 1.80117)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80110, 1.80117) = (-0.78411, 0.513117)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot \\big((-2.80110, 1.80117) \\odot (-2.80110, 1.80117)\\big) = (0.0252298, 0.0108426)$
- **Bias correction:**

  $$
  \\begin{aligned}
  \\hat{m}_3 &= \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78411, 0.513117)}{1 - 0.9^3} = (-2.893395, 1.89342) \\\\
  \\hat{v}_3 &= \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252298, 0.0108426)}{1 - 0.999^3} = (8.418349, 3.617817)
  \\end{aligned}
  $$

- **Decay:** $\\theta_2 = (1 - \\eta \\lambda)\\, \\theta_2 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1988973, -0.1988335) = (0.196908, -0.196845)$
- **Update:**

  $$
  \\begin{aligned}
  \\theta_3 &= \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon}\\, \\hat{m}_3 = (0.196908, -0.196845) - \\frac{0.1}{\\sqrt{(8.418349, 3.617817)}} \\cdot (-2.893395, 1.89342) \\\\
  &= (0.2966312, -0.2963911)
  \\end{aligned}
  $$
`,
  },
};
