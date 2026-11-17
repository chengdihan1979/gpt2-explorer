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
**\`fc_in\` is the first layer of the MLP. It turns each token's 768-dimensional vector into 3072 numbers, and each of those numbers measures how well the token matches one learned pattern.**

## Where it sits

\`\`\`python
self.fc_in = nn.Linear(n_embd, 4 * n_embd)    # 768 → 3072
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

The MLP's input is \`ln2(x)\`, the residual stream after LayerNorm, with shape $(B,\\ T,\\ C)$ where $C$ = \`n_embd\` = 768. The full MLP is:

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   ← this layer: widen 4×
→ GELU       (B, T, 3072)   non-linearity, shape unchanged
→ fc_out     (B, T, 768)    back to 768
→ dropout    (B, T, 768)
\`\`\`

\`nn.Linear\` only acts on the **last** dimension. Every token goes through the same weights, but separately: the MLP never mixes information between tokens. (Mixing between tokens is attention's job.)

## Each output is a pattern detector

[\`nn.Linear\` stores its weight as (out, in)](#hover:linear_weight_notes), so \`fc_in.weight\` has shape (3072, 768): 3072 rows, each with 768 numbers. For one token, let $h$ be its 768-dimensional input vector, i.e. \`ln2(x)[b, t, :]\`. Output number $j$ is

\`\`\`text
out[j] = row_j · h + bias[j]      (dot product, j = 0 … 3071)
\`\`\`

A dot product measures **how similar** two vectors are: large and positive when they point the same way, near 0 when unrelated, negative when they point opposite ways. So each row is a **pattern**, and \`out[j]\` is "how well this token matches pattern j". The rows are learned during training.

**A small example** with $C = 4$, bias 0, and input \`h = [1, 0, 2, -1]\`:

\`\`\`text
row 0 = [ 1, 0, 1, 0]   →  1×1 + 0×0 + 1×2 + 0×(-1) =  3    strong match
row 1 = [ 0, 1, 0, 1]   →  0×1 + 1×0 + 0×2 + 1×(-1) = -1    weak mismatch
row 2 = [-1, 0, 0, 1]   → -1×1 + 0×0 + 0×2 + 1×(-1) = -2    opposite
\`\`\`

In GPT-2 there are 3072 such detectors, so every token gets 3072 match scores.

## Why wider than 768

- The 768 dimensions are the [bandwidth of the residual stream](#hover:proj_residual_stream_notes): the shared "whiteboard" that every layer reads from and writes to.
- The model wants to recognize far more than 768 patterns: "this is a plural noun", "the previous word is *the*", "we are inside a code block", "this is the second half of a person's name", and so on.
- Widening to 3072 gives each layer 3072 detectors to check at the same time. Afterwards [\`fc_out\`](#info:mlp_fc_out_notes) compresses the results back to 768 dimensions and writes them onto the whiteboard.

\`\`\`text
768  ──fc_in──→  3072 match scores  ──GELU──→  ──fc_out──→  768
read whiteboard   wide "workspace"                          write back
\`\`\`

> The widening only helps because of the [GELU](#info:gelu) in the middle. Without it, \`fc_out(fc_in(h))\` would be two linear maps in a row, which equal a single 768 × 768 linear map, and the 3072 detectors would add nothing.

## Why exactly 4×

There is no derivation; it is an empirical choice. The original Transformer paper used 512 → 2048, which is 4×. A wider layer gives the model more capacity but costs more parameters and compute; 4× was a good trade-off in their experiments, and GPT-2 kept it.

> 🆕 LLaMA and many later models use **SwiGLU** instead of GELU. SwiGLU has 3 weight matrices instead of 2, so the width is reduced to about 8/3 × to keep the parameter count similar. For example, LLaMA-7B uses 4096 → 11008 (8/3 × 4096 ≈ 10923, rounded up to a multiple of 256).

## Parameters

For this layer: weight $3072 \\times 768$ plus bias $3072$:

\`\`\`text
768 × 3072 + 3072 = 2,362,368
\`\`\`

Compared with attention, the MLP has about **twice** as many parameters per block:

| Part | Layers | Parameters |
| --- | --- | --- |
| MLP | \`fc_in\`: 768 × 3072 + 3072 | 2,362,368 |
| | \`fc_out\`: 3072 × 768 + 768 | 2,360,064 |
| | **total** | **4,722,432** (≈ 4.72M) |
| Attention | \`c_attn\`: 768 × 2304 + 2304 | 1,771,776 |
| | \`proj\`: 768 × 768 + 768 | 590,592 |
| | **total** | **2,362,368** (≈ 2.36M) |

## Variants

- 🆕 Gated variants (GEGLU, SwiGLU) replace "Linear → GELU" with two parallel Linears whose outputs are multiplied; the idea is still widen → non-linearity → project back.
- GPT-2 runs attention first and the MLP second, each in its own residual branch. 🆕 Some later models (e.g. GPT-J, PaLM) run the attention and MLP branches in parallel on the same input.
`,
  },

  mlp_fc_out_notes: {
    title: "MLP projection back: nn.Linear(4 * n_embd, n_embd)",
    md: `
**\`fc_out\` is the last layer of the MLP. It turns the 3072 numbers from GELU back into a 768-dimensional vector, the update that is added to the residual stream.**

## Where it sits

\`\`\`python
self.fc_out = nn.Linear(4 * n_embd, n_embd)    # 3072 → 768
...
return self.drop(self.fc_out(self.act(self.fc_in(x))))
\`\`\`

\`\`\`text
input        (B, T, 768)
→ fc_in      (B, T, 3072)   widen: 3072 pattern scores
→ GELU       (B, T, 3072)   decide which patterns "fire"
→ fc_out     (B, T, 768)    ← this layer: back to 768
→ dropout    (B, T, 768)
\`\`\`

The result must be 768-dimensional because the block adds it to the residual stream, which is 768-dimensional: \`x = x + mlp(ln2(x))\`. Adding vectors only works when their shapes match.

## How nn.Linear computes

\`nn.Linear(in_features, out_features)\` computes

\`\`\`text
y = x @ weight.T + bias
\`\`\`

where \`weight\` is **stored** with shape (out, in). For \`fc_out = nn.Linear(3072, 768)\`, \`weight\` is (768, 3072), and the multiplication uses its transpose (3072, 768):

\`\`\`text
(B, T, 3072) @ (3072, 768) → (B, T, 768)
\`\`\`

More on \`nn.Linear\`: [why (out, in), why \`x @ W.T\` instead of \`W x\`, and what actually runs](#hover:linear_weight_notes).

## A worked example: 2 → 4 → 2

The real sizes 768 → 3072 → 768 cannot be computed by hand, so shrink them to **2 → 4 → 2**. It widens 2× instead of 4×, but the process is the same: first wider, then narrower. All biases are 0.

### Step 1: fc_in, 2 → 4

Input \`h = [1, 2]\`. \`fc_in.weight\` has shape (out, in) = (4, 2). Each of its 4 rows is a pattern detector, and each output is a dot product:

\`\`\`text
row 0 = [ 1,  1]  →  1×1 + 1×2    =  3
row 1 = [ 1, -1]  →  1×1 + (-1)×2 = -1
row 2 = [-1,  0]  → -1×1 + 0×2    = -1
row 3 = [ 0,  1]  →  0×1 + 1×2    =  2

scores = [3, -1, -1, 2]       ← 2 numbers became 4
\`\`\`

### Step 2: GELU, one number at a time

[GELU](#info:gelu) acts on each number separately, so the shape does not change. Roughly, it keeps positive numbers and pushes negative ones close to 0:

\`\`\`text
GELU(3)  ≈  3.00    pattern 0 fires
GELU(-1) ≈ -0.16    almost off
GELU(-1) ≈ -0.16    almost off
GELU(2)  ≈  1.95    pattern 3 fires

a = [3.00, -0.16, -0.16, 1.95]
\`\`\`

### Step 3: fc_out, 4 → 2

\`fc_out = nn.Linear(4, 2)\`. Its stored \`weight\` is (out, in) = (2, 4); the multiplication uses \`weight.T\`, which is (4, 2):

\`\`\`text
weight (2×4):              weight.T (4×2):
[ 1, 0, 1,  0 ]            [ 1,  0 ]   ← row 0
[ 0, 1, 1, -1 ]            [ 0,  1 ]   ← row 1
                           [ 1,  1 ]   ← row 2
                           [ 0, -1 ]   ← row 3
\`\`\`

\`a\` has shape [(4,)](#hover:vector_1d_notes), so \`a @ weight.T\` is (4,) @ (4, 2) → (2,). Multiplying a vector by a matrix from the left means: **use the vector's numbers as weights, and add up the matrix's rows**:

\`\`\`text
 3.00 × [1,  0] = [ 3.00,  0   ]
-0.16 × [0,  1] = [ 0,    -0.16]
-0.16 × [1,  1] = [-0.16, -0.16]
 1.95 × [0, -1] = [ 0,    -1.95]
                  ──────────────
out             = [ 2.84, -2.27]    ← 4 numbers back to 2
\`\`\`

Computing one output number at a time gives the same result. Each output number is a row of the stored \`weight\` dotted with \`a\`:

\`\`\`text
out[0] = [1, 0, 1,  0] · a = 3.00 − 0.16         =  2.84
out[1] = [0, 1, 1, -1] · a = −0.16 − 0.16 − 1.95 = −2.27
\`\`\`

Checked in PyTorch:

\`\`\`python
fc_out = nn.Linear(4, 2, bias=False)
fc_out.weight.shape        # torch.Size([2, 4])
# after setting weight = [[1, 0, 1, 0], [0, 1, 1, -1]]:
a = torch.tensor([3.00, -0.16, -0.16, 1.95])
fc_out(a)                  # tensor([ 2.8400, -2.2700])
a @ fc_out.weight.T        # tensor([ 2.8400, -2.2700])  same
\`\`\`

### The whole example

\`\`\`text
h = [1, 2]
  → fc_in  → [3, -1, -1, 2]               2 → 4: 4 detectors score the input
  → GELU   → [3.00, -0.16, -0.16, 1.95]   patterns 0 and 3 fire; 1 and 2 are nearly off
  → fc_out → [2.84, -2.27]                4 → 2: the firing patterns write their rows
\`\`\`

The output comes mostly from rows 0 and 3 of \`weight.T\`, because only patterns 0 and 3 fired.

## The MLP as "if … then write …" rules

The example suggests a way to read the whole MLP. Rule $j$ consists of:

- **row $j$ of \`fc_in.weight\`**: the pattern to check ("if the token looks like this …");
- **GELU of the score**: how strongly the rule fires (near 0 means it did not fire);
- **row $j$ of \`fc_out.weight.T\`** (the same numbers as column $j$ of the stored \`fc_out.weight\`): the 768-dimensional vector to write ("… then write this").

With the real sizes:

\`\`\`text
a: the 3072 numbers after GELU, a[j] = how much rule j fired
fc_out.weight.T: (3072, 768), row j = what rule j writes

out = a[0] × row 0 + a[1] × row 1 + … + a[3071] × row 3071 + bias
    = one 768-dimensional vector
\`\`\`

After GELU most \`a[j]\` are close to 0, so in practice only the few rules that fired write into the output.

So 3072 is the **number of rules per block**, and 768 is the **width of what gets written**. These are separate: many rules can write into the same directions, which is why the MLP can have far more rules than the residual stream has dimensions. Each of the 12 blocks has its own MLP with its own 3072 rules.

> 🆕 This view comes from the 2021 paper *Transformer Feed-Forward Layers Are Key-Value Memories* (Geva et al.): the rows of \`fc_in\` are keys and the rows of \`fc_out.weight.T\` are values. Later work such as ROME edits facts a model has memorized by changing MLP weights.

## Parameters

Weight 768 × 3072 plus bias 768:

\`\`\`text
3072 × 768 + 768 = 2,360,064
\`\`\`

Together with [\`fc_in\`](#info:mlp_fc_in_notes) (2,362,368), the MLP has 4,722,432 parameters per block.
`,
  },

  linear_weight_notes: {
    title: "How nn.Linear computes (hover)",
    md: `
**How \`nn.Linear\` computes**

\`nn.Linear(in_features, out_features)\` computes

\`\`\`text
y = x @ weight.T + bias
\`\`\`

- \`weight\` is **stored** as (out, in). For \`nn.Linear(4, 2)\` it is (2, 4).
- The multiplication uses the transpose \`weight.T\`, which is (in, out) = (4, 2), with \`x\` on the left:

\`\`\`text
x       @  weight.T  =  y
(4,)       (4, 2)       (2,)
\`\`\`

**Why store it as (out, in)? One output = one row.** Row $i$ of \`weight\` holds all the weights used by output $i$:

\`\`\`text
weight[0] = [1, 0, 1,  0]  → only used for out[0] = weight[0] · x
weight[1] = [0, 1, 1, -1]  → only used for out[1] = weight[1] · x
\`\`\`

So \`weight[i]\` gives output $i$'s weights directly. This is why each row of \`fc_in.weight\` can be read as one pattern detector. Stored as (in, out), the same weights would be a column, \`weight[:, i]\`. The results would be identical; (out, in) is a convention that matches the math y = W x.

**Why \`x @ W.T\` and not \`W x\`?** They give the same numbers; only the direction of the vectors differs. Math writes $x$ as a column, PyTorch stores each token as a **row**, and the two forms are transposes of each other: (W x)ᵀ = xᵀ Wᵀ.

| | Vector direction | Formula |
| --- | --- | --- |
| Math / papers | column | y = W x |
| PyTorch | row (one row per token) | y = x @ W.T |

Since \`x\` is (B, T, in) with one token per row, the matrix has to go on the right. \`x @ W.T\` works directly, while \`W @ x\` fails on the shapes.

**Leading dimensions are kept.** \`@\` only acts on the last dimension of \`x\`. If \`x\` is (B, T, 4), think of it as B × T separate rows of 4 numbers, each multiplied by the same (4, 2) matrix:

\`\`\`text
(B, T, 4) @ (4, 2) → (B, T, 2)
\`\`\`

**What actually runs.** \`out[i] = weight[i] · x\` is the mathematical definition, not the order of execution:

- \`.T\` copies nothing. It only swaps the strides, so \`weight.T\` reads the same memory in a different order.
- The whole layer is **one** matrix multiplication (\`addmm\`: bias + x @ weight.T). The library (BLAS on CPU, cuBLAS on GPU) splits it into tiles and computes all tokens and all outputs in parallel, not one output after another.
- The (out, in) layout is convenient: both \`x[t]\` and \`weight[i]\` are contiguous along the "in" dimension that the dot product sums over. But the libraries are fast with other layouts too, so the main reason for (out, in) is convention, not speed.

In GPT-2:

| Layer | Stored weight | Multiplication |
| --- | --- | --- |
| \`fc_in = nn.Linear(768, 3072)\` | (3072, 768) | (B, T, 768) @ (768, 3072) → (B, T, 3072) |
| \`fc_out = nn.Linear(3072, 768)\` | (768, 3072) | (B, T, 3072) @ (3072, 768) → (B, T, 768) |
`,
  },

  vector_1d_notes: {
    title: "(4,) vs (1, 4) (hover)",
    md: `
**\`(4,)\` is a 1-D tensor, not a 1×4 matrix**

\`(4,)\` means one dimension of length 4. It is neither 1 × 4 nor 4 × 1:

\`\`\`text
(4,)     1-D, 4 numbers                      a.dim() = 1
(1, 4)   2-D, 1 row and 4 columns (row)      a.dim() = 2
(4, 1)   2-D, 4 rows and 1 column (column)   a.dim() = 2
\`\`\`

The comma is Python syntax: a tuple with one element is written \`(4,)\`, because \`(4)\` is just the number 4.

**How \`(4,) @ (4, 2)\` works.** For \`@\`, PyTorch temporarily treats a 1-D tensor on the **left** as a row (1, 4), multiplies, then removes the extra 1:

\`\`\`text
(4,)  →  treated as (1, 4)  @ (4, 2)  =  (1, 2)  →  drop the 1  →  (2,)
\`\`\`

The numbers are the same as with a 1 × 4 matrix; only the shape lacks the 1. Checked in PyTorch, with \`W\` the (4, 2) matrix \`weight.T\` from the example:

\`\`\`python
a.shape                        # torch.Size([4])
(a @ W).shape                  # torch.Size([2])     1-D in, 1-D out
(a.reshape(1, 4) @ W).shape    # torch.Size([1, 2])  2-D in, 2-D out
a.reshape(4, 1) @ W            # error: 4x1 and 4x2 cannot be multiplied
\`\`\`

The last line shows that a column (4, 1) would not work here. On the left of \`@\`, a 1-D tensor behaves like a row.

A 1-D tensor on the **right** of \`@\` is treated as a column instead: \`W.T @ a\` is (2, 4) @ (4,) → (2,).

In the real model, \`x\` is (B, T, 3072) rather than 1-D. Each token's 3072 numbers play the role of \`a\`.
`,
  },

  erf_derivation_notes: {
    title: "Deriving Φ(x) = ½(1 + erf(x/√2)) (hover)",
    md: `
**Deriving Φ(x) = ½(1 + erf(x/√2))**

Start from the two definitions:

$$
\\Phi(x) = \\int_{-\\infty}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt, \\qquad \\operatorname{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-u^2}\\, du
$$

They differ in two ways: the exponent ($-t^2/2$ vs. $-u^2$) and the lower limit ($-\\infty$ vs. 0). Fix them one at a time.

*Step 1: split the integral at 0.*

$$
\\Phi(x) = \\int_{-\\infty}^{0} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt + \\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
$$

The first part is $\\tfrac{1}{2}$: the total area under the PDF is 1, and the curve is symmetric around 0, so exactly half lies left of 0. (This is $\\Phi(0) = 0.5$.)

*Step 2: substitute $t = \\sqrt{2}\\, u$ to turn the exponent into $-u^2$.* Three things change:

- exponent: $t^2/2 = (\\sqrt{2}\\,u)^2/2 = u^2$;
- differential: $dt = \\sqrt{2}\\, du$ (the derivative of $t = \\sqrt{2}\\,u$ is $\\sqrt{2}$);
- limits: $t = 0 \\Rightarrow u = 0$, and $t = x \\Rightarrow u = x/\\sqrt{2}$ (from $u = t/\\sqrt{2}$).

$$
\\int_{0}^{x} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-t^2/2}\\, dt
= \\int_{0}^{x/\\sqrt{2}} \\frac{1}{\\sqrt{2\\pi}}\\, e^{-u^2} \\sqrt{2}\\, du
= \\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
$$

using $\\dfrac{\\sqrt{2}}{\\sqrt{2\\pi}} = \\dfrac{\\sqrt{2}}{\\sqrt{2}\\,\\sqrt{\\pi}} = \\dfrac{1}{\\sqrt{\\pi}}$.

*Step 3: match erf.* erf has $\\tfrac{2}{\\sqrt{\\pi}}$ in front, and we have $\\tfrac{1}{\\sqrt{\\pi}}$, exactly half:

$$
\\frac{1}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du
= \\frac{1}{2} \\cdot \\underbrace{\\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x/\\sqrt{2}} e^{-u^2}\\, du}_{\\operatorname{erf}(x/\\sqrt{2})}
= \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)
$$

*Step 4: add the two parts.*

$$
\\Phi(x) = \\frac{1}{2} + \\frac{1}{2} \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right) = \\frac{1}{2}\\left(1 + \\operatorname{erf}\\!\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)
$$

*Check with x = 1:* erf(1/√2) = erf(0.7071) = 0.6827, so Φ(1) = 0.5 × (1 + 0.6827) = 0.8413 ✓.

- **Negative x works too.** The second integral then runs from 0 to the left and is negative; erf is negative for negative inputs too: erf(−z) = −erf(z). For example Φ(−1) = 0.5 × (1 − 0.6827) = 0.159.
- **erf(x/√2) has its own meaning:** it is the share of numbers between −x and x. The 0.6827 above is the 68.3% between −1 and 1 from the CDF popup. Reason: the second integral is the area from 0 to x, which equals ½ erf(x/√2); by symmetry the area from −x to x is twice that, erf(x/√2).
- **Why 2/√π in front of erf:** it makes erf(∞) = 1, i.e. 100% of the numbers lie between −∞ and ∞.
`,
  },

  gelu_backprop_notes: {
    title: "Backward through GELU: a worked example (hover)",
    md: `
**Backward through GELU: a worked example**

How does the gradient actually travel back from the loss, through GELU, to the weights of fc_in? Here is a tiny MLP, 2 → 4 → 2, computed step by step. All numbers come from PyTorch autograd and were checked by hand. It uses the exact GELU (GPT-2 uses the tanh version, which differs only in the 4th decimal).

## The setup

- input \`x = [1, 2]\` (one token, 2 features instead of 768)
- fc_in: \`W1\` is (4, 2), 4 detectors instead of 3072; its rows are \`[1, 1]\`, \`[1, −1]\`, \`[−1, 0]\`, \`[0, 1]\`
- fc_out: \`W2\` is (2, 4): \`[[1, 0, 1, 0], [0, 1, 1, −1]]\`
- no biases, to keep the arithmetic short
- a target \`y = [2, −2]\` and a simple loss, the squared error: \`loss = ½ × Σ (out − y)²\`. (GPT-2 uses cross entropy instead; the backward mechanism is the same.)

## Forward

\`\`\`text
h    = W1 · x     = [3, -1, -1, 2]
a    = GELU(h)    = [2.996, -0.159, -0.159, 1.954]
out  = W2 · a     = [2.837, -2.272]
loss = ½ × (0.837² + 0.272²) = 0.387
\`\`\`

For example \`h[1] = [1, −1] · [1, 2] = 1 − 2 = −1\`, and \`out[0] = a[0] + a[2] = 2.996 − 0.159 = 2.837\`.

## Backward: start at the loss, walk back one layer at a time

The rule is the **chain rule**: each time you go back through a layer, multiply the gradient you are holding by that layer's local derivative. "\`g_v\`" below means "the gradient of the loss with respect to v": how much the loss changes when v increases by 1.

### Step 1: the gradient of out

The derivative of \`½ (out − y)²\` with respect to out is \`out − y\`:

\`\`\`text
g_out = [2.837 − 2, −2.272 + 2] = [0.837, −0.272]
\`\`\`

out[0] is 0.837 too high, so it should go down; out[1] is 0.272 too low, so it should go up.

### Step 2: back through fc_out (out = W2 · a)

This layer produces two things.

**(a) The gradient of its weights W2**, used later to update W2. \`W2[i][j]\` affects the loss only through the term \`W2[i][j] × a[j]\` inside \`out[i]\`, so:

\`\`\`text
g_W2[i][j] = g_out[i] × a[j]

g_W2 = [[ 2.509, -0.133, -0.133,  1.637],
        [-0.814,  0.043,  0.043, -0.531]]
\`\`\`

For example \`g_W2[0][0] = 0.837 × 2.996 = 2.509\`.

**(b) The gradient of a**, passed further back. Write fc_out out in full:

\`\`\`text
out[0] = a[0]        + a[2]
out[1] =        a[1] + a[2] − a[3]
\`\`\`

One \`a[j]\` can appear in **several** outputs. The coefficient in front of \`a[j]\` in \`out[i]\` is \`W2[i][j]\`; going down the outputs, those coefficients are column j of W2. If \`a[j]\` increases by a small amount, every out[i] it appears in moves, and each of those moves changes the loss. So the gradient is the **sum over all paths**:

\`\`\`text
g_a[j] = Σ_i  g_out[i] × W2[i][j]
\`\`\`

Plugging in the numbers for all four:

\`\`\`text
g_a[0] = 0.837×1 + (−0.272)×0    =  0.837
g_a[1] = 0.837×0 + (−0.272)×1    = −0.272
g_a[2] = 0.837×1 + (−0.272)×1    =  0.565
g_a[3] = 0.837×0 + (−0.272)×(−1) =  0.272
\`\`\`

\`a[3]\` is not in out[0] (coefficient 0), so that path contributes nothing; its coefficient in out[1] is −1, which flips −0.272 into +0.272. In matrix form this is \`g_a = W2ᵀ · g_out\`: transposing turns column j of W2 into row j.

### Step 3: back through GELU (the key step)

GELU acts on each number separately, so its backward is also one number at a time: multiply by the GELU gradient at that point (the values in the gradient table of the GELU note).

\`\`\`text
g_h[j] = g_a[j] × GELU'(h[j])
\`\`\`

| j | h[j] | GELU'(h[j]) | g_a[j] | g_h[j] |
| --- | --- | --- | --- | --- |
| 0 | 3 | 1.012 | 0.837 | 0.847 |
| 1 | −1 | −0.083 | −0.272 | 0.023 |
| 2 | −1 | −0.083 | 0.565 | −0.047 |
| 3 | 2 | 1.085 | 0.272 | 0.295 |

- Detectors 0 and 3 fired (h > 0): GELU' ≈ 1, so the gradient passes almost unchanged.
- Detectors 1 and 2 did not fire (h = −1): GELU' = −0.083, so the gradient shrinks to about 1/12, but it is **not 0**.

**With ReLU instead**, ReLU' at h = [3, −1, −1, 2] is [1, 0, 0, 1]:

\`\`\`text
g_h = [0.837, 0, 0, 0.272]
\`\`\`

Detectors 1 and 2 get exactly 0. This is where "dying ReLU" comes from: a detector that did not fire learns nothing from this step.

### Step 4: back through fc_in (h = W1 · x)

Same rule as step 2(a): \`W1[j][k]\` only acts through \`W1[j][k] × x[k]\` inside \`h[j]\`:

\`\`\`text
g_W1[j][k] = g_h[j] × x[k]          x = [1, 2]

g_W1 = [[ 0.847,  1.695],
        [ 0.023,  0.045],
        [-0.047, -0.094],
        [ 0.295,  0.590]]
\`\`\`

Row j of \`g_W1\` is \`g_h[j] × x\`. Rows 1 and 2 are small but nonzero, thanks to GELU; with ReLU they would be exactly 0, and those two detectors' weights would not move.

In the real model the backward continues: it also computes \`g_x = W1ᵀ · g_h\` and passes it on to ln2, the residual stream, and back to the embeddings. This small example stops here.

## Update the weights (learning rate 0.1)

Each weight takes a small step against its gradient: \`w ← w − 0.1 × g_w\`.

\`\`\`text
W1 row 0: [1,  1] → [1 − 0.0847, 1 − 0.1695] = [0.915,  0.831]
W1 row 1: [1, −1] → [1 − 0.0023, −1 − 0.0045] = [0.998, -1.005]
W2 changes the same way.
\`\`\`

Forward again with the new weights:

\`\`\`text
loss: 0.387 → 0.163     ✓ smaller
\`\`\`

## The pattern

Going back, every layer does two things:

1. **Gradient of its own weights:** (gradient arriving here) × (the input this layer saw in the forward pass).
2. **Gradient to pass back:** (gradient arriving here) × (this layer's local derivative).

| Layer | Forward | Gradient passed back | Gradient of its weights |
| --- | --- | --- | --- |
| loss | ½ (out − y)² | out − y | none |
| fc_out | W2 · a | W2ᵀ · g_out | g_out × a |
| GELU | GELU(h), per number | g_a × GELU'(h), per number | none (no weights) |
| fc_in | W1 · x | W1ᵀ · g_h | g_h × x |

The backward needs values from the forward pass: \`a\` for W2's gradient, \`h\` for GELU', \`x\` for W1's gradient. PyTorch therefore keeps these intermediate values in memory after the forward pass, which is why training uses much more memory than inference.
`,
  },

  attn_residual_notes: {
    title: "Attention residual",
    md: `
**Code:** \`x = x + self.attn(self.ln1(x))\`

**This line is the first half of a Transformer block: normalize the residual stream, let attention compute an update for each token from the tokens before it, and add that update back.** The second half, \`x = x + self.mlp(self.ln2(x))\`, has the same shape (see [MLP residual](#info:mlp_residual_notes)).

## Where it sits

\`\`\`python
def forward(self, x):                    # x: (B, T, C), the residual stream
    x = x + self.attn(self.ln1(x))       # first half: attention  ← this line
    x = x + self.mlp(self.ln2(x))        # second half: MLP
    return x
\`\`\`

\`x\` here is the **residual stream**: each token's 768-number vector. In block 1 it is token embedding + position embedding; after that, every sublayer (12 blocks × 2 = 24 of them) adds its update to it. A useful picture is a notebook per token: the embedding writes the first draft, and every sublayer adds a note.

## The three steps

1. **\`self.ln1(x)\`, normalize first (pre-LN).** [LayerNorm](#info:layernorm) rescales each token's 768 numbers to roughly mean 0 and variance 1, so attention always sees inputs on a similar scale, which keeps the q·k scores and the softmax in a stable range. It produces a normalized **copy**; \`x\` itself is not changed.
2. **\`self.attn(...)\`, compute the update.** Causal self-attention lets each token take information from itself and the tokens before it (never after it, because of the causal mask), and returns an update of shape $(B, T, C)$. It ends with the output projection and [residual dropout](#info:resid_dropout_notes).
3. **\`x + ...\`, add it back.** The update is added number by number to the residual stream.

Shapes, which must match for the add:

\`\`\`text
x                 (B, T, C)
self.ln1(x)       (B, T, C)
self.attn(...)    (B, T, C)
x + ...           (B, T, C)
\`\`\`

The difference from the MLP half: **attention is the only place where tokens exchange information.** The MLP works on each token separately. So after this line, the token "sat" in "The cat sat" has information from "The" and "cat" in its \`x\`.

## Forward: a worked example

One token, first 4 dimensions, using the numbers from the [residual dropout note](#info:resid_dropout_notes) (training, dimension 1 dropped):

\`\`\`text
x (residual stream):      [0.20, -0.50,  1.00, 0.30]
attn(ln1(x)):             [1.30,  0,    -0.33, 0.89]   ← the update, after dropout
new x:                    [1.50, -0.50,  0.67, 1.19]
\`\`\`

The original information in \`x\` is kept; attention only adds a correction. In dimension 1 the update was dropped, and \`x\`'s −0.50 passes through unchanged.

## Backward: at the "+", the gradient is copied to both sides

Write the line as \`new_x = old_x + branch\`, where \`branch = attn(ln1(old_x))\`.

- If \`old_x\` increases by 0.001, \`new_x\` increases by 0.001.
- If \`branch\` increases by 0.001, \`new_x\` also increases by 0.001.

So the local derivative of "+" is **1 for both sides**, and the gradient arriving at \`new_x\` is passed to both sides unchanged. Suppose the gradient arriving from later layers is \`g = [0.3, −0.1, 0.2, 0.05]\`:

\`\`\`text
      ┌──→ straight to old_x:  [0.3, -0.1, 0.2, 0.05]   ← unchanged, × 1
g  ───┤
      └──→ into the branch:    [0.3, -0.1, 0.2, 0.05]   ← also unchanged; then back through
                                                           dropout, proj, attention, ln1
\`\`\`

The copy that enters the branch passes back through attention and ln1 and arrives at \`old_x\` as well; call it \`g_branch\`. So, by the "sum over paths" rule (see the [worked backward example](#hover:gelu_backprop_notes)):

\`\`\`text
gradient of old_x = g  (straight path)  +  g_branch  (path through attention)
\`\`\`

One difference from the MLP: attention mixed tokens in the forward pass, so in the backward pass the branch's gradient also flows back to the **earlier tokens** it read from. The gradient of token "cat"'s \`x\` includes a share from "sat"'s update, because "sat" took information from "cat".

## Why the residual matters: a gradient highway

GPT-2 has 24 lines of the form \`x = x + branch\`.

- **With residuals:** along the straight paths of all 24 "+", the gradient from the loss reaches the embeddings **unchanged**. Each branch adds its own contribution on top, but never blocks the straight path.
- **Without residuals** (\`x = attn(ln1(x))\`): the gradient would have to pass through all 24 sublayers one after another, multiplied by each one's local derivative. Suppose each sublayer happened to halve it on average: after 24 sublayers it would be 0.5²⁴ ≈ 0.00000006, and the first layers would learn almost nothing (**vanishing gradients**). If each doubled it, it would explode.

A related benefit at the start of training: the weights are small, so the update is close to 0 and \`new_x ≈ old_x\`. Every block starts close to "do nothing", and the deep stack trains like a shallow one until the blocks learn useful updates.

## Why LayerNorm goes inside the branch (pre-LN)

There are two ways to combine LayerNorm with the residual:

\`\`\`text
pre-LN  (GPT-2):                  x = x + attn(LN(x))     ← LN only on the branch
post-LN (original Transformer):   x = LN(x + attn(x))     ← LN after the add, on the main path
\`\`\`

- **Pre-LN:** the straight path \`x → x + ...\` contains only the "+", so the gradient along it is multiplied by exactly 1 in every block.
- **Post-LN:** the straight path goes through LN after every add, so the gradient along it is multiplied by LN's local derivative 24 times. These factors can compound and shrink or grow the gradient. Post-LN models are known to need a careful learning-rate warmup to train at all, and become hard to train when deep.

GPT-2 (2019) switched to pre-LN, and adds one extra LayerNorm, \`ln_f\`, after the last block, because with pre-LN the stream itself is never normalized on the main path. For the same argument written with Jacobians, see [Prevents vanishing/exploding](#info:prevents_vanishing_exploding_notes).

> 🆕 Pre-LN is now the standard. Many recent models (e.g. LLaMA) keep the same layout but use RMSNorm instead of LayerNorm: it only rescales and does not subtract the mean, which is a little cheaper.

## Summary

| | Forward | Backward |
| --- | --- | --- |
| Straight path (\`x\`) | information passes through complete | gradient passes back unchanged |
| Branch (\`attn(ln1(x))\`) | adds an update read from earlier tokens | adds its gradient, also to earlier tokens |
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
**\`resid_drop\` randomly zeros part of the attention output during training, right before that output is added to the residual stream.** It is the last step of causal self-attention.

## Where it sits

\`\`\`python
self.resid_drop = nn.Dropout(dropout)    # in __init__; this site uses dropout = 0.1
...
y = self.resid_drop(self.proj(y))        # end of forward; y: (B, T, C)
return y
\`\`\`

The returned \`y\` is what the block adds to the residual stream \`x\`:

\`\`\`python
x = x + self.attn(self.ln1(x))           # self.attn(...) is the y above
\`\`\`

So the order is: [output projection](#info:attention_output_linear_notes) \`proj\` → \`resid_drop\` → add to \`x\`. The name "resid" comes from this position: it sits directly before the residual add.

## Same rule as attention dropout

\`resid_drop\` is the same \`nn.Dropout\` as [attention dropout](#info:attn_dropout_notes) (\`attn_drop\`), with the same rule:

- **Training** (\`model.train()\`): each number is independently set to 0 with probability $p = 0.1$; every number that survives is multiplied by $1/(1-p) = 1/0.9 \\approx 1.11$. The multiplication keeps the **average** output equal to the input (why this works is explained in the attention dropout note).
- **Inference** (\`model.eval()\`): it does nothing; the output equals the input.

The only difference is **what it acts on**:

| | Acts on | Shape |
| --- | --- | --- |
| \`attn_drop\` | the attention weights after softmax ("how much to look at each token") | $(B,\\ n\\_head,\\ T,\\ T)$ |
| \`resid_drop\` | the output of \`proj\`: the update this attention layer is about to write into the residual stream | $(B,\\ T,\\ C)$ |

## A worked example

Continue the $C = 4$ example from the output projection note, for the token \`black\`. The first number 1.17 is the \`out[0]\` computed there; the other numbers, and the residual stream \`x\`, are made up for illustration:

\`\`\`text
dimension:           0      1      2      3
x (residual stream): [0.20, -0.50,  1.00,  0.30]
update = proj(y):    [1.17,  0.40, -0.30,  0.80]
\`\`\`

**Without dropout** (inference), the whole update is added:

\`\`\`text
x + update         = [1.37, -0.10,  0.70,  1.10]
\`\`\`

**With dropout** (training), suppose this time dimension 1 is drawn to be dropped:

\`\`\`text
dimension 1 → 0:     [1.17,  0,    -0.30,  0.80]
survivors × 1/0.9:   [1.30,  0,    -0.33,  0.89]    ← 1.17/0.9 = 1.30, -0.30/0.9 = -0.33, 0.80/0.9 = 0.89
x + dropped update = [1.50, -0.50,  0.67,  1.19]    ← 0.20+1.30, -0.50+0, 1.00-0.33, 0.30+0.89
\`\`\`

Look at dimension 1: the update was dropped, but \`x\`'s original **−0.50 is still there**. Dropout acts only on this layer's update; the residual stream \`x\` itself is untouched. The effect is "this time, attention did not write anything into dimension 1", not "the information in dimension 1 is lost".

The next training step draws a new random mask, so a different set of dimensions is dropped.

## Why use it

### 1. Regularization

Like every dropout, it prevents the model from relying on any single dimension of the attention output. Every number in the update may disappear in a given training step, so later layers learn not to depend on one particular write. The information has to be spread over several dimensions (and several heads), which generalizes better to unseen text.

The two dropouts in attention perturb different things:

- \`attn_drop\` perturbs the **read** side: which tokens each token takes information from.
- \`resid_drop\` perturbs the **write** side: what the layer writes back into the residual stream. Even if one head always attends to the same token, what it writes is still randomly dropped sometimes.

### 2. It does not damage the main path

Dropout is placed on the **branch** (\`attn(...)\`), not on \`x\`. The residual path, \`x\` passing straight through \`x + ...\`, always stays complete.

Suppose instead dropout were applied to \`x\` itself in every block. Then each of the 12 blocks would randomly erase part of everything the earlier layers had accumulated, and the losses would compound with depth. Deep networks would become much harder to train. Dropping only the branch update means each step loses at most this one layer's contribution.

## Not only in attention

The MLP branch has the same residual dropout at the same position: \`x = x + mlp(ln2(x))\`, where the MLP output also goes through dropout before the add. See [Residual dropout in MLP](#info:mlp_dropout_notes).

## Choosing p

- GPT-2 uses $p = 0.1$.
- 🆕 Many recent LLMs (e.g. LLaMA) pretrain with dropout 0 here too: the training data is huge and usually seen about once, so overfitting is not the main problem.
`,
  },

  mlp_residual_notes: {
    title: "MLP residual",
    md: `
**Code:** \`x = x + self.mlp(self.ln2(x))\`

**This line is the second half of a Transformer block: normalize the residual stream, let the MLP compute an update for each token, and add that update back.** After it, the block is finished and \`x\` goes on to the next block.

## Where it sits

\`\`\`python
def forward(self, x):                    # x: (B, T, C), the residual stream
    x = x + self.attn(self.ln1(x))       # first half: attention
    x = x + self.mlp(self.ln2(x))        # second half: MLP  ← this line
    return x
\`\`\`

It has exactly the same shape as the [attention residual](#info:attn_residual_notes) line; only the sublayer differs.

\`x\` here is the **residual stream**: each token's 768-number vector. It starts as token embedding + position embedding, and every sublayer (12 blocks × 2 = 24 of them) adds its update to it. By the time this line runs in block 5, \`x\` already contains the embedding plus the updates of 4 full blocks and of block 5's attention.

## The three steps

1. **\`self.ln2(x)\`, normalize first (pre-LN).** [LayerNorm](#info:layernorm) rescales each token's 768 numbers to roughly mean 0 and variance 1, so the MLP always sees inputs on a similar scale. This produces a normalized **copy**; \`x\` itself is not changed.
2. **\`self.mlp(...)\`, compute the update.** For each token separately (no mixing between tokens): [fc_in](#info:mlp_fc_in_notes) 768 → 3072, [GELU](#info:gelu), [fc_out](#info:mlp_fc_out_notes) 3072 → 768, [dropout](#info:mlp_dropout_notes).
3. **\`x + ...\`, add it back.** The update is added number by number to the residual stream.

Shapes, which must match for the add:

\`\`\`text
x                 (B, T, C)
self.ln2(x)       (B, T, C)
self.mlp(...)     (B, T, C)
x + ...           (B, T, C)
\`\`\`

## Forward: a worked example

One token, first 4 dimensions, continuing the example from the [MLP dropout note](#info:mlp_dropout_notes) (training, dimension 2 dropped):

\`\`\`text
x (residual stream):  [0.20,  -0.50,  1.00, 0.30]
mlp(ln2(x)):          [0.556, -0.222, 0,    0.111]   ← the update, after dropout
new x:                [0.756, -0.722, 1.00, 0.411]
\`\`\`

The original information in \`x\` is kept; the MLP only adds a correction on top of it.

## Backward: at the "+", the gradient is copied to both sides

Write the line as \`new_x = old_x + branch\`, where \`branch = mlp(ln2(old_x))\`.

- If \`old_x\` increases by 0.001, \`new_x\` increases by 0.001.
- If \`branch\` increases by 0.001, \`new_x\` also increases by 0.001.

So the local derivative of "+" is **1 for both sides**, and the gradient arriving at \`new_x\` is passed to both sides unchanged. Suppose the gradient arriving from later layers is \`g = [0.3, −0.1, 0.2, 0.05]\`:

\`\`\`text
      ┌──→ straight to old_x:  [0.3, -0.1, 0.2, 0.05]   ← unchanged, × 1
g  ───┤
      └──→ into the branch:    [0.3, -0.1, 0.2, 0.05]   ← also unchanged; then back through
                                                           dropout, fc_out, GELU, fc_in, ln2
\`\`\`

The copy that enters the branch passes back through the MLP and ln2 (as in the [worked backward example](#hover:gelu_backprop_notes)) and finally also arrives at \`old_x\`; call it \`g_branch\`. So:

\`\`\`text
gradient of old_x = g  (straight path)  +  g_branch  (path through the MLP)
\`\`\`

This is the "sum over paths" rule: \`old_x\` affects the loss along two paths (directly into \`new_x\`, and through ln2 and the MLP), so the two gradients are added.

## Why this matters: a gradient highway

GPT-2 has 12 blocks, so 24 lines of the form \`x = x + branch\`.

- **With residuals:** along the straight paths of all 24 "+", the gradient from the loss reaches the embeddings **unchanged**. Each branch adds its own contribution on top, but never blocks the straight path.
- **Without residuals** (\`x = mlp(ln2(x))\`): the gradient would have to pass through all 24 sublayers one after another, multiplied by each one's local derivative. Suppose each sublayer happened to halve it on average: after 24 sublayers it would be 0.5²⁴ ≈ 0.00000006, and the first layers would learn almost nothing. This is the **vanishing gradient** problem. (If each one doubled it instead, it would explode.)

A related benefit shows up at the start of training: the weights are small, so the update \`mlp(ln2(x))\` is close to 0 and \`new_x ≈ old_x\`. Every block starts out close to "do nothing", and the deep stack behaves like a shallow one until the blocks learn useful updates. That makes deep networks much easier to start training.

For the same argument in Jacobian form, and why putting LayerNorm **before** the sublayer (pre-LN, as here) keeps the straight path clean while the original Transformer's post-LN does not, see [Prevents vanishing/exploding](#info:mlp_prevents_vanishing_exploding_notes).

## Summary

| | Forward | Backward |
| --- | --- | --- |
| Straight path (\`x\`) | the original information passes through complete | the gradient passes back unchanged |
| Branch (\`mlp(ln2(x))\`) | adds a per-token correction | adds its own gradient contribution |

The residual connection is what makes it possible to stack many blocks: forward, nothing is lost; backward, the gradient reaches every layer.
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

   Guo et al., On Calibration of Modern Neural Networks (ICML 2017). Shows that modern networks tend to be overconfident (their confidence is higher than their accuracy) and proposes temperature scaling as a fix. Useful background for why reducing overconfidence, as label smoothing does, matters.

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

(take the log of both sides of the softmax definition). Plug it into the loss:

$$
L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i \\left(z_i - \\log \\sum_{k=1}^C e^{z_k}\\right) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
$$

Because $y$ is a probability distribution, $\\sum_{i=1}^C y_i = 1$, so

$$
L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
$$

Differentiate w.r.t. $z_j$ (in the first sum only the $i = j$ term contains $z_j$; for the second term use the rule that the derivative of $\\log u$ is $u'/u$):

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
    title: "Optimizers: from SGD to AdamW",
    md: `
**An optimizer turns gradients into parameter updates.**

[\`loss.backward()\`](#hover:backward_notes) computes every parameter's gradient; \`optimizer.step()\` then decides how far, and in which direction, to move each parameter. Optimizers differ only in that rule. They are used in every kind of training, pretraining and finetuning alike.

This note follows the history: each optimizer fixes a problem of the one before. Click a name for the details and a worked example. The last section explains the AdamW settings in this site's code.

## Two problems with plain SGD

The simplest rule, [SGD](#info:sgd_notes), moves every parameter against its gradient $g$ by a fixed learning rate $\\eta$ (\`lr\` in PyTorch): $w \\leftarrow w - \\eta \\cdot g$. It has two problems, and the optimizers below were invented to fix them:

1. **Gradients are noisy.** Each step estimates the gradient from one small batch, so part of every step is noise and the path zigzags. Fix: **momentum**, an average of recent gradients, so the noise cancels and the consistent direction remains.
2. **One learning rate for all parameters.** Parameters can have gradients of very different sizes. With $\\eta = 0.01$, a parameter whose gradient is about 100 moves 1 per step (it may oscillate), while one whose gradient is about 0.01 moves 0.0001 (it barely learns). Fix: a **per-parameter step size** that shrinks for parameters with consistently large gradients and grows for those with small ones.

\`\`\`text
SGD
 ├─ problem 1 (noise)            → Momentum
 └─ problem 2 (one η for all)    → Adagrad → RMSProp
                                               ↓
                     Momentum + RMSProp  =  Adam
                                               ↓
             decoupled weight decay  =  AdamW   ← used in this code
\`\`\`

## Timeline

The eras below say what language models **used** at the time; several of these methods are much older.

### Before 2014: the building blocks

- **[SGD](#info:sgd_notes)**: the basic rule. Stochastic approximation goes back to Robbins & Monro (1951).
- **[SGD with momentum](#info:sgd_momentum_notes)**: adds a velocity to smooth the updates. Proposed by Polyak (1964); shown to matter for deep networks by Sutskever et al. (2013).
- **[Adagrad](#info:adagrad_notes)** (2011): the first popular per-parameter learning rate. It helped with sparse, high-variance gradients such as word embeddings, where a rare word's row gets a gradient only now and then.
- **[RMSProp](#info:rmsprop_notes)** (2012): fixes Adagrad's step size shrinking toward zero over time; widely used in early RNN language models.

### 2014–2017: Adam becomes the default

- **[Adam](#info:adam_notes)** (2014) becomes the default in NLP (RNN language models, seq2seq with attention). It combines momentum with RMSProp's per-parameter scaling, so it needs much less learning-rate tuning than SGD.

### 2017–2019: Transformers

- **Adam + warmup becomes standard:** a short linear warmup, then a decay (inverse-sqrt in the original Transformer, later usually cosine).
- Post-norm gives way to pre-norm; together with Adam's adaptivity and warmup, this makes deep stacks train stably.
- **Decoupled weight decay.** The AdamW paper (Loshchilov & Hutter, 2017) shows that weight decay should be applied directly to the weights, not added to the gradient as L2 regularization. BERT (2018) already used Adam with this kind of decoupled decay, and excluded LayerNorm and bias from it; this recipe spread with BERT.

### 2019–2021: scaling up (GPT-2/3, T5)

- **[AdamW](#info:adamw_notes)** becomes the default for large language models, used with warmup + cosine decay, global-norm [gradient clipping](#hover:clip_grad_norm_notes), and no decay on norm/bias weights.
- The GPT-2 paper gives no optimizer details. The GPT-3 paper does: Adam with $\\beta_1 = 0.9$, $\\beta_2 = 0.95$, gradient clipping at 1.0, weight decay 0.1, and warmup followed by cosine decay. This site's code uses the same values.
- **Adafactor** (memory-efficient and Adam-like, with factored second moments) is used for T5/mT5 when optimizer memory is tight.

### 2022–2023: AdamW holds

- Pretraining stays on AdamW (or Adafactor where memory is the limit).
- New optimizers such as Lion and Sophia show gains in some setups, but none broadly replaces AdamW.
- Fused AdamW kernels become standard for speed: PyTorch's built-in \`AdamW(..., fused=True)\` (since PyTorch 2.0), Apex, DeepSpeed.

### 2024–today

- AdamW is still the most common default for LLM pretraining and full-parameter finetuning; Adafactor remains an option when memory must be minimized.
- 🆕 **Muon** (Keller Jordan, late 2024, first used in the NanoGPT speedrun) updates each 2D weight matrix with an orthogonalized version of its momentum, so no direction in the matrix dominates the update. Moonshot scaled it up and trained Kimi K2 (2025, 1T total parameters) with **MuonClip**: Muon plus a "QK-clip" that rescales the query/key weights whenever attention logits grow too large. Muon is used only for the hidden weight matrices; embeddings, the output layer and norm weights still use AdamW. It is the best-known case so far of an optimizer replacing AdamW for pretraining at this scale.

## The AdamW in this code

\`\`\`python
# weight decay only on 2D weights (Linear, Embedding), not on biases or LayerNorm
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
\`\`\`

### Two parameter groups

- **\`decay\`**: parameters with 2 or more dimensions: every \`nn.Linear\` weight and the embedding tables. These hold almost all of the model's parameters, and weight decay keeps them from growing too large.
- **\`no_decay\`**: 1D parameters: every bias, and LayerNorm's $\\gamma$ and $\\beta$. They are a tiny share of the parameters, so decaying them does little against overfitting. Decaying $\\gamma$ would even work against LayerNorm: it pulls $\\gamma$ toward 0, which shrinks the normalized output that LayerNorm is meant to rescale.
- The tied weight (\`tok_emb.weight\` = \`lm_head.weight\`, see [weight tying](#hover:weight_tying_notes)) appears only once, because \`model.parameters()\` skips duplicates. It lands in \`decay\`.

This is the same split nanoGPT uses: \`p.dim() >= 2\` decides.

### The hyperparameters

- **\`lr=3e-4\`**: a common safe learning rate for Adam on small models. (The GPT-3 paper used 6e-4 for its 125M model, the size of GPT-2 small.)
- **\`betas=(0.9, 0.95)\`**: the averaging rates of Adam's two running averages. A rate $\\beta$ averages over roughly the last $\\frac{1}{1 - \\beta}$ steps:
  - $\\beta_1 = 0.9$: the momentum (average gradient) covers about the last 10 steps.
  - $\\beta_2 = 0.95$: the average squared gradient covers about the last 20 steps. PyTorch's default is 0.999, about 1000 steps. The shorter window reacts faster when gradient sizes change, which makes large-model training more stable; GPT-3 used 0.95.
- **\`weight_decay=0.1\`** (only in the \`decay\` group): each step shrinks the weights a little toward 0, in proportion to the learning rate. The value comes from GPT-3. See [AdamW](#info:adamw_notes) for how decoupled decay works.

### What this code leaves out

The learning rate stays at 3e-4 for the whole run. Real GPT-2/GPT-3 training uses a **schedule**: a linear warmup from near 0 over the first steps, then a cosine decay down to a small value. Warmup avoids large updates while Adam's running averages are still unreliable; the decay lets the model settle into a minimum at the end.

## References

1. **Decoupled Weight Decay Regularization** (AdamW). Loshchilov & Hutter, 2017. [arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
2. **Language Models are Few-Shot Learners** (GPT-3), Appendix B: training details. Brown et al., 2020. [arXiv:2005.14165 ↗](https://arxiv.org/abs/2005.14165)
3. **Muon: An optimizer for hidden layers in neural networks**. Keller Jordan, 2024. [kellerjordan.github.io ↗](https://kellerjordan.github.io/posts/muon/)
4. **Kimi K2: Open Agentic Intelligence** (MuonClip). Moonshot AI, 2025. [arXiv:2507.20534 ↗](https://arxiv.org/abs/2507.20534)
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

## Two problems with plain SGD

SGD works, but it has two problems, and the optimizers that followed were invented to fix them:

1. **Gradients are noisy.** In real training each step estimates the gradient from one small batch (in this site's code, 4 × 32 = 128 tokens), so part of every step is noise, and the path zigzags instead of heading straight downhill. Fix: [momentum](#info:sgd_momentum_notes), which averages recent gradients so the noise cancels.
2. **One learning rate for all parameters.** Parameters can have gradients of very different sizes. Take two parameters whose gradients are about 100 and about 0.01, with $\\eta = 0.01$:

   \`\`\`text
   parameter A:  moves 0.01 × 100  = 1       per step   ← may be too large and oscillate
   parameter B:  moves 0.01 × 0.01 = 0.0001  per step   ← too small, barely learns
   \`\`\`

   Raise $\\eta$ and A diverges; lower it and B stops learning. No single learning rate suits both. Fix: give each parameter its own step size, as [Adagrad](#info:adagrad_notes) does.

> The example above has a single data point, so its gradient has no noise, and $w$ and $b$ have gradients of similar size. That is why plain SGD does so well there.
`,
  },

  sgd_momentum_notes: {
    title: "SGD with momentum",
    md: `
**Instead of stepping with the current gradient alone, keep a velocity: a running average of past gradients, and step with that.**

## What it is

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

- $g = \\nabla_{\\theta} L$ is the current gradient.
- $\\beta$ (usually 0.9) controls how much "memory" to keep.
- $\\eta$ is the learning rate.

Each step keeps 90% of the old velocity and adds 10% of the new gradient. Unrolled, $v$ is a weighted average of all past gradients: the newest has weight 0.1, the one before $0.1 \\times 0.9$, then $0.1 \\times 0.81$, and so on, fading by a factor 0.9 per step. This is an **exponential moving average** (EMA). With $\\beta = 0.9$ it covers roughly the last $\\frac{1}{1 - \\beta} = 10$ steps.

The name comes from physics: $v$ behaves like a ball with inertia. Pushed the same way step after step, it keeps rolling that way; pushed back and forth, it barely moves.

## Why it helps: noise cancels

Plain [SGD](#info:sgd_notes) steps with each batch's gradient, and part of every such gradient is noise. Momentum averages the noise away.

Take a parameter with two coordinates. The true downhill direction is $x$, but every batch adds a large noise in $y$ whose sign flips from step to step:

\`\`\`text
odd steps:   g = [1, +3]
even steps:  g = [1, −3]
\`\`\`

**SGD** steps with $g$ itself: each step moves 1 in $x$ and ±3 in $y$. The noise (3) is larger than the signal (1), so the path zigzags across far more than it moves forward.

**Momentum** ($\\beta = 0.9$) steps with $v$:

\`\`\`text
step  1:  v = [0.100,  0.300]
step  2:  v = [0.190, -0.030]
step  3:  v = [0.271,  0.273]
step  4:  v = [0.344, -0.054]
step 10:  v = [0.651, -0.103]
step 20:  v = [0.878, -0.139]
step 40:  v = [0.985, -0.156]
\`\`\`

- **$x$ (signal):** every gradient is 1, so their average is 1. $v_x$ climbs from 0 toward 1.
- **$y$ (noise):** +3 and −3 alternate, so they cancel in the average. $v_y$ ends up swinging only about ±0.16.

Where ±0.16 comes from: once things settle, $v_y$ alternates between $+a$ and $-a$. An odd step turns $-a$ into $+a$: $a = 0.9 \\cdot (-a) + 0.1 \\cdot 3$, so $1.9a = 0.3$ and $a = \\frac{0.3}{1.9} \\approx 0.16$.

The noise shrinks from 3 to about 0.16, roughly 1/19 of its size, while the signal is kept. The path becomes nearly straight. The same thing happens in a narrow valley (an "ill-conditioned" loss), where the gradient keeps flipping across the valley while pointing consistently along it.

## The cost: a slow start

SGD has no velocity: from step 1 it moves the full gradient. Momentum's $v$ starts at 0 and has to build up. Compare the $x$ direction above:

\`\`\`text
          SGD steps with    momentum steps with v
step  1:       1                 0.100
step  2:       1                 0.190
step  3:       1                 0.271
step 10:       1                 0.651
step 20:       1                 0.878
step 40:       1                 0.985   → approaches 1
\`\`\`

Each value is 0.9 × the previous one plus 0.1: $0.1$, then $0.1 \\cdot 0.9 + 0.1 = 0.19$, then $0.19 \\cdot 0.9 + 0.1 = 0.271$, and so on.

- **Early on**, momentum is slower than SGD: it takes about 10+ steps to warm up.
- **Once warmed up**, both move at the same speed along the signal: $\\eta \\times 1$ per step. In the EMA form, $v$ settles at the average gradient and never exceeds it, because each step adds only $(1 - \\beta) = 10\\%$ of the new gradient.

So momentum is not faster along the signal; its gain is in the noise direction (±3 per step for SGD vs about ±0.16). [Adam](#info:adam_notes) removes the slow start with a "bias correction".

## Full worked example: one data point, no noise

This example has a single data point, so the gradient has no noise to cancel. It shows only the cost (the slow start), step by step, so it is a useful contrast to the example above.

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

> You can see the loss decreasing at each step (4.5 → 4.0613 → 3.3089). That's the optimizer using backprop's gradients to move $w, b$ toward better values. Compare with SGD, where the loss decreases faster (4.5 → 1.125 → 0.28125) in the first 3 steps. The reason is the slow start explained above: in steps 1–3 the velocity is only 0.1×, 0.19× and 0.271× of a full gradient.

## Why momentum loses on this example

There is no noise here, so momentum pays the cost without getting the benefit. It is also hurt later: as $w$ and $b$ approach the target, the gradient shrinks, but $v$ still remembers the larger gradients of earlier steps. So momentum overshoots the target and oscillates around it. After 10 steps momentum's loss is about 0.44, versus about $4 \\times 10^{-6}$ for SGD.

Momentum pays off on harder problems: noisy gradients (as in the example above) and narrow valleys. Training a language model on small batches is exactly that kind of problem.

## Making momentum look better earlier

If we want momentum to look better earlier, we can:

1. Lower $\\beta$ (e.g., 0.8), so the velocity builds up faster.
2. Slightly raise $\\eta$ (e.g., 0.12) to make up for the smaller steps early on.

## PyTorch's form is different

This note uses the EMA form $v \\leftarrow \\beta v + (1 - \\beta) g$, which is also the form [Adam](#info:adam_notes) uses for its first moment. PyTorch's \`torch.optim.SGD(momentum=0.9)\` uses a different form, without the $(1 - \\beta)$:

$$
v \\leftarrow \\beta v + g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

With a constant gradient of 1, this $v$ settles at $\\frac{1}{1 - \\beta} = 10$ instead of 1. The two forms differ by exactly the factor $(1 - \\beta)$ at every step, so PyTorch's form with learning rate $\\eta$ takes the same steps as the EMA form with learning rate $\\frac{\\eta}{1 - \\beta} = 10\\eta$. Keep this in mind when copying a learning rate from one form to the other.
`,
  },

  adagrad_notes: {
    title: "Adagrad",
    md: `
**Adagrad (Adaptive gradient)** gives each parameter its own learning rate, set automatically from that parameter's past gradients.

It fixes the second of the [two problems of SGD](#info:sgd_notes) (see the end of the SGD note): one learning rate for all parameters.

## Update rule

Adagrad keeps a running **sum of squared gradients** for each parameter $i$:

$$
s_i \\leftarrow s_i + g_i^2
$$

where $g_i$ is the gradient of parameter $i$ (a single number) and $s_i$ starts at 0. It then updates the parameter with the learning rate divided by $\\sqrt{s_i}$:

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

$\\epsilon$ is a tiny constant (e.g. $10^{-8}$) that only prevents division by zero.

Two quantities are worth keeping apart:

- **Effective learning rate** $\\eta_{\\text{eff},\\,i} = \\frac{\\eta}{\\sqrt{s_i} + \\epsilon}$: parameter $i$'s own learning rate. It depends on how much squared gradient the parameter has **accumulated so far**.
- **Actual step** $\\eta_{\\text{eff},\\,i} \\cdot g_i$: how far the parameter really moves. It is the effective learning rate times the current gradient.

### Vector form and the $\\odot$ symbol

Written for all parameters at once, as vectors, the accumulator is $s \\leftarrow s + g \\odot g$. The symbol $\\odot$ means **element-wise multiplication**: multiply the numbers in matching positions; the result is a vector of the same shape. For example, $(-3, 2) \\odot (-3, 2) = (9, 4)$. (Unlike the dot product $(-3, 2) \\cdot (-3, 2) = 9 + 4 = 13$, nothing is added up.) So $g \\odot g$ squares each parameter's gradient separately: each parameter accumulates only its own squared gradients, which is why each one gets its own learning rate. In PyTorch this is \`g * g\`.

## Effect 1: the step size does not depend on the gradient's scale

Take two parameters: A's gradient is always 100, B's is always 0.01, and $\\eta = 0.01$.

**SGD** moves each by $\\eta \\cdot g$:

\`\`\`text
A: 0.01 × 100  = 1      per step
B: 0.01 × 0.01 = 0.0001 per step      ← 10,000× apart
\`\`\`

**Adagrad, step 1:**

\`\`\`text
A: s = 100²  = 10000,   √s = 100   →  moves 0.01 × 100  / 100  = 0.01
B: s = 0.01² = 0.0001,  √s = 0.01  →  moves 0.01 × 0.01 / 0.01 = 0.01
\`\`\`

Both take the **same** step, 0.01. A's effective learning rate is much smaller than B's, but multiplied by its much larger gradient, the actual step comes out the same.

The reason: the numerator is $g$, and the denominator $\\sqrt{s}$ is proportional to the size of $g$. Scale the gradient by 10,000 and both scale together, so their ratio does not change. Each step moves about $\\eta$, in the direction given by the gradient's sign; how large the gradient is no longer matters.

At step $t$, with a constant gradient $g$:

$$
s = t\\, g^2, \\qquad \\sqrt{s} = \\sqrt{t}\\, |g|, \\qquad \\text{step} = \\frac{\\eta\\, g}{\\sqrt{t}\\, |g|} = \\pm\\frac{\\eta}{\\sqrt{t}}
$$

The same for A and B.

## Effect 2: a parameter that rarely gets a gradient takes larger steps

What matters is not only how **large** a parameter's gradients are, but how **often** it gets one. $s$ records the total accumulated so far, so a parameter that rarely gets a gradient keeps a small $s$.

Example: a common word's embedding row gets a gradient of 1 every step; a rare word's row gets a gradient of 1 only once every 100 steps. At step 10,000:

\`\`\`text
common word:  10,000 gradients so far,  s = 10,000,  √s = 100
              step now:  η × 1 / 100 = 0.01 η

rare word:    100 gradients so far,     s = 100,     √s = 10
              step when it appears:  η × 1 / 10 = 0.1 η
\`\`\`

The gradients are the same size (1), yet when the rare word appears it moves **10 times** as far as the common word. A rare word gets few chances to learn, so each chance counts for more; a common word, updated almost every step, takes small steps instead of being pushed around again and again. This is why Adagrad became popular for word vectors.

The same situation exists in GPT-2: the [token embedding](#info:embedding_tok) row of a rare token receives an input-side gradient only when that token is in the batch.

## Why the square root?

**It is what makes Effect 1 work.** The step is $\\frac{g}{\\sqrt{s}}$ times $\\eta$, and $\\sqrt{s}$ grows in proportion to the size of $g$, so the two cancel. Dividing by $s$ instead would give $\\frac{g}{t\\, g^2} = \\frac{1}{t\\, g}$: larger gradients would get *smaller* steps, the opposite of what we want.

**It gives a gentle decay.** When the gradients keep a similar size, $s_i$ grows about linearly with time, $s_i \\approx t\\, g_i^2$, so

$$
\\eta_{\\text{eff},\\,i} \\approx \\frac{\\eta}{|g_i| \\sqrt{t}}
$$

and the actual step shrinks like $\\frac{\\eta}{\\sqrt{t}}$. This decay is steady but not so fast that learning dies early. Dividing by $s_i$ would instead give a $\\frac{1}{t}$ decay, which becomes tiny too quickly.

> The $\\frac{1}{\\sqrt{t}}$ rate also matches classic results in online convex optimization, where it gives good regret/convergence bounds.

## Worked example

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Accumulator:** $s \\leftarrow s + g \\odot g$, starting from $s_0 = (0, 0)$
- **Update:** $\\theta \\leftarrow \\theta - \\Delta$, with $\\Delta = \\eta_{\\text{eff}} \\cdot g$ and $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **Hyperparameters:** $\\eta = 1.0$, $\\epsilon = 10^{-8}$ (negligible here)
- **Init:** $\\theta_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)$
- $\\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3,\\; 1/2)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-1, 1) = (1, -1)$

The two coordinates have different effective learning rates, $1/3$ and $1/2$, yet the gradients were $-3$ and $2$ and **both coordinates moved exactly 1** $= \\eta$. This is Effect 1: on the first step $s = g^2$, so every coordinate moves $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$.

### Step 2

- $\\theta_1 = (1, -1)$
- $g_2 = (1 - 3,\\; -1 + 2) = (-2, 1)$
- $s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)$
- $\\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)$
- $\\theta_2 = \\theta_1 - \\Delta = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)$

Both effective learning rates shrank: $1/3 \\to 0.277$ and $1/2 \\to 0.447$.

### Step 3

- $\\theta_2 = (1.554, -1.447)$
- $g_3 = (1.554 - 3,\\; -1.447 + 2) = (-1.446, 0.553)$
- $s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)$
- $\\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)$
- $\\theta_3 = \\theta_2 - \\Delta = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)$

They shrank again: $0.277 \\to 0.257$ and $0.447 \\to 0.434$.

### What the example shows

- Each coordinate has its own effective learning rate, and it only shrinks as $s$ accumulates.
- $\\theta_1$ has the larger gradients, so it gets the **smaller** effective learning rate ($0.257$ vs $0.434$ by step 3). Its actual steps are not smaller, though: equal in step 1 (both 1), and after that even slightly larger ($0.554$ vs $0.447$, $0.372$ vs $0.24$), because its larger gradient makes up for the smaller learning rate. (Its gradients stay larger because it started farther from its target: 3 away vs 2.) Adagrad evens the coordinates out instead of letting the larger gradient dominate.

## The problem: the steps only shrink

$s$ is a running **sum**; it only grows, never shrinks. So the effective learning rate $\\frac{\\eta}{\\sqrt{s}}$ only gets smaller. With a gradient of constant size:

\`\`\`text
step      1:  moves η / 1
step    100:  moves η / 10
step 10,000:  moves η / 100
\`\`\`

Training a language model takes hundreds of thousands of steps; by then the steps are so small that learning stalls. And if a parameter's gradients were large early on and become small later, $s$ does not go down: the past keeps holding its steps down.

\`\`\`text
first 100 steps, gradient 10:   s = 100 × 10² = 10,000,  √s = 100
step 101, gradient 1:           moves η × 1 / 100 = 0.01 η    ← held down by the past
\`\`\`

## Summary

| Case | Actual step |
|---|---|
| Gradients always large vs always small | About the same size: Adagrad evens them out (Effect 1) |
| Gradients frequent vs rare | Larger for the rare one, when it gets a gradient (Effect 2) |
| Large gradients in the past, small now | Small now: held down by the past (the problem) |

[RMSProp](#info:rmsprop_notes) fixes the problem: instead of summing the whole history, it averages only the recent past.
`,
  },

  rmsprop_notes: {
    title: "RMSProp",
    md: `
**RMSProp** keeps Adagrad's per-parameter learning rate, but averages only the **recent** squared gradients instead of summing all of them.

It fixes [Adagrad](#info:adagrad_notes)'s problem: Adagrad's sum only grows, so its steps only shrink, and large gradients from the past keep holding the steps down. (Proposed by Geoffrey Hinton in a 2012 lecture, never formally published.)

## Update rule

Only the accumulator changes. For each parameter $i$:

$$
\\begin{aligned}
\\text{Adagrad:}\\quad & s_i \\leftarrow s_i + g_i^2 \\\\
\\text{RMSProp:}\\quad & s_i \\leftarrow \\beta\\, s_i + (1 - \\beta)\\, g_i^2
\\end{aligned}
$$

The update is the same as Adagrad's:

$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$

- $g_i$ is the current gradient of parameter $i$; $s_i$ starts at 0.
- $\\beta$ is the decay rate, usually 0.9.
- $\\eta$ is the learning rate; $\\epsilon$ is a tiny constant that prevents division by zero.
- In vector form the accumulator is $s \\leftarrow \\beta s + (1 - \\beta)(g \\odot g)$, where $\\odot$ is element-wise multiplication (explained in the [Adagrad](#info:adagrad_notes) note).

### What $s$ means now

The new accumulator is an **exponential moving average** (EMA), the same kind of average [momentum](#info:sgd_momentum_notes) uses: each step keeps 90% of the old $s$ and adds 10% of the new $g^2$, so older gradients fade by a factor 0.9 per step. With $\\beta = 0.9$, $s$ is roughly the average of the last $\\frac{1}{1 - \\beta} = 10$ squared gradients.

So $s$ changes meaning:

- **Adagrad:** $s$ is "how much in total so far", so it keeps growing.
- **RMSProp:** $s$ is "how large the squared gradients have been recently". If the gradient size stays the same, $s$ stays the same.

$\\sqrt{s}$ is then the **root mean square** (RMS) of the recent gradients; dividing the gradient by it gives the method its name.

## What it fixes

### Effect 1 is kept; the shrinking is gone

With a constant gradient $g$, $s$ settles at $g^2$, so each step moves $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$:

- As in Adagrad, the step size does not depend on how large the gradient is (Adagrad's Effect 1).
- Unlike Adagrad, the step **does not shrink over time**: step 10,000 moves as far as step 100.

### It follows changes in gradient size

Adagrad's "held down by the past" case: the gradient is 10 for the first 100 steps, then 1 from step 101 on. How far does each step move afterwards (in units of $\\eta$)?

\`\`\`text
steps after the switch    Adagrad    RMSProp
         1                 0.010      0.105
        10                 0.010      0.168
        20                 0.010      0.277
        30                 0.010      0.439
        40                 0.010      0.637
        60                 0.010      0.921   → approaches 1
\`\`\`

- **Adagrad:** $s$ still holds the 100 large gradients and never lets them go, so the step stays stuck at $0.01\\eta$.
- **RMSProp:** the old large gradients fade by a factor 0.9 per step. After a few dozen steps $s$ reflects "the gradient is now 1", and the step recovers to nearly $\\eta$.

In LLM training the size of the gradients does change between phases of training, so being able to follow it matters.

## An analogy with momentum

RMSProp relates to Adagrad the way SGD with momentum relates to SGD: both add an EMA, a short memory of recent steps.

- **Adagrad → RMSProp:** replace the forever-growing sum of squared gradients with an EMA of squared gradients.
- **SGD → SGD with momentum:** replace the current gradient with an EMA of gradients.

The differences:

- **What gets averaged:** RMSProp averages **squared** gradients (their size); momentum averages the gradients themselves (their direction).
- **Per-parameter learning rate:** RMSProp, like Adagrad, gives each parameter its own; momentum, like SGD, uses one learning rate for all.
- **Problem fixed:** RMSProp fixes Adagrad's steps shrinking toward zero; momentum fixes SGD's noisy, zigzag path.

[Adam](#info:adam_notes) combines the two: an EMA of the gradients (momentum) divided by the root of an EMA of the squared gradients (RMSProp).

## Worked example

The same problem as in the [Adagrad](#info:adagrad_notes) note, so the two can be compared.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Accumulator:** $s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g)$, starting from $s_0 = (0, 0)$
- **Update:** $\\theta \\leftarrow \\theta - \\Delta$, with $\\Delta = \\eta_{\\text{eff}} \\cdot g$ and $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s} + \\epsilon}$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta = 0.9$, $\\epsilon = 10^{-8}$ (negligible here)
- **Init:** $\\theta_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.9, 0.4)$
- $\\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)$
- $\\theta_1 = \\theta_0 - \\Delta = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)$

Both coordinates moved $0.3162$, the same distance even though the gradients were $-3$ and $2$ (Effect 1). But $0.3162$ is $3.16\\eta$, more than three times the "normal" step $\\eta = 0.1$. This is the start-up problem described below.

### Step 2

- $\\theta_1 = (0.3162, -0.3162)$
- $g_2 = (0.3162 - 3,\\; -0.3162 + 2) = (-2.6838, 1.6838)$
- $s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot \\big((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)\\big) = (1.5303, 0.6435)$
- $\\sqrt{s_2} = (1.2370, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1247)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1247) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)$
- $\\theta_2 = \\theta_1 - \\Delta = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)$

### Step 3

- $\\theta_2 = (0.5332, -0.5261)$
- $g_3 = (0.5332 - 3,\\; -0.5261 + 2) = (-2.4668, 1.4739)$
- $s_3 = 0.9 \\cdot (1.5303, 0.6435) + (1 - 0.9) \\cdot \\big((-2.4668, 1.4739) \\odot (-2.4668, 1.4739)\\big) = (1.9858, 0.7964)$
- $\\sqrt{s_3} = (1.4092, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121)$
- $\\Delta = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4739) = (-0.1751, 0.1652)$
- $\\theta_3 = \\theta_2 - \\Delta = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)$

### What the example shows

- The steps shrink over steps 1–3 ($0.316 \\to 0.217 \\to 0.175$ for $\\theta_1$), but not because $s$ only grows, as in Adagrad. $s$ is still warming up from 0 (see below), and the gradients themselves get smaller as $\\theta$ approaches the minimum.
- As in Adagrad, $\\theta_1$ (larger gradients) gets the smaller effective learning rate, but its actual steps are not smaller.

## The new problem: steps too large at the start

$s$ starts at 0, and the first step adds only $(1 - \\beta) = 10\\%$ of $g^2$:

\`\`\`text
s = 0.1 × g²,   √s = 0.316 × |g|
step 1 moves:  η × g / (0.316 |g|) = ±3.16 η      ← more than 3× the normal step
\`\`\`

This is the same cause as momentum's slow start: an EMA that starts at 0 is too small for the first several steps. In momentum the too-small average is the step itself, so the steps start **too small**; here it sits in the **denominator**, so the steps start **too large**. It takes roughly 10+ steps for $s$ to warm up. [Adam](#info:adam_notes) fixes both with its "bias correction".

## In PyTorch

\`torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8)\`: the decay rate $\\beta$ is called \`alpha\`, and its default is **0.99** (an average over roughly the last 100 steps), not the 0.9 used in this note and in Hinton's lecture.
`,
  },

  adam_notes: {
    title: "Adam",
    md: `
**Adam** combines the two ideas before it: [momentum](#info:sgd_momentum_notes) (an average of the gradients) and [RMSProp](#info:rmsprop_notes) (a per-parameter step size from an average of the squared gradients). It also fixes the start-up problem both of them have.

## Update rule

For each parameter, Adam keeps two EMAs (exponential moving averages), both starting at 0:

$$
\\begin{aligned}
m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t && \\text{(EMA of gradients: momentum)} \\\\
v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2)\\, g_t^2 && \\text{(EMA of squared gradients: RMSProp)} \\\\
\\hat{m}_t &= \\frac{m_t}{1 - \\beta_1^t} && \\text{(bias correction)} \\\\
\\hat{v}_t &= \\frac{v_t}{1 - \\beta_2^t} && \\text{(bias correction)} \\\\
\\theta_{t} &= \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t && \\text{(update)}
\\end{aligned}
$$

- $g_t$ is the gradient at step $t$; for vectors of parameters, $g_t^2$ means $g_t \\odot g_t$ (element-wise, see [Adagrad](#info:adagrad_notes)).
- $m_t$ tracks the **direction** of recent gradients; $v_t$ tracks their **size**.
- $\\beta_1$ (usually 0.9) averages $m$ over roughly the last $\\frac{1}{1 - \\beta_1} = 10$ steps; $\\beta_2$ (usually 0.999) averages $v$ over roughly the last 1000 steps.
- $\\hat{m}_t$ and $\\hat{v}_t$ are the bias-corrected versions, explained below.
- $\\eta$ is the learning rate; $\\epsilon$ is a tiny constant that prevents division by zero.

Leaving out the bias correction for a moment, the update is simply $\\eta \\cdot \\frac{m}{\\sqrt{v}}$: momentum's average gradient, divided by RMSProp's RMS of the gradient.

## What combining them adds: noisy gradients get smaller steps

Look at the ratio $\\frac{m}{\\sqrt{v}}$: the numerator is the **average** gradient, the denominator is the **root mean square** of the gradient.

**Consistent gradients**, e.g. always 1:

\`\`\`text
m ≈ 1,  √v ≈ 1   →  m/√v ≈ 1,   each step moves η
\`\`\`

**Noisy gradients**, e.g. 4, −2, 4, −2, … (the average is still 1, but with large noise):

\`\`\`text
m ≈ average            = 1
v ≈ average of squares = (16 + 4) / 2 = 10,   √v ≈ 3.16
m/√v ≈ 1 / 3.16 = 0.32,   each step moves only 0.32 η
\`\`\`

When the signs flip back and forth, they cancel in $m$ (the momentum part), but the squares in $v$ do not cancel. So the less consistent the gradients, the smaller $\\frac{m}{\\sqrt{v}}$, and the more cautiously Adam steps; the more consistent, the closer it gets to the full step $\\eta$.

Think of $\\frac{m}{\\sqrt{v}}$ as a signal-to-noise ratio: how trustworthy the direction is. Its size is usually at most about 1, so each Adam step is **at most about $\\eta$**, however large the gradient itself is. This keeps the scale independence from Adagrad and RMSProp.

## Bias correction

### The problem: both averages start at 0

$m$ and $v$ start at 0, so for the first several steps both are too small. This is the same start-up problem as before:

- In [momentum](#info:sgd_momentum_notes), the too-small average is the step itself, so the steps start **too small**.
- In [RMSProp](#info:rmsprop_notes), the too-small average is in the denominator, so the steps start **too large**.

### How large the bias is

Suppose the gradient is always $g$. Unroll $m$:

\`\`\`text
m1 = 0.1 g
m2 = 0.9 × 0.1 g + 0.1 g = 0.19 g
m3 = 0.9 × 0.19 g + 0.1 g = 0.271 g
...
m_t = (1 − 0.9^t) × g
\`\`\`

In general $m_t = (1 - \\beta_1^t)\\, g$, and in the same way $v_t = (1 - \\beta_2^t)\\, g^2$. The average is too small by exactly the factor $(1 - \\beta^t)$.

### The fix: divide it back out

$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
$$

\`\`\`text
step 1:  m̂ = 0.1 g / 0.1       = g      ✓
         v̂ = 0.001 g² / 0.001  = g²     ✓
\`\`\`

As $t$ grows, $\\beta^t$ goes to 0 and the divisor goes to 1, so the correction fades away by itself. With the correction, the very first step is exactly $\\eta \\cdot \\frac{g}{|g|} = \\pm\\eta$.

> The paper's Section 3 derives this for gradients that change over time: [Adam: A Method for Stochastic Optimization ↗](https://arxiv.org/pdf/1412.6980).

### Without the correction

With a constant gradient, the uncorrected step $\\eta \\cdot \\frac{m_t}{\\sqrt{v_t}}$ is too large by the factor

$$
F_t = \\frac{1 - \\beta_1^t}{\\sqrt{1 - \\beta_2^t}}
$$

For $\\beta_1 = 0.9$, $\\beta_2 = 0.999$:

\`\`\`text
step     1:  F = 3.16       (0.1 / √0.001)
step     2:  F = 4.25
step     3:  F = 4.95
step     4:  F = 5.44
step    12:  F = 6.57       ← the peak
step   100:  F = 3.24
step  1000:  F = 1.26
step  3000:  F = 1.03
\`\`\`

$m$ (10-step average) warms up quickly, but $v$ (1000-step average) takes thousands of steps, and until then the steps stay too large, by up to 6.6×. At the start of training the parameters are random and the gradients are large; steps several times too large then can easily make training diverge.

## Worked example

The same problem as in the [Adagrad](#info:adagrad_notes) and [RMSProp](#info:rmsprop_notes) notes.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update:** $\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (negligible here)
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

After correction, $\\hat{m}_1$ and $\\hat{v}_1$ are exactly $g_1$ and $g_1^2$, so both coordinates move exactly $\\eta = 0.1$. Without the correction they would move $3.16 \\times 0.1 = 0.316$, the same too-large first step as in the RMSProp example.

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

- **Every step moves about $\\eta = 0.1$** in both coordinates: $\\theta_1$ goes $0 \\to 0.1 \\to 0.2 \\to 0.3$, and $\\theta_2$ goes $0 \\to -0.1 \\to -0.2 \\to -0.3$. The gradients are $-3$ and $2$, yet the steps are the same: the step size does not depend on how large the gradient is.
- **The gradients here are perfectly consistent** (no noise), so $\\frac{\\hat{m}}{\\sqrt{\\hat{v}}} \\approx \\pm 1$ and Adam takes full steps. With noisy gradients the steps would be smaller.
- **Bias correction** makes step 1 exactly $\\eta$ instead of $3.16\\eta$.

## In PyTorch

\`torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)\`: \`betas\` is $(\\beta_1, \\beta_2)$. Bias correction is always applied.

This site's GPT-2 code uses [AdamW](#info:adamw_notes) (Adam with decoupled weight decay) and sets $\\beta_2 = 0.95$ instead of 0.999, following GPT-3: a shorter average for $v$ reacts faster when gradient sizes change. See the [optimizer overview](#info:optim_adamw_notes).

## References

1. **Adam: A Method for Stochastic Optimization**

   Kingma & Ba, 2014. The original paper.

   [arXiv:1412.6980 ↗](https://arxiv.org/pdf/1412.6980)

2. **StackExchange discussion**

   Why bias correction matters in Adam.

   [StackExchange Discussion ↗](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
`,
  },

  adamw_notes: {
    title: "AdamW",
    md: `
**AdamW** is [Adam](#info:adam_notes) plus **weight decay**, applied directly to the weights instead of being added to the gradient. The "W" stands for this decoupled **W**eight decay. It is the optimizer this site's GPT-2 code uses.

## What weight decay is

Before each update, shrink every weight a little:

$$
\\theta \\leftarrow (1 - \\eta \\lambda)\\, \\theta
$$

- $\\lambda$ is the weight decay strength (0.1 in this site's code); $\\eta$ is the learning rate.
- With $\\eta = 0.1$ and $\\lambda = 0.1$, $1 - \\eta\\lambda = 0.99$: every step first multiplies each weight by 0.99.

**Why do this?** A weight that the loss does not need gets no consistent push from the gradient, so the decay slowly pulls it toward 0. A weight that does help lower the loss keeps getting pushed back by its gradient and stays large. The model is therefore discouraged from relying on a few very large weights to memorize the training data. This is a form of **regularization**.

## In SGD, L2 regularization is the same thing

Another common way to keep weights small is **L2 regularization**: add a penalty to the loss,

$$
L_{\\text{total}} = L + \\frac{\\lambda}{2}\\, \\theta^2
$$

The penalty's gradient is $\\lambda \\theta$, so the SGD update becomes

$$
\\theta \\leftarrow \\theta - \\eta\\, (g + \\lambda \\theta) = (1 - \\eta \\lambda)\\, \\theta - \\eta\\, g
$$

That is "shrink by $(1 - \\eta\\lambda)$, then take the normal step": exactly weight decay. In SGD the two are the same, which is why the names were long used interchangeably.

## In Adam, L2 goes wrong

With Adam the two are **no longer the same**. The L2 way (what \`torch.optim.Adam(weight_decay=...)\` does) adds $\\lambda\\theta$ to the gradient and hands the result to Adam:

\`\`\`text
g' = g + λθ
m and v are computed from g', and the step is η · m̂ / √v̂
\`\`\`

Adam cannot tell which part of $g'$ comes from the loss and which part is decay. It divides **all** of it by $\\sqrt{\\hat{v}}$, the root mean square of the past values of $g'$ (over roughly the last $\\frac{1}{1 - \\beta_2} = 1000$ steps, or all steps so far if there have been fewer; see [Adam](#info:adam_notes)). So the decay part is scaled too:

- **Parameters with large gradients:** $\\sqrt{\\hat{v}}$ is large, the decay is divided down to almost nothing, and the weight barely decays.
- **Parameters with small gradients:** $\\sqrt{\\hat{v}}$ is small, the decay is magnified, and the weight decays very fast.

### Example: one $\\lambda$, a 100× difference

Two parameters A and B, both at $\\theta = 1$, with $\\eta = 0.01$ and $\\lambda = 0.1$, so the decay adds $\\lambda\\theta = 0.1$ to each gradient. A's own gradient alternates between $+10$ and $-10$; B's alternates between $+0.01$ and $-0.01$. Both average to 0: the loss does not need either weight, which is exactly when decay should pull them toward 0.

**Step 1: what is $\\sqrt{\\hat{v}}$?** It is the root mean square of the past $g' = g + 0.1$:

\`\`\`text
A:  g' = 10.1, −9.9, 10.1, …    squares 102.01, 98.01   →  mean 100.01,  √v̂ ≈ 10
B:  g' = 0.11, 0.09, 0.11, …    squares 0.0121, 0.0081  →  mean 0.0101,  √v̂ ≈ 0.1
\`\`\`

A's $g'$ is dominated by its own gradient, so the extra 0.1 hardly shows. B's $g'$ is almost entirely the decay term, with its own gradient as a small wobble.

**Step 2: how much does the decay part move $\\theta$?** The step is $\\eta \\cdot \\hat{m} / \\sqrt{\\hat{v}}$, and $\\hat{m}$ averages $g'$, so the decay term contributes $\\eta \\lambda \\theta / \\sqrt{\\hat{v}}$:

\`\`\`text
A:  0.01 × 0.1 / 10   = 0.0001   →  shrinks 0.01% per step
B:  0.01 × 0.1 / 0.1  = 0.01     →  shrinks 1% per step
\`\`\`

**Step 3: compare with what was intended.** $\\lambda = 0.1$ was meant to shrink each weight by $\\eta\\lambda = 0.001$, i.e. 0.1% per step:

| | Intended (AdamW) | Adam + L2 |
|---|---|---|
| A (large gradients) | 0.1% | 0.01%, 10× too weak |
| B (small gradients) | 0.1% | 1%, 10× too strong |

Same $\\lambda$, but a **100×** difference in actual decay, caused only by dividing the decay by $\\sqrt{\\hat{v}}$.

**Over 200 steps** (run in PyTorch):

\`\`\`text
                 A (gradient ±10)   B (gradient ±0.01)
no decay              0.982              0.982
Adam + L2             0.963              0.018     ← A barely decays, B is nearly 0
AdamW                 0.804              0.804     ← both decay the same
\`\`\`

- **A:** 0.0001 less per step, about 0.02 over 200 steps: 0.982 → 0.963.
- **B:** about 0.01 less per step, so it reaches nearly 0 within about 100 steps. (As $\\theta$ shrinks, the decay term and $\\sqrt{\\hat{v}}$ shrink with it, so it does not go exactly to 0.)
- **AdamW:** each step multiplies by $0.999$, and $0.999^{200} \\approx 0.819$, so $0.982 \\times 0.819 \\approx 0.804$ for both.

## Update rule

AdamW computes $m_t$, $\\hat{m}_t$, $v_t$, $\\hat{v}_t$ exactly as [Adam](#info:adam_notes) does, from the plain gradient $g_t$ (no $\\lambda\\theta$ added). Only the last line changes:

$$
\\begin{aligned}
\\text{Adam:}\\quad & \\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t \\\\
\\text{AdamW:}\\quad & \\theta_t = (1 - \\eta \\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t
\\end{aligned}
$$

The decay $(1 - \\eta\\lambda)$ multiplies the weight directly and never passes through $\\sqrt{\\hat{v}}$, so every parameter shrinks by the same fraction, whatever its gradients look like.

## Worked example

The same problem as in the [Adam](#info:adam_notes) note, so the two can be compared.

### Setup

- **Parameters:** $\\theta = (\\theta_1, \\theta_2)$
- **Loss:** $L = \\frac{1}{2} \\left[(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2\\right]$, smallest at $\\theta = (3, -2)$
- **Gradients:** $g = (g_1, g_2) = (\\theta_1 - 3,\\; \\theta_2 + 2)$
- **Update:** $\\theta_t = (1 - \\eta\\lambda)\\, \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t$
- **Hyperparameters:** $\\eta = 0.1$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$ (negligible here), $\\lambda = 0.1$, so the decay factor is $1 - \\eta\\lambda = 0.99$
- **Init:** $\\theta_0 = (0, 0)$, $m_0 = (0, 0)$, $v_0 = (0, 0)$

### Step 1

- $\\theta_0 = (0, 0)$
- $g_1 = (-3, 2)$
- $m_1 = 0.9 \\cdot (0, 0) + 0.1 \\cdot (-3, 2) = (-0.3, 0.2)$
- $v_1 = 0.999 \\cdot (0, 0) + 0.001 \\cdot \\big((-3, 2) \\odot (-3, 2)\\big) = (0.009, 0.004)$
- $\\hat{m}_1 = \\frac{(-0.3, 0.2)}{1 - 0.9} = (-3, 2)$, $\\quad \\hat{v}_1 = \\frac{(0.009, 0.004)}{1 - 0.999} = (9, 4)$, $\\quad \\sqrt{\\hat{v}_1} = (3, 2)$
- **Decay:** $0.99 \\cdot (0, 0) = (0, 0)$
- **Adam step:** $0.1 \\cdot \\frac{(-3, 2)}{(3, 2)} = (-0.1, 0.1)$
- $\\theta_1 = (0, 0) - (-0.1, 0.1) = (0.1, -0.1)$

The decay does nothing yet: $0.99 \\times 0 = 0$. Step 1 is identical to Adam's.

### Step 2

- $\\theta_1 = (0.1, -0.1)$
- $g_2 = (0.1 - 3,\\; -0.1 + 2) = (-2.9, 1.9)$
- $m_2 = 0.9 \\cdot (-0.3, 0.2) + 0.1 \\cdot (-2.9, 1.9) = (-0.56, 0.37)$
- $v_2 = 0.999 \\cdot (0.009, 0.004) + 0.001 \\cdot \\big((-2.9, 1.9) \\odot (-2.9, 1.9)\\big) = (0.017401, 0.007606)$
- $\\hat{m}_2 = \\frac{(-0.56, 0.37)}{1 - 0.9^2} = (-2.9474, 1.9474)$, $\\quad \\hat{v}_2 = \\frac{(0.017401, 0.007606)}{1 - 0.999^2} = (8.7049, 3.8049)$, $\\quad \\sqrt{\\hat{v}_2} = (2.9504, 1.9506)$
- **Decay:** $0.99 \\cdot (0.1, -0.1) = (0.099, -0.099)$
- **Adam step:** $0.1 \\cdot \\frac{(-2.9474, 1.9474)}{(2.9504, 1.9506)} = (-0.0999, 0.0998)$
- $\\theta_2 = (0.099, -0.099) - (-0.0999, 0.0998) = (0.1989, -0.1988)$

### Step 3

- $\\theta_2 = (0.1989, -0.1988)$
- $g_3 = (0.1989 - 3,\\; -0.1988 + 2) = (-2.8011, 1.8012)$
- $m_3 = 0.9 \\cdot (-0.56, 0.37) + 0.1 \\cdot (-2.8011, 1.8012) = (-0.7841, 0.5131)$
- $v_3 = 0.999 \\cdot (0.017401, 0.007606) + 0.001 \\cdot \\big((-2.8011, 1.8012) \\odot (-2.8011, 1.8012)\\big) = (0.025230, 0.010843)$
- $\\hat{m}_3 = \\frac{(-0.7841, 0.5131)}{1 - 0.9^3} = (-2.8934, 1.8934)$, $\\quad \\hat{v}_3 = \\frac{(0.025230, 0.010843)}{1 - 0.999^3} = (8.4183, 3.6178)$, $\\quad \\sqrt{\\hat{v}_3} = (2.9014, 1.9021)$
- **Decay:** $0.99 \\cdot (0.1989, -0.1988) = (0.1969, -0.1968)$
- **Adam step:** $0.1 \\cdot \\frac{(-2.8934, 1.8934)}{(2.9014, 1.9021)} = (-0.0997, 0.0995)$
- $\\theta_3 = (0.1969, -0.1968) - (-0.0997, 0.0995) = (0.2966, -0.2964)$

### What the example shows

Compare $\\theta_1$ with plain Adam on the same problem:

\`\`\`text
          Adam      AdamW
step 1    0.1000    0.1000
step 2    0.1999    0.1989    ← 0.001 smaller
step 3    0.2996    0.2966    ← 0.003 smaller
\`\`\`

- The difference is the decay: each step removes 1% of the current weight ($0.99 \\times 0.1 = 0.099$, i.e. 0.001 at step 2), and the effect adds up over steps.
- The Adam step itself stays about $\\eta = 0.1$ per step, just as in Adam, because the gradients that feed $m$ and $v$ do not include the decay.
- Here decay pulls $\\theta$ toward 0, away from the minimum at $(3, -2)$. In this toy problem every weight is needed, so decay only gets in the way; its benefit appears in a real model, where many weights are not needed.

## In PyTorch and in this site's GPT-2 code

\`torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)\`. Each step first multiplies every weight by $1 - \\eta\\lambda$ (\`1 - lr * weight_decay\`), then takes the Adam step.

### How strong the decay actually is

Note the $\\eta$ in $\\eta\\lambda$: \`weight_decay=0.1\` does **not** mean "shrink 10% per step". The code uses \`lr=3e-4\` and \`weight_decay=0.1\`:

\`\`\`text
η·λ = 0.0003 × 0.1 = 0.00003
each step:  θ ← 0.99997 · θ          ← shrinks only 0.003% per step
\`\`\`

To see what decay can do on its own, take a weight with **no gradient at all**: then $m = 0$, the Adam step is 0, and only the decay acts. Starting from 1, after $n$ steps it is $0.99997^n$:

\`\`\`text
after   1,000 steps:   0.970     ← this site's code runs max_steps = 1000
after  10,000 steps:   0.741
after  50,000 steps:   0.223
after 100,000 steps:   0.050
\`\`\`

A quick estimate: $0.99997^n \\approx e^{-0.00003\\, n}$. For example, at 10,000 steps $0.00003 \\times 10{,}000 = 0.3$ and $e^{-0.3} \\approx 0.741$. After about $1 / 0.00003 \\approx 33{,}000$ steps the weight is down to $1/e \\approx 37\\%$.

So weight decay is a slow, gentle pull. It does not erase useful weights, whose gradients push far harder than 0.003% per step; it acts on weights that go unneeded for tens of thousands of steps. With only 1000 steps, this site's training gets at most a 3% shrink from it; real GPT runs take hundreds of thousands of steps.

Because the decay is multiplied by $\\eta$, it also follows the learning rate: with an LR schedule (GPT-3 used warmup followed by cosine decay), the decay weakens as $\\eta$ goes down. This site's code has no LR schedule (see the [optimizer overview](#info:optim_adamw_notes)).

### Which parameters are decayed

The code splits the parameters into two groups by number of dimensions:

\`\`\`python
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
\`\`\`

For layer 0, for example:

\`\`\`text
decayed (2D):
  tok_emb.weight           (50304, 768)
  pos_emb.weight           (1024, 768)
  attn.c_attn.weight       (2304, 768)
  attn.proj.weight         (768, 768)
  mlp.fc_in.weight         (3072, 768)
  mlp.fc_out.weight        (768, 3072)

not decayed (1D):
  ln1.weight (γ), ln1.bias (β)     (768,)
  ln2.weight (γ), ln2.bias (β)     (768,)
  attn.c_attn.bias                 (2304,)
  attn.proj.bias                   (768,)
  mlp.fc_in.bias                   (3072,)
  mlp.fc_out.bias                  (768,)
\`\`\`

Over the whole model:

\`\`\`text
decayed:      50 tensors,  124,354,560 parameters   (99.9%)
not decayed:  98 tensors,      121,344 parameters   (0.1%)
\`\`\`

(\`lm_head.weight\` is the same tensor as \`tok_emb.weight\` because of [weight tying](#info:weight_tying_notes), so it is counted once.)

- **LayerNorm $\\gamma$ and $\\beta$:** in PyTorch's \`nn.LayerNorm\`, \`.weight\` is $\\gamma$ (initialized to 1) and \`.bias\` is $\\beta$ (initialized to 0), in $y = \\gamma \\cdot \\hat{x} + \\beta$. The natural value of $\\gamma$ is 1, not 0, so pulling it toward 0 would just shrink LayerNorm's output; that damages the model rather than simplifying it. $\\beta$ only shifts the output, so decaying it serves no regularization purpose either.
- **Biases:** a bias is added, not multiplied with the input. Memorizing the data with large weights happens in the matrices that multiply the input, not in biases. And biases are only 0.1% of the parameters, so decaying them would bring almost no benefit.

Decay is aimed at the large matrices that multiply the input (Linear and Embedding weights), which is where the model's capacity lies. \`p.dim() >= 2\` separates exactly those from the biases and LayerNorm parameters. This recipe comes from BERT and is used by GPT-3, nanoGPT, and others.

## References

1. **Decoupled Weight Decay Regularization** (AdamW). Loshchilov & Hutter, 2017. [arXiv:1711.05101 ↗](https://arxiv.org/abs/1711.05101)
`,
  },
  final_ln_notes: {
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
  zero_grad_notes: {
    title: "optimizer.zero_grad()",
    md: `
**Why every training step starts by clearing the gradients.**

## Where it sits in the training loop

Each step of the loop does five things:

\`\`\`python
for step in range(max_steps):
    x, y = dataloader.get_batch("train")    # 1. get a batch
    optimizer.zero_grad(set_to_none=True)   # 2. clear the previous step's gradients
    _, loss = model(x, y)                   # 3. forward pass, compute the loss
    loss.backward()                         # 4. backward pass, compute every parameter's gradient
    torch.nn.utils.clip_grad_norm_(...)     #    (clip the gradients)
    optimizer.step()                        # 5. update the parameters with the gradients
\`\`\`

This note is about step 2.

## The reason: gradients accumulate, they are not overwritten

\`backward()\` **adds** the new gradient to each parameter's \`.grad\`; it does not replace the old value.

Check with one parameter: $w = 2$ and $\\text{loss} = (3w)^2$, so $\\frac{d\\,\\text{loss}}{dw} = 18w = 36$.

\`\`\`text
backward three times without clearing:
step 0:  w.grad = 36
step 1:  w.grad = 72     ← 36 + 36
step 2:  w.grad = 108    ← 36 + 36 + 36

set w.grad back to None, then backward:
         w.grad = 36     ← correct
\`\`\`

Without clearing, the gradient used at step 100 would be the sum of the gradients from all 100 steps, and the parameters would be pushed further and further off.

## Why PyTorch accumulates

Accumulating is the behavior you want in two common cases:

1. **One parameter is used more than once.** [Weight tying](#hover:weight_tying_notes) is an example: \`tok_emb.weight\` is used at the input and again at the output, and the two gradients have to be added. Accumulation does exactly that.
2. **Gradient accumulation.** When a large batch does not fit on the GPU, split it into several micro-batches. Call \`backward()\` on each without clearing, then call \`step()\` once. The result is the same as training on the large batch. Karpathy's build-nanogpt reproduction of GPT-2 (124M) uses this to reach a batch of about 0.5M tokens.

\`\`\`python
optimizer.zero_grad(set_to_none=True)
for micro_step in range(grad_accum_steps):
    x, y = dataloader.get_batch("train")
    _, loss = model(x, y)
    loss = loss / grad_accum_steps   # so the sum equals the mean over the whole large batch
    loss.backward()                  # gradients add up in p.grad
optimizer.step()
\`\`\`

> The \`loss / grad_accum_steps\` matters: each micro-batch's loss is already a mean over its own tokens, so without the division the accumulated gradient would be \`grad_accum_steps\` times too large.

## What \`set_to_none=True\` does

There are two ways to clear:

\`\`\`text
set_to_none=False:  p.grad is filled with zeros (the tensor still takes memory)
set_to_none=True:   p.grad = None (the memory is freed)
\`\`\`

On the next \`backward()\`, if \`.grad\` is \`None\`, PyTorch writes the new gradient straight in, instead of reading a tensor of zeros and adding to it. This saves memory and is slightly faster. Since PyTorch 2.0, \`set_to_none=True\` is the default.

> One difference to be aware of: after clearing with \`None\`, a parameter that receives no gradient in the next step keeps \`.grad = None\`, not a tensor of zeros. Optimizers skip parameters whose \`.grad\` is \`None\`.
`,
  },
  backward_notes: {
    title: "loss.backward()",
    md: `
**What \`loss.backward()\` does: one backward pass that computes the gradient of every parameter.**

## \`backward()\` computes gradients; \`step()\` updates parameters

\`backward()\` does **not** change any weight. It only computes the gradients and stores them in each parameter's \`.grad\`. The next line, \`optimizer.step()\`, reads \`.grad\` and updates the parameters.

\`\`\`python
loss.backward()     # compute gradients, store them in p.grad; weights unchanged
optimizer.step()    # read p.grad, update the weights
\`\`\`

Check with one parameter: $w = 2$, $\\text{loss} = (3w)^2$, so the gradient is $18w = 36$. The optimizer is plain SGD with learning rate $\\eta = 0.01$:

\`\`\`text
before backward:  w = 2.0    w.grad = None
after backward:   w = 2.0    w.grad = 36      ← gradient computed, w unchanged
after step:       w = 1.64   w.grad = 36      ← now w changes: 2 − 0.01 × 36 = 1.64
\`\`\`

So \`.grad\` has to be stored: it is how \`backward()\` hands its result to \`step()\`.

### Why keep them as two separate steps

If \`backward()\` updated the parameters right away, none of these would be possible:

1. **Process the gradients before updating.** Between \`backward()\` and \`step()\` the code calls [\`clip_grad_norm_\`](#hover:clip_grad_norm_notes): it measures how large all the gradients are together and scales them down if they are too large. That needs every gradient to be computed first.
2. **Gradient accumulation.** Several micro-batches each call \`backward()\`, their gradients add up in \`.grad\`, and \`step()\` runs once. See [zero_grad](#hover:zero_grad_notes).
3. **Different update rules.** SGD does $w \\leftarrow w - \\eta \\cdot g$. AdamW, used for GPT-2, also uses running averages of past gradients and adapts the step size per parameter. \`backward()\` only provides the gradient; the optimizer you choose decides how to use it.

So the work in each training step is split like this:

\`\`\`text
zero_grad()  clear .grad
backward()   compute gradients, write them into .grad
(clip)       adjust .grad
step()       read .grad, update the parameters
\`\`\`

## One backward pass covers every parameter

In this code each training step calls \`loss.backward()\` once, and that one call fills the \`.grad\` of **all** ~124M parameters.

During the forward pass, PyTorch records every operation and links them into a **computation graph**, from the embeddings, through the 12 blocks, \`ln_f\` and \`lm_head\`, to the loss. \`backward()\` starts at the loss and walks this graph **once, in reverse**. At each layer it uses the chain rule to pass the gradient on, and computes that layer's parameter gradients along the way:

\`\`\`text
loss → lm_head → ln_f → block 12 → block 11 → ... → block 1 → pos_emb, tok_emb
\`\`\`

When the walk is done, every parameter that took part in the forward pass has its gradient. No parameter needs its own backward call.

> The one case with several \`backward()\` calls per step is gradient accumulation. Even then, each call computes all parameters' gradients; the calls just add up in \`.grad\` before the single \`step()\`.

## What each layer does: two things

During the backward pass, each layer receives one gradient from the layer after it: the gradient of the loss with respect to **this layer's output**. Call it $\\frac{\\partial L}{\\partial y}$, the upstream gradient. With it, the layer does two things:

\`\`\`text
1. Gradients of its own parameters:  dL/dW, dL/db   → stored in W.grad, b.grad
2. Gradient of its input:            dL/dx          → handed to the previous layer as its dL/dy
\`\`\`

Thing 2 is what lets the gradient travel back layer by layer: this layer's $\\frac{\\partial L}{\\partial x}$ is the previous layer's $\\frac{\\partial L}{\\partial y}$.

### Worked example: one Linear layer

\`nn.Linear(2, 3)\`: 2 inputs, 3 outputs, $y = x W^\\top + b$.

\`\`\`text
x = [1, 2]

W = [[1, 0],      b = [0, 0, 0]
     [0, 1],
     [1, 1]]

y = x @ W.T + b = [1, 2, 3]
\`\`\`

Suppose the layer after it hands back this upstream gradient:

\`\`\`text
dL/dy = [0.5, -1, 2]
\`\`\`

**Thing 1: the parameter gradients.** Each output is $y_i = x \\cdot W[i] + b_i$, so:

\`\`\`text
# row i = the i-th upstream number × the input
dL/dW[i] = dL/dy[i] × x

W.grad = [[ 0.5 × [1, 2]],   = [[ 0.5,  1],
          [-1   × [1, 2]],      [-1,   -2],
          [ 2   × [1, 2]]]      [ 2,    4]]

# b[i] is added straight to y[i], so its gradient passes through unchanged
b.grad = dL/dy = [0.5, -1, 2]
\`\`\`

**Thing 2: the input gradient.** $x$ feeds every $y_i$, so its gradient is the sum of the rows $W[i]$, weighted by $\\frac{\\partial L}{\\partial y_i}$:

\`\`\`text
dL/dx = 0.5 × W[0] + (−1) × W[1] + 2 × W[2]
      = 0.5 × [1, 0] − [0, 1] + 2 × [1, 1]
      = [2.5, 1]                 → passed to the previous layer
\`\`\`

> For a longer chain, where the gradient is multiplied by each layer's Jacobian in turn, see [this two-layer example](#hover:gradient_notes).

## Across the whole of GPT-2

\`\`\`text
loss
 ↓ dL/dlogits = (p − y) / (B*T)        ← derived in the cross-entropy note
lm_head:  compute W.grad, pass dL/dx to ln_f
 ↓
ln_f:     compute γ.grad and β.grad, pass dL/dx to block 12
 ↓
block 12 → block 11 → ... → block 1    every Linear and LayerNorm inside does the same two things
 ↓
tok_emb, pos_emb:  only thing 1 (they are the first layer; nothing comes before them)
\`\`\`

The starting gradient $\\frac{(p - y)}{B \\times T}$ is derived in [Cross Entropy](#hover:cross_entropy_notes).

Layers **without parameters**, such as GELU, softmax and dropout, only do thing 2: they transform the gradient and pass it on.

> The [weight tying](#hover:weight_tying_notes) example already used both things. Its "output side" gradient is \`lm_head\`'s thing 1 (parameter gradient); its "input side" gradient is \`lm_head\`'s thing 2: the $\\frac{\\partial L}{\\partial x}$ that flows back to the embedding row.
`,
  },
  clip_grad_norm_notes: {
    title: "clip_grad_norm_",
    md: `
**Gradient clipping: when the gradients are too large, scale them all down together.**

\`\`\`python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
\`\`\`

This line sits between \`backward()\` and \`step()\`. The gradients have been computed but not yet used to update the parameters, so this is the moment to check whether they are too large. (Why the two steps are separate: see [backward](#hover:backward_notes).)

## What it does

1. Treat the gradients of **all** parameters as one long vector (about 124M numbers for GPT-2) and compute its length, the norm: square every number, add them all up, take the square root.

   $$
   \\text{norm} = \\sqrt{\\sum_{\\text{all gradient entries } g} g^2}
   $$

2. If the norm is at most \`max_norm\` (here 1.0), do nothing.
3. If the norm is larger, multiply **every** gradient by the same factor $\\frac{\\text{max\\_norm}}{\\text{norm}}$, so the total length becomes exactly \`max_norm\`.

## Worked example

Suppose the model had only two parameters, \`a\` and \`b\`, with 2 numbers each:

\`\`\`text
a.grad = [3, 0]
b.grad = [0, 4]

step 1:  total norm = √(3² + 0² + 0² + 4²) = √25 = 5
step 3:  5 > 1.0, so the scale factor = 1.0 / 5 = 0.2

a.grad = [3, 0] × 0.2 = [0.6, 0]
b.grad = [0, 4] × 0.2 = [0, 0.8]

total norm after clipping = √(0.6² + 0.8²) = 1.0
\`\`\`

If the gradients are already small, nothing changes:

\`\`\`text
a.grad = [0.3, 0],  b.grad = [0, 0.4]
total norm = 0.5 ≤ 1.0   →  unchanged
\`\`\`

## Two key points

**1. It changes the length, not the direction.** Every gradient is multiplied by the same factor, so the ratios between parameters stay the same (\`a\` and \`b\` are still 3 : 4). The update points the same way as before; the step is just shorter.

**2. The norm is global, not per parameter.** If each parameter were clipped on its own, the ratio between \`a\` and \`b\` could change, and so would the direction of the update. Clipping by the global norm keeps the direction.

## Why it is needed

Most steps produce gradients of roughly similar size. Occasionally a "bad batch", such as an unusual stretch of data, produces a very large gradient. Without clipping, that one step would push the parameters far away and the loss would jump (a **loss spike**); in the worst case training never recovers.

Clipping puts a cap on how far any single step can move: a bad batch still affects that step, but only by a limited amount.

## The return value

\`clip_grad_norm_\` returns the total norm **before** clipping (5.0 in the example). Many training scripts print it:

\`\`\`python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"step {step} loss {loss.item():.4f} norm {norm:.4f}")
\`\`\`

This number is worth watching: a sudden jump in the norm is an early sign that training is going wrong.

## Details

- The trailing \`_\` in the name is a PyTorch convention for **in-place** operations: it modifies \`.grad\` directly instead of returning new gradients.
- The threshold 1.0 is the value used in the GPT-3 paper ("we clip the global norm of the gradient at 1.0"); most LLM training runs since then use 1.0 as well.
`,
  },
};
