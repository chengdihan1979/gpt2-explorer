import { ORIGINAL_INFO } from "../GPT2DecoderExplorer";

const prevents_vanishing_exploding_md = `
$$
\\textbf{Two concrete toy examples that show why pre-LN keep gradients from vanishing/exploding in deep stack}\\\\[10pt]
$$

---

$$
\\textbf{Example 1 — "Safe at init" path (identity highway)}\\\\[6pt]
\\text{Setup (early training): attention weights are small, so the sublayer output is near zero: } \\\\[6pt]
\\mathrm{attn}(\\mathrm{LN}(x)) \\approx 0.\\\\[6pt]
\\bullet\\space\\text{Pre-LN block: } x_{l+1} = x_l + \\mathrm{attn}(\\mathrm{LN}(x_l)) \\approx x_l.\\\\[6pt]
\\text{Backward: } \\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx I \\ \\Rightarrow\\ \\text{gradient passes nearly unchanged.}\\\\[6pt]
\\bullet\\space\\text{Post-LN (contrast): } x_{l+1} = \\mathrm{LN}(x_l + \\mathrm{attn}(x_l)) \\approx \\mathrm{LN}(x_l),\\\\[6pt]
\\text{so } \\dfrac{\\partial x_{l+1}}{\\partial x_l} \\approx J_{\\mathrm{LN}}(x_l)\\ \\text{and repeated products can shrink or grow gradients.}\\\\[10pt]
$$

---

$$
\\textbf{Example 2 — Jacobian picture with residual vs. no residual}\\\\[6pt]
\\text{Consider one block as a function } F(x) = x + f(\\mathrm{LN}(x)).\\ \\ \\\\[6pt]
\\bullet\\space\\textbf{Jacobian (pre-LN):}\\\\[6pt]
J_F(x) = I + J_f(\\mathrm{LN}(x))\\,J_{\\mathrm{LN}}(x).\\\\[6pt]
\\text{If }\\|J_f\\|\\ \\text{is small and }J_{\\mathrm{LN}}\\ \\text{is bounded, then } J_F \\approx I + \\epsilon\\ \\text{with small }\\epsilon.\\\\[6pt]
\\text{Across }L\\text{ layers: } \\prod_{\\ell=1}^{L}(I+\\epsilon_\\ell)\\ \\text{stays near }I\\ \\Rightarrow\\ \\text{no vanishing/exploding.}\\\\[6pt]
\\textbf{No residual: } \\\\[6pt]
G(x) = f(\\mathrm{LN}(x)),\\ \\ J_G = J_f\\,J_{\\mathrm{LN}}.\\\\[6pt]
\\text{Across depth: } \\prod_{\\ell=1}^{L} J_G(x_\\ell)\\ \\text{can vanish if }\\|J_G\\|<1\\ \\text{or explode if }\\|J_G\\|>1.\\\\[6pt]
\\href{https://mathworld.wolfram.com/SpectralNorm.html}{\\|J_G\\|\\text{ Spectral norm}}.\\\\
$$

---

$$
\\textbf{Why pre-LN helps more than post-LN}\\\\[6pt]
\\bullet\\space\\text{Pre-LN: } x_{l+1}=x_l + f(\\mathrm{LN}(x_l))\\\\[6pt] 
\\space\\space\\space\\text{keeps an identity path in both forward and backward.}\\\\[6pt]
\\bullet\\space\\text{Post-LN: } x_{l+1}=\\mathrm{LN}(x_l+f(x_l))\\\\[6pt] 
\\space\\space\\space\\text{Even the skip connection is passed through }J_{\\mathrm{LN}}\\ \\text{every layer, the identity gets scaled repeated,}\\\\
\\space\\space\\space\\text{which can accumulate into vanishing/exploding.}\\\\[10pt]
$$

---

$$
\\textbf{Bottom line}\\\\[6pt]
\\text{Pre-LN residual attention provides an identity gradient highway and bounds the non-identity part, }\\\\
\\text{preventing vanishing/exploding in deep stacks.}
$$
    `;

export const INFO = {
  ...(ORIGINAL_INFO || {}),
  input_data_notes: {
    title: "Input Data (Text)",
    md: `
**What it is**

- Raw text strings (documents, prompts) that will be tokenized before entering the model.

**Common preprocessing**

- Normalize whitespace/newlines if needed.
- Decide on max sequence length (T) and segmentation strategy (e.g., sliding windows, paragraph chunks).
- Split into train/val/test to avoid leakage.

**Batching shapes**

- After tokenization, inputs are **token IDs**: $(B, T)$.
- Optionally an attention mask: $(B, T)$ to mark padding (not typically used in GPT-2 causal-only training, but useful in general).
    `,
  },
  
  tokenizer_notes: {
    title: "Tokenizer (BPE for GPT-2)",
    md: `
$\\text{This line of code returns the GPT-2's byte-level BPE (BPE) tokenizer (vocab size 50257, includes the special token}$
$\\texttt{<|endoftext|>}\\text{ with id 50256).}\\\\$
$$\\text{It contains the byte}\\leftrightarrow\\text{\\href{#info:gpt2_byte_unicode_map_notes}{\\text{unicode map}}, \\href{#info:gpt2_merge_ranks_notes}{\\text{GPT-2 merge ranks}}, \\href{#info:regex_pre_tokenizer_notes}{\\text{regex pre-tokenizer}}, and special tokens.}\\\\[10pt]$$

$$\\textbf{How tokenization works}\\\\[4pt]$$
$$
1)\\ \\textbf{Byte encoding: }\\text{Encode the input text as UTF-8 bytes (case and whitespace are preserved).} \\\\
\\text{Map each byte to a safe Unicode symbol via the byte↔unicode table so every byte is representable.}\\\\[4pt]
\\text{Example:}\\\\[4pt]
\\text{" café"} \\\\[2pt]
\\rightarrow\\; \\text{Bytes: }[\\mathtt{0x20},\\ \\mathtt{0x63},\\ \\mathtt{0x61},\\ \\mathtt{0x66},\\ \\mathtt{0xC3},\\ \\mathtt{0xA9}] \\ \\text{(space and accents preserved).}\\\\[2pt]
\\rightarrow\\; \\text{Unicode: }[\\mathtt{U+0120}\\ \\text{'Ġ'},\\ \\mathtt{U+0063}\\ \\text{'c'},\\ \\mathtt{U+0061}\\ \\text{'a'},\\ \\mathtt{U+0066}\\ \\text{'f'},\\ \\mathtt{U+00C3}\\ \\text{'Ã'},\\ \\mathtt{U+00A9}\\ \\text{'©'}\\ ] \\\\[2pt]
\\rightarrow\\; \\text{the string "ĠcafÃ©".}\\\\[4pt]
\\href{#info:gpt2_byte_unicode_map_notes}{\\text{Deep dive into byte-unicode map}}
$$

$$
2)\\ \\textbf{Pre-tokenize: } \\text{Split the text into pieces with GPT-2's regex (separates runs of letters, digits, punctuation,} \\\\[2pt]
\\text{leading spaces, etc.).}\\\\[4pt]
\\text{Example:}\\\\[4pt]
\\text{"Hello, world!" → pieces: ["Hello", ",", "⎵world", "!"] (⎵ = leading space).}\\\\[4pt]
\\text{"2025-09-19" → ["2025", "-", "09", "-", "19"].}\\\\[4pt]
\\href{#info:regex_pre_tokenizer_notes}{\\text{Deep dive into regex pre-tokenizer}}
$$

$$
3)\\ \\textbf{BPE merges: } \\text{For each piece, iteratively merge symbol pairs using GPT-2's merge-ranks until no more merges apply,} \\\\
\\text{yielding subword units.}\\\\[10pt]
$$

$$
\\textbf{Notes: } \\text{Because it's byte-level, every input string is tokenizable without "unknown" tokens, and the original}\\\\[2pt]
\\text{spacing/casing is maintained.}\\\\[4pt]
$$

$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. Primary source (architecture + input representation):}$$
GPT-2 technical report. It spells out byte-level BPE, the motivation for byte-level processing, and the expanded vocab (50,257). See §2.2 "Input Representation" and Table 2.
$$\\href{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com}{\\texttt{GPT-2 Technical Report}}$$

$$\\textbf{2. Tokenizer implementation:}$$
OpenAI's tiktoken. Clear, practical reference for GPT-2's byte-level BPE (lossless, reversible), and how bytes ↔ tokens map is implemented.
$$\\href{https://github.com/openai/tiktoken}{\\texttt{tiktoken}}$$

$$\\textbf{3. Foundational BPE paper (why subword/byte-level):}$$
Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units." Not GPT-2-specific, but it's the classic reference behind subword/BPE vocabularies GPT-2 builds embeddings over.
$$\\href{https://aclanthology.org/P16-1162/?utm_source=chatgpt.com}{\\texttt{Neural Machine Translation of Rare Words with Subword Units}}$$

$$\\textbf{4. Let's build the GPT Tokenizer:}$$
Karpathy's video on GPT Tokenizer. Practical guides on GPT-2's byte-level BPE (from 1430s).
$$\\href{https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s}{\\texttt{BPE - Let's build the GPT Tokenizer}}$$

$$\\textbf{5. Hands-on/educational BPE references:}$$
Karpathy's minBPE (minimal byte-level BPE)
$$\\href{https://github.com/karpathy/minbpe?utm_source=chatgpt.com}{\\texttt{minbpe}}$$

$$\\textbf{6. Empirical analysis of GPT-2 embeddings:}$$
Interpretability deep-dive into GPT-2's embedding weights (distributional properties, relation to positions). Advanced exploration inside wte.
$$\\href{https://www.alignmentforum.org/posts/BMghmAxYxeSdAteDc/an-exploration-of-gpt-2-s-embedding-weights?utm_source=chatgpt.com}{\\texttt{An exploration of GPT-2's embedding weights}}$$
    `,
  },

  gpt2_byte_unicode_map_notes: {
    title: "GPT-2 byte-unicode map",
    md: `
$$\\textbf{What it is}\\\\[2pt]$$
$$
\\text{A fixed, 1-to-1 mapping between the 256 possible bytes (0-255) and 256 printable Unicode code points.}\\\\[4pt]
\\textbf{byte }\\to\\ \\textbf{unicode ("byte encoder"):}\\ \\text{ turns each raw byte into a single visible Unicode character.}\\\\
\\textbf{unicode }\\to\\ \\textbf{byte ("byte decoder"):}\\ \\text{ the exact inverse, so it's perfectly reversible.}\\\\[12pt]
$$
$$\\textbf{Why it exists:}\\\\[4pt]$$
$$
\\text{Classical BPE operates on "characters." GPT-2, however, is byte-level: it wants to handle any byte sequence}\\\\
\\text{(emoji, accents, binary-ish bytes, etc.) losslessly. Many bytes are control/whitespace and would confuse regex}\\\\
\\text{splitting or get normalized away. The map solves this by:}\\\\[4pt]
1. \\text{Giving every byte a safe, visible stand-in,}\\\\[4pt]
2. \\text{Avoiding collisions with whitespace/control characters,}\\\\[4pt]
3. \\text{Making regex pre-tokenization and BPE run on a "printable" string while staying reversible.}\\\\[12pt]
$$
$$\\textbf{How it's constructed (conceptually)}\\\\[2pt]$$
$$
\\bullet\\space \\text{Bytes that are already "nice printable" (e.g., ASCII and many Latin-1 characters) are mapped to themselves}\\\\
\\text{(same code point).}\\\\
\\bullet\\space \\text{The remaining bytes (e.g., space, tabs, newlines, etc.) are mapped to } \\textbf{distinct printable code points}\\\\
\\text{in higher Unicode ranges, ensuring they're visible and not whitespace.}\\\\
\\space\\space\\space\\text{Example:}\\\\[4pt]
\\space\\space\\space\\bullet\\space\\text{Byte 0x41 ('A')}\\rightarrow\\text{is mapped to standard-'A' (same character)}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Byte 0x20 (space)}\\rightarrow\\text{stand-in "Ġ" (U+0120) — that's why " world" encoded as Ġworld with a leading space.}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Bytes 0xC3 0xA9 (é)}\\rightarrow\\text{stand-ins something like Ã © (two Unicode chars, one per byte)}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Bytes 0xE2 0x80 0x94 (-)(em dash)}\\rightarrow\\text{three stand-ins (one per byte)}\\\\[6pt]
\\text{This yields two tables:}\\\\[4pt]
\\mathtt{byte\\_encoder: \\{0..255 \\to one-char\\ unicode\\ strings\\}}\\\\
\\mathtt{byte\\_decoder: \\{that\\ unicode\\ char \\to original\\ byte\\}}\\\\[12pt]
\\text{In GPT-2's map, the following bytes map to themselves: }\\\\[4pt]
\\text{33\\text{-}126 } \\; (\\text{printable ASCII excluding space}) \\; \\text{and } 161\\text{-}172,\\; 174\\text{-}255\\; (\\text{printable Latin-1}).\\\\[6pt]
\\text{Bytes outside those ranges (e.g., control chars }0\\text{-}31,\\; \\text{space }32,\\; 127\\text{-}160,\\; \\text{and }173) \\text{ get special stand-ins (e.g., }\\\\
\\mathrm{U+0100},\\ \\mathrm{U+0102}, \\ldots\\text{).}\\text{ That's why space (0x20) becomes } \\texttt{Ġ}\\text{.}\\\\[10pt]
\\textbf{Emoji example (UTF-8: F0 9F 98 8A)}\\\\[4pt]
\\space\\space\\space\\bullet\\space\\texttt{0xF0} \\; (=240) \\; \\text{is in the range 174 - 255} \\Rightarrow \\text{ maps to itself (prints as } \\texttt{ð} \\text{ in Latin-1).}\\\\[4pt]
\\space\\space\\space\\bullet\\space\\texttt{0x9F}\\; (159),\\; \\texttt{0x98}\\; (152),\\; \\texttt{0x8A}\\; (138) \\; \\text{are not in the printable sets } \\Rightarrow \\text{ each maps to a printable stand-in}\\\\
\\space\\space\\space\\text{Latin-Extended characters).}\\\\[6pt]
\\space\\space\\space\\text{So you get a mix: one "self" character plus three stand-ins — still one visible char per byte, fully reversible.}\\\\[10pt]
$$
$$\\textbf{How it's used (encode path)}\\\\[2pt]$$
$$
1.\\space\\text{Text }\\to\\ \\text{ UTF-8 bytes.}\\\\
2.\\space\\text{Map each byte } b \\text{ to its stand-in Unicode char } u = \\mathtt{byte\\_encoder}[b].\\\\
\\space\\space\\space\\space\\bullet\\space\\text{Example: " Hello" (leading space) }\\to\\ \\text{ bytes [0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F]}\\\\
\\quad \\to\\ \\text{ mapped string "ĠHello" (space became "Ġ"; ASCII letters map to themselves).}\\\\
\\space\\space\\space\\space\\bullet\\space\\text{Non-ASCII (e.g., é }\\to\\ \\text{ bytes C3 A9) becomes two mapped chars, one for each byte; both are visible stand-ins.}\\\\
3.\\space\\text{Run regex pre-tokenizer and BPE merges on this printable string.}\\\\[12pt]
$$
$$\\textbf{Decoding (exact inverse)}\\\\[2pt]$$
$$
1.\\space\\text{Take the merged "printable" token pieces }\\to\\ \\text{ concatenate to a string of mapped chars.}\\\\[4pt]
2.\\space\\text{Map each char back to a byte via } \\mathtt{byte\\_decoder}.\\\\[4pt]
3.\\space\\text{UTF-8 decode the bytes}\\to\\text{original text.}\\\\[4pt]
\\text{This guarantees lossless round-trip: } \\mathtt{decode(encode(s)) == s}
$$
    `,
  },

  regex_pre_tokenizer_notes: {
    title: "Regex pre-tokenizer",
    md: `
$$\\textbf{What it is:}$$
$$
\\text{A regular expression based splitter that chops the input text into chunks before BPE merging happens.}\\\\[4pt]
\\text{BPE merges are then applied within each chunk only, never across chunk boundaries.}\\\\[10pt]
$$
$$\\textbf{Why it exists:}$$
$$
\\text{Speed \\& determinism — fewer candidate merges, stable behavior.}\\\\[4pt]
\\text{Semantics — keeps leading spaces with the following token (so you get tokens like "}\\texttt{Ġworld}\\text{"),}\\\\
\\text{groups letters together, groups numbers together, and groups punctuation/symbols together.}\\\\[4pt]
\\text{Reversibility — pairs neatly with byte-level encoding for lossless round-trip.}\\\\[10pt]
$$
$$\\textbf{Rough pattern GPT-2 uses (conceptually):}$$
$$
\\text{contractions: }\\texttt{'s, 't, 're, 've, 'm, 'll, 'd}\\\\
\\text{optional leading space + letters: }\\texttt{?\\p\\{L\\}+}\\\\
\\text{optional leading space + digits: }\\texttt{?\\p\\{N\\}+}\\\\
\\text{optional leading space + other symbols: }\\texttt{?[\\s\\p\\{L\\}\\p\\{N\\}]+}\\\\
\\text{whitespace runs (including end-of-string cases).}\\\\[10pt]
$$
$$\\textbf{How it affects BPE:}$$
$$
\\text{Start from bytes (byte-level BPE) }\\to\\ \\text{ apply the regex pre-tokenizer }\\to\\ \\text{ for each chunk, repeatedly}\\\\
\\text{merge the lowest-rank adjacent pair until no more merges apply.}\\\\
\\text{Merges do not cross chunk boundaries. That's why " world" (with its leading space) is a single chunk where merges}\\\\
\\text{can happen internally, but it won't merge with the preceding token.}\\\\[10pt]

\\textbf{Special tokens:}\\;\\text{handled separately (e.g., }\\texttt{<|endoftext|>)\\text{). They bypass normal splitting when allowed.}\\\\[10pt]
$$

$$\\textbf{Examples:}$$
$$
\\textbf{Letters grouped}\\\\[4pt]
\\text{"Hello world"}\\ \\to\\ \\text{ chunks: ["Hello", " world"]}\\\\
\\text{(letters stay together; the leading space sticks to the next chunk).}\\\\
\\text{"AB12"}\\ \\to\\ \\text{ chunks: ["AB", "12"]}\\\\
\\text{(letters as one chunk, digits as another).}\\\\[8pt]
\\textbf{Numbers grouped}\\\\[4pt]
\\text{"Room 1234"}\\ \\to\\ \\text{ chunks: ["Room", " 1234"]}\\\\
\\text{"1,234.50"}\\ \\to\\ \\text{ chunks: [" 1", ",", "234", ".", "50"]}\\\\
\\text{(digits group, but punctuation like "," and "." are separate symbol chunks).}\\\\[8pt]
\\textbf{Punctuation / symbols grouped}\\\\[4pt]
\\text{"Hello, world!!"}\\ \\to\\ \\text{ chunks: ["Hello", ",", " world", "!!"]}\\\\
\\text{(",", and the "!!" run are symbol chunks).}\\\\
\\text{"Good-morning :) "}\\ \\to\\ \\text{ chunks: ["Good", "-", "morning", " :)", " "]}\\\\
\\text{(" :)" is a single symbols chunk; trailing space can be its own whitespace chunk).}\\\\[8pt]
\\textbf{Contractions handled specially}\\\\[4pt]
\\text{"can't"}\\ \\to\\ \\text{ chunks: ["can", "'t"]}\\\\
\\text{(the pre-tokenizer has specific rules for pieces like "'t", "'s", "'re", etc.).}\\\\[8pt]
\\textbf{Non-ASCII symbols / emoji}\\\\[4pt]
\\text{"Hi — ok 😊"}\\ \\to\\ \\text{ chunks: ["Hi", " —", " ok", " 😊"]}\\\\
\\text{(em dash and emoji are symbols; each becomes its own symbol chunk with any leading space).}\\\\[10pt]
$$
$$\\textbf{Key idea}$$
$$
\\text{It first splits the UTF-8 bytes into chunks using a regex: letters together, digits together, and symbols together,}\\\\
\\text{keeping any leading space with the next chunk. Then BPE merges run inside each chunk only—they don't cross}\\\\
\\text{chunk boundaries.} \\\\[10pt]
$$
$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. Transformer positional encodings:}$$
Vaswani et al., "Attention Is All You Need," §3.5. Introduces adding position information to token embeddings via fixed sinusoids.
$$\\href{https://arxiv.org/pdf/1706.03762v2}{\\texttt{Attention Is All You Need}}$$

$$\\textbf{2. GPT-2 uses learned absolute positional embeddings::}$$
Hugging Face GPT-2 docs (model card/config) explicitly note GPT-2's absolute position embeddings (not sinusoidal), hence right-padding guidance.
$$\\href{https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2?utm_source=chatgpt.com}{\\texttt{GPT-2 Configs}}$$

$$\\textbf{3. GPT-2 technical report (input representation):}$$
Radford et al., Language Models are Unsupervised Multitask Learners—describes inputs as token + position embeddings summed before the stack.
$$\\href{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com}{\\texttt{GPT-2 Technical Report}}$$

$$\\textbf{4. Relative position encodings:}$$
Shaw et al., Self-Attention with Relative Position Representations. Introduces relative positional embeddings as an alternative to absolute positional embeddings.
$$\\href{https://arxiv.org/pdf/1803.02155}{\\texttt{Self-Attention with Relative Position Representations}}$$

$$\\textbf{5. RoPE (Rotary Position Embeddings):}$$
Su et al., RoPE: Rotary Position Embedding. Introduces a novel method for encoding relative positions in self-attention.
$$\\href{https://arxiv.org/pdf/2104.09864}{\\texttt{RoPE: Rotary Position Embedding}}$$

$$\\textbf{6. ALiBi (Attention with Linear Biases):}$$
Press et al., uses distance-proportional biases in attention scores; enables length extrapolation.
$$\\href{https://arxiv.org/pdf/2108.12409}{\\texttt{ALiBi: Attention with Linear Biases}}$$

`,
  },

  lossless_round_trip_notes: {
    title: "Lossless round-trip",
    md: `
$$
\\textbf{Lossless round-trip:}\\\\[4pt]
\\text{Lossless round-trip" means if you encode text into token IDs and then decode those IDs back to text,}\\\\[1pt]
\\text{you get exactly the same bytes you started with—nothing is altered or lost.}\\\\[6pt]
\\text{Formally: }\\; \\mathtt{decode(encode(s)) = s}.\\\\[10pt]
\\textbf{Why true for GPT-2 (byte-level BPE):}\\\\[4pt]
1)\\ \\text{It operates on raw bytes (no lowercasing/normalization/space stripping).}\\\\
2)\\ \\text{Every byte sequence is representable.}\\\\
3)\\ \\text{Decode exactly inverts encode.}\\\\[10pt]
$$

    `,
  },

  mlp_fc_in_notes: {
    title: "MLP expansion: nn.Linear(n_embd, 4 * n_embd)",
    md: `
**What this layer does**

- Projects each token's hidden vector from size $n\\_embd$ up to a larger **feed-forward** size $4 * n\\_embd$.
- This is the first linear in the Transformer block's MLP (a.k.a. FFN):
  $$
  h' = \\phi(\\underbrace{W_1 h + b_1}_{\\text{this layer}}), \\quad \\
  y = W_2 h' + b_2
  $$

**Why 4X? (common design in Transformers)**

- The MLP is the main place where non-linearity and per-token mixing capacity live.
- Using a wider hidden dimension increases expressive power at modest depth.
- Empirically, 4x is a strong trade-off between quality and compute/memory for GPT-2 style models. Larger ratios (e.g., 8x) can improve quality but cost more; smaller (2x) may underfit.

**Shapes**

$$
\\boxed{\\begin{array}{l}
- \\text{Input per token}: h \\in \\mathbb{R}^{n\\_embd} \\\\[6pt]
- \\text{After this layer}: h_1 = W_1 h + b_1 \\in \\mathbb{R}^{4 \\times n\\_embd} \\\\[6pt]
- \\text{After activation (GELU): same shape} (4 \\times n\\_embd) \\\\[6pt]
- \\text{Second linear maps back}: (4 \\times n\\_embd) \\to (n\\_embd)
\\end{array}}
$$


**Parameter count (this layer only)**

$$
\\boxed{\\begin{array}{l}
- \\text{Weights}: 4 \\times n\\_embd \\times n\\_embd = 4 \\times n\\_embd^2 \\\\[6pt]
- \\text{Bias}: 4 \\times n\\_embd \\\\[6pt]
- \\text{Total}: ~4 \\times n\\_embd^2 + 4 \\times n\\_embd
\\end{array}}
$$


**Alternatives / notes**

- Some variants use ratios other than 4, gated linear units (GEGLU/SwishGLU), or different activations; the idea is similar: widen, nonlinearity, project back.
- In Pre-LN GPT-2 blocks, this comes after LayerNorm and parallel to attention in a residual structure.
    `,
  },
  mlp_fc_out_notes: {
    title: "MLP projection back: nn.Linear(4 * n_embd, n_embd)",
    md: `
**What this layer does**

- Projects features back from $4 * n\\_embd$ to $n\\_embd$ so the residual add aligns with the block input.
- FFN pattern recap:
  $$
  h' = \\phi(W_1 h + b_1), \\quad y = \\underbrace{W_2 h' + b_2}_{\\text{this layer}}
  $$

&nbsp;

**Why project back?**

- Residual connections need matching dimensions; compressing to $n\\_embd$ keeps compute stable across layers.

&nbsp;

**Shapes**

$$
\\boxed{\\begin{array}{l}
- \\text{Input}: h' \\in \\mathbb{R}^{4 \\times n\\_embd} \\\\[6pt]
- \\text{Output}: y \\in \\mathbb{R}^{n\\_embd}
\\end{array}}
$$

**Parameter count (this layer only)**

$$
\\boxed{\\begin{array}{l}
- \\text{Weights}: n\\_embd \\times (4 \\times n\\_embd) = 4 \\times n\\_embd^2 \\\\[6pt]
- \\text{Bias}: n\\_embd \\\\[6pt]
- \\text{Total}: ~4 \\times n\\_embd^2 + n\\_embd
\\end{array}}
$$

    `,
  },

  attn_residual_notes: {
    title: "Residual connection x = x + self.attn(self.ln1(x))",
    md: `
$$
\\textbf{This line is the pre-norm residual attention block. It does three things in one:}\\\\[8pt]
\\textbf{1. Normalize before attention (pre-LN).}\\\\[6pt]
\\mathtt{self.ln1(x)}\\ \\text{applies LayerNorm to each token's features so the attention sees a well-conditioned input}\\\\
\\text{(zero-mean, unit-variance per token). This stabilizes training.}\\\\[8pt]
\\textbf{2. Compute a context-dependent update.}\\\\[6pt]
\\mathtt{self.attn(...)}\\ \\text{(masked multi-head self-attention) looks at other tokens and returns an update with}\\\\
\\text{the same shape as } x\\ \\text{ (}B, T, C\\text{). You can think of it as a learned correction }\\Delta x\\ \\text{telling each token what to borrow from its context.}\\\\[8pt]
\\textbf{3. Residual (skip) connection.}\\\\[6pt]
\\mathtt{x = x + (...)}\\ \\text{adds that update onto the original stream. The residual path:}\\\\[6pt]
\\text{(i) gives a gradient highway (there's an identity path, so gradients flow even if the attention sublayer is poorly scaled early in training)}\\\\[6pt]
\\text{(ii) lets the block learn a small perturbation around identity (if the attention output is near zero, the block behaves like a no-op),}\\\\
\\text{which makes deep stacks much easier to optimize.}\\\\[10pt]
$$

---

$$
\\textbf{Why GPT-2 uses pre-LN specifically (}x + \\mathrm{Attn}(\\mathrm{LN}(x))\\textbf{) instead of post-LN (}\\mathrm{LN}(x + \\mathrm{Attn}(x))\\textbf{):}\\\\[6pt]
\\text{Pre-LN keeps the residual path's \\href{#info:gradient_notes}{\\text{gradient}} close to 1, which \\href{#info:prevents_vanishing_exploding_notes}{\\text{prevents vanishing/exploding}} and enables training many layers more reliably.}\\\\[6pt]
\\text{Attention also receives a normalized input, which helps its softmax and projections operate in a stable range.}\\\\[10pt]
$$

---

$$
\\textbf{Shape check (so the add works):}\\\\[6pt]
x:\\ (B, T, C)\\\\[6pt]
\\mathtt{self.ln1(x)}:\\ (B, T, C)\\\\[6pt]
\\mathtt{self.attn(...)}:\\ (B, T, C)\\\\[6pt]
\\text{Sum is elementwise, returning } (B, T, C).\\\\[10pt]
$$

---

$$
\\textbf{Big picture:}\\ \\text{the line implements "take your current token representations, normalize them, compute a context-aware}\\\\
\\text{adjustment via attention, and add that adjustment back"---a stabilized, residual refinement step.}
$$
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
$$
\\boxed{\\begin{array}{l} 
\\textbf{Here's a tiny 2-D numeric example that shows how the gradient is multiplied by each layer's Jacobian (matrix) in backprop.}\\\\[12pt] 
\\textbf{Setup}\\\\[4pt] 
\\text{Let }\\\\ W_1 = \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix},\\quad W_2 = \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix},\\quad x = \\begin{bmatrix}1\\\\ -2\\end{bmatrix}.\\\\[12pt] 
\\textbf{Forward pass}\\\\[4pt] 
z \\,=\\, W_1 x \\;=\\; \\begin{bmatrix}1.0 & 0.5\\\\ -0.2 & 1.2\\end{bmatrix} \\begin{bmatrix}1\\\\ -2\\end{bmatrix} \\;=\\; \\begin{bmatrix}1.0 + 0.5(-2)\\\\ -0.2 + 1.2(-2)\\end{bmatrix} \\;=\\; \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix}.\\\\[12pt] 
y \\,=\\, W_2 z \\;=\\; \\begin{bmatrix}0.8 & -0.3\\\\ 0.4 & 1.1\\end{bmatrix} \\begin{bmatrix}0.0\\\\ -2.6\\end{bmatrix} \\;=\\; \\begin{bmatrix}0.8\\cdot 0.0 + (-0.3)(-2.6)\\\\ 0.4\\cdot 0.0 + 1.1(-2.6)\\end{bmatrix} \\;=\\; \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}.\\\\[12pt] 
L \\,=\\, \\tfrac{1}{2}\\,\\|y\\|^2 \\,=\\, \\tfrac{1}{2}\\big(0.78^2 + (-2.86)^2\\big) \\,=\\, \\tfrac{1}{2}(0.6084 + 8.1796) \\,=\\, 4.394.\\\\[12pt] 
\\textbf{Backprop (chain rule = Jacobian multiplications)}\\\\[4pt] 
\\text{Since } L = \\tfrac{1}{2}\\|y\\|^2,\\quad \\dfrac{\\partial L}{\\partial y} = y = \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix}.\\\\[12pt] 
\\dfrac{\\partial L}{\\partial x} \\,=\\, \\Big(\\dfrac{\\partial y}{\\partial x}\\Big)^{\\!\\top}\\,\\dfrac{\\partial L}{\\partial y} \\,=\\, (W_2 W_1)^{\\!\\top} y \\,=\\, W_1^{\\top} W_2^{\\top} y.\\\\[12pt] 
\\text{Step 1: multiply by } W_2^{\\top}:\\\\[6pt] 
W_2^{\\top} = \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix},\\quad g_1 = W_2^{\\top}\\,\\dfrac{\\partial L}{\\partial y} \\,=\\, \\begin{bmatrix}0.8 & 0.4\\\\ -0.3 & 1.1\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} \\,=\\, \\begin{bmatrix}0.8\\cdot 0.78 + 0.4(-2.86)\\\\ -0.3\\cdot 0.78 + 1.1(-2.86)\\end{bmatrix} \\,=\\, \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix}.\\\\[12pt] 
\\text{Step 2: multiply by } W_1^{\\top}:\\\\[6pt] 
W_1^{\\top} = \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix},\\quad \\dfrac{\\partial L}{\\partial x} \\,=\\, W_1^{\\top} g_1 \\,=\\, \\begin{bmatrix}1.0 & -0.2\\\\ 0.5 & 1.2\\end{bmatrix} \\begin{bmatrix}-0.52\\\\ -3.38\\end{bmatrix} \\,=\\, \\begin{bmatrix}1.0(-0.52) + (-0.2)(-3.38)\\\\ 0.5(-0.52) + 1.2(-3.38)\\end{bmatrix} \\,=\\, \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}.\\\\[12pt] 
\\textbf{Check (single combined Jacobian)}\\\\[4pt] 
W_2 W_1 \\,=\\, \\begin{bmatrix}0.86 & 0.04\\\\ 0.18 & 1.52\\end{bmatrix},\\quad (W_2 W_1)^{\\top} y \\,=\\, \\begin{bmatrix}0.86 & 0.18\\\\ 0.04 & 1.52\\end{bmatrix} \\begin{bmatrix}0.78\\\\ -2.86\\end{bmatrix} \\,=\\, \\begin{bmatrix}0.86\\cdot 0.78 + 0.18(-2.86)\\\\ 0.04\\cdot 0.78 + 1.52(-2.86)\\end{bmatrix} \\,=\\, \\begin{bmatrix}0.156\\\\ -4.316\\end{bmatrix}.\\\\[12pt] 
\\textbf{What this shows}\\\\[4pt] 
\\text{1. Backprop takes the upstream gradient and left-multiplies by each layer's Jacobian transpose in reverse order.}\\\\ 
\\text{2. In deep stacks, this becomes a product of many matrices; that product can shrink or grow rapidly—vanishing/exploding gradients.}\\\\ 
\\text{3. Residual/identity paths keep each factor near identity, keeping the overall product well-behaved.} \\\\
\\text{4. NOTE : For } \\mathtt{𝑦=𝑊𝑥+𝑏}, \\text{the Jacobian w.r.t. 𝑥 is exactly } \\mathtt{𝐽=∂𝑦/∂𝑥=𝑊}.
\\text{In backprop you then multiply the upstream gradient by } \\mathtt{𝐽^⊤=𝑊^⊤}. \\\\
\\text{Outside of this special case, no—the Jacobian is not just } \\mathtt{𝑊^⊤}.
\\end{array}}
$$
    `
  },

  gpt2_merge_ranks_notes: {
    title: "GPT-2 merge ranks",
    md: `
$$\\textbf{Byte-Pair Encoding (BPE) refresher:}$$
$$
\\text{Start from a sequence of basic symbols (for GPT-2, bytes). A BPE model has a list of symbol-pair merges}\\\\
\\text{learned from a large corpus. The most frequent pair gets merge rank 0, the next gets rank 1, etc. This order is the}\\\\
\\text{merge ranks.}\\\\[10pt]
$$
$$\\textbf{Why ranks matter:}$$
$$
\\text{During encoding, the tokenizer repeatedly looks at adjacent pairs and merges the pair with the lowest rank}\\\\
\\text{number (highest priority) that appears in the list. It updates the sequence and repeats until no mergeable pairs}\\\\
\\text{remain. The final merged chunks are your tokens.}\\\\[10pt]
$$
$$\\textbf{What }\\mathtt{tiktoken}\\textbf{ stores:}$$
$$
\\text{GPT-2's tokenizer ships with a mapping that effectively represents these ranks. In practice:}\\\\[4pt]
\\bullet\\ \\text{Lower rank } \\Rightarrow \\text{ earlier (more frequent) merge rule } \\Rightarrow \\text{ applied before higher-rank rules.}\\\\[4pt]
\\bullet\\ \\text{This makes tokenization deterministic and matches GPT-2's training.}\\\\[10pt]
$$
$$\\textbf{Relation to token IDs:}$$
$$
\\text{In GPT-2/}\\mathtt{tiktoken}\\text{, every final merged result that BPE learns becomes an entry in the GPT-2 vocabulary with}\\\\[2pt] 
\\text{an integer token id. For normal tokens, tiktoken reuses the BPE rank as the token id. But special tokens,} \\\\[2pt]
\\text{e.g. <|endoftext|>, are assigned separate ids like 50256 and don't follow the rank rule.} \\\\[2pt]
\\text{So, in practice: If a merged string got rank 12345, then in GPT-2/tiktoken its token id is 12345 (unless it's a special token).} \\\\[4pt]
$$
$$\\textbf{Tiny toy example (made up):}$$
$$
\\text{Text: } \\mathtt{banana} \\;\\to\\; \\text{bytes} \\;\\to\\; \\text{initial symbols: } b\\ a\\ n\\ a\\ n\\ a\\\\[4pt]
\\text{Suppose merges (with ranks):}\\\\[2pt]
\\text{rank 0: } (a,n) \\to an\\\\[2pt]
\\text{rank 1: } (an,a) \\to ana\\\\[2pt]
\\text{rank 2: } (b,an) \\to ban\\\\[4pt]
\\text{Encoding proceeds:}\\\\[2pt]
\\text{Merge lowest-rank pairs present } \\to\\ b\\ an\\ a\\ n\\ a\\\\[2pt]
\\text{Now } (an,a) \\text{ exists } \\to\\ b\\ ana\\ n\\ a\\\\[2pt]
\\text{No more lower-rank merges available (or continue if others exist) } \\to\\ \\text{final tokens.}\\\\[8pt]
$$
$$\\textbf{Space handling:}$$
$$
\\text{GPT-2 is byte-level. Space is part of the bytes and many tokens encode "space+word" as a single token}\\\\[2pt]
\\text{(often shown with a leading \\texttt{Ġ} in visualizations). Those arise from specific merges and their ranks.}\\\\[8pt]
$$
    `,
  },

  attention_linear_notes: {
    title: "Attention linear",
    md: `
$\\textbf{What it is:}\\\\[4pt]$
$$\\mathtt{self.c\\_attn = nn.Linear(n\\_embd,\\ 3 * n\\_embd,\\ bias=True)}$$ is a single fused projection that computes $\\textbf{Q, K, V}$ in one matmul.
Equivalent to stacking three $\\mathtt{Linear(n\\_embd, n\\_embd)}$ layers:

$$
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}.\\\\[20pt]
$$

$\\textbf{Parameter shapes:}\\\\[4pt]$
$$
\\text{Weight: } (3\\times n\\_{\\text{embd}},\\ n\\_{\\text{embd}}),\\quad
\\text{Bias: } (3\\times n\\_{\\text{embd}}).\\\\[12pt]
$$

$\\textbf{What it does to the tensor:}\\\\[4pt]$
$$
\\text{Input } x \\in \\mathbb{R}^{B\\times T\\times n\\_{\\text{embd}}}.\\\\[4pt]
\\mathtt{qkv = c\\_attn(x)}\\ \\Rightarrow\\ qkv \\in \\mathbb{R}^{B\\times T\\times 3\\times n\\_{\\text{embd}}}.\\\\[4pt]
\\mathtt{q,\\ k,\\ v = qkv.split(n\\_embd,\\ 2)}\\ \\Rightarrow\\ q,k,v \\in \\mathbb{R}^{B\\times T\\times n\\_{\\text{embd}}}.\\\\[4pt]
\\text{Split q, k, and v into heads with } hs = n\\_{\\text{embd}}/n\\_{\\text{head}}\\\\[4pt]
\\mathtt{view(B, T, n\\_head, hs).transpose(1, 2)}\\ \\Rightarrow\\ (B,\\ n\\_{\\text{head}},\\ T,\\ hs).\\\\[12pt]
$$

$\\textbf{Why fuse Q/K/V?}\\\\[4pt]$
$$
\\text{Fewer kernel launches / better cache use (one big GEMM vs. three separate linears). Results are identical to three separate linears.}\\\\[12pt]
$$

$\\textbf{Parameter count:}\\\\[4pt]$
$$
\\text{Weights: } 3\\times n\\_{\\text{embd}}^2,\\quad \\text{Bias: } 3\\times n\\_{\\text{embd}}.\\\\
\\text{Example } (n\\_{\\text{embd}}=768):\\ \\text{weights } 3\\cdot 768^2=1{,}769{,}472,\\ \\text{bias } 3\\cdot 768=2{,}304.\\\\[12pt]
$$

$\\textbf{PyTorch default initialization:}\\\\[8pt]$
$$\\text{By default in PyTorch, nn.Linear initializes weights with }\\textbf{Kaiming uniform}\\text{ and biases with a plain uniform with the same bound.}\\\\[6pt]$$
$\\bullet\\ \\textbf{Weights: }\\text{PyTorch initializes Linear.weight with Kaiming (He) uniform:}\\\\[4pt]$

&nbsp;

\`\`\`python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
\`\`\`

&nbsp;

Passing $\\text{a=math.sqrt(5)}$ is a PyTorch trick so that the resulting uniform range becomes $\\mathcal{U}(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}}, \\tfrac{1}{\\sqrt{\\text{fan\\_in}}})$,
numerically the same range you'd get from a plain uniform with that bound. This was done for backward-compatibility when they refactored to use $\\mathtt{kaiming\\_uniform\\_}. \\href{https://discuss.pytorch.org/t/nn-linear-default-weight-initialisation-assumes-leaky-relu-activation/46712/2?utm_source=chatgpt.com}{\\texttt{Forum post}}$.

$$
W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right),\\quad
\\text{with } \\text{fan\\_in}=n\\_{\\text{embd}}.\\ \\text{(For } n\\_{\\text{embd}}=768:\\ \\approx\\pm 0.0361\\text{.)}\\\\[12pt]
$$

$\\bullet\\ \\textbf{Biases: }\\text{Bias is not Kaiming; it's sampled explicitly from a plain uniform with the same bound:}\\\\[4pt]$

&nbsp;

\`\`\`python
fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
bound = 1 / sqrt(fan_in)
init.uniform_(self.bias, -bound, bound)
\`\`\`

&nbsp;

$$
\\Rightarrow b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right),\\\\
$$

So: weights use Kaiming-uniform configured to yield the same range as a simple uniform, while biases use that simple uniform directly.

&nbsp;

**Note**: Many Transformer/GPT-2 implementations override this and use a normal init $\\mathcal{N}(0,0.02^{2})$
for linear layers/embeddings. 

&nbsp;

$\\textbf{References:}\\\\[4pt]$$
$\\textbf{1. Kaiming initialization: }$
introduces the initialization for rectifiers (ReLU/LeakyReLU) and derives the variance-preserving scheme.
$\\href{https://arxiv.org/pdf/1502.01852.pdf}{\\texttt{He et al., 2015}} \\\\[4pt]$

$\\textbf{2. PyTorch defaults: }$
See the PyTorch init code that points to the exact behavior.
$\\href{https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50}{\\texttt{PyTorch init docs}} \\\\[4pt]$

$\\textbf{3. GPT-2 / HF Transformers practice (overriding PyTorch defaults): }\\\\[4pt]$
GPT-2 (HF Transformers): weight matrices initialized from a (truncated) Normal(0, 0.02) controlled by initializer_range=0.02 in the config. This is the widely used default when you instantiate GPT-2 from Transformers.
$\\href{https://huggingface.co/docs/transformers/main/model_doc/gpt2?utm_source=chatgpt.com}{\\texttt{GPT-2 Configs}} \\\\[4pt]$

You can also see this policy described in general Transformer init notes (HF/others): "weights ~ Normal(0, initializer_range) (often 0.02), biases 0; LayerNorm γ=1, β=0."
$\\href{https://huggingface.co/transformers/v3.1.0/internal/modeling_utils.html?utm_source=chatgpt.com}{\\texttt{HF Transformers Modeling Utils}} \\\\[4pt]$

`,
  },

  attention_output_linear_notes: {
    title: "Attention output projection",
    md: `
Intuitively, the projection layer increases expressivity, this layer learns a linear combination so information from
different heads can interact before entering the residual. Without it, heads would stay in disjoint slices and couldn't
"talk" before the next block.

&nbsp;

$\\text{On the high level, think of it as following}\\\\[2pt]$
$$
\\bullet\\ \\text{Q/K do routing (which heads to attend to which other heads)}\\\\[2pt]
\\bullet\\ \\text{V carries content}\\\\[2pt]
\\bullet\\ \\text{Projection chooses where/how to write that content into the residual space (a learned basis change)}\\\\[2pt]
\\text{It lets the model reweight heads, rotate features, and gate what survives to the next layer.}\\\\[10pt]
$$

$\\textbf{Parameter shapes}\\\\[4pt]$
$$
\\text{Weight: } (n\\_embd, n\\_embd),\\quad \\text{Bias: } (n\\_embd).\\\\[2pt]
\\text{More strictly, the weight dimension is } (n\\_head * hs, n\\_embd). \\text{ But for GPT-2, they are often the same.}\\\\[4pt]
$$

$\\textbf{PyTorch default initialization:}\\\\[4pt]$
Refer to the "PyTorch default initialization" section in the $\\href{#info:attention_linear_notes}{\\text{Attention linear notes}}$ for the details.

&nbsp;

The weight of this layer is also called the "residual-writing weights" because it is final projection matrix that writes the residual update to the original stream.

&nbsp;

Why I call this out: at initialization some implementations scale these residual-writing weights by $\\frac{1}{\\sqrt{N}}$ (or $\\frac{1}{\\sqrt{2N}}$) where N is the number of residual layers.
so that the sum of $\\frac{1}{\\sqrt{N}}$ residual additions doesn't blow up in variance with depth. But this scaling technique is not universal anymore because:

&nbsp;

$\\textbf{1. Pre-norm (often RMSNorm) stabilizes depth by design}\\\\[4pt]$
In post-norm Transformers, the residual stream's variance can grow with depth, so 
$\\frac{1}{\\sqrt{N}}$ helps keep it in range. Modern stacks use pre-norm/RMSNorm: each block first normalizes the stream, then applies the sublayer, then adds it back. That normalization pins the scale right before the residual add, so extra 
$\\frac{1}{\\sqrt{N}}$ at init is largely redundant.

&nbsp;

$\\textbf{See a tiny numeric \\href{#info:post_norm_scaling_example_notes}{example} on why Post-norm needs this scaling but Pre-norm doesn't.}$

`,
  },

  attn_dropout_notes: {
    title: "Attention dropout",
    md: `
$$\\text{Attention dropout applies dropout to the attention matrix after softmax and masking, before the value mix:}\\\\[2pt]$$
$$A=\\mathrm{softmax}(S+\\text{mask}),\\quad \\tilde A=\\mathrm{Dropout}_p(A),\\quad Y=\\tilde A\\,V.\\\\[2pt]$$
$$\\text{In code: }\\mathtt{att = self.attn\\_drop(att)}\\ \\text{ where }\\mathtt{att}\\text{ is already softmaxed.}\\\\[8pt]$$

$$\\textbf{Why it helps (intuition)}\\\\[2pt]$$
$$\\text{- Think of attention as a graph from each query token to earlier tokens; dropout randomly removes some edges.}\\\\[2pt]$$
$$\\text{- Prevents over-reliance on a single edge, nudging heads to find backup paths and generalize better.}\\\\[2pt]$$
$$\\text{- Acts like gentle exploration: sometimes forcing the model to look elsewhere.}\\\\[8pt]$$

$$\\textbf{When it is active}\\\\[2pt]$$
$$\\text{Training only: enabled in }\\mathtt{model.train()};\\\\[2pt]$$
$$\\text{Disabled at inference: }\\mathtt{model.eval()}. \\\\[8pt]$$

$$\\textbf{Shape}\\\\[2pt]$$
$$\\mathtt{att}\\in\\mathbb{R}^{B\\times n\\_head\\times T\\times T}\\ \\longrightarrow\\ \\text{same shape after dropout.}\\\\[8pt]$$

$$\\textbf{Mechanics (inverted dropout)}\\\\[2pt]$$
$$\\tilde A \\;=\\; \\frac{A\\odot M}{1-p},\\qquad M_{ij}\\sim\\mathrm{Bernoulli}(1-p).\\\\[2pt]$$
$$\\text{This keeps the expected value unchanged:: }\\mathbb{E}[\\tilde A]=A\\\\[2pt]$$
$$\\text{(row sums are not exactly 1 per sample, but the expectation matches).}\\\\[8pt]$$

$$\\textbf{Practical notes}\\\\[2pt]$$
$$\\text{- Apply after masking and softmax (never drop masked zeros).}\\\\[2pt]$$
$$\\text{- Typical }p\\approx 0.1\\ \\text{(e.g., GPT-2 uses ~0.1) }\\\\[2pt]$$
$$\\text{Smaller datasets may benefit from a larger p; huge models sometimes set it to 0 if other regularizers suffice..}\\\\[8pt]$$

$$\\textbf{One intuition example}\\\\[2pt]$$

$$\\bullet\\ \\text{Suppose a head usually puts most attention on "dogs" to decide the verb should be "were".}\\\\[2pt]$$
$$\\bullet\\ \\text{With attention dropout, that strongest edge is sometimes zeroed during training.}\\\\[2pt]$$
$$\\bullet\\ \\text{When that happens, the model must use other cues (plurality elsewhere, other heads) to solve the task correctly.}\\\\[2pt]$$
$$\\bullet\\ \\text{Over time, this builds redundancy and avoids brittle shortcuts, making predictions more robust.}\\\\[4pt]$$

    `,
  },

  resid_dropout_notes: {
    title: "Residual dropout in Causal Self-Attention",
    md: `
  $$\\textbf{Causal residual dropout (a.k.a. dropout on the attention branch's residual update)}\\\\[4pt]$$
  Refer to the $$\\href{#info:mlp_dropout_notes}{\\text{MLP residual-dropout notes}}$$ for: what dropout is, why residual dropout is useful, train vs. eval behavior and example. 
  Everything below focuses on what's specific to causal self-attention.

  &nbsp;

  $$\\textbf{What it does}\\\\[4pt]$$
  The Causal residual dropout is applied on the mixed feature vector after heads are concatenated and projected by $$\\mathtt{W_O}$$, right before the residual add.
  At this point, all attention heads have been combined into a single update vector for each position.
  
  &nbsp;

  $$\\textbf{Why it is useful}\\\\[4pt]$$
  Heads can become spiky (e.g., lock onto one token path). Even if the attention pattern is spiky, the written features still pass through $$\\mathtt{W_O}$$
  before being added to the original stream. Randomly muting parts of that write:
  
  $$
  \\bullet \\; \\text{discourages dependence on a single head/channel's features}\\\\[2pt]
  \\bullet \\; \\text{encourages redundant cues across heads,}
  $$

  Think of $$\\mathtt{Q/K/V}$$ as the "read side" of the attention mechanism. Then $$\\mathtt{W_O}$$ is the "write side".
  Attention dropout perturbs the "read side" pattern, while residual dropout regularizes and encourages more robust, distributedthe learned write directions
  and reducing reliance on any single head/channel even if the read pattern is spiky.
  `,
  },

  mlp_residual_notes: {
    title: "MLP residual",
    md: `
$$
\\textbf{Code:}\\ \\mathtt{x = x + self.mlp(self.ln2(x))}\\\\[8pt]
\\textbf{1. Normalize before attention (pre-LN).}\\\\[6pt]
\\mathtt{self.ln2(x)}\\ \\text{applies LayerNorm per token (across embedding dimensions) so the MLP sees a well-conditioned input }\\\\
\\text{(roughly zero-mean, unit-variance per token). This stabilizes training.}\\\\[8pt]
$$
$$
\\textbf{2. Position-wise MLP transforms features.}\\\\[6pt]
\\text{The feed-forward sublayer is applied independently at each time step (no token mixing). Typically:}\\\\[4pt]
\\mathtt{x\\ \\to\\ \\mathrm{Linear}({n\\_embd}\\!\\to\\!4\\times {n\\_embd})\\ \\to\\ \\mathrm{GELU}\\ \\to\\ \\mathrm{Linear}(4\\times {n\\_embd}\\!\\to\\!{n\\_embd})\\ \\to\\ (\\mathrm{Dropout)}\\\\[4pt]
\\text{This expand}\\to\\text{nonlinearity}\\to\\text{project back pattern builds rich, non-linear feature interactions per token.}\\\\[8pt]
$$
$$
\\textbf{3. Residual (skip) connection.}\\\\[6pt]
\\mathtt{x + (\\ldots)}\\ \\text{adds the MLP's output back to the original stream. Benefits:}\\\\[6pt]
\\text{(i) Gradient highway: there is an identity path, helping \\href{#info:mlp_prevents_vanishing_exploding_notes}{\\text{prevent vanishing/exploding}} early in training.}\\\\[6pt]
\\text{(ii) Small refinements: if the MLP output is near zero, the block behaves like a near-identity map.}\\\\[10pt]
$$
$$
\\textbf{Shapes (so the add is valid):}\\\\[6pt]
x:\\ (B, T, C)\\\\[6pt]
\\mathtt{self.ln2(x)}:\\ (B, T, C)\\\\[6pt]
\\mathtt{self.mlp(\\ldots)}:\\ (B, T, C)\\\\[6pt]
\\text{Elementwise sum }\\Rightarrow\\ (B, T, C).\\\\[10pt]
$$
$$
\\textbf{Big picture:}\\ \\text{normalize token features }\\to\\ \\text{compute a per-token non-linear update with the MLP }\\to\\ \\text{add that update back.} \\\\
\\text{This is the pre-norm residual MLP block that improves stability and optimization in deep Transformers.}
$$

    `,
  },

  highlevel_transformer_introduction_notes: {
    title: "High-level Transformer Analogy",
    md: `
$$\\textbf{A high-level analogy of a GPT-2 transformer block.}\\\\[8pt]$$

$$\\textbf{Tokens = people in a meeting.}\\\\[2pt]$$
Each token is like a person holding their own notes (its current representation).

&nbsp;

$$\\textbf{Attention = the conversation.}\\\\[2pt]$$
Everyone briefly scans what others are saying and decides whose points matter for them. Each person builds a short
"summary I need" by selectively listening.

Why it matters: this is how long-range connections form—facts from far away can instantly influence you.

&nbsp;

$$\\textbf{MLP (feed-forward) = private thinking.}\\\\[2pt]$$
After listening, each person goes off for a moment of inner reasoning: "Given what I heard, how should I update my notes?"

Why it matters: it lets each token transform and rephrase ideas, adding nonlinearity and capacity.

&nbsp;

$$\\textbf{Two LayerNorms = the organizer's ground rules (applied twice).}\\\\[2pt]$$
Before the conversation and before the private thinking, the organizer reminds everyone to keep comments calm and on-topic—
no one's voice too loud, none too quiet.

Why it matters: it keeps both phases stable and comparable so the block behaves predictably.

&nbsp;

$$\\textbf{Residual connections = a running draft.}\\\\[2pt]$$

After each phase (conversation, then private thinking), you add your new insights to your previous notes rather than replacing them.

Why it matters: you never lose what you already knew; updates are incremental and safe.

&nbsp;

$$\\textbf{Dropout = occasional silence to build resilience.}\\\\[2pt]$$
Sometimes random remarks are withheld during practice meetings.

Why it matters: the group learns to make good decisions even if a few voices are missing, preventing overreliance on any single cue.

&nbsp;


$$\\textbf{Putting it together:}\\\\[2pt]$$

Each block is a two-step cycle—communicate (attention) then compute (MLP)—with the organizer (LayerNorm) keeping order,
a running draft (residual) preserving history, and practice with partial attendance (dropout) to ensure robustness.
Stacking many such blocks lets tokens repeatedly share, rethink, and refine.

`,
},

  causal_analogy_notes: {
    title: "Causal Analogy",
    md: `
$$

\\href{#info:causal_analogy_notes}{\\text{Causal Analogy}}\\quad\\href{#info:attn_inference_example_notes}{\\text{Attention Inference Example}}\\\\[8pt]

\\textbf{Below is an analogy to show how the GPT-2 Multi-Head Self Attention block works in order to give you an intuitive understanding.}\\\\[8pt]
\\textbf{Setting:}\\
\\text{A row of people, one per position } t \\text{ in the sequence. Each person holds notes } x_t \\text{ (corresponding to the embedding of the token at position t)}\\text{.}\\\\[10pt]

\\textbf{1) Make three cards from your notes — } \\mathtt{c\\_attn(x) \\to q, k, v}\\\\[4pt]
\\quad \\textbf{Query (}q\\textbf{): } \\text{what I'm looking for now (built by } W_q\\text{).}\\\\[4pt]
\\quad \\textbf{Key (}k\\textbf{): } \\text{what I can offer (be matched on by key) (built by } W_k\\text{); an index used for matching only.}\\\\[4pt]
\\quad \\textbf{Value (}v\\textbf{): } \\text{what I will share if you listen (built by } W_v\\text{); the actual content that gets copied/mixed into someone else's representation..}\\\\[4pt]
\\quad \\textbf{Why separate Key vs Value?}~\\\\[4pt]
\\text{Keys are for addressing (similarity search), Values are for content.}\\\\
\\text{You choose \\emph{who} to listen to via } q\\!\\cdot\\!k \\text{ and copy what matters from those people via } v\\text{.}\\\\[10pt]

\\textbf{2) Split into committees (heads) — reshape to } (B, n\\_{\\text{head}}, T, hs)\\\\[4pt]
\\text{Each committee (head) is a parallel committee with its own } W_q, W_k, W_v \\text{ and looks at a different perspective (syntax, long-range cues, etc.).}\\\\[10pt]

\\textbf{3) Score who to listen to — } \\mathtt{att\\_logits = q @ k^\\top / \\sqrt{hs}}\\\\[4pt]
\\text{Each person compares each query with everyone's keys to get compatibility scores; divide by } \\sqrt{hs} \\text{ to keep them numerically calm so softmax doesn't explode.}\\\\[10pt]

\\textbf{4) Causal rule (no looking ahead) — mask future positions to } -\\infty\\\\[4pt]
\\text{This enforces the LM factorization } p(x_{1:T}) = \\prod_{t=1}^{T} p(x_t \\mid x_{\\le t-1}).\\\\[4pt]
\\textbf{Why it exists:}\\\\[4pt]
\\quad \\bullet~\\textbf{No leakage during training: } \\text{even though we compute all positions in parallel for speed, the mask guarantees position t never uses t + 1 ...}.\\\\[4pt]
\\quad \\bullet~\\textbf{Matches generation: } \\text{at inference we generate left-to-right; causality during training makes the learned behavior consistent with how we use the model.}\\\\[4pt]
\\quad \\bullet~\\textbf{Prevents cheating: } \\text{without masking, the model could peek at the answer (future tokens) and cheat.}\\\\[10pt]

\\textbf{5) Turn scores into listening weights — } \\mathtt{att = softmax(att\\_logits,\\ dim=-1)}\\\\[4pt]
\\text{Now each person has a probability distribution over allowed earlier positions: }\\textit{how much attention do I give to each source? Rows sum to 1 → interpretable mixing weights.}\\\\[10pt]

\\textbf{6) Build your personalized summary from Values — } \\mathtt{y = att @ v}\\\\[4pt]
\\text{Mix others' Value vectors using your attention weights (e.g., } 0.6\\,v_3 + 0.3\\,v_7 + \\cdots \\text{).}\\\\[4pt]
\\text{Remember: Keys help you choose; Values are what you take.}\\\\[10pt]

\\textbf{7) Practice with partial connections — } \\mathtt{attn\\_drop(att)}\\\\[4pt]
\\text{Randomly drop some links during training to build robustness and avoid overreliance on one path.}\\\\[10pt]

\\textbf{8) Reassemble committees \\& standardize — merge heads, then } \\mathtt{proj(y)}\\\\[4pt]
\\text{Stitch head outputs back to one vector per position, project to model width, and feed the residual stream cleanly.}\\\\[10pt]

\\textbf{Big picture.}\\
\\text{At each position } t\\text{: select earlier positions via Query-Key (addressing), copy useful content via Values,}\\\\[4pt]
\\text{combine multiple perspectives (heads), and do it under a causal rule that matches left-to-right generation.} \\\\[4pt]

$$

    `,
  },

  attn_inference_example_notes: {
    title: "Attention Inference Example",
    md: `
  
  $$
  \\href{#info:causal_analogy_notes}{\\text{Causal Analogy}}\\quad\\href{#info:attn_inference_example_notes}{\\text{Attention Inference Example}}\\\\[8pt]
  \\textbf{In the below example, we will show how a single attention head can infer the correct verb form (were vs was) from the context of the sentence.} \\\\[8pt]
  \\textbf{Input Sentence: The\\ dogs\\ that\\ barked\\ loudly\\ were\\ tired.} \\\\[8pt]
  \\text{We focus on predicting the token }\\mathtt{were}\\text{ (position 6) from the prefix }\\mathtt{The\\ dogs\\ that\\ barked\\ loudly}\\text{.} \\\\[4pt]
  $$
  
  ---
  
  $$
  \\textbf{What happens (one head, simplified)} \\\\[4pt]
  
  \\text{Goal: decide between }\\mathtt{were}\\text{ vs }\\mathtt{was}\\text{.} \\\\[4pt]
  \\text{In order to do this, we need to know the true subject of the clause so we can infer the correct verb form.} \\\\[4pt]
  \\text{In this example, the true subject is }\\mathtt{dogs}\\text{ (plural), even though there's an intervening clause "that barked loudly."} \\\\[4pt]
  
  \\textbf{Notes:}\\\\[4pt]
  \\text{1. In real models, the model doesn't actually know it's a verb position. There are no POS tags or grammar rules baked in.}\\\\
  \\text{Instead, the model learns from training to recognize contexts where a verb is likely next.}\\\\
  \\text{The model has a hidden state } x_t \\text{ that's a function of}\\\\
  \\bullet\\ \\text{buildt from all previous tokens and itself }\\\\
  \\bullet\\ \\text{the positional embedding } p_t\\\\
  \\bullet\\ \\text{many layers of self-attention and feed-forward networks}\\\\[4pt]
  \\textit{The hidden state } x_t \\text{ encodes contextual features like "we're in a place where a verb typically follows a plural subject..."}\\\\
  \\textit{but it could also encode other things such as "likely punctuation next". In this case, the model might predict a comma instead: "The dogs that barked loudly, …".}\\\\
  \\textit{the hidden state could also encode things "Alternate continuations". In this case, the model might predict a different verb such as 'seemed' or 'are': "The dogs that barked loudly are tired."}\\\\[4pt]
  \\textbf{but in this toy example, we focus on why the model predicts "were" instead of "was"}.
  
  \\\\[4pt]

  \\text{2. Assumption: This is an inference example, so we assume all model parameters are already trained: the projections }\\mathbf{W}_q,\\ \\mathbf{W}_k,\\ \\mathbf{W}_v\\ \\text{(and the output projection }\\mathbf{W}_o\\text{), the }\\mathtt{MLP}\\text{s, }\\mathtt{LayerNorm}\\text{s, embeddings, and the }\\mathtt{LM\\ head.}\\\\
  \\text{At inference time these weights are fixed. The example's numbers are illustrative, but they reflect what trained parameters typically make happen at inference.}\\\\[4pt]

  $$
  
  ---
  
  $$
  \\textbf{Step A --- Values carry useful features} \\\\[4pt]
  
  \\text{Suppose this head's Value vectors include:} \\\\[4pt]
  
  \\bullet\\ \\text{pluralness (singular }-1\\,\\dots\\,+1\\text{ plural)}\\\\[4pt]
  \\bullet\\ \\text{subjectness (is this token a good subject head?)} \\\\[4pt]

  \\textbf{Notes: }\\text{This is a }\\textbf{simplification} \\text{. In real models, the Value vectors are }\\textbf{a high-dimensional learned representation - the head size. } \\text{(e.g. GPT-2 small has 768 dimensions and 12 heads, so head size is 64).}\\\\
  \\text{So in real models it isn't literally a 2-number feature; we just projected the idea into two dimensions to illustrate what the head might encode.}\\\\[4pt]
  
  \\text{Toy values:} \\\\[4pt]
  \\mathtt{The}:\\ \\mathbf{V}\\approx\\begin{bmatrix}0.0\\\\[2pt]0.0\\end{bmatrix} \\\\[4pt]
  \\mathtt{dogs}:\\ \\mathbf{V}\\approx\\begin{bmatrix}+1.0\\\\[2pt]+0.9\\end{bmatrix}\\ \\leftarrow\\ \\text{plural noun, likely subject} \\\\[4pt]
  \\mathtt{that}:\\ \\mathbf{V}\\approx\\begin{bmatrix}0.0\\\\[2pt]0.1\\end{bmatrix} \\\\[4pt]
  \\mathtt{barked}:\\ \\mathbf{V}\\approx\\begin{bmatrix}0.0\\\\[2pt]0.0\\end{bmatrix} \\\\[4pt]
  \\mathtt{loudly}:\\ \\mathbf{V}\\approx\\begin{bmatrix}0.0\\\\[2pt]0.0\\end{bmatrix} \\\\[4pt]
  $$

  ---

  $$
  \\textbf{Step B --- Queries/Keys score who to listen to} \\\\[4pt]
  \\text{The head computes the Query }\\mathbf{q}_5\\text{ to each Key }\\mathbf{k}_j\\ (j=1\\ldots5)\\text{.} \\\\[4pt]
  \\text{Made-up attention logits (after the }1/\\sqrt{hs}\\text{ scale):} \\\\[4pt]
  
  \\bullet\\ \\text{to }\\mathtt{The}:\\ -0.5\\\\
  \\bullet\\ \\text{to }\\mathtt{dogs}:\\ +2.2\\ \\leftarrow\\ \\text{strong match}\\\\
  \\bullet\\ \\text{to }\\mathtt{that}:\\ +0.3\\\\
  \\bullet\\ \\text{to }\\mathtt{barked}:\\ -0.2\\\\
  \\bullet\\ \\text{to }\\mathtt{loudly}:\\ -0.1
  
  \\\\[4pt]
  \\textbf{Notes: }\\text{The numerical values we listed are made up to show the pattern (e.g., subject getting the highest score). In the real model they come from those dot products with the trained }\\mathbf{W}_q,\\ \\mathbf{W}_k,\\ \\text{ plus masking.}\\\\[4pt]
  
  \\text{Then we apply softmax to get the attention weights (sum to 1):} \\\\[4pt]
  
  \\bullet\\ w(\\mathtt{The})=0.04,\\\\
  \\bullet\\ w(\\mathtt{dogs})=0.78,\\\\
  \\bullet\\ w(\\mathtt{that})=0.10,\\\\
  \\bullet\\ w(\\mathtt{barked})=0.04,\\\\
  \\bullet\\ w(\\mathtt{loudly})=0.04
  
  $$

  ---
  
  $$
  \\textbf{Step C --- Mix Values with those weights (att @ v)} \\\\[4pt]
  
  \\text{Weighted average produces the head's output at position }5\\text{:} \\\\[4pt]
  
  \\mathbf{y}_6
  =0.04\\cdot\\begin{bmatrix}0,0\\end{bmatrix}
  +0.78\\cdot\\begin{bmatrix}+1.0,+0.9\\end{bmatrix}
  +0.10\\cdot\\begin{bmatrix}0,0.1\\end{bmatrix}
  +0.04\\cdot\\begin{bmatrix}0,0\\end{bmatrix}
  +0.04\\cdot\\begin{bmatrix}0,0\\end{bmatrix}
  \\\\[4pt]

  \\mathbf{y}_6\\ \\approx\\ \\begin{bmatrix}+0.78,+0.71\\end{bmatrix}\\\\[4pt]
  
  \\textbf{Interpretation:} \\text{ position 5 now carries a strong plural + subject cue.} \\\\[4pt]
  $$

  ---
  
  $$
  \\textbf{Step D --- Why this flips the logits the right way} \\\\[4pt]
  
  \\text{Think of the LM head as rows (one per vocab item) that dot with the final hidden vector.} \\\\[4pt]
  
  \\text{The rows for just the two verbs:} \\\\[4pt]
  
  \\bullet\\ \\operatorname{row}(\\mathtt{were})\\ \\approx\\ \\begin{bmatrix}+1.4,+0.5\\end{bmatrix}
  \\quad
  \\text{(likes plural \\& subject)}
  
  \\\\

  \\bullet\\ \\operatorname{row}(\\mathtt{was})\\ \\approx\\ \\begin{bmatrix}-1.1,+0.2\\end{bmatrix}
  \\quad
  \\text{(penalizes plural)}

  \\\\[4pt]
  
  \\text{Dot products from this head's contribution:} \\\\[4pt]
  
  \\bullet\\ \\operatorname{logit}(\\mathtt{were})\\approx 1.4\\cdot 0.78 + 0.5\\cdot 0.71 \\approx +1.45\\\\
  \\bullet\\ \\operatorname{logit}(\\mathtt{was})\\approx (-1.1)\\cdot 0.78 + 0.2\\cdot 0.71 \\approx -0.66\\\\[4pt]
  
  \\text{After combining with other heads/layers and softmax over the vocab, }\\mathtt{were}\\text{ gets much higher probability than }\\mathtt{was}\\text{.}
  $$

  ---
  
  $$
  \\textbf{Why }\\mathtt{dogs}\\text{ has high influence here} \\\\[4pt]
  \\bullet\\ \\text{Syntactic role: It's the true subject of the clause containing the verb, even across "that barked loudly."}\\\\[4pt]
  \\bullet\\ \\text{Training pressure: Across millions of examples, gradients push the parameters so that a verb's Query aligns strongly with the Key of its governing subject. That makes the attention weight on the subject high.}\\\\[4pt]
  \\bullet\\ \\text{Value payload: The subject's Value carries exactly the features (plurality, subjectness) that help pick the correct verb form. If the model predicts the wrong form, gradients increase the }q\\cdot k\\text{ for the right subject and shape its }\\mathbf{V}\\text{ to be more informative.}\\\\[4pt]
  $$

  ---

  $$
  \\textbf{What if we flip to singular "dog" in the example: }\\\\[4pt]
  \\text{With "}\\mathtt{dog}\\text{" (singular) instead of "}\\mathtt{dogs}\\text{" (plural), the same attention head that tracks subject--verb agreement will now pass singular evidence forward, so the LM head will score "}\\mathtt{was}\\text{" higher than "}\\mathtt{were}\\text{."}\\\\[4pt]
  $$

  $$
  \\textbf{Here's the same toy walk-through you saw, just flipped to singular:}

  \\textbf{Prefix: }\\text{"}\\mathtt{The\\ dog\\ that\\ barked\\ loudly}\\text{"}
  $$

  $$
  \\textbf{A) Values carry features (toy features: }[\\text{pluralness},\\ \\text{subjectness}]\\text{):}\\\\[4pt]
  \\bullet\\ \\mathtt{The}:\\ \\ \\mathbf{V}\\ \\approx\\ \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}\\\\
  \\bullet\\ \\mathtt{dog}:\\ \\ \\mathbf{V}\\ \\approx\\ \\begin{bmatrix}-1.0 & +0.9\\end{bmatrix}\\ \\leftarrow\\ \\text{singular noun, likely subject}\\\\
  \\bullet\\ \\mathtt{that}:\\ \\ \\mathbf{V}\\ \\approx\\ \\begin{bmatrix}0.0 & 0.1\\end{bmatrix}\\\\
  \\bullet\\ \\mathtt{barked}:\\ \\ \\mathbf{V}\\ \\approx\\ \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}\\\\
  \\bullet\\ \\mathtt{loudly}:\\ \\ \\mathbf{V}\\ \\approx\\ \\begin{bmatrix}0.0 & 0.0\\end{bmatrix}
  $$

  $$
  \\textbf{B) Queries/Keys decide who to listen to (for the next-token slot):}\\\\[4pt]
  \\text{Potion 5's query compares against keys of position 1 to 5. As before, it strongly matches the subject:}\\\\[4pt]
  \\bullet\\ \\text{to }\\mathtt{The}: -0.5\\\\
  \\bullet\\ \\text{to }\\mathtt{dog}: +2.2\\ \\leftarrow\\ \\text{"Strong match"}\\\\
  \\bullet\\ \\text{to }\\mathtt{that}: +0.3\\\\
  \\bullet\\ \\text{to }\\mathtt{barked}: -0.2\\\\
  \\bullet\\ \\text{to }\\mathtt{loudly}: -0.1\\\\[4pt]
  \\text{Softmax }\\to\\ \\text{attention weights: }\\quad
  w(\\mathtt{The})=0.04,\\ \\ w(\\mathtt{dog})=0.78,\\ \\ w(\\mathtt{that})=0.10,\\ \\ w(\\mathtt{barked})=0.04,\\ \\ w(\\mathtt{loudly})=0.04
  $$

  $$
  \\textbf{C) Mix values with those weights (att @ v):}\\\\[4pt]
  \\mathbf{y}\\ \\approx\\ 0.78\\,\\cdot\\,\\begin{bmatrix}-1.0, +0.9\\end{bmatrix}\\ +\\ 0.10\\,\\cdot\\,\\begin{bmatrix}0.0, 0.1\\end{bmatrix}\\ \\approx\\ \\boxed{\\begin{bmatrix}-0.78, +0.71\\end{bmatrix}}\\\\[4pt]
  \\text{Interpretation: strong singular signal plus subject cue at position 5.}
  $$

  $$
  \\textbf{D) LM head rows (toy) and resulting logits:}\\\\[4pt]
  \\text{Think of each vocab token as a row that dots with the hidden vector.}\\\\[4pt]
  \\bullet\\ \\operatorname{row}(\\mathtt{were})\\ \\approx\\ \\begin{bmatrix}+1.4 & +0.5\\end{bmatrix}\\quad(\\text{likes plural \\& subject})\\\\
  \\operatorname{logit}(\\mathtt{were})\\approx 1.4\\cdot(-0.78) + 0.5\\cdot 0.71 \\approx -0.73\\\\
  \\bullet\\ \\operatorname{row}(\\mathtt{was})\\ \\approx\\ \\begin{bmatrix}-1.1 & +0.2\\end{bmatrix}\\quad(\\text{penalizes plural, okay with subject})\\\\
  \\operatorname{logit}(\\mathtt{was})\\approx (-1.1)\\cdot(-0.78) + 0.2\\cdot 0.71 \\approx +1.00
  $$

  $$
  \\text{So }\\operatorname{logit}(\\mathtt{was}) > \\operatorname{logit}(\\mathtt{were})\\text{. After combining all heads/layers and softmax over the full vocab, the model strongly prefers "}\\mathtt{was}\\text{" in the singular-subject context.}

  $$
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
  $$\\textbf{What it is:}\\\\[4pt]$$
  Transformers need order information because self-attention has no sense of sequence.
  Positional embeddings give each token a unique vector based on its position.
  
  &nbsp;

  Note that GPT-2's positional embeddings are learnable parameters in a lookup table that get trained by backpropagation exactly like any other weight.
  It's not the fixed sinusoids like in the original Transformer paper.
  
  &nbsp;

  The shape of the positional embedding weight matrix is:
  $$
  P \\in \\mathbb{R}^{\\text{max\\_toks} \\times n\\_embd}
  $$
  
  $$\\text{- Here, max\\_toks is the maximum sequence length (e.g. 1024 for GPT-2).}$$
  $$\\text{- Each row }P_j\\text{ corresponds to position }j\\text{ in the sequence.}$$
  
  &nbsp;
  
  ---
  
  &nbsp;

  $$\\textbf{Why do we need positional embeddings?}\\\\[4pt]$$
  Intuitively, we need positional embeddings to give the model a sense of order and position in the sequence. But how does it exactly work?

  Suppose we don't add the positional embeddings, only the token embeddings $$\\mathtt{X} \\in \\mathbb{R}^{T \\times n\\_embd}$$ (We ignore the batch dimension for simplicity).

  Self attention computes the following:
  $$
  \\mathtt{Q} = \\mathtt{X} \\cdot \\mathtt{W}_q, \\quad \\mathtt{K} = \\mathtt{X} \\cdot \\mathtt{W}_k, \\quad scores = \\mathtt{Q} \\cdot \\mathtt{K}^\\top
  $$

  Now permute (shuffle) the sequence with a permutation matrix $$\\mathtt{P}$$.
  The new inputs are $$\\mathtt{X'} = \\mathtt{P} \\cdot \\mathtt{X}$$. Then

  $$
  \\mathtt{Q'} = \\mathtt{X'} \\cdot \\mathtt{W}_q = \\mathtt{P} \\cdot \\mathtt{X} \\cdot \\mathtt{W}_q = \\mathtt{P} \\cdot \\mathtt{Q} \\\\[2pt]
  \\mathtt{K'} = \\mathtt{X'} \\cdot \\mathtt{W}_k = \\mathtt{P} \\cdot \\mathtt{X} \\cdot \\mathtt{W}_k = \\mathtt{P} \\cdot \\mathtt{K}
  $$
  
  So the scores become:
  $$
  \\mathtt{scores'} = \\mathtt{Q'} \\cdot \\mathtt{K'}^\\top = (\\mathtt{P} \\cdot \\mathtt{Q}) \\cdot (\\mathtt{P} \\cdot \\mathtt{K})^\\top = \\mathtt{P} \\cdot (\\mathtt{Q} \\cdot \\mathtt{K}^\\top) \\cdot \\mathtt{P}^\\top
  $$

  That's the same numbers as before, just with rows/columns permuted. In other words, self-attention is permutation-equivariant when there's no position info: 
  shuffling inputs just shuffles the attention matrix in the same way. The model can "see" which tokens are present but not their order.

  &nbsp;

  Adding positional embeddings $$\\mathtt{P_t}$$ breaks that symmetry: it changes $$\\mathtt{Q}$$ and $$\\mathtt{K}$$ so that swapping tokens also swaps which
  $$\\mathtt{P_t}$$ they get. Now the dot products include position-dependent terms, so $$\\mathtt{QK}^\\top$$ changes in value, not just by a row/column reordering.
  That's how the model becomes sensitive to order.

  &nbsp;

  If you are still confused, look at the tiny example $$\\href{#info:positional_embeddings_example_notes}{\\text{here}}$$

  &nbsp;

  ---

  &nbsp;
  
  $$\\textbf{Runtime usage}\\\\[8pt]$$
  
  1. Token IDs → token embeddings:  
  $$
  \\text{tok\\_emb}(\\text{input\\_ids}) \\in \\mathbb{R}^{B \\times T \\times n\\_embd}
  $$
  
  2. Positions $0,1,\\dots,T-1$ → positional embeddings:  
  $$
  \\text{pos\\_emb}(\\text{positions}) \\in \\mathbb{R}^{T \\times n\\_embd}
  $$
  
  3. Add them together via broadcasting: ([What is broadcasting?](#info:broadcasting)) 
  $$
  x = \\text{tok\\_emb}(\\text{input\\_ids}) + \\text{pos\\_emb}(\\text{positions})
  $$
  
  ---
  
  &nbsp;
  
  $$\\textbf{Parameters}\\\\[4pt]$$
  
  $$
  \\text{params} = \\text{max\\_toks} \\times n\\_embd
  $$
  
  For GPT-2 small:  
  $$
  1024 \\times 768 \\approx 7.86 \\times 10^5
  $$
  
  Much smaller than the token embedding table (~38M params).
  
  &nbsp;
  
  ---
  
  &nbsp;
  
  $$\\textbf{Shapes at a glance}\\\\[8pt]$$
  $$\\bullet$$ Weight: $[\\text{max\\_toks}, n\\_embd]$  
  $$\\bullet$$ Positions: $[T]$  
  $$\\bullet$$ Pos embeddings: $[T, n\\_embd]$  
  $$\\bullet$$ Final sum: $[B, T, n\\_embd]$  

  &nbsp;

  ---

  &nbsp;

  $$\\textbf{Extra Notes:}\\\\[4pt]$$
  Learned absolute positional embeddings used in GPT-2 are not widely used in the newest state-of-the-art LLMs, 
  which have largely transitioned to relative positional embeddings and other novel methods like $$\\href{https://arxiv.org/pdf/2104.09864}{\\texttt{RoPE (Rotary Position Embeddings)}}$$ 
  or $$\\href{https://arxiv.org/pdf/2108.12409}{\\texttt{ALiBi (Attention with Linear Biases)}}$$ .

  &nbsp;

  ---

  &nbsp;

  $$\\textbf{References:}\\\\[4pt]$$
  $$\\textbf{1. Transformer positional encodings:}$$
  Vaswani et al., "Attention Is All You Need," §3.5. Introduces adding position information to token embeddings via fixed sinusoids.
  $$\\href{https://arxiv.org/pdf/1706.03762v2}{\\texttt{Attention Is All You Need}}$$

  $$\\textbf{2. GPT-2 uses learned absolute positional embeddings::}$$
  Hugging Face GPT-2 docs (model card/config) explicitly note GPT-2's absolute position embeddings (not sinusoidal), hence right-padding guidance.
  $$\\href{https://huggingface.co/docs/transformers/v4.34.0/model_doc/gpt2?utm_source=chatgpt.com}{\\texttt{GPT-2 Configs}}$$
  
  $$\\textbf{3. GPT-2 technical report (input representation):}$$
  Radford et al., Language Models are Unsupervised Multitask Learners—describes inputs as token + position embeddings summed before the stack.
  $$\\href{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com}{\\texttt{GPT-2 Technical Report}}$$
  
  $$\\textbf{4. Relative position encodings:}$$
  Shaw et al., Self-Attention with Relative Position Representations. Introduces relative positional embeddings as an alternative to absolute positional embeddings.
  $$\\href{https://arxiv.org/pdf/1803.02155}{\\texttt{Self-Attention with Relative Position Representations}}$$

  $$\\textbf{5. RoPE (Rotary Position Embeddings):}$$
  Su et al., RoPE: Rotary Position Embedding. Introduces a novel method for encoding relative positions in self-attention.
  $$\\href{https://arxiv.org/pdf/2104.09864}{\\texttt{RoPE: Rotary Position Embedding}}$$

  $$\\textbf{6. ALiBi (Attention with Linear Biases):}$$
  Press et al., uses distance-proportional biases in attention scores; enables length extrapolation.
  $$\\href{https://arxiv.org/pdf/2108.12409}{\\texttt{ALiBi: Attention with Linear Biases}}$$

  `,
  },

  positional_embeddings_example_notes: {
    title: "Positional Embeddings Example",
    md: `
  $$\\textbf{Setup (no positions):}\\\\[4pt]$$

  Let token embeddings be 2-D vectors: $$e(A) = \\begin{bmatrix}1,0\\end{bmatrix}, e(B) = \\begin{bmatrix}0,1\\end{bmatrix}.$$

  &nbsp;

  Use identity projections so $$Q=X,\\ K=X,\\ \\text{scores } S = QK^\\top = XX^\\top.$$

  $$
  \\text{Sequence } [A, B]\\Rightarrow X=\\begin{bmatrix}1&0\\\\0&1\\end{bmatrix}\\\\[4pt]
  \\text{Score S } = XX^\\top = \\begin{bmatrix}1&0\\\\0&1\\end{bmatrix}.\\\\[4pt]
  \\text{Sequence }[B, A] \\text{ just permutes rows of } X.\\\\[4pt]
  \\text{According to what we said in "\\href{#info:embedding_pos}{\\text{Positional Embedding notes}}". Scores S' = }PSP^{\\top}.\\\\[4pt]
  \\text{So, it's same numbers as S, just rows/cols swapped. Self-attention can't tell AB vs BA—only a permutation.}
  $$

  $$\\textbf{Add positional embeddings:}$$

  &nbsp;

  Let position vectors be $$p_0 = \\begin{bmatrix}1,1\\end{bmatrix}, p_1 = \\begin{bmatrix}-1,1\\end{bmatrix}.\\\\[4pt]$$
  We feed $$\\mathtt{h}_t = e(token_t) + p_t.$$

  $$
  \\bullet\\ \\text{Sequence [A, B] (A at pos 0, B at pos 1):}\\\\[4pt]
  \\mathtt{h}_0 == \\begin{bmatrix}1,0\\end{bmatrix} + \\begin{bmatrix}1,1\\end{bmatrix} = \\begin{bmatrix}2,1\\end{bmatrix}\\\\[4pt]
  \\mathtt{h}_1 == \\begin{bmatrix}0,1\\end{bmatrix} + \\begin{bmatrix}-1,1\\end{bmatrix} = \\begin{bmatrix}-1,2\\end{bmatrix}\\\\[4pt]
  \\mathtt{h} == \\begin{bmatrix}2&1\\\\-1&2\\end{bmatrix}\\\\[4pt]
  \\text{Score S = }\\mathtt{h}\\mathtt{h}^\\top = \\begin{bmatrix}2&1\\\\-1&2\\end{bmatrix}\\begin{bmatrix}2&-1\\\\1&2\\end{bmatrix} = \\begin{bmatrix}5&0\\\\0&5\\end{bmatrix}.
  $$

  $$
  \\bullet\\ \\text{Sequence [B, A] (B at pos 0, A at pos 1):}\\\\[4pt]
  \\mathtt{h}_0 == \\begin{bmatrix}0,1\\end{bmatrix} + \\begin{bmatrix}1,1\\end{bmatrix} = \\begin{bmatrix}1,2\\end{bmatrix}\\\\[4pt]
  \\mathtt{h}_1 == \\begin{bmatrix}1,0\\end{bmatrix} + \\begin{bmatrix}-1,1\\end{bmatrix} = \\begin{bmatrix}0,1\\end{bmatrix}\\\\[4pt]
  \\mathtt{h} == \\begin{bmatrix}1&2\\\\0&1\\end{bmatrix}\\\\[4pt]
  \\text{Score S = }\\mathtt{h}\\mathtt{h}^\\top = \\begin{bmatrix}1&2\\\\0&1\\end{bmatrix}\\begin{bmatrix}1&0\\\\2&1\\end{bmatrix} = \\begin{bmatrix}5&2\\\\2&1\\end{bmatrix}.
  $$

  Now $$S'$$ is not just a row/column permutation of $$S$$; the dot products changed in value.
  Without positions, shuffling only permutes the attention matrix; with positions, the numbers themselves change, so the model becomes order-sensitive.

  `,
  },

  layernorm: {
    title: "LayerNorm",
    md: `
$\\textbf{nn.LayerNorm}$ — applies layer normalization over the last dimension of the input (the embedding dimension $n\\_embd$).

Mathematically, given any input vector $h \\in \\mathbb{R}^{d}$, LayerNorm computes
$$
\\text{LN}(h) = \\gamma \\odot \\frac{h - \\mu(h)}{\\sigma(h) + \\epsilon} + \\beta
$$
where $\\mu,\\sigma$ are the mean and std over the last (feature) dimension, and $\\gamma \\in \\mathbb{R}^{d},\\beta \\in \\mathbb{R}^{d}$ are learned scale/shift.

&nbsp;

Here is sometimes called the "final layer norm", which applies to each token individually, across its embedding dimensions before the final projection layer (lm_head).

&nbsp;

$\\textbf{Example}$ ($\\gamma$ and $\\beta$ are ignored for simplicity).

&nbsp;

For a token embedding vector

$$
x = [5.0,\\; 4.5,\\; 6.2,\\; 4.8],
$$

LayerNorm computes the mean and standard deviation over the 4 dimensions:

$$
\\mu = 5.125, \\quad \\sigma \\approx 0.64
$$

and normalizes each element:

$$
\\hat{x}_i = \\frac{x_i - \\mu}{\\sigma + \\epsilon}.
$$

So the normalized output becomes approximately

$$
\\hat{x} = [-0.20,\\; -0.98,\\; 1.68,\\; -0.49].
$$

&nbsp;

---

&nbsp;

$$\\textbf{Why It's Used in GPT-Style Models}\\\\[4pt]$$

&nbsp;

$$\\textbf{1. Stabilizes training by controlling magnitude}\\\\[2pt]$$  
After $x \\rightarrow \\hat{x} = \\frac{x - \\mu}{\\sigma}$, the softmax in attention sees $\\frac{QK^\\top}{\\sqrt{d}}$ on a stable scale,
reducing exploding logits and saturated softmax; MLP nonlinearity also operate in a stable range.

&nbsp;

$\\href{#info:stabilize_training_notes}{\\text{Why is it? A tiny numeric example}}$

&nbsp;

$$\\textbf{2. Improves gradient flow}\\\\[4pt]$$  
The Jacobian around normalized activations is better conditioned (roughly scaled by 
$\\frac{1}{\\sigma}$
), which curbs exploding/vanishing gradients and reduces sensitivity to learning-rate choice.

&nbsp;

$\\href{#info:improves_gradient_flow_notes}{\\text{Why is it? A tiny numeric example}}$

&nbsp;

$$\\textbf{3. Provides scale invariance}\\\\[4pt]$$  
rescaling upstream weights changes $\\mu$ and $\\sigma$ similarly, so the normalized activations barely change.
the model learns "directional" features while LayerNorm's $\\gamma$ and $\\beta$ reintroduce any useful scale/shift

&nbsp;

$\\text{Quick numeric example:}$

Let $x = \\begin{bmatrix}2, -1, 1\\end{bmatrix}$ scale it by $c = 10$ to $x' = \\begin{bmatrix}20, -10, 10\\end{bmatrix}$.

$\\bullet\\ $ Without LN: x' is 10x larger -> Q/K dot products and logis become ~10x larger -> likely softmax saturation.

$\\bullet\\ $ With LN: $\\hat{x}' = \\hat{x}$. Forward behavior after LN is the same, backprop through LN also accounts for $\\frac{1}{\\sigma}$, 
so gradients aren't inflated by the scale.

&nbsp;

$\\textbf{Summary:}$

The above three points are related, but not the same. Think of the three points as tackling stability from three different angles:

&nbsp;

$\\textbf{1. Forward magnitude control (to avoid softmax saturation):}$
LN standardizes each token's vector so dot products/logits don't blow up $\\Rightarrow$ softmax stays in a small range
$\\Rightarrow \\triangledown_z = p - y$ doesn't vanish. 
It targets the numerical size of forward activations after LN.

&nbsp;

$\\textbf{2. Gradient flow conditioning (LN's Jacobian):}$
In backprop, LN applies a specific linear map:
$$
  \\triangledown_{x} L = \\frac{1}{\\sigma} (g - \\mu(g)\\mathbf{1} - \\hat{y}\\mu(g\\odot \\hat{y}))
$$
The $\\frac{1}{\\sigma}$ scales and the two projections directly shape gradients, reducing exploding/vanishing.
It targets the local conditioning of gradients at LN during back prop - independent of upstream rescaling and even out any upstreaming component magnitudes. 
Intuitively, no matter what gradient arrives at LN, it reshapes it into a well conditioned, scale-controlled signal before sending it further back.

&nbsp;

$\\textbf{3. Scale invariance (decouples direction from scale):}$
Upstream scaling won't affect LN's output since $\\mu$ and $\\sigma$ scale accordingly.
It targets the invariance to arbitrary rescaling from earlier layers.

&nbsp;

---

&nbsp;

$\\textbf{4. More notes:}$

&nbsp;

$\\textbf{1.}$ The layer norm used in GPT-2 is also called Pre-norm, which is different from the Post-norm used in the original paper.
Pre-norm vs post-norm is just about where you put LayerNorm relative to the residual add in the Transformer block.

&nbsp;

$\\textbf{The two layouts:}\\\\[4pt]$

$\\textbf{Post-norm (original paper):}$

$$
SA: x_1 = LN(x + MHA(x))\\\\[4pt]
FFN: y = LN(x_1 + FFN(x_1))
$$

Here LN normalizes after residual addition.

&nbsp;

$\\textbf{Pre-norm (widely used now for deep stacks includingGPT-2):}$

$$
SA: x_1 = x + MHA(LN(x))\\\\[4pt]
FFN: y = x_1 + FFN(LN(x_1))
$$

Here LN is before each sublayer.

&nbsp;

The main reason Pre-norm is widely used now is that there is an identity path for gradients through the residual: 
$\\frac{\\partial y}{\\partial x} = I + ...$. This makes deep stacks train stably.

&nbsp;

**See a tiny numeric $\\href{#info:post_norm_gradient_example_notes}{example}$ to illustrate the issue of Post-norm.**

&nbsp;

$\\textbf{2.}$ Amone recent open weight LLMs, RMSNorm is the default more often than LayerNorm. Examples that explicitly use RMSNorm include:

&nbsp;

$\\textbf{LLaMA(v2)}:$ 
The paper explicitly says: "we … apply pre-normalization using RMSNorm …" (Sec. 2.2 Training Details).
$\\href{https://ar5iv.labs.arxiv.org/html/2307.09288}{\\texttt{LLaMA(v2)}}\\\\[4pt]$

$\\textbf{LLaMA(v3)}:$
Meta's implementation and community annotated walkthroughs of the official code note RMSNorm in the stack (implementation detail consistent with Llama 2).
$\\href{https://ttumiel.com/blog/LLaMA3/?utm_source=chatgpt.com}{\\texttt{Tom Tumiel}}$

$\\textbf{Mistral 7B}:$
Hugging Face's official Mistral model code defines and uses MistralRMSNorm.
$\\href{https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py?utm_source=chatgpt.com}{\\texttt{GitHub}}$

$\\textbf{Mixtral}:$
HF's Mixtral implementation (derived from Mistral codebase) uses RMSNorm; more broadly, normalization surveys/papers state "RMSNorm is used by many LLMs such as Llama and Mistral."
$\\href{https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py?utm_source=chatgpt.com}{\\texttt{GitHub}}$

$\\textbf{Qwen}:$
model cards list the architecture as "Transformers with RoPE, SwiGLU, RMSNorm …" (example: Qwen 2.5-1.5B).
$\\href{https://huggingface.co/Qwen/Qwen2.5-1.5B?utm_source=chatgpt.com}{\\texttt{Hugging Face}}$

$\\textbf{Gemma/Gemma2}:$
HF's official Gemma docs describe the model as using SwiGLU and RMSNorm layer normalization.
$\\href{https://huggingface.co/docs/transformers/en/model_doc/gemma?utm_source=chatgpt.com}{\\texttt{Hugging Face}}$

&nbsp;

**See a tiny numeric $\\href{#info:rmsnorm_gradient_example_notes}{example}$ to illustrate how RMSNorm works.**

&nbsp;

---

&nbsp;

$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. Original paper — Layer Normalization (2016).}$$
Ba, Kiros, Hinton. Defines LN, contrasts with BatchNorm, and motivates per-example normalization.
$$\\href{https://arxiv.org/pdf/1607.06450}{\\texttt{Layer Normalization}}$$

$$\\textbf{2. GPT-2 technical report — where LN sits in GPT-2 (Pre-LN).}$$
Radford et al. note LN is moved to the input of each sub-block (pre-activation) and add a final LN.
$$\\href{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com}{\\texttt{GPT-2 Technical Report}}$$

$$\\textbf{3. PyTorch API — exact math \\& args used in practice.}$$
$\\mathtt{torch.nn.LayerNorm(normalized\\_shape, \\epsilon, elementwise\\_affine, …)}$—helpful for implementation details and the 
$\\epsilon$ stabilization you'll see in configs.
$$\\href{https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html}{\\texttt{torch.nn.LayerNorm}}$$

$$\\textbf{4. Pre-LN vs Post-LN analysis (why GPT-2 uses Pre-LN).}$$
Xiong et al., On Layer Normalization in the Transformer Architecture—theoretical & empirical comparison; shows Pre-LN stabilizes training (often no warm-up).
$$\\href{https://arxiv.org/pdf/2002.04745}{\\texttt{On Layer Normalization in the Transformer Architecture}}$$

$$\\textbf{5. RMSNorm}$$
Zhang & Sennrich introduce RMSNorm, a lighter alternative (no centering) with similar quality and speed gains—useful contrast to LN.
$$\\href{https://arxiv.org/pdf/1910.07467}{\\texttt{RMSNorm}}$$
`,
  },

  stabilize_training_notes: {
    title: "Stabilize Training Example",
    md: `
  Here's a tiny numeric example to show why LayerNorm stabilizes training by controlling magnitude (so attention logits don't blow up and softmax doesn't saturate).

  &nbsp;

  $\\textbf{Setup (2 tokens, d=3, use identity projections so Q = X, K = X, scale by 1/sqrt(d))}$

  Raw embeddings:

  $$
  e_1 = \\begin{bmatrix}10 & -5 & 7\\end{bmatrix}, e_2 = \\begin{bmatrix}-8 & 12 & -4\\end{bmatrix}
  $$

  $\\textbf{1) Without LayerNorm:}$

  Scores for token 1 attending to {1,2}:

  $$
  \\bullet\\ s_{11} = \\frac{e_1 \\cdot e_1}{\\sqrt{3}} = \\frac{10 \\cdot 10 + (-5) \\cdot (-5) + 7 \\cdot 7}{\\sqrt{3}} = \\frac{100 + 25 + 49}{\\sqrt{3}} = \\frac{174}{\\sqrt{3}} \\approx 100.46\\\\[4pt]
  \\bullet\\ s_{12} = \\frac{e_1 \\cdot e_2}{\\sqrt{3}} = \\frac{10 \\cdot (-8) + (-5) \\cdot 12 + 7 \\cdot (-4)}{\\sqrt{3}} = \\frac{-80 - 60 - 28}{\\sqrt{3}} = \\frac{-168}{\\sqrt{3}} \\approx -96.99
  $$

  Softmax over [100.46, -96.99] $\\Rightarrow \\approx$ [1.00, 0.00], totally saturated which will lead to vanishing gradients.
  (Why? Step by Step explanation [here](#info:stabilize_training_no_layernorm_backprop_notes))

  &nbsp;

  $\\textbf{2) With LayerNorm:}$

  Normalize each token (numbers rounded):

  $$
  \\bullet\\ \\text{For } e_1: mean = 4, std \\approx 6.4807 \\\\[4pt]
  \\hat{e}_1 = \\begin{bmatrix}0.9258, -1.3887, 0.4630\\end{bmatrix}\\\\[4pt]
  \\bullet\\ \\text{For } e_2: mean = 0, std \\approx 8.6403 \\\\[4pt]
  \\hat{e}_2 = \\begin{bmatrix}-0.9258, 1.3887, -0.4630\\end{bmatrix}
  $$

  Scores for token 1 attending to {1,2}:

  $$
  \\bullet\\ s_{11} = \\frac{\\hat{e}_1 \\cdot \\hat{e}_1}{\\sqrt{3}} = \\frac{0.9258 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.4630 \\cdot 0.4630}{\\sqrt{3}} \\approx 1.732\\\\[4pt]
  \\bullet\\ s_{12} = \\frac{\\hat{e}_1 \\cdot \\hat{e}_2}{\\sqrt{3}} = \\frac{0.9258 \\cdot (-0.9258) + (-1.3887) \\cdot 1.3887 + 0.4630 \\cdot (-0.4630)}{\\sqrt{3}} \\approx -1.732
  $$
  
  Softmax over [1.732, -1.732] $\\Rightarrow \\approx$ [0.9696, 0.0304], not saturated which will not lead to vanishing gradients.
  
  &nbsp;

  $\\textbf{Bottom line:}\\\\[4pt]$
  LN standardizes activations so during forward pass, downstream dot-products/logits are in a reasonable numeric range → softmax avoids saturation → probabilities remain sensitive to changes.
  `,
  },

  stabilize_training_no_layernorm_backprop_notes: {
    title: "Why satuated softmax leads to vanishing gradients",
    md: `
  
  $\\textbf{1. Softmax from logits to probabilities:}\\\\[4pt]$

  The logits from the example: $z=[100.46,-96.99]\\\\[4pt]$

  We also have after softmax: $p \\approx [1.00, 0.00]$ up to an $\\epsilon$ of about $10^{-86}$.
  
  &nbsp;

  $\\textbf{2. Cross-entropy loss and its gradient w.r.t logits:}\\\\[4pt]$

  For a one-hot target $y$ (say the true class is 1: $y=[1,0]$), the cross-entropy loss is:
  $$
  L = -\\sum_i y_i \\log p_i = -log(p_1)
  $$
  
  The gradient of the loss w.r.t logits is:
  $$
  \\frac{\\partial L}{\\partial z_i} = p_i - y_i
  $$
  
  Read the mathematical proof $\\href{#info:gradient_loss_to_logits_proof_notes}{\\text{here}}$.

  &nbsp;

  Plugging in the values:
  $$
  \\bullet\\ P_1 \\approx 1 - 10^{-86}, y_1 = 1 \\Rightarrow \\frac{\\partial L}{\\partial z_1} \\approx (1 - 10^{-86}) - 1 = -10^{-86} \\\\[4pt]
  \\bullet\\ P_2 \\approx 10^{-86}, y_2 = 0 \\Rightarrow \\frac{\\partial L}{\\partial z_2} \\approx 10^{-86} - 0 = 10^{-86}\\\\[4pt]
  \\Rightarrow \\triangledown_z L = [-10^{-86}, 10^{-86}] \\; \\textbf{essentially zero in magnitude.}\\\\[4pt]
  \\text{Intuition: when softmax is saturated and confidently correct, the loss surface is almost flat, so the gradient is near 0.}
  $$

  $\\textbf{3. Backprop to earlier layers becomes tiny:}\\\\[4pt]$
  Suppose logits came from a linear layer: $z = Wh + b$ (where $h$ is the previous layer's activations). Then
  $$
  \\frac{\\partial L}{\\partial W} = (\\triangledown_z L)h^\\top \\\\[4pt]
  \\frac{\\partial L}{\\partial h} = W^\\top (\\triangledown_z L) \\\\[4pt]
  \\frac{\\partial L}{\\partial b} = \\triangledown_z L \\\\[4pt]
  $$
  Because $\\triangledown_z L$ has entries on the order of $10^{-86}$, all the weight gradients and the gradient flowing back into $h$ become $10^{-86}$ in magnitude.
  That means earlier layers receive essentially no learning signal.
  
  &nbsp;
  
  $\\textbf{4. Some thoughts:}\\\\[4pt]$
  $\\bullet\\ $ Early in training, we don't want logits so huge that softmax saturates, because we stop getting useful gradients to shape the representations.
  
  $\\bullet\\ $ If the model is $\\textbf{confident but wrong}$ (e.g, p $\\approx$ [0,1] while y = [1,0]), then $\\triangledown_z L \\approx$ [-1,1] - the network does 
  get a strong corrective signal. Vanishing happens in the "confidently correct" case.
  
  $\\bullet\\ $ $\\href{#info:label_smoothing_notes}{\\text{Label smoothing}}$ keeps gradients from collapsing completely even when predictions are confident.
  
  &nbsp;

  $\\textbf{5. References:}\\\\[4pt]$
  `,
  },

  label_smoothing_notes: {
    title: "Label Smoothing",
    md: `
Label smoothing replaces the one-hot target with a soft target so the gradient 
$\\triangledown_z L = p - y$
never becomes exactly zero—even when the model is extremely confident.

&nbsp;

$\\bullet\\ $ Standard (one-hot) CE: if the model is right and saturated, 
$$
p \\approx [1,0, \\ldots] \\text{ and } y = [1,0, \\ldots] \\Rightarrow p - y \\approx [0,0, \\ldots] \\Rightarrow \\text{near-zero gradient (collapse).}
$$

$\\bullet\\ $ With label smoothing $\\epsilon$: set the target to
$$
y_c^{LS} = 1 - \\epsilon,\\; \\;\\; y_{j\\neq c}^{LS} = \\frac{\\epsilon}{C-1} \\\\[4pt]
$$
Then, even if $p \\approx [1,0, \\ldots]$, we get
$$
\\triangledown_z L = p - y^{LS} \\approx [\\epsilon, \\frac{\\epsilon}{C-1}, \\ldots]
$$
which is non-zero. So logits still receive a useful updates. Intuitively, label smoothing reduces overconfidence by nudging the top logit 
down a bit and pushing up the others slightly.
This prevesnts complete gradient shutdown when the model is already confident, keeping training dynamics active and reducing brittle overfitting.

$$\\textbf{Tiny example (C=3, $\\epsilon=0.1$):}\\\\[4pt]$$
$\\bullet\\ $ Smoothed targeet: $y^{LS} = [0.9, 0.05, 0.05]\\\\[4pt]$
$\\bullet\\ $ If $p = [0.999,0.0005,0.0005] (overconfident),\\\\[4pt]$.
$\\triangledown_z L = [0.999 - 0.9, 0.0005 - 0.05, 0.0005 - 0.05] = [0.099, -0.0495, -0.0495] \\; \\text{non-zero, weights still get updated.}$

&nbsp;

$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. Original paper:}$$
Rethinking the Inception Architecture for Computer Vision (Inception-v3); introduces and motivates label smoothing as a regularizer
$$\\href{https://arxiv.org/pdf/1512.00567}{\\texttt{Rethinking the Inception Architecture for Computer Vision}}$$

$$\\textbf{2. Transformer training detail:}$$
Attention Is All You Need—explicitly states they use label smoothing with 
$\\epsilon=0.1$ and notes its effects.
$$\\href{https://arxiv.org/pdf/1706.03762v2}{\\texttt{Attention Is All You Need}}$$

$$\\textbf{3. Theory \\& empirical study:}$$
When Does Label Smoothing Help? (Müller, Kornblith, Hinton, NeurIPS 2019) — effects on generalization, calibration, and why it can hurt distillation.
$$\\href{https://arxiv.org/pdf/1906.02629}{\\texttt{When Does Label Smoothing Help?}}$$

$$\\textbf{4. Related regularizer (confidence penalty) and connection to LS:}$$
Pereyra et al., ICLR 2017 workshop—penalizing low-entropy outputs; shows relation to label smoothing via direction of KL divergence.
$$\\href{https://arxiv.org/pdf/1701.06548}{\\texttt{Regularizing Neural Networks by Penalizing Confident Output Distributions}}$$

$$\\textbf{5. Calibration context:}$$
Guo et al., On Calibration of Modern Neural Networks (ICML 2017)—why model confidence is not calibrated and how label smoothing helps.
$$\\href{https://arxiv.org/pdf/1706.04599}{\\texttt{On Calibration of Modern Neural Networks}}$$

`,
  },
  
  gradient_loss_to_logits_proof_notes: {
    title: "Gradient of loss to logits",
    md: `
  $\\textbf{Setup:}\\\\[4pt]$
  $$
  \\bullet\\ \\text{Logits: } z \\in \\mathbb{R}^C\\\\[4pt]
  \\bullet\\ \\text{Softmax: } p_i = \\frac{e^{z_i}}{\\sum_{k=1}^C e^{z_k}}\\\\[4pt]
  \\bullet\\ \\text{Target: } y \\text{(usually one-hot, but can be any distribution for label smoothing)}\\\\[4pt]
  \\bullet\\ \\text{Cross-entropy loss: } L = -\\sum_{i=1}^C y_i \\log p_i
  $$

  $\\textbf{Proof:}\\\\[4pt]$

  Use the identity
  
  $$
  \\log p_i = z_i - \\log \\sum_{k=1}^C e^{z_k}
  $$

  Plug it into the loss:

  $$
  L = -\\sum_{i=1}^C y_i \\log p_i = -\\sum_{i=1}^C y_i (z_i - \\log \\sum_{k=1}^C e^{z_k}) = -\\sum_{i=1}^C y_i z_i + \\sum_{i=1}^C y_i \\log \\sum_{k=1}^C e^{z_k}
  $$

  Because $y$ is a probability distribution, $\\sum_{i=1}^C y_i = 1$, so

  $$
  L = -\\sum_{i=1}^C y_i z_i + \\log \\sum_{k=1}^C e^{z_k}
  $$

  Differentiate w.r.t $z_j$:
  
  $$
  \\frac{\\partial L}{\\partial z_j} = -y_j + \\frac{e^{z_j}}{\\sum_{k=1}^C e^{z_k}} = p_j - y_j
  $$
  `,
  },

  improves_gradient_flow_notes: {
    title: "Improves gradient flow",
    md: `
  Here is a tiny numeric example to compare the gradient w.r.t. the pre-LN activations with and without LN.

  &nbsp;

  $\\textbf{Setup:}\\\\[4pt]$ a single token vector of size n = 3
  $$
  x = \\begin{bmatrix}10, -5, 7\\end{bmatrix}
  $$

  Mean and standard deviation:
  $$
  \\mu = \\frac{10 + (-5) + 7}{3} = 4, \\quad \\sigma = \\sqrt{\\frac{(10-4)^2 + (-5-4)^2 + (7-4)^2}{3}} = 6.4807
  $$

  To make the example simple, let a simple linear head follow with zero target:
  
  $W = diag(3, 1, 0.5)$ and $b = 0$. Use a squared loss to the zero target:
  
  $$
  L = \\frac{1}{2} ||Wy + b - 0||^2
  $$

  where $y$ is eitehr the identity output without LN or the LN output.

  &nbsp;
  
  ---
  
  &nbsp;

  $\\textbf{1) Without LN:}\\\\[4pt]$
  Because there is no LN, so it's identity pass: $y = x$.
  
  $\\bullet\\ $ $\\text{Forward: } Wy = diag(3, 1, 0.5) \\cdot x = \\begin{bmatrix}30, -5, 3.5\\end{bmatrix}\\\\[4pt]$
  $\\bullet\\ $ $\\text{Gradient w.r.t. y: } \\triangledown_y L = W^\\top (Wy) = W^2{y}\\\\[4pt]$
  $\\bullet\\ $ $\\text{Since W is diagonal, } W^2 = diag(9, 1, 0.25)\\\\[4pt]$
  $\\bullet\\ $ $\\text{Gradient w.r.t. input x (here y = x): } \\triangledown_x L = W^2{x} = \\begin{bmatrix}9*10, 1*(-5), 0.25*7\\end{bmatrix} = \\begin{bmatrix}90, -5, 1.75\\end{bmatrix}\\\\[4pt]$

  This gradient is large and very uneven—dominated by the first coordinate. If upstream layers rescale 
  $x$ by some factor $c$, the gradient scales by $c$ as well (learning rate becomes sensitive to scale).

  &nbsp;

  ---

  &nbsp;

  $\\textbf{2. With LN (pre-LN, GPT-2 style):}\\\\[4pt]$
  First compute the normalized vector:
  $$
  \\hat{y} = LN(x) = \\frac{x - \\mu}{\\sigma} \\approx \\begin{bmatrix}0.9258, -1.3887, 0.4630\\end{bmatrix}
  $$

  Forward:
  $$
  W\\hat{y} = diag(3, 1, 0.5)\\hat{y}. 
  $$

  Gradient w.r.t. $\\hat{y}$:
  $$
  g = \\triangledown_{\\hat{y}} L = W^\\top (W\\hat{y}) = W^2\\hat{y} = diag(9, 1, 0.25)\\hat{y} = \\begin{bmatrix}8.3322, -1.3887, 0.1158\\end{bmatrix}
  $$

  Now backprop through the LN. For LN over a vector (affine removed for simplicity), the closed form is:
  $$
  \\triangledown_{x} L = \\frac{1}{\\sigma} (g - \\mu(g)\\mathbf{1} - \\hat{y}\\mu(g\\odot \\hat{y}))
  $$

  where $\\mathbf{1}$ is a vector of all ones, $\\mu(g)$ is the mean of $g$ and $\\odot$ is the elementwise product.

  Compute the two mean terms:
  $$
  \\mu(g) = \\frac{8.3322 + (-1.3887) + 0.1158}{3} \\approx 2.3532\\\\[4pt]
  \\mu(g\\odot \\hat{y}) = \\frac{8.3322 \\cdot 0.9258 + (-1.3887) \\cdot (-1.3887) + 0.1158 \\cdot 0.4630}{3} = 3.2325
  $$

  Assemble together:
  $$
  g - \\mu(g)\\mathbf{1} \\approx \\begin{bmatrix}5.979, -3.7417, -2.2372\\end{bmatrix}\\\\[4pt]
  \\hat{y}\\mu(g\\odot \\hat{y}) \\approx \\begin{bmatrix}2.990, -4.487, 1.497\\end{bmatrix}
  $$

  Subtract and scale by $\\frac{1}{\\sigma}$:
  $$
  \\triangledown_{x} L = \\frac{1}{6.4807} (\\begin{bmatrix}2.989, 0.7453, -3.7342\\end{bmatrix}) \\approx \\begin{bmatrix}0.461, 0.115, -0.576\\end{bmatrix}
  $$

  Now the gradient is much more even—no single dimension dominates.
  
  `,
  },

  post_norm_gradient_example_notes: {
    title: "Post-norm gradient example",
    md: `

  We'll use a single residual block in $\\textbf{Post-norm}$ form to illustrate the issue.

  $$
  \\text{(post-norm) } \\mathtt{z = x + f(x), \\quad y = LN(z)} \\\\[4pt]
  $$

  For simplicity, we assume the following settings:
  $$
  \\bullet\\ \\text{hidden size m = 2, so LN normalizes across 2 features}\\\\[4pt]
  \\bullet\\ \\text{input x = [1, -1] (mean = 0, variance = 1)}\\\\[4pt]
  \\bullet\\ \\text{a simple residual f(x) = ax with a = 2 (amplifying the residual path by factor of 2)}
  $$

  $\\textbf{Forward pass:}\\\\[4pt]$
  $$
  \\bullet\\ \\text{Residual add: } z = x + f(x) = x + 2x = 3x = [3, -3]\\\\[4pt]
  \\bullet\\ \\text{LayerNorm over features:}\\\\[4pt]
  \\ \\ \\ \\ \\bullet\\ mean \\ \\mu = (3 + (-3)) / 2 = 0\\\\[4pt]
  \\ \\ \\ \\ \\bullet\\ variance \\ v = ((3 - 0)^2 + (-3 - 0)^2) / 2 = 9, std \\ \\sigma = 3\\\\[4pt]
  \\ \\ \\ \\ \\bullet\\ y = LN(z) =(z - \\mu) / \\sigma = [3, -3]/3 = [1, -1].
  $$

  The forward pass is fine.

  &nbsp;

  $\\textbf{Backward pass (the problem!):}\\\\[4pt]$
  For LayerNorm (per-feature LN with m features), the Jacobian at the point is has the following closed form $\\href{#info:ln_jacobian_closed_form_proof_notes}{proof}$:

  $$
  J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} (\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top),
  $$

  Where $\\mathbf{I}$ is the identity matrix of size m $\\times$ m. $\\mathbf{1}$ is the vector of all ones in $\\mathbb{R}^m$.

  &nbsp;

  Plug in the values:

  Here $m = 2, \\sigma = 3, y = [1, -1]$.
  $$
  \\bullet\\ \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top = \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top = \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix}\\\\[4pt]
  \\bullet\\ \\frac{1}{m} yy^\\top = \\frac{1}{2} \\frac{1}{2} yy^\\top = \\frac{1}{2} \\begin{bmatrix}1 & -1\\\\-1 & 1\\end{bmatrix} = \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix}\\\\[4pt]
  \\bullet\\ \\mathbf{I} - \\frac{1}{2} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{2} \\frac{1}{2} yy^\\top = \\mathbf{I} - \\begin{bmatrix}0.5 & 0.5\\\\0.5 & 0.5\\end{bmatrix} - \\begin{bmatrix}0.5 & -0.5\\\\-0.5 & 0.5\\end{bmatrix} = \\begin{bmatrix}0 & 0\\\\0 & 0\\end{bmatrix}
  $$

  Thus $J = \\frac{1}{\\sigma} \\times 0 = 0$.

  &nbsp;

  $\\textbf{Interpretation: }$ at this configuration, LayerNor's Jacobian is exactly zero. That means for any upstream gradient $\\triangledown_y L$,
  the gradient propagated back to $z$ is
  $$
  \\triangledown_z L = J^\\top \\triangledown_y L = 0.
  $$ 
  So the gradient dies completely before it even reaches the residual add.
  Also note the $\\frac{1}{\\sigma}$ in the Jacobian, with more layers, many such Jacobians chain up, so even when not exactly zero,
  the repeated products of $\\frac{1}{\\sigma}$ can still shrink the gradient to near zero.

  &nbsp;

  In pre-norm, you'd compute $y = x + f(LN(x))$. The sublayer always sees unit-scale inputs, and the LN Jacobian is applied before the residual add.
  So there's always a "gradient highway". Even if the residual branch's Jacobian goes to zero, the "gradient highway" still carries gradient through.
  
  `,
  },

  rmsnorm_gradient_example_notes: {
    title: "RMSNorm gradient example",
    md: `
  A tiny numeric example to illustrate how RMSNorm works.

  &nbsp;

  The formula of RMSNorm is:
  
  $$
  RMSNorm(x) = \\gamma \\cdot \\frac{x}{\\sqrt{\\frac{1}{d} \\sum_{i}x_i^2 + \\epsilon}}
  $$

  where $\\gamma$ is a learnable scaling factor we'll talk about what it's used for later.
  $\\epsilon$ is a small constant to avoid division by zero.

  &nbsp;

  $\\textbf{Setup:}\\\\[4pt]$
  $$
  \\text{Given } x = \\begin{bmatrix}2, -1, 3, 0\\end{bmatrix}, \\gamma = [1.0, 0.5, 2.0, 1.5]
  $$
  So in this example, $d = 4$, we'll also ignore the $\\epsilon$ term for simplicity.

  &nbsp;

  $\\textbf{Step 1 - compute the RMS):}\\\\[4pt]$
  $$
  RMS(x) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2} = \\sqrt{\\frac{1}{4} (2^2 + (-1)^2 + 3^2 + 0^2)} = \\sqrt{\\frac{1}{4} (4 + 1 + 9 + 0)} = \\sqrt{\\frac{14}{4}} = \\sqrt{3.5} \\approx 1.8708
  $$

  $\\textbf{Step 2 - normalize by the RMS:}\\\\[4pt]$
  $$
  x = \\frac{x}{RMS(x)} = \\begin{bmatrix}2/1.8708, -1/1.8708, 3/1.8708, 0/1.8708\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.5345, 1.6035, 0.0000\\end{bmatrix}
  $$

  $\\textbf{Step 3 - apply per-feature scaling:}\\\\[4pt]$
  $$
  y = \\begin{bmatrix}1.0690 \\cdot 1.0, -0.5345 \\cdot 0.5, 1.6035 \\cdot 2.0, 0.0000 \\cdot 1.5\\end{bmatrix} = \\begin{bmatrix}1.0690, -0.2673, 3.2070, 0.0000\\end{bmatrix}
  $$

  This is the RMSNorm output.

  &nbsp;

  ---
  
  &nbsp;

  $\\textbf{Why do we need scale factor } \\gamma: \\\\[4pt]$
  Pure normalization fixes the output of RMS to $\\approx 1$, which is too restrictive and not ideal.
  $\\gamma$ let the model learn and reshape that constraint per feature. Different features can be scaled
  up/down as training discovers which directions are useful.
  `,
  },

  post_norm_scaling_example_notes: {
    title: "Post-norm scaling example",
    md: `
  In this practice, we'll first see an example that under Post-normwithout scaling, the gradient vanishes with depth.
  Then we'll add scaling and show that the gradient does not vanish with depth.
  Finally we'll see an example of even without scaling, gradient still doesn't vanish with Pre-norm.
  
  &nbsp;

  $\\textbf{Setup (post-norm)}$

  In order to make the example simple, we'll use a 2-D LayerNorm and a super simple residual branch tha simply 
  $\\textbf{adds a fixed vector u}$ each layer so we can see the growth cleanly.

  &nbsp;

  For each layer $l$:
  $$
  z_l = x_{l-1} + \\alpha u
  $$

  where

  $$
  \\bullet \\ \\text{Feature dimension m = 2}\\\\[4pt]
  \\bullet \\ \\text{Start with }x_0 = [0, 0]\\\\[4pt]
  \\bullet \\ \\text{Fixed residual vector }u = [1, 0] - \\text{a fixed unit step on the first feature, just to keep arithmetic simple}.\\\\[4pt]
  \\bullet \\ \\text{LayerNorm uses }\\gamma=1, \\beta=0 - \\text{no shift/scale}.\\\\[4pt]
  \\bullet \\ \\text{Let }\\sigma(z) = \\sqrt{\\frac{z_1^2 + z_2^2}{2}} = ||z||/\\sqrt{2} - \\text{the L2 norm of z}.\\\\[4pt]
  \\bullet \\ \\text{An informative assumption for the per-layer gradient attenuation through LN is }\\approx 1/\\sigma(z_l).\\\\[2pt]
  \\text{(LN's Jacobian operator norm scales up to }\\approx 1/\\sigma(z_l)).
  \\href{#info:ln_gradient_attenuation_upper_bound_proof_notes}{Why} \\\\[4pt]
  $$

  With this setup, we'll compare $\\textbf{no scaling }(\\alpha = 1)$ vs $\\textbf{depth scaling for N = 4 layers }(\\alpha = 1/\\sqrt{N} = 0.5)$.
  
  &nbsp;

  $\\textbf{Case 1: no scaling } \\alpha = 1, N = 4:\\\\[4pt]$

  Layer 1
  $$
  \\bullet \\ z_1 = x_0 + \\alpha u = [0, 0] + 1 \\times [1, 0] = [1, 0]\\\\[4pt]
  \\bullet \\ \\mu_1 = (1 + 0)/2 = 0.5\\\\[4pt]
  \\bullet \\ \\sigma_1 = \\sqrt{(1-0.5)^2 + (0-0.5)^2)/2} = 0.5\\\\[4pt]
  \\bullet \\ x_1 = (z_1 - \\mu_1)/\\sigma_1 = [1, -1]
  $$

  Layer 2
  $$
  \\bullet \\ z_2 = x_1 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1]\\\\[4pt]
  \\bullet \\ \\mu_2 = (2 + (-1))/2 = 0.5\\\\[4pt]
  \\bullet \\ \\sigma_2 = \\sqrt{(2-0.5)^2 + (-1-0.5)^2)/2} = 1.5\\\\[4pt]
  \\bullet \\ x_2 = (z_2 - \\mu_2)/\\sigma_2 = [1.5, -1.5]/1.5 = [1, -1]
  $$

  Layer 3
  $$
  \\bullet \\ z_3 = x_2 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1]\\\\[4pt]
  \\bullet \\ \\mu_3 = (2 + (-1))/2 = 0.5\\\\[4pt]
  \\bullet \\ \\sigma_3 = \\sqrt{(2-0.5)^2 + (-1-0.5)^2)/2} = 1.5\\\\[4pt]
  \\bullet \\ x_3 = (z_3 - \\mu_3)/\\sigma_3 = [1.5, -1.5]/1.5 = [1, -1]
  $$

  Layer 4
  $$
  \\bullet \\ z_4 = x_3 + \\alpha u = [1, -1] + 1 \\times [1, 0] = [2, -1]\\\\[4pt]
  \\bullet \\ \\mu_4 = (2 + (-1))/2 = 0.5\\\\[4pt]
  \\bullet \\ \\sigma_4 = \\sqrt{(2-0.5)^2 + (-1-0.5)^2)/2} = 1.5\\\\[4pt]
  \\bullet \\ x_4 = (z_4 - \\mu_4)/\\sigma_4 = [1.5, -1.5]/1.5 = [1, -1]
  $$

  Accumulate the per-layer factors:
  $$
  \\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\times \\frac{1}{1.5} \\approx 0.5926
  $$

  Gradients get shrunk by about 40% by the four LNs and would keep shrinking with more layers.
  
  &nbsp;

  $\\textbf{Case 2: depth scaling } \\alpha = 0.5, N = 4:\\\\[4pt]$

  Layer 1
  $$
  \\bullet \\ z_1 = x_0 + \\alpha u = [0, 0] + 0.5 \\times [1, 0] = [0.5, 0]\\\\[4pt]
  \\bullet \\ \\mu_1 = (0.5 + 0)/2 = 0.25\\\\[4pt]
  \\bullet \\ \\sigma_1 = \\sqrt{(0.5-0.25)^2 + (0-0.25)^2)/2} = 0.25\\\\[4pt]
  \\bullet \\ x_1 = (z_1 - \\mu_1)/\\sigma_1 = [0.25, -0.25]/0.25 = [1, -1]
  $$

  Layer 2
  $$
  \\bullet \\ z_2 = x_1 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1]\\\\[4pt]
  \\bullet \\ \\mu_2 = (1.5 + (-1))/2 = 0.25\\\\[4pt]
  \\bullet \\ \\sigma_2 = \\sqrt{(1.5-0.25)^2 + (-1-0.25)^2)/2} = 1.25\\\\[4pt]
  \\bullet \\ x_2 = (z_2 - \\mu_2)/\\sigma_2 = [1.25, -1.25]/1.25 = [1, -1]
  $$

  Layer 3
  $$
  \\bullet \\ z_3 = x_2 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1]\\\\[4pt]
  \\bullet \\ \\mu_3 = (1.5 + (-1))/2 = 0.25\\\\[4pt]
  \\bullet \\ \\sigma_3 = \\sqrt{(1.5-0.25)^2 + (-1-0.25)^2)/2} = 1.25\\\\[4pt]
  \\bullet \\ x_3 = (z_3 - \\mu_3)/\\sigma_3 = [1.25, -1.25]/1.25 = [1, -1]
  $$

  Layer 4
  $$
  \\bullet \\ z_4 = x_3 + \\alpha u = [1, -1] + 0.5 \\times [1, 0] = [1.5, -1]\\\\[4pt]
  \\bullet \\ \\mu_4 = (1.5 + (-1))/2 = 0.25\\\\[4pt]
  \\bullet \\ \\sigma_4 = \\sqrt{(1.5-0.25)^2 + (-1-0.25)^2)/2} = 1.25\\\\[4pt]
  \\bullet \\ x_4 = (z_4 - \\mu_4)/\\sigma_4 = [1.25, -1.25]/1.25 = [1, -1]
  $$

  Accumulate the per-layer factors:
  $$
  \\prod_{l=1}^4 \\frac{1}{\\sigma_l} = \\frac{1}{0.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\times \\frac{1}{1.25} \\approx 2.05
  $$

  Chained LNs keep the gradient at a stable level without vanishing.
  `,
  },

  ln_jacobian_closed_form_proof_notes: {
    title: "LayerNorm Jacobian closed form proof",
    md: `

  Let's prove the closed form of Jacobian of LayerNorm as follows:

  $$
  J = \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} (\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top), \\text{ with } y = \\frac{z - \\mu}{\\sigma}
  $$

  $\\textbf{Setup:}\\\\[4pt]$
  For a vector $z \\in \\mathbb{R}^m:\\\\[4pt]$
  $$
  \\bullet \\ \\mu = \\frac{1}{m} \\mathbf{1}^\\top z \\text{ (feature mean)}\\\\[4pt]
  \\bullet \\ var = \\frac{1}{m} \\sum_{i} (z_i - \\mu)^2 = \\frac{1}{m} z^\\top z - \\mu^2\\\\[4pt]
  \\bullet \\ \\sigma = \\sqrt{var} \\text{ (feature standard deviation, ignore } \\epsilon \\text{ for simplicity)}\\\\[4pt]
  \\bullet \\ y = LN(z) = \\frac{z - \\mu}{\\sigma}
  $$

  We want the Jacobian $J = \\frac{\\partial y}{\\partial z}$.

  &nbsp;

  $\\textbf{First, let's derive some useful derivatives:}\\\\[4pt]$

  $\\text{1. Mean:}$
  $$
  \\frac{\\partial \\mu}{\\partial z} = \\frac{1}{m} \\mathbf{1}
  $$

  $\\text{2. Std:}$
  $$
  \\frac{\\partial var}{\\partial z} = \\frac{2}{m} z - 2\\mu\\frac{\\partial \\mu}{\\partial z} = \\frac{2}{m} (z - \\mu \\textbf{1}),\\\\[4pt]
  \\frac{\\partial \\sigma}{\\partial z} = \\frac{\\partial \\sigma}{\\partial var} \\times \\frac{\\partial var}{\\partial z} = \\frac{1}{2\\sigma} \\times \\frac{2}{m} (z - \\mu \\textbf{1}) = \\frac{1}{m} \\times \\frac{z - \\mu \\textbf{1}}{\\sigma} = \\frac{1}{m} \\times y
  $$

  $\\text{3. Numerator }z - \\mu \\textbf{1}$:
  $$
  \\frac{\\partial (z - \\mu \\textbf{1})}{\\partial z} = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top
  $$

  $\\textbf{Chain rule for } y = \\frac{z - \\mu \\textbf{1}}{\\sigma}:$
  $$
  \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma} \\times \\frac{\\partial (z - \\mu \\textbf{1})}{\\partial z} + (z - \\mu \\textbf{1}) \\times \\frac{\\partial (1/\\sigma)}{\\partial z}\\\\[4pt]
  $$

  Since $\\frac{\\partial (1/\\sigma)}{\\partial z} = -\\frac{1}{\\sigma^2} \\times \\frac{\\partial \\sigma}{\\partial z}$, we have
  $$
  \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma}(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top) - \\frac{z - \\mu \\textbf{1}}{\\sigma^2}(\\frac{1}{m} y)^\\top
  $$

  Now plug in $z = y\\sigma + \\mu \\textbf{1}$:
  $$
  \\frac{\\partial y}{\\partial z} = \\frac{1}{\\sigma}(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top) - \\frac{(y\\sigma + \\mu \\textbf{1}) - \\mu \\textbf{1}}{\\sigma^2}(\\frac{1}{m} y)^\\top = \\frac{1}{\\sigma}(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top) - \\frac{y}{\\sigma}(\\frac{1}{m} y)^\\top = \\frac{1}{\\sigma}(\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top)
  $$

  That's exactly the formula we wanted to prove.
  `,
  },

  ln_gradient_attenuation_upper_bound_proof_notes: {
    title: "Proof of gradient attenuation upper bound",
    md: `
  Let's prove the upper bound of the gradient attenuation of LayerNorm.

  &nbsp;

  As approved $\\href{#info:ln_jacobian_closed_form_proof_notes}{here}$, LayerNorm's Jacobian at a vector $z \\in \\mathbb{R}^m$ can be written as:

  $$
  J = \\frac{1}{\\sigma} (\\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top)\\\\[4pt]
  \\text{ with } y = \\frac{z - \\mu}{\\sigma}, \\mu = \\frac{1}{m} \\mathbf{1}^\\top z \\text{ and } \\sigma = \\sqrt{\\frac{1}{m} \\sum_{i=1}^m (z_i - \\mu)^2}\\\\[4pt]
  $$

  $\\bullet \\ $ The $1/\\sigma$ in front scales the Jacobian.$\\\\[4pt]$
  $\\bullet \\ $ The matrix in parentheses,
  $$
  P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top
  $$
  $P$ is an $\\textbf{orthogonal projection}$ onto the subspace orthogonal to both the all-ones vector $\\mathbf{1}$ and the vector $y$.
  Orthogonal projection matrix has the property that its 2-norm (a.k.a. the spectral/operator norm) $\\leq 1$, so $||P||_2 \\leq 1$ (See the related $\\href{#info:orthogonal_projection_notes}{math}$ here).

  &nbsp;

  Hence, the 2-norm of the Jacobian is bounded by:
  $$
  ||J||_2 = \\frac{1}{\\sigma} ||P||_2 \\leq \\frac{1}{\\sigma}
  $$

  Here we use the one important property of the 2-norm of a matrix:
  The 2-norm of a matrix represents the largest factor by which the matrix can stretch a vector's Euclidean length, over any direction. 
  Thus, $||J||_2$ is the directly upper-bounds $\\textbf{per-layer gradient attenuation/amplification}$.
  `
  },

  orthogonal_projection_notes: {
    title: "Orthogonal projection",
    md: `
  The matrix we are interested in is
  $$
  \\ \\ \\ P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top \\\\[4pt]
  \\bullet \\ \\mathbf{I} = \\text{identity matrix of size } m \\times m.\\\\[4pt]
  \\bullet \\ \\mathbf{1} = \\text{all-ones vector of size } m.\\\\[4pt]
  \\bullet \\ y = \\text{the LayerNorm output (zero mean, and } \\frac{1}{m} y^\\top y = 1).\\\\[4pt]
  $$

  $\\textbf{Theorem:}$
  If $P$ is [symmetric](#hover:orthogonal_projection_symmetric_notes) $(P^\\top = P)$ and [idempotent](#hover:orthogonal_projection_idempotent_notes) $(P^2 = P)$, then $P$ is an orthogonal projector to its range $Range(P)$ [Proof](#hover:orthogonal_projection_proof_notes).
  
  &nbsp;
  
  $\\textbf{A. Prove } P \\textbf{ is a orthogonal projector to the subspace that is orthogonal to both } \\mathbf{1} \\textbf{ and } y \\textbf{:}$
  
  &nbsp;

  Under the LayerNorm conditions, we have
  $$
  \\mathbf{1}^\\top y = 0,\\  y^\\top y = m
  $$

  Let
  $$
  A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top,\\ B = \\frac{1}{m} yy^\\top
  $$
  
  Define the subspace $S$ as the subspace that is orthogonal to both $\\mathbf{1}$ and $y$.
  $$
  S = \\{v \\in \\mathbb{R}^m: \\mathbf{1}^\\top v = 0, y^\\top v = 0\\}
  $$
  
  &nbsp;

  We'll prove the following facts:

  &nbsp;

  $\\textbf{1). } P \\textbf{ project any } z \\textbf{ into } S$

  &nbsp;

  Take any vector $z$ and set $w = Pz$. We'll show $w$ is orthogonal to both $\\mathbf{1}$ and $y$.
  $$
  \\bullet \\ \\text{Orthogonal to } \\mathbf{1}:\\\\[4pt]
  \\mathbf{1}^\\top w = \\mathbf{1}^\\top (I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top)z = \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top \\mathbf{1}}_{=m} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{\\mathbf{1}^\\top y}_{=0} y^\\top z = 0 \\\\[4pt]
  \\bullet \\ \\text{Orthogonal to } y:\\\\[4pt]
  y^\\top w = y^\\top z - \\frac{1}{m} \\underbrace{y^\\top \\mathbf{1}}_{=0} \\mathbf{1}^\\top z - \\frac{1}{m} \\underbrace{y^\\top y}_{=m} y^\\top z = 0
  $$

  Then $w = Pz \\in S$. So $Range(P) \\in S$.

  &nbsp;

  $\\textbf{2). } Pz = z \\text{ for any } z \\in S$
  
  &nbsp;

  Take any $z \\in S$, we have $1^\\top z = 0$ and $y^\\top z = 0$. So
  
  $$
  Pz = (I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top)z \\\\[4pt]
  = z - \\frac{1}{m} \\mathbf{1}(\\mathbf{1}^\\top z) - \\frac{1}{m} y(y^\\top z) \\\\[4pt]
  = z - \\frac{1}{m} \\mathbf{1} \\cdot 0 - \\frac{1}{m} y \\cdot 0 \\\\[4pt]
  = z \\\\[4pt]
  $$
  
  $\\textbf{3). } P \\textbf{ is the orthogonal projector onto } S$
  
  &nbsp;

  From $\\textbf{1)}$ and $\\textbf{2)}$ we have:

  &nbsp;

  $\\bullet \\ $ We showed $Range(P) \\in S$ $\\textbf{from 1)}$ and $Pz = z$ for any $z \\in S \\textbf{ from 2)}$. Hence $Range(P) = S$ and $P$ acts as identity on $S$.

  &nbsp;

  $\\bullet \\ $ We also showed $P$ is symmetric and idempotent at beginning of this note.

  &nbsp;

  **According to the theorem, $P$ is the orthogonal projector onto its range $Range(P)$**
  **and $Range(P) = S$, so $P$ is the orthogonal projector onto $S$.**

  &nbsp;

  ---

  &nbsp;
  
  $\\textbf{B. Prove that the 2-norm of any orthogonal projection matrix}$ $P \\neq 0$ $\\textbf{ is 1.}$

  &nbsp;

  We'll prove that for an orthogonal projection matrix $P$ (i.e., $P^\\top = P$ and $P^2 = P$), the 2-norm satisfies
  $$
  ||P||_2 \\leq 1
  $$
  and in fact $||P||_2 = 1$ unless $P = 0$.

  &nbsp;

  Here are the quick proof.

  &nbsp;

  $\\textbf{Geometric (Pythagoras) proof}\\\\[4pt]$
  Any vector $x$ decomposes uniquely as $x = u + s$ with $u \\in range(P)$ and $s \\in ker(P)$.
  
  Since $Px = u$,

  $$
  ||Px||_2^2 = ||u||_2^2 \\leq ||u||_2^2 + ||s||_2^2 = ||x||_2^2 \\Rightarrow ||Px||_2 \\leq ||x||_2 \\Rightarrow \\frac{||Px||_2}{||x||_2} \\leq 1
  $$

  The definition of 2-norm is $||P||_2 = \\max_{x \\neq 0} \\frac{||Px||_2}{||x||_2}$, so $||P||_2 \\leq 1$.

  &nbsp;

  Actually, if $P \\neq 0$, pick $x \\in range(P)$, then $Px = x$, so $||Px||_2 = ||x||_2$, and we have $||P||_2 = 1$.
  `,
  },

  orthogonal_projection_idempotent_notes: {
    title: "Idempotent projection (hover)",
    md: `
Prove $P$ is idempotent $P^2 = P$.
$$
P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top \\\\[4pt]
$$

under the LayerNorm conditions
$$
\\mathbf{1}^\\top y = 0,\\  y^\\top y = m
$$

Let
$$
A = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top,\\ B = \\frac{1}{m} yy^\\top
$$

We'll prove the following facts:

&nbsp;

$\\textbf{1. A and B are idempotnet}$
$$
A^2 = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}\\mathbf{1}(m)\\mathbf{1}^\\top = \\frac{1}{m}\\mathbf{1}\\mathbf{1}^\\top = A
$$
Similaryly, using $y^\\top y = m$, we have
$$
B^2 = \\frac{1}{m^2}y(y^\\top y)y^\\top = \\frac{1}{m^2}y(m)y^\\top = \\frac{1}{m}yy^\\top = B
$$

&nbsp;

$\\textbf{2. A and B are orthogonal}$
$$
\\text{Using } \\mathbf{1}^\\top y = 0,\\\\[4pt]
AB = \\frac{1}{m^2}\\mathbf{1}(\\mathbf{1}^\\top y)y^\\top = \\frac{1}{m^2}\\mathbf{1}(0)y^\\top = 0\\\\[4pt]
BA = \\frac{1}{m^2}y(y^\\top\\mathbf{1})\\mathbf{1}^\\top = \\frac{1}{m^2}y(0)\\mathbf{1}^\\top = 0
$$

&nbsp;

$\\textbf{3. Expand } P^2 \\textbf{ and simplify:}$
$$
P^2 = (I - A - B)^2 \\\\[4pt]
= I - 2A - 2B + A^2 + B^2 + AB + BA \\\\[4pt]
= I - 2A - 2B + A + B + 0 + 0 \\ \\ \\text{(By facts 1 and 2)} \\\\[4pt]
= I - A - B \\\\[4pt]
= P
$$

Hence $P^2 = P$
    `,
  },

  orthogonal_projection_symmetric_notes: {
    title: "Symmetric projection (hover)",
    md: `
Prove $P$ is symmetric $P^\\top = P$.
$$
\\ \\ \\ P = \\mathbf{I} - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top \\\\[4pt]
$$

Take the transpose term-by-term, using $(A+B)^\\top = A^\\top + B^\\top$, $(AB)^\\top = B^\\top A^\\top$ and $(cA)^\\top = cA^\\top$ for scalar $c$.

&nbsp;

$\\textbf{1.}$ $I^\\top = I$ (theidentity is symmetric).

&nbsp;

$\\textbf{2.}$ $(\\mathbf{1}\\mathbf{1}^\\top)^\\top = (\\mathbf{1}^\\top)^\\top \\mathbf{1}^\\top = \\mathbf{1}\\mathbf{1}^\\top$.

&nbsp;

$\\textbf{3.}$ $(yy^\\top)^\\top = (y^\\top)^\\top y^\\top = yy^\\top$.

&nbsp;

Therefore,.

$$
P^\\top = I^\\top - \\frac{1}{m} (\\mathbf{1}\\mathbf{1}^\\top)^\\top - \\frac{1}{m} (yy^\\top)^\\top = I - \\frac{1}{m} \\mathbf{1}\\mathbf{1}^\\top - \\frac{1}{m} yy^\\top = P
$$


    `,
  },
  orthogonal_projection_proof_notes: {
    title: "Symmetric + idempotent ⇒ orthogonal projector",
    md: `

  Let's approve the theorem that a symmetric idempotent matrix is the orthogonal projector onto its range.

  &nbsp;

  $\\bullet \\ $ Idempotent ($P^2 = P$) alone $\\Rightarrow P$ is a projection (onto $range(P)$ along $ker(P)$), but not necessarily orthogonal.

  $\\bullet \\ $ Idempotnet + symmetric ($P^\\top = P$) $\\Rightarrow P$ is the orthogonal projector onto its column space $S = Range(P)$.
  
  &nbsp;

  $\\textbf{Proof:}\\\\[4pt]$
  $\\textbf{1. Idempoent} \\Rightarrow projection.\\\\[4pt]$

  For any $x$,$\\\\[4pt]$

  $\\bullet \\ $ $Px \\in range(P).\\\\[4pt]$
  $\\bullet \\ $ $x - Px \\in ker(P)$ because $P(x - Px) = Px - P^2x = Px - Px = 0.\\\\[4pt]$
  So every $x$ decomposes uniquely as $x = Px + (x - Px)$ with the first part in $range(P)$ and the second in $ker(P)$. That's exactly a projection.

  &nbsp;

  $\\textbf{2. Add symmetry} \\Rightarrow \\textbf{orthogonal projector}.\\\\[4pt]$

  &nbsp;

  Let $S = range(P)$. $\\\\[4pt]$ 
  Choose any vector $z \\in ker(P)$, so $Pz = 0$. $\\\\[4pt]$
  Choose any vector $y = Pw \\in S$, then
  $$
  y^\\top z = w^\\top P^\\top z = w^\\top P z = 0
  $$

  So vector $z$ and $y$ are orthogonal, and because $z$ is chosen arbitrarily from $ker(P)$, $y$ is chosen arbitrarily from $S$.
  Therefore, $ker(P) \\perp S$, i.e., $P$ is the orthogonal projector onto $S$.

    `,
  },

  optim_adamw_notes: {
    title: "torch.optim.AdamW",
    md: `
$\\textbf{What it is}\\\\[4pt]$

AdamW is the dominant optimizer in the current LLM pretraining. An optimizer is the decision making algorithm in pre-training. At each step the optimizer answers the question: 
"Given the current parameters and the gradient, how should I update the parameters to reduce loss?", so

$\\bullet \\ $ The *gradient* tells us the local slope of the loss with respect to each parameters.

&nbsp;

$\\bullet \\ $ The *optimizer* gives us the algorithm that turns the gradient into an update.

&nbsp;

---

&nbsp;

$\\textbf{History of optimizers - From SGD to AdamW}\\\\[4pt]$

From my learning experience, I find it's difficult to understand AdamW without understanding the history of optimizers.
So below is a language model focused optimizer timeline with the key ideas like who used that and why it changed.
Click the links under each optimizer to see the details.

&nbsp;

$\\textbf{2010-2013: Sparse features era (word vectors, classic NLP)}\\\\[4pt]$
$\\bullet \\ $ **$\\href{#info:sgd_notes}{\\text{SGD}}$** - simple and fast; the original optimizer.

$\\bullet \\ $ **$\\href{#info:sgd_momentum_notes}{\\text{SGD with momentum}}$** - adds velocity to smooth updates.

$\\bullet \\ $ **$\\href{#info:adagrad_notes}{\\text{Adagrad}}$** - popular for sparse, high-variance gradients; helped early embeddings & logistic/softmax models. It adjusts the learning rate for each parameter during training
and keeps adjusting it over time.

$\\bullet \\ $ **$\\href{#info:rmsprop_notes}{\\text{RMSProp}}$** - fix Adagrad's vanishing step size; widely used in early RNN LMs.

&nbsp;

$\\textbf{2014-2017: RNNs → early seq2seq}\\\\[4pt]$
$\\bullet \\ $ **$\\href{#info:adam_notes}{\\text{Adam}}$** - becomes default in NLP (RNN LMs, seq2seq with attention).
Why: combines momentum + RMSProp-style variance normalization, much less LR fiddling than SGD.

&nbsp;

$\\textbf{2017-2019: Transformer takeoff}\\\\[4pt]$
$\\bullet \\ $ **Adam + warmup standardizes**: small linear warmup then decay (often inverse-sqrt or cosine).

$\\bullet \\ $ Post-norm → Pre-norm transitions; Adam's adaptivity + warmup stabilize deep stacks.

$\\bullet \\ $ Weight decay starts to matter. BERT popularizes weight decay with Adam; the community converges on decoupled weight decay (AdamW) soon after for predictability and better generalization.

&nbsp;

$\\textbf{2019-2021: Scaling up (T5, GPT-2/3, mT5)}\\\\[4pt]$

$\\bullet \\ $ **$\\href{#info:adamw_notes}{\\text{AdamW}}$** becomes the go-to for big LMs (bf16, warmup+cosine, grad-clip, exclude decay on norm/bias).

$\\bullet \\ $ Adafactor (memory-efficient Adam-like) gains traction for T5/mT5-style models when optimizer-state RAM is tight (factored second moments).

&nbsp;

$\\textbf{2022-2023: Scaling up further}\\\\[4pt]$
$\\bullet \\ $ Pretraining still: AdamW (or Adafactor where memory is the limiter).

$\\bullet \\ $ Finetuning: experiments with Lion, Adan, AdaBelief, Sophia, etc. show promise on some setups, but none broadly displace AdamW for foundation-model pretraining.

$\\bullet \\ $ Systems stacks adopt fused AdamW kernels (Apex, DeepSpeed, xFormers) for throughput.

&nbsp;

$\\textbf{2024-today}\\\\[4pt]$

$\\bullet \\ $ AdamW remains the mainstream default for LLM pretraining and most full-parameter finetunes.

$\\bullet \\ $ Adafactor remains a practical alternative when optimizer memory must be minimized.

$\\bullet \\ $ Research optimizers appear regularly, but production LMs stick to the stable duo above, paired with: pre-norm/RMSNorm, bf16, linear warmup (≈0.5–3% steps), cosine decay, global-norm clipping, and no-decay on norm/bias.

&nbsp;

---

&nbsp;


`,
  },

  sgd_notes: {
    title: "SGD (Stochastic Gradient Descent)",
    md: `
$\\textbf{What it is}\\\\[4pt]$

The classic and simplest optimizer that updates parameters in the negative gradient direction with a fixed learning rate.

$\\bullet \\ $ The **gradient** tells you which way the loss increases fastest.

$\\bullet \\ $ To go **downhill**, move **against** the gradient.

&nbsp;

Say we have parameters $\\theta$ and loss $L(\\theta)$ we want to minimize.

Update rule:
$$
\\theta \\leftarrow \\theta - \\eta \\triangledown_{\\theta} L(\\theta)
$$

$\\bullet \\ \\eta$ is the **learning rate** (step size). Too big => overshoot. Too small => slow.
$\\bullet \\ $ $\\triangledown_{\\theta} L(\\theta)$ is the gradient of the loss with respect to the parameters $\\theta$.

Think of it as you are walking downhill in fog. The gradient is your local down slope. The learning rate is how big a step you take.
If you take a big step, you might miss the bottown. If you take a small step, you might take a long time to get to the bottom.

&nbsp;

---

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$

&nbsp;

$\\textbf{Setup:}\\\\[4pt]$
$\\text{Model: } \\hat{y} = wx + b\\\\[4pt]$
$\\text{Loss: } L = \\frac{1}{2} (\\hat{y} - y)^2 \\ \\ \\ \\ \\textit{(y is the target value)}\\\\[4pt]$
$\\text{Data point: } x = 2, y = 5\\\\[4pt]$
$\\text{Init: } w = 1.0, b = 0.0\\\\[4pt]$
$\\text{Learning rate: } \\eta = 0.1\\\\[4pt]$

&nbsp;

$\\textbf{Step 1:}\\\\[4pt]$
$\\text{Forward pass: }\\\\[4pt]$
$\\bullet \\ \\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2\\\\[4pt]$
$\\bullet \\ \\text{Error } e = \\hat{y} - y = 2 - 5 = -3\\\\[4pt]$
$\\bullet \\ \\text{Loss } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5\\\\[8pt]$

$\\text{Backprop: }\\\\[4pt]$
$\\bullet \\ \\triangledown_{\\hat{y}} L = \\hat{y} - y = e = -3 \\\\[4pt]$
$\\bullet \\ \\triangledown_w L = \\triangledown_{\\hat{y}} L \\cdot \\triangledown_w \\hat{y} = e \\cdot x = -3 \\cdot 2 = -6\\\\[4pt]$
$\\bullet \\ \\triangledown_b L = \\triangledown_{\\hat{y}} L \\cdot \\triangledown_b \\hat{y} = e = -3\\\\[8pt]$

$\\text{Optimizer update: }\\\\[4pt]$
$\\bullet \\ w \\leftarrow w - \\eta \\triangledown_w L = 1.0 - 0.1 \\cdot (-6) = 1.6\\\\[4pt]$
$\\bullet \\ b \\leftarrow b - \\eta \\triangledown_b L = 0.0 - 0.1 \\cdot (-3) = 0.3\\\\[8pt]$

$\\textbf{Step 2:}\\\\[4pt]$
$\\text{Forward pass: }\\\\[4pt]$
$\\bullet \\ \\hat{y} = wx + b = 1.6 \\cdot 2 + 0.3 = 3.5\\\\[4pt]$
$\\bullet \\ \\text{Error } e = \\hat{y} - y = 3.5 - 5 = -1.5\\\\[4pt]$
$\\bullet \\ \\text{Loss } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-1.5)^2 = 1.125\\\\[8pt]$

$\\text{Backprop: }\\\\[4pt]$
$\\bullet \\ \\triangledown_{\\hat{y}} L = e = -1.5 \\\\[4pt]$
$\\bullet \\ \\triangledown_w L = e \\cdot x = -1.5 \\cdot 2 = -3\\\\[4pt]$
$\\bullet \\ \\triangledown_b L = e = -1.5\\\\[8pt]$

$\\text{Optimizer update: }\\\\[4pt]$
$\\bullet \\ w = 1.6 - 0.1(-3) = 1.9\\\\[4pt]$
$\\bullet \\ b = 0.3 - 0.1(-1.5) = 0.45\\\\[8pt]$

$\\textbf{Step 3:}\\\\[4pt]$
$\\text{Forward pass: }\\\\[4pt]$
$\\bullet \\ \\hat{y} = wx + b = 1.9 \\cdot 2 + 0.45 = 4.25\\\\[4pt]$
$\\bullet \\ \\text{Error } e = \\hat{y} - y = 4.25 - 5 = -0.75\\\\[4pt]$
$\\bullet \\ \\text{Loss } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-0.75)^2 = 0.28125\\\\[8pt]$

$\\text{Backprop: }\\\\[4pt]$
$\\bullet \\ \\triangledown_{\\hat{y}} L = e = -0.75 \\\\[4pt]$
$\\bullet \\ \\triangledown_w L = e \\cdot x = -0.75 \\cdot 2 = -1.5\\\\[4pt]$
$\\bullet \\ \\triangledown_b L = e = -0.75\\\\[8pt]$

$\\text{Optimizer update: }\\\\[4pt]$
$\\bullet \\ w = 1.9 - 0.1(-1.5) = 2.05\\\\[4pt]$
$\\bullet \\ b = 0.45 - 0.1(-0.75) = 0.525\\\\[8pt]$

You can see the loss decreasing at each step (4.5 -> 1.125 -> 0.28125). That's the optimizer using backprop's gradients to move $w, b$ toward better values.
`,
  },

  sgd_momentum_notes: {
    title: "SGD with momentum",
    md: `
$\\textbf{What it is}\\\\[4pt]$

Instead of stepping only with the current gradient, we keep a velocity (an exponential moving average of past gradients).
The velocity smooths noise and accelerates motion in directions that stay consistently downhill.

&nbsp;

We use the common EMA form of gradients as below,

$$
v \\leftarrow \\beta v + (1 - \\beta) g \\\\[4pt]
\\theta \\leftarrow \\theta - \\eta v
$$

$\\bullet \\ g = \\triangledown_{\\theta} L $ is the current gradient.

$\\bullet \\ \\beta $ (e.g., 0.9) controls how much "memory" to keep.

$\\bullet \\ \\eta $ is the learning rates.

&nbsp;

---

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$

$\\textbf{Setup:}\\\\[4pt]$
$\\text{We use the same setup as the numeric example in SGD notes.}\\\\[4pt]$
$\\text{Model: } \\hat{y} = wx + b\\\\[4pt]$
$\\text{Loss: } L = \\frac{1}{2} (\\hat{y} - y)^2 \\ \\ \\ \\ \\textit{(y is the target value)}\\\\[4pt]$
$\\text{Data point: } x = 2, y = 5\\\\[4pt]$
$\\text{Init: } w = 1.0, b = 0.0, v_w = v_b = 0\\\\[4pt]$
$\\text{Hyperparams: Learning rate: } \\eta = 0.1 \\text{, momentum } \\beta = 0.9 \\\\[4pt]$

$\\text{Remember gradients of loss w.r.t. w and b are } 
$$
\\triangledown_w L = (\\hat{y} - y) \\cdot x = e \\cdot x \\\\[4pt]
\\triangledown_b L = \\hat{y} - y = e \\\\[4pt]
$$

$\\textbf{Step 1:}\\\\[4pt]$
$\\text{Forward pass: } \\hat{y} = wx + b = 1.0 \\cdot 2 + 0.0 = 2\\\\[4pt]$
$\\text{Error: } e = \\hat{y} - y = 2 - 5 = -3\\\\[4pt]$
$\\text{Loss: } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-3)^2 = 4.5\\\\[8pt]$
$\\text{Gradients: } \\triangledown_w L = e \\cdot x = -3 \\cdot 2 = -6, \\triangledown_b L = e = -3\\\\[4pt]$

$\\text{Velocity: }\\\\[4pt]$
$$
\\bullet \\ v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\triangledown_w L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-6) = -0.6\\\\[4pt]
\\bullet \\ v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\triangledown_b L = 0.9 \\cdot 0 + (1 - 0.9) \\cdot (-3) = -0.3\\\\[4pt]
$$

$\\text{Optimizer update: }\\\\[4pt]$
$$
\\bullet \\ w \\leftarrow w - \\eta \\cdot v_w = 1.0 - 0.1 \\cdot (-0.6) = 1.06\\\\[4pt]
\\bullet \\ b \\leftarrow b - \\eta \\cdot v_b = 0.0 - 0.1 \\cdot (-0.3) = 0.03\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 2:}\\\\[4pt]$
$\\text{Forward pass: } \\hat{y} = wx + b = 1.06 \\cdot 2 + 0.03 = 2.15\\\\[4pt]$
$\\text{Error: } e = \\hat{y} - y = 2.15 - 5 = -2.85\\\\[4pt]$
$\\text{Loss: } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.85)^2 = 4.0613\\\\[8pt]$
$\\text{Gradients: } \\triangledown_w L = e \\cdot x = -2.85 \\cdot 2 = -5.7, \\triangledown_b L = e = -2.85\\\\[4pt]$

$\\text{Velocity: }\\\\[4pt]$
$$
\\bullet \\ v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\triangledown_w L = 0.9 \\cdot (-0.6) + (1 - 0.9) \\cdot (-5.7) = -0.54 + (-0.57) = -1.11\\\\[4pt]
\\bullet \\ v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\triangledown_b L = 0.9 \\cdot (-0.3) + (1 - 0.9) \\cdot (-2.85) = -0.27 + (-0.285) = -0.555\\\\[4pt]
$$

$\\text{Optimizer update: }\\\\[4pt]$
$$
\\bullet \\ w \\leftarrow w - \\eta \\cdot v_w = 1.06 - 0.1 \\cdot (-1.11) = 1.171\\\\[4pt]
\\bullet \\ b \\leftarrow b - \\eta \\cdot v_b = 0.03 - 0.1 \\cdot (-0.555) = 0.0855\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 3:}\\\\[4pt]$
$\\text{Forward pass: } \\hat{y} = wx + b = 1.171 \\cdot 2 + 0.0855 = 2.4275\\\\[4pt]$
$\\text{Error: } e = \\hat{y} - y = 2.4275 - 5 = -2.5725\\\\[4pt]$
$\\text{Loss: } L = \\frac{1}{2} e^2 = \\frac{1}{2} (-2.5725)^2 = 3.3088\\\\[8pt]$
$\\text{Gradients: } \\triangledown_w L = e \\cdot x = -2.5725 \\cdot 2 = -5.145, \\triangledown_b L = e = -2.5725\\\\[4pt]$

$\\text{Velocity: }\\\\[4pt]$
$$
\\bullet \\ v_w = \\beta \\cdot v_w + (1 - \\beta) \\cdot \\triangledown_w L = 0.9 \\cdot (-1.11) + (1 - 0.9) \\cdot (-5.145) = -0.999 + (-0.5145) = -1.5135\\\\[4pt]
\\bullet \\ v_b = \\beta \\cdot v_b + (1 - \\beta) \\cdot \\triangledown_b L = 0.9 \\cdot (-0.555) + (1 - 0.9) \\cdot (-2.5725) = -0.4995 + (-0.25725) = -0.75675\\\\[4pt]
$$

$\\text{Optimizer update: }\\\\[4pt]$
$$
\\bullet \\ w \\leftarrow w - \\eta \\cdot v_w = 1.171 - 0.1 \\cdot (-1.5135) = 1.32235\\\\[4pt]
\\bullet \\ b \\leftarrow b - \\eta \\cdot v_b = 0.0855 - 0.1 \\cdot (-0.75675) = 0.161175\\\\[4pt]
$$

You can see the loss decreasing at each step (4.5 -> 4.0613 -> 3.3088). That's the optimizer using backprop's gradients to move $w, b$ toward better values.
Compare with SGD where the loss decreases more faster (4.5 -> 1.125 -> 0.28125) in the first 3 steps as shown in the example. The reaon is:

&nbsp;

$\\text{Why momentum lags at first:}\\\\[4pt]$

Velocity needs to build up. With $\\beta = 0.9$, the velocity starts at 0 and at early steps it only tracks the current gradient by a fraction.
That makes the initial steps smaller than plain SGD. 

&nbsp;

For example, in our tiny numeric example, 

**Step 1** accumulates 90% of the past velocity ($0$ for now because $v_w = 0, v_b = 0$) and 10% of the current gradient, so step 1 is 0.1x the raw gradient.

**Step 2** accumulates 90% of the past velocity (0.1x raw gradient from step 1) and 10% of the current gradient, so step 2 is roughly 0.19x the raw gradient (0.1 * 0.9 + 0.1 = 0.19).

**Step 3** accumulates 90% of the past velocity (0.19x raw gradient from step 1/2) and 10% of the current gradient, so step 3 is roughly 0.271x the raw gradient (0.19 * 0.9 + 0.1 = 0.271).


&nbsp;

$\\text{When/why momentum wins later:}\\\\[4pt]$
Once gradients stay roughly aligned, the velocity accumulates, so steps become larger and smoother than SGD. With more steps, momentum catches up and passes SGD.

&nbsp;

$\\text{If we want momentum to look better earlier, we can:}\\\\[4pt]$

$\\text{1.}$ Lower $\\beta$ (e.g., 0.8), so velocity builds up faster.

$\\text{2.}$ Slightly raise $\\eta$ (e.g., 0.12) to make up for the smaller steps at early steps.

`,
  },

  adagrad_notes: {
    title: "Adagrad",
    md: `
$\\textbf{What it is}\\\\[4pt]$

Adagrad (Adaptive gradient) is an optimizer that gives each parameter its own learning rate and automatically adjusts it based on that parameter's past gradients.
Thus parameters with large/frequent past gradients get smaller future steps; parameters with small/rare gradiets get larger future steps.

&nbsp;

$\\textbf{Update rule:}\\\\[4pt]$
Adagrad keeps a running **sum of squared gradients** for each parameter:
$$
s_i \\leftarrow s_i + g_i \\odot g_i
$$

Where $g_i$ is the gradient of the $i$-th parameter, and $s_i$ is the running sum of squared gradients for the $i$-th parameter.
Then adagrad divides the learning rate by the square root of the sum of squared gradients plus a small constant $\\epsilon$ to get the learning rate for the $i$-th parameter:
$$
\\theta_i \\leftarrow \\theta_i - \\frac{\\eta}{\\sqrt{s_i} + \\epsilon} \\cdot g_i
$$
This way, the parameters with large/frequent past gradients get smaller future steps; parameters with small/rare gradiets get larger future steps.

&nbsp;

Because $s_i$ is a cumulative sum, it grows about linearly over time for stationary gradients ($s_i \\propto t$). Using $\\sqrt{s_i}$ in the denominator makes the effective learning rate decay like:
$$
\\eta_{\\text{effective, }i} = \\frac{\\eta}{\\sqrt{s_i}} \\approx \\frac{\\eta}{\\sqrt{t}}.
$$
The $\\frac{1}{\\sqrt{t}}$ decay is gentle: it shrinks steps over time stabily but not too fast that learning dies early. If we devided by $s_i$, we'd get $\\frac{1}{t}$ decay,
which becomes tiny too quickly and can stall progress.

&nbsp;

This is also tied to classic online convex optimization results where $\\frac{1}{\\sqrt{t}}$ yields good regret/convergence bounds.

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$
$\\textbf{Setup:}\\\\[4pt]$
$$
\\bullet \\ \\text{Parameters: } \\theta = (\\theta_1, \\theta_2)\\\\[4pt]
\\bullet \\ \\text{Loss: } L = \\frac{1}{2} [(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2]  \\\\[4pt]
\\bullet \\ \\text{Gradients: } g = (g_1, g_2) = (\\theta_1 - 3, \\theta_2 + 2) \\\\[4pt]
\\bullet \\ \\text{Adagrad accumulator: } s \\leftarrow s + g \\odot g, \\text{Start } s_0 = (0, 0)\\\\[4pt]
\\bullet \\ \\text{Updates rule: } \\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon} g \\\\[4pt]
\\bullet \\ \\text{Hyperparams: } \\eta = 1.0, \\epsilon = 1e-8 \\text{ tiny so numerically negligible} \\\\[4pt]
\\bullet \\ \\text{Init: } \\theta_0 = (0, 0)\\\\[4pt]
$$

Below we define $\\eta_{\\text{eff}} = \\frac{\\eta}{\\sqrt{s_{t}} + \\epsilon}$ as the effective learning rate at time $t$.

&nbsp;

$\\textbf{Step 1:}\\\\[4pt]$
$$
\\bullet \\ \\theta_0 = (0, 0)\\\\[4pt]
\\bullet \\ g_1 = (-3, 2)\\\\[4pt]
\\bullet \\ s_1 = (0, 0) + (-3, 2) \\odot (-3, 2) = (9, 4)\\\\[4pt]
\\bullet \\ \\sqrt{s_1} = (3, 2) \\Rightarrow \\eta_{\\text{eff}} = (1/3, 1/2) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (1/3, 1/2) \\cdot (-3, 2) = (-1, 1)\\\\[4pt]
\\bullet \\ \\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-1, 1) = (1, -1)\\\\[4pt]
$$

Coord1 and coord2 have different effective learning rates now : 1/3 vs 1/2.

&nbsp;

$\\textbf{Step 2:}\\\\[4pt]$
$$
\\bullet \\ \\theta_1 = (1, -1)\\\\[4pt]
\\bullet \\ g_2 = (1 - 3, -1 + 2) = (-2, 1)\\\\[4pt]
\\bullet \\ s_2 = (9, 4) + (-2, 1) \\odot (-2, 1) = (13, 5)\\\\[4pt]
\\bullet \\ \\sqrt{s_2} = (3.606, 2.236) \\Rightarrow \\eta_{\\text{eff}} = (0.277, 0.447) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_2 \\approx (0.277, 0.447) \\cdot (-2, 1) = (-0.554, 0.447)\\\\[4pt]
\\bullet \\ \\theta_2 = \\theta_1 - \\triangledown = (1, -1) - (-0.554, 0.447) = (1.554, -1.447)\\\\[4pt]
$$

Coord1's step shrank from 1/3 to 0.277, coord2's step shrank from 1/2 to 0.447.

&nbsp;

$\\textbf{Step 3:}\\\\[4pt]$
$$
\\bullet \\ \\theta_2 = (1.554, -1.447)\\\\[4pt]
\\bullet \\ g_3 = (1.554 - 3, -1.447 + 2) = (-1.446, 0.553)\\\\[4pt]
\\bullet \\ s_3 = (13, 5) + (-1.446, 0.553) \\odot (-1.446, 0.553) = (15.091, 5.306)\\\\[4pt]
\\bullet \\ \\sqrt{s_3} = (3.885, 2.306) \\Rightarrow \\eta_{\\text{eff}} = (0.257, 0.434) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_3 \\approx (0.257, 0.434) \\cdot (-1.446, 0.553) = (-0.372, 0.24)\\\\[4pt]
\\bullet \\ \\theta_3 = \\theta_2 - \\triangledown = (1.554, -1.447) - (-0.372, 0.24) = (1.926, -1.687)\\\\[4pt]
$$

Coord1's step shrank from 0.277 to 0.257, coord2's step shrank from 0.447 to 0.434.

&nbsp;

This example shows each coordinate $i$ gets its own learning rage $\\eta_{\\text{eff, }i}$ which depends on the past gradients of the $i$-th coordinate.
These learning rates shrink over time as $s_i$ accumulates past squared gradients.

&nbsp;

Coordinates with larger/frequent gradients (here $\\theta_1$) get faster shrinkage so they get smaller steps, 
while coordinates with small/rare gradients (here $\\theta_2$) get slower shrinkage so they get larger steps.

`,
  },

  rmsprop_notes: {
    title: "RMSProp",
    md: `
$\\textbf{What it does:}\\\\[4pt]$

RMSProp keeps an exponential moving average (EMA) of each parameter's squared gradient and scales the step by the RMS (square root) of that average:

$$
s_t = \\beta s_{t-1} + (1 - \\beta) (g_t \\odot g_t) \\\\[4pt]
\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{s_t} + \\epsilon} g_t
$$

Where $\\beta$ is the decay rate (e.g., 0.9), $s_t$ is the EMA of the squared gradient, $g_t$ is the gradient at time $t$, $\\theta_t$ is the parameter at time $t$, and $\\eta$ is the learning rate.
$\\epsilon$ is a small constant to avoid division by zero.

&nbsp;

---

&nbsp;

$\\textbf{What RMSProp fixes (vs. Adagrad):}\\\\[4pt]$

Adagrad sums squared gradients forever, so per-parameter step sizes shrink monotonically ($~1/\\sqrt{t}$) and can get too small in long runs.

RMSProp replaces the forever growing sum with an EMA, this prevents the learning rate from decreasing too quickly, making the algorithm more effective in trainig deep neural networks.

&nbsp;

We can think of RMSProd to Adagrad as SGD+momentum to SGD in the following way:

$\\bullet \\ $ Adagrap -> RMSProp: add an EMA (memory) to the squared gradient statistic instead of the forever growing sum.

$\\bullet \\ $ SGD -> SGD+momentum: add an EMA (memory) of the gradient itself. 

&nbsp;

But there are still some key differences:

$\\bullet \\ $ $\\textbf{What gets the EMA: }$ RMSProp = squared gradients. SGD+momentum = gradients.

$\\bullet \\ $ $\\textbf{Per-parameter LR: }$ RMSProp (like Adagrad) keeps per-parameter adaptive step sizes; SGD+momentum (like SGD) uses one global LR (no per-parameter).

$\\bullet \\ $ $\\textbf{Failure mode fixed: }$ Adagrad's LR decays to near-zero on long runs; RMSProd avoids that. SGD is noisy/zig-zaggy; SGD+momentum smooths and accelerates it.

&nbsp;

---

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$
$\\textbf{Setup:}\\\\[4pt]$
$$
\\bullet \\ \\text{Parameters: } \\theta = (\\theta_1, \\theta_2)\\\\[4pt]
\\bullet \\ \\text{Loss: } L = \\frac{1}{2} [(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2]  \\\\[4pt]
\\bullet \\ \\text{Gradients: } g = (g_1, g_2) = (\\theta_1 - 3, \\theta_2 + 2) \\\\[4pt]
\\bullet \\ \\text{RMSProp accumulator: } s \\leftarrow \\beta s + (1 - \\beta) (g \\odot g), \\text{Start } s_0 = (0, 0)\\\\[4pt]
\\bullet \\ \\text{Updates rule: } \\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{s} + \\epsilon} g \\\\[4pt]
\\bullet \\ \\text{Hyperparams: } \\eta = 0.1, \\beta = 0.9, \\epsilon = 1e-8 \\text{ tiny so numerically negligible} \\\\[4pt]
\\bullet \\ \\text{Init: } \\theta_0 = (0, 0)\\\\[4pt]
$$

$\\textbf{Step 1:}\\\\[4pt]$
$$
\\bullet \\ \\theta_0 = (0, 0)\\\\[4pt]
\\bullet \\ g_1 = (-3, 2)\\\\[4pt]
\\bullet \\ s_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot ((-3, 2) \\odot (-3, 2)) = (0.9, 0.4)\\\\[4pt]
\\bullet \\ \\sqrt{s_1} = (0.9487, 0.6325) \\Rightarrow \\eta_{\\text{eff}} = (0.1054, 0.1581) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_1 = (0.1054, 0.1581) \\cdot (-3, 2) = (-0.3162, 0.3162)\\\\[4pt]
\\bullet \\ \\theta_1 = \\theta_0 - \\triangledown = (0, 0) - (-0.3162, 0.3162) = (0.3162, -0.3162)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 2:}\\\\[4pt]$
$$
\\bullet \\ \\theta_1 = (0.3162, -0.3162)\\\\[4pt]
\\bullet \\ g_2 = (0.3162 - 3, -0.3162 + 2) = (-2.6838, 1.6838)\\\\[4pt]
\\bullet \\ s_2 = 0.9 \\cdot (0.9, 0.4) + (1 - 0.9) \\cdot ((-2.6838, 1.6838) \\odot (-2.6838, 1.6838)) = (1.5307, 0.6435)\\\\[4pt]
\\bullet \\ \\sqrt{s_2} = (1.2373, 0.8022) \\Rightarrow \\eta_{\\text{eff}} = (0.0808, 0.1246) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_2 = (0.0808, 0.1246) \\cdot (-2.6838, 1.6838) = (-0.2170, 0.2099)\\\\[4pt]
\\bullet \\ \\theta_2 = \\theta_1 - \\triangledown = (0.3162, -0.3162) - (-0.2170, 0.2099) = (0.5332, -0.5261)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 3:}\\\\[4pt]$
$$
\\bullet \\ \\theta_2 = (0.5332, -0.5261)\\\\[4pt]
\\bullet \\ g_3 = (0.5332 - 3, -0.5261 + 2) = (-2.4668, 1.4738)\\\\[4pt]
\\bullet \\ s_3 = 0.9 \\cdot (1.5307, 0.6435) + (1 - 0.9) \\cdot ((-2.4668, 1.4738) \\odot (-2.4668, 1.4738)) = (1.9862, 0.7964)\\\\[4pt]
\\bullet \\ \\sqrt{s_3} = (1.4093, 0.8924) \\Rightarrow \\eta_{\\text{eff}} = (0.0710, 0.1121) \\\\[4pt]
\\bullet \\ \\text{Update } \\triangledown = \\eta_{\\text{eff}} \\cdot g_3 = (0.0710, 0.1121) \\cdot (-2.4668, 1.4738) = (-0.1751, 0.1652)\\\\[4pt]
\\bullet \\ \\theta_3 = \\theta_2 - \\triangledown = (0.5332, -0.5261) - (-0.1751, 0.1652) = (0.7083, -0.6913)\\\\[4pt]
$$

`,
  },

  adam_notes: {
    title: "Adam",
    md: `
$\\textbf{What it is}\\\\[4pt]$

Adam combines the best ideas of RMSProp and SGD+momentum:

&nbsp;

$\\bullet \\ $ **RMSProp**: keeps track of the squared gradient (which captures the scale of the gradient) to normalize the step size.

$\\bullet \\ $ **SGD+momentum**: keeps track of the gradient itself (which captures the direction of the gradient) to smooth the updates.

&nbsp;

Adam then combines these two ideas in a single update rule:

$$
m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\ \\ \\ \\ \\text{(EMA of gradients: momentum)} \\\\[4pt]
v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) (g_t \\odot g_t) \\ \\ \\ \\ \\text{(EMA of squared gradients: RMS scaling)} \\\\[4pt]
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\ \\ \\ \\ \\text{(bias correction)} \\\\[4pt]
\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\ \\ \\ \\ \\text{(bias correction)} \\\\[4pt]
\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t \\ \\ \\ \\ \\text{(adaptive step per parameter)} \\\\[4pt]
$$

Where $\\beta_1$ and $\\beta_2$ are the momentum and RMS scaling coefficients, respectively.
$\\epsilon$ is a small constant to avoid division by zero.

&nbsp;

Bias correction fixes the biased EMA (exponential moving average) problem. We'll discuss it at the end of the notes.

&nbsp;

---

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$

$\\textbf{Setup:}\\\\[4pt]$
$$
\\bullet \\ \\text{Parameters: } \\theta = (\\theta_1, \\theta_2)\\\\[4pt]
\\bullet \\ \\text{Loss: } L = \\frac{1}{2} [(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2]  \\\\[4pt]
\\bullet \\ \\text{Gradients: } g = (g_1, g_2) = (\\theta_1 - 3, \\theta_2 + 2) \\\\[4pt]
\\bullet \\ \\text{Updates rule: } \\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t \\\\[4pt]
\\bullet \\ \\text{Hyperparams: } \\eta = 0.1, \\beta_1 = 0.9, \\beta_2 = 0.999, \\epsilon = 1e-8 \\text{ tiny so numerically negligible} \\\\[4pt]
\\bullet \\ \\text{Init: } \\theta_0 = (0, 0), m_0 = (0, 0), v_0 = (0, 0)\\\\[4pt]
$$

$\\textbf{Step 1:}\\\\[4pt]$
$$
\\bullet \\ \\theta_0 = (0, 0)\\\\[4pt]
\\bullet \\ g_1 = (-3, 2)\\\\[4pt]
\\bullet \\ m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)\\\\[4pt]
\\bullet \\ v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot ((-3, 2) \\odot (-3, 2)) = (0.009, 0.004)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_1 = \\frac{m_1}{1 - \\beta_1^1} = \\frac{(-0.3,0.2)}{1 - 0.9^1} = (-3,2) \\\\[4pt]
\\hat{v}_1 = \\frac{v_1}{1 - \\beta_2^1} = \\frac{(0.009,0.004)}{1 - 0.999^1} = (9, 4)\\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon} \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9,4)}} \\cdot (-3, 2) = (0.1, -0.1)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 2:}\\\\[4pt]$
$$
\\bullet \\ \\theta_1 = (0.1, -0.1)\\\\[4pt]
\\bullet \\ g_2 = (0.1 - 3, -0.1 + 2) = (-2.9, 1.9)\\\\[4pt]
\\bullet \\ m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)\\\\[4pt]
\\bullet \\ v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot ((-2.9, 1.9) \\odot (-2.9, 1.9)) = (0.017401, 0.007606)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_2 = \\frac{m_2}{1 - \\beta_1^2} = \\frac{(-0.56,0.37)}{1 - 0.9^2} = (-2.94737, 1.94737) \\\\[4pt]
\\hat{v}_2 = \\frac{v_2}{1 - \\beta_2^2} = \\frac{(0.017401,0.007606)}{1 - 0.999^2} = (8.70485,3.80490)\\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_2 = \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon} \\hat{m}_2 = (0.1, -0.1) - \\frac{0.1}{\\sqrt{(8.70485,3.80490)}} \\cdot (-2.94737, 1.94737) \\\\[4pt]
= (0.1998973,-0.1998335)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 3:}\\\\[4pt]$
$$
\\bullet \\ \\theta_2 = (0.1998973,-0.1998335)\\\\[4pt]
\\bullet \\ g_3 = (0.1998973 - 3, -0.1998335 + 2) = (-2.80010,1.80017)\\\\[4pt]
\\bullet \\ m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80010, 1.80017) = (-0.78401, 0.513017)\\\\[4pt]
\\bullet \\ v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot ((-2.80010, 1.80017) \\odot (-2.80010, 1.80017)) = (0.0252242, 0.010839)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_3 = \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78401,0.513017)}{1 - 0.9^3} = (-2.89303, 1.89305) \\\\[4pt]
\\hat{v}_3 = \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252242,0.010839)}{1 - 0.999^3} = (8.41647,3.61661)\\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_3 = \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon} \\hat{m}_3 = (0.1998973,-0.1998335) - \\frac{0.1}{\\sqrt{(8.41647,3.61661)}} \\cdot (-2.89303, 1.89305) \\\\[4pt]
= (0.2996185,-0.2993766)\\\\[4pt]
$$

&nbsp;

This example shows how Adam combines the best ideas of RMSProp and SGD+momentum to get a more effective optimizer.
$$
\\bullet \\ \\textbf{Momentum: } \\hat{m}_t \\text{ is a smoothed version of the raw gradient, reducing noise and zigzagging. } \\\\[4pt]
\\bullet \\ \\textbf{Per-parameter adaptivity: } \\hat{v}_t \\text{ differs across coordinates, so each parameter get its own effective step size } \\frac{\\eta}{\\sqrt{\\hat{v}_{t,i}} + \\epsilon}.
$$

&nbsp;

---

&nbsp;

$\\textbf{Bias correction:}\\\\[4pt]$

Adam keeps **EMA (exponential moving average)** of:
$$
\\bullet \\ \\text{the gradient } m_t \\text{ (momentum), and } \\\\[4pt]
\\bullet \\ \\text{the squared gradient } v_t \\text{ (RMS scaling)}.
$$

If both EMAs start at zero ($m_0 = 0, v_0 = 0$). This makes the bias estimation large at the early steps, especially with common $\\beta_1 = 0.9, \\beta_2 = 0.999$.

&nbsp;

For example, for a constant gradient $g$, the first step would be:
$$
\\bullet \\ m_1 = \\beta_1 m_0 + (1 - \\beta_1)g \\ \\ \\ (m_1 \\text{ starts at 0.1}g), \\\\[4pt]
\\bullet \\ v_1 = \\beta_2 v_0 + (1 - \\beta_2)g^2 \\ \\ \\ (v_1 \\text{ starts at 0.001}g^2). \\\\[4pt]
$$

So without bias correction, step 1 would use 0.1X the intended numerator and $\\sqrt{0.001} \\approx 0.0316$X the intended denominator. The raw (uncorrected) step magnitude scales like
$$
\\eta \\cdot \\frac{m_1}{\\sqrt{v_1}} = \\eta \\cdot \\frac{0.1g}{0.0316g} = \\eta \\cdot 3.16 = \\eta \\cdot F_1
$$

So with common setup $\\beta_1 = 0.9, \\beta_2 = 0.999, m_0 = 0, v_0 = 0$:

$\\bullet \\ $ $F_1 = 3.16$, the first step is large by a factor of 3.16.

$\\bullet \\ $ Use similar reasoning we can get $F_2 = 4.25, F_3 = 4.95, F_4 = 5.44.$

$\\bullet \\ $ At $t = 12$, we reach a high of $F_{12} = 6.568$, and from there it descends to 1.

&nbsp;

On the other hand, with bias correction, $\\hat{m}_1 = g_1$, and $\\hat{v}_1 = g_1^2$,
the $\\frac{\\hat{m}_1}{\\sqrt{\\hat{v}_1}}$ term becomes less sensitive to the choice of $\\beta$ and the initial value of $m_0$ and $v_0$.

&nbsp;

$\\textbf{The fix:}\\\\[4pt]$

Adam fixes this by dividing by:
$$
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\[4pt]
$$

For the complete proof, refer to section 3 in the original paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980).

&nbsp;

$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. Adam: A Method for Stochastic Optimization}$$
Original paper of Adam.
$$\\href{https://arxiv.org/pdf/1412.6980}{\\texttt{Adam: A Method for Stochastic Optimization}}$$

$$\\textbf{References:}\\\\[4pt]$$
$$\\textbf{1. StackExchange Discussion}$$
Discussion about the bias correction in Adam.
$$\\href{https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for}{\\texttt{StackExchange Discussion}}$$


`,
  },

  adamw_notes: {
    title: "AdamW",
    md: `

AdamW = Adam (momentum + RMS scaling + bias correction) plus **weight decay** applied separately:

$$
m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\ \\ \\ \\ \\text{(EMA of gradients: momentum)} \\\\[4pt]
v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) (g_t \\odot g_t) \\ \\ \\ \\ \\text{(EMA of squared gradients: RMS scaling)} \\\\[4pt]
\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\ \\ \\ \\ \\text{(bias correction)} \\\\[4pt]
\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\ \\ \\ \\ \\text{(bias correction)} \\\\[4pt]
\\theta_t = (1 - \\eta \\lambda) \\theta_t \\ \\ \\ \\ \\text{(decoupled weight decay)} \\\\[4pt]
\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t \\ \\ \\ \\ \\text{(adaptive step per parameter)} \\\\[4pt]
$$

Where $\\beta_1$ and $\\beta_2$ are the momentum and RMS scaling coefficients, respectively.
$\\epsilon$ is a small constant to avoid division by zero. $\\lambda$ is the weight decay strength (e.g. 0.1).

&nbsp;

---

&nbsp;

$\\textbf{Tiny numeric example:}\\\\[4pt]$

$\\textbf{Setup:}\\\\[4pt]$
$$
\\bullet \\ \\text{Parameters: } \\theta = (\\theta_1, \\theta_2)\\\\[4pt]
\\bullet \\ \\text{Loss: } L = \\frac{1}{2} [(\\theta_1 - 3)^2 + (\\theta_2 + 2)^2]  \\\\[4pt]
\\bullet \\ \\text{Gradients: } g = (g_1, g_2) = (\\theta_1 - 3, \\theta_2 + 2) \\\\[4pt]
\\bullet \\ \\text{Updates rule: } \\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t \\\\[4pt]
\\bullet \\ \\text{Hyperparams: } \\eta = 0.1, \\beta_1 = 0.9, \\beta_2 = 0.999, \\epsilon = 1e-8, \\lambda = 0.1 \\text{ tiny so numerically negligible} \\\\[4pt]
\\bullet \\ \\text{Init: } \\theta_0 = (0, 0), m_0 = (0, 0), v_0 = (0, 0)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 1:}\\\\[4pt]$
$$
\\bullet \\ \\theta_0 = (0, 0)\\\\[4pt]
\\bullet \\ g_1 = (-3, 2)\\\\[4pt]
\\bullet \\ m_1 = 0.9 \\cdot (0, 0) + (1 - 0.9) \\cdot (-3, 2) = (-0.3, 0.2)\\\\[4pt]
\\bullet \\ v_1 = 0.999 \\cdot (0, 0) + (1 - 0.999) \\cdot ((-3, 2) \\odot (-3, 2)) = (0.009, 0.004)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_1 = \\frac{m_1}{1 - \\beta_1^1} = \\frac{(-0.3,0.2)}{1 - 0.9^1} = (-3,2) \\\\[4pt]
\\hat{v}_1 = \\frac{v_1}{1 - \\beta_2^1} = \\frac{(0.009,0.004)}{1 - 0.999^1} = (9, 4)\\\\[4pt]
\\bullet \\ \\textbf{Decay: } \\theta_0 = (1 - \\eta \\lambda) \\theta_0 = (1 - 0.1 \\cdot 0.1) \\cdot (0, 0) = (0, 0) \\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_1 = \\theta_0 - \\frac{\\eta}{\\sqrt{\\hat{v}_1} + \\epsilon} \\hat{m}_1 = (0, 0) - \\frac{0.1}{\\sqrt{(9,4)}} \\cdot (-3, 2) = (0.1, -0.1)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 2:}\\\\[4pt]$
$$
\\bullet \\ \\theta_1 = (0.1, -0.1)\\\\[4pt]
\\bullet \\ g_2 = (0.1 - 3, -0.1 + 2) = (-2.9, 1.9)\\\\[4pt]
\\bullet \\ m_2 = 0.9 \\cdot (-0.3, 0.2) + (1 - 0.9) \\cdot (-2.9, 1.9) = (-0.56, 0.37)\\\\[4pt]
\\bullet \\ v_2 = 0.999 \\cdot (0.009, 0.004) + (1 - 0.999) \\cdot ((-2.9, 1.9) \\odot (-2.9, 1.9)) = (0.017401, 0.007606)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_2 = \\frac{m_2}{1 - \\beta_1^2} = \\frac{(-0.56,0.37)}{1 - 0.9^2} = (-2.94737, 1.94737) \\\\[4pt]
\\hat{v}_2 = \\frac{v_2}{1 - \\beta_2^2} = \\frac{(0.017401,0.007606)}{1 - 0.999^2} = (8.70485,3.80490)\\\\[4pt]
\\bullet \\ \\textbf{Decay: } \\theta_1 = (1 - \\eta \\lambda) \\theta_1 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1, -0.1) = (0.099, -0.099) \\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_2 = \\theta_1 - \\frac{\\eta}{\\sqrt{\\hat{v}_2} + \\epsilon} \\hat{m}_2 = (0.099, -0.099) - \\frac{0.1}{\\sqrt{(8.70485,3.80490)}} \\cdot (-2.94737, 1.94737) \\\\[4pt]
= (0.1988973,-0.1988335)\\\\[4pt]
$$

&nbsp;

$\\textbf{Step 3:}\\\\[4pt]$
$$
\\bullet \\ \\theta_2 = (0.1988973,-0.1988335)\\\\[4pt]
\\bullet \\ g_3 = (0.1988973 - 3, -0.1988335 + 2) = (-2.80110,1.80117)\\\\[4pt]
\\bullet \\ m_3 = 0.9 \\cdot (-0.56, 0.37) + (1 - 0.9) \\cdot (-2.80110, 1.80117) = (-0.78411, 0.513117)\\\\[4pt]
\\bullet \\ v_3 = 0.999 \\cdot (0.017401, 0.007606) + (1 - 0.999) \\cdot ((-2.80110, 1.80117) \\odot (-2.80110, 1.80117)) = (0.0252298, 0.0108426)\\\\[4pt]
\\bullet \\ \\textbf{Bias correction: } \\\\[4pt]
\\hat{m}_3 = \\frac{m_3}{1 - \\beta_1^3} = \\frac{(-0.78411,0.513117)}{1 - 0.9^3} = (-2.893395, 1.89342) \\\\[4pt]
\\hat{v}_3 = \\frac{v_3}{1 - \\beta_2^3} = \\frac{(0.0252298,0.0108426)}{1 - 0.999^3} = (8.418349,3.617817)\\\\[4pt]
\\bullet \\ \\textbf{Decay: } \\theta_2 = (1 - \\eta \\lambda) \\theta_2 = (1 - 0.1 \\cdot 0.1) \\cdot (0.1988973,-0.1988335) = (0.196908,-0.196845) \\\\[4pt]
\\bullet \\ \\textbf{Update: } \\theta_3 = \\theta_2 - \\frac{\\eta}{\\sqrt{\\hat{v}_3} + \\epsilon} \\hat{m}_3 = (0.196908,-0.196845) - \\frac{0.1}{\\sqrt{(8.418349,3.617817)}} \\cdot (-2.893395, 1.89342) \\\\[4pt]
= (0.2966312,-0.2963911)\\\\[4pt]
$$

    `,
  },
};
