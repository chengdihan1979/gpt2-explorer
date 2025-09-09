import { ORIGINAL_INFO } from "../GPT2DecoderExplorer";

const prevents_vanishing_exploding_md = `
$$
\\textbf{Two concrete toy examples that show why pre-LN keep gradients from vanishing/exploding in deep stack}\\\\[10pt]
$$

---

$$
\\textbf{Example 1 — “Safe at init” path (identity highway)}\\\\[6pt]
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
$$
\\textbf{Library \\& call}\\\\[4pt]
\\mathtt{tokenizer = tiktoken.get\\_encoding("gpt2")}\\;\\to\\;\\text{ returns a built-in }\\mathtt{tiktoken.Encoding}\\\\[10pt]
\\textbf{What the object is}\\\\[4pt]
\\text{GPT-2's byte-level BPE (BBPE) tokenizer; vocab size }50{,}257\\text{ (includes }\\texttt{<|endoftext|>}\\text{ with id }50256\\text{).}\\\\
\\text{Holds the byte}\\leftrightarrow\\text{\\href{#info:gpt2_byte_unicode_map_notes}{\\text{unicode map}}, \\href{#info:gpt2_merge_ranks_notes}{\\text{GPT-2 merge ranks}}, \\href{#info:regex_pre_tokenizer_notes}{\\text{regex pre-tokenizer}}, and special tokens.}\\\\[10pt]
\\textbf{How tokenization works}\\\\[4pt]
1)\\ \\text{Convert input text to UTF-8 bytes (case/whitespace preserved).}\\\\
2)\\ \\text{Split bytes with GPT-2's regex into chunks.}\\\\
3)\\ \\text{Apply BPE merges to produce token ids.}\\\\[10pt]
\\textbf{API you can use}\\\\[4pt]
\\mathtt{enc = tiktoken.get\\_encoding("gpt2")}\\\\
\\mathtt{ids = enc.encode("Hello\\ world!")}\\;\\to\\;\\text{ list of ints}\\\\
\\mathtt{text = enc.decode(ids)}\\;\\to\\;\\href{#info:lossless_round_trip_notes}{\\text{lossless round-trip}}\\\\
\\mathtt{enc.encode("<|endoftext|>")}\\;\\text{ raises by default; allow via }\\mathtt{allowed\\_special=\\{"all"\\}}\\\\[10pt]
\\textbf{Gotchas}\\\\[4pt]
\\text{Leading spaces/punctuation matter (byte-level, space-sensitive).} \\\\[6pt]
\\text{No BOS auto-insert; }\\texttt{<|endoftext|>}\\text{ is special.}\\\\[6pt]
\\textbf{References:}\\\\[2pt] 
\\href{https://github.com/openai/tiktoken}{\\texttt{tiktoken}}\\\\[2pt]
\\href{https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s}{\\texttt{BPE - Let's build the GPT Tokenizer}}\\\\[2pt]
\\href{https://huggingface.co/learn/llm-course/en/chapter6/5}{\\texttt{Hugging Face - Byte-Pair Encoding tokenization}}
$$
    `,
  },

  gpt2_byte_unicode_map_notes: {
    title: "GPT-2 byte-unicode map",
    md: `
$$
\\boxed{\\begin{array}{l}
\\textbf{What it is}\\\\[2pt]
\\text{A fixed, 1-to-1 mapping between the 256 possible bytes (0-255) and 256 printable Unicode code points.}\\\\[8pt]
\\textbf{byte }\\to\\ \\textbf{unicode (“byte encoder”):}\\ \\text{ turns each raw byte into a single visible Unicode character.}\\\\
\\textbf{unicode }\\to\\ \\textbf{byte (“byte decoder”):}\\ \\text{ the exact inverse, so it’s perfectly reversible.}\\\\[12pt]
\\textbf{Why it exists:}\\\\[2pt]
\\text{Classical BPE operates on “characters.” GPT-2, however, is byte-level: it wants to handle any byte sequence}\\\\
\\text{(emoji, accents, binary-ish bytes, etc.) losslessly. Many bytes are control/whitespace and would confuse regex}\\\\
\\text{splitting or get normalized away. The map solves this by:}\\\\
1. \\text{giving every byte a safe, visible stand-in,}\\\\
2. \\text{avoiding collisions with whitespace/control characters,}\\\\
3. \\text{making regex pre-tokenization and BPE run on a “printable” string while staying reversible.}\\\\[12pt]
\\textbf{How it's constructed (conceptually)}\\\\[2pt]
\\bullet\\space \\text{Bytes that are already “nice printable” (e.g., ASCII and many Latin-1 characters) are mapped to themselves}\\\\
\\text{(same code point).}\\\\
\\bullet\\space \\text{The remaining bytes (e.g., space, tabs, newlines, etc.) are mapped to } \\textbf{distinct printable code points}\\\\
\\text{in higher Unicode ranges, ensuring they're visible and not whitespace.}\\\\
\\space\\space\\space\\text{Example:}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Byte 0x41 ('A')}\\rightarrow\\text{is mapped to standard-in 'A' (same character)}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Byte 0x20 (space)}\\rightarrow\\text{stand-in “Ġ” (U+0120) — that's why " world" encoded as Ġworld with a leading space.}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Bytes 0xC3 0xA9 (é)}\\rightarrow\\text{stand-ins something like Ã © (two Unicode chars, one per byte)}\\\\[2pt]
\\space\\space\\space\\bullet\\space\\text{Bytes 0xE2 0x80 0x94 (-)(em dash)}\\rightarrow\\text{three stand-ins (one per byte)}\\\\[6pt]
\\text{This yields two tables:}\\\\
\\mathtt{byte\\_encoder: \\{0..255 \\to one-char\\ unicode\\ strings\\}}\\\\
\\mathtt{byte\\_decoder: \\{that\\ unicode\\ char \\to original\\ byte\\}}\\\\[12pt]
\\textbf{Which bytes map to themselves?}\\\\[2pt]
\\text{In GPT-2's map, these do: }\\\\
\\text{33\\text{-}126 } \\; (\\text{printable ASCII excluding space}) \\; \\text{and } 161\\text{-}172,\\; 174\\text{-}255\\; (\\text{printable Latin-1}).\\\\[6pt]
\\text{Bytes outside those ranges (e.g., control chars }0\\text{-}31,\\; \\text{space }32,\\; 127\\text{-}160,\\; \\text{and }173) \\text{ get special stand-ins (e.g., }\\\\
\\mathrm{U+0100},\\ \\mathrm{U+0102}, \\ldots\\text{).}\\text{ That's why space (0x20) becomes } \\texttt{Ġ}\\text{.}\\\\[10pt]
\\space\\space\\space\\textbf{Emoji example (UTF-8: F0 9F 98 8A)}\\\\[4pt]
\\space\\space\\space\\bullet\\space\\texttt{0xF0} \\; (=240) \\; \\text{is in } 174\\text{-}255 \\Rightarrow \\text{ maps to itself (prints as } \\texttt{ð} \\text{ in Latin-1).}\\\\[4pt]
\\space\\space\\space\\bullet\\space\\texttt{0x9F}\\; (159),\\; \\texttt{0x98}\\; (152),\\; \\texttt{0x8A}\\; (138) \\; \\text{are not in the printable sets } \\Rightarrow \\text{ each maps to a printable stand-in}\\\\
\\space\\space\\space\\space\\space\\space\\text{Latin-Extended characters).}\\\\[6pt]
\\space\\space\\space\\text{So you get a mix: one “self” character plus three stand-ins—still one visible char per byte, fully reversible.}\\\\[10pt]
\\textbf{How it's used (encode path)}\\\\[2pt]
1.\\space\\text{Text }\\to\\ \\text{ UTF-8 bytes.}\\\\
2.\\space\\text{Map each byte } b \\text{ to its stand-in Unicode char } u = \\mathtt{byte\\_encoder}[b].\\\\
\\space\\space\\space\\space\\bullet\\space\\text{Example: " Hello" (leading space) }\\to\\ \\text{ bytes [0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F]}\\\\
\\quad \\to\\ \\text{ mapped string "ĠHello" (space became “Ġ”; ASCII letters map to themselves).}\\\\
\\space\\space\\space\\space\\bullet\\space\\text{Non-ASCII (e.g., é }\\to\\ \\text{ bytes C3 A9) becomes two mapped chars, one for each byte; both are visible stand-ins.}\\\\
3.\\space\\text{Run regex pre-tokenizer and BPE merges on this printable string.}\\\\[12pt]
\\textbf{Decoding (exact inverse)}\\\\[2pt]
1.\\space\\text{Take the merged “printable” token pieces }\\to\\ \\text{ concatenate to a string of mapped chars.}\\\\
2.\\space\\text{Map each char back to a byte via } \\mathtt{byte\\_decoder}.\\\\
3.\\space\\text{UTF-8 decode the bytes }\\to\\ \\text{ original text.}\\\\
\\space\\space\\space\\space\\text{This guarantees lossless round-trip: } \\mathtt{decode(encode(s)) == s}
\\end{array}}
$$

    `,
  },

  regex_pre_tokenizer_notes: {
    title: "Regex pre-tokenizer",
    md: `
$$
\\textbf{What it is:}\\;\\text{a regular-expression–based splitter that chops the input text into chunks before BPE merging happens.}\\\\
\\text{BPE merges are then applied within each chunk only, never across chunk boundaries.}\\\\[10pt]

\\textbf{Why it exists:}\\\\
\\text{Speed \\& determinism — fewer candidate merges, stable behavior.}\\\\
\\text{Semantics — keeps leading spaces with the following token (so you get tokens like “}\\texttt{Ġworld}\\text{”),}\\\\
\\text{groups letters together, groups numbers together, and groups punctuation/symbols together.}\\\\
\\text{Reversibility — pairs neatly with byte-level encoding for lossless round-trip.}\\\\[10pt]

\\textbf{Rough pattern GPT-2 uses (conceptually):}\\\\
\\text{contractions: }\\texttt{’s, ’t, ’re, ’ve, ’m, ’ll, ’d}\\\\
\\text{optional leading space + letters: }\\texttt{?\\p\\{L\\}+}\\\\
\\text{optional leading space + digits: }\\texttt{?\\p\\{N\\}+}\\\\
\\text{optional leading space + other symbols: }\\texttt{?[\\s\\p\\{L\\}\\p\\{N\\}]+}\\\\
\\text{whitespace runs (including end-of-string cases).}\\\\[10pt]

\\textbf{How it affects BPE:}\\\\
\\text{Start from bytes (byte-level BPE) }\\to\\ \\text{ apply the regex pre-tokenizer }\\to\\ \\text{ for each chunk, repeatedly}\\\\
\\text{merge the lowest-rank adjacent pair until no more merges apply.}\\\\
\\text{Merges do not cross chunk boundaries. That’s why " world" (with its leading space) is a single chunk where merges}\\\\
\\text{can happen internally, but it won’t merge with the preceding token.}\\\\[10pt]

\\textbf{Special tokens:}\\;\\text{handled separately (e.g., }\\texttt{<|endoftext|>}\\text{). They bypass normal splitting when allowed.}\\\\[10pt]

\\textbf{Examples:}\\\\[6pt]
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
\\textbf{Key idea}\\\\[4pt]
\\text{It first splits the UTF-8 bytes into chunks using a regex: letters together, digits together, and symbols together,}\\\\
\\text{keeping any leading space with the next chunk. Then BPE merges run inside each chunk only—they don't cross}\\\\
\\text{chunk boundaries.} \\\\[10pt]

\\textbf{References:}\\\\[2pt] 
\\href{https://www.youtube.com/watch?v=zduSFxRajkE&t=3457s}{\\texttt{Pre Tokenizer - Let's build the GPT Tokenizer}}\\\\[2pt]
$$

    `,
  },

  lossless_round_trip_notes: {
    title: "Lossless round-trip",
    md: `
$$
\\boxed{\\begin{array}{l}
\\textbf{Lossless round-trip:}\\\\[4pt]
\\text{Lossless round-trip” means if you encode text into token IDs and then decode those IDs back to text,}\\\\[1pt]
\\text{you get exactly the same bytes you started with—nothing is altered or lost.}\\\\[6pt]
\\text{Formally: }\\; \\mathtt{decode(encode(s)) = s}.\\\\[10pt]
\\textbf{Why true for GPT-2 (byte-level BPE):}\\\\[4pt]
1)\\ \\text{It operates on raw bytes (no lowercasing/normalization/space stripping).}\\\\
2)\\ \\text{Every byte sequence is representable.}\\\\
3)\\ \\text{Decode exactly inverts encode.}\\\\[10pt]
\\end{array}}
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
\\quad\\bullet\\;\\text{gives a gradient highway (there's an identity path, so gradients flow even if the attention sublayer is poorly scaled early in training)}\\\\[6pt]
\\quad\\bullet\\;\\text{lets the block learn a small perturbation around identity (if the attention output is near zero, the block behaves like a no-op),}\\\\
\\;\\;\\;\\;\\text{which makes deep stacks much easier to optimize.}\\\\[10pt]
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
$$
\\boxed{\\begin{array}{l}
\\textbf{Byte-Pair Encoding (BPE) refresher:}\\\\[4pt]
\\text{Start from a sequence of basic symbols (for GPT-2, bytes). A BPE model has a list of symbol-pair merges}\\\\
\\text{learned from a large corpus. The most frequent pair gets merge \\#0, the next gets \\#1, etc. This order is the}\\\\
\\text{merge ranks.}\\\\[10pt]
\\textbf{Why ranks matter:}\\\\[4pt]
\\text{During encoding, the tokenizer repeatedly looks at adjacent pairs and merges the pair with the lowest rank}\\\\
\\text{number (highest priority) that appears in the list. It updates the sequence and repeats until no mergeable pairs}\\\\
\\text{remain. The final merged chunks are your tokens.}\\\\[10pt]

\\textbf{What }\\mathtt{tiktoken}\\textbf{ stores:}\\\\[4pt]
\\text{GPT-2's tokenizer ships with a mapping that effectively represents these ranks. In practice:}\\\\
\\quad \\text{Lower rank } \\Rightarrow \\text{ earlier (more frequent) merge rule } \\Rightarrow \\text{ applied before higher-rank rules.}\\\\
\\quad \\text{This makes tokenization deterministic and matches GPT-2's training.}\\\\[10pt]

\\textbf{Relation to token IDs:}\\\\[4pt]
\\text{In GPT-2/}\\mathtt{tiktoken}\\text{, the learned merged strings correspond to entries in the vocabulary. The internal}\\\\
\\text{"rank" concept aligns with how tokens are prioritized/created; in }\\mathtt{tiktoken}\\text{ many of these ranks are used}\\\\
\\text{directly as token ids (special tokens handled separately).}\\\\[10pt]
\\textbf{Tiny toy example (made up):}\\\\[4pt]
\\text{Text: } \\mathtt{banana} \\;\\to\\; \\text{bytes} \\;\\to\\; \\text{initial symbols: } b\\ a\\ n\\ a\\ n\\ a\\\\[4pt]
\\text{Suppose merges (with ranks):}\\\\
\\text{rank 0: } (a,n) \\to an\\\\
\\text{rank 1: } (an,a) \\to ana\\\\
\\text{rank 2: } (b,an) \\to ban\\\\[4pt]
\\text{Encoding proceeds:}\\\\
\\text{Merge lowest-rank pairs present } \\to\\ b\\ an\\ a\\ n\\ a\\\\
\\text{Now } (an,a) \\text{ exists } \\to\\ b\\ ana\\ n\\ a\\\\
\\text{No more lower-rank merges available (or continue if others exist) } \\to\\ \\text{final tokens.}\\\\[10pt]
\\textbf{Space handling:}\\\\[4pt]
\\text{GPT-2 is byte-level. Space is part of the bytes and many tokens encode "space+word" as a single token}\\\\
\\text{(often shown with a leading \\texttt{Ġ} in visualizations). Those arise from specific merges and their ranks.}\\\\[10pt]
\\textbf{Bottom line:}\\\\[4pt]
\\text{Merge ranks are the ordered list of learned pair-merge rules that dictate exactly how GPT-2's byte-level BPE}\\\\
\\text{groups characters/bytes into tokens.}
\\end{array}}
$$
    `,
  },

  attention_linear_notes: {
    title: "Attention linear: nn.Linear(n_embd, 3 * n_embd, bias=True)",
    md: `
$$
\\boxed{\\begin{array}{l}
\\textbf{What it is}\\\\[4pt]
\\mathtt{self.c\\_attn = nn.Linear(n\\_embd,\\ 3 * n\\_embd,\\ bias=True)}\\ \\text{ is a single fused projection that computes }\\textbf{Q, K, V}\\ \\text{ in one matmul.}\\\\
\\text{Equivalent to stacking three }\\mathtt{Linear(n\\_embd, n\\_embd)}\\text{ layers:}\\\\[4pt]
W_{qkv} = \\begin{bmatrix} W_Q \\\\ W_K \\\\ W_V \\end{bmatrix},\\quad
b_{qkv} = \\begin{bmatrix} b_Q \\\\ b_K \\\\ b_V \\end{bmatrix}.\\\\[20pt]
\\textbf{Parameter shapes}\\\\[4pt]
\\text{Weight: } (3\\,n\\_{\\text{embd}},\\ n\\_{\\text{embd}}),\\quad
\\text{Bias: } (3\\,n\\_{\\text{embd}}).\\\\[12pt]
\\textbf{What it does to the tensor}\\\\[4pt]
\\text{Input } x \\in \\mathbb{R}^{B\\times T\\times n\\_{\\text{embd}}}.\\\\
\\mathtt{qkv = c\\_attn(x)}\\ \\Rightarrow\\ qkv \\in \\mathbb{R}^{B\\times T\\times 3n\\_{\\text{embd}}}.\\\\
\\mathtt{q,\\ k,\\ v = qkv.split(n\\_embd,\\ -1)}\\ \\Rightarrow\\ q,k,v \\in \\mathbb{R}^{B\\times T\\times n\\_{\\text{embd}}}.\\\\
\\text{Split into heads with } hs = n\\_{\\text{embd}}/n\\_{\\text{head}}:\\quad
\\mathtt{view(B, T, n\\_head, hs).transpose(1, 2)}\\ \\Rightarrow\\ (B,\\ n\\_{\\text{head}},\\ T,\\ hs).\\\\[12pt]
\\textbf{Why fuse Q/K/V?}\\\\[4pt]
\\text{Fewer kernel launches / better cache use (one big GEMM vs. three). Results are identical to three separate linears.}\\\\[12pt]
\\textbf{Parameter count}\\\\[4pt]
\\text{Weights: } 3\\times n\\_{\\text{embd}}^2,\\quad \\text{Bias: } 3\\times n\\_{\\text{embd}}.\\\\
\\text{Example } (n\\_{\\text{embd}}=768):\\ \\text{weights } 3\\cdot 768^2=1{,}769{,}472,\\ \\text{bias } 3\\cdot 768=2{,}304.\\\\[12pt]
\\textbf{PyTorch default initialization}\\\\[4pt]
\\text{For }\\mathtt{nn.Linear}\\text{ with }\\mathtt{bias=True}:\\\\
\\text{Default initialization: } W_{ij} \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right),\\quad
b_i \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{\\text{fan\\_in}}},\\ \\tfrac{1}{\\sqrt{\\text{fan\\_in}}}\\right),\\\\
\\text{with } \\text{fan\\_in}=n\\_{\\text{embd}}.\\ \\text{(For } n\\_{\\text{embd}}=768:\\ \\approx\\pm 0.0361\\text{.)}\\\\[12pt]
\\textbf{Historical GPT-2 initialization}\\\\[4pt]
W \\sim \\mathcal{N}(0,\\ 0.02^2),\\quad b=0.\\ \\text{(Re-init if you want to match GPT-2 exactly.)}\\\\[12pt]
\\textbf{Bias = True — what it means}\\\\[4pt]
\\text{Learns three separate bias vectors (for Q, K, V), concatenated into a single } (3\\times n\\_{\\text{embd}}) \\text{ parameter.}\\\\[12pt]
\\textbf{Naming note}\\\\[4pt]
\\text{The name }\\mathtt{c\\_attn}\\ \\text{comes from original OpenAI GPT-2 implementation where a small 1 Conv1D wrapper (1×1 conv = linear) was used. In PyTorch it's }\\mathtt{nn.Linear}.\\\\
\\text{but the name stay the same} \\\\[4pt]
\\end{array}}
$$
    `,
  },

  mlp_residual_notes: {
    title: "MLP residual: x = x + self.mlp(self.ln2(x))",
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
\\mathtt{x\\ \\to\\ \\mathrm{Linear}({n\\_embd}\\!\\to\\!4\\times {n\\_embd})\\ \\to\\ \\mathrm{GELU}\\ \\to\\ \\mathrm{Linear}(4\\times {n\\_embd}\\!\\to\\!{n\\_embd})\\ \\to\\ (\\mathrm{Dropout})}\\\\[4pt]
\\text{This expand}\\to\\text{nonlinearity}\\to\\text{project back pattern builds rich, non-linear feature interactions per token.}\\\\[8pt]
$$
$$
\\textbf{3. Residual (skip) connection.}\\\\[6pt]
\\mathtt{x + (\\ldots)}\\ \\text{adds the MLP's output back to the original stream. Benefits:}\\\\[6pt]
\\quad\\text{(i) Gradient highway: there is an identity path, helping \\href{#info:mlp_prevents_vanishing_exploding_notes}{\\text{prevent vanishing/exploding}} early in training.}\\\\[6pt]
\\quad\\text{(ii) Small refinements: if the MLP output is near zero, the block behaves like a near-identity map.}\\\\[10pt]
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
$$
\\textbf{A high-level analogy of a GPT-2 transformer block.}\\\\[8pt]
\\textbf{Tokens = people in a meeting.}\\\\
\\text{Each token is like a person holding their own notes (its current representation).}\\\\[8pt]
\\textbf{Attention = the conversation.}\\\\
\\text{Everyone briefly scans what others are saying and decides whose points matter for them. Each person builds a short}\\\\
\\text{"summary I need" by selectively listening.}\\\\
\\textit{Why it matters:}~\\text{this is how long-range connections form—facts from far away can instantly influence you.}\\\\[8pt]
\\textbf{MLP (feed-forward) = private thinking.}\\\\
\\text{After listening, each person goes off for a moment of inner reasoning: "Given what I heard, how should I update my notes?"}\\\\
\\textit{Why it matters:}~\\text{it lets each token transform and rephrase ideas, adding nonlinearity and capacity.}\\\\[8pt]
\\textbf{Two LayerNorms = the organizer's ground rules (applied twice).}\\\\
\\text{Before the conversation and before the private thinking, the organizer reminds everyone to keep comments calm and on-topic—}\\\\
\\text{no one's voice too loud, none too quiet.}\\\\
\\textit{Why it matters:}~\\text{it keeps both phases stable and comparable so the block behaves predictably.}\\\\[8pt]
\\textbf{Residual connections = a running draft.}\\\\
\\text{After each phase (conversation, then private thinking), you add your new insights to your previous notes rather than replacing them.}\\\\
\\textit{Why it matters:}~\\text{you never lose what you already knew; updates are incremental and safe.}\\\\[8pt]
\\textbf{Dropout = occasional silence to build resilience.}\\\\
\\text{Sometimes random remarks are withheld during practice meetings.}\\\\
\\textit{Why it matters:}~\\text{the group learns to make good decisions even if a few voices are missing, preventing overreliance on any single cue.}\\\\[10pt]
\\textbf{Putting it together:}\\\\
\\text{Each block is a two-step cycle—communicate (attention) then compute (MLP)—with the organizer (LayerNorm) keeping order,}\\\\
\\text{a running draft (residual) preserving history, and practice with partial attendance (dropout) to ensure robustness.}\\\\
\\text{Stacking many such blocks lets tokens repeatedly share, rethink, and refine.}
$$
    
`,
  },
};
