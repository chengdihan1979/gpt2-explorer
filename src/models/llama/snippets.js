// LLaMA 3 8B reference code (minimal PyTorch, same style and names as the GPT-2 code where possible).
// Verified in PyTorch at a small size: shapes, causality, RoPE depends only on relative position,
// RMSNorm output has RMS 1, the training loop lowers the loss; at full size it has 8.03B parameters.

export const llamaTrainingCode = `from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataLoader:
    def __init__(self, batch_size: int, block_size: int, val_fraction: float):
        self.batch_size = batch_size
        self.block_size = block_size

        # at init load tokens from local file
        with open("input.txt", "r") as f:
            text = f.read()                   # str
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")  # BPE, 128,256 tokens
        ids = tokenizer.encode(text) # return the token ids: list of N ints
        ids = torch.tensor(ids, dtype=torch.long)    # (N,)
        n = int((1.0 - val_fraction) * len(ids))     # int, 90% of N
        self.train_ids = ids[:n]                     # (n,)
        self.val_ids = ids[n:]                       # (N - n,)

    def get_batch(self, split: str):
        data = self.train_ids if split == "train" else self.val_ids   # (n,)
        # B = batch_size = 4, T = block_size = 32
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))  # (B,) start positions
        x = torch.stack([data[i:i+self.block_size] for i in ix])      # (B, T)
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])  # (B, T), x shifted by one
        return x, y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))    # (C,), no bias

    def forward(self, x):                              # x: (B, T, C)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # (B, T, 1)
        return x / rms * self.weight                   # (B, T, C)


class LLaMA(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, n_kv_head, hidden, max_toks):
        super().__init__()
        # V = vocab_size = 128256, C = n_embd = 4096
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # weight (V, C); no position embedding
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, hidden=hidden, max_toks=max_toks)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)                       # weight (C,)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # weight (V, C), not tied to tok_emb

    # B = batch, T = sequence length, C = 4096, V = 128256
    def forward(self, idx, targets=None):    # idx: (B, T), targets: (B, T)
        x = self.tok_emb(idx)         # (B, T, C)
        for blk in self.blocks:
            x = blk(x)                # (B, T, C) -> (B, T, C)
        x = self.ln_f(x)              # (B, T, C)
        logits = self.lm_head(x)      # (B, T, C) @ (C, V) -> (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss           # (B, T, V), () scalar

# ----- Simple training loop -----
device = "cuda" if torch.cuda.is_available() else "cpu"
# LLaMA 3 8B
model = LLaMA(vocab_size=128256, n_layer=32, n_embd=4096, n_head=32, n_kv_head=8, hidden=14336, max_toks=8192).to(device)
# weight decay only on 2D weights; LLaMA has no biases, so no_decay is just the RMSNorm weights
decay    = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))

batch_size = 4
block_size = 32
val_fraction = 0.1
dataloader = DataLoader(batch_size=batch_size, block_size=block_size, val_fraction=val_fraction)
max_steps = 1000
for step in range(max_steps):
    x, y = dataloader.get_batch("train")    # x: (4, 32), y: (4, 32)
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)   # every p.grad -> None
    _, loss = model(x, y)                   # logits (4, 32, 128256), loss () scalar
    loss.backward()                         # each p.grad gets the same shape as p
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # rescale all grads so total norm <= 1.0
    optimizer.step()                        # each p updated in place, shape unchanged
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")
`;

export const blockCode = `import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, hidden, max_toks):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, max_toks=max_toks)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, hidden=hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
`;

export const attnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(hs, max_toks, theta=500000.0):
    # one rotation frequency per pair of dimensions: fast for the first pairs, slow for the last
    inv_freq = 1.0 / theta ** (torch.arange(0, hs, 2).float() / hs)   # (hs/2,)
    angles = torch.outer(torch.arange(max_toks).float(), inv_freq)    # (max_toks, hs/2): position × frequency
    return angles.cos(), angles.sin()                                 # each (max_toks, hs/2)


def apply_rope(x, cos, sin):                         # x: (B, nh, T, hs), cos/sin: (T, hs/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]              # each pair of dims (x1, x2): (B, nh, T, hs/2)
    out1 = x1 * cos - x2 * sin                       # rotate each pair by its angle
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)   # (B, nh, T, hs)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, max_toks):
        super().__init__()
        self.n_head = n_head                         # 32 query heads
        self.n_kv_head = n_kv_head                   # 8 key/value heads (GQA)
        hs = n_embd // n_head                        # 128 (4096 // 32)
        self.q_proj = nn.Linear(n_embd, n_head * hs, bias=False)     # weight (4096, C)
        self.k_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # weight (1024, C)
        self.v_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)  # weight (1024, C)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)          # weight (C, C)
        cos, sin = rope_cache(hs, max_toks)
        self.register_buffer("cos", cos)             # (max_toks, hs/2)
        self.register_buffer("sin", sin)

    def forward(self, x):                            # x: (B, T, C)
        B, T, C = x.size()
        hs = C // self.n_head                        # 128

        # nh = n_head (32), nkv = n_kv_head (8)
        q = self.q_proj(x).view(B, T, self.n_head, hs).transpose(1, 2)     # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_kv_head, hs).transpose(1, 2)  # (B, nkv, T, hs)
        v = self.v_proj(x).view(B, T, self.n_kv_head, hs).transpose(1, 2)  # (B, nkv, T, hs)

        q = apply_rope(q, self.cos[:T], self.sin[:T])  # position enters here, only into q and k
        k = apply_rope(k, self.cos[:T], self.sin[:T])

        rep = self.n_head // self.n_kv_head          # 4 query heads share each k/v head
        k = k.repeat_interleave(rep, dim=1)          # (B, nh, T, hs)
        v = v.repeat_interleave(rep, dim=1)          # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)                   # (B, nh, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))                # (T, T)
        att = att.masked_fill(mask == 0, float('-inf'))                     # (B, nh, T, T)
        att = F.softmax(att, dim=-1)                                        # (B, nh, T, T)
        y = att @ v                                                         # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                    # (B, T, C)
        return self.o_proj(y)                                               # (B, T, C)
`;

export const mlpCode = `import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_embd, hidden):
        super().__init__()
        # C = n_embd = 4096, H = hidden = 14336
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)   # weight (H, C)
        self.up_proj   = nn.Linear(n_embd, hidden, bias=False)   # weight (H, C)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)   # weight (C, H)

    def forward(self, x):                              # x: (B, T, C)
        gate = F.silu(self.gate_proj(x))               # (B, T, H)
        up = self.up_proj(x)                           # (B, T, H)
        return self.down_proj(gate * up)               # (B, T, H) -> (B, T, C)
`;

// One entry per diagram node: the code to show and the exact lines to highlight.
// Every highlight string must appear in its code; this is checked when the module loads.
export const content = {
  gpt2: {
    title: "LLaMA (model & training)",
    summary: "LLaMA 3 8B: RMSNorm, RoPE, grouped-query attention, SwiGLU, no biases.",
    code: llamaTrainingCode,
    highlight: ["class LLaMA(nn.Module):"],
  },
  input_data: {
    title: "Input Data",
    summary: "Read raw text from a file.",
    code: llamaTrainingCode,
    highlight: ['with open("input.txt", "r") as f:', "text = f.read()"],
  },
  tokenizer: {
    title: "Tokenizer",
    summary: "LLaMA 3's BPE tokenizer: 128,256 tokens (GPT-2: 50,257).",
    code: llamaTrainingCode,
    highlight: ['tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")', "ids = tokenizer.encode(text)"],
  },
  token_embed: {
    title: "Token Embeddings",
    summary: "Token ids to vectors. There is no position embedding: position enters through RoPE inside attention.",
    code: llamaTrainingCode,
    highlight: ["self.tok_emb = nn.Embedding(vocab_size, n_embd)", "x = self.tok_emb(idx)"],
  },
  stack: {
    title: "Transformer Block",
    summary: "Pre-norm block, as in GPT-2, with RMSNorm, attention and a SwiGLU MLP.",
    code: blockCode,
    highlight: ["class TransformerBlock(nn.Module):"],
  },
  block_ln: {
    title: "RMSNorm before Attention",
    summary: "Normalizes by the root mean square only: no mean subtraction, no bias.",
    code: blockCode,
    highlight: ["self.ln1 = RMSNorm(n_embd)", "self.ln1(x)"],
  },
  block_attn: {
    title: "Grouped-Query Attention",
    summary: "32 query heads share 8 key/value heads.",
    code: blockCode,
    highlight: ["self.attn = CausalSelfAttention(", "x = x + self.attn(self.ln1(x))"],
  },
  block_attn_plus: {
    title: "Attention residual",
    summary: "Add the attention output back to x.",
    code: blockCode,
    highlight: ["x = x + self.attn(self.ln1(x))"],
  },
  block_ln2: {
    title: "RMSNorm before MLP",
    summary: "Normalization before the SwiGLU MLP.",
    code: blockCode,
    highlight: ["self.ln2 = RMSNorm(n_embd)", "self.ln2(x)"],
  },
  block_mlp: {
    title: "SwiGLU MLP",
    summary: "Gated MLP: SiLU(gate) × up, then down.",
    code: blockCode,
    highlight: ["self.mlp = MLP(n_embd=n_embd, hidden=hidden)", "x = x + self.mlp(self.ln2(x))"],
  },
  block_mlp_plus: {
    title: "MLP residual",
    summary: "Add the MLP output back to x.",
    code: blockCode,
    highlight: ["x = x + self.mlp(self.ln2(x))"],
  },
  attn_class: {
    title: "Grouped-Query Attention",
    summary: "Causal self-attention with RoPE and shared key/value heads.",
    code: attnCode,
    highlight: ["class CausalSelfAttention(nn.Module):"],
  },
  attn_linear_qkv: {
    title: "Linear for Q, K, V",
    summary: "Three projections without bias; K and V produce only 8 heads (1024 dims) instead of 32.",
    code: attnCode,
    highlight: [
      "self.q_proj = nn.Linear(n_embd, n_head * hs, bias=False)",
      "self.k_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)",
      "self.v_proj = nn.Linear(n_embd, n_kv_head * hs, bias=False)",
      "q = self.q_proj(x).view(B, T, self.n_head, hs).transpose(1, 2)",
      "k = self.k_proj(x).view(B, T, self.n_kv_head, hs).transpose(1, 2)",
      "v = self.v_proj(x).view(B, T, self.n_kv_head, hs).transpose(1, 2)",
    ],
  },
  attn_rope: {
    title: "RoPE (rotary position embedding)",
    summary: "Rotates each pair of dimensions of q and k by an angle that grows with the position.",
    code: attnCode,
    highlight: [
      "def rope_cache(hs, max_toks, theta=500000.0):",
      "def apply_rope(x, cos, sin):",
      "q = apply_rope(q, self.cos[:T], self.sin[:T])",
      "k = apply_rope(k, self.cos[:T], self.sin[:T])",
    ],
  },
  attn_repeat_kv: {
    title: "Repeat K, V heads (GQA)",
    summary: "Each of the 8 key/value heads is shared by 4 query heads.",
    code: attnCode,
    highlight: ["rep = self.n_head // self.n_kv_head", "k = k.repeat_interleave(rep, dim=1)", "v = v.repeat_interleave(rep, dim=1)"],
  },
  attn_matmul_qk: {
    title: "Matmul (Q × Kᵀ)",
    summary: "Scaled dot products, as in GPT-2.",
    code: attnCode,
  },
  attn_masked: {
    title: "Masked Attention",
    summary: "Causal mask, as in GPT-2.",
    code: attnCode,
    highlight: ["mask = torch.tril(torch.ones(T, T, device=x.device))", "att = att.masked_fill(mask == 0, float('-inf'))"],
  },
  attn_softmax: {
    title: "Softmax",
    summary: "Attention weights, as in GPT-2 (no attention dropout).",
    code: attnCode,
  },
  attn_matmul_v: {
    title: "Matmul (att × V)",
    summary: "Weighted sum of values, as in GPT-2.",
    code: attnCode,
  },
  attn_linear_out: {
    title: "Linear (output projection)",
    summary: "Merge the heads and project back, without bias or residual dropout.",
    code: attnCode,
    highlight: ["self.o_proj = nn.Linear(n_embd, n_embd, bias=False)", "return self.o_proj(y)"],
  },
  mlp_subdiagram: {
    title: "SwiGLU MLP",
    summary: "down( SiLU(gate(x)) ⊙ up(x) ), hidden size 14336.",
    code: mlpCode,
    highlight: ["class MLP(nn.Module):"],
  },
  mlp_linear_gate: {
    title: "Linear (gate)",
    summary: "C → H projection whose output decides how much of each hidden unit passes.",
    code: mlpCode,
    highlight: ["self.gate_proj = nn.Linear(n_embd, hidden, bias=False)", "self.gate_proj(x)"],
  },
  mlp_linear_up: {
    title: "Linear (up)",
    summary: "C → H projection carrying the values to be gated.",
    code: mlpCode,
    highlight: ["self.up_proj   = nn.Linear(n_embd, hidden, bias=False)", "up = self.up_proj(x)"],
  },
  mlp_silu: {
    title: "SiLU",
    summary: "SiLU(z) = z · sigmoid(z), applied to the gate.",
    code: mlpCode,
    highlight: ["gate = F.silu(self.gate_proj(x))"],
  },
  mlp_mul: {
    title: "Multiply (⊙)",
    summary: "Element-wise product of the gate and up branches.",
    code: mlpCode,
    highlight: ["gate * up"],
  },
  mlp_linear_down: {
    title: "Linear (down)",
    summary: "H → C projection back to the residual stream.",
    code: mlpCode,
    highlight: ["self.down_proj = nn.Linear(hidden, n_embd, bias=False)", "return self.down_proj(gate * up)"],
  },
  final_ln: {
    title: "Final RMSNorm",
    summary: "Normalize before the LM head.",
    code: llamaTrainingCode,
    highlight: ["class RMSNorm(nn.Module):", "self.ln_f = RMSNorm(n_embd)", "x = self.ln_f(x)"],
  },
  lm_head: {
    title: "LM Head",
    summary: "Projection to vocabulary logits; its own weight, not tied to the token embedding.",
    code: llamaTrainingCode,
    highlight: ["self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)", "logits = self.lm_head(x)"],
  },
  loss_fn: {
    title: "Training Loss",
    summary: "Cross-entropy on next-token targets, as in GPT-2.",
    code: llamaTrainingCode,
    highlight: ["loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))"],
  },
};

for (const [id, entry] of Object.entries(content)) {
  for (const h of entry.highlight || []) {
    if (!entry.code.includes(h)) throw new Error(`LLaMA content "${id}": highlight not found in its code: ${h}`);
  }
}
