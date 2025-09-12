export const gpt2TrainingCode = `import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataLoader:
    def __init__(self, batch_size: int, block_size: int, val_fraction: float):
        self.batch_size = batch_size
        self.block_size = block_size
        
        # at init load tokens from local file
        with open("input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(text) # return the token ids
        n = int((1.0 - val_fraction) * len(ids))
        self.train_ids = ids[:n]
        self.val_ids = ids[n:]
        
    def get_batch(self, split: str):
        data = self.train_ids if split == "train" else self.val_ids
        if len(data) <= self.block_size + 1:
            # pad by cycling if dataset is too small
            data = data.repeat((self.block_size * 2) // max(1, len(data)) + 1)
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])
        return x, y;
        

class GPT2(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, max_toks, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_toks, n_embd)
        self.emb_drop = nn.Dropout(dropout) # embedding dropout
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying: LM head shares weights with token embeddings
        self.lm_head.weight = self.tok_emb.weight

    # Dimension changes during forward
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.emb_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----- Simple training loop -----
model = GPT2(vocab_size=50304, n_layer=12, n_embd=768, n_head=12, max_toks=1024, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

batch_size = 4
block_size = 32
val_fraction = 0.1
dataloader = DataLoader(batch_size=batch_size, block_size=block_size, val_fraction=val_fraction)
max_steps = 1000
for step in range(max_steps):
    x, y = dataloader.get_batch("train")
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")
`;

export const dataLoaderCode = `
import tiktoken


class DataLoader:
    def __init__(self, batch_size: int, block_size: int, val_fraction: float):
        self.batch_size = batch_size
        self.block_size = block_size
        
        # at init load tokens from local file
        with open("input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(text)
        n = int((1.0 - val_fraction) * len(ids))
        self.train_ids = ids[:n]
        self.val_ids = ids[n:]
        
    def get_batch(self, split: str):
        data = self.train_ids if split == "train" else self.val_ids
        if len(data) <= self.block_size + 1:
            # pad by cycling if dataset is too small
            data = data.repeat((self.block_size * 2) // max(1, len(data)) + 1)
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])
        return x, y;
`;

export const attnCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.proj  = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y
`;

export const mlpCode = `import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd)
        try:
            self.act = nn.GELU(approximate="tanh")
        except TypeError:
            self.act = nn.GELU()
        self.fc_out = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc_out(self.act(self.fc_in(x))))
`;

export const transformerBlockCode = `import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
`;

export const content = {
  gpt2: {
    title: "GPT-2 (model & training)",
    summary: "Full model definition and a minimal training loop.",
    code: gpt2TrainingCode,
    anchors: [
      { id: "emb_drop_def",  label: "define drop", match: "self.emb_drop = nn.Dropout(dropout) # embedding dropout" },
      { id: "emb_drop_apply",label: "apply drop",  match: "x = self.emb_drop(x)" },
      { id: "lnf_def",   label: "final ln def", match: "self.ln_f = nn.LayerNorm(n_embd)" },
      { id: "lnf_apply", label: "apply ln_f",   match: "x = self.ln_f(x)" },
      { id: "lm_head_def",   label: "lm_head def",   match: "self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)" },
      { id: "lm_head_apply", label: "apply lm_head", match: "logits = self.lm_head(x)" },
      { id: "blocks_assign", label: "blocks assign", match: "self.blocks = nn.ModuleList([" },
      { id: "blocks_ctor",   label: "TransformerBlock call", match: "TransformerBlock(n_embd=n_embd, n_head=n_head, dropout=dropout)" },
      { id: "blocks_for",    label: "for loop",   match: "for _ in range(n_layer)" },
      { id: "blocks_close",  label: "close",      match: "])" },
      { id: "blocks_use_for",   label: "iterate blocks", match: "for blk in self.blocks:" },
      { id: "blocks_use_apply", label: "apply block",    match: "x = blk(x)" },
    ],
  },
  input_data: {
    title: "Input Data",
    summary: "Raw text strings prior to tokenization.",
    code: gpt2TrainingCode,
    anchors: [
      { id: "open_file", label: "open file", match: 'with open("input.txt", "r") as f:' },
      { id: "read_text", label: "read text", match: "text = f.read()" },
    ],
  },
  tokenizer: {
    title: "Tokenizer",
    summary: "Converts text to token IDs (BPE for GPT-2).",
    code: gpt2TrainingCode,
    anchors: [
      { id: "load_tok", label: "load tokenizer", match: "tokenizer = tiktoken.get_encoding(\"gpt2\")" },
      { id: "encode", label: "encode", match: "ids = tokenizer.encode(text)" },
    ],
  },
  token_embed: { 
    title: "Token Embeddings", 
    summary: "Maps token indices to dense vectors.", 
    code: gpt2TrainingCode, 
    anchors: [
      { id: "tok_emb",   label: "tok_emb",   match: "self.tok_emb = nn.Embedding(vocab_size, n_embd)" },
      { id: "tok_apply", label: "use tok",   match: "self.tok_emb(idx)" },
    ],
  },
  pos_embed:   { 
    title: "Positional Embeddings", 
    summary: "Learned positional embeddings.", 
    code: gpt2TrainingCode, 
    anchors: [
      { id: "pos_emb",   label: "pos_emb",   match: "self.pos_emb = nn.Embedding(max_toks, n_embd)" },
      { id: "pos_apply", label: "use pos",   match: "self.pos_emb(pos)[None, :, :]" },
    ] 
  },
  emb_dropout: { title: "Embedding Dropout", summary: "Dropout applied to token+pos embeddings before Transformer blocks.", code: "", anchors: [] },
  stack: {
    title: "Transformer Block",
    summary: "GPT-2 repeats the Transformer Block many times.",
    code: transformerBlockCode,
    anchors: [
      { id: "class_def", label: "class", match: "class TransformerBlock(nn.Module):" },
      { id: "forward",  label: "forward", match: "def forward(self, x):" },
    ]
  },
  block_ln:  { title: "LayerNorm before Attention", summary: "Normalization before self-attention.", code: transformerBlockCode, anchors: [
    { id: "ln1", label: "ln1", match: "self.ln1 = nn.LayerNorm(n_embd)" },
    { id: "ln1_use", label: "ln1 use", match: "self.ln1(x)" },
  ] },
  block_attn:{ title: "Masked Multi-Head Attention", summary: "Performs causal self-attention.", code: transformerBlockCode, anchors: [
    { id: "attn_assign", label: "self.attn assignment", match: "self.attn = CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)" },
    { id: "attn_call", label: "attn call", match: "self.attn(self.ln1(x))" },
  ] },
  attn_class:{ title: "Causal Self-Attention", summary: "Implementation of masked self-attention.", code: attnCode, anchors: [{ id: "class_def", label: "class", match: "class CausalSelfAttention(nn.Module):" }, { id: "mask", label: "mask", match: 'self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)).unsqueeze(0).unsqueeze(0))' }] },
  attn_linear_qkv: {
    title: "Linear for Q, K, V",
    summary: "Fused projection to compute qkv",
    code: attnCode,
    anchors: [
      { id: "c_attn_def", label: "define c_attn", match: "self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)" },
      { id: "c_attn_call", label: "call c_attn", match: "qkv = self.c_attn(x)  # [B, T, 3C]" },
      { id: "qkv_split", label: "split qkv", match: "q, k, v = qkv.split(C, dim=2)" },
    ],
  },
  attn_matmul_qk: {
    title: "Matmul (Q × Kᵀ)",
    summary: "Scaled dot-product before masking",
    code: attnCode,
    anchors: [],
  },
  attn_masked: {
    title: "Masked Attention",
    summary: "Apply causal mask to attention logits",
    code: attnCode,
    anchors: [],
  },
  attn_softmax: {
    title: "Softmax",
    summary: "Normalize logits over keys",
    code: attnCode,
    anchors: [],
  },
  attn_dropout: {
    title: "Attention dropout",
    summary: "Regularization on attention probabilities",
    code: attnCode,
    anchors: [
      { id: "attn_drop_def", label: "define attn_drop", match: "self.attn_drop  = nn.Dropout(dropout)" },
      { id: "attn_drop_call", label: "apply attn_drop", match: "att = self.attn_drop(att)" },
    ],
  },
  attn_matmul_v: {
    title: "Matmul (attn × V)",
    summary: "Weighted sum of values by attention",
    code: attnCode,
    anchors: [],
  },
  attn_linear_out: {
    title: "Linear",
    summary: "Project back to model dimension",
    code: attnCode,
    anchors: [
      { id: "proj_def", label: "define proj", match: "self.proj  = nn.Linear(n_embd, n_embd, bias=True)" },
      { id: "proj_call", label: "apply proj", match: "self.proj(y)" },
    ],
  },
  attn_resid_dropout: {
    title: "Residual dropout",
    summary: "Dropout applied after projection (residual path)",
    code: attnCode,
    anchors: [
      { id: "resid_drop_def", label: "define resid_drop", match: "self.resid_drop = nn.Dropout(dropout)" },
      { id: "resid_drop_call", label: "apply resid_drop", match: "self.resid_drop(" },
    ],
  },
  block_ln2: { title: "LayerNorm before MLP", summary: "Normalization before feed-forward MLP.", code: transformerBlockCode, anchors: [
    { id: "ln2", label: "ln2", match: "self.ln2 = nn.LayerNorm(n_embd)" },
    { id: "ln2_use", label: "ln2 use", match: "self.ln2(x)" },
  ] },
  block_mlp: { title: "MLP / Feed-Forward", summary: "Two linear layers with GELU in between.", code: transformerBlockCode, anchors: [
    { id: "mlp_assign", label: "self.mlp assignment", match: "self.mlp = MLP(n_embd=n_embd, dropout=dropout)" },
    { id: "mlp_call", label: "mlp call", match: "self.mlp(self.ln2(x))" },
  ] },
  final_ln:  { title: "Final LayerNorm", summary: "Normalizes hidden states.", code: "", anchors: [] },
  lm_head:   { title: "LM Head (Linear Layer)", summary: "Linear projection to vocab logits.", code: "", anchors: [] },
  loss_fn: {
    title: "Training Loss",
    summary: "CrossEntropy loss for next-token prediction.",
    code: gpt2TrainingCode,
    anchors: [{ id: "loss_line", label: "loss line", match: "loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))" }],
  },
  mlp_subdiagram: {
    title: "MLP Architecture",
    summary: "Linear → GELU → Linear → Dropout.",
    code: mlpCode,
    anchors: [
      { id: "mlp_linear1", label: "fc_in", match: "self.fc_in = nn.Linear(n_embd, 4 * n_embd)" },
      { id: "mlp_fc_in_use", label: "fc_in(x) use", match: "self.fc_in(x)" },
      { id: "mlp_gelu", label: "act", match: '        try:\n            self.act = nn.GELU(approximate="tanh")\n        except TypeError:\n            self.act = nn.GELU()' },
      { id: "mlp_gelu_in_use", label: "act use", match: 'self.act(self.fc_in(x))' },
      { id: "mlp_linear2", label: "fc_out", match: "self.fc_out = nn.Linear(4 * n_embd, n_embd)" },
      { id: "mlp_fc_out_use", label: "fc_out use", match: "self.fc_out(self.act(self.fc_in(x)))" },
      { id: "mlp_dropout", label: "dropout", match: "self.drop = nn.Dropout(dropout)" },
      { id: "mlp_dropout_in_use", label: "dropout use", match: "self.drop(self.fc_out(self.act(self.fc_in(x))))" },
    ],
  },
};
