export function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function buildCodeHtml({ component, selectedId }) {
  if (!component?.code) return null;
  let html = escapeHtml(component.code);
  const anchors = component.anchors || [];
  const find = (id) => anchors.find((x) => x.id === id);
  const mark = (needle) => {
    const esc = escapeHtml(needle);
    html = html.replace(esc, `<mark>${esc}</mark>`);
  };

  // 1) Highlight anchors for LN/dropout in block
  if ((selectedId === "block_ln" || selectedId === "block_ln2" || selectedId === "block_dropout_attn" || selectedId === "block_dropout_mlp") && anchors.length) {
    anchors.forEach((a) => {
      if (
        (selectedId === "block_ln" && a.id === "ln1") ||
        (selectedId === "block_ln2" && a.id === "ln2") ||
        (selectedId === "block_dropout_attn" && a.id === "dropout1")
      ) {
        mark(a.match);
      }
    });
  } else if (["mlp_linear1", "mlp_linear2", "mlp_gelu", "mlp_dropout"].includes(selectedId) && anchors.length) {
    anchors.forEach((a) => {
      const shouldMark =
        (selectedId === "mlp_linear1" && (a.id === "mlp_linear1" || a.id === "mlp_fc_in_use")) ||
        (selectedId === "mlp_linear2" && (a.id === "mlp_linear2" || a.id === "mlp_fc_out_use")) ||
        (selectedId === "mlp_gelu" && (a.id === "mlp_gelu" || a.id === "mlp_gelu_in_use")) ||
        (selectedId === "mlp_dropout" && (a.id === "mlp_dropout" || a.id === "mlp_dropout_in_use"));
      if (shouldMark) {
        const esc = escapeHtml(a.match);
        html = html.replaceAll ? html.replaceAll(esc, `<mark>${esc}</mark>`) : html.replace(esc, `<mark>${esc}</mark>`);
      }
    });
  }

  // 2) Make constructor calls clickable
  {
    const attnAssignRaw = "CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=dropout)";
    const esc = escapeHtml(attnAssignRaw);
    const wrapped = selectedId === "block_attn"
      ? `<span data-action="select:attn_class" style="text-decoration: underline; cursor: pointer;"><mark>${esc}</mark></span>`
      : `<span data-action="select:attn_class" style="text-decoration: underline; cursor: pointer;">${esc}</span>`;
    html = html.replaceAll ? html.replaceAll(esc, wrapped) : html.replace(esc, wrapped);
  }
  {
    const mlpAssignRaw = 'MLP(n_embd=n_embd, dropout=dropout)';
    const esc = escapeHtml(mlpAssignRaw);
    const wrapped = `<span data-action="select:mlp_subdiagram" style="text-decoration: underline; cursor: pointer;">${selectedId === "block_mlp" ? `<mark>${esc}</mark>` : esc}</span>`;
    html = html.replaceAll ? html.replaceAll(esc, wrapped) : html.replace(esc, wrapped);
  }

  // 3) Highlight specific areas by selection
  const highlightKeys = (keys) => {
    keys.forEach((key) => {
      const a = find(key);
      if (!a) return;
      const esc = escapeHtml(a.match);
      html = html.replace(esc, `<mark>${esc}</mark>`);
    });
  };
  if (selectedId === "token_embed") highlightKeys(["tok_emb", "tok_apply"]);
  if (selectedId === "pos_embed") highlightKeys(["pos_emb", "pos_apply"]);
  if (selectedId === "emb_dropout") highlightKeys(["drop_def", "drop_apply"]);
  if (selectedId === "final_ln") highlightKeys(["lnf_def", "lnf_apply"]);
  if (selectedId === "lm_head") highlightKeys(["lm_head_def", "lm_head_apply"]);
  if (selectedId === "loss_fn") {
    const a = find("loss_line");
    if (a) mark(a.match);
  }
  // Keep constructor clickable for blocks and highlight when selected
  {
    const ctor = find("blocks_ctor");
    if (ctor) {
      const escCtor = escapeHtml(ctor.match);
      html = html.replace(escCtor, `<span data-action="select:stack" style="text-decoration: underline; cursor: pointer;">${escCtor}</span>`);
    }
  }
  if (selectedId === "gpt2_blocks") highlightKeys(["blocks_ctor", "blocks_assign", "blocks_for", "blocks_close", "blocks_use_for", "blocks_use_apply"]);

  // 4) Contextual replacements for info links
  if (selectedId === "emb_dropout") {
    html = html.replace(/nn\.Dropout\(dropout\)/g, '<span data-action="open:emb_dropout_notes" style="text-decoration: underline; cursor: pointer;">nn.Dropout(dropout)</span>');
  }
  if (selectedId === "mlp_dropout") {
    html = html.replace(/nn\.Dropout\(dropout\)/g, '<span data-action="open:mlp_dropout_notes" style="text-decoration: underline; cursor: pointer;">nn.Dropout(dropout)</span>');
  }

  // 5) Add links on phrases
  html = html.replace(/Dimension\s+changes\s+during\s+forward/g, '<span data-action="open:dimension_change_notes" style="text-decoration: underline; cursor: pointer;">Dimension changes during forward</span>');
  html = html.replace(/weight\s+tying/g, '<span data-action="open:weight_tying_notes" style="text-decoration: underline; cursor: pointer;">weight tying</span>');
  html = html.replace(/LM head shares weights with token embeddings/g, '<span data-action="open:weight_tying_notes" style="text-decoration: underline; cursor: pointer;">LM head shares weights with token embeddings</span>');
  
  // 6) Token clickable spans
  html = html.replace(/nn\.GELU\(approximate=\"tanh\"\)/g, '<span data-action="open:gelu" style="text-decoration: underline; cursor: pointer;">nn.GELU(approximate="tanh")</span>');
  html = html.replace(/nn\.GELU\(\)/g, '<span data-action="open:gelu" style="text-decoration: underline; cursor: pointer;">nn.GELU()</span>');

  // 7) Linear links: wrap generic, but exclude the specific lm_head form, then specific
  //html = html.replace(/nn\.Linear\((?!n_embd,\s*vocab_size,\s*bias=False)[^)]*\)/g, (m) => `<span data-action="open:linear" style="text-decoration: underline; cursor: pointer;">${m}</span>`);
  html = html.replace(/self\.fc_in\s*=\s*nn\.Linear\(n_embd,\s*4\s*\*\s*n_embd\)/g, '<span data-action="open:mlp_fc_in_notes" style="text-decoration: underline; cursor: pointer;">self.fc_in = nn.Linear(n_embd, 4 * n_embd)</span>');
  html = html.replace(/self\.fc_out\s*=\s*nn\.Linear\(4\s*\*\s*n_embd,\s*n_embd\)/g, '<span data-action="open:mlp_fc_out_notes" style="text-decoration: underline; cursor: pointer;">self.fc_out = nn.Linear(4 * n_embd, n_embd)</span>');
  html = html.replace(/nn\.Linear\(n_embd,\s*vocab_size,\s*bias=False\)/g, '<span data-action="open:lm_head_notes" style="text-decoration: underline; cursor: pointer;">nn.Linear(n_embd, vocab_size, bias=False)</span>');

  // 8) LayerNorm, Embeddings, loss, attention ops
  html = html.replace(/nn\.LayerNorm\([^)]*\)/g, (m) => `<span data-action="open:layernorm" style="text-decoration: underline; cursor: pointer;">${m}</span>`);
  html = html.replace(/nn\.Embedding\(vocab_size,\s*n_embd\)/g, '<span data-action="open:embedding_tok" style="text-decoration: underline; cursor: pointer;">nn.Embedding(vocab_size, n_embd)</span>');
  html = html.replace(/nn\.Embedding\(max_toks,\s*n_embd\)/g, '<span data-action="open:embedding_pos" style="text-decoration: underline; cursor: pointer;">nn.Embedding(max_toks, n_embd)</span>');
  html = html.replace(/F\.cross_entropy(\([^)]*\))/g, '<span data-action="open:cross_entropy_notes" style="text-decoration: underline; cursor: pointer;">F.cross_entropy($1)</span>');

  html = html.replace(/k\s*=\s*self\.key\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g, '<span data-action="open:key_view_notes" style="text-decoration: underline; cursor: pointer;">k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>');
  html = html.replace(/q\s*=\s*self\.query\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g, '<span data-action="open:query_view_notes" style="text-decoration: underline; cursor: pointer;">q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>');
  html = html.replace(/v\s*=\s*self\.value\(x\)\.view\(B, T, self\.n_head, hs\)\.transpose\(1, 2\)/g, '<span data-action="open:value_view_notes" style="text-decoration: underline; cursor: pointer;">v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)</span>');
  html = html.replace(/\s*\(\s*q\s*@\s*k\.transpose\(\s*-2\s*,\s*-1\s*\)\s*\)\s*\*\s*\(\s*1\.0\s*\/\s*hs\*\*0\.5\s*\)/g, '<span data-action="open:q_k_product_notes" style="text-decoration: underline; cursor: pointer;"> (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)</span>');
  html = html.replace(/\s*att\.masked_fill\(\s*self\.mask\s*\[\s*:\s*,\s*:\s*,\s*:\s*T\s*,\s*:\s*T\s*\]\s*==\s*0\s*,\s*float\(\s*'-inf'\s*\)\s*\)/g, '<span data-action="open:causal_mask_notes" style="text-decoration: underline; cursor: pointer;"> att.masked_fill(self.mask[:, :, :T, :T] == 0, float(\'-inf\'))</span>');
  html = html.replace(/self\.attn_drop\s*=\s*nn\.Dropout\(dropout\)/g, '<span data-action="open:attn_dropout_notes" style="text-decoration: underline; cursor: pointer;">self.attn_drop = nn.Dropout(dropout)</span>');
  html = html.replace(/self\.resid_drop\s*=\s*nn\.Dropout\(dropout\)/g, '<span data-action="open:resid_dropout_notes" style="text-decoration: underline; cursor: pointer;">self.resid_drop = nn.Dropout(dropout)</span>');
  html = html.replace(/self\.register_buffer/g, '<span data-action="open:register_buffer_notes" style="text-decoration: underline; cursor: pointer;">self.register_buffer</span>');
  html = html.replace(/torch\.tril/g, '<span data-action="open:torch_tril_notes" style="text-decoration: underline; cursor: pointer;">torch.tril</span>');
  html = html.replace(/torch\.ones/g, '<span data-action="open:torch_ones_notes" style="text-decoration: underline; cursor: pointer;">torch.ones</span>');
  html = html.replace(/unsqueeze/g, '<span data-action="open:unsqueeze_notes" style="text-decoration: underline; cursor: pointer;">unsqueeze</span>');
  html = html.replace(/F\.softmax\(att, dim=-1\)/g, '<span data-action="open:softmax_notes" style="text-decoration: underline; cursor: pointer;">F.softmax(att, dim=-1)</span>');
  html = html.replace(/y = att @ v/g, '<span data-action="open:att_v_product_notes" style="text-decoration: underline; cursor: pointer;">y = att @ v</span>');
  html = html.replace(/y\.transpose\(1, 2\)\.contiguous\(\)\.view\(B, T, C\)/g, '<span data-action="open:reorder_merge_heads_notes" style="text-decoration: underline; cursor: pointer;"> y.transpose(1, 2).contiguous().view(B, T, C)</span>');
  html = html.replace(/nn\.Dropout\(dropout\)\s*#\s*embedding\s*dropout/g, '<span data-action="open:emb_dropout_notes" style="text-decoration: underline; cursor: pointer;">nn.Dropout(dropout) # embedding dropout</span>');

  
  return html;
}
