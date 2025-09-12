import React, { useState, useRef, useLayoutEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import { useInfo } from "../context/InfoContext";
import { INFO } from "../content/info";

export default function InfoPanel() {
  const { active, previous, close, open } = useInfo();
  const [hoverInfoId, setHoverInfoId] = useState(null);
  const [hoverPos, setHoverPos] = useState({ top: 0, left: 0 });
  const popupRef = useRef(null);
  const anchorRectRef = useRef(null);

  const recomputePopup = useCallback(() => {
    if (!hoverInfoId || !popupRef.current || !anchorRectRef.current) return;
    const r = popupRef.current.getBoundingClientRect();
    const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
    const vh = typeof window !== "undefined" ? window.innerHeight : 800;
    const ar = anchorRectRef.current;
    const m = 8;
    // Candidates: above, below, right, left
    const candidates = [];
    // Above
    candidates.push({ top: ar.top - r.height - m, left: Math.min(Math.max(ar.left, m), vw - r.width - m), fits: () => ar.top - r.height - m >= m });
    // Below
    candidates.push({ top: ar.bottom + m, left: Math.min(Math.max(ar.left, m), vw - r.width - m), fits: () => ar.bottom + r.height + m <= vh - m });
    // Right
    candidates.push({ top: Math.min(Math.max(ar.top, m), vh - r.height - m), left: ar.right + m, fits: () => ar.right + r.width + m <= vw - m });
    // Left
    candidates.push({ top: Math.min(Math.max(ar.top, m), vh - r.height - m), left: ar.left - r.width - m, fits: () => ar.left - r.width - m >= m });
    let placed = null;
    for (const c of candidates) {
      if (c.fits()) { placed = { top: Math.max(m, Math.min(c.top, vh - r.height - m)), left: Math.max(m, Math.min(c.left, vw - r.width - m)) }; break; }
    }
    if (!placed) {
      placed = { top: Math.max(m, Math.min(ar.bottom + m, vh - r.height - m)), left: Math.max(m, Math.min(ar.left, vw - r.width - m)) };
    }
    setHoverPos(placed);
  }, [hoverInfoId]);

  useLayoutEffect(() => { recomputePopup(); }, [recomputePopup]);

  useLayoutEffect(() => {
    if (!popupRef.current) return;
    const ro = new ResizeObserver(() => recomputePopup());
    ro.observe(popupRef.current);
    const onResize = () => recomputePopup();
    window.addEventListener('resize', onResize);
    return () => { ro.disconnect(); window.removeEventListener('resize', onResize); };
  }, [recomputePopup, popupRef]);
  if (!active) return null;

  return (
    <div className="mt-4 pt-3 border-t text-xs text-slate-700 prose prose-sm max-w-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          {active === "weight_tying" && (
            <button className="text-xs underline" onClick={() => open("embedding_tok")}>← Back</button>
          )}
          {active === "broadcasting" && (
            <button className="text-xs underline" onClick={() => open("embedding_pos")}>← Back</button>
          )}
          {active === "positional_embeddings_example_notes" && (
            <button className="text-xs underline" onClick={() => open("embedding_pos")}>← Back</button>
          )}
          {(active === "torch_ones_notes" || active === "torch_tril_notes" || active === "unsqueeze_notes") && (
            <button className="text-xs underline" onClick={() => open("causal_mask_notes")}>← Back</button>
          )}
          {active === "contiguous_notes" && (
            <button className="text-xs underline" onClick={() => open("reorder_merge_heads_notes")}>← Back</button>
          )}
          {(active === "lossless_round_trip_notes" || active === "gpt2_merge_ranks_notes" || active === "regex_pre_tokenizer_notes" || active === "gpt2_byte_unicode_map_notes") && (
            <button className="text-xs underline" onClick={() => open("tokenizer_notes")}>← Back</button>
          )}
          {active === "lm_head_insights_notes" && (
            <button className="text-xs underline" onClick={() => open("lm_head_notes")}>← Back</button>
          )}
          {active === "stabilize_training_notes" && (
            <button className="text-xs underline" onClick={() => open("layernorm")}>← Back</button>
          )}
          {active === "improves_gradient_flow_notes" && (
            <button className="text-xs underline" onClick={() => open("layernorm")}>← Back</button>
          )}
          {active === "post_norm_gradient_example_notes" && (
            <button className="text-xs underline" onClick={() => open("layernorm")}>← Back</button>
          )}
          {active === "rmsnorm_gradient_example_notes" && (
            <button className="text-xs underline" onClick={() => open("layernorm")}>← Back</button>
          )}
          {active === "post_norm_scaling_example_notes" && (
            <button className="text-xs underline" onClick={() => open("attention_output_linear_notes")}>← Back</button>
          )}
          {active === "ln_jacobian_closed_form_proof_notes" && (
            <button className="text-xs underline" onClick={() => open("post_norm_gradient_example_notes")}>← Back</button>
          )}
          {active === "ln_gradient_attenuation_upper_bound_proof_notes" && (
            <button className="text-xs underline" onClick={() => open("post_norm_scaling_example_notes")}>← Back</button>
          )}
          {active === "orthogonal_projection_notes" && (
            <button className="text-xs underline" onClick={() => open("ln_gradient_attenuation_upper_bound_proof_notes")}>← Back</button>
          )}
          {active === "stabilize_training_no_layernorm_backprop_notes" && (
            <button className="text-xs underline" onClick={() => open("stabilize_training_notes")}>← Back</button>
          )}
          {(active === "prevents_vanishing_exploding_notes" || active === "gradient_notes") && (
            <button className="text-xs underline" onClick={() => open("attn_residual_notes")}>← Back</button>
          )}
          {active === "mlp_prevents_vanishing_exploding_notes" && (
            <button className="text-xs underline" onClick={() => open("mlp_residual_notes")}>← Back</button>
          )}
          {active === "mlp_dropout_notes" && previous === "resid_dropout_notes" && (
            <button className="text-xs underline" onClick={() => open("resid_dropout_notes")}>← Back</button>
          )}
          {active === "label_smoothing_notes" && (
            <button className="text-xs underline" onClick={() => open("stabilize_training_no_layernorm_backprop_notes")}>← Back</button>
          )}
          {active === "gradient_loss_to_logits_proof_notes" && (
            <button className="text-xs underline" onClick={() => open("stabilize_training_no_layernorm_backprop_notes")}>← Back</button>
          )}
          {active === "sgd_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          {active === "sgd_momentum_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          {active === "adagrad_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          {active === "rmsprop_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          {active === "adam_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          {active === "adamw_notes" && (
            <button className="text-xs underline" onClick={() => open("optim_adamw_notes")}>← Back</button>
          )}
          <h3 className="font-medium my-0">{INFO[active]?.title ?? "Info"}</h3>
        </div>
        <button className="text-xs underline" onClick={close}>Close</button>
      </div>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
        components={{
          a: ({ href, children, ...props }) => {
            if (href && href.startsWith("#info:")) {
              const id = href.slice("#info:".length);
              return (
                <a href={href} onClick={(e) => { e.preventDefault(); e.stopPropagation(); if (INFO[id]) open(id); }} className="underline cursor-pointer" {...props}>
                  {children}
                </a>
              );
            }
            if (href && href.startsWith("#hover:")) {
              const id = href.slice("#hover:".length);
              return (
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const rect = e.currentTarget.getBoundingClientRect();
                    anchorRectRef.current = rect;
                    // Seed a provisional position; final position will be adjusted by recomputePopup
                    const left = Math.min(rect.left, (typeof window !== "undefined" ? window.innerWidth : 1200) - 360);
                    const top = Math.max(8, rect.top - 16);
                    setHoverPos({ top, left });
                    anchorRectRef.current = rect;
                    setHoverInfoId((prev) => (prev === id ? null : id));
                    // In next frame, recompute with actual popup size
                    setTimeout(() => recomputePopup(), 0);
                  }}
                  className="underline cursor-pointer"
                  {...props}
                >
                  {children}
                </a>
              );
            }
            if (href === "#cdf") {
              return (
                <a href={href} onClick={(e) => { e.preventDefault(); e.stopPropagation(); open("cdf"); }} className="underline cursor-pointer" {...props}>
                  {children}
                </a>
              );
            }
            const isExternal = /^https?:\/\//.test(href || "");
            return (
              <a href={href} className="underline cursor-pointer" {...(isExternal ? { target: "_blank", rel: "noopener noreferrer" } : {})} {...props}>
                {children}
              </a>
            );
          },
        }}
      >
        {INFO[active]?.md || "_No content_"}
      </ReactMarkdown>
      {hoverInfoId && INFO[hoverInfoId] && (
        <div
          ref={popupRef}
          className="fixed z-50 bg-white border rounded shadow-lg p-3 text-xs overflow-auto"
          style={{ top: hoverPos.top, left: hoverPos.left, maxWidth: 720, maxHeight: '70vh', minWidth: 360 }}
          onClick={(e) => { e.stopPropagation(); }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[[rehypeKatex, { trust: (ctx) => ctx.command === "\\href", strict: "ignore", fleqn: true }], rehypeSlug]}
          >
            {INFO[hoverInfoId]?.md || "_No content_"}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}
