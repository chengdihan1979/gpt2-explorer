import React from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import { useInfo } from "../context/InfoContext";
import { INFO } from "../content/info";

export default function InfoPanel() {
  const { active, previous, close, open } = useInfo();
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
    </div>
  );
}
