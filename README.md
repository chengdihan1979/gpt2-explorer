### GPT‑2 Decoder Explorer

Large language model (LLM) training involves many moving parts: shapes, masks, normalizations, activations, weight tying, and more. The details are easy to lose across papers, blogs, and code. This interactive visual explorer organizes these concepts in one place—so developers at any level can click through the decoder, see the exact code lines, and open concise math-backed notes to understand how it all fits together.

An interactive visual explainer of the GPT‑2 decoder, built with React + Vite + Tailwind CSS. Click nodes in the left diagram to:
- highlight relevant lines in minimal PyTorch reference code on the right
- open inline explainers with math (KaTeX) for concepts like LayerNorm, GELU, causal mask, weight tying, broadcasting, etc.

---

### Features
- Interactive SVG diagram of the GPT‑2 decoder (embeddings → dropout → Transformer blocks × N → final LayerNorm → LM head → loss).
- Clickable sub-parts that deep-link into code and open compact notes with math.
- Clean PyTorch reference snippets for model, attention, MLP, and a simple training loop.
- Tailwind typography and KaTeX rendering for readable math.
- Added “Input Data (Text)” and “Tokenizer” components before embeddings, with links to their code and notes.

---

### Requirements
- Node.js 18+ and npm

---

### Quick start
Install deps and run dev:
```bash
npm install
npm run dev
```
Vite prints a local URL (e.g. http://localhost:5173).

Build and preview:
```bash
npm run build
npm run preview
```

One‑shot (install + run):
```bash
bash scripts/setup-and-run.sh
```

Docker (no local Node required):
```bash
docker build -t gpt2-explorer .
docker run --rm -p 8080:80 gpt2-explorer
```
Open http://localhost:8080

---

### Project structure (refactored)
- `src/main.jsx`: React root
- `src/App.jsx`: renders `RefExplorer`
- `src/RefExplorer.jsx`: thin wrapper → `InfoProvider` + `GPT2DecoderExplorerInner`
- `src/GPT2DecoderExplorerInner.jsx`: main UI container (diagram + code + InfoPanel)
- `src/context/InfoContext.jsx`: global info panel state (`useInfo`, `InfoProvider`)
- `src/components/InfoPanel.jsx`: Markdown + math renderer (remark-math, rehype-katex)
- `src/components/CodeWithActions.jsx`: clickable code block (event delegation via `data-action`)
- `src/components/diagram/`:
  - `Node.jsx`, `Arrow.jsx`, `GPT2OuterFrame.jsx`, `TransformerBlockGroup.jsx`, `MLPSubDiagram.jsx`
- `src/content/snippets.js`: all code snippets and the `content` map (titles, summaries, anchors)
- `src/content/info.js`: info note registry (titles + markdown)
- `src/utils/highlight.js`: `buildCodeHtml` rules (highlight/links) and `escapeHtml`
- Styling:
  - `src/index.css` (Tailwind entry), `src/katex-overrides.css`
  - `tailwind.config.js`, `postcss.config.js`
- Tooling:
  - `vite.config.js`, `eslint.config.js`
  - `scripts/setup-and-run.sh`
  - `Dockerfile`, `nginx.conf`, `.dockerignore`

Notes:
- The legacy `src/GPT2DecoderExplorer.jsx` is kept for reference and to source original info notes; the app renders the refactored path via `RefExplorer`.

---

### How it works
- Diagram (left): SVG components emit `onClick(id)` to set `selectedId`.
- Content selection (right): `content` (from `src/content/snippets.js`) provides code + anchors; `buildCodeHtml` injects clickable spans and highlights.
- Info notes: `InfoPanel` renders markdown+math from `INFO` (from `src/content/info.js`). By default the InfoPanel is shown under the left diagram; the code panel stays on the right.
- Links & highlights are powered by predictable strings/regex in `src/utils/highlight.js`.

---

### Extend the explorer
- Add a new info topic:
  1) Add an entry to `src/content/info.js` (`id`, `title`, `md`).
  2) Reference it in code via a clickable span rule (update `buildCodeHtml`) or link to `#info:<id>` inside note markdown.

- Add a new diagram node:
  1) Place a `Node` within the SVG in `src/GPT2DecoderExplorerInner.jsx` (or a sub-diagram component).
  2) Ensure there’s a matching `content[<id>]` entry in `src/content/snippets.js`.
  3) Add anchors in that content block to highlight relevant lines.

- Add/modify code highlighting rules:
  - Edit `src/utils/highlight.js` → `buildCodeHtml(...)` (regex replacements and anchor-based `mark(...)`).

- Add another code snippet group:
  1) Create snippet string(s) in `src/content/snippets.js`.
  2) Add a `content[<id>]` block with `title`, `summary`, `code`, `anchors`.
  3) Wire a node or clickable from another snippet using `data-action="select:<id>"` in `buildCodeHtml`.

---

### Scripts
- `npm run dev`: start dev server
- `npm run build`: build to `dist/`
- `npm run preview`: preview production build
- `npm run lint`: ESLint

---

### Dependencies
- Runtime: `react`, `react-dom`, `react-markdown`, `remark-math`, `rehype-katex`, `katex`
  - Also: `rehype-slug` (adds stable heading ids in notes)
- Dev/build: `vite`, `@vitejs/plugin-react`, `tailwindcss`, `@tailwindcss/postcss`, `@tailwindcss/typography`, `postcss`, `autoprefixer`, `eslint`

---

### License
This project is licensed under the [MIT License](./LICENSE). You are free to use, modify, and distribute the code with proper attribution.


