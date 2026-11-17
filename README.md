### LLM Explorer

Large language model (LLM) training involves many moving parts: shapes, masks, normalizations, activations, weight tying, and more. The details are easy to lose across papers, blogs, and code. This interactive visual explorer organizes these concepts in one place—so developers at any level can click through the decoder, see the exact code lines, and open concise math-backed notes to understand how it all fits together.

An interactive visual explainer of LLM architectures, starting with the GPT‑2 decoder (more open models to follow), built with React + Vite + Tailwind CSS. Notes are available in English and Chinese. Click nodes in the left diagram to:
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
docker build -t llm-explorer .
docker run --rm -p 8080:80 llm-explorer
```
Open http://localhost:8080

---

### Project structure
- `src/main.jsx`: React root → `App.jsx` → `RefExplorer.jsx`
- `src/RefExplorer.jsx`: picks the model page from the URL hash (`#/gpt2`) via `useHashRoute.js`
- `src/ModelExplorer.jsx`: the shared page layout (header + model nav, diagram, code panel, notes drawer)
- `src/models/index.js`: list of models shown in the nav
- `src/models/gpt2/`: everything GPT‑2-specific
  - `index.js`: model config (diagram, code snippets, file names, node → code mapping)
  - `snippets.js`: PyTorch code and the `content` map (titles, summaries, anchors per diagram node)
  - `diagram/`: the GPT‑2 SVG diagram (`GPT2Diagram.jsx` and its sub-diagrams)
- `src/components/`: shared UI
  - `diagram/`: `Node`, `Arrow`, `FrameTitle`, `palette`, `SelectionContext`
  - `CodeWithActions.jsx`: clickable code block (event delegation via `data-action`)
  - `InfoPanel.jsx`: notes drawer and popups (Markdown + KaTeX)
- `src/content/notes/`: notes library, one file per topic (`attention.js`, `norm.js`, `training.js`, …); each note keeps its English and Chinese versions side by side; `index.js` merges them and exports `NOTES` and `getNote`
- `src/context/InfoContext.jsx`: notes drawer state and EN/中文 language
- `src/utils/highlight.js`: `buildCodeHtml` rules (highlights and code → note links)
- Styling: `src/index.css`, `src/katex-overrides.css`, `tailwind.config.js`, `postcss.config.js`
- Tooling: `vite.config.js`, `eslint.config.js`, `scripts/setup-and-run.sh`, `Dockerfile`, `nginx.conf`

---

### How it works
- Diagram (left): SVG nodes call `onSelect(id)`; the model config maps the id to an entry in its `content`.
- Code (right): `buildCodeHtml` highlights the anchored lines and turns known code into links (`data-action="open:<note>"` or `select:<node>`).
- Notes: `InfoPanel` renders a note from `NOTES` in the chosen language (falling back to English). Inside notes, `#info:<id>` opens another note in the drawer and `#hover:<id>` opens a popup.

---

### Extend the explorer
- Add a note: add `<id>: { en: { title, md }, zh: { title, md } }` to the matching topic file in `src/content/notes/`, then link it from code (a rule in `src/utils/highlight.js`) or from another note (`#info:<id>`).
- Add a diagram node: place a `Node` in the model's diagram and add a matching `content[<id>]` entry (with anchors) in its `snippets.js`, or map it to an existing entry in the model config's node → code mapping.
- Add a model: create `src/models/<id>/` with the same shape as `src/models/gpt2/` and add it to `MODELS` in `src/models/index.js`; it appears in the nav at `#/<id>`.

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


