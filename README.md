### GPT‑2 Decoder Explorer

An interactive visual explainer of the GPT‑2 decoder architecture, built with React + Vite + Tailwind CSS. Click nodes in the left diagram to:
- highlight the relevant lines in minimal PyTorch reference code on the right
- open inline explainers with math (rendered via KaTeX) for concepts like LayerNorm, GELU, causal mask, weight tying, broadcasting, and more

The goal is to help learners and practitioners connect the boxes-and-arrows diagram of a decoder block to clean reference code and concise explanations.

---

### Features
- Interactive SVG diagram of the GPT‑2 decoder stack (token/positional embeddings → dropout → Transformer blocks × N → final LayerNorm → LM head → CrossEntropy).
- Clickable sub-parts (e.g., attention, MLP, LayerNorm) that highlight and deep‑link into reference code snippets.
- In‑context info panels with math and citations rendered via KaTeX, powered by React Markdown + remark‑math + rehype‑katex.
- Clean, minimal PyTorch reference code strings for: GPT‑2 module, causal self‑attention, MLP, and a tiny training loop.
- Tailwind typography and small UX touches for readability.

---

### Tech stack
- React 19 (Vite 7)
- Tailwind CSS 4 with `@tailwindcss/typography`
- React Markdown + remark‑math + rehype‑katex for math/MD rendering
- ESLint 9 with React Hooks and React Refresh configs

---

### Quick start

Prerequisites:
- Node.js 18+ and npm

Install dependencies:
```bash
npm install
```

Run the dev server:
```bash
npm run dev
```
Vite will print a local URL (typically http://localhost:5173). Open it in your browser.

Build for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

---

### Project structure (key files)
- `index.html`: Vite entry HTML
- `src/main.jsx`: React root render
- `src/App.jsx`: mounts `GPT2DecoderExplorer`
- `src/GPT2DecoderExplorer.jsx`: all interactive diagram/infopanel logic and reference code strings
- `src/index.css`: Tailwind entry; includes `@import "tailwindcss"`
- `src/katex-overrides.css`: left‑align display math for readability
- `tailwind.config.js`: Tailwind v4 config (top‑level)
- `postcss.config.js`: Tailwind + autoprefixer PostCSS setup
- `vite.config.js`: Vite with React plugin
- `eslint.config.js`: ESLint setup

Assets:
- `src/assets/gpt2-explorer.png`: README banner

---

### How it works (at a glance)
- The left panel is an SVG diagram. Clicking a node sets a `selectedId`.
- The right panel shows a minimal PyTorch code string and uses anchor rules to wrap/mark matching lines. Some spans are converted to clickable links that either:
  - switch the right panel to a different component/code view, or
  - open the inline info panel for topical explanations (e.g., causal mask, softmax, GELU, LayerNorm, weight tying).
- Math in info panels is authored in Markdown with LaTeX blocks and rendered via KaTeX.

---

### Development tips
- Tailwind scan paths are defined in `src/tailwind.config.js` for v3‑style scanning used by Tailwind 4 CLI integration. Keep JSX/HTML under `src/` for styles to be picked up.
- If you add new info topics, extend the `INFO` registry in `src/GPT2DecoderExplorer.jsx` and reference them via `#info:<id>` links in Markdown or by `data-action="open:<id>"` spans in code.
- If you add new diagram nodes, also extend the `content` map and optional `anchors` so highlighting works.

---

### Scripts
- `npm run dev`: start Vite dev server
- `npm run build`: build to `dist/`
- `npm run preview`: preview built app
- `npm run lint`: run ESLint

---

### License
Personal/educational use. Adapt as needed inside your org or projects.

### One‑shot: setup and run

If you want a single command to install and start the dev server:
```bash
bash scripts/setup-and-run.sh
```
Requires Node.js (npm) installed locally.

### Docker (no local Node required)

Build the image and start a static server:
```bash
docker build -t gpt2-explorer .
docker run --rm -p 8080:80 gpt2-explorer
```
Open http://localhost:8080


