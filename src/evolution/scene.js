// A "scene" is a whole architecture drawing as plain data: { [id]: element }.
// The evolution player tweens between two scenes: elements with the same id move/resize/relabel,
// elements only in the old scene fade out, elements only in the new one fade in.
//
// Element types (all coordinates in the SVG's user space):
//   frame  { x, y, w, h, dashed }               rounded group box
//   title  { x, y, label, anchor }              pill-shaped group heading
//   node   { x, y, w, h, label, cat? }          a block (color from palette category of its id, or `cat`)
//   link   { from, to, x? }                     vertical arrow from the bottom of node `from` to the top of node `to`,
//                                               at x (default: center of `from`); follows the nodes while they move
//   line   { x1, y1, x2, y2, variant? }         straight arrow ("detail" = dashed sub-diagram connector)
//   poly   { points: [[x, y], ...], variant? }  polyline arrow ("tie" = dashed amber, no arrowhead)
//   plus   { cx, cy }                           residual-add circle
//   text   { x, y, label, size?, color?, rotate? }

// Draw order: boxes at the back, arrows, then blocks and labels on top.
export const TYPE_ORDER = ["frame", "line", "poly", "link", "plus", "node", "title", "text"];

// Build a scene object from a list of elements.
export function scene(elements) {
  const s = {};
  for (const e of elements) s[e.id] = e;
  return s;
}

// Replace every element whose id starts with `prefix` by `elements` (these may also overwrite other ids).
export function replaceGroup(s, prefix, elements) {
  for (const id of Object.keys(s)) if (id.startsWith(prefix)) delete s[id];
  for (const e of elements) s[e.id] = e;
}

export function relabel(s, labels) {
  for (const [id, label] of Object.entries(labels)) s[id] = { ...s[id], label };
}

export function remove(s, ids) {
  for (const id of ids) delete s[id];
}

const lerp = (a, b, t) => a + (b - a) * t;

const NUM_KEYS = ["x", "y", "w", "h", "x1", "y1", "x2", "y2", "cx", "cy", "size", "rotate"];

// Interpolate one element that exists in both scenes.
function mix(a, b, t) {
  const out = { ...b };
  for (const k of NUM_KEYS) if (typeof a[k] === "number" && typeof b[k] === "number") out[k] = lerp(a[k], b[k], t);
  if (a.points && b.points && a.points.length === b.points.length) {
    out.points = a.points.map((p, i) => [lerp(p[0], b.points[i][0], t), lerp(p[1], b.points[i][1], t)]);
  }
  if (a.label !== b.label) {
    out.label = b.label;
    out.prevLabel = a.label; // drawn fading out while `label` fades in
  }
  out.labelT = t;
  return out;
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

// The frame at time t (0..1, linear) between scenes A and B. Each element gets an `opacity` and a `status`.
// Two phases, so removals are seen before things move:
//   0.00-0.30  removed elements are shown outlined in red, then fade out by 0.50;
//   0.30-1.00  kept elements move/resize/relabel (eased); added elements fade in from 0.60.
export function frameAt(A, B, t) {
  const move = easeInOut(clamp01((t - 0.3) / 0.7));
  const out = {};
  for (const id of Object.keys(A)) {
    if (B[id]) out[id] = { ...mix(A[id], B[id], move), opacity: 1, status: "kept" };
    else out[id] = { ...A[id], opacity: 1 - clamp01((t - 0.25) / 0.25), status: "removed" };
  }
  for (const id of Object.keys(B)) {
    if (!A[id]) out[id] = { ...B[id], opacity: clamp01((t - 0.6) / 0.4), status: "added" };
  }
  return out;
}

// Geometry of a link arrow, computed from the (possibly moving) nodes it connects.
export function linkGeometry(link, frame) {
  const a = frame[link.from];
  const b = frame[link.to];
  if (!a || !b) return null;
  const x = link.x ?? a.x + a.w / 2;
  return { x1: x, y1: a.y + a.h, x2: x, y2: b.y };
}

export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
