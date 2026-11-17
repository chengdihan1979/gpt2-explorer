// All notes, keyed by id. Each topic file holds { [id]: { en: { title, md }, zh: { title, md } } }.
import model from "./model.js";
import input from "./input.js";
import embedding from "./embedding.js";
import position from "./position.js";
import attention from "./attention.js";
import mlp from "./mlp.js";
import residual from "./residual.js";
import norm from "./norm.js";
import output from "./output.js";
import training from "./training.js";
import pytorch from "./pytorch.js";
import scaling from "./scaling.js";

const TOPICS = { model, input, embedding, position, attention, mlp, residual, norm, output, training, pytorch, scaling };

export const NOTES = {};
for (const [topic, notes] of Object.entries(TOPICS)) {
  for (const [id, note] of Object.entries(notes)) {
    if (NOTES[id]) throw new Error(`Duplicate note id "${id}" in notes/${topic}.js`);
    NOTES[id] = note;
  }
}

// The note to show in the chosen language; a note with no Chinese version falls back to English.
export function getNote(id, lang) {
  const n = NOTES[id];
  const zh = lang === "zh" ? n?.zh : null;
  return { title: zh?.title ?? n?.en?.title, md: zh?.md ?? n?.en?.md, missingZh: lang === "zh" && !zh };
}
