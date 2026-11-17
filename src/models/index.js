import gpt2 from "./gpt2";
import gpt3 from "./gpt3";

// Models in the order they appear in the top navigation. Each page is reached at "#/<id>".
export const MODELS = [gpt2, gpt3];

export const DEFAULT_MODEL_ID = "gpt2";

export function findModel(id) {
  return MODELS.find((m) => m.id === id) ?? null;
}
