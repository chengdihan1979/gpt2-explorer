import React from "react";
import { DETAIL_LINE, LINE } from "./palette";

// variant="detail" draws a dashed connector meaning "this block expands into that sub-diagram".
export default function Arrow({ x1, y1, x2, y2, variant }) {
  const detail = variant === "detail";
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={detail ? DETAIL_LINE : LINE}
      strokeWidth={1.25}
      strokeDasharray={detail ? "4 4" : undefined}
      markerEnd={detail ? "url(#arrowhead-detail)" : "url(#arrowhead)"}
    />
  );
}
