from dataclasses import dataclass

import cv2
import numpy as np

from .model import BBox, Rule


@dataclass
class TableGrid:
    bbox: BBox
    cells: list[BBox]


def detect_rules(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Rule]]:
    """Extracts long horizontal/vertical strokes (table borders, separators, underlines)."""
    h, w = binary.shape
    hk, vk = max(25, w // 35), max(25, h // 45)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    max_thick = max(8, int(min(h, w) * 0.006))
    rules: list[Rule] = []
    for mask, is_h in ((horiz, True), (vert, False)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            thick = area / max(cw if is_h else ch, 1)
            pos = y + ch / 2 if is_h else x + cw / 2
            limit = h if is_h else w
            at_edge = pos < 0.015 * limit or pos > 0.985 * limit  # paper edge in photos
            if thick > max_thick or at_edge:
                mask[lab == i] = 0
                continue
            if is_h:
                yc = y + ch / 2
                rules.append(Rule(x, yc, x + cw, yc, max(1.0, thick)))
            else:
                xc = x + cw / 2
                rules.append(Rule(xc, y, xc, y + ch, max(1.0, thick)))
    return horiz, vert, rules


def detect_tables(horiz: np.ndarray, vert: np.ndarray) -> list[TableGrid]:
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 80 or h < 40:
            continue
        h_sub, v_sub = horiz[y:y + h, x:x + w], vert[y:y + h, x:x + w]
        if cv2.connectedComponents(h_sub)[0] - 1 < 2 or cv2.connectedComponents(v_sub)[0] - 1 < 2:
            continue
        inside = cv2.bitwise_not(grid[y:y + h, x:x + w])
        n, _, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=4)
        cells = []
        for i in range(1, n):
            cx, cy, cw, ch, area = stats[i]
            if cx == 0 or cy == 0 or cx + cw >= w or cy + ch >= h:
                continue  # region outside the table frame
            if cw < 12 or ch < 12 or area < 0.6 * cw * ch:
                continue
            cells.append((x + cx, y + cy, x + cx + cw, y + cy + ch))
        if len(cells) >= 2:
            cells.sort(key=lambda b: (b[1] // 10, b[0]))
            tables.append(TableGrid((x, y, x + w, y + h), cells))
    return tables
