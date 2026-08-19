"""A tiny fixed-grid ASCII canvas.

The v0.7 systems maps have to stay column-aligned at 100%, 125% and 150% zoom,
which means every glyph must land on an exact grid cell. Hand-typed box art
drifts the moment a label changes length, so the maps are *composed* here
instead: boxes and connectors are placed by coordinate and the canvas
guarantees the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

# Box-drawing sets. "solid" is a runtime/process boundary, "double" is a trust
# boundary, "dashed" is an advisory or deferred edge.
STYLES: dict[str, tuple[str, str, str, str, str, str]] = {
    #        tl   tr   bl   br   h    v
    "solid": ("┌", "┐", "└", "┘", "─", "│"),
    "double": ("╔", "╗", "╚", "╝", "═", "║"),
    "heavy": ("┏", "┓", "┗", "┛", "━", "┃"),
    "round": ("╭", "╮", "╰", "╯", "─", "│"),
    "dashed": ("┌", "┐", "└", "┘", "╌", "╎"),
}

ARROWS = {"right": "▶", "left": "◀", "up": "▲", "down": "▼"}


class CanvasError(ValueError):
    """Raised when a drawing operation would fall outside the canvas."""


@dataclass(slots=True)
class Box:
    row: int
    col: int
    width: int
    height: int

    @property
    def top(self) -> int:
        return self.row

    @property
    def bottom(self) -> int:
        return self.row + self.height - 1

    @property
    def left(self) -> int:
        return self.col

    @property
    def right(self) -> int:
        return self.col + self.width - 1

    @property
    def mid_row(self) -> int:
        return self.row + self.height // 2

    @property
    def mid_col(self) -> int:
        return self.col + self.width // 2


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid = [[" "] * width for _ in range(height)]

    # ------------------------------------------------------------------ basics

    def put(self, row: int, col: int, char: str, *, overwrite: bool = True) -> None:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise CanvasError(f"({row},{col}) outside {self.height}x{self.width} canvas")
        if not overwrite and self.grid[row][col] != " ":
            return
        self.grid[row][col] = char

    def text(self, row: int, col: int, value: str, *, overwrite: bool = True) -> None:
        for offset, char in enumerate(value):
            self.put(row, col + offset, char, overwrite=overwrite)

    def center_text(self, row: int, box: Box, value: str) -> None:
        inner = box.width - 2
        value = value[:inner]
        start = box.left + 1 + max(0, (inner - len(value)) // 2)
        self.text(row, start, value)

    # -------------------------------------------------------------------- box

    def box(
        self,
        row: int,
        col: int,
        width: int,
        height: int,
        *,
        title: str = "",
        lines: tuple[str, ...] | list[str] = (),
        style: str = "solid",
        align: str = "left",
    ) -> Box:
        if width < 4 or height < 3:
            raise CanvasError("a box needs at least 4x3 cells")
        tl, tr, bl, br, h, v = STYLES[style]
        box = Box(row, col, width, height)
        self.text(row, col, tl + h * (width - 2) + tr)
        self.text(row + height - 1, col, bl + h * (width - 2) + br)
        for offset in range(1, height - 1):
            self.put(row + offset, col, v)
            self.put(row + offset, col + width - 1, v)
            self.text(row + offset, col + 1, " " * (width - 2))
        if title:
            self.center_text(row, box, f" {title[: width - 4]} ")
        for index, line in enumerate(lines):
            target = row + 1 + index
            if target >= row + height - 1:
                break
            if align == "center":
                self.center_text(target, box, line)
            else:
                self.text(target, col + 2, line[: width - 4])
        return box

    # ------------------------------------------------------------- connectors

    def hline(self, row: int, start_col: int, end_col: int, *, char: str = "─") -> None:
        lo, hi = sorted((start_col, end_col))
        for col in range(lo, hi + 1):
            self.put(row, col, char)

    def vline(self, col: int, start_row: int, end_row: int, *, char: str = "│") -> None:
        lo, hi = sorted((start_row, end_row))
        for row in range(lo, hi + 1):
            self.put(row, col, char)

    def arrow_h(
        self, row: int, start_col: int, end_col: int, *, label: str = "", char: str = "─"
    ) -> None:
        """Horizontal connector with the arrow head at ``end_col``."""

        if end_col >= start_col:
            self.hline(row, start_col, end_col - 1, char=char)
            self.put(row, end_col, ARROWS["right"])
            label_start = start_col + max(0, (end_col - start_col - len(label)) // 2)
        else:
            self.hline(row, end_col + 1, start_col, char=char)
            self.put(row, end_col, ARROWS["left"])
            label_start = end_col + max(0, (start_col - end_col - len(label)) // 2)
        if label:
            self.text(row - 1, label_start, label)

    def arrow_v(
        self, col: int, start_row: int, end_row: int, *, label: str = "", char: str = "│"
    ) -> None:
        """Vertical connector with the arrow head at ``end_row``."""

        if end_row >= start_row:
            self.vline(col, start_row, end_row - 1, char=char)
            self.put(end_row, col, ARROWS["down"])
        else:
            self.vline(col, end_row + 1, start_row, char=char)
            self.put(end_row, col, ARROWS["up"])
        if label:
            mid = (start_row + end_row) // 2
            self.text(mid, col + 2, label)

    def elbow(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        label: str = "",
        char_h: str = "─",
        char_v: str = "│",
        via_row: int | None = None,
    ) -> None:
        """Route ``start`` to ``end`` with one horizontal leg and one vertical leg."""

        (r0, c0), (r1, c1) = start, end
        row = via_row if via_row is not None else r1
        self.vline(c0, r0, row, char=char_v)
        self.hline(row, c0, c1, char=char_h)
        corner = "└" if row > r0 else "┌"
        self.put(row, c0, corner if c1 > c0 else ("┘" if row > r0 else "┐"))
        if row == r1:
            self.put(r1, c1, ARROWS["right"] if c1 > c0 else ARROWS["left"])
        else:
            self.vline(c1, row, r1, char=char_v)
            self.put(row, c1, "┐" if r1 > row else "┘")
            self.put(r1, c1, ARROWS["down"] if r1 > row else ARROWS["up"])
        if label:
            mid = c0 + (c1 - c0) // 2 - len(label) // 2
            self.text(row - 1, max(0, mid), label)

    def connect_h(
        self,
        left: Box,
        right: Box,
        *,
        row: int | None = None,
        label: str = "",
        reverse: bool = False,
        char: str = "─",
    ) -> None:
        """Connect two boxes horizontally, refusing labels that would overflow.

        A label is centred in the gap on the row above the arrow. If it does not
        fit the gap it raises instead of silently overwriting a box, which is
        how the first draft of these maps corrupted itself.
        """

        row = row if row is not None else left.mid_row
        start, end = left.right + 1, right.left - 1
        gap = end - start + 1
        if gap < 3:
            raise CanvasError(f"gap of {gap} columns is too narrow to connect")
        if label:
            if len(label) > gap:
                raise CanvasError(f"label {label!r} ({len(label)}) exceeds {gap}-column gap")
            self.text(row - 1, start + (gap - len(label)) // 2, label)
        if reverse:
            self.arrow_h(row, end, start, char=char)
        else:
            self.arrow_h(row, start, end, char=char)

    def connect_v(
        self,
        top: Box,
        bottom: Box,
        *,
        col: int | None = None,
        label: str = "",
        reverse: bool = False,
        char: str = "│",
    ) -> None:
        """Connect two boxes vertically; the label sits to the right of the line."""

        col = col if col is not None else top.mid_col
        start, end = top.bottom + 1, bottom.top - 1
        if end - start + 1 < 1:
            raise CanvasError("no vertical gap between the boxes")
        if reverse:
            self.arrow_v(col, end, start, char=char)
        else:
            self.arrow_v(col, start, end, char=char)
        if label:
            self.text((start + end) // 2, col + 2, label)

    def badge(self, row: int, col: int, value: str) -> None:
        self.text(row, col, f"[{value}]")

    # ----------------------------------------------------------------- output

    def render(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.grid).rstrip("\n")


__all__ = ["ARROWS", "STYLES", "Box", "Canvas", "CanvasError"]
