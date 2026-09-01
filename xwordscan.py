#!/usr/bin/env python3
"""
xwordscan – convert an image of an empty crossword grid to .puz format.

Usage:
    python xwordscan.py <image_path> [output.puz]

The script:
  1. Loads the image and detects the crossword grid using OpenCV.
  2. Uses PaddleOCR to read any pre-filled letters or numbers in the cells.
  3. Infers the grid dimensions and black-square positions.
  4. Writes a valid .puz file using the `puz` library.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import puz


# ---------------------------------------------------------------------------
# Grid detection helpers
# ---------------------------------------------------------------------------

def _load_gray(image_path: str) -> np.ndarray:
    """Load an image and return a grayscale version."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold to isolate grid lines."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )
    return binary


def _find_grid_contour(binary: np.ndarray) -> np.ndarray:
    """Return the bounding rectangle of the largest rectangular contour."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found in image.")
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def _crop_grid(gray: np.ndarray, binary: np.ndarray):
    """Return the gray and binary images cropped to the grid bounding box."""
    x, y, w, h = _find_grid_contour(binary)
    return gray[y : y + h, x : x + w], binary[y : y + h, x : x + w]


def _detect_grid_size(binary_grid: np.ndarray) -> tuple[int, int]:
    """
    Estimate the number of rows and columns in the crossword grid by
    analysing horizontal and vertical line projections.
    """
    h, w = binary_grid.shape

    # Horizontal projection – count rows of high-density pixels (grid lines).
    h_proj = np.sum(binary_grid, axis=1).astype(float)
    h_proj /= (w * 255)

    # Vertical projection
    v_proj = np.sum(binary_grid, axis=0).astype(float)
    v_proj /= (h * 255)

    threshold = 0.3  # fraction of pixels that must be set to count as a line

    h_lines = _count_line_groups(h_proj, threshold)
    v_lines = _count_line_groups(v_proj, threshold)

    # Number of cells = number of grid lines – 1  (fences-and-posts)
    rows = max(1, h_lines - 1)
    cols = max(1, v_lines - 1)
    return rows, cols


def _count_line_groups(proj: np.ndarray, threshold: float) -> int:
    """Count distinct groups of high-value pixels in a 1-D projection."""
    above = proj > threshold
    groups = 0
    in_group = False
    for val in above:
        if val and not in_group:
            groups += 1
            in_group = True
        elif not val:
            in_group = False
    return groups


def _extract_cells(gray_grid: np.ndarray, rows: int, cols: int) -> list[list[np.ndarray]]:
    """Slice the grid into individual cell images."""
    h, w = gray_grid.shape
    cell_h = h // rows
    cell_w = w // cols
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            row_cells.append(gray_grid[y0:y1, x0:x1])
        cells.append(row_cells)
    return cells


def _is_black_cell(cell: np.ndarray, black_threshold: float = 0.5) -> bool:
    """Return True if the cell is mostly dark (a black/filled square)."""
    mean = np.mean(cell) / 255.0
    return mean < (1.0 - black_threshold)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _run_ocr(cells: list[list[np.ndarray]]) -> list[list[str]]:
    """
    Use PaddleOCR to extract any pre-filled letters from each cell.
    Returns a 2-D list of strings (single character or empty string).
    """
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "paddleocr is required. Install it with: pip install paddleocr"
        ) from exc

    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

    result_grid: list[list[str]] = []
    for row in cells:
        result_row: list[str] = []
        for cell in row:
            # PaddleOCR expects a file path or a numpy BGR array.
            cell_bgr = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            ocr_result = ocr.ocr(cell_bgr, cls=False)
            text = ""
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        candidate = line[1][0].strip().upper()
                        if candidate.isalpha() and len(candidate) == 1:
                            text = candidate
                            break
            result_row.append(text)
        result_grid.append(result_row)
    return result_grid


# ---------------------------------------------------------------------------
# .puz assembly
# ---------------------------------------------------------------------------

def _build_puz(
    rows: int,
    cols: int,
    black_cells: list[list[bool]],
    fill: list[list[str]],
    title: str = "",
    author: str = "",
) -> puz.Puzzle:
    """Assemble a puz.Puzzle from the detected grid data."""
    puzzle = puz.Puzzle()
    puzzle.title = title
    puzzle.author = author
    puzzle.width = cols
    puzzle.height = rows

    # Build solution and fill strings.
    # In .puz format:
    #   '.' = black square
    #   '-' = empty white square (in fill / answer unknown)
    #   letter = pre-filled letter
    solution_chars = []
    fill_chars = []
    for r in range(rows):
        for c in range(cols):
            if black_cells[r][c]:
                solution_chars.append(".")
                fill_chars.append(".")
            else:
                letter = fill[r][c] if fill[r][c] else "-"
                solution_chars.append(letter)
                fill_chars.append("-")

    puzzle.solution = "".join(solution_chars)
    puzzle.fill = "".join(fill_chars)

    # Clue numbering – standard crossword rules:
    # A cell gets a number if it starts an across or down word.
    clue_num = 1
    numbering = [[0] * cols for _ in range(rows)]
    across_clues: list[str] = []
    down_clues: list[str] = []

    for r in range(rows):
        for c in range(cols):
            if black_cells[r][c]:
                continue
            starts_across = (c == 0 or black_cells[r][c - 1]) and (
                c + 1 < cols and not black_cells[r][c + 1]
            )
            starts_down = (r == 0 or black_cells[r - 1][c]) and (
                r + 1 < rows and not black_cells[r + 1][c]
            )
            if starts_across or starts_down:
                numbering[r][c] = clue_num
                if starts_across:
                    across_clues.append(f"{clue_num} Across")
                if starts_down:
                    down_clues.append(f"{clue_num} Down")
                clue_num += 1

    # puz library expects clues in reading order: across clues for each
    # numbered square (in order), then down clues.
    puzzle.clues = across_clues + down_clues

    return puzzle


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert(
    image_path: str,
    output_path: str | None = None,
    title: str = "",
    author: str = "",
    use_ocr: bool = True,
) -> puz.Puzzle:
    """
    Convert a crossword grid image to a .puz file.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str, optional
        Path for the output .puz file.  If None the file is written next to
        the image with a .puz extension.
    title : str
        Puzzle title embedded in the .puz file.
    author : str
        Puzzle author embedded in the .puz file.
    use_ocr : bool
        Whether to run PaddleOCR to detect pre-filled letters.

    Returns
    -------
    puz.Puzzle
        The assembled puzzle object (also written to disk).
    """
    gray = _load_gray(image_path)
    binary = _binarise(gray)
    gray_grid, binary_grid = _crop_grid(gray, binary)
    rows, cols = _detect_grid_size(binary_grid)

    cells = _extract_cells(gray_grid, rows, cols)

    # Determine black squares from pixel intensity.
    black_cells = [
        [_is_black_cell(cells[r][c]) for c in range(cols)]
        for r in range(rows)
    ]

    # Optionally run OCR for pre-filled letters.
    if use_ocr:
        fill = _run_ocr(cells)
    else:
        fill = [["" for _ in range(cols)] for _ in range(rows)]

    # Override fill with '.' for black cells.
    for r in range(rows):
        for c in range(cols):
            if black_cells[r][c]:
                fill[r][c] = ""

    puzzle = _build_puz(rows, cols, black_cells, fill, title=title, author=author)

    # Determine output path.
    if output_path is None:
        output_path = str(Path(image_path).with_suffix(".puz"))

    puzzle.save(output_path)
    print(f"Saved {rows}×{cols} puzzle to {output_path}")
    return puzzle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a crossword grid image to .puz format using PaddleOCR."
    )
    parser.add_argument("image", help="Path to the crossword grid image.")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output .puz file path (defaults to <image>.puz).",
    )
    parser.add_argument("--title", default="", help="Puzzle title.")
    parser.add_argument("--author", default="", help="Puzzle author.")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR; leave all white cells empty.",
    )
    args = parser.parse_args(argv)

    try:
        convert(
            image_path=args.image,
            output_path=args.output,
            title=args.title,
            author=args.author,
            use_ocr=not args.no_ocr,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
