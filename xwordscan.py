#!/usr/bin/env python3
"""
xwordscan – convert an image of an empty crossword grid to .puz format.

Usage:
    python xwordscan.py <image_path> [output.puz] [--title TITLE] [--author AUTHOR]
    python xwordscan.py <image_path> --no-ocr   # skip OCR, structure only

Pipeline:
  1. Preprocess: convert to grayscale, denoise, normalise contrast.
  2. Deskew: correct any rotation in the scanned image.
  3. Detect and crop the grid boundary.
  4. Estimate grid dimensions from line projections.
  5. Slice into individual cells; classify black vs white.
  6. For each white cell, crop the top-left quadrant and run PaddleOCR to
     detect the small clue number printed there.
  7. Assemble and save a valid .puz file.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import puz


def _save_debug_image(
    debug_dir: Path | None, filename: str, image: np.ndarray
) -> None:
    """Write a debug image when a debug directory is configured."""
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / filename
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Cannot write debug image: {output_path}")


# ---------------------------------------------------------------------------
# 1. Image preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    image_path: str, debug_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image and return ``(gray, binary)`` after denoising and
    contrast normalisation.

    ``gray``   – cleaned grayscale image, uint8.
    ``binary`` – binarised (black lines on white background) version, uint8.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # Convert to grayscale and remove colour/JPEG artefacts.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_debug_image(debug_dir, "01-grayscale.png", gray)

    # Bilateral filter: removes noise while preserving edges (grid lines).
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    _save_debug_image(debug_dir, "02-denoised.png", denoised)

    # Normalise contrast so faded prints and shadows are handled uniformly.
    gray = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    _save_debug_image(debug_dir, "03-normalized.png", gray)

    # Adaptive threshold → binary image with dark features on white background.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )
    _save_debug_image(debug_dir, "04-thresholded.png", binary)

    # Morphological closing to join broken grid lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    _save_debug_image(debug_dir, "05-closed.png", binary)

    return gray, binary


# ---------------------------------------------------------------------------
# 2. Deskew
# ---------------------------------------------------------------------------

def deskew(gray: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and correct any skew in the image using the dominant line angle
    found by the Hough transform.  Returns corrected ``(gray, binary)``.
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None:
        return gray, binary

    # Collect angles of near-horizontal lines (within 10° of 0 or 180°).
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < 10:
            angles.append(angle_deg)

    if not angles:
        return gray, binary

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray, binary  # negligible skew

    h, w = gray.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    gray_deskewed = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    binary_deskewed = cv2.warpAffine(
        binary, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return gray_deskewed, binary_deskewed


# ---------------------------------------------------------------------------
# 3. Grid detection and cropping
# ---------------------------------------------------------------------------

def find_grid_bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return ``(x, y, w, h)`` bounding box of the largest contour, which
    should be the outer border of the crossword grid.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError(
            "No contours found – check that the image contains a grid."
        )
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def crop_to_grid(
    gray: np.ndarray, binary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both images to the detected grid bounding box."""
    x, y, w, h = find_grid_bbox(binary)
    return gray[y : y + h, x : x + w], binary[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# 4. Grid dimension estimation
# ---------------------------------------------------------------------------

def _count_line_groups(proj: np.ndarray, threshold: float) -> int:
    """Count distinct groups of high-value pixels in a 1-D projection."""
    groups = 0
    in_group = False
    for val in proj > threshold:
        if val and not in_group:
            groups += 1
            in_group = True
        elif not val:
            in_group = False
    return groups


def detect_grid_size(binary_grid: np.ndarray) -> tuple[int, int]:
    """
    Estimate ``(rows, cols)`` by analysing horizontal and vertical line
    projections of the binarised grid image.
    """
    h, w = binary_grid.shape

    h_proj = np.sum(binary_grid, axis=1).astype(float) / (w * 255)
    v_proj = np.sum(binary_grid, axis=0).astype(float) / (h * 255)

    # 0.3 = 30 % of pixels in a row/column must be set to count as a grid line.
    threshold = 0.3
    rows = max(1, _count_line_groups(h_proj, threshold) - 1)
    cols = max(1, _count_line_groups(v_proj, threshold) - 1)
    return rows, cols


# ---------------------------------------------------------------------------
# 5. Cell extraction and black-square detection
# ---------------------------------------------------------------------------

def extract_cells(
    gray_grid: np.ndarray, rows: int, cols: int
) -> list[list[np.ndarray]]:
    """Slice the grid into a 2-D list of individual cell images."""
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


def is_black_cell(cell: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Return True if the cell is mostly dark.

    ``threshold`` is the brightness level (0–1 scale) below which a cell is
    classified as black.  The default of 0.5 means cells whose mean pixel
    brightness is below 50 % are treated as black squares.
    """
    return float(np.mean(cell)) / 255.0 < threshold


# ---------------------------------------------------------------------------
# 6. OCR for clue numbers
# ---------------------------------------------------------------------------

def _prepare_number_crop(cell: np.ndarray, upscale_to: int = 96) -> np.ndarray:
    """
    Extract and upscale the top-left quadrant of a cell where the clue
    number is printed, then binarise and convert to BGR for PaddleOCR.
    """
    h, w = cell.shape
    # Take the top-left 35 % of each dimension.
    crop_h = max(1, int(h * 0.35))
    crop_w = max(1, int(w * 0.35))
    crop = cell[:crop_h, :crop_w]

    # Upscale to a fixed size so small digits are large enough for OCR.
    crop_up = cv2.resize(crop, (upscale_to, upscale_to), interpolation=cv2.INTER_CUBIC)

    # Sharpen after upscaling.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    crop_up = cv2.filter2D(crop_up, -1, kernel)
    crop_up = np.clip(crop_up, 0, 255).astype(np.uint8)

    # Binarise to remove residual artefacts.
    _, crop_bin = cv2.threshold(crop_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2BGR)


def read_cell_numbers(
    cells: list[list[np.ndarray]],
    black_cells: list[list[bool]],
    debug_dir: Path | None = None,
) -> list[list[int]]:
    """
    Use PaddleOCR to read the clue number (integer ≥ 1) from the top-left
    corner of each white cell.  Black cells always get 0.

    Returns a 2-D list of integers (0 = no number / black cell).
    """
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "paddleocr is required.  Install it with:  pip install paddleocr"
        ) from exc

    # Clue numbers are upright crops, so document preprocessing is unnecessary.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    numbers: list[list[int]] = []

    for r in range(rows):
        row_nums: list[int] = []
        for c in range(cols):
            if black_cells[r][c]:
                row_nums.append(0)
                continue

            crop_bgr = _prepare_number_crop(cells[r][c])
            _save_debug_image(
                debug_dir / "ocr-crops" if debug_dir else None,
                f"row-{r + 1:02d}-col-{c + 1:02d}.png",
                crop_bgr,
            )
            ocr_results = ocr.predict(crop_bgr)

            num = 0
            for result in ocr_results:
                for text in result.get("rec_texts", []):
                    text = text.strip()
                    if text.isdigit():
                        num = int(text)
                        break
                if num:
                    break
            row_nums.append(num)
        numbers.append(row_nums)

    return numbers


# ---------------------------------------------------------------------------
# 7. .puz assembly
# ---------------------------------------------------------------------------

def build_puz(
    rows: int,
    cols: int,
    black_cells: list[list[bool]],
    cell_numbers: list[list[int]],
    title: str = "",
    author: str = "",
) -> puz.Puzzle:
    """
    Assemble a ``puz.Puzzle`` from the detected grid data.

    ``cell_numbers`` is used to cross-check the OCR-detected numbers against
    the standard crossword numbering algorithm.  If OCR numbers are present
    and consistent they are used; otherwise the algorithm falls back to
    standard sequential numbering.
    """
    puzzle = puz.Puzzle()
    puzzle.title = title
    puzzle.author = author
    puzzle.width = cols
    puzzle.height = rows

    # .puz format: '.' = black, '-' = empty white square.
    solution = []
    fill = []
    for r in range(rows):
        for c in range(cols):
            if black_cells[r][c]:
                solution.append(".")
                fill.append(".")
            else:
                solution.append("-")
                fill.append("-")

    puzzle.solution = "".join(solution)
    puzzle.fill = "".join(fill)

    # Standard crossword clue numbering.
    across_clues: list[str] = []
    down_clues: list[str] = []
    clue_num = 1

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
                # Use OCR number only when it matches the expected sequential
                # value, guarding against misreads.  If it differs, fall back
                # to the sequential number so clue ordering stays consistent.
                ocr_num = cell_numbers[r][c]
                num = ocr_num if ocr_num == clue_num else clue_num
                if starts_across:
                    across_clues.append(f"{num} Across")
                if starts_down:
                    down_clues.append(f"{num} Down")
                clue_num += 1

    # .puz clue order: all Across in grid-reading order, then all Down.
    puzzle.clues = across_clues + down_clues
    return puzzle


def build_ipuz(
    rows: int,
    cols: int,
    black_cells: list[list[bool]],
    cell_numbers: list[list[int]],
    title: str = "",
    author: str = "",
) -> dict:
    """Assemble an IPUZ crossword document from the detected grid data."""
    grid: list[list[int | str]] = []
    across_clues: list[list[int | str]] = []
    down_clues: list[list[int | str]] = []
    clue_num = 1

    for r in range(rows):
        grid_row: list[int | str] = []
        for c in range(cols):
            if black_cells[r][c]:
                grid_row.append("#")
                continue

            starts_across = (c == 0 or black_cells[r][c - 1]) and (
                c + 1 < cols and not black_cells[r][c + 1]
            )
            starts_down = (r == 0 or black_cells[r - 1][c]) and (
                r + 1 < rows and not black_cells[r + 1][c]
            )
            if starts_across or starts_down:
                ocr_num = cell_numbers[r][c]
                num = ocr_num if ocr_num == clue_num else clue_num
                if starts_across:
                    across_clues.append([num, f"{num} Across"])
                if starts_down:
                    down_clues.append([num, f"{num} Down"])
                clue_num += 1
            else:
                num = 0
            grid_row.append(num)
        grid.append(grid_row)

    return {
        "version": "http://ipuz.org/v2",
        "kind": ["http://ipuz.org/crossword#1"],
        "title": title,
        "author": author,
        "dimensions": {"width": cols, "height": rows},
        "puzzle": grid,
        "clues": {"Across": across_clues, "Down": down_clues},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert(
    image_path: str,
    output_path: str | None = None,
    title: str = "",
    author: str = "",
    use_ocr: bool = True,
    output_format: str = "puz",
    debug_dir: str | Path | None = None,
) -> puz.Puzzle | dict:
    """
    Convert a crossword grid image to a .puz file.

    Parameters
    ----------
    image_path : str
        Path to the input image (JPEG, PNG, TIFF, etc.).
    output_path : str, optional
        Destination .puz path.  Defaults to ``<image>.puz``.
    title : str
        Puzzle title written into the .puz header.
    author : str
        Puzzle author written into the .puz header.
    use_ocr : bool
        Run PaddleOCR to detect clue numbers.  Set to False to rely solely
        on standard sequential numbering.
    output_format : str
        Output format: ``puz`` (default) or ``ipuz``.
    debug_dir : str or pathlib.Path, optional
        Directory for intermediate preprocessing images and OCR input crops.

    Returns
    -------
    puz.Puzzle
        The assembled puzzle (also saved to disk).
    """
    debug_path = Path(debug_dir) if debug_dir else None

    # 1. Preprocess.
    gray, binary = preprocess(image_path, debug_dir=debug_path)

    # 2. Deskew.
    gray, binary = deskew(gray, binary)
    _save_debug_image(debug_path, "06-deskewed-grayscale.png", gray)
    _save_debug_image(debug_path, "07-deskewed-binary.png", binary)

    # 3. Crop to grid.
    gray_grid, binary_grid = crop_to_grid(gray, binary)
    _save_debug_image(debug_path, "08-grid-grayscale.png", gray_grid)
    _save_debug_image(debug_path, "09-grid-binary.png", binary_grid)

    # 4. Estimate grid dimensions.
    rows, cols = detect_grid_size(binary_grid)

    # 5. Slice cells and classify black/white.
    cells = extract_cells(gray_grid, rows, cols)
    black_cells = [
        [is_black_cell(cells[r][c]) for c in range(cols)]
        for r in range(rows)
    ]

    # 6. OCR for clue numbers.
    if use_ocr:
        cell_numbers = read_cell_numbers(cells, black_cells, debug_dir=debug_path)
    else:
        cell_numbers = [[0] * cols for _ in range(rows)]

    # 7. Build and save the requested puzzle format.
    if output_format == "puz":
        puzzle = build_puz(rows, cols, black_cells, cell_numbers, title=title, author=author)
    elif output_format == "ipuz":
        puzzle = build_ipuz(rows, cols, black_cells, cell_numbers, title=title, author=author)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_path is None:
        output_path = str(Path(image_path).with_suffix(f".{output_format}"))

    if output_format == "puz":
        puzzle.save(output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(puzzle, output_file, indent=2)
            output_file.write("\n")
    print(f"Saved {rows}×{cols} {output_format.upper()} puzzle to {output_path}")
    return puzzle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert an image of an empty crossword grid to .puz format.\n"
            "Uses PaddleOCR to read the clue numbers printed in each cell."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to the crossword grid image.")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output .puz file path (default: <image>.puz).",
    )
    parser.add_argument("--title", default="", help="Puzzle title.")
    parser.add_argument("--author", default="", help="Puzzle author.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Save intermediate processing images and OCR input crops here.",
    )
    parser.add_argument(
        "--format",
        choices=("puz", "ipuz"),
        default="puz",
        help="Output format (default: puz).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR and use sequential clue numbering only.",
    )
    args = parser.parse_args(argv)

    try:
        convert(
            image_path=args.image,
            output_path=args.output,
            title=args.title,
            author=args.author,
            use_ocr=not args.no_ocr,
            output_format=args.format,
            debug_dir=args.debug_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
