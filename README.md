# xwordscan

A utility to convert an image of an empty crossword grid to [.puz format](https://code.google.com/archive/p/puz/wikis/FileFormat.wiki).

## How it works

1. **Preprocess** – converts to grayscale, denoises with a bilateral filter, normalises contrast, and binarises with an adaptive threshold to handle colour images, JPEG artefacts, shadows, and uneven lighting.
2. **Deskew** – detects and corrects any rotation using the Hough line transform.
3. **Detect grid** – finds the outer grid boundary via contour detection and crops to it.
4. **Estimate dimensions** – counts horizontal and vertical grid lines from pixel projections to determine rows × columns.
5. **Classify cells** – each cell is classified as black (filled) or white (empty) by its mean brightness.
6. **OCR for clue numbers** – for each white cell, the top-left quadrant (where the clue number is printed) is cropped, upscaled, sharpened, and binarised before being passed to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to read the number.
7. **Build .puz** – assembles a valid `.puz` file using standard crossword clue-numbering rules.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

Dependencies (all from official PyPI maintainers, pinned to versions ≥ 2 weeks old):

| Package | Source |
|---|---|
| `opencv-python-headless` | [opencv/opencv-python](https://github.com/opencv/opencv-python) |
| `numpy` | [numpy.org](https://numpy.org) |
| `paddlepaddle` | [paddlepaddle.org.cn](https://www.paddlepaddle.org.cn) (CPU build sufficient) |
| `paddleocr` | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `puzpy` | [alexdej/puzpy](https://github.com/alexdej/puzpy) |

### Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) installs exact versions from the committed `uv.lock` file,
guaranteeing reproducible, auditable installs regardless of when you run the command.

```bash
# Install uv (one-time)
pip install uv

# Create a virtual environment and install locked dependencies
uv sync

# Run the tool inside the uv-managed environment
uv run xwordscan grid.jpg
```

Use `uv sync --frozen` in CI to enforce the exact locked versions without allowing updates.

### Install with pip (alternative)

```bash
pip install -r requirements.txt
```

> **Note:** PaddlePaddle installation varies by platform. See the
> [official guide](https://www.paddlepaddle.org.cn/install/quick) if the above
> command fails.

## Usage

```bash
# Basic usage – output written to <image>.puz
uv run xwordscan grid.jpg

# Specify output file
uv run xwordscan grid.png my_puzzle.puz

# Add metadata
uv run xwordscan grid.png --title "Daily Crossword" --author "A. Setter"

# Skip OCR (use sequential clue numbering only)
uv run xwordscan grid.png --no-ocr

# Write an IPUZ file instead of PUZ
uv run xwordscan grid.png --format ipuz
```

## Tips for best results

- Use a **flat, well-lit scan** rather than a phone photo where possible.
- Minimum recommended image width: **800 px** (larger is better for OCR accuracy).
- The grid should occupy most of the image frame.
- JPEG, PNG, and TIFF formats are all supported.

## Running tests

```bash
uv sync                  # installs dev dependencies including pytest
uv run pytest tests/
```
