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
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en) (CPU build is sufficient)
- PaddleOCR
- OpenCV
- NumPy
- [puz](https://github.com/alexdej/puzpy)

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** PaddlePaddle installation varies by platform. See the
> [official guide](https://www.paddlepaddle.org.cn/install/quick) if the above
> command fails.

## Usage

```bash
# Basic usage – output written to <image>.puz
python xwordscan.py grid.jpg

# Specify output file
python xwordscan.py grid.png my_puzzle.puz

# Add metadata
python xwordscan.py grid.png --title "Daily Crossword" --author "A. Setter"

# Skip OCR (use sequential clue numbering only)
python xwordscan.py grid.png --no-ocr
```

## Tips for best results

- Use a **flat, well-lit scan** rather than a phone photo where possible.
- Minimum recommended image width: **800 px** (larger is better for OCR accuracy).
- The grid should occupy most of the image frame.
- JPEG, PNG, and TIFF formats are all supported.

## Running tests

```bash
pip install pytest
pytest tests/
```
