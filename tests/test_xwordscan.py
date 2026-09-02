"""
Unit tests for xwordscan.

These tests cover the pure-Python / NumPy logic that does not require
PaddleOCR, a real image file, or the puz library to be installed.
"""

import numpy as np
import pytest

import sys
import os
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import xwordscan


# ---------------------------------------------------------------------------
# _count_line_groups
# ---------------------------------------------------------------------------

class TestCountLineGroups:
    def test_no_lines(self):
        proj = np.zeros(20)
        assert xwordscan._count_line_groups(proj, 0.3) == 0

    def test_single_line(self):
        proj = np.zeros(20)
        proj[5:8] = 0.9
        assert xwordscan._count_line_groups(proj, 0.3) == 1

    def test_multiple_lines(self):
        proj = np.zeros(50)
        for start in (5, 15, 25, 35, 45):
            proj[start : start + 2] = 0.9
        assert xwordscan._count_line_groups(proj, 0.3) == 5

    def test_threshold_boundary(self):
        proj = np.array([0.3, 0.3, 0.0, 0.3])
        # values equal to threshold are NOT above it
        assert xwordscan._count_line_groups(proj, 0.3) == 0

    def test_all_above_threshold(self):
        proj = np.ones(10) * 0.5
        assert xwordscan._count_line_groups(proj, 0.3) == 1


# ---------------------------------------------------------------------------
# detect_grid_size
# ---------------------------------------------------------------------------

class TestDetectGridSize:
    def _make_grid_binary(self, rows, cols, cell_size=20):
        """Create a synthetic binary grid image with ``rows`` × ``cols`` cells."""
        h = rows * cell_size + rows + 1
        w = cols * cell_size + cols + 1
        img = np.zeros((h, w), dtype=np.uint8)
        # Draw horizontal lines.
        for r in range(rows + 1):
            y = r * (cell_size + 1)
            img[y, :] = 255
        # Draw vertical lines.
        for c in range(cols + 1):
            x = c * (cell_size + 1)
            img[:, x] = 255
        return img

    def test_5x5(self):
        binary = self._make_grid_binary(5, 5)
        rows, cols = xwordscan.detect_grid_size(binary)
        assert rows == 5
        assert cols == 5

    def test_15x15(self):
        binary = self._make_grid_binary(15, 15)
        rows, cols = xwordscan.detect_grid_size(binary)
        assert rows == 15
        assert cols == 15

    def test_rectangular(self):
        binary = self._make_grid_binary(7, 11)
        rows, cols = xwordscan.detect_grid_size(binary)
        assert rows == 7
        assert cols == 11


# ---------------------------------------------------------------------------
# is_black_cell
# ---------------------------------------------------------------------------

class TestIsBlackCell:
    def test_all_black(self):
        cell = np.zeros((20, 20), dtype=np.uint8)
        assert xwordscan.is_black_cell(cell) is True

    def test_all_white(self):
        cell = np.full((20, 20), 255, dtype=np.uint8)
        assert xwordscan.is_black_cell(cell) is False

    def test_threshold_boundary(self):
        # mean = 127 ≈ 0.498 → below threshold 0.5 → True (black)
        cell = np.full((20, 20), 127, dtype=np.uint8)
        assert xwordscan.is_black_cell(cell, threshold=0.5) is True

    def test_above_threshold(self):
        # mean = 200 ≈ 0.784 → above threshold 0.5 → False (white)
        cell = np.full((20, 20), 200, dtype=np.uint8)
        assert xwordscan.is_black_cell(cell, threshold=0.5) is False

    def test_mostly_white_with_number(self):
        # Simulate a white cell with a small dark number in one corner.
        cell = np.full((40, 40), 240, dtype=np.uint8)
        cell[0:8, 0:8] = 20  # dark corner
        assert xwordscan.is_black_cell(cell) is False


# ---------------------------------------------------------------------------
# extract_cells
# ---------------------------------------------------------------------------

class TestExtractCells:
    def test_shape(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        cells = xwordscan.extract_cells(gray, rows=5, cols=5)
        assert len(cells) == 5
        assert len(cells[0]) == 5

    def test_cell_values(self):
        # Fill quadrants with different intensities so we can verify slicing.
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:50, :50] = 64
        gray[:50, 50:] = 128
        gray[50:, :50] = 192
        gray[50:, 50:] = 255
        cells = xwordscan.extract_cells(gray, rows=2, cols=2)
        assert np.mean(cells[0][0]) == pytest.approx(64.0)
        assert np.mean(cells[0][1]) == pytest.approx(128.0)
        assert np.mean(cells[1][0]) == pytest.approx(192.0)
        assert np.mean(cells[1][1]) == pytest.approx(255.0)


# ---------------------------------------------------------------------------
# read_cell_numbers
# ---------------------------------------------------------------------------

class TestReadCellNumbers:
    def test_uses_paddleocr_3_arguments(self, monkeypatch):
        created_with = {}

        class FakeOCR:
            def __init__(self, **kwargs):
                created_with.update(kwargs)

            def predict(self, image):
                return [{"rec_texts": ["17"]}]

        monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakeOCR))
        cells = [[np.full((20, 20), 255, dtype=np.uint8)]]

        assert xwordscan.read_cell_numbers(cells, [[False]]) == [[17]]
        assert created_with == {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

    def test_saves_prepared_ocr_crop(self, monkeypatch, tmp_path):
        class FakeOCR:
            def __init__(self, **kwargs):
                pass

            def predict(self, image):
                return [{"rec_texts": []}]

        monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakeOCR))
        cells = [[np.full((20, 20), 255, dtype=np.uint8)]]

        xwordscan.read_cell_numbers(cells, [[False]], debug_dir=tmp_path)

        assert (tmp_path / "ocr-crops" / "row-01-col-01.png").is_file()


# ---------------------------------------------------------------------------
# build_puz
# ---------------------------------------------------------------------------

class TestBuildPuz:
    def _simple_black(self, rows, cols):
        return [[False] * cols for _ in range(rows)]

    def _no_numbers(self, rows, cols):
        return [[0] * cols for _ in range(rows)]

    def test_all_white_3x3(self):
        import puz  # noqa: PLC0415
        rows, cols = 3, 3
        bc = self._simple_black(rows, cols)
        cn = self._no_numbers(rows, cols)
        puzzle = xwordscan.build_puz(rows, cols, bc, cn)
        assert puzzle.width == cols
        assert puzzle.height == rows
        assert "." not in puzzle.solution
        assert len(puzzle.solution) == rows * cols

    def test_black_cell_encoding(self):
        import puz  # noqa: PLC0415
        rows, cols = 3, 3
        bc = self._simple_black(rows, cols)
        bc[1][1] = True  # centre cell is black
        cn = self._no_numbers(rows, cols)
        puzzle = xwordscan.build_puz(rows, cols, bc, cn)
        # Centre cell (index 4 in 3×3) must be '.'
        assert puzzle.solution[4] == "."
        assert puzzle.fill[4] == "."

    def test_title_and_author(self):
        import puz  # noqa: PLC0415
        bc = self._simple_black(3, 3)
        cn = self._no_numbers(3, 3)
        puzzle = xwordscan.build_puz(3, 3, bc, cn, title="My Puzzle", author="Jane")
        assert puzzle.title == "My Puzzle"
        assert puzzle.author == "Jane"

    def test_clue_count_all_white_5x5(self):
        import puz  # noqa: PLC0415
        # All-white 5×5: 5 across clues (one per row) + 5 down clues (one per col).
        rows, cols = 5, 5
        bc = self._simple_black(rows, cols)
        cn = self._no_numbers(rows, cols)
        puzzle = xwordscan.build_puz(rows, cols, bc, cn)
        assert len(puzzle.clues) == rows + cols


# ---------------------------------------------------------------------------
# build_ipuz
# ---------------------------------------------------------------------------

class TestBuildIpuz:
    def test_grid_metadata_and_clues(self):
        black_cells = [[False, False, False], [False, True, False], [False, False, False]]
        cell_numbers = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        puzzle = xwordscan.build_ipuz(
            3,
            3,
            black_cells,
            cell_numbers,
            title="My Puzzle",
            author="Jane",
        )

        assert puzzle["version"] == "http://ipuz.org/v2"
        assert puzzle["kind"] == ["http://ipuz.org/crossword#1"]
        assert puzzle["dimensions"] == {"width": 3, "height": 3}
        assert puzzle["title"] == "My Puzzle"
        assert puzzle["author"] == "Jane"
        assert puzzle["puzzle"] == [[1, 0, 2], [0, "#", 0], [3, 0, 0]]
        assert puzzle["clues"] == {
            "Across": [[1, "1 Across"], [3, "3 Across"]],
            "Down": [[1, "1 Down"], [2, "2 Down"]],
        }


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shapes_match(self, tmp_path):
        # Create a small white PNG and verify preprocess returns arrays of the
        # same spatial dimensions.
        import cv2  # noqa: PLC0415
        img_path = str(tmp_path / "white.png")
        cv2.imwrite(img_path, np.full((100, 100, 3), 255, dtype=np.uint8))
        gray, binary = xwordscan.preprocess(img_path)
        assert gray.shape == binary.shape

    def test_saves_intermediate_images(self, tmp_path):
        import cv2  # noqa: PLC0415
        img_path = str(tmp_path / "white.png")
        cv2.imwrite(img_path, np.full((100, 100, 3), 255, dtype=np.uint8))

        xwordscan.preprocess(img_path, debug_dir=tmp_path / "debug")

        debug_dir = tmp_path / "debug"
        assert {path.name for path in debug_dir.iterdir()} == {
            "01-grayscale.png",
            "02-denoised.png",
            "03-normalized.png",
            "04-thresholded.png",
            "05-closed.png",
        }

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            xwordscan.preprocess("/nonexistent/path/image.png")
