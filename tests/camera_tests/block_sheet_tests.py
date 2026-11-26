import contextlib
import dataclasses
import io
import os
import re
import tempfile
import unittest

import cv2
import numpy as np

from src.block_stacking.stacking_config import StackingConfig, aruco_dictionary
from src.camera.aruco_generation import block_sheet
from src.camera.aruco_generation.block_sheet import A4_LONG, A4_SHORT, BoardPage, MarkerPage, Rect, SheetError
from src.camera.block_detection import board_image, make_board
from src.camera.util import detect_aruco_markers, interpolate_charuco_corners
from tests.block_stacking_tests.frozen_config import FROZEN_STACKING_CONFIG_PATH, load_frozen_config

PPC = 100  # pixels per cm of the rendered pages, a pixel is 0.1 mm
# A4 in points, a point is 1/72 inch
A4_LONG_POINTS = A4_LONG / 2.54 * 72
A4_SHORT_POINTS = A4_SHORT / 2.54 * 72


def to_pixels(page_height: float, x: float, y: float) -> tuple[float, float]:
    """
    where the point (x, y) cm of the page is in OpenCV's pixel coordinates of the rendered image: row 0 is the top and
    the middle of the first pixel is 0, while the page has its edge at 0
    """
    return x * PPC - 0.5, (page_height - y) * PPC - 0.5


class SheetTestCase(unittest.TestCase):
    config: StackingConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_frozen_config()


class LayoutTests(SheetTestCase):
    def test_board_is_centred_on_the_landscape_page(self) -> None:
        page = block_sheet.layout_board_page(self.config.board)
        self.assertEqual((A4_LONG, A4_SHORT), (page.width, page.height))
        board = self.config.board
        width = board.squares_x * board.square_length
        height = board.squares_y * board.square_length
        self.assertAlmostEqual(width, page.board.width)
        self.assertAlmostEqual(height, page.board.height)
        self.assertAlmostEqual((A4_LONG - width) / 2, page.board.left)
        self.assertAlmostEqual((A4_SHORT - height) / 2, page.board.bottom)
        self.assertAlmostEqual(page.board.left, page.width - page.board.right)
        self.assertAlmostEqual(page.board.bottom, page.height - page.board.top)

    def test_the_origin_is_the_bottom_right_corner_of_the_board(self) -> None:
        page = block_sheet.layout_board_page(self.config.board)
        self.assertEqual((page.board.right, page.board.bottom), page.origin)

    def test_the_default_board_is_15_by_9_cm_with_room_for_the_arrows(self) -> None:
        # the numbers of the default configuration, the margin right of the board holds the whole x arrow and its tip
        page = block_sheet.layout_board_page(self.config.board)
        self.assertAlmostEqual(15.0, page.board.width)
        self.assertAlmostEqual(9.0, page.board.height)
        self.assertAlmostEqual(7.35, page.board.left)
        self.assertAlmostEqual(6.0, page.board.bottom)
        self.assertEqual(block_sheet.ARROW_LENGTH, page.arrow_length)

    def test_the_x_arrow_is_shorter_when_the_margin_right_of_the_board_is_narrow(self) -> None:
        page = block_sheet.layout_board_page(dataclasses.replace(self.config.board, squares_x=9))
        self.assertAlmostEqual(1.35, page.width - page.board.right)
        self.assertLess(page.arrow_length, block_sheet.ARROW_LENGTH)
        self.assertGreater(page.arrow_length, 0.0)
        self.assertLessEqual(page.board.right + block_sheet.ARROW_OFFSET + page.arrow_length, page.width - 0.4)

    def test_the_ruler_is_at_the_bottom_left_away_from_the_origin_and_the_x_arrow(self) -> None:
        page = block_sheet.layout_board_page(self.config.board)
        self.assertTrue(page.ruler.text_on_right)
        # the text is about 8 cm long: it ends well before the origin mark, which is at the right edge of the board
        self.assertLess(page.ruler.end[0] + 0.3 + 9.0, page.origin[0])

    def test_the_layout_follows_the_configuration(self) -> None:
        board = dataclasses.replace(self.config.board, squares_x=6, squares_y=4, square_length=4.0, marker_length=3.0)
        page = block_sheet.layout_board_page(board)
        self.assertAlmostEqual(24.0, page.board.width)
        self.assertAlmostEqual(16.0, page.board.height)
        self.assertAlmostEqual((A4_LONG - 24.0) / 2, page.board.left)
        self.assertAlmostEqual((A4_SHORT - 16.0) / 2, page.board.bottom)

    def test_the_ruler_is_10_cm_and_inside_the_page(self) -> None:
        for page in (block_sheet.layout_board_page(self.config.board),
                     block_sheet.layout_marker_page(self.config.blocks)):
            (start_x, start_y), (end_x, end_y) = page.ruler.start, page.ruler.end
            self.assertAlmostEqual(10.0, end_x - start_x)
            self.assertEqual(start_y, end_y)
            self.assertGreaterEqual(start_x, 0.5)
            self.assertLessEqual(end_x, page.width - 0.5)
            self.assertGreater(start_y, 0.0)

    def test_marker_cells_are_blocks_with_the_marker_centred(self) -> None:
        page = block_sheet.layout_marker_page(self.config.blocks)
        self.assertEqual((A4_SHORT, A4_LONG), (page.width, page.height))
        self.assertEqual(list(self.config.blocks.marker_ids), [cell.marker_id for cell in page.cells])
        for cell in page.cells:
            self.assertAlmostEqual(3.0, cell.cut.width)
            self.assertAlmostEqual(3.0, cell.cut.height)
            self.assertAlmostEqual(2.4, cell.marker.width)
            self.assertAlmostEqual(2.4, cell.marker.height)
            self.assertAlmostEqual(0.3, cell.marker.left - cell.cut.left)
            self.assertAlmostEqual(0.3, cell.cut.right - cell.marker.right)
            self.assertAlmostEqual(0.3, cell.marker.bottom - cell.cut.bottom)
            self.assertAlmostEqual(0.3, cell.cut.top - cell.marker.top)

    def test_marker_cells_are_apart_and_on_the_page(self) -> None:
        page = block_sheet.layout_marker_page(self.config.blocks)
        for index, cell in enumerate(page.cells):
            self.assertGreaterEqual(cell.cut.left, 1.0)
            self.assertLessEqual(cell.cut.right, page.width - 1.0)
            self.assertGreaterEqual(cell.cut.bottom, 2.0)
            self.assertLessEqual(cell.cut.top, page.height - 1.0)
            for other in page.cells[index + 1:]:
                gap_x = max(other.cut.left - cell.cut.right, cell.cut.left - other.cut.right)
                gap_y = max(other.cut.bottom - cell.cut.top, cell.cut.bottom - other.cut.top)
                self.assertGreaterEqual(max(gap_x, gap_y), 1.0 - 1e-9)

    def test_a_board_that_does_not_fit_is_refused_with_a_message(self) -> None:
        # 10 squares of 3 cm are 30 cm, which is more than the 29.7 cm of the page
        wide = dataclasses.replace(self.config.board, squares_x=10, squares_y=5, square_length=3.0, marker_length=2.2)
        with self.assertRaisesRegex(SheetError, r'30 x 15 cm.*does not fit.*margin of 1 cm'):
            block_sheet.layout_board_page(wide)
        # 7 squares of 3 cm are 21 cm, which is the whole height of the page
        high = dataclasses.replace(self.config.board, squares_x=7, squares_y=7, square_length=3.0, marker_length=2.2)
        with self.assertRaisesRegex(SheetError, r'21 x 21 cm.*does not fit'):
            block_sheet.layout_board_page(high)
        # just too big for the margins of 1 cm: 27.7 cm is the most
        edge = dataclasses.replace(self.config.board, squares_x=2, squares_y=2, square_length=13.9, marker_length=10.0)
        with self.assertRaisesRegex(SheetError, r'27.8 x 27.8 cm.*does not fit'):
            block_sheet.layout_board_page(edge)

    def test_a_board_that_just_fits_is_accepted(self) -> None:
        page = block_sheet.layout_board_page(dataclasses.replace(self.config.board, squares_x=9, square_length=3.0,
                                                                 marker_length=2.2, squares_y=6))
        self.assertAlmostEqual(1.35, page.board.left)
        self.assertAlmostEqual(1.5, page.board.bottom)

    def test_too_many_markers_for_one_page_are_refused_with_a_message(self) -> None:
        many = dataclasses.replace(self.config.blocks, marker_ids=tuple(range(50)))
        with self.assertRaisesRegex(SheetError, r'50 markers.*do not fit on one A4 page'):
            block_sheet.layout_marker_page(many)

    def test_blocks_that_are_too_big_for_the_page_are_refused(self) -> None:
        big = dataclasses.replace(self.config.blocks, size=25.0, marker_length=24.0)
        with self.assertRaisesRegex(SheetError, r'do not fit'):
            block_sheet.layout_marker_page(big)

    def test_the_most_markers_that_fit_are_accepted(self) -> None:
        # 5 columns and 6 rows with the gap of 1 cm
        full = dataclasses.replace(self.config.blocks, marker_ids=tuple(range(30)))
        page = block_sheet.layout_marker_page(full)
        self.assertEqual(30, len(page.cells))
        with self.assertRaises(SheetError):
            block_sheet.layout_marker_page(dataclasses.replace(self.config.blocks, marker_ids=tuple(range(31))))


class BoardPageTests(SheetTestCase):
    page: BoardPage
    image: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.page = block_sheet.layout_board_page(cls.config.board)
        cls.image = block_sheet.render_page(cls.page, cls.config, PPC)

    def detect_charuco(self) -> dict[int, np.ndarray]:
        """the inner corners of the board found in the image, by corner id, in pixels"""
        board = make_board(self.config.board)
        corners, ids = detect_aruco_markers(self.image, aruco_dictionary(self.config.board.dictionary))
        self.assertIsNotNone(ids)
        assert ids is not None
        self.assertEqual(self.config.board.squares_x * self.config.board.squares_y // 2, len(ids))  # every marker
        charuco_corners, charuco_ids = interpolate_charuco_corners(board, self.image, corners, ids)
        assert charuco_corners is not None and charuco_ids is not None
        return {int(corner_id): point for corner_id, point in zip(charuco_ids.ravel(), charuco_corners.reshape(-1, 2))}

    def test_image_has_the_size_of_the_page(self) -> None:
        self.assertEqual((2100, 2970), self.image.shape)

    def test_the_squares_are_the_configured_size(self) -> None:
        found = self.detect_charuco()
        # every inner corner of the board
        self.assertEqual((self.config.board.squares_x - 1) * (self.config.board.squares_y - 1), len(found))
        columns = self.config.board.squares_x - 1
        for corner_id, point in found.items():
            # a neighbour in the same row is one square away, and so is one in the same column
            for neighbour_id, in_row in ((corner_id + 1, True), (corner_id + columns, False)):
                if neighbour_id not in found or (in_row and (corner_id + 1) % columns == 0):
                    continue
                distance = float(np.linalg.norm(found[neighbour_id] - point)) / PPC
                self.assertAlmostEqual(self.config.board.square_length, distance,
                                       delta=self.config.board.square_length * 0.01)

    def test_corners_are_where_the_board_frame_places_them(self) -> None:
        board = make_board(self.config.board)
        width = self.config.board.squares_x * self.config.board.square_length
        height = self.config.board.squares_y * self.config.board.square_length
        object_corners = board.getChessboardCorners()
        for corner_id, point in self.detect_charuco().items():
            # the corner in OpenCV's frame is (x, y) from the top left, in ours (x - width, height - y) from the bottom
            # right corner, which is the origin
            x, y_opencv, _ = object_corners[int(corner_id)]
            expected = to_pixels(self.page.height, self.page.origin[0] + float(x) - width,
                                 self.page.origin[1] + height - float(y_opencv))
            self.assertAlmostEqual(expected[0], point[0], delta=0.03 * PPC)
            self.assertAlmostEqual(expected[1], point[1], delta=0.03 * PPC)

    def test_the_corner_of_the_origin_is_the_first_of_the_board_frame_x_to_the_right_y_up(self) -> None:
        found = self.detect_charuco()
        columns = self.config.board.squares_x - 1
        rows = self.config.board.squares_y - 1
        square = self.config.board.square_length
        # corner 0 is the first inner corner of the top row, so the last corner of the last row is the one closest to
        # the origin at the bottom right: one square left of it, which is x = -1 square, and one square up
        closest = found[rows * columns - 1]
        origin_x, origin_y = to_pixels(self.page.height, *self.page.origin)
        self.assertAlmostEqual(origin_x - square * PPC, closest[0], delta=0.03 * PPC)
        self.assertAlmostEqual(origin_y - square * PPC, closest[1], delta=0.03 * PPC)

    def test_the_vector_board_is_the_board_image_of_the_detection_module(self) -> None:
        # At 60 pixels per cm the squares (180 pixels) and the modules of the markers (22 pixels) are whole pixels and
        # the board starts at a whole pixel of the page. The black squares then have to be the same as those of
        # board_image. OpenCV places a marker in its square by truncating floats, which puts it 0 or 1 pixel off the
        # middle (the vector marker is exactly in the middle): so a marker has to be the same after a shift of at most
        # 1 pixel, and with that shift all its bits are the same. That is a rounding of OpenCV's picture, not a
        # difference of the sheet.
        ppc = 60
        image = block_sheet.render_page(self.page, self.config, ppc)
        board = self.page.board
        top = round((self.page.height - board.top) * ppc)
        left = round(board.left * ppc)
        self.assertAlmostEqual((self.page.height - board.top) * ppc, top)
        self.assertAlmostEqual(board.left * ppc, left)
        reference = board_image(self.config.board, ppc) < 128
        crop = image[top:top + reference.shape[0], left:left + reference.shape[1]] < 128
        self.assertEqual(reference.shape, crop.shape)
        square = round(self.config.board.square_length * ppc)
        markers = 0
        for row in range(self.config.board.squares_y):
            for column in range(self.config.board.squares_x):
                expected = reference[row * square:(row + 1) * square, column * square:(column + 1) * square]
                drawn = crop[row * square:(row + 1) * square, column * square:(column + 1) * square]
                if (row + column) % 2 == 0:
                    self.assertTrue(np.array_equal(expected, drawn), 'black square {}, {}'.format(column, row))
                    continue
                markers += 1
                differences = [int((np.roll(drawn, (dy, dx), axis=(0, 1))[2:-2, 2:-2] != expected[2:-2, 2:-2]).sum())
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                self.assertEqual(0, min(differences), 'marker in square {}, {}'.format(column, row))
        self.assertEqual(self.config.board.squares_x * self.config.board.squares_y // 2, markers)

    def test_edges_are_crisp(self) -> None:
        # the board has no grey but on the border pixels of its edges: of all the pixels of a square in the
        # middle of the board, the grey ones are only those on the edge of a module
        board = self.page.board
        square = self.config.board.square_length
        top = round((self.page.height - board.top) * PPC)
        left = round(board.left * PPC)
        inside = self.image[top + 2:top + round(square * PPC) - 2, left + 2:left + round(square * PPC) - 2]
        self.assertEqual(0, int(inside.max()))  # the first square is black all through, no grey blur in it

    def test_the_x_arrow_runs_to_the_right_and_the_y_arrow_up_from_the_origin_mark(self) -> None:
        board = self.page.board
        offset = block_sheet.ARROW_OFFSET
        length = self.page.arrow_length
        mark_x, mark_y = board.right + offset, board.bottom - offset
        # the x arrow is a line under the bottom edge of the board, from the origin mark away from the board
        row_top = round((self.page.height - mark_y) * PPC) - 2
        band = self.image[row_top:row_top + 5, :] < 128
        for x in np.arange(mark_x + 0.1, mark_x + length - 0.3, 0.1):
            self.assertTrue(band[:, round(x * PPC)].any(), 'no x arrow at {:.1f} cm'.format(x))
        # and nothing of it is to the left of the origin mark
        self.assertFalse(band[:, round((mark_x - 0.4) * PPC):round((mark_x - 0.1) * PPC)].any())
        # the y arrow is a line right of the right edge, from the origin mark up
        column_left = round(mark_x * PPC) - 2
        band = self.image[:, column_left:column_left + 5] < 128
        for y in np.arange(mark_y + 0.1, mark_y + length - 0.3, 0.1):
            self.assertTrue(band[round((self.page.height - y) * PPC), :].any(), 'no y arrow at {:.1f} cm'.format(y))
        # the arrows are on the paper next to the board, not on it
        self.assertGreater(offset, 0.0)
        self.assertLess(mark_x + length, self.page.width)

    def test_the_margin_next_to_the_right_edge_is_clear_but_for_the_y_arrow_and_the_origin(self) -> None:
        board = self.page.board
        rows = slice(round((self.page.height - board.top) * PPC), round((self.page.height - board.bottom) * PPC) - 5)
        strip = self.image[rows, round(board.right * PPC) + 2:round((board.right + 0.3) * PPC)]
        self.assertEqual(255, int(strip.min()))
        # the left edge of the board has no arrow any more
        left = self.image[rows, round((board.left - 0.3) * PPC):round(board.left * PPC) - 2]
        self.assertEqual(255, int(left.min()))

    def test_the_ruler_does_not_touch_the_origin_mark_and_the_arrow(self) -> None:
        y = self.page.ruler.start[1]
        band = self.image[round((self.page.height - y - 0.3) * PPC):round((self.page.height - y + 0.3) * PPC), :] < 128
        columns = np.nonzero(band.any(axis=0))[0]
        self.assertGreater(len(columns), 0)
        # the line and its text are the only things in the band, from the margin to the end of the text
        self.assertGreaterEqual(columns.min() / PPC, 0.5)
        self.assertLess(columns.max() / PPC, self.page.origin[0] - 1.0)

    def test_the_ruler_is_10_cm_in_the_image(self) -> None:
        (start_x, y), (end_x, _) = self.page.ruler.start, self.page.ruler.end
        # a row just above the line crosses only the two end ticks
        row = self.image[round((self.page.height - y - 0.1) * PPC)]
        ticks = []
        for x in (start_x, end_x):
            window = row[round((x - 0.2) * PPC):round((x + 0.2) * PPC)]
            columns = np.nonzero(window < 128)[0]
            self.assertGreater(len(columns), 0)
            ticks.append((columns.mean() + 0.5) / PPC + x - 0.2)  # the middle of the dark pixels, in cm
        self.assertAlmostEqual(10.0, ticks[1] - ticks[0], delta=0.02)


class MarkerPageTests(SheetTestCase):
    page: MarkerPage
    image: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.page = block_sheet.layout_marker_page(cls.config.blocks)
        cls.image = block_sheet.render_page(cls.page, cls.config, PPC)

    def detect(self) -> dict[int, np.ndarray]:
        corners, ids = detect_aruco_markers(self.image, aruco_dictionary(self.config.blocks.dictionary))
        self.assertIsNotNone(ids)
        assert ids is not None
        found: dict[int, np.ndarray] = {}
        for marker_corners, marker_id in zip(corners, ids.ravel()):
            self.assertNotIn(int(marker_id), found, 'id {} is found twice'.format(marker_id))
            found[int(marker_id)] = marker_corners.reshape(4, 2)
        return found

    def test_image_has_the_size_of_the_page(self) -> None:
        self.assertEqual((2970, 2100), self.image.shape)

    def test_every_configured_id_is_found_exactly_once(self) -> None:
        self.assertEqual(sorted(self.config.blocks.marker_ids), sorted(self.detect()))

    def test_markers_have_the_configured_size(self) -> None:
        marker_length = self.config.blocks.marker_length
        for marker_id, corners in self.detect().items():
            for side in range(4):
                length = float(np.linalg.norm(corners[(side + 1) % 4] - corners[side])) / PPC
                self.assertAlmostEqual(marker_length, length, delta=marker_length * 0.02,
                                       msg='side {} of marker {}'.format(side, marker_id))

    def test_markers_are_centred_in_their_cut_square(self) -> None:
        found = self.detect()
        for cell in self.page.cells:
            centre = found[cell.marker_id].mean(axis=0)
            expected = to_pixels(self.page.height, *cell.cut.centre)
            self.assertAlmostEqual(expected[0], centre[0], delta=0.03 * PPC)
            self.assertAlmostEqual(expected[1], centre[1], delta=0.03 * PPC)

    def test_cut_lines_are_squares_of_the_size_of_a_block(self) -> None:
        # the grey line of every cut square: find it in the image as the grey pixels of the middle row of the cell
        size = self.config.blocks.size
        for cell in self.page.cells:
            row = self.image[round((self.page.height - cell.cut.centre[1]) * PPC) - 1]
            # the white edge of the cell is 0.3 cm, so the marker starts that far in from the line
            window = row[round((cell.cut.left - 0.3) * PPC):round((cell.cut.right + 0.3) * PPC)]
            grey = np.nonzero((window > 60) & (window < 235))[0]
            self.assertGreater(len(grey), 0)
            first_line = window[:round(0.6 * PPC)]
            last_line = window[-round(0.6 * PPC):]
            left = np.nonzero((first_line > 60) & (first_line < 235))[0].mean()
            right = np.nonzero((last_line > 60) & (last_line < 235))[0].mean() + len(window) - len(last_line)
            self.assertAlmostEqual(size, (right - left) / PPC, delta=0.02)

    def test_the_ruler_is_10_cm_in_the_image(self) -> None:
        (start_x, y), (end_x, _) = self.page.ruler.start, self.page.ruler.end
        row = self.image[round((self.page.height - y - 0.1) * PPC)]
        ticks = []
        for x in (start_x, end_x):
            window = row[round((x - 0.2) * PPC):round((x + 0.2) * PPC)]
            columns = np.nonzero(window < 128)[0]
            self.assertGreater(len(columns), 0)
            ticks.append((columns.mean() + 0.5) / PPC + x - 0.2)
        self.assertAlmostEqual(10.0, ticks[1] - ticks[0], delta=0.02)

    def test_edges_of_the_black_modules_are_crisp(self) -> None:
        # a module of 0.4 cm is 40 pixels: the first one of the border is black through, no grey in its middle
        cell = self.page.cells[0]
        module = self.config.blocks.marker_length / (aruco_dictionary(self.config.blocks.dictionary).markerSize + 2)
        top = round((self.page.height - cell.marker.top) * PPC)
        left = round(cell.marker.left * PPC)
        inside = self.image[top + 2:top + round(module * PPC) - 2, left + 2:left + round(module * PPC) - 2]
        self.assertEqual(0, int(inside.max()))


class PdfTests(SheetTestCase):
    def test_pdf_has_a_landscape_and_a_portrait_a4_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sheet.pdf')
            block_sheet.write_pdf(self.config, path)
            with open(path, 'rb') as pdf_file:
                data = pdf_file.read()
        self.assertTrue(data.startswith(b'%PDF'))
        self.assertEqual(2, len(re.findall(rb'/Type\s*/Page\b', data)))
        boxes = [[float(number) for number in match.split()]
                 for match in (box.decode('ascii') for box in re.findall(rb'/MediaBox\s*\[([^\]]*)\]', data))]
        self.assertEqual(2, len(boxes))
        sizes = sorted((box[2] - box[0], box[3] - box[1]) for box in boxes)
        # the portrait page first, it is the narrow one
        self.assertAlmostEqual(A4_SHORT_POINTS, sizes[0][0], delta=0.1)
        self.assertAlmostEqual(A4_LONG_POINTS, sizes[0][1], delta=0.1)
        self.assertAlmostEqual(A4_LONG_POINTS, sizes[1][0], delta=0.1)
        self.assertAlmostEqual(A4_SHORT_POINTS, sizes[1][1], delta=0.1)
        # and in the order of the sheet: the board first
        first = boxes[0]
        self.assertGreater(first[2] - first[0], first[3] - first[1])

    def test_the_figures_have_the_size_of_the_pages_in_inches(self) -> None:
        for page in block_sheet.make_layouts(self.config):
            width, height = block_sheet.build_figure(page, self.config).get_size_inches()
            self.assertAlmostEqual(page.width / 2.54, width)
            self.assertAlmostEqual(page.height / 2.54, height)

    def test_the_black_is_vector_graphics(self) -> None:
        # no image in the pdf, so there is no resolution and no smoothing of the edges
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sheet.pdf')
            block_sheet.write_pdf(self.config, path)
            with open(path, 'rb') as pdf_file:
                data = pdf_file.read()
        self.assertNotIn(b'/Subtype /Image', data)
        self.assertNotIn(b'/Subtype/Image', data)

    def test_main_writes_the_file_it_is_asked_to(self) -> None:
        # with the live configuration of the project, which the user edits: it only has to make a sheet from it
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'inner', 'sheet.pdf')
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                block_sheet.main(['--output', path])
            self.assertIn(path, printed.getvalue())
            self.assertTrue(os.path.getsize(path) > 1000)

    def test_main_writes_the_file_with_the_frozen_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sheet.pdf')
            with contextlib.redirect_stdout(io.StringIO()):
                block_sheet.main(['--config', FROZEN_STACKING_CONFIG_PATH, '--output', path])
            self.assertTrue(os.path.getsize(path) > 1000)

    def test_main_stops_with_a_message_when_the_board_does_not_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, 'stacking.yaml')
            with open(FROZEN_STACKING_CONFIG_PATH, encoding='utf-8') as source:
                text = source.read().replace('squares: [5, 3]', 'squares: [10, 5]')
            with open(config_path, 'w', encoding='utf-8') as target:
                target.write(text)
            output = os.path.join(directory, 'sheet.pdf')
            with self.assertRaises(SystemExit) as context:
                block_sheet.main(['--config', config_path, '--output', output])
            self.assertIn('does not fit', str(context.exception))
            self.assertFalse(os.path.exists(output))

    def test_default_output_is_in_the_documentation_folder_of_the_project(self) -> None:
        path = block_sheet.DEFAULT_OUTPUT_PATH
        self.assertEqual('block_stacking_markers_a4.pdf', path.name)
        self.assertEqual('documentation', path.parent.name)
        self.assertTrue((path.parent.parent / 'src' / 'camera').is_dir())


class RectTests(unittest.TestCase):
    def test_edges_and_centre(self) -> None:
        rect = Rect(1.0, 2.0, 3.0, 4.0)
        self.assertEqual((4.0, 6.0), (rect.right, rect.top))
        self.assertEqual((2.5, 4.0), rect.centre)


if __name__ == '__main__':
    unittest.main()
