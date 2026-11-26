"""
Makes the A4 pdf to print for the block stacking, at true size: 1 cm on the paper is 1 cm in the configuration.

Page 1 (A4 landscape) is the charuco board, with the origin and the two arrows of the board frame in the margin: the
origin is the bottom right corner of the board, x points right (away from the board) and y points up along the right
edge.
Page 2 (A4 portrait) has the aruco marker of every block id, centred in a square of the size of a block with a cut line
round it. Both pages have a line of 10 cm to check that the printer did not scale the page.

The black parts are drawn as vector rectangles, from the bits of the markers and the squares of the board, so the edges
are as sharp as the printer makes them and the size does not depend on a resolution. The layout is worked out in cm by
layout_board_page and layout_marker_page, without drawing anything, and render_page draws a page into an image.

Run from the project root: python -m src.camera.aruco_generation.block_sheet [--output PATH] [--config PATH]
"""
import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from src.block_stacking.stacking_config import BlockConfig, BoardConfig, StackingConfig, StackingConfigError, \
    aruco_dictionary, load_stacking_config
from src.camera.block_detection import make_board

# the project root is 3 folders above this file: src/camera/aruco_generation/block_sheet.py. The get_project_root of
# src/utils/os_utils.py gives the src folder, which is not what the documentation folder lives in.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'documentation' / 'block_stacking_markers_a4.pdf'

CM_PER_INCH = 2.54
A4_LONG = 29.7  # cm
A4_SHORT = 21.0
# printers can not print up to the edge of the paper, and the board needs white paper round it to be found
MIN_MARGIN = 1.0
RULER_LENGTH = 10.0
RULER_TEXT = '10 cm: measure this line, print at 100% (no fit to page)'
_RULER_TICK = 0.2  # half the height of an end tick
_RULER_LINE_WIDTH = 0.8  # points, like the other lines
ARROW_LENGTH = 3.0  # at most: it is shorter when the margin right of the board is too narrow for it
ARROW_OFFSET = 0.4  # cm between the edge of the board and its arrows
_ARROW_MIN_LENGTH = 0.5
_ARROW_END_SPACE = 0.6  # between the tip of the x arrow and the edge of the page
_INFO_FONT_SIZE = 7  # points
_LABEL_FONT_SIZE = 9
_LINE_WIDTH = 0.8
_CUT_LINE_WIDTH = 0.5
_CUT_LINE_COLOR = '0.6'
# between two cells of the markers page, the nicer gap is used when the cells fit with it
_CELL_GAPS = (1.5, 1.0)
_CELL_LABEL_SPACE = 0.5  # under a cell, for its id
_HEADER_HEIGHT = 2.5  # from the top of the markers page to the first row of cells
_FOOTER_HEIGHT = 2.0  # from the bottom of the markers page to the last row of cells, the ruler is in it
_FIGURE_EXTRA_PIXELS = 0.01  # so that a rounding error in the size of the figure never loses a pixel of the image


class SheetError(ValueError):
    """the sheet can not be made as configured, the message says why"""


@dataclass(frozen=True)
class Rect:
    """a rectangle on the page in cm, from the bottom left corner of the page, y up"""
    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height

    @property
    def centre(self) -> tuple[float, float]:
        return self.left + self.width / 2, self.bottom + self.height / 2


@dataclass(frozen=True)
class Ruler:
    """the line that is 10 cm long when the page was printed at 100%, from start to end, and the side of its text"""
    start: tuple[float, float]
    end: tuple[float, float]
    text_on_right: bool = False


@dataclass(frozen=True)
class BoardPage:
    """page 1, the size of the page and where the board is on it, all in cm"""
    width: float
    height: float
    board: Rect
    ruler: Ruler
    arrow_length: float  # of the x and the y arrow

    @property
    def origin(self) -> tuple[float, float]:
        """the corner of the board that is the origin of the board frame: the bottom right one"""
        return self.board.right, self.board.bottom


@dataclass(frozen=True)
class MarkerCell:
    """the cell of one block marker: the cut square of the size of the block and the marker in it"""
    marker_id: int
    cut: Rect
    marker: Rect


@dataclass(frozen=True)
class MarkerPage:
    """page 2, the size of the page and the cells on it, all in cm"""
    width: float
    height: float
    cells: tuple[MarkerCell, ...]
    ruler: Ruler


Page = BoardPage | MarkerPage


def _ruler(page_width: float, y: float) -> Ruler:
    """the ruler along the bottom of the page, its end is at the right margin and its text is left of it"""
    end_x = page_width - MIN_MARGIN
    return Ruler((end_x - RULER_LENGTH, y), (end_x, y))


def _ruler_at_left(y: float) -> Ruler:
    """the ruler along the bottom of the page, its start is at the left margin and its text is right of it"""
    return Ruler((MIN_MARGIN, y), (MIN_MARGIN + RULER_LENGTH, y), text_on_right=True)


def layout_board_page(board_config: BoardConfig) -> BoardPage:
    """
    the board centred on an A4 page in landscape, with the origin at its bottom right corner
    :raises SheetError: if the board does not fit with a margin of at least MIN_MARGIN all around
    """
    width = board_config.squares_x * board_config.square_length
    height = board_config.squares_y * board_config.square_length
    free_width = A4_LONG - 2 * MIN_MARGIN
    free_height = A4_SHORT - 2 * MIN_MARGIN
    if width > free_width or height > free_height:
        raise SheetError('the board is {:g} x {:g} cm, which does not fit on an A4 page in landscape ({:g} x {:g} cm) '
                         'with a margin of {:g} cm all around: at most {:g} x {:g} cm. Use fewer or smaller squares in '
                         'the board of the stacking configuration.'.format(
                             width, height, A4_LONG, A4_SHORT, MIN_MARGIN, free_width, free_height))
    board = Rect((A4_LONG - width) / 2, (A4_SHORT - height) / 2, width, height)
    # the origin mark and the x arrow are at the bottom right, so the ruler goes to the bottom left, in the margin under
    # the board halfway between the board and the edge of the paper
    arrow_length = min(ARROW_LENGTH, max(_ARROW_MIN_LENGTH, A4_LONG - board.right - ARROW_OFFSET - _ARROW_END_SPACE))
    return BoardPage(A4_LONG, A4_SHORT, board, _ruler_at_left(board.bottom / 2), arrow_length)


def layout_marker_page(blocks_config: BlockConfig) -> MarkerPage:
    """
    a cell for every id of the blocks on an A4 page in portrait, row by row from the top left, centred
    :raises SheetError: if the cells do not fit on one page, with MIN_MARGIN round them and a gap to cut them out
    """
    size = blocks_config.size
    ids = blocks_config.marker_ids
    width, height = A4_SHORT, A4_LONG
    free_width = width - 2 * MIN_MARGIN
    free_height = height - _HEADER_HEIGHT - _FOOTER_HEIGHT
    capacity = 0
    columns = rows = 0
    gap = _CELL_GAPS[-1]
    for gap in _CELL_GAPS:
        columns = int((free_width + gap) // (size + gap))
        rows = int((free_height - _CELL_LABEL_SPACE + gap) // (size + gap))
        capacity = columns * rows
        if capacity >= len(ids):
            break
    if capacity < len(ids):
        raise SheetError('{} markers of blocks of {:g} cm do not fit on one A4 page in portrait: with a gap of {:g} cm '
                         'between the cells there is room for {} by {}, which is {}'.format(
                             len(ids), size, gap, columns, rows, capacity))
    used_columns = min(columns, len(ids))
    grid_left = (width - (used_columns * size + (used_columns - 1) * gap)) / 2
    cells = []
    for index, marker_id in enumerate(ids):
        row, column = divmod(index, columns)
        left = grid_left + column * (size + gap)
        bottom = height - _HEADER_HEIGHT - size - row * (size + gap)
        cut = Rect(left, bottom, size, size)
        margin = (size - blocks_config.marker_length) / 2
        cells.append(MarkerCell(marker_id, cut, Rect(left + margin, bottom + margin, blocks_config.marker_length,
                                                     blocks_config.marker_length)))
    return MarkerPage(width, height, tuple(cells), _ruler(width, _FOOTER_HEIGHT / 2))


def _marker_black_modules(dictionary: cv2.aruco.Dictionary, marker_id: int) -> list[tuple[int, int, int]]:
    """
    the black modules of a marker with its border as runs (row, first column, number of columns), the row counted from
    the top of the marker
    """
    bits = dictionary.markerSize + 2  # the border is one module wide
    image = dictionary.generateImageMarker(marker_id, bits, borderBits=1)
    runs = []
    for row in range(bits):
        column = 0
        while column < bits:
            if image[row, column] == 0:
                start = column
                while column < bits and image[row, column] == 0:
                    column += 1
                runs.append((row, start, column - start))
            else:
                column += 1
    return runs


def marker_rectangles(dictionary: cv2.aruco.Dictionary, marker_id: int, marker: Rect
                      ) -> list[tuple[float, float, float, float]]:
    """the black parts of a marker that fills marker, as rectangles (left, bottom, width, height) in cm"""
    module = marker.width / (dictionary.markerSize + 2)
    return [(marker.left + column * module, marker.top - (row + 1) * module, count * module, module)
            for row, column, count in _marker_black_modules(dictionary, marker_id)]


def board_rectangles(board_config: BoardConfig, board: Rect) -> list[tuple[float, float, float, float]]:
    """
    the black parts of the board placed at board, as rectangles (left, bottom, width, height) in cm: the squares
    without a marker, which are black, and the black modules of the markers
    """
    charuco = make_board(board_config)
    dictionary = aruco_dictionary(board_config.dictionary)
    square = board_config.square_length
    with_marker = set()
    rectangles: list[tuple[float, float, float, float]] = []
    for corners, marker_id in zip(charuco.getObjPoints(), np.asarray(charuco.getIds()).ravel()):
        points = np.asarray(corners)[:, :2]
        # OpenCV's board has its origin at the top left and y down, so the first row of squares is the top one
        left, top = points.min(axis=0)
        centre_x, centre_y = points.mean(axis=0)
        with_marker.add((int(centre_x // square), int(centre_y // square)))
        side = float(points[:, 0].max() - left)
        placed = Rect(board.left + float(left), board.top - float(top) - side, side, side)
        rectangles.extend(marker_rectangles(dictionary, int(marker_id), placed))
    for row in range(board_config.squares_y):
        for column in range(board_config.squares_x):
            if (column, row) not in with_marker:
                rectangles.append((board.left + column * square, board.top - (row + 1) * square, square, square))
    return rectangles


def _black_path(rectangles: Sequence[tuple[float, float, float, float]]) -> MplPath:
    """all the rectangles as one path, so that neighbours are filled together and no hairline shows between them"""
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for left, bottom, width, height in rectangles:
        # every rectangle is counter clockwise, so the fill of overlapping ones never cancels out
        vertices.extend([(left, bottom), (left + width, bottom), (left + width, bottom + height),
                         (left, bottom + height), (left, bottom)])
        codes.extend(int(code) for code in (MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
                                            MplPath.CLOSEPOLY))
    return MplPath(vertices, codes)


def build_figure(page: Page, config: StackingConfig, pixels_per_cm: float | None = None) -> Figure:
    """
    the figure of one page in cm: one unit of its axes is 1 cm on the paper. When pixels_per_cm is given, the figure
    has page size * pixels_per_cm pixels at dpi pixels_per_cm * 2.54, else it is the size of the page in inches.
    """
    if pixels_per_cm is None:
        figure = Figure(figsize=(page.width / CM_PER_INCH, page.height / CM_PER_INCH))
    else:
        width_pixels = round(page.width * pixels_per_cm)
        height_pixels = round(page.height * pixels_per_cm)
        dpi = pixels_per_cm * CM_PER_INCH
        figure = Figure(figsize=((width_pixels + _FIGURE_EXTRA_PIXELS) / dpi,
                                 (height_pixels + _FIGURE_EXTRA_PIXELS) / dpi), dpi=dpi)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_xlim(0.0, page.width)
    axes.set_ylim(0.0, page.height)
    axes.axis('off')
    if isinstance(page, BoardPage):
        _draw_board_page(axes, page, config)
    else:
        _draw_marker_page(axes, page, config)
    _draw_ruler(axes, page.ruler)
    return figure


def _draw_black(axes: Axes,rectangles: Sequence[tuple[float, float, float, float]]) -> None:
    axes.add_patch(PathPatch(_black_path(rectangles), facecolor='black', edgecolor='none', linewidth=0))


def _draw_ruler(axes: Axes, ruler: Ruler) -> None:
    (start_x, y), (end_x, _) = ruler.start, ruler.end
    axes.plot([start_x, end_x], [y, y], color='black', linewidth=_RULER_LINE_WIDTH, solid_capstyle='butt')
    for x in (start_x, end_x):
        axes.plot([x, x], [y - _RULER_TICK, y + _RULER_TICK], color='black', linewidth=_RULER_LINE_WIDTH,
                  solid_capstyle='butt')
    if ruler.text_on_right:
        axes.text(end_x + 0.3, y, RULER_TEXT, fontsize=_INFO_FONT_SIZE, ha='left', va='center')
    else:
        axes.text(start_x - 0.3, y, RULER_TEXT, fontsize=_INFO_FONT_SIZE, ha='right', va='center')


def _draw_arrow(axes: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, color='black',
                                   linewidth=_LINE_WIDTH, shrinkA=0, shrinkB=0))


def _draw_board_page(axes: Axes, page: BoardPage, config: StackingConfig) -> None:
    board_config = config.board
    board = page.board
    _draw_black(axes, board_rectangles(board_config, board))
    right, bottom = page.origin
    # the axes of the board frame, in the margin and parallel to the edges they belong to, from the origin mark: the x
    # arrow runs along the bottom edge and away from the board, the y arrow up along the right edge
    mark_x, mark_y = right + ARROW_OFFSET, bottom - ARROW_OFFSET
    length = page.arrow_length
    axes.plot([mark_x, right], [mark_y, bottom], color='black', linewidth=_LINE_WIDTH)
    axes.plot([mark_x], [mark_y], marker='o', color='black', markersize=5)
    axes.text(mark_x, mark_y - 0.15, 'origin', fontsize=_INFO_FONT_SIZE + 1, ha='center', va='top')
    _draw_arrow(axes, (mark_x, mark_y), (mark_x + length, mark_y))
    _draw_arrow(axes, (mark_x, mark_y), (mark_x, mark_y + length))
    # the x is over the shaft near the tip, so that it stays on the page when the margin is narrow
    axes.text(mark_x + length - 0.3, mark_y + 0.12, 'x', fontsize=_LABEL_FONT_SIZE, ha='center', va='bottom')
    axes.text(mark_x, mark_y + length + 0.15, 'y', fontsize=_LABEL_FONT_SIZE, ha='center', va='bottom')
    description = ('Charuco board {}: {} x {} squares of {:g} cm, markers of {:g} cm, so {:g} x {:g} cm. Bottom right '
                   'is the origin, x goes right (away from the board) and y goes up.').format(
                       board_config.dictionary, board_config.squares_x, board_config.squares_y,
                       board_config.square_length, board_config.marker_length, board.width, board.height)
    note = ('Tape it flat on the table, then measure it: the yaml wants the x, y of the origin corner and the '
            'rotation of the x arrow.')
    axes.text(board.left, board.top + 0.55, description, fontsize=_INFO_FONT_SIZE, ha='left', va='bottom')
    axes.text(board.left, board.top + 0.2, note, fontsize=_INFO_FONT_SIZE, ha='left', va='bottom')


def _draw_marker_page(axes: Axes, page: MarkerPage, config: StackingConfig) -> None:
    blocks = config.blocks
    dictionary = aruco_dictionary(blocks.dictionary)
    rectangles = []
    for cell in page.cells:
        rectangles.extend(marker_rectangles(dictionary, cell.marker_id, cell.marker))
        axes.add_patch(Rectangle((cell.cut.left, cell.cut.bottom), cell.cut.width, cell.cut.height, fill=False,
                                 edgecolor=_CUT_LINE_COLOR, linewidth=_CUT_LINE_WIDTH))
        centre_x, _ = cell.cut.centre
        axes.text(centre_x, cell.cut.bottom - 0.1, 'id {}'.format(cell.marker_id), fontsize=_INFO_FONT_SIZE,
                  ha='center', va='top')
    _draw_black(axes, rectangles)
    header = ('Block markers {}: marker of {:g} cm centred in a block of {:g} cm, cut along the grey lines.'
              .format(blocks.dictionary, blocks.marker_length, blocks.size))
    axes.text(MIN_MARGIN, page.height - MIN_MARGIN, header, fontsize=_INFO_FONT_SIZE, ha='left', va='top')


def render_page(page: Page, config: StackingConfig, pixels_per_cm: int) -> np.ndarray:
    """
    the page as a gray image of page size * pixels_per_cm pixels, row 0 on top: the point (x, y) cm of the page is at
    the pixel position (x * pixels_per_cm, (height - y) * pixels_per_cm), the corner of the pixel at integer values
    """
    figure = build_figure(page, config, pixels_per_cm)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    expected = (round(page.height * pixels_per_cm), round(page.width * pixels_per_cm))
    if rgba.shape[:2] != expected:
        raise SheetError('the image is {} pixels but should be {}'.format(rgba.shape[:2], expected))
    gray: np.ndarray = rgba[:, :, :3].mean(axis=2).astype(np.uint8)
    return gray


def make_layouts(config: StackingConfig) -> tuple[BoardPage, MarkerPage]:
    """both pages, before anything is written
    :raises SheetError: if one of them does not fit"""
    return layout_board_page(config.board), layout_marker_page(config.blocks)


def write_pdf(config: StackingConfig, output_path: str | os.PathLike[str]) -> None:
    """
    write the two pages of the sheet to a pdf file
    :raises SheetError: if the board or the markers do not fit on their page
    """
    pages = make_layouts(config)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # the fonts are embedded as text, and no creation date so that the same configuration gives the same file
    metadata: dict[str, object] = {'Title': 'Block stacking markers', 'Creator': 'block_sheet.py', 'CreationDate': None}
    with matplotlib.rc_context({'pdf.fonttype': 42}):
        with PdfPages(output_path, metadata=metadata) as pdf:
            for page in pages:
                pdf.savefig(build_figure(page, config))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Make the A4 pdf with the charuco board and the markers of the blocks '
                                                 'at true size. Print it at 100%, without fit to page.')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='the pdf to write (default: %(default)s)')
    parser.add_argument('--config', default=None, help='the stacking configuration (default: the one of the project)')
    arguments = parser.parse_args(argv)
    try:
        config = load_stacking_config() if arguments.config is None else load_stacking_config(arguments.config)
        write_pdf(config, arguments.output)
    except (StackingConfigError, SheetError) as e:
        raise SystemExit('can not make the sheet: {}'.format(e)) from e
    print('wrote {}: print it at 100% and measure the 10 cm line'.format(arguments.output))


if __name__ == '__main__':
    main()
