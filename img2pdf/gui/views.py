import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QImageReader, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from ..model import Page

KIND_COLORS = {
    "paragraph": QColor(30, 120, 230),
    "heading": QColor(240, 130, 0),
    "cell": QColor(20, 160, 60),
    "figure": QColor(220, 30, 60),
    "table": QColor(150, 40, 190),
}


def numpy_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def load_qimage(path: str, max_side: int | None = None) -> QImage:
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    if max_side:
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > max_side:
            reader.setScaledSize(size.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio))
    return reader.read()


class ZoomView(QGraphicsView):
    """Graphics view that fits its content until the user zooms (Ctrl + wheel)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(90, 94, 102)))
        self._fit = True
        self._pix: QGraphicsPixmapItem | None = None

    def set_image(self, img: QImage | None, keep_zoom: bool = False):
        self.scene().clear()
        self._pix = None
        if img is None or img.isNull():
            return
        self._pix = self.scene().addPixmap(QPixmap.fromImage(img))
        self._pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().setSceneRect(QRectF(img.rect()).adjusted(-20, -20, 20, 20))
        if not keep_zoom:
            self._fit = True
        if self._fit:
            self.fit()

    def fit(self):
        if self._pix is not None:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float):
        self._fit = False
        self.scale(factor, factor)

    def reset_zoom(self):
        self._fit = True
        self.fit()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.15 if e.angleDelta().y() > 0 else 1 / 1.15)
        else:
            super().wheelEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()


class SourceView(ZoomView):
    """Processed page image with the detected regions drawn on top."""

    elementClicked = pyqtSignal(object)  # ("block", i) | ("figure", i)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: list[tuple[QRectF, tuple]] = []
        self._items: dict[tuple, tuple[QGraphicsRectItem, QPen, QBrush]] = {}
        self.show_regions = True

    def show_page(self, page: Page | None, original: QImage | None, keep_zoom=False):
        self._regions, self._items = [], {}
        if page is None:
            self.set_image(original, keep_zoom)
            return
        self.set_image(numpy_to_qimage(page.image), keep_zoom)
        if not self.show_regions:
            return
        for t in page.tables:
            self._add(t, "table", None)
        for i, f in enumerate(page.figures):
            self._add(f.bbox, "figure", ("figure", i))
        for i, b in enumerate(page.blocks):
            self._add(b.bbox, b.kind, ("block", i))

    def _add(self, box, kind, key):
        x0, y0, x1, y1 = box
        rect = QRectF(x0 - 3, y0 - 3, x1 - x0 + 6, y1 - y0 + 6)
        color = KIND_COLORS[kind]
        pen = QPen(color, 3 if kind != "table" else 5)
        fill = QColor(color)
        fill.setAlpha(28 if kind != "table" else 0)
        brush = QBrush(fill)
        item = self.scene().addRect(rect, pen, brush)
        if key is not None:
            self._regions.append((rect, key))
            self._items[key] = (item, pen, brush)

    def highlight(self, key):
        for k, (item, pen, brush) in self._items.items():
            if k == key:
                item.setBrush(QBrush(QColor(255, 220, 0, 90)))
                item.setPen(QPen(QColor(255, 170, 0), 6))
                self.ensureVisible(item, 40, 40)
            else:
                item.setPen(pen)
                item.setBrush(brush)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._regions:
            p = self.mapToScene(e.position().toPoint())
            hits = [(r.width() * r.height(), key) for r, key in self._regions if r.contains(p)]
            if hits:
                self.elementClicked.emit(min(hits)[1])
        super().mousePressEvent(e)


class PreviewView(ZoomView):
    """Rendered PDF page; clicks are mapped back to page-pixel coordinates."""

    pointClicked = pyqtSignal(QPointF)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pix is not None:
            self.pointClicked.emit(self.mapToScene(e.position().toPoint()))
        super().mousePressEvent(e)
