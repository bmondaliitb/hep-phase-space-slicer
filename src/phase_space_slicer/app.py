from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .data import FeatureData, load_feature_files


class CorrelationPanel(QtWidgets.QFrame):
    variables_changed = QtCore.Signal()
    slice_requested = QtCore.Signal(object, int, int)
    remove_requested = QtCore.Signal(object)
    maximize_requested = QtCore.Signal(object)
    minimize_requested = QtCore.Signal(object)

    def __init__(
        self,
        data: FeatureData,
        x_index: int,
        y_index: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.minimized = False
        color_map = pg.colormap.get("viridis")
        self.plot = pg.PlotWidget()
        self.image = pg.ImageItem()
        self.image.setLookupTable(color_map.getLookupTable(0.0, 1.0, 256))
        self.plot.addItem(self.image)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setBackground("w")
        self.slice_region_item = QtWidgets.QGraphicsPolygonItem()
        self.slice_region_item.setPen(pg.mkPen("#f97316", width=2))
        self.slice_region_item.setBrush(pg.mkBrush(249, 115, 22, 55))
        self.slice_region_item.setZValue(20)
        self.plot.addItem(self.slice_region_item)

        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        for combo in (self.x_combo, self.y_combo):
            combo.addItems(data.names)
        self.x_combo.setCurrentIndex(x_index)
        self.y_combo.setCurrentIndex(y_index)

        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(20, 600)
        self.bins_spin.setSingleStep(10)
        self.bins_spin.setValue(160)
        self.bins_spin.valueChanged.connect(self.variables_changed)
        self.slice_angle_spin = QtWidgets.QDoubleSpinBox()
        self.slice_angle_spin.setRange(-180.0, 180.0)
        self.slice_angle_spin.setDecimals(1)
        self.slice_angle_spin.setSingleStep(5.0)
        self.slice_angle_spin.setValue(0.0)
        self.slice_angle_spin.valueChanged.connect(self._update_slice_region)
        self.slice_x1_spin = self._make_range_spin()
        self.slice_y1_spin = self._make_range_spin()
        self.slice_x2_spin = self._make_range_spin()
        self.slice_y2_spin = self._make_range_spin()
        for spin in (
            self.slice_x1_spin,
            self.slice_y1_spin,
            self.slice_x2_spin,
            self.slice_y2_spin,
        ):
            spin.valueChanged.connect(self._update_slice_region)

        self.x_min_spin = self._make_range_spin()
        self.x_max_spin = self._make_range_spin()
        self.y_min_spin = self._make_range_spin()
        self.y_max_spin = self._make_range_spin()
        for spin in (self.x_min_spin, self.x_max_spin, self.y_min_spin, self.y_max_spin):
            spin.valueChanged.connect(self.variables_changed)

        self.bin_count_label = QtWidgets.QLabel()
        self.bin_count_label.setMinimumWidth(130)

        remove_button = QtWidgets.QToolButton()
        remove_button.setText("×")
        remove_button.setToolTip("Remove plot")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.minimize_button = QtWidgets.QToolButton()
        self.minimize_button.setText("_")
        self.minimize_button.setToolTip("Minimize plot")
        self.minimize_button.clicked.connect(lambda: self.minimize_requested.emit(self))
        self.maximize_button = QtWidgets.QToolButton()
        self.maximize_button.setText("□")
        self.maximize_button.setToolTip("Maximize plot")
        self.maximize_button.clicked.connect(lambda: self.maximize_requested.emit(self))

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("X"))
        header.addWidget(self.x_combo, 1)
        header.addWidget(QtWidgets.QLabel("Y"))
        header.addWidget(self.y_combo, 1)
        header.addWidget(self.minimize_button)
        header.addWidget(self.maximize_button)
        header.addWidget(remove_button)

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("X min"), 0, 0)
        controls.addWidget(self.x_min_spin, 0, 1)
        controls.addWidget(QtWidgets.QLabel("X max"), 0, 2)
        controls.addWidget(self.x_max_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Y min"), 1, 0)
        controls.addWidget(self.y_min_spin, 1, 1)
        controls.addWidget(QtWidgets.QLabel("Y max"), 1, 2)
        controls.addWidget(self.y_max_spin, 1, 3)
        controls.addWidget(QtWidgets.QLabel("Bins"), 2, 0)
        controls.addWidget(self.bins_spin, 2, 1)
        auto_range_button = QtWidgets.QPushButton("Auto Range")
        auto_range_button.clicked.connect(lambda: self.reset_ranges())
        controls.addWidget(auto_range_button, 2, 2, 1, 2)
        controls.addWidget(QtWidgets.QLabel("x1"), 3, 0)
        controls.addWidget(self.slice_x1_spin, 3, 1)
        controls.addWidget(QtWidgets.QLabel("y1"), 3, 2)
        controls.addWidget(self.slice_y1_spin, 3, 3)
        controls.addWidget(QtWidgets.QLabel("x2"), 4, 0)
        controls.addWidget(self.slice_x2_spin, 4, 1)
        controls.addWidget(QtWidgets.QLabel("y2"), 4, 2)
        controls.addWidget(self.slice_y2_spin, 4, 3)
        controls.addWidget(QtWidgets.QLabel("Angle"), 5, 0)
        controls.addWidget(self.slice_angle_spin, 5, 1)
        apply_slice_button = QtWidgets.QPushButton("Apply Slice")
        apply_slice_button.clicked.connect(self._emit_slice)
        controls.addWidget(apply_slice_button, 5, 2, 1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        self.controls_layout = controls
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addLayout(controls)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.bin_count_label)

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMinimumSize(460, 380)
        self.reset_ranges(emit=False)
        for combo in (self.x_combo, self.y_combo):
            combo.currentIndexChanged.connect(self._variable_changed)
        self._update_labels()

    @property
    def x_index(self) -> int:
        return self.x_combo.currentIndex()

    @property
    def y_index(self) -> int:
        return self.y_combo.currentIndex()

    @property
    def bins(self) -> int:
        return self.bins_spin.value()

    @property
    def x_range(self) -> tuple[float, float]:
        return self.x_min_spin.value(), self.x_max_spin.value()

    @property
    def y_range(self) -> tuple[float, float]:
        return self.y_min_spin.value(), self.y_max_spin.value()

    def set_selection_enabled(self, enabled: bool) -> None:
        del enabled
        self.slice_region_item.setVisible(True)

    def set_minimized(self, minimized: bool) -> None:
        self.minimized = minimized
        self.plot.setVisible(not minimized)
        self.bin_count_label.setVisible(not minimized)
        for index in range(self.controls_layout.count()):
            item = self.controls_layout.itemAt(index)
            if item.widget():
                item.widget().setVisible(not minimized)
        self.minimize_button.setText("+" if minimized else "_")
        self.minimize_button.setToolTip("Restore plot" if minimized else "Minimize plot")

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        if x_max <= x_min or y_max <= y_min:
            self.image.clear()
            self.bin_count_label.setText("Bin count: invalid range")
            self._update_labels()
            return

        in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x = x[in_range]
        y = y[in_range]
        if x.size == 0:
            self.image.clear()
            self.bin_count_label.setText("Bin count: empty")
            self._update_labels()
            return

        histogram, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=self.bins,
            range=((x_min, x_max), (y_min, y_max)),
        )
        max_count = float(histogram.max())
        self.image.setImage(histogram, autoLevels=False, levels=(0.0, max(max_count, 1.0)))
        self.image.setRect(
            QtCore.QRectF(
                float(x_edges[0]),
                float(y_edges[0]),
                float(x_edges[-1] - x_edges[0]),
                float(y_edges[-1] - y_edges[0]),
            )
        )
        self.bin_count_label.setText(f"Bin count color: 0 .. {int(max_count):,}")
        self.plot.setXRange(x_min, x_max, padding=0.0)
        self.plot.setYRange(y_min, y_max, padding=0.0)
        self._update_labels()

    def refresh_variable_names(self, names: list[str]) -> None:
        x_index = self.x_index
        y_index = self.y_index
        for combo, index in ((self.x_combo, x_index), (self.y_combo, y_index)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._update_labels()

    def apply_settings(
        self,
        bins: int | None = None,
        x_range: list[float] | tuple[float, float] | None = None,
        y_range: list[float] | tuple[float, float] | None = None,
        slice_angle: float | None = None,
        slice_points: list[list[float]] | tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self._block_range_signals(True)
        if bins is not None:
            self.bins_spin.setValue(int(bins))
        if slice_angle is not None:
            self.slice_angle_spin.setValue(float(slice_angle))
        if x_range is not None and len(x_range) == 2:
            self.x_min_spin.setValue(float(x_range[0]))
            self.x_max_spin.setValue(float(x_range[1]))
        if y_range is not None and len(y_range) == 2:
            self.y_min_spin.setValue(float(y_range[0]))
            self.y_max_spin.setValue(float(y_range[1]))
        self._block_range_signals(False)
        if slice_points is not None and len(slice_points) == 2:
            self._set_slice_points(
                (float(slice_points[0][0]), float(slice_points[0][1])),
                (float(slice_points[1][0]), float(slice_points[1][1])),
            )
        else:
            self._update_slice_region()

    def reset_ranges(self, emit: bool = True) -> None:
        x_range = self._column_range(self.x_index)
        y_range = self._column_range(self.y_index)
        self._block_range_signals(True)
        self.x_min_spin.setValue(x_range[0])
        self.x_max_spin.setValue(x_range[1])
        self.y_min_spin.setValue(y_range[0])
        self.y_max_spin.setValue(y_range[1])
        self._reset_slice_region(x_range, y_range)
        self._block_range_signals(False)
        if emit:
            self.variables_changed.emit()

    def _emit_slice(self) -> None:
        self.slice_requested.emit(self.slice_box(), self.x_index, self.y_index)

    def _update_labels(self) -> None:
        x_name = self.data.names[self.x_index]
        y_name = self.data.names[self.y_index]
        self.plot.setLabel("bottom", x_name)
        self.plot.setLabel("left", y_name)

    def _variable_changed(self) -> None:
        self.reset_ranges(emit=False)
        self.variables_changed.emit()

    def _make_range_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1.0e300, 1.0e300)
        spin.setDecimals(6)
        spin.setKeyboardTracking(False)
        return spin

    def _block_range_signals(self, blocked: bool) -> None:
        for widget in (
            self.bins_spin,
            self.slice_angle_spin,
            self.x_min_spin,
            self.x_max_spin,
            self.y_min_spin,
            self.y_max_spin,
        ):
            widget.blockSignals(blocked)

    def _column_range(self, index: int) -> tuple[float, float]:
        values = self.data.values[:, index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            delta = abs(low) * 0.01 or 1.0
            return low - delta, high + delta
        return low, high

    def _reset_slice_region(
        self, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> None:
        point_a = (
            x_range[0] + (x_range[1] - x_range[0]) / 3.0,
            y_range[0] + (y_range[1] - y_range[0]) / 3.0,
        )
        point_b = (
            x_range[0] + 2.0 * (x_range[1] - x_range[0]) / 3.0,
            y_range[0] + 2.0 * (y_range[1] - y_range[0]) / 3.0,
        )
        self._set_slice_points(point_a, point_b)

    def _set_slice_points(
        self, point_a: tuple[float, float], point_b: tuple[float, float]
    ) -> None:
        widgets = (
            (self.slice_x1_spin, point_a[0]),
            (self.slice_y1_spin, point_a[1]),
            (self.slice_x2_spin, point_b[0]),
            (self.slice_y2_spin, point_b[1]),
        )
        for widget, value in widgets:
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self._update_slice_region()

    def _update_slice_region(self) -> None:
        polygon = slice_box_polygon(
            (
                self.slice_x1_spin.value(),
                self.slice_y1_spin.value(),
            ),
            (
                self.slice_x2_spin.value(),
                self.slice_y2_spin.value(),
            ),
            self.slice_angle_spin.value(),
        )
        self.slice_region_item.setPolygon(
            QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in polygon])
        )

    def slice_points(self) -> list[list[float]]:
        return [
            [float(self.slice_x1_spin.value()), float(self.slice_y1_spin.value())],
            [float(self.slice_x2_spin.value()), float(self.slice_y2_spin.value())],
        ]

    def slice_box(self) -> dict[str, float]:
        return {
            "x1": float(self.slice_x1_spin.value()),
            "y1": float(self.slice_y1_spin.value()),
            "x2": float(self.slice_x2_spin.value()),
            "y2": float(self.slice_y2_spin.value()),
            "angle": float(self.slice_angle_spin.value()),
        }


class DistributionPanel(QtWidgets.QFrame):
    variables_changed = QtCore.Signal()
    slice_requested = QtCore.Signal(object, int)
    remove_requested = QtCore.Signal(object)
    maximize_requested = QtCore.Signal(object)
    minimize_requested = QtCore.Signal(object)

    def __init__(self, data: FeatureData, x_index: int, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.minimized = False
        self.plot = pg.PlotWidget()
        self.histogram = pg.PlotCurveItem(
            pen=pg.mkPen("#2563eb", width=1.5),
            brush=pg.mkBrush(37, 99, 235, 90),
            fillLevel=0,
        )
        self.plot.addItem(self.histogram)
        self.slice_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            orientation="vertical",
            movable=True,
            brush=pg.mkBrush(249, 115, 22, 45),
            pen=pg.mkPen("#f97316", width=2),
        )
        self.plot.addItem(self.slice_region, ignoreBounds=True)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setBackground("w")

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.addItems(data.names)
        self.x_combo.setCurrentIndex(x_index)

        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(20, 2000)
        self.bins_spin.setSingleStep(10)
        self.bins_spin.setValue(160)
        self.bins_spin.valueChanged.connect(self.variables_changed)

        self.x_min_spin = self._make_range_spin()
        self.x_max_spin = self._make_range_spin()
        for spin in (self.x_min_spin, self.x_max_spin):
            spin.valueChanged.connect(self.variables_changed)

        self.count_label = QtWidgets.QLabel()
        self.count_label.setMinimumWidth(130)

        remove_button = QtWidgets.QToolButton()
        remove_button.setText("×")
        remove_button.setToolTip("Remove plot")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.minimize_button = QtWidgets.QToolButton()
        self.minimize_button.setText("_")
        self.minimize_button.setToolTip("Minimize plot")
        self.minimize_button.clicked.connect(lambda: self.minimize_requested.emit(self))
        self.maximize_button = QtWidgets.QToolButton()
        self.maximize_button.setText("□")
        self.maximize_button.setToolTip("Maximize plot")
        self.maximize_button.clicked.connect(lambda: self.maximize_requested.emit(self))

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("1D"))
        header.addWidget(self.x_combo, 1)
        header.addWidget(self.minimize_button)
        header.addWidget(self.maximize_button)
        header.addWidget(remove_button)

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("X min"), 0, 0)
        controls.addWidget(self.x_min_spin, 0, 1)
        controls.addWidget(QtWidgets.QLabel("X max"), 0, 2)
        controls.addWidget(self.x_max_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Bins"), 1, 0)
        controls.addWidget(self.bins_spin, 1, 1)
        auto_range_button = QtWidgets.QPushButton("Auto Range")
        auto_range_button.clicked.connect(lambda: self.reset_ranges())
        controls.addWidget(auto_range_button, 1, 2, 1, 2)
        apply_slice_button = QtWidgets.QPushButton("Apply Slice")
        apply_slice_button.clicked.connect(self._emit_slice)
        controls.addWidget(apply_slice_button, 2, 2, 1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        self.controls_layout = controls
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addLayout(controls)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.count_label)

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMinimumSize(460, 300)
        self.reset_ranges(emit=False)
        self.x_combo.currentIndexChanged.connect(self._variable_changed)
        self._update_labels()

    @property
    def x_index(self) -> int:
        return self.x_combo.currentIndex()

    @property
    def bins(self) -> int:
        return self.bins_spin.value()

    @property
    def x_range(self) -> tuple[float, float]:
        return self.x_min_spin.value(), self.x_max_spin.value()

    def set_selection_enabled(self, enabled: bool) -> None:
        self.slice_region.setVisible(enabled)

    def set_minimized(self, minimized: bool) -> None:
        self.minimized = minimized
        self.plot.setVisible(not minimized)
        self.count_label.setVisible(not minimized)
        for index in range(self.controls_layout.count()):
            item = self.controls_layout.itemAt(index)
            if item.widget():
                item.widget().setVisible(not minimized)
        self.minimize_button.setText("+" if minimized else "_")
        self.minimize_button.setToolTip("Restore plot" if minimized else "Minimize plot")

    def set_data(self, x: np.ndarray) -> None:
        x_min, x_max = self.x_range
        if x_max <= x_min:
            self.histogram.setData([], [])
            self.count_label.setText("Bin count: invalid range")
            self._update_labels()
            return

        in_range = (x >= x_min) & (x <= x_max)
        x = x[in_range]
        if x.size == 0:
            self.histogram.setData([], [])
            self.count_label.setText("Bin count: empty")
            self._update_labels()
            return

        counts, edges = np.histogram(x, bins=self.bins, range=(x_min, x_max))
        step_x = np.repeat(edges, 2)[1:-1]
        step_y = np.repeat(counts, 2)
        self.histogram.setData(step_x, step_y)
        max_count = int(counts.max()) if counts.size else 0
        self.count_label.setText(f"Bin count: 0 .. {max_count:,}")
        self.plot.setXRange(x_min, x_max, padding=0.0)
        self.plot.setYRange(0, max(max_count, 1), padding=0.05)
        self._update_labels()

    def refresh_variable_names(self, names: list[str]) -> None:
        x_index = self.x_index
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItems(names)
        self.x_combo.setCurrentIndex(x_index)
        self.x_combo.blockSignals(False)
        self._update_labels()

    def apply_settings(
        self,
        bins: int | None = None,
        x_range: list[float] | tuple[float, float] | None = None,
        y_range: list[float] | tuple[float, float] | None = None,
    ) -> None:
        del y_range
        self._block_range_signals(True)
        if bins is not None:
            self.bins_spin.setValue(int(bins))
        if x_range is not None and len(x_range) == 2:
            self.x_min_spin.setValue(float(x_range[0]))
            self.x_max_spin.setValue(float(x_range[1]))
        self._block_range_signals(False)

    def reset_ranges(self, emit: bool = True) -> None:
        x_range = self._column_range(self.x_index)
        self._block_range_signals(True)
        self.x_min_spin.setValue(x_range[0])
        self.x_max_spin.setValue(x_range[1])
        self.slice_region.setRegion(
            (x_range[0] + (x_range[1] - x_range[0]) / 3.0, x_range[0] + 2.0 * (x_range[1] - x_range[0]) / 3.0)
        )
        self._block_range_signals(False)
        if emit:
            self.variables_changed.emit()

    def _emit_slice(self) -> None:
        self.slice_requested.emit(np.array(self.slice_region.getRegion(), dtype=float), self.x_index)

    def _update_labels(self) -> None:
        x_name = self.data.names[self.x_index]
        self.plot.setLabel("bottom", x_name)
        self.plot.setLabel("left", "count")

    def _variable_changed(self) -> None:
        self.reset_ranges(emit=False)
        self.variables_changed.emit()

    def _make_range_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1.0e300, 1.0e300)
        spin.setDecimals(6)
        spin.setKeyboardTracking(False)
        return spin

    def _block_range_signals(self, blocked: bool) -> None:
        for widget in (self.bins_spin, self.x_min_spin, self.x_max_spin):
            widget.blockSignals(blocked)

    def _column_range(self, index: int) -> tuple[float, float]:
        values = self.data.values[:, index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            delta = abs(low) * 0.01 or 1.0
            return low - delta, high + delta
        return low, high


class RenameColumnsDialog(QtWidgets.QDialog):
    def __init__(self, names: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Columns")
        self.resize(520, 680)

        self.table = QtWidgets.QTableWidget(len(names), 2)
        self.table.setHorizontalHeaderLabels(["Column", "Label"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        for index, name in enumerate(names):
            column_item = QtWidgets.QTableWidgetItem(f"{index:02d}")
            column_item.setFlags(column_item.flags() & ~QtCore.Qt.ItemIsEditable)
            label_item = QtWidgets.QTableWidgetItem(name)
            self.table.setItem(index, 0, column_item)
            self.table.setItem(index, 1, label_item)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply_requested)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(buttons)

    def names(self) -> list[str]:
        result = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            name = item.text().strip() if item is not None else ""
            result.append(name or f"feature_{row:02d}")
        return result

    def apply_requested(self) -> None:
        parent = self.parent()
        if isinstance(parent, MainWindow):
            parent.rename_columns(self.names())


class SpreadsheetWidget(QtWidgets.QWidget):
    labels_changed = QtCore.Signal(object)
    column_added = QtCore.Signal(str, str, object)

    def __init__(self, data: FeatureData, parent=None) -> None:
        super().__init__(parent)
        self.data = data

        self.label_table = QtWidgets.QTableWidget()
        self.label_table.setColumnCount(2)
        self.label_table.setHorizontalHeaderLabels(["Column", "Label"])
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        apply_labels_button = QtWidgets.QPushButton("Apply Labels")
        apply_labels_button.clicked.connect(self.apply_labels)

        self.new_column_name = QtWidgets.QLineEdit()
        self.new_column_name.setPlaceholderText("new_column")
        self.expression_edit = QtWidgets.QLineEdit()
        self.expression_edit.setPlaceholderText("np.log(col0) + sqrt(col1)")
        add_column_button = QtWidgets.QPushButton("Add Column")
        add_column_button.clicked.connect(self.add_column_from_expression)
        self.expression_status = QtWidgets.QLabel()
        self.expression_status.setWordWrap(True)

        expression_layout = QtWidgets.QGridLayout()
        expression_layout.addWidget(QtWidgets.QLabel("New column"), 0, 0)
        expression_layout.addWidget(self.new_column_name, 0, 1)
        expression_layout.addWidget(QtWidgets.QLabel("Expression"), 1, 0)
        expression_layout.addWidget(self.expression_edit, 1, 1)
        expression_layout.addWidget(add_column_button, 2, 1)
        expression_layout.addWidget(self.expression_status, 3, 0, 1, 2)

        self.preview_rows = QtWidgets.QSpinBox()
        self.preview_rows.setRange(10, 5000)
        self.preview_rows.setValue(500)
        self.preview_rows.setSingleStep(100)
        self.preview_rows.valueChanged.connect(self.refresh_preview)

        self.data_table = QtWidgets.QTableWidget()
        self.data_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("Column Labels"))
        left_layout.addWidget(self.label_table)
        left_layout.addWidget(apply_labels_button)
        left_layout.addSpacing(12)
        left_layout.addLayout(expression_layout)
        left.setMaximumWidth(420)

        preview_controls = QtWidgets.QHBoxLayout()
        preview_controls.addWidget(QtWidgets.QLabel("Preview rows"))
        preview_controls.addWidget(self.preview_rows)
        preview_controls.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addLayout(preview_controls)
        right_layout.addWidget(self.data_table)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

        self.refresh_all(data)

    def refresh_all(self, data: FeatureData | None = None) -> None:
        if data is not None:
            self.data = data
        self.refresh_labels()
        self.refresh_preview()

    def refresh_labels(self) -> None:
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(len(self.data.names))
        for index, name in enumerate(self.data.names):
            index_item = QtWidgets.QTableWidgetItem(str(index))
            index_item.setFlags(index_item.flags() & ~QtCore.Qt.ItemIsEditable)
            label_item = QtWidgets.QTableWidgetItem(name)
            self.label_table.setItem(index, 0, index_item)
            self.label_table.setItem(index, 1, label_item)
        self.label_table.blockSignals(False)

    def refresh_preview(self) -> None:
        rows = min(self.preview_rows.value(), self.data.values.shape[0])
        columns = self.data.values.shape[1]
        self.data_table.setRowCount(rows)
        self.data_table.setColumnCount(columns)
        self.data_table.setHorizontalHeaderLabels(self.data.names)
        preview = self.data.values[:rows]
        for row in range(rows):
            for column in range(columns):
                self.data_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(f"{preview[row, column]:.6g}"),
                )
        self.data_table.resizeColumnsToContents()

    def apply_labels(self) -> None:
        names = []
        for row in range(self.label_table.rowCount()):
            item = self.label_table.item(row, 1)
            name = item.text().strip() if item is not None else ""
            names.append(name or f"feature_{row:02d}")
        self.labels_changed.emit(names)

    def add_column_from_expression(self) -> None:
        name = self.new_column_name.text().strip()
        expression = self.expression_edit.text().strip()
        if not name or not expression:
            self.expression_status.setText("Column name and expression are required.")
            return
        try:
            values = evaluate_column_expression(expression, self.data)
        except Exception as error:  # noqa: BLE001 - show expression error to user
            self.expression_status.setText(f"Expression error: {error}")
            return
        self.expression_status.setText(f"Added {name}")
        self.column_added.emit(name, expression, values)


def evaluate_column_expression(expression: str, data: FeatureData) -> np.ndarray:
    namespace: dict[str, object] = {"np": np}
    for name in (
        "abs",
        "arccos",
        "arcsin",
        "arctan",
        "arctan2",
        "ceil",
        "clip",
        "cos",
        "cosh",
        "deg2rad",
        "exp",
        "floor",
        "log",
        "log10",
        "maximum",
        "minimum",
        "rad2deg",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "where",
    ):
        namespace[name] = getattr(np, name)

    for index, column_name in enumerate(data.names):
        values = data.values[:, index]
        namespace[f"col{index}"] = values
        namespace[f"feature_{index:02d}"] = values
        if column_name.isidentifier():
            namespace[column_name] = values

    result = eval(expression, {"__builtins__": {}}, namespace)
    values = np.asarray(result, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(data.values.shape[0], float(values), dtype=np.float64)
    if values.shape != (data.values.shape[0],):
        raise ValueError(
            f"Expression must produce one value per row, got shape {values.shape}."
        )
    return values


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data: FeatureData) -> None:
        super().__init__()
        self.data = data
        self.state = self._load_state()
        self.derived_columns: list[dict[str, str]] = []
        if self._state_source_matches(self.state):
            saved_names = self.state.get("names")
            if isinstance(saved_names, list) and len(saved_names) >= len(self.data.names):
                self.data.names[:] = [str(name) for name in saved_names[: len(self.data.names)]]
            self._restore_derived_columns_from_state()
            saved_names = self.state.get("names")
            if isinstance(saved_names, list) and len(saved_names) == len(self.data.names):
                self.data.names[:] = [str(name) for name in saved_names]
        self.mask = np.ones(data.values.shape[0], dtype=bool)
        self.panels: list[CorrelationPanel | DistributionPanel] = []
        self.maximized_panel: CorrelationPanel | DistributionPanel | None = None

        self.setWindowTitle("Phase Space Slicer")
        self.resize(1400, 900)
        pg.setConfigOptions(antialias=False)

        self.columns = QtWidgets.QSpinBox()
        self.columns.setRange(1, 6)
        self.columns.setValue(
            int(self.state.get("columns", 2)) if self._state_matches_data(self.state) else 2
        )
        self.columns.valueChanged.connect(self._rebuild_grid)

        self.slice_toggle = QtWidgets.QCheckBox("Slice tool")
        self.slice_toggle.setToolTip("Show slice regions on plots.")
        self.slice_toggle.toggled.connect(self._set_slice_enabled)

        clear_button = QtWidgets.QPushButton("Clear Slice")
        clear_button.clicked.connect(self.clear_slice)

        add_button = QtWidgets.QPushButton("Add 2D Plot")
        add_button.clicked.connect(lambda: self.add_panel())

        add_distribution_button = QtWidgets.QPushButton("Add 1D Plot")
        add_distribution_button.clicked.connect(lambda: self.add_distribution_panel())

        open_button = QtWidgets.QPushButton("Open File")
        open_button.clicked.connect(self.open_file)

        self.status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(QtWidgets.QLabel("Data file"))
        self.path_label = QtWidgets.QLabel(self._data_path_label())
        self.path_label.setWordWrap(True)
        side_layout.addWidget(self.path_label)
        side_layout.addWidget(open_button)
        side_layout.addSpacing(16)
        side_layout.addWidget(QtWidgets.QLabel("Plot columns"))
        side_layout.addWidget(self.columns)
        side_layout.addSpacing(16)
        side_layout.addWidget(self.slice_toggle)
        side_layout.addWidget(clear_button)
        side_layout.addWidget(add_button)
        side_layout.addWidget(add_distribution_button)
        side_layout.addStretch(1)
        side.setMaximumWidth(280)

        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)

        plot_splitter = QtWidgets.QSplitter()
        plot_splitter.addWidget(side)
        plot_splitter.addWidget(scroll)
        plot_splitter.setStretchFactor(1, 1)

        self.spreadsheet = SpreadsheetWidget(self.data)
        self.spreadsheet.labels_changed.connect(self.rename_columns)
        self.spreadsheet.column_added.connect(self.add_derived_column)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(plot_splitter, "Plots")
        tabs.addTab(self.spreadsheet, "Spreadsheet")
        self.setCentralWidget(tabs)

        self._restore_mask_from_state()
        restored = self._restore_panels_from_state()
        if not restored:
            self.add_panel(0, 1)
            self.add_panel(2, 3)
            self.add_panel(4, 5)
            self.add_panel(6, 7)
        self.refresh_all()

    def add_panel(
        self,
        x_index: int | None = None,
        y_index: int | None = None,
    ) -> CorrelationPanel:
        count = len(self.panels)
        x = 0 if x_index is None else x_index
        y = min(count + 1, len(self.data.names) - 1) if y_index is None else y_index
        panel = CorrelationPanel(self.data, x, y)
        panel.variables_changed.connect(self.refresh_all)
        panel.slice_requested.connect(self.apply_line_slice)
        panel.remove_requested.connect(self.remove_panel)
        panel.maximize_requested.connect(self.toggle_maximize_panel)
        panel.minimize_requested.connect(self.toggle_minimize_panel)
        panel.set_selection_enabled(self.slice_toggle.isChecked())
        self.panels.append(panel)
        self._rebuild_grid()
        self.refresh_all()
        return panel

    def add_distribution_panel(self, x_index: int | None = None) -> DistributionPanel:
        count = len(self.panels)
        x = min(count, len(self.data.names) - 1) if x_index is None else x_index
        panel = DistributionPanel(self.data, x)
        panel.variables_changed.connect(self.refresh_all)
        panel.slice_requested.connect(self.apply_range_slice)
        panel.remove_requested.connect(self.remove_panel)
        panel.maximize_requested.connect(self.toggle_maximize_panel)
        panel.minimize_requested.connect(self.toggle_minimize_panel)
        panel.set_selection_enabled(self.slice_toggle.isChecked())
        self.panels.append(panel)
        self._rebuild_grid()
        self.refresh_all()
        return panel

    def remove_panel(self, panel: CorrelationPanel | DistributionPanel) -> None:
        if len(self.panels) <= 1:
            return
        self.panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()
        self._rebuild_grid()
        self.refresh_all()

    def open_file(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open numpy feature files",
            str(self.data.path.parent),
            "Numpy files (*.npy *.npz);;All files (*)",
        )
        if paths:
            self._replace_data(load_feature_files(paths))

    def rename_columns(self, names: list[str]) -> None:
        if len(names) != len(self.data.names):
            return
        self.data.names[:] = [name.strip() or f"feature_{index:02d}" for index, name in enumerate(names)]
        self._refresh_names_after_rename()

    def _refresh_names_after_rename(self) -> None:
        for panel in self.panels:
            panel.refresh_variable_names(self.data.names)
        self.spreadsheet.refresh_all(self.data)
        self.refresh_all()

    def add_derived_column(self, name: str, expression: str, values: np.ndarray) -> None:
        self.data.values = np.column_stack([self.data.values, values])
        self.data.names.append(name)
        self.derived_columns.append({"name": name, "expression": expression})
        self._refresh_names_after_rename()

    def clear_slice(self) -> None:
        self.mask = np.ones(self.data.values.shape[0], dtype=bool)
        self.refresh_all()

    def apply_line_slice(self, box: dict, x_index: int, y_index: int) -> None:
        x = self.data.values[:, x_index]
        y = self.data.values[:, y_index]
        finite = np.isfinite(x) & np.isfinite(y)
        inside = np.zeros_like(self.mask)
        inside[finite] = points_in_slice_box(x[finite], y[finite], box)
        self.mask &= inside
        self.refresh_all()

    def apply_range_slice(self, positions: np.ndarray, x_index: int) -> None:
        x = self.data.values[:, x_index]
        finite = np.isfinite(x)
        inside = np.zeros_like(self.mask)
        low, high = sorted(float(position) for position in positions)
        inside[finite] = (x[finite] >= low) & (x[finite] <= high)
        self.mask &= inside
        self.refresh_all()

    def refresh_all(self) -> None:
        active_indices = np.flatnonzero(self.mask)

        for panel in self.panels:
            x = self.data.values[active_indices, panel.x_index]
            if isinstance(panel, CorrelationPanel):
                y = self.data.values[active_indices, panel.y_index]
                finite = np.isfinite(x) & np.isfinite(y)
                panel.set_data(x[finite], y[finite])
            else:
                finite = np.isfinite(x)
                panel.set_data(x[finite])

        self.status_label.setText(
            f"{active_indices.size:,} / {self.data.values.shape[0]:,} rows in slice"
        )

    def _rebuild_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if self.maximized_panel is not None and self.maximized_panel in self.panels:
            for panel in self.panels:
                panel.setVisible(panel is self.maximized_panel)
            self.grid.addWidget(self.maximized_panel, 0, 0)
            return

        columns = self.columns.value()
        for index, panel in enumerate(self.panels):
            panel.setVisible(True)
            self.grid.addWidget(panel, index // columns, index % columns)

    def _set_slice_enabled(self, enabled: bool) -> None:
        for panel in self.panels:
            panel.set_selection_enabled(enabled)

    def toggle_maximize_panel(self, panel: CorrelationPanel | DistributionPanel) -> None:
        self.maximized_panel = None if self.maximized_panel is panel else panel
        self._rebuild_grid()

    def toggle_minimize_panel(self, panel: CorrelationPanel | DistributionPanel) -> None:
        panel.set_minimized(not panel.minimized)
        self._rebuild_grid()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.save_state()
        super().closeEvent(event)

    def save_state(self) -> None:
        state = {
            "source_path": str(self.data.path),
            "source_paths": [str(path) for path in (self.data.paths or [self.data.path])],
            "shape": list(self.data.values.shape),
            "names": self.data.names,
            "derived_columns": self.derived_columns,
            "panels": [
                self._panel_state(panel)
                for panel in self.panels
            ],
            "columns": self.columns.value(),
            "mask": self._encode_mask(self.mask),
            "geometry": bytes(self.saveGeometry()).hex(),
        }
        try:
            self._state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError as error:
            self.status_label.setText(f"Could not save state: {error}")

    def _panel_state(self, panel: CorrelationPanel | DistributionPanel) -> dict:
        if isinstance(panel, CorrelationPanel):
            return {
                "type": "2d",
                "x": panel.x_index,
                "y": panel.y_index,
                "bins": panel.bins,
                "x_range": list(panel.x_range),
                "y_range": list(panel.y_range),
                "slice_angle": panel.slice_angle_spin.value(),
                "slice_points": panel.slice_points(),
                "minimized": panel.minimized,
            }
        return {
            "type": "1d",
            "x": panel.x_index,
            "bins": panel.bins,
            "x_range": list(panel.x_range),
            "minimized": panel.minimized,
        }

    def _state_path(self) -> Path:
        return self.data.path.parent / ".phase_space_slicer_state.json"

    def _load_state(self) -> dict:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _state_matches_data(self, state: dict) -> bool:
        if not self._state_source_matches(state):
            return False
        saved_shape = state.get("shape")
        if not isinstance(saved_shape, list) or len(saved_shape) != 2:
            return False
        return saved_shape[0] == self.data.values.shape[0] and saved_shape[1] == self.data.values.shape[1]

    def _state_source_matches(self, state: dict) -> bool:
        current_paths = [str(path) for path in (self.data.paths or [self.data.path])]
        saved_paths = state.get("source_paths")
        if saved_paths is not None and saved_paths != current_paths:
            return False
        saved_shape = state.get("shape")
        if not isinstance(saved_shape, list) or len(saved_shape) != 2:
            return False
        return state.get("source_path") == str(self.data.path) and saved_shape[0] == self.data.values.shape[0]

    def _restore_derived_columns_from_state(self) -> None:
        derived_columns = self.state.get("derived_columns")
        if not isinstance(derived_columns, list):
            return
        for column in derived_columns:
            if not isinstance(column, dict):
                continue
            name = str(column.get("name", "")).strip()
            expression = str(column.get("expression", "")).strip()
            if not name or not expression:
                continue
            try:
                values = evaluate_column_expression(expression, self.data)
            except Exception:
                continue
            self.data.values = np.column_stack([self.data.values, values])
            self.data.names.append(name)
            self.derived_columns.append({"name": name, "expression": expression})

    def _restore_mask_from_state(self) -> None:
        if not self._state_matches_data(self.state):
            return
        encoded_mask = self.state.get("mask")
        if isinstance(encoded_mask, str):
            mask = self._decode_mask(encoded_mask, self.data.values.shape[0])
            if mask is not None:
                self.mask = mask

    def _restore_panels_from_state(self) -> bool:
        if not self._state_matches_data(self.state):
            return False
        panels = self.state.get("panels")
        if not isinstance(panels, list):
            return False

        restored = 0
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            panel_type = panel.get("type", "2d")
            x_index = int(panel.get("x", -1))
            if panel_type == "1d" and self._valid_column_index(x_index):
                created = self.add_distribution_panel(x_index)
                created.apply_settings(
                    bins=panel.get("bins", self.state.get("bins", 160)),
                    x_range=panel.get("x_range"),
                )
                created.set_minimized(bool(panel.get("minimized", False)))
                restored += 1
                continue

            y_index = int(panel.get("y", -1))
            if self._valid_column_index(x_index) and self._valid_column_index(y_index):
                created = self.add_panel(x_index, y_index)
                created.apply_settings(
                    bins=panel.get("bins", self.state.get("bins", 160)),
                    x_range=panel.get("x_range"),
                    y_range=panel.get("y_range"),
                    slice_angle=panel.get("slice_angle"),
                    slice_points=panel.get("slice_points"),
                )
                created.set_minimized(bool(panel.get("minimized", False)))
                restored += 1

        geometry = self.state.get("geometry")
        if isinstance(geometry, str):
            self.restoreGeometry(QtCore.QByteArray.fromHex(geometry.encode("ascii")))
        return restored > 0

    def _valid_column_index(self, index: int) -> bool:
        return 0 <= index < len(self.data.names)

    def _encode_mask(self, mask: np.ndarray) -> str:
        packed = np.packbits(mask.astype(np.uint8))
        return base64.b64encode(packed.tobytes()).decode("ascii")

    def _decode_mask(self, encoded: str, length: int) -> np.ndarray | None:
        try:
            packed = np.frombuffer(base64.b64decode(encoded.encode("ascii")), dtype=np.uint8)
            return np.unpackbits(packed, count=length).astype(bool)
        except (ValueError, TypeError):
            return None

    def _replace_data(self, data: FeatureData) -> None:
        self.save_state()
        self.data = data
        self.state = self._load_state()
        self.derived_columns = []
        if self._state_source_matches(self.state):
            saved_names = self.state.get("names")
            if isinstance(saved_names, list) and len(saved_names) >= len(self.data.names):
                self.data.names[:] = [str(name) for name in saved_names[: len(self.data.names)]]
            self._restore_derived_columns_from_state()
            saved_names = self.state.get("names")
            if isinstance(saved_names, list) and len(saved_names) == len(self.data.names):
                self.data.names[:] = [str(name) for name in saved_names]
        self.mask = np.ones(data.values.shape[0], dtype=bool)
        self.path_label.setText(self._data_path_label())
        if self._state_matches_data(self.state):
            self.columns.setValue(int(self.state.get("columns", self.columns.value())))
        for panel in self.panels:
            panel.setParent(None)
            panel.deleteLater()
        self.panels = []
        self.maximized_panel = None
        self.spreadsheet.refresh_all(self.data)
        self._restore_mask_from_state()
        restored = self._restore_panels_from_state()
        if not restored:
            self.add_panel(0, 1)
        self.refresh_all()

    def _data_path_label(self) -> str:
        paths = self.data.paths or [self.data.path]
        if len(paths) == 1:
            return paths[0].name
        return "\n".join(path.name for path in paths)


def rotated_coordinates(
    x: np.ndarray, y: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(angle)
    along = x * np.cos(radians) + y * np.sin(radians)
    across = -x * np.sin(radians) + y * np.cos(radians)
    return along, across


def slice_box_polygon(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    angle: float,
) -> list[tuple[float, float]]:
    along, across = rotated_coordinates(
        np.array([point_a[0], point_b[0]], dtype=float),
        np.array([point_a[1], point_b[1]], dtype=float),
        angle,
    )
    along_low, along_high = float(np.min(along)), float(np.max(along))
    across_low, across_high = float(np.min(across)), float(np.max(across))
    return [
        inverse_rotated_coordinates(along_low, across_low, angle),
        inverse_rotated_coordinates(along_high, across_low, angle),
        inverse_rotated_coordinates(along_high, across_high, angle),
        inverse_rotated_coordinates(along_low, across_high, angle),
    ]


def inverse_rotated_coordinates(along: float, across: float, angle: float) -> tuple[float, float]:
    radians = np.deg2rad(angle)
    x = along * np.cos(radians) - across * np.sin(radians)
    y = along * np.sin(radians) + across * np.cos(radians)
    return float(x), float(y)


def points_in_slice_box(x: np.ndarray, y: np.ndarray, box: dict) -> np.ndarray:
    angle = float(box["angle"])
    along, across = rotated_coordinates(x, y, angle)
    corner_along, corner_across = rotated_coordinates(
        np.array([float(box["x1"]), float(box["x2"])]),
        np.array([float(box["y1"]), float(box["y2"])]),
        angle,
    )
    along_low, along_high = np.min(corner_along), np.max(corner_along)
    across_low, across_high = np.min(corner_across), np.max(corner_across)
    return (
        (along >= along_low)
        & (along <= along_high)
        & (across >= across_low)
        & (across <= across_high)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive phase-space correlation slicer.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["clustercalib_nn_500k_batch001.npy"],
        help="Paths to one or more .npy or .npz feature tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_feature_files([Path(path) for path in args.paths])
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Phase Space Slicer")
    window = MainWindow(data)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
