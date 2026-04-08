"""PySide6 GUI for relativistic two-body nuclear reaction kinematics."""
import os
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from nuclear_kinematics.core.kinematics import (
    cm_energy,
    cm_kinetic_energies,
    kinematic_compression,
    kinematic_locus,
    lab_to_cm_solutions,
    max_lab_angle,
    threshold_energy,
)
from nuclear_kinematics.core.constants import U_TO_MEV
from nuclear_kinematics.core.masses import format_nuclide, get_mass_MeV, load_ame2020, parse_nuclide, qvalue


class KinematicsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear Kinematics Calculator")
        self._build_ui()
        load_ame2020()
        # Populate initial mass labels after AME table is loaded
        for text, lbl in (
            (self.inp_a.text(), self.mass_lbl_a),
            (self.inp_A.text(), self.mass_lbl_A),
            (self.inp_b.text(), self.mass_lbl_b),
            (self.inp_B.text(), self.mass_lbl_B),
        ):
            self._update_mass_label(text, lbl)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Reaction inputs
        reaction_group = QGroupBox("Reaction  (a + A \u2192 b + B)")
        reaction_hbox = QHBoxLayout()

        grid = QGridLayout()

        grid.addWidget(QLabel("Projectile (a):"), 0, 0)
        self.inp_a = QLineEdit("p")
        self.inp_a.setMaximumWidth(80)
        grid.addWidget(self.inp_a, 0, 1)
        self.mass_lbl_a = QLabel("\u2014")
        grid.addWidget(self.mass_lbl_a, 0, 2)

        grid.addWidget(QLabel("Target (A):"), 1, 0)
        self.inp_A = QLineEdit("12C")
        self.inp_A.setMaximumWidth(80)
        grid.addWidget(self.inp_A, 1, 1)
        self.mass_lbl_A = QLabel("\u2014")
        grid.addWidget(self.mass_lbl_A, 1, 2)

        grid.addWidget(QLabel("Ejectile (b):"), 2, 0)
        self.inp_b = QLineEdit("p")
        self.inp_b.setMaximumWidth(80)
        grid.addWidget(self.inp_b, 2, 1)
        self.mass_lbl_b = QLabel("\u2014")
        grid.addWidget(self.mass_lbl_b, 2, 2)

        grid.addWidget(QLabel("Residual (B):"), 3, 0)
        self.inp_B = QLineEdit("12C")
        self.inp_B.setMaximumWidth(80)
        grid.addWidget(self.inp_B, 3, 1)
        self.mass_lbl_B = QLabel("\u2014")
        grid.addWidget(self.mass_lbl_B, 3, 2)

        grid.setColumnStretch(3, 1)  # push grid to the left
        reaction_hbox.addLayout(grid)

        # Logo on the right
        logo_lbl = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "FENRISLogo-notext.png")
        pixmap = QPixmap(os.path.normpath(logo_path))
        logo_lbl.setPixmap(pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation))
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        reaction_hbox.addWidget(logo_lbl)

        reaction_group.setLayout(reaction_hbox)

        self.inp_a.textChanged.connect(self._auto_fill_B)
        self.inp_A.textChanged.connect(self._auto_fill_B)
        self.inp_b.textChanged.connect(self._auto_fill_B)

        self.inp_a.textChanged.connect(lambda t: self._update_mass_label(t, self.mass_lbl_a))
        self.inp_A.textChanged.connect(lambda t: self._update_mass_label(t, self.mass_lbl_A))
        self.inp_b.textChanged.connect(lambda t: self._update_mass_label(t, self.mass_lbl_b))
        self.inp_B.textChanged.connect(lambda t: self._update_mass_label(t, self.mass_lbl_B))

        layout.addWidget(reaction_group)

        # Beam energy
        energy_group = QGroupBox("Beam Energy")
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("T_a:"))
        self.inp_energy = QLineEdit("20.0")
        energy_layout.addWidget(self.inp_energy)
        self.energy_units = QComboBox()
        self.energy_units.addItems(["MeV", "MeV/u"])
        energy_layout.addWidget(self.energy_units)
        energy_group.setLayout(energy_layout)

        # Lab angle
        angle_group = QGroupBox("Lab Angle")
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("\u03b8_lab (b, degrees):"))
        self.inp_angle = QLineEdit("30.0")
        angle_layout.addWidget(self.inp_angle)
        angle_group.setLayout(angle_layout)

        beam_angle_row = QHBoxLayout()
        beam_angle_row.addWidget(energy_group)
        beam_angle_row.addWidget(angle_group)
        layout.addLayout(beam_angle_row)

        # Calculate + Move detector buttons
        btn_row = QHBoxLayout()
        self.calc_btn = QPushButton("Calculate")
        self.calc_btn.clicked.connect(self._calculate)
        btn_row.addWidget(self.calc_btn)
        self.move_detector_btn = QPushButton("Move detector")
        btn_row.addWidget(self.move_detector_btn)
        layout.addLayout(btn_row)

        for field in (self.inp_a, self.inp_A, self.inp_b, self.inp_B,
                      self.inp_energy, self.inp_angle):
            field.returnPressed.connect(self.calc_btn.click)

        # Global metrics (scalar quantities)
        metrics_group = QGroupBox("Global Metrics")
        mg = QGridLayout()

        def _add_metric_row(row, name):
            mg.addWidget(QLabel(name), row, 0)
            val_lbl = QLabel("\u2014")
            mg.addWidget(val_lbl, row, 1)
            return val_lbl

        self.lbl_qvalue = _add_metric_row(0, "Q-value:")
        self.lbl_sqrts  = _add_metric_row(1, "\u221as:")
        self.lbl_thresh = _add_metric_row(2, "Threshold T\u2090:")
        self.lbl_maxang = _add_metric_row(3, "Max \u03b8_b_lab:")

        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet("color: red;")
        mg.addWidget(self.lbl_warning, 4, 0, 1, 2)

        metrics_group.setLayout(mg)
        layout.addWidget(metrics_group)

        # Per-solution results table
        headers = [
            "Sol",
            "T_b_lab (MeV)",
            "\u03b8_b_lab (deg)",
            "T_b_cm (MeV)",
            "\u03b8_b_cm (deg)",
            "T_B_lab (MeV)",
            "\u03b8_B_lab (deg)",
            "T_B_cm (MeV)",
            "\u03b8_B_cm (deg)",
            "kp (rad\u207b\u00b9)",
        ]
        self.results_table = QTableWidget(0, len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setMaximumHeight(120)
        layout.addWidget(self.results_table)

        # Kinematic locus plot
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

    def _update_mass_label(self, text, label):
        try:
            Z, A = parse_nuclide(text.strip())
            mass = get_mass_MeV(Z, A)
            label.setText(f"{mass / U_TO_MEV:.6f} u")
        except Exception:
            label.setText("\u2014")

    def _auto_fill_B(self):
        try:
            Za, Aa = parse_nuclide(self.inp_a.text().strip())
            ZA, AA = parse_nuclide(self.inp_A.text().strip())
            Zb, Ab = parse_nuclide(self.inp_b.text().strip())
        except ValueError:
            return
        ZB = Za + ZA - Zb
        AB = Aa + AA - Ab
        if ZB < 0 or AB <= 0:
            return
        try:
            text = format_nuclide(ZB, AB)
        except ValueError:
            return
        self.inp_B.blockSignals(True)
        self.inp_B.setText(text)
        self.inp_B.blockSignals(False)
        self._update_mass_label(text, self.mass_lbl_B)

    def _clear_metrics(self):
        for lbl in (self.lbl_qvalue, self.lbl_sqrts, self.lbl_thresh, self.lbl_maxang):
            lbl.setText("\u2014")
        self.lbl_warning.setText("")

    def _calculate(self):
        self._clear_metrics()
        try:
            ZA_a = parse_nuclide(self.inp_a.text().strip())
            ZA_A = parse_nuclide(self.inp_A.text().strip())
            ZA_b = parse_nuclide(self.inp_b.text().strip())
            ZA_B = parse_nuclide(self.inp_B.text().strip())

            m_a = get_mass_MeV(*ZA_a)
            m_A = get_mass_MeV(*ZA_A)
            m_b = get_mass_MeV(*ZA_b)
            m_B = get_mass_MeV(*ZA_B)

            T_input = float(self.inp_energy.text())
            T_a = T_input * ZA_a[1] if self.energy_units.currentText() == "MeV/u" else T_input

            theta_lab_deg = float(self.inp_angle.text())
            theta_lab_rad = np.radians(theta_lab_deg)

        except (ValueError, KeyError) as e:
            self.lbl_warning.setText(f"Input error: {e}")
            return

        try:
            Q       = qvalue(ZA_a, ZA_A, ZA_b, ZA_B)
            sqrts   = cm_energy(m_a, m_A, T_a)
            T_thresh = threshold_energy(m_a, m_A, m_b, m_B)
            T_b_cm, T_B_cm = cm_kinetic_energies(m_b, m_B, sqrts)
            max_ang = max_lab_angle(m_a, m_A, m_b, m_B, T_a)
            solutions = lab_to_cm_solutions(theta_lab_rad, m_a, m_A, m_b, m_B, T_a)
        except Exception as e:
            self.lbl_warning.setText(f"Calculation error: {e}")
            return

        # --- Global metrics ---
        self.lbl_qvalue.setText(f"{Q:+.4f} MeV")
        self.lbl_sqrts.setText(f"{sqrts:.4f} MeV")
        self.lbl_thresh.setText(f"{T_thresh:.4f} MeV" if T_thresh > 0 else "N/A (exothermic)")
        self.lbl_maxang.setText(
            f"{np.degrees(max_ang):.4f} \u00b0" if max_ang is not None else "(full range)"
        )

        # --- Per-solution table ---
        self.results_table.setRowCount(0)

        if T_thresh > 0 and T_a < T_thresh:
            self.lbl_warning.setText(
                f"WARNING: T_a = {T_a:.4f} MeV is below threshold {T_thresh:.4f} MeV"
            )
        elif not solutions:
            self.lbl_warning.setText(
                f"No solution: \u03b8_b_lab = {theta_lab_deg:.4f} deg exceeds the kinematic limit"
            )
        else:
            self.results_table.setRowCount(len(solutions))
            for i, sol in enumerate(solutions):
                theta_b_cm_deg = np.degrees(sol['theta_b_cm'])
                theta_B_cm_deg = 180.0 - theta_b_cm_deg
                theta_B_lab_deg = np.degrees(sol['theta_B_lab'])
                kp = kinematic_compression(sol["theta_b_cm"], m_a, m_A, m_b, m_B, T_a)[0]

                values = [
                    f"{i + 1}",
                    f"{sol['T_b_lab']:.4f}",
                    f"{theta_lab_deg:.4f}",
                    f"{T_b_cm:.4f}",
                    f"{theta_b_cm_deg:.4f}",
                    f"{sol['T_B_lab']:.4f}",
                    f"{theta_B_lab_deg:.4f}",
                    f"{T_B_cm:.4f}",
                    f"{theta_B_cm_deg:.4f}",
                    f"{kp:.4f}",
                ]
                for col, val in enumerate(values):
                    item = QTableWidgetItem(val)
                    self.results_table.setItem(i, col, item)

            self.results_table.resizeColumnsToContents()

        self._update_plot(m_a, m_A, m_b, m_B, T_a)

    def _update_plot(self, m_a, m_A, m_b, m_B, T_a):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        theta_deg, T_lab = kinematic_locus(m_a, m_A, m_b, m_B, T_a)
        ax.plot(theta_deg, T_lab, color="steelblue")
        ax.set_xlabel("\u03b8_b_lab (deg)")
        ax.set_ylabel("T_b_lab (MeV)")
        a_text = self.inp_a.text().strip()
        A_text = self.inp_A.text().strip()
        b_text = self.inp_b.text().strip()
        B_text = self.inp_B.text().strip()
        T_a_display = float(self.inp_energy.text())
        units = self.energy_units.currentText()
        ax.set_title(
            f"{A_text}({a_text},{b_text}){B_text}  T_a = {T_a_display:.1f} {units}"
        )
        ax.grid(True)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = KinematicsApp()
    window.resize(950, 800)
    window.show()
    sys.exit(app.exec())
