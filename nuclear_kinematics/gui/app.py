"""PySide6 GUI for relativistic two-body nuclear reaction kinematics."""
import sys

import numpy as np
from PySide6.QtGui import QFont
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
    QTextEdit,
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
from nuclear_kinematics.core.masses import format_nuclide, get_mass_MeV, load_ame2020, parse_nuclide, qvalue


class KinematicsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear Kinematics Calculator")
        self._build_ui()
        load_ame2020()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Reaction inputs
        reaction_group = QGroupBox("Reaction  (a + A \u2192 b + B)")
        grid = QGridLayout()
        grid.addWidget(QLabel("Projectile (a):"), 0, 0)
        self.inp_a = QLineEdit("p")
        grid.addWidget(self.inp_a, 0, 1)
        grid.addWidget(QLabel("Target (A):"), 1, 0)
        self.inp_A = QLineEdit("12C")
        grid.addWidget(self.inp_A, 1, 1)
        grid.addWidget(QLabel("Ejectile (b):"), 2, 0)
        self.inp_b = QLineEdit("p")
        grid.addWidget(self.inp_b, 2, 1)
        grid.addWidget(QLabel("Residual (B):"), 3, 0)
        self.inp_B = QLineEdit("12C")
        grid.addWidget(self.inp_B, 3, 1)
        reaction_group.setLayout(grid)

        self.inp_a.textChanged.connect(self._auto_fill_B)
        self.inp_A.textChanged.connect(self._auto_fill_B)
        self.inp_b.textChanged.connect(self._auto_fill_B)
        layout.addWidget(reaction_group)

        # Beam energy
        energy_group = QGroupBox("Beam Energy")
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("T_a:"))
        self.inp_energy = QLineEdit("100.0")
        energy_layout.addWidget(self.inp_energy)
        self.energy_units = QComboBox()
        self.energy_units.addItems(["MeV", "MeV/u"])
        energy_layout.addWidget(self.energy_units)
        energy_group.setLayout(energy_layout)
        layout.addWidget(energy_group)

        # Lab angle
        angle_group = QGroupBox("Lab Angle")
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("\u03b8_lab (b, degrees):"))
        self.inp_angle = QLineEdit("30.0")
        angle_layout.addWidget(self.inp_angle)
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        # Calculate button
        self.calc_btn = QPushButton("Calculate")
        self.calc_btn.clicked.connect(self._calculate)
        layout.addWidget(self.calc_btn)

        for field in (self.inp_a, self.inp_A, self.inp_b, self.inp_B,
                      self.inp_energy, self.inp_angle):
            field.returnPressed.connect(self.calc_btn.click)

        # Global metrics (scalar quantities)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Courier"))
        self.metrics_text.setMaximumHeight(160)
        layout.addWidget(self.metrics_text)

        # Per-solution results table
        headers = [
            "Sol",
            "\u03b8_b_cm (deg)",
            "T_b_lab (MeV)",
            "\u03b8_b_lab (deg)",
            "T_B_lab (MeV)",
            "\u03b8_B_cm (deg)",
            "\u03b8_B_lab (deg)",
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

    def _calculate(self):
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
            self.metrics_text.setPlainText(f"Input error: {e}")
            return

        try:
            Q       = qvalue(ZA_a, ZA_A, ZA_b, ZA_B)
            sqrts   = cm_energy(m_a, m_A, T_a)
            T_thresh = threshold_energy(m_a, m_A, m_b, m_B)
            T_b_cm, T_B_cm = cm_kinetic_energies(m_b, m_B, sqrts)
            max_ang = max_lab_angle(m_a, m_A, m_b, m_B, T_a)
            solutions = lab_to_cm_solutions(theta_lab_rad, m_a, m_A, m_b, m_B, T_a)
        except Exception as e:
            self.metrics_text.setPlainText(f"Calculation error: {e}")
            return

        # --- Global metrics block ---
        lines = []
        lines.append(f"Q-value:          {Q:+.4f} MeV")
        lines.append(f"sqrt(s):          {sqrts:.4f} MeV")
        if T_thresh > 0:
            lines.append(f"Threshold T_a:    {T_thresh:.4f} MeV")
        else:
            lines.append(f"Threshold T_a:    N/A (exothermic)")
        if max_ang is not None:
            lines.append(f"Max \u03b8_b_lab:       {np.degrees(max_ang):.4f} deg")
        else:
            lines.append(f"Max \u03b8_b_lab:       (full range)")
        lines.append(f"T_b_cm:           {T_b_cm:.4f} MeV")
        lines.append(f"T_B_cm:           {T_B_cm:.4f} MeV")
        self.metrics_text.setPlainText("\n".join(lines))

        # --- Per-solution table ---
        self.results_table.setRowCount(0)

        if T_thresh > 0 and T_a < T_thresh:
            self.metrics_text.append(
                f"\nWARNING: T_a = {T_a:.4f} MeV is below threshold {T_thresh:.4f} MeV"
            )
        elif not solutions:
            self.metrics_text.append(
                f"\nNo solution: \u03b8_b_lab = {theta_lab_deg:.4f} deg exceeds the kinematic limit"
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
                    f"{theta_b_cm_deg:.4f}",
                    f"{sol['T_b_lab']:.4f}",
                    f"{theta_lab_deg:.4f}",
                    f"{sol['T_B_lab']:.4f}",
                    f"{theta_B_cm_deg:.4f}",
                    f"{theta_B_lab_deg:.4f}",
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
            f"{a_text}({A_text},{b_text}){B_text}  T_a = {T_a_display:.1f} {units}"
        )
        ax.grid(True)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = KinematicsApp()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())
