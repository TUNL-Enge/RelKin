"""PySide6 GUI for relativistic two-body nuclear reaction kinematics."""
import sys

import numpy as np
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nuclear_kinematics.core.kinematics import (
    cm_energy,
    cm_kinetic_energies,
    kinematic_compression,
    lab_to_cm_solutions,
    max_lab_angle,
    threshold_energy,
)
from nuclear_kinematics.core.masses import get_mass_MeV, load_ame2020, parse_nuclide, qvalue


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

        # Output
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Courier"))
        layout.addWidget(self.output)

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
            self.output.setPlainText(f"Input error: {e}")
            return

        try:
            Q       = qvalue(ZA_a, ZA_A, ZA_b, ZA_B)
            sqrts   = cm_energy(m_a, m_A, T_a)
            T_thresh = threshold_energy(m_a, m_A, m_b, m_B)
            T_b_cm, T_B_cm = cm_kinetic_energies(m_b, m_B, sqrts)
            max_ang = max_lab_angle(m_a, m_A, m_b, m_B, T_a)
            solutions = lab_to_cm_solutions(theta_lab_rad, m_a, m_A, m_b, m_B, T_a)
        except Exception as e:
            self.output.setPlainText(f"Calculation error: {e}")
            return

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
        lines.append("")

        if T_thresh > 0 and T_a < T_thresh:
            lines.append(
                f"WARNING: T_a = {T_a:.4f} MeV is below threshold {T_thresh:.4f} MeV"
            )
        elif not solutions:
            lines.append(
                f"No solution: \u03b8_b_lab = {theta_lab_deg:.4f} deg exceeds the kinematic limit"
            )
        else:
            for i, sol in enumerate(solutions, 1):
                lines.append(
                    f"Solution {i}  (\u03b8_b_cm = {np.degrees(sol['theta_b_cm']):.4f} deg):"
                )
                lines.append(f"  T_b_lab:  {sol['T_b_lab']:.4f} MeV")
                lines.append(f"  T_B_lab:  {sol['T_B_lab']:.4f} MeV")
                kp = kinematic_compression(sol["theta_b_cm"], m_a, m_A, m_b, m_B, T_a)[0]
                lines.append(f"  kp:       {kp:.4f} rad^-1")
                if i < len(solutions):
                    lines.append("")

        self.output.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    window = KinematicsApp()
    window.resize(420, 560)
    window.show()
    sys.exit(app.exec())
