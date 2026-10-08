import sys
import os
import csv
import numpy as np
import pandas as pd
# PyQt6 core and GUI components
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QAction, QColor
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# PyQt6 widgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QMenuBar, QMenu, QFrame, QScrollArea,
    QGroupBox, QSizePolicy, QComboBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QFileDialog)
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QFrame, QPushButton, QScrollArea
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout,QFrame, QDialog)
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Matplotlib for embedding plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import io
from PyQt6.QtGui import QPixmap, QImage
from matplotlib import rcParams
import os
from inductance_megnetisante import *
from rotorbloqueefunctions import *
from graph_dispatcher import *
from nominale import *
def latex_to_pixmap(latex_str, dpi=75):
    # Use modern font and white text
    rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "mathtext.default": "regular",
        "text.color": "white",  # White LaTeX formula
        "figure.facecolor": "#121924",  # Background color for figure
        "savefig.facecolor": "#121924",  # Save figure with same background
        "savefig.edgecolor": "#121924",
    })

    fig = plt.figure(figsize=(0.01, 0.01))  # Tiny initial size
    fig.patch.set_facecolor("#121924")

    # Render the LaTeX formula at the center
    fig.text(0.0, 0.0, f"${latex_str}$", fontsize=20)  # fontsize < 15 looks better compact

    buf = io.BytesIO()
    plt.axis('off')
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    image = QImage()
    image.loadFromData(buf.read())
    return QPixmap.fromImage(image)

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        # Set up window geometry and style
        self.setGeometry(200, 70, 500, 540)
        self.setWindowTitle("Electrical Machine Analyzer")
        self.setStyleSheet("background-color: #111721;")

        # Set up layout and pages
        self.create_layout()

    def create_layout(self):
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create and add pages
        splash_page = self.create_splash_screen()
        self.main = self.mainpage()
        self.input_page = self.create_input_page()
        self.analytic_graph_page = self.create_analytic_graph_page()
        self.nominal_param_page = self.create_nominal_param_page()

        self.stacked_widget.addWidget(splash_page)
        self.stacked_widget.addWidget(self.main)
        self.stacked_widget.addWidget(self.input_page)
        self.stacked_widget.addWidget(self.analytic_graph_page)
        self.stacked_widget.addWidget(self.nominal_param_page)

        # Show main page after splash screen delay
        QTimer.singleShot(1000, lambda: self.stacked_widget.setCurrentWidget(self.main)) 

# LOGO APP
    def create_header(self):
        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # header_layout.setContentsMargins(0, 0, 0, 0)  # Adding top margin for header
        # Logo
        logo_label = QLabel()
        import os

        # Path relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "Artboard 1.png")

        logo_pixmap = QPixmap(image_path)


        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label.setText("⚠️ Logo")
            logo_label.setStyleSheet("color: red;")
        header_layout.addWidget(logo_label)

        # Title
        title_label = QLabel("""<p style="font-family:Angled; font-size: 16px; font-weight: bold; color: white; margin-top: 0px; text-align: left;">Electrical Machine Analysis</p>""")
        title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Center vertically
        header_layout.addWidget(title_label)

        # Add stretch for spacing
        header_layout.addStretch()

        # Create and return header widget
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        return header_widget

#welcoming page
    def create_splash_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Logo for splash screen
        logo_label = QLabel()
        import os

        # Path relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "Artboard 1.png")
        
        logo_pixmap = QPixmap(image_path)


        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label.setText("Logo not found")
            logo_label.setStyleSheet("color: red;")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)


        # Welcome Message
        welcome_label = QLabel("<span style='font-size: 18px; font-weight: bold; color: white;'>Bienvenue...</span>")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        # Footer Message
        source_label = QLabel("<p style='font-size: 10px; color: white;'>Funded by Electrical students 2024-2025</p>")
        source_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(source_label)

        page.setLayout(layout)
        page.setStyleSheet("background-color: ##0e0d12;")
        return page

#THE MAIN PAGE
    def mainpage(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(30)

        # Menu bar
        menu_bar = QMenuBar(self)
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #2C3E50;  # Darker background for the bar
                color: white;
                font-size: 14px;  # Slightly larger font for readability
                font-family: 'Montserrat', sans-serif;
                border: none;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: #34495e;
                color: white;
                padding: 8px 15px;
                border-radius: 6px;
                margin: 0 2px;
            }
            QMenuBar::item:selected {
                background-color: #1abc9c;  # Highlight color for selected items
                color: white;
                border-radius: 6px;
            }
            QMenuBar::item:hover {
                background-color: #2980b9;  # Hover effect
                color: white;
            }
            QMenu {
                background-color: #34495e;
                border: 1px solid #2c3e50;
                border-radius: 6px;
                min-width: 150px;
                padding: 6px;
            }
            QMenu::item {
                color: white;
                padding: 8px 15px;
                background-color: #34495e;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #1abc9c;
                color: white;
            }
            QMenu::item:hover {
            background-color: #2980b9;
                    color: white;
                }
            """)
        # Créer le menu 'Modèle Analytique'
        info_menu = QMenu("Modèle Analytique de la Machine Asynchrone", self)
        
        # Sous-menus pour les trois régions
        region1_action = QAction("Région I : Entrefer", self)
        region2_action = QAction("Région II : Barres rotoriques", self)
        region3_action = QAction("Région III : Encoches statoriques", self)
        
        # Connecter chaque action à une fonction d'affichage
        region1_action.triggered.connect(lambda: self.display_region("region1"))
        region2_action.triggered.connect(lambda: self.display_region("region2"))
        region3_action.triggered.connect(lambda: self.display_region("region3"))
        
        # Ajouter les actions au menu principal
        info_menu.addAction(region1_action)
        info_menu.addAction(region2_action)
        info_menu.addAction(region3_action)
        
        # Ajouter ce menu à la barre de menus
        menu_bar.addMenu(info_menu)
        layout.setMenuBar(menu_bar)


        # Header
        layout.addWidget(self.create_header())


        # Abstract/Intro
        intro_label = QLabel("""
            <div style='font-family: Montserrat; font-size: 12px; color: #ffffff; line-height: 1.9; max-width: 520px;'>
                <p style="font-size: 16px; font-weight: bold;">⚙️ Conception & Analyse des Machines Électriques</p>
                <p>Bienvenue dans une plateforme dédiée à l'étude des <b>machines électriques</b>, et plus particulièrement à la <b>machine asynchrone à cage</b>.</p>
                <p>Grâce à cette application, vous pouvez :</p>
                <ul style="margin-left: -15px;">
                    <li>🔍 Explorer la structure interne de la machine (entrefer, rotor, stator)</li>
                    <li>📊 Déterminer ses paramètres caractéristiques à partir d’essais pratiques</li>
                    <li>🧠 Approfondir vos connaissances à travers des simulations guidées</li>
                </ul>
                <p>Un outil moderne pour les étudiants, les enseignants et les passionnés d’électrotechnique.</p>
            </div>
        """)
        intro_label.setWordWrap(True)
        intro_box = QFrame()
        intro_box.setStyleSheet("background-color: #121924; border-radius: 12px; padding: 5px; border: 1px solid #121924;")
        box_layout = QVBoxLayout(intro_box)
        box_layout.addWidget(intro_label)
        layout.addWidget(intro_box)

        # Title
        title_label = QLabel("""
            <div style='text-align: left;'>
            <span style='font-size: 33px; font-weight: bold; color: white;'>Sélectionner votre </span>
            <span style='font-size: 33px; font-weight: bold; color: #00a5cf;'>objectif</span></div>
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title_label)

        # Objective Boxes
        objectives_layout = QHBoxLayout()
        objectives_layout.setSpacing(30)
        objectives_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Première box : Entrée des paramètres de la machine
        parametres_box = self.create_objective_container(
            "Paramètres de la Machine",
            "Entrez les paramètres de votre machine asynchrone pour initier l’analyse.",
            "Configurer",
            "#00a5cf",
            lambda: self.stacked_widget.setCurrentWidget(self.input_page)
        )

        # Deuxième box : Visualisation des graphes analytiques
        # Create Analyse Graphique button and disable it initially
        self.analyse_box = self.create_objective_container(
            "Analyse Graphique",
            "Visualisez les courbes issues de l’étude analytique de la machine.",
            "Visualiser",
            "#00a5cf",
            lambda: self.stacked_widget.setCurrentWidget(self.analytic_graph_page)
        )
        self.analyse_box.setEnabled(False)
        
        # Create Régime Nominal button and disable it initially
        self.fonctionnement_box = self.create_objective_container(
            "Régime Nominal",
            "Affichez les paramètres caractéristiques en fonctionnement nominal.",
            "Afficher",
            "#00a5cf",
            lambda: self.stacked_widget.setCurrentWidget(self.nominal_param_page)
        )
        self.fonctionnement_box.setEnabled(False)

        # Ajout des boîtes à l'interface
        objectives_layout.addWidget(parametres_box)
        objectives_layout.addWidget(self.analyse_box)
        objectives_layout.addWidget(self.fonctionnement_box)

        layout.addLayout(objectives_layout)

        # Footer
        source_label = QLabel("<p style='font-size: 10px; color: white;'>Funded by Electrical students 2024-2025</p>")
        source_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(source_label)
        page.setLayout(layout)
        return page

#INFO MENU OF THE REGIONS
    def display_region(self, region):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(1)
        scroll.setWidget(content_widget)

        layout.addWidget(scroll)

        self.stacked_widget.addWidget(page)
        self.stacked_widget.setCurrentWidget(page)

        title_map = {
            "region1": "🌐 Région I : Entrefer",
            "region2": "🧲 Région II : Barres Rotoriques",
            "region3": "🔩 Région III : Encoches Statoriques"
        }

        title_label = QLabel(title_map.get(region, "📘 Région"))
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        layout.insertWidget(0, title_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        def section(title_text, latex_equation, icon=""):
            box = QGroupBox()
            box.setStyleSheet("""
                QGroupBox {
                    background-color: #121924;
                    border: 2px solid #121924;
                    border-radius: 12px;
                    padding: 5px;
                    margin-top: 4px;
                }
            """)
            box_layout = QVBoxLayout(box)

            title = QLabel(f"{icon} {title_text}")
            title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
            title.setStyleSheet("color: #ffffff; background-color: transparent;")
            title.setWordWrap(True)

            image_label = QLabel()
            image_label.setPixmap(latex_to_pixmap(latex_equation, dpi=80))
            image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            box_layout.addWidget(title)
            box_layout.addSpacing(10)
            box_layout.addWidget(image_label)

            return box

        def centered_image(path):
            img_label = QLabel()
            pixmap = QPixmap(path)
            pixmap = pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            return img_label

        if region == "region1":
            content_layout.addWidget(section("Équation du potentiel scalaire", 
                r"AI(r,\theta)= A_{10} + A_{20}\ln (r)+\sum_{n=1}^{\infty} ((A_{1n}r^n+A_{2n}r^{-n})\sin(n\theta)+(A_{3n}r^n+A_{4n}r^{-n})\cos(n\theta))", "🧮"))

            content_layout.addWidget(section("Champ magnétique radial", 
                r"BI_r(r,\theta)= \frac{1}{r}\sum_{n=1}^{\infty}n((A_{1n}r^n+A_{2n}r^{-n})\cos(n\theta)-(A_{3n}r^n+A_{4n}r^{-n})\sin(n\theta))", "🧲"))

            content_layout.addWidget(section("Champ magnétique tangentiel", 
                r"BI_{\theta}(r,\theta)= -\frac{A_{20}}{r}-\sum_{n=1}^{\infty} n ((A_{1n}r^{n-1}-A_{2n}r^{-n-1})\sin(n\theta)+(A_{3n}r^{n-1}-A_{4n}r^{-n-1})\cos(n\theta))", "📐"))

        elif region == "region2":
            content_layout.addWidget(centered_image("A1.png"))

            content_layout.addWidget(section("Barres rotoriques saines", 
                r"AII_{z,j} (r,\theta) = B_{j0} f(r) + \sum_{m=1}^{\infty} B_{jm} g_m(r) \cos\left(\frac{m\pi}{b} (\theta - g_j + \frac{b}{2})\right)", "🧱"))

            content_layout.addWidget(section("Fonction f(r)", 
                r"f(r) = J_0(\gamma r) - \frac{J_1(\gamma R_1) Y_0(\gamma r)}{Y_1(\gamma R_1)}", "📊"))

            content_layout.addWidget(section("Champ radial B_r", 
                r"B_r(r, \theta) = -\frac{1}{r} \sum_{m=1}^{\infty} \frac{m\pi}{b} B_{jm} g_m(r) \sin \left( \frac{m\pi}{b}(\theta - g_j + \frac{b}{2}) \right)", "📏"))

            content_layout.addWidget(section("Barres rotoriques cassées", 
                r"A_{II,c}(r,\theta) = W_{c,0} + \sum_{h=1}^{\infty} W_{c,h} h_l(r) \cos \left( \frac{h\pi}{b} (\theta - g_j + \frac{b}{2}) \right)", "❌"))

        elif region == "region3":
            content_layout.addWidget(centered_image("A2.png"))

            content_layout.addWidget(section("Potentiel scalaire dans l'encoche statorique", 
                r"AIII_{z,i} (r, \theta) = C_{i,0} + \frac{1}{2} \mu_0 J_i R_4^2 \ln(r) - \frac{1}{4} \mu_0 J_i r^2 + \sum_{l=1}^{\infty} C_{i,l} h_l(r) \cos \left( \frac{l\pi}{c} (\theta - \beta_i + \frac{c}{2}) \right)", "🧾"))

            content_layout.addWidget(section("Champ radial", 
                r"BIII_r(r, \theta) = -\frac{1}{r} \sum_{l=1}^{\infty} \frac{l\pi}{c} C_{i,l} h_l(r) \sin \left( \frac{l\pi}{c} (\theta - \beta_i + \frac{c}{2}) \right)", "📐"))

        # Back button
                # BACK BUTTON
        layout.addSpacing(30)
        layout.addWidget(self.create_back_button(self.main))

#BOX STYLESHEET
    def create_objective_container(self, title, description, button_text, color, callback):
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #121924;
                border-radius: 16px;
                padding: 10px;
            }
        """)
        container.setFixedWidth(280)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel(f"<b style='color: white; font-size: 16px'>{title}</b>")
        desc_label = QLabel(f"<p style='color: #dcdcdc; font-size: 13px'>{description}</p>")
        desc_label.setWordWrap(True)

        button = QPushButton(button_text)
        button.setFixedHeight(40)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: bold;
                border-radius: 10px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #2c3e50;
            }}
        """)
        button.clicked.connect(callback)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()
        layout.addWidget(button)

        container.setLayout(layout)
        return container

#INPUTS ARE ALL DEFINED HERE
    def create_input_page(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(20)

        self.input_fields = {}

        def section_title(text):
            label = QLabel(text)
            label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: 600;
                    color: #0075c4;
                    margin-bottom: 8px;
                    font-family: 'Segoe UI', sans-serif;
                    border-bottom: 1px solid #2c3e50;
                    padding-bottom: 4px;
                }
            """)
            return label

        def add_input_row(label_text, key, default=""):
            container = QWidget()
            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            label = QLabel(label_text)
            label.setStyleSheet("color: #ffffff; font-size: 14px;")
            label.setFixedWidth(280)

            input_field = QLineEdit()
            input_field.setStyleSheet("""
                QLineEdit {
                    background-color: #1e2b38;
                    color: #ffffff;
                    border: 1px solid #3a4a5a;
                    border-radius: 6px;
                    padding: 6px;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 1px solid #00ffaa;
                    background-color: #273745;
                }
            """)
            input_field.setText(str(default))
            row_layout.addWidget(label)
            row_layout.addWidget(input_field)
            layout.addWidget(container)
            self.input_fields[key] = input_field

        # --- Sections ---
        layout.addWidget(section_title("1. PARAMÈTRES GÉOMÉTRIQUES"))
        add_input_row("R1 (Rayon interne rotor)", "R1", 0.038)
        add_input_row("R2 (Rayon externe rotor)", "R2", 0.060)
        add_input_row("R3 (Rayon interne stator)", "R3", 0.061)
        add_input_row("R4 (Rayon externe stator)", "R4", 0.085)
        add_input_row("R_EXT (Rayon extérieur machine)", "R_EXT", 0.1)
        add_input_row("G (Longueur d'entrefer)", "G", 1e-3)
        add_input_row("E (Épaisseur d'entrefer)", "E", 1e-3)

        layout.addWidget(section_title("2. PARAMÈTRES ÉLECTRIQUES"))
        add_input_row("slip", "slip", 0.0001)
        add_input_row("FREQ (Fréquence)", "FREQ", 50)
        add_input_row("MU_0 (Perméabilité du vide)", "MU_0", 4 * np.pi * 1e-7)
        add_input_row("MU_R (Perméabilité relative rotor)", "MU_R", 1)
        add_input_row("SIGMA (Conductivité saine)", "SIGMA", 35e6)
        add_input_row("SIGMA_DEFECT (Conductivité défectueuse)", "SIGMA_DEFECT", 0.1 * 35e6)
        add_input_row("SIGMA_BROKEN (Conductivité cassée)", "SIGMA_BROKEN", 1e-8)
        add_input_row("P (Nombre de paires de pôles)", "P", 2)

        layout.addWidget(section_title("3. PARAMÈTRES DE SIMULATION"))
        add_input_row("Ph (Nombre de phases)", "Ph", 3)
        add_input_row("Q_S (Nombre d'encoches statoriques)", "Q_S", 36)
        add_input_row("Q_R (Nombre des barres rotoriques)", "Q_R", 28)
        add_input_row("N_HARM (Harmoniques entrefer)", "N_HARM", 100)
        add_input_row("M_HARM (Harmoniques rotor)", "M_HARM", 3)
        add_input_row("L_HARM (Harmoniques stator)", "L_HARM", 3)

        layout.addWidget(section_title("4. AUTRES PARAMÈTRES"))
        add_input_row("N_C (Conducteurs par encoche statorique)", "N_C", 15)
        add_input_row("L_U (Longueur axiale)", "L_U", 0.2)
        add_input_row("V_m (Tension max par phase)", "V_m", 220)
        add_input_row("V1", "V1", 220)
        add_input_row("I_M (Courant statorique)", "I_M", 20)

        # Submit button
        submit_btn = QPushButton("📥 Enregistrer les Données")
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ec4b6;
                color: #ffffff;
                font-weight: bold;
                font-size: 15px;
                padding: 10px 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #00ddaa;
                color: #001d3d;
            }
        """)
        submit_btn.clicked.connect(lambda: self.save_manual_inputs(trigger_graph=True))

        layout.addWidget(submit_btn)
        goto_graph_btn = QPushButton("📊 Accéder aux Graphiques")
        goto_graph_btn.setStyleSheet("""
            QPushButton {
                background-color: #0075c4;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
        """)
        goto_graph_btn.clicked.connect(self.switch_to_graph_page)
        layout.addWidget(goto_graph_btn)


        # Back button
        layout.addStretch()
        layout.addWidget(self.create_back_button(self.main))

        scroll_content.setLayout(layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        return page

    def save_manual_inputs(self, trigger_graph=False):
        values = {key: field.text() for key, field in self.input_fields.items()}

        try:
            # Convert string values to correct types
            for k in values:
                try:
                    values[k] = float(values[k])
                except ValueError:
                    pass  # Keep string if not convertible

            # Fixed Q_R
            Q_R = int(values['Q_R']) 
            Q_S = int(values['Q_S'])  # Q_S comes from user input

            # Compute B and C
            B = 360 / Q_R * 0.5 * np.pi / 180
            C = 360 / Q_S * 0.5 * np.pi / 180
                    # === Save to inputs.py in same directory as main script ===
            script_dir = os.path.dirname(os.path.abspath(__file__))
            inputs_path = os.path.join(script_dir, "inputs.py")

            # Write to input.py
            with open(inputs_path, "w", encoding="utf-8") as f:
                f.write("import numpy as np\n\n")
                f.write("# ========== PARAMÈTRES GÉOMÉTRIQUES ==========\n")
                f.write(f"R1 = {values['R1']}\n")
                f.write(f"R2 = {values['R2']}\n")
                f.write(f"R3 = {values['R3']}\n")
                f.write(f"R4 = {values['R4']}\n")
                f.write(f"Rg = (R2 + R3) / 2\n")
                f.write(f"R_EXT = {values['R_EXT']}\n")
                f.write(f"G = {values['G']}\n")
                f.write(f"E = {values['E']}\n\n")

                f.write("# ========== PARAMÈTRES ÉLECTRIQUES ==========\n")
                f.write(f"slip = {values['slip']}\n")
                f.write(f"ss = 1 + 5 * (1 - slip)\n")
                f.write(f"FREQ = {values['FREQ']}\n")
                f.write(f"OMEGA_s = 2 * np.pi * FREQ\n")
                f.write(f"wrh = slip * OMEGA_s\n")
                f.write(f"fg = ss * FREQ\n")
                f.write(f"MU_0 = {values['MU_0']}\n")
                f.write(f"MU_R = {values['MU_R']}\n")
                f.write(f"SIGMA = {values['SIGMA']}\n")
                f.write(f"SIGMA_DEFECT = {values['SIGMA_DEFECT']}\n")
                f.write(f"SIGMA_BROKEN = {values['SIGMA_BROKEN']}\n")
                f.write(f"P = {int(values['P'])}\n\n")

                f.write("# ========== CONFIGURATION DES BARRES ROTORIQUES ==========\n")
                f.write(f"Q_R = {Q_R}\n")
                f.write("bar_status = np.full(Q_R, SIGMA)\n")
                f.write("bar_status[[0]] = SIGMA_BROKEN\n")
                f.write("bar_status[[5, 12]] = SIGMA_DEFECT\n")
                f.write("def get_sigma(j: int) -> float:\n")
                f.write("    assert 0 <= j < Q_R, f\"Indice de barre invalide : {j}\"\n")
                f.write("    return bar_status[j]\n\n")

                f.write("# ========== PARAMÈTRES DE SIMULATION ==========\n")
                f.write("g = np.linspace(0.1, 1, 50)\n")
                f.write(f"Ph = {int(values['Ph'])}\n")
                f.write(f"Q_S = {Q_S}\n")
                f.write(f"N_HARM = {int(values['N_HARM'])}\n")
                f.write(f"M_HARM = {int(values['M_HARM'])}\n")
                f.write(f"L_HARM = {int(values['L_HARM'])}\n\n")

                f.write("# ========== AUTRES PARAMÈTRES ==========\n")
                f.write(f"N_C = {int(values['N_C'])}\n")
                f.write(f"L_U = {values['L_U']}\n")
                f.write(f"V_m = {values['V_m']}\n")
                f.write(f"V1 = {values['V1']}\n")
                f.write(f"I_M = {values['I_M']}\n\n")

                f.write("# ========== PARAMÈTRES ANGULAIRES DÉDUITS ==========\n")
                f.write(f"B = {B}\n")
                f.write(f"C = {C}\n")
            self.data_saved = True  # <-- flag to trigger graph later
            print("✅ Paramètres enregistrés avec succès dans input.py")
            self.inputs_submitted = True  # ✅ Mark inputs as submitted
            # Enable main page buttons after successful input
            self.analyse_box.setEnabled(True)
            self.fonctionnement_box.setEnabled(True)


        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")
 
    def switch_to_graph_page(self):
        if self.inputs_submitted:
            self.stacked_widget.setCurrentWidget(self.analytic_graph_page)
            # self.stacked_widget.setCurrentWidget(self. nominal_param_page)
        else:
            QMessageBox.warning(self, "Entrée requise", "Veuillez d'abord enregistrer les données d'entrée.")

    def create_analytic_graph_page(self):
        class GraphCanvas(FigureCanvas):
            def __init__(self, title):
                self.title = title
                self.fig = Figure(figsize=(4.5, 2.5), dpi=100)
                super().__init__(self.fig)
                self.ax = self.fig.add_subplot(111)

            def plot_graph(self):
                plot_to_ax(self.ax, self.title)
                self.draw()

        def create_graph_card(title, description="Aperçu graphique"):
            container = QFrame()
            container.setStyleSheet("""
                QFrame {
                    background-color: #121924;
                    border-radius: 16px;
                    padding: 12px;
                }
                QFrame:hover {
                    border: 2px solid #0075c4;
                    border-radius: 10px;
                }
            """)
            container.setFixedWidth(400)
            container.setCursor(Qt.CursorShape.PointingHandCursor)

            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            title_label = QLabel(f"<b style='color: white; font-size: 16px'>{title}</b>")
            desc_label = QLabel(f"<p style='color: #dcdcdc; font-size: 13px'>{description}</p>")
            desc_label.setWordWrap(True)

            canvas = GraphCanvas(title)
            canvas.setMinimumHeight(250)
            toolbar = NavigationToolbar(canvas, container)

            plot_btn = QPushButton("Afficher les graphes")
            plot_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0075c4;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #005fa3;
                }
            """)

            def refresh_canvas():
                canvas.plot_graph()

            plot_btn.clicked.connect(refresh_canvas)

            def open_fullscreen_dialog():
                dialog = QDialog()
                dialog.setWindowTitle(title)
                dialog.setStyleSheet("background-color: #e2e2e2;")
                dialog.setMinimumSize(900, 600)
                dialog_layout = QVBoxLayout(dialog)

                full_canvas = GraphCanvas(title)
                full_canvas.setMinimumHeight(550)
                full_toolbar = NavigationToolbar(full_canvas, dialog)

                dialog_layout.addWidget(full_toolbar)
                dialog_layout.addWidget(full_canvas)
                full_canvas.plot_graph()
                dialog.exec()

            container.mousePressEvent = lambda event: open_fullscreen_dialog()

            layout.addWidget(title_label)
            layout.addWidget(desc_label)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            layout.addWidget(plot_btn)
            layout.addStretch()
            container.setLayout(layout)

            return container

        # === Main Page Layout ===
        page = QWidget()
        main_layout = QVBoxLayout(page)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        grid = QGridLayout(scroll_content)
        grid.setSpacing(20)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        titles = [
            ("Induction à vide - Radiale", "Induction radiale dans l’entrefer à vide (g ≈ 0)."),
            ("Induction à vide - Tangentielle", "Induction tangentielle dans l’entrefer à vide (g ≈ 0)."),
            ("Induction à rotor bloqué - Radiale", "Induction radiale  dans l’entrefer à rotor bloqué (g = 1)."),
            ("Induction à rotor bloqué - Tangentielle", "Induction tangentielle  dans l’entrefer à rotor bloqué (g = 1)."),
            ("Densité de courant dans une barre", "Densité de courant en fonction du rayon (barre saine/défectueuse/cassée)."),
            ("Courant dans les barres à g=1", "Courant dans les barres rotoriques à rotor bloqué (g = 1)."),
            ("inductance ramenée au rotor en fonction du glissement", "Variation de l'inductance ramenee au  rotor selon le glissement."),
            ("Résistance ramenée au rotor en fonction du glissement", "Variation de la résistance ramenee au  rotor selon le glissement."),
            ("Résistance statorique en fonction du glissement", "Variation de Résistance statorique selon le glissement."),
            ("Courant statorique vs glissement", "Variation du courant statorique selon le glissement."),
            ("Couple électromagnétique vs glissement", "Variation du couple électromagnétique en fonction du glissement."),
        ]

        positions = [(i // 3, i % 3) for i in range(len(titles))]
        for pos, (title, desc) in zip(positions, titles):
            grid.addWidget(create_graph_card(title, desc), *pos)

        main_layout.addWidget(self.create_back_button(self.main))
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        return page

    def create_nominal_param_page(self):
        page = QWidget()
        self.nominal_labels = {}  # Store references to all value labels in a dict

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("Paramètres Caractéristiques – Schéma Équivalent")
        title.setStyleSheet("font-size: 24px; font-weight: 600; color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Machine asynchrone à cage")
        subtitle.setStyleSheet("font-size: 14px; color: #bdc3c7;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(30)

        # === Section 1: Schéma équivalent ===
        main_layout.addWidget(self._section_label("⚙️ Paramètres du schéma équivalent"))
        for label_key, description in [
            ("rs", "Résistance statorique"),
            ("xm", "Réactance magnétisante"),
            ("rr", "Résistance ramenée au rotor"),
            ("xr", "Réactance ramenée au rotor")
        ]:
            self.nominal_labels[label_key] = QLabel("...")
            main_layout.addLayout(self._create_line_item(description, self.nominal_labels[label_key]))

        # === Section 2: Résultats calculés ===
        main_layout.addSpacing(30)
        main_layout.addWidget(self._section_label("🔍 Résultats électriques calculés"))
        for label_key, description in [
            ("ptr", "Puissance active transmise"),
            ("str", "Puissance apparente transmise"),
            ("ttr", "Couple électromagnétique"),
            ("i1", "Courant statorique")
        ]:
            self.nominal_labels[label_key] = QLabel("...")
            main_layout.addLayout(self._create_line_item(description, self.nominal_labels[label_key]))

        # === Calculer Button ===
        main_layout.addSpacing(20)
        calc_button = QPushButton("Calculer")
        calc_button.setStyleSheet("""
            QPushButton {
                background-color: #0075c4;
                color: white;
                padding: 8px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
        """)
        calc_button.clicked.connect(self._calculate_nominal_params)
        main_layout.addWidget(calc_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Back Button
        main_layout.addSpacing(30)
        main_layout.addWidget(self.create_back_button(self.main))

        page.setLayout(main_layout)
        return page
    def _section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        return label

    def _calculate_nominal_params(self):
        print("🟡 Bouton 'Calculer' cliqué")

        try:
            # Call your actual computation functions
            S_tr, P_tr, T_tr, _, I1, Rs, N2_prime, R2_prime = nominale()
            Xm = calculate_Lm()

            # Format helper
            def fmt(val):
                if isinstance(val, np.ndarray):
                    val = val.item()
                return f"{val:.3f}"

            # Fill in all value labels
            self.nominal_labels["rs"].setText(f"{fmt(Rs)} Ω")
            self.nominal_labels["xm"].setText(f"{fmt(Xm)} Ω")
            self.nominal_labels["rr"].setText(f"{fmt(R2_prime)} Ω")
            self.nominal_labels["xr"].setText(f"{fmt(N2_prime)} H")

            self.nominal_labels["ptr"].setText(f"{fmt(P_tr)} W")
            self.nominal_labels["str"].setText(f"{fmt(abs(S_tr))} VA")
            self.nominal_labels["ttr"].setText(f"{fmt(T_tr)} Nm")
            self.nominal_labels["i1"].setText(f"{fmt(abs(I1))} A")

        except Exception as e:
            print(f"❌ Erreur de calcul : {e}")

    def _h_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #34495e;")
        return line


#STYLESHEET
    def _create_line_item(self, label_text, value_text):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)

        label = QLabel(label_text)
        label.setStyleSheet("font-size: 14px; color: #ecf0f1; font-weight: 500;")
        value = QLabel(value_text)
        value.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        return layout

#BOUTON DE RETOUR
    def create_back_button(self, target_page):
        back_btn = QPushButton(" Back")
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3590f3 ;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                padding:8px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #2c3e50;
            }}
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(target_page))

        return back_btn



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



