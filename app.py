"""
Bio-Oracle Application Launcher
Run this file to start the GUI application
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import BioOracleWindow


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Bio-Oracle")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Bio-Oracle Project")
    
    # Create and show main window
    window = BioOracleWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
