#  pyuic5 -x gui/gui.ui -o gui/gui.py   
import sys
from PyQt5.QtWidgets import QApplication
from qt_processing.meeting_summarizer_gui import MainWindow

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
