import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QHBoxLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QMovie, QIcon, QPixmap

# Import your processing functions
from detect_mkrs_bkg_grid_blobs_2 import calibrate_markers, process_video

OUTPUT_VIDEO = "rugby_analysis_output_2.mp4"

class VideoProcessorThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.log_signal.emit("Calibrating markers...")
            calibration_data = calibrate_markers(self.video_path)
            self.log_signal.emit("Processing video...")
            process_video(self.video_path, calibration_data)
            self.log_signal.emit("Processing complete!")
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
        self.finished_signal.emit()

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rugby Video Processor")
        self.resize(700, 600)
        self.video_path = None

        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_button)

        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)

        self.view_button = QPushButton("View Output Video")
        self.view_button.clicked.connect(self.view_output_video)
        self.view_button.setEnabled(False)
        button_layout.addWidget(self.view_button)

        layout.addLayout(button_layout)

        # Video player widgets
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.setLayout(layout)

    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.process_button.setText("Process Video")  # Reset text
            self.process_button.setEnabled(True)
            self.view_button.setEnabled(False)
        else:
            self.process_button.setText("Process Video")  # Reset text
            self.process_button.setEnabled(False)
            self.view_button.setEnabled(False)

    def process_video(self):
        if not self.video_path:
            return
        self.process_button.setText("Processing Video...")
        self.process_button.setEnabled(False)
        self.view_button.setEnabled(False)
        self.thread = VideoProcessorThread(self.video_path)
        self.thread.log_signal.connect(self.handle_log)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.start()

    def handle_log(self, message):
        print(message)

    def on_processing_finished(self):
        self.process_button.setText("Done")
        self.process_button.setEnabled(True)
        if os.path.exists(OUTPUT_VIDEO):
            self.view_button.setEnabled(True)
        else:
            print("Output video not found.")

    def view_output_video(self):
        if os.path.exists(OUTPUT_VIDEO):
            self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(OUTPUT_VIDEO)))
            self.media_player.play()
        else:
            print("Output video not found.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
