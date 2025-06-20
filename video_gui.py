import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QHBoxLayout, QSlider, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QMovie, QIcon, QPixmap
# Ensure you have the necessary imports for your video processing 

# Import your processing functions
from detect_mkrs_bkg_grid_blobs_2 import calibrate_markers, process_video
# Constants for file paths
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

        # --- Media controls layout ---
        media_controls_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        media_controls_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_video)
        media_controls_layout.addWidget(self.pause_button)

        # Playback speed combo box
        self.speed_box = QComboBox()
        self.speed_box.addItems(["0.1x", "0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_box.setCurrentIndex(3)  # Default to 1x
        self.speed_box.currentIndexChanged.connect(self.change_speed)
        # --- Update: Use a named label for styling ---
        self.speed_label = QLabel("Playback Speed:")
        self.speed_label.setObjectName("playbackSpeedLabel")
        media_controls_layout.addWidget(self.speed_label)
        media_controls_layout.addWidget(self.speed_box)

        layout.addLayout(media_controls_layout)

        # --- Video slider ---
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        layout.addWidget(self.position_slider)

        # Video player widgets
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Connect signals for slider and state
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.update_play_pause_buttons)

        self.setLayout(layout)

        # Set blue/light blue theme
        self.setStyleSheet("""
            QWidget {
                background-color: #e3f0fc; /* light blue */
            }
            QPushButton {
                background-color: #1976d2; /* blue */
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 15px;
            }
            QPushButton:disabled {
                background-color: #90caf9; /* lighter blue for disabled */
                color: #e3f0fc;
            }
            QPushButton:hover {
                background-color: #1565c0; /* darker blue on hover */
            }
            QVideoWidget {
                border: 2px solid #1976d2;
                border-radius: 8px;
                background-color: #bbdefb; /* very light blue */
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #bbdefb;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1976d2;
                border: 1px solid #1565c0;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox, QComboBox QAbstractItemView {
                color: black;
                background: #bbdefb;
                selection-background-color: #90caf9;
                selection-color: black;
                font-size: 15px;
            }
            QLabel#playbackSpeedLabel {
                color: #1565c0; /* dark blue */
                font-weight: bold;
                font-size: 15px;
            }
        """)

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
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(True)
        else:
            print("Output video not found.")

    # --- Media player controls ---
    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def change_speed(self):
        speeds = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
        self.media_player.setPlaybackRate(speeds[self.speed_box.currentIndex()])

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def update_play_pause_buttons(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState or state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec())
