import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QHBoxLayout, QSlider, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QMovie, QIcon, QPixmap
import shutil  # <-- for saving the video
# Ensure you have the necessary imports for your video processing 

# Import your processing functions
from detect_mkrs_bkg_grid_blobs_2 import calibrate_markers, process_video
# Constants for file paths
OUTPUT_VIDEO = "rugby_analysis_output_2.mp4"

class VideoProcessorThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)  # Pass the output directory path

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.log_signal.emit("Calibrating markers...")      
            calibration_data = calibrate_markers(self.video_path)
            self.log_signal.emit("Processing video...")
            process_video(self.video_path, calibration_data)
            
            # Get the output directory path
            from detect_mkrs_bkg_grid_blobs_2 import OUTPUT_DIRS
            output_dir = OUTPUT_DIRS['base'] if OUTPUT_DIRS else "Unknown"
            
            self.log_signal.emit("Processing complete!")
            self.finished_signal.emit(output_dir)
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit("")

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rugby Video Processor")
        self.resize(700, 600)
        self.video_path = None
        self.output_directory = None  # Store the output directory path

        layout = QVBoxLayout()

        # --- Top buttons ---
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

        self.save_button = QPushButton("Save Video")
        self.save_button.clicked.connect(self.save_video)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # --- Video player widget ---
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        layout.addWidget(self.video_widget)

        # --- Controls under video ---
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        controls_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_video)
        controls_layout.addWidget(self.pause_button)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)

        layout.addLayout(controls_layout)

        # --- Playback speed under video ---
        speed_layout = QHBoxLayout()
        self.speed_label = QLabel("Playback Speed:")
        self.speed_label.setObjectName("playbackSpeedLabel")
        speed_layout.addWidget(self.speed_label)
        self.speed_box = QComboBox()
        self.speed_box.addItems(["0.1x", "0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_box.setCurrentIndex(3)  # Default to 1x
        self.speed_box.currentIndexChanged.connect(self.change_speed)
        speed_layout.addWidget(self.speed_box)
        speed_layout.addStretch()
        layout.addLayout(speed_layout)

        # --- Media player setup ---
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
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
            self.process_button.setText("Process Video")
            self.process_button.setEnabled(True)
            self.view_button.setEnabled(False)
            self.save_button.setEnabled(False)
        else:
            self.process_button.setText("Process Video")
            self.process_button.setEnabled(False)
            self.view_button.setEnabled(False)
            self.save_button.setEnabled(False)

    def process_video(self):
        if not self.video_path:
            return
        self.process_button.setText("Processing Video...")
        self.process_button.setEnabled(False)
        self.view_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.thread = VideoProcessorThread(self.video_path)
        self.thread.log_signal.connect(self.handle_log)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.start()

    def handle_log(self, message):
        print(message)

    def on_processing_finished(self, output_dir):
        self.process_button.setText("Done")
        self.process_button.setEnabled(True)
        self.output_directory = output_dir
        
        # Check for output video in the new directory structure
        if output_dir and os.path.exists(output_dir):
            video_path = os.path.join(output_dir, 'videos', 'rugby_analysis_output_2.mp4')
            if os.path.exists(video_path):
                self.view_button.setEnabled(True)
                self.save_button.setEnabled(True)
                # Update the global OUTPUT_VIDEO path for the GUI
                global OUTPUT_VIDEO
                OUTPUT_VIDEO = video_path
            else:
                print("Output video not found in expected location.")
        else:
            print("Output directory not found.")

    def view_output_video(self):
        if self.output_directory:
            video_path = os.path.join(self.output_directory, 'videos', 'rugby_analysis_output_2.mp4')
            if os.path.exists(video_path):
                self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
                self.media_player.play()
                self.play_button.setEnabled(True)
                self.pause_button.setEnabled(True)
            else:
                print("Output video not found.")
        else:
            print("No output directory available.")

    def save_video(self):
        if not self.output_directory:
            return
            
        source_video = os.path.join(self.output_directory, 'videos', 'rugby_analysis_output_2.mp4')
        if not os.path.exists(source_video):
            return
            
        # Suggest a filename based on the original video
        if self.video_path:
            base, ext = os.path.splitext(os.path.basename(self.video_path))
            suggested = f"{base}_processed{ext}"
        else:
            suggested = "processed_video.mp4"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Video", suggested, "Video Files (*.mp4 *.avi *.mov)")
        if save_path:
            try:
                shutil.copyfile(source_video, save_path)
            except Exception as e:
                print(f"Error saving video: {e}")

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
