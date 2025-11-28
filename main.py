import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets


class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    finished_signal = QtCore.pyqtSignal()
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self, video_path: str, model_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self._running = True
        self._paused = False

    def run(self):

        self.status_signal.emit("Загрузка модели...")
        device = 0 if torch.cuda.is_available() else "cpu"

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.status_signal.emit(f"Ошибка загрузки модели: {e}")
            self.finished_signal.emit()
            return

        self.status_signal.emit("Модель загружена ✔")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_signal.emit("Не удалось открыть видеофайл.")
            self.finished_signal.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_time = 1.0 / fps
        self.status_signal.emit(f"FPS исходного видео: {fps:.2f}")


        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break


            try:
                results = model.predict(frame, conf=0.35, device=device, verbose=False)
                annotated = results[0].plot()
            except Exception as e:
                self.status_signal.emit(f"Ошибка обработки кадра: {e}")
                continue

            self.frame_ready.emit(annotated)

            time.sleep(frame_time)

        cap.release()
        self.status_signal.emit("Обработка завершена.")
        self.finished_signal.emit()

    def stop(self):
        self._running = False

    def pause(self, p: bool):
        self._paused = p


class DroneDetectorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Система распознавания дронов")
        self.setMinimumSize(1000, 650)
        self._worker = None

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)


        video_row = QtWidgets.QHBoxLayout()
        self.video_edit = QtWidgets.QLineEdit()
        self.video_edit.setPlaceholderText("Путь к видеофайлу...")
        self.video_btn = QtWidgets.QPushButton("Обзор")
        video_row.addWidget(self.video_edit)
        video_row.addWidget(self.video_btn)
        layout.addLayout(video_row)


        model_row = QtWidgets.QHBoxLayout()
        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("Путь к модели YOLO (.pt)")
        self.model_btn = QtWidgets.QPushButton("Обзор")
        model_row.addWidget(self.model_edit)
        model_row.addWidget(self.model_btn)
        layout.addLayout(model_row)


        btn_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Запустить")
        self.pause_btn = QtWidgets.QPushButton("Пауза")
        self.stop_btn = QtWidgets.QPushButton("Остановить")

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)


        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(800, 450)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setText("Видео не загружено")
        layout.addWidget(self.video_label)

        self.info = QtWidgets.QTextEdit()
        self.info.setReadOnly(True)
        layout.addWidget(self.info)

        # Строка состояния
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self.video_btn.clicked.connect(self.browse_video)
        self.model_btn.clicked.connect(self.browse_model)
        self.run_btn.clicked.connect(self.on_run)
        self.pause_btn.clicked.connect(self.on_pause)
        self.stop_btn.clicked.connect(self.on_stop)

    def apply_styles(self):

        style = """
        QMainWindow {
            background: #F5F8FF;
        }
        QPushButton {
            background-color: #0B5FFF;
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: 600;
        }
        QPushButton:disabled {
            background-color: #AFC7FF;
            color: #E8E8E8;
        }
        QLineEdit {
            background: white;
            border: 1px solid #C9DFFF;
            border-radius: 6px;
            padding: 5px;
        }
        QTextEdit {
            background: white;
            border: 1px solid #C9DFFF;
            border-radius: 6px;
            padding: 5px;
        }
        QLabel {
            color: #0B2A66;
            font-weight: 600;
        }
        """
        self.setStyleSheet(style)

    def browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите видео", str(Path.home()),
            "Видео (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_edit.setText(path)
            self.info.append(f"Выбрано видео: {path}")

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите модель YOLO", str(Path.home()),
            "YOLO model (*.pt)")
        if path:
            self.model_edit.setText(path)
            self.info.append(f"Выбрана модель: {path}")

    def on_run(self):
        video_path = self.video_edit.text().strip()
        model_path = self.model_edit.text().strip()

        if not Path(video_path).exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Видео не найдено.")
            return

        if not Path(model_path).exists():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Файл модели не найден.")
            return

        self.info.append("▶ Запуск обработки...")
        self.status.showMessage("Запуск...")

        self.run_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.model_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self._worker = VideoWorker(video_path, model_path)
        self._worker.frame_ready.connect(self.update_frame)
        self._worker.finished_signal.connect(self.on_worker_finished)
        self._worker.status_signal.connect(self.set_status)
        self._worker.start()


    def on_pause(self):
        if not self._worker:
            return

        if self.pause_btn.text() == "Пауза":
            self._worker.pause(True)
            self.pause_btn.setText("Продолжить")
            self.status.showMessage("Пауза...")
        else:
            self._worker.pause(False)
            self.pause_btn.setText("Пауза")
            self.status.showMessage("Продолжение воспроизведения")


    def on_stop(self):
        if self._worker:
            self._worker.stop()
            self.status.showMessage("Остановка...")


    def on_worker_finished(self):
        self.info.append("✔ Обработка завершена.")

        self.run_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.model_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Пауза")

        self._worker = None

    @QtCore.pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled)

    def set_status(self, msg: str):
        self.status.showMessage(msg)
        self.info.append(msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = DroneDetectorGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
