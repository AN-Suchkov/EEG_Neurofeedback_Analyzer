import sys
import os
import numpy as np
import pandas as pd
import joblib
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QWidget, QGroupBox, QTextEdit, QProgressBar,
                             QMessageBox)
from PyQt5.QtCore import QTimer
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score


class EEGAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.test_data = None
        self.test_labels = None
        self.feature_names = None
        self.target_names = None
        self.current_test_index = 0
        self.is_real_sensor_mode = False

        # таймер
        self.sensor_timer = QTimer()
        self.sensor_timer.timeout.connect(self.update_sensor_data)
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.auto_play_next)

        self.label_mapping = {
            'NEGATIVE': 'Стресс',
            'NEUTRAL': 'Концентрация',
            'POSITIVE': 'Расслабленность'
        }

        self.initUI()
        self.load_model()
        self.load_test_data()

    def initUI(self):
        # Окно
        self.setWindowTitle('Анализатор ЭЭГ сигналов')
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        connection_group = QGroupBox("Настройки подключения")
        connection_layout = QHBoxLayout()

        self.connection_combo = QComboBox()
        self.connection_combo.addItems(["Тестовые данные", "Реальный датчик"])
        self.connection_combo.currentTextChanged.connect(self.connection_changed)

        self.connect_btn = QPushButton("Подключить")
        self.connect_btn.clicked.connect(self.toggle_connection)

        connection_layout.addWidget(QLabel("Режим работы:"))
        connection_layout.addWidget(self.connection_combo)
        connection_layout.addWidget(self.connect_btn)
        connection_layout.addStretch()
        connection_group.setLayout(connection_layout)

        status_group = QGroupBox("Состояние системы")
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Не подключено")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)

        # визуализация сигнала
        visualization_group = QGroupBox("Визуализация сигнала")
        visualization_layout = QVBoxLayout()

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        visualization_layout.addWidget(self.canvas)

        visualization_group.setLayout(visualization_layout)

        results_group = QGroupBox("Результаты классификации")
        results_layout = QVBoxLayout()

        self.prediction_label = QLabel("Предсказание: -")
        self.prediction_label.setStyleSheet("font-size: 16pt; font-weight: bold;")

        self.confidence_label = QLabel("Уверенность: -")
        self.confidence_label.setStyleSheet("font-size: 12pt;")

        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)

        results_layout.addWidget(self.prediction_label)
        results_layout.addWidget(self.confidence_label)
        results_layout.addWidget(self.details_text)
        results_group.setLayout(results_layout)

        test_controls_group = QGroupBox("Управление тестовыми данными")
        test_controls_layout = QHBoxLayout()

        self.prev_btn = QPushButton("Предыдущий")
        self.prev_btn.clicked.connect(self.prev_test_data)

        self.next_btn = QPushButton("Следующий")
        self.next_btn.clicked.connect(self.next_test_data)

        self.auto_play_btn = QPushButton("Автовоспроизведение")
        self.auto_play_btn.setCheckable(True)
        self.auto_play_btn.toggled.connect(self.toggle_auto_play)

        self.test_accuracy_btn = QPushButton("Проверить точность")
        self.test_accuracy_btn.clicked.connect(self.check_accuracy)

        test_controls_layout.addWidget(self.prev_btn)
        test_controls_layout.addWidget(self.next_btn)
        test_controls_layout.addWidget(self.auto_play_btn)
        test_controls_layout.addWidget(self.test_accuracy_btn)
        test_controls_layout.addStretch()
        test_controls_group.setLayout(test_controls_layout)

        main_layout.addWidget(connection_group)
        main_layout.addWidget(status_group)
        main_layout.addWidget(visualization_group)
        main_layout.addWidget(results_group)
        main_layout.addWidget(test_controls_group)

        # Обновление доступности элементов управления
        self.update_controls()

    def load_model(self):
        # Загрузка обученной модели
        try:
            model_path = os.path.join('models', 'random_forest_model_3_classes.pkl')
            encoder_path = os.path.join('models', 'label_encoder_3_classes.pkl')

            if os.path.exists(model_path) and os.path.exists(encoder_path):
                self.model = joblib.load(model_path)
                self.encoder = joblib.load(encoder_path)
                self.status_label.setText("Модель загружена успешно")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
            else:
                self.status_label.setText("Файлы модели не найдены")
        except Exception as e:
            self.status_label.setText(f"Ошибка загрузки модели: {str(e)}")

    def load_test_data(self):
        try:
            test_data_path = os.path.join('models', 'test_data.pkl')

            if os.path.exists(test_data_path):
                test_data_dict = joblib.load(test_data_path)

                self.test_data = test_data_dict['X_test']
                self.test_labels = test_data_dict['y_test']
                self.feature_names = test_data_dict['feature_names']
                self.target_names = test_data_dict['target_names']

                self.status_label.setText(f"Тестовая выборка загружена: {len(self.test_data)} примеров")
                self.current_test_index = 0
                self.update_display()
            else:
                self.status_label.setText("Тестовая выборка не найдена")
        except Exception as e:
            self.status_label.setText(f"Ошибка загрузки тестовых данных: {str(e)}")

    def connection_changed(self, mode):
        self.is_real_sensor_mode = (mode == "Реальный датчик")
        self.update_controls()

        if self.is_real_sensor_mode:
            self.status_label.setText("Режим реального датчика выбран")
        else:
            self.status_label.setText("Режим тестовых данных выбран")
            self.stop_sensor()

    def toggle_connection(self):
        if self.is_real_sensor_mode:
            if self.sensor_timer.isActive():
                self.stop_sensor()
                self.connect_btn.setText("Подключить")
            else:
                self.start_sensor()
                self.connect_btn.setText("Отключить")
        else:
            self.update_display()

    def start_sensor(self):
        # Запуск датчика
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.sensor_timer.start(1000)

    def stop_sensor(self):
        # остановка
        self.sensor_timer.stop()
        self.progress_bar.setVisible(False)

    def update_sensor_data(self):
        # Обновление данных с датчика (***заглушка***); Нужен специальный датчик
        if self.model and self.feature_names is not None:
            n_features = len(self.feature_names)
            synthetic_data = np.random.rand(1, n_features)

            synthetic_df = pd.DataFrame(synthetic_data, columns=self.feature_names)

            prediction = self.model.predict(synthetic_df)
            prediction_proba = self.model.predict_proba(synthetic_df)

            english_label = self.target_names[prediction[0]]
            predicted_label = self.label_mapping.get(english_label, english_label)
            confidence = np.max(prediction_proba) * 100

            self.update_prediction_display(predicted_label, confidence, prediction_proba[0])
            self.visualize_signal(synthetic_data[0][:50])

    def update_display(self):
        # Обновление отображения тестовых данных
        if (self.test_data is None or self.model is None or
                self.feature_names is None or self.target_names is None or
                self.test_labels is None):
            return

        if self.current_test_index >= len(self.test_data) or self.current_test_index < 0:
            return

        try:
            current_data = self.test_data.iloc[self.current_test_index].values.reshape(1, -1)

            if current_data.shape[1] != self.model.n_features_in_:
                return

            current_df = pd.DataFrame(current_data, columns=self.feature_names)

            true_label_idx = self.test_labels.iloc[self.current_test_index]

            # translation
            english_true_label = self.target_names[true_label_idx]
            true_label_text = self.label_mapping.get(english_true_label, english_true_label)

            prediction = self.model.predict(current_df)
            prediction_proba = self.model.predict_proba(current_df)

            english_predicted_label = self.target_names[prediction[0]]
            predicted_label = self.label_mapping.get(english_predicted_label, english_predicted_label)
            confidence = np.max(prediction_proba) * 100

            self.update_prediction_display(predicted_label, confidence, prediction_proba[0], true_label_text)
            self.visualize_signal(current_data[0][:50])

            self.status_label.setText(f"Тестовые данные: {self.current_test_index + 1}/{len(self.test_data)}")
        except Exception as e:
            print(f"Ошибка в update_display: {str(e)}")

    def update_prediction_display(self, prediction, confidence, probabilities, true_label=None):
        # результаты предсказаний
        self.prediction_label.setText(f"Предсказание: {prediction}")

        if true_label:
            color = "green" if prediction == true_label else "red"
            self.prediction_label.setStyleSheet(f"font-size: 16pt; font-weight: bold; color: {color};")
            self.confidence_label.setText(f"Уверенность: {confidence:.2f}% (Истинный класс: {true_label})")
        else:
            self.prediction_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: blue;")
            self.confidence_label.setText(f"Уверенность: {confidence:.2f}%")

        # вероятность
        details = "Вероятности по классам:\n"
        for i, class_name in enumerate(self.target_names):
            russian_name = self.label_mapping.get(class_name, class_name)
            details += f"{russian_name}: {probabilities[i] * 100:.2f}%\n"

        self.details_text.setPlainText(details)

    def visualize_signal(self, signal_data):
        # Визуализация
        try:
            self.figure.clear()

            plot_data = signal_data[:50]

            ax = self.figure.add_subplot(111)
            ax.plot(plot_data)
            ax.set_title('Визуализация сигнала ЭЭГ')
            ax.set_xlabel('Время (отсчеты)')
            ax.set_ylabel('Амплитуда')
            ax.grid(True)

            self.canvas.draw()
        except Exception as e:
            print(f"Ошибка визуализации: {e}")

    def prev_test_data(self):
        if self.test_data is not None and self.current_test_index > 0:
            self.current_test_index -= 1
            self.update_display()

    def next_test_data(self):
        # Переход к следующим тестовым данным
        if self.test_data is not None and self.current_test_index < len(self.test_data) - 1:
            self.current_test_index += 1
            self.update_display()

    def toggle_auto_play(self, checked):
        # Включение/выключение автовоспроизведения
        if checked:
            self.auto_play_timer.start(2000)
            self.auto_play_btn.setText("Остановить")
        else:
            self.auto_play_timer.stop()
            self.auto_play_btn.setText("Автовоспроизведение")

    def auto_play_next(self):
        # Автоматический переход
        if self.test_data is not None:
            self.current_test_index = (self.current_test_index + 1) % len(self.test_data)
            self.update_display()

    def check_accuracy(self):
        # Проверка точности модели на тестовой выборке
        if self.test_data is not None and self.model is not None:
            test_df = pd.DataFrame(self.test_data, columns=self.feature_names)
            predictions = self.model.predict(test_df)
            accuracy = accuracy_score(self.test_labels, predictions)

            msg = QMessageBox()
            msg.setWindowTitle("Точность модели")
            msg.setText(f"Точность на тестовой выборке: {accuracy:.4f}\n"
                        f"Размер выборки: {len(self.test_data)} примеров")
            msg.exec_()

    def update_controls(self):
        # Обновление доступности элементов управления
        is_test_mode = not self.is_real_sensor_mode

        self.prev_btn.setEnabled(is_test_mode)
        self.next_btn.setEnabled(is_test_mode)
        self.auto_play_btn.setEnabled(is_test_mode)
        self.test_accuracy_btn.setEnabled(is_test_mode)

        if self.is_real_sensor_mode:
            self.connect_btn.setText("Подключить" if not self.sensor_timer.isActive() else "Отключить")
        else:
            self.connect_btn.setText("Обновить")

    def closeEvent(self, event):
        # Обработчик закрытия приложения
        self.stop_sensor()
        if self.auto_play_timer.isActive():
            self.auto_play_timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = EEGAnalyzerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()