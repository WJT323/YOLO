# -*- coding: utf-8 -*-
"""
道路病害检测系统 - PyQt5 主程序 (单窗口大屏展示版)
请将本文件与 win1.py 放在同一目录。
"""

import os
import sys
import traceback

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from PyQt5 import QtCore, QtGui, QtWidgets

# 引入你的 UI 文件
from win1 import Ui_MainWindow

# YOLOv5 原生组件
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# ======================== 配置 ========================
MODEL_PATH = os.path.join("runs", "train", "exp13", "weights", "best_fixed.pt")
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1000
USE_HALF = False


class RoadDiseaseWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.current_result = None
        self.current_image_path = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.stride = 32
        self.names = []

        self._setup_window()
        self._setup_menu()
        self._setup_connections()
        self._setup_model()

    def _setup_window(self):
        self.setWindowTitle("道路病害智能检测系统 - 大屏展示端")
        self.setMinimumSize(1000, 700)
        self.resize(1280, 850)

        # ================= 【终极修复】顶部标题 =================
        # 隐藏原本那个总是截断的 QLineEdit
        self.ui.lineEdit.hide()

        # 动态创建一个 QLabel 替代它
        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtCore import Qt

        self.custom_title = QLabel("道路病害智能检测系统")
        self.custom_title.setAlignment(Qt.AlignCenter)
        self.custom_title.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #172b4d;
                padding: 10px;
            }
        """)
        # 把它插入到原来的头部布局里
        self.ui.headerLayout.insertWidget(0, self.custom_title)
        # ==========================================================

        # 图像显示区样式（只针对单图显示 label_2）
        self.ui.label_2.setText("请点击下方按钮上传道路图片")
        self.ui.label_2.setAlignment(Qt.AlignCenter)
        self.ui.label_2.setStyleSheet("""
            QLabel {
                background-color: #101827;
                color: #9fb3c8;
                border: 2px dashed #42536b;
                border-radius: 12px;
                padding: 8px;
                font-size: 18px;
            }
        """)

        self.ui.pushButton.setText("选择图片并开始检测")
        self.ui.pushButton.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.ui.statusbar.showMessage("正在初始化检测系统...")

    def _setup_menu(self):
        file_menu = self.ui.menubar.addMenu("文件")

        open_action = QtWidgets.QAction("打开图片", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QtWidgets.QAction("保存检测结果", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QtWidgets.QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.ui.menubar.addMenu("帮助")
        about_action = QtWidgets.QAction("关于系统", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _setup_connections(self):
        self.ui.pushButton.clicked.connect(self.open_image)

    def _setup_model(self):
        if not os.path.isfile(MODEL_PATH):
            QtWidgets.QMessageBox.critical(self, "错误", f"找不到模型文件：\n{MODEL_PATH}")
            self.ui.pushButton.setEnabled(False)
            return

        try:
            self.ui.statusbar.showMessage("正在加载 YOLOv5 模型...")
            QtWidgets.QApplication.processEvents()
            self.model = attempt_load(MODEL_PATH, device=self.device, inplace=True, fuse=True)
            self.model.to(self.device).eval()

            self.stride = int(self.model.stride.max())
            self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names

            self.use_half = USE_HALF and self.device.type == "cuda"
            self.model.half() if self.use_half else self.model.float()

            device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            self.ui.statusbar.showMessage(f"模型加载成功 | 设备：{device_name} | 类别数：{len(self.names)}")
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "错误", f"模型加载失败：\n{exc}")
            self.ui.pushButton.setEnabled(False)

    def open_image(self):
        if self.model is None:
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            return

        self.current_image_path = file_path
        self.ui.label_2.clear()
        self.ui.label_2.setText("正在进行 AI 检测，请稍候...")
        self.ui.pushButton.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        try:
            result_image, detections = self.detect_image(image)
            self.current_result = result_image
            self._show_cv_image(self.ui.label_2, result_image)

            if detections:
                summary = self._format_detection_summary(detections)
                self.ui.statusbar.showMessage(f"检测完成 | 发现 {len(detections)} 个目标 | {summary}")
            else:
                self.ui.statusbar.showMessage("检测完成 | 未发现病害目标")
        except Exception as exc:
            traceback.print_exc()
            self.ui.label_2.setText("检测失败")
            QtWidgets.QMessageBox.critical(self, "错误", f"检测出错：\n{exc}")
        finally:
            self.ui.pushButton.setEnabled(True)

    def detect_image(self, original_bgr):
        with torch.inference_mode():
            im = letterbox(original_bgr, IMAGE_SIZE, stride=self.stride, auto=True)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            tensor = torch.from_numpy(im).to(self.device)
            tensor = tensor.half() if self.use_half else tensor.float()
            tensor /= 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)

            prediction = self.model(tensor, augment=False)[0]
            prediction = non_max_suppression(prediction, CONF_THRESHOLD, IOU_THRESHOLD, max_det=MAX_DETECTIONS)

            # 使用 PIL 渲染完美中文
            img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            try:
                # 字体大小 24，在 4K 大屏上非常清晰
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size=24)
            except:
                font = ImageFont.load_default()

            detections = []
            for det in prediction:
                if det is None or len(det) == 0:
                    continue

                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], original_bgr.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                    confidence = float(conf.item())
                    class_id = int(cls.item())
                    class_name = str(self.names[class_id])

                    detections.append({"class_name": class_name, "confidence": confidence, "box": (x1, y1, x2, y2)})

                    # 画框
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
                    # 写文字
                    label = f"{class_name} {confidence:.2f}"
                    text_bbox = draw.textbbox((x1, y1 - 32), label, font=font)
                    draw.rectangle(text_bbox, fill=(0, 255, 0))
                    draw.text((x1, y1 - 32), label, font=font, fill=(0, 0, 0))

            annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return annotated, detections

    @staticmethod
    def _format_detection_summary(detections):
        counts = {}
        for item in detections:
            name = item["class_name"]
            counts[name] = counts.get(name, 0) + 1
        return "、".join(f"{name} {count}个" for name, count in counts.items())

    def _show_cv_image(self, label, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()

        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_result is not None:
            self._show_cv_image(self.ui.label_2, self.current_result)

    def save_result(self):
        if self.current_result is None:
            QtWidgets.QMessageBox.information(self, "提示", "当前没有可保存的检测结果。")
            return

        default_name = "road_disease_result.jpg"
        if self.current_image_path:
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_name = base + "_检测结果.jpg"

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存结果", default_name,
                                                             "JPEG (*.jpg);;PNG (*.png)")
        if file_path:
            if not os.path.splitext(file_path)[1]:
                file_path += ".jpg"
            cv2.imwrite(file_path, self.current_result)
            self.ui.statusbar.showMessage(f"已保存至：{file_path}")

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "关于系统", "道路病害智能检测系统\n基于 PyQt5 与 YOLOv5 的单图大屏展示端")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RoadDiseaseWindow()
    window.showMaximized()  # 直接以最大化窗口启动
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()