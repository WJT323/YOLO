# -*- coding: utf-8 -*-
"""Responsive PyQt5 UI for the road-disease detection application.
Save this file as win1.py in the same directory as road_disease_pyqt_main.py.
"""
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # 初始窗口调大，适配单图大屏展示
        MainWindow.resize(1280, 850)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 700))
        MainWindow.setWindowTitle("道路病害智能检测系统")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Main vertical layout: header / image area / controls
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(34, 24, 34, 22)
        self.mainLayout.setSpacing(20)

        # ================= Header (标题区) =================
        self.headerLayout = QtWidgets.QVBoxLayout()
        self.headerLayout.setSpacing(5)

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("道路病害智能检测系统")
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setMinimumHeight(54)
        # 【修复】取消最大宽度限制，让标题不再被截断
        self.lineEdit.setMaximumWidth(16777215)
        self.headerLayout.addWidget(self.lineEdit, 0, QtCore.Qt.AlignHCenter)

        self.subtitleLabel = QtWidgets.QLabel("基于 YOLOv5 的道路病害识别与分析")
        self.subtitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitleLabel.setObjectName("subtitleLabel")
        self.headerLayout.addWidget(self.subtitleLabel)

        self.mainLayout.addLayout(self.headerLayout)

        # ================= 图像区 (单窗口逻辑) =================
        # 直接使用一个主面板，不再区分左右
        self.imageLayout = QtWidgets.QHBoxLayout()
        self.imageLayout.setSpacing(24)

        # 主结果面板
        self.resultPanel = QtWidgets.QFrame(self.centralwidget)
        self.resultPanel.setObjectName("imagePanel")
        self.resultPanel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultPanelLayout = QtWidgets.QVBoxLayout(self.resultPanel)
        self.resultPanelLayout.setContentsMargins(16, 14, 16, 16)
        self.resultPanelLayout.setSpacing(10)

        self.resultTitle = QtWidgets.QLabel("病害识别结果")
        self.resultTitle.setObjectName("panelTitle")
        self.resultTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.resultPanelLayout.addWidget(self.resultTitle)

        # 用于显示图片的 Label
        self.label_2 = QtWidgets.QLabel("检测结果将在此处显示")
        self.label_2.setObjectName("label_2")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setMinimumSize(320, 260)
        self.label_2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_2.setWordWrap(True)
        self.resultPanelLayout.addWidget(self.label_2, 1)

        self.imageLayout.addWidget(self.resultPanel, 1)
        self.mainLayout.addLayout(self.imageLayout, 1)

        # ================= 底部按钮区 =================
        self.actionLayout = QtWidgets.QHBoxLayout()
        self.actionLayout.addStretch(1)

        self.pushButton = QtWidgets.QPushButton("上传图片并开始检测")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setMinimumSize(210, 52)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.actionLayout.addWidget(self.pushButton)

        self.actionLayout.addStretch(1)
        self.mainLayout.addLayout(self.actionLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        # Menu bar and status bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 25))
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ================= 样式表 =================
        MainWindow.setStyleSheet("""
            QMainWindow, QWidget#centralwidget {
                background: #f3f6fb;
                color: #243247;
                font-family: "Microsoft YaHei", "微软雅黑";
                font-size: 10pt;
            }
            QLineEdit#lineEdit {
                background: transparent;
                border: none;
                color: #172b4d;
                font-size: 26pt; /* 字体稍微加大 */
                font-weight: bold;
                padding: 8px;
            }
            QLabel#subtitleLabel {
                color: #718096;
                font-size: 11pt;
                padding-bottom: 4px;
            }
            QFrame#imagePanel {
                background: #ffffff;
                border: 1px solid #d9e2ef;
                border-radius: 14px;
            }
            QLabel#panelTitle {
                color: #334e68;
                font-size: 14pt; /* 标题字体加大 */
                font-weight: 600;
                padding: 5px;
                background: transparent;
            }
            QLabel#label_2 {
                background: #111827;
                color: #cbd5e1;
                border: 1px solid #26364d;
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton#pushButton {
                background: #2878e5;
                color: white;
                border: none;
                border-radius: 9px;
                padding: 10px 24px;
                font-size: 12pt;
                font-weight: 600;
            }
            QPushButton#pushButton:hover { background: #1766ce; }
            QPushButton#pushButton:pressed { background: #1255ad; }
            QMenuBar { background: #ffffff; padding: 4px; }
            QMenuBar::item { padding: 6px 12px; }
            QMenuBar::item:selected { background: #e8f0fe; border-radius: 5px; }
            QStatusBar {
                background: #ffffff;
                color: #52657d;
                border-top: 1px solid #dce4ef;
            }
        """)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "道路病害智能检测系统"))