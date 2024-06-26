# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'anotation_tool.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import images_qr    # 消さない

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(35, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.buttons.setObjectName("buttons")
        self.penButton = QtWidgets.QPushButton(self.centralwidget)
        self.penButton.setCursor(QtGui.QCursor(QtCore.Qt.ForbiddenCursor))
        self.penButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.penButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/src/enpitsu.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.penButton.setIcon(icon)
        self.penButton.setDefault(False)
        self.penButton.setFlat(False)
        self.penButton.setObjectName("penButton")
        self.buttons.addWidget(self.penButton)
        self.keshiButton = QtWidgets.QPushButton(self.centralwidget)
        self.keshiButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.keshiButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/src/keshi.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.keshiButton.setIcon(icon1)
        self.keshiButton.setDefault(False)
        self.keshiButton.setFlat(False)
        self.keshiButton.setObjectName("keshiButton")
        self.buttons.addWidget(self.keshiButton)
        self.handButton = QtWidgets.QPushButton(self.centralwidget)
        self.handButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.handButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/src/drag.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.handButton.setIcon(icon2)
        self.handButton.setDefault(False)
        self.handButton.setFlat(False)
        self.handButton.setObjectName("handButton")
        self.buttons.addWidget(self.handButton)
        self.zoomButton = QtWidgets.QPushButton(self.centralwidget)
        self.zoomButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.zoomButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/src/zoom.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoomButton.setIcon(icon3)
        self.zoomButton.setDefault(False)
        self.zoomButton.setFlat(False)
        self.zoomButton.setObjectName("zoomButton")
        self.buttons.addWidget(self.zoomButton)
        self.unzoomButton = QtWidgets.QPushButton(self.centralwidget)
        self.unzoomButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.unzoomButton.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/src/unzoom.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.unzoomButton.setIcon(icon4)
        self.unzoomButton.setDefault(False)
        self.unzoomButton.setFlat(False)
        self.unzoomButton.setObjectName("unzoomButton")
        self.buttons.addWidget(self.unzoomButton)
        self.gridLayout.addLayout(self.buttons, 1, 0, 1, 1)
        self.imageViewer = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.imageViewer.sizePolicy().hasHeightForWidth())
        self.imageViewer.setSizePolicy(sizePolicy)
        self.imageViewer.setMinimumSize(QtCore.QSize(460, 100))
        self.imageViewer.setSizeIncrement(QtCore.QSize(0, 0))
        self.imageViewer.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.imageViewer.setMouseTracking(True)
        self.imageViewer.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.imageViewer.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.imageViewer.setAutoFillBackground(False)
        self.imageViewer.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imageViewer.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.imageViewer.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.imageViewer.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.imageViewer.setObjectName("imageViewer")
        self.gridLayout.addWidget(self.imageViewer, 0, 0, 1, 1)
        self.sliders = QtWidgets.QVBoxLayout()
        self.sliders.setObjectName("sliders")
        self.sliderRaw = QtWidgets.QHBoxLayout()
        self.sliderRaw.setObjectName("sliderRaw")
        self.checkBox_Raw = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Raw.setToolTip("Set visibility of Raw Image")
        self.checkBox_Raw.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_Raw.sizePolicy().hasHeightForWidth())
        self.checkBox_Raw.setSizePolicy(sizePolicy)
        self.checkBox_Raw.setMinimumSize(QtCore.QSize(70, 0))
        self.checkBox_Raw.setMaximumSize(QtCore.QSize(110, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox_Raw.setFont(font)
        self.checkBox_Raw.setFocusPolicy(QtCore.Qt.NoFocus)
        self.checkBox_Raw.setObjectName("checkBox_Raw")
        self.sliderRaw.addWidget(self.checkBox_Raw)
        self.horizontalSlider_Raw = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_Raw.setToolTip("Change transparency of Raw Image")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_Raw.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_Raw.setSizePolicy(sizePolicy)
        self.horizontalSlider_Raw.setMinimumSize(QtCore.QSize(100, 0))
        self.horizontalSlider_Raw.setMaximumSize(QtCore.QSize(140, 11))
        self.horizontalSlider_Raw.setFocusPolicy(QtCore.Qt.NoFocus)
        self.horizontalSlider_Raw.setMaximum(100)
        self.horizontalSlider_Raw.setSliderPosition(100)
        self.horizontalSlider_Raw.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Raw.setObjectName("horizontalSlider_Raw")
        self.sliderRaw.addWidget(self.horizontalSlider_Raw)
        self.sliders.addLayout(self.sliderRaw)
        self.sliderFiltered = QtWidgets.QHBoxLayout()
        self.sliderFiltered.setObjectName("sliderFiltered")
        self.checkBox_Filtered = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Filtered.setToolTip("Set visibility of Filtered Image")
        self.checkBox_Filtered.setMinimumSize(QtCore.QSize(70, 0))
        self.checkBox_Filtered.setMaximumSize(QtCore.QSize(110, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox_Filtered.setFont(font)
        self.checkBox_Filtered.setFocusPolicy(QtCore.Qt.NoFocus)
        self.checkBox_Filtered.setObjectName("checkBox_Filtered")
        self.sliderFiltered.addWidget(self.checkBox_Filtered)
        self.horizontalSlider_Filtered = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_Filtered.setToolTip("Change transparency of Filtered Image")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_Filtered.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_Filtered.setSizePolicy(sizePolicy)
        self.horizontalSlider_Filtered.setMinimumSize(QtCore.QSize(100, 0))
        self.horizontalSlider_Filtered.setMaximumSize(QtCore.QSize(140, 11))
        self.horizontalSlider_Filtered.setFocusPolicy(QtCore.Qt.NoFocus)
        self.horizontalSlider_Filtered.setMaximum(100)
        self.horizontalSlider_Filtered.setProperty("value", 100)
        self.horizontalSlider_Filtered.setSliderPosition(100)
        self.horizontalSlider_Filtered.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Filtered.setObjectName("horizontalSlider_Filtered")
        self.sliderFiltered.addWidget(self.horizontalSlider_Filtered)
        self.sliders.addLayout(self.sliderFiltered)
        self.sliderPrevious = QtWidgets.QHBoxLayout()
        self.sliderPrevious.setObjectName("sliderPrevious")
        self.checkBox_Previous = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Previous.setToolTip("Set visibility of Masking Image")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_Previous.sizePolicy().hasHeightForWidth())
        self.checkBox_Previous.setSizePolicy(sizePolicy)
        self.checkBox_Previous.setMinimumSize(QtCore.QSize(70, 0))
        self.checkBox_Previous.setMaximumSize(QtCore.QSize(110, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox_Previous.setFont(font)
        self.checkBox_Previous.setFocusPolicy(QtCore.Qt.NoFocus)
        self.checkBox_Previous.setObjectName("checkBox_Previous")
        self.sliderPrevious.addWidget(self.checkBox_Previous)
        self.horizontalSlider_Previous = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_Previous.setToolTip("Change transparency of Masking Image")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_Previous.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_Previous.setSizePolicy(sizePolicy)
        self.horizontalSlider_Previous.setMinimumSize(QtCore.QSize(100, 0))
        self.horizontalSlider_Previous.setMaximumSize(QtCore.QSize(140, 11))
        self.horizontalSlider_Previous.setFocusPolicy(QtCore.Qt.NoFocus)
        self.horizontalSlider_Previous.setMaximum(100)
        self.horizontalSlider_Previous.setProperty("value", 30)
        self.horizontalSlider_Previous.setSliderPosition(30)
        self.horizontalSlider_Previous.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Previous.setObjectName("horizontalSlider_Previous")
        self.sliderPrevious.addWidget(self.horizontalSlider_Previous)
        self.sliders.addLayout(self.sliderPrevious)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        self.listWidget.setSizePolicy(sizePolicy)
        self.listWidget.setMinimumSize(QtCore.QSize(70, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.listWidget.setFont(font)
        self.listWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.listWidget.setObjectName("listWidget")
        self.listWidget.currentItemChanged.connect(self.load)
        self.sliders.addWidget(self.listWidget)
        self.gridLayout.addLayout(self.sliders, 0, 1, 1, 1)
        self.linkButtons = QtWidgets.QHBoxLayout()
        self.linkButtons.setObjectName("linkButtons")
        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.prevButton.setFont(font)
        self.prevButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.prevButton.setStyleSheet("color: rgba(0, 0, 255, 200);\n"
"")
        self.prevButton.setFlat(True)
        self.prevButton.setObjectName("prevButton")
        self.linkButtons.addWidget(self.prevButton)
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.nextButton.setFont(font)
        self.nextButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.nextButton.setStyleSheet("color: rgba(0, 0, 255, 200);")
        self.nextButton.setFlat(True)
        self.nextButton.setObjectName("nextButton")
        self.linkButtons.addWidget(self.nextButton)
        self.gridLayout.addLayout(self.linkButtons, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 705, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.setShortcut("Ctrl+S")
        self.actionOpen_Previous = QtWidgets.QAction(MainWindow)
        self.actionOpen_Previous.setObjectName("actionOpen_Previous")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionRaw_Image")
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionUndo = QtWidgets.QAction(MainWindow)
        self.actionUndo.setObjectName("actionUndo")
        self.actionUndo.setShortcut("Ctrl+Z")
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionUndo)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.prevButton.clicked.connect(MainWindow.prevFile) # type: ignore
        self.nextButton.clicked.connect(MainWindow.nextFile) # type: ignore
        self.penButton.clicked.connect(MainWindow.setPen) # type: ignore
        self.keshiButton.clicked.connect(MainWindow.erase) # type: ignore
        self.zoomButton.clicked.connect(MainWindow.zoom) # type: ignore
        self.unzoomButton.clicked.connect(MainWindow.unzoom) # type: ignore
        self.actionOpen.triggered.connect(MainWindow.open) # type: ignore
        self.actionUndo.triggered.connect(MainWindow.undo) # type: ignore
        self.horizontalSlider_Raw.valueChanged['int'].connect(lambda value, key="raw": MainWindow.setOpacity(value, key)) # type: ignore
        self.horizontalSlider_Filtered.valueChanged['int'].connect(lambda value, key="filtered": MainWindow.setOpacity(value, key)) # type: ignore
        self.horizontalSlider_Previous.valueChanged['int'].connect(lambda value, key="previous": MainWindow.setOpacity(value, key)) # type: ignore
        self.actionSave.triggered.connect(MainWindow.saveImage) # type: ignore
        self.checkBox_Raw.toggled['bool'].connect(lambda state, key="raw": MainWindow.toggle_image(state, key)) # type: ignore
        self.checkBox_Filtered.toggled['bool'].connect(lambda state, key="filtered": MainWindow.toggle_image(state, key)) # type: ignore
        self.checkBox_Previous.toggled['bool'].connect(lambda state, key="previous": MainWindow.toggle_image(state, key)) # type: ignore
        self.handButton.clicked.connect(MainWindow.hand) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.MainWindow = MainWindow

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AnotationTool"))
        self.checkBox_Raw.setText(_translate("MainWindow", "Raw"))
        self.checkBox_Filtered.setText(_translate("MainWindow", "Filtered"))
        self.checkBox_Previous.setText(_translate("MainWindow", "Previous"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.prevButton.setText(_translate("MainWindow", "← Previous"))
        self.nextButton.setText(_translate("MainWindow", "Next →"))
        self.menu.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save as..."))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z"))