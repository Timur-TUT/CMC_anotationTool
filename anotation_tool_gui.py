import sys
import re
import numpy as np
import cv2
from anotation_tool_ui import Ui_MainWindow
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class CMC:

    def __init__(
        self,
        filename,  #.raw形式のデータファイル名
        w=1344,  #画像データの幅
        h=1344,  #画像データの高さ
        tl=(400, 550),  #切り取り用(左上座標)
        br=(860, 650)):  #切り取り用(右下座標) おすすめ：br[0] = 810 or 860 or 945
        s = re.search("_", filename).span()
        self.id = filename[7:s[1] - 1]  #負荷レベル

        #データの読み込み
        with open(filename, 'rb') as f:
            rawdata = f.read()
            data = np.frombuffer(rawdata, dtype=np.int16).reshape(w, h)
            self.data = data[tl[1]:br[1], tl[0]:br[0]]

    # ひとつ前のマスキング画像を読み込む(対象：png等)
    def read_previous(self, path):
        # 画像の読み込み
        self.prev_img = np.array(Image.open(path))

    # 配列の正規化
    def normalize(self, array=None):
        if array is None: array = self.data
        normalized_data = (array - array.min()) / (array.max() -
                                                   array.min()) * 255
        return normalized_data.astype(np.uint8)
    
    # フーリエ変換
    def fourier_transform(self, r=14):
        # 2次元高速フーリエ変換で周波数領域の情報を取り出す
        f_transformed = np.fft.fft2(self.data)

        # 画像の中心に低周波数の成分がくるように並べかえる
        shifted_ft = np.fft.fftshift(f_transformed)
        # フィルターをかける
        shifted_ft[:, 230 - r:230 + r] = 0

        # 元通りに並び替える
        data2invert = np.fft.ifftshift(shifted_ft)
        # 逆フーリエ変換
        self.inverted_data = np.abs(np.fft.ifft2(data2invert))

        return self.normalize(self.inverted_data)

class AnotationApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(AnotationApp, self).__init__(parent)
        self.setupUi(self)

        # メソッドのオーバーライド
        self.imageViewer.wheelEvent = self.wheelEvent
        self.imageViewer.mousePressEvent = self.mousePressEvent
        self.imageViewer.mouseMoveEvent = self.mouseMoveEvent
        self.imageViewer.mouseReleaseEvent = self.mouseReleaseEvent

        # 変数の初期値設定
        self.image_dict = {k:None for k in ["raw", "filtered", "previous"]} # 3つの画像の保存を行う辞書. keyの型はQGraphicsPixmapItem
        self.numScheduledScalings = 0
        self.total_scaling = 1

        # 画像表示用のシーンの準備
        self.scene = QGraphicsScene(self)
        self.imageViewer.setScene(self.scene)

        # 初期状態設定
        self._hand = False
        self._drawing = False

    def resizeEvent(self, event):
        # ウィンドウサイズが変更されたときに呼び出されるメソッド
        for img in self.image_dict.values():
            if img:
                self.imageViewer.fitInView(img, Qt.KeepAspectRatio)
        super().resizeEvent(event)

    # qimageをndarrayに変換する関数
    def qimage_to_cv(self, qimage):
        w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
        bytes_ = qimage.bits().asstring(w * h * d // 8)
        arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
        return arr

    # 前の負荷画像で作成したマスキング画像を開くメソッド. メニューで ファイル＞Open＞Masking Image から実行される
    @pyqtSlot()
    def openPrev(self):
        # ダイアログ画面の設定
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;PNG Files (*.png)", options=options)
        file_name = file_name.split("/")[-1] # 絶対パスの最後を取得する(現時点では同じディレクトリにあるファイルのみを開く仕様になっている)
        
        if file_name:
            print(f"Opening file: {file_name}")
            
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) # グレースケールで読み込む
            height, width = img.shape # 画像サイズ
            red_img = np.zeros((height, width, 3), dtype=np.uint8) # RGB画像を作成し、赤色表示
            red_img[np.where(img==255)] = [255, 0, 0]
            
            q_image = QImage(red_img.copy(), width, height, QImage.Format_RGB888) #.copy()しないとエラーが起きる
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image)) # pixmapItemにすることで透明度の設定や表示・非表示が可能
            # pixmap_item = QGraphicsPixmapItem(QPixmap(file_name)) # グレースケール表示
            self.image_dict["previous"] = pixmap_item # 辞書に画像を登録
            self.checkBox_Previous.setChecked(True) # チェックボックスの更新
            pixmap_item.setOpacity(0.5)
            self.scene.addItem(pixmap_item) # 画像の表示
            self.imageViewer.fitInView(pixmap_item, Qt.KeepAspectRatio)
        else: return

    # 作業に必要なraw画像を開くメソッド.自動でフィルタリングされた画像も生成され、表示される. メニューで ファイル＞Open＞Raw Image から実行される
    @pyqtSlot()
    def openRaw(self):
        # ダイアログ画面の設定
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Raw Files (*.raw)", options=options)
        file_name = file_name.split("/")[-1]
        
        if file_name:
            print(f"Opening file: {file_name}")

            # CMCインスタンスを作成
            self.cmc = CMC(file_name)
            
            # NumPy配列からQImageに変換
            raw_image = self.cmc.normalize() # 正規化したraw画像
            ft_image = self.cmc.fourier_transform() # フーリエ変換のフィルター画像
            height, width = raw_image.shape # 画像サイズ

            for key, image, cb in zip(self.image_dict, (raw_image, ft_image, np.zeros_like(raw_image)), (self.checkBox_Raw, self.checkBox_Filtered, self.checkBox_Previous)):
                q_image = QImage(image.copy(), width, height, QImage.Format_Grayscale8) #.copy()しないとエラーが起きる
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image)) # pixmapItemにすることで透明度の設定や表示・非表示が可能
                self.image_dict[key] = pixmap_item # 辞書に画像を登録
                cb.setChecked(True) # チェックボックスの更新
                if key == "previous":
                    pixmap_item.setOpacity(0.3)
                self.scene.addItem(pixmap_item) # 画像の表示
                self.imageViewer.fitInView(pixmap_item, Qt.KeepAspectRatio)
        else: return        

    @pyqtSlot()
    def setOpacity(self, value, key):
        if self.image_dict[key]:
            self.image_dict[key].setOpacity(value / 100)

    @pyqtSlot()
    def setPen(self):
        pass

    @pyqtSlot()
    def setEraser(self):
        pass

    @pyqtSlot()
    def hand(self):
        if self._hand:
            self._hand = False
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
        else:
            self._hand = True
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)

    @pyqtSlot()
    def zoom(self, factor = 1.5):
        if 0.95 < self.total_scaling * factor < 50:
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)

    @pyqtSlot()
    def unzoom(self, factor = 1/1.5):
        if 0.95 < self.total_scaling * factor < 50:
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)
    
    @pyqtSlot()
    def saveImage(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                          "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
 
        if filePath == "":
            return

        img = self.qimage_to_cv(self.image_dict["previous"].pixmap().toImage()) # qimageをndarrayに変換
        gray_img = img[:, :, 2] # 赤色のみを書き出し
        
        # 日本語パスに対応した保存
        _, buf = cv2.imencode('*.png', gray_img) 
        buf.tofile(filePath)

    @pyqtSlot()
    def toggle_image(self, state, key):
        if self.image_dict[key]:
            if state:
                self.image_dict[key].setVisible(True)
            else:
                self.image_dict[key].setVisible(False)

    def draw(self, event):
        # クリックした位置を取得
        pos_scene = self.imageViewer.mapToScene(event.pos())
        x, y = int(pos_scene.x()), int(pos_scene.y())
        
        image = self.image_dict["previous"].pixmap().toImage()
        if 0 <= x < image.width() and 0 <= y < image.height():
            if event.buttons() & Qt.RightButton:
                image.setPixelColor(x, y, QColor(0, 0, 0)) # クリックしたピクセルを黒くする
            elif event.buttons() & Qt.LeftButton:
                image.setPixelColor(x, y, QColor(255, 0, 0)) # クリックしたピクセルを白くする
        
        # キャンバスの画像を QPixmap に変換して表示
        canvas_pixmap = QPixmap.fromImage(image)
        self.image_dict["previous"].setPixmap(canvas_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)
            event = QMouseEvent(QEvent.GraphicsSceneDragMove, event.pos(), Qt.MouseButton.LeftButton, 
                                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            QGraphicsView.mousePressEvent(self.imageViewer, event)

        else:
            if self.image_dict["previous"] and self._hand == False:
                self._drawing = True
                self.draw(event)
            else:
                QGraphicsView.mousePressEvent(self.imageViewer, event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            self.draw(event)
        QGraphicsView.mouseMoveEvent(self.imageViewer, event)
        
    def mouseReleaseEvent(self, event):
        if self._hand == False:
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
            if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
                self._drawing = False
        QGraphicsView.mouseReleaseEvent(self.imageViewer, event)        

    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 8
        numSteps = numDegrees / 15
        self.numScheduledScalings += numSteps
        if self.numScheduledScalings * numSteps < 0:
            self.numScheduledScalings = numSteps
        self.scale_animation = QTimeLine(350, self)
        self.scale_animation.setUpdateInterval(20)
        self.scale_animation.valueChanged.connect(self.scalingTime)
        self.scale_animation.finished.connect(self.animFinished)
        self.scale_animation.start()

    def scalingTime(self, x):
        factor = 1.0 + float(self.numScheduledScalings) / 300.0
        if 0.95 < self.total_scaling * factor < 50:
            self.total_scaling *= factor
            self.imageViewer.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.imageViewer.scale(factor, factor)
        else:
            self.numScheduledScalings = 0
            self.scale_animation.stop()

    def animFinished(self):
        if self.numScheduledScalings > 0:
            self.numScheduledScalings -= 1
        else:
            self.numScheduledScalings += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnotationApp()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())

"""
# 改善・バグメモ
ok・長押しで描画を行えるようにする
ok・Previous Imageがなくても描画が出来るようにしたい
・クリアボタンが欲しい ⇒消しゴムボタンの代わりにするのが良さそう
ok・拡大する時にマウスの位置を中心に拡大縮小を行いたい(現在は画面中央を拡大している)
ok・拡大と縮小に最大値を設ける
ok・ショートカットキーが使えるようにする
ok・透明度の設定だけではわかりづらいからRawとFilteredをグレースケールではなく、色を付けて表示する？
"""