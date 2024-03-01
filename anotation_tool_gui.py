import sys
import re
import numpy as np
import cv2
from anotation_tool_ui import Ui_MainWindow
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
        self.image_dict = {k:None for k in ["raw", "filtered", "previous"]} # 3つの画像の保存を行う辞書. valueの型はQGraphicsPixmapItem
        self.undo_image = None
        self.numScheduledScalings = 0
        self.total_scaling = 1

        # 画像表示用のシーンの準備
        self.scene = QGraphicsScene(self)
        self.imageViewer.setScene(self.scene)

        # 初期状態設定
        self._hand = False
        self._drawing = False

    # ウィンドウサイズが変更された際に呼び出されるメソッド
    def resizeEvent(self, event):
        for img in self.image_dict.values():
            if img:
                self.imageViewer.fitInView(img, Qt.KeepAspectRatio)
        super().resizeEvent(event)

    # Qimageをndarrayに変換するメソッド
    def qimage_to_cv(self, qimage):
        w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
        bytes_ = qimage.bits().asstring(w * h * d // 8)
        arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
        return arr
    
    # 前の負荷画像で作成したマスキング画像を開くスロット. メニューで ファイル＞Open＞Masking Image から実行される
    @pyqtSlot()
    def openPrev(self):
        # ダイアログ画面の設定
        if self.image_dict["raw"] is None:
            self.show_error_dialog(str("Open '.raw' file first"))
            return
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;PNG Files (*.png)", options=options)
        
        if file_name:
            print(f"Opening file: {file_name}")
            
            try:
                _img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) # グレースケールで読み込む
                height, width = _img.shape # 画像サイズ
                img = np.zeros((height, width, 3), dtype=np.uint8) # RGB画像を作成し
                img[np.where(self.ft_image>100)] = [225, 147, 56] # 明らかにき裂である画素をオレンジ色で表示
                img[np.where(_img==255)] = [255, 0, 0] # 赤色で表示
                
                q_image = QImage(img.copy(), width, height, QImage.Format_RGB888) #.copy()しないとエラーが起きる
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image)) # pixmapItemにすることで透明度の設定や表示・非表示が可能
                # pixmap_item = QGraphicsPixmapItem(QPixmap(file_name)) # グレースケール表示
                self.image_dict["previous"] = pixmap_item # 辞書に画像を登録
                self.checkBox_Previous.setChecked(True) # チェックボックスの更新
                pixmap_item.setOpacity(0.5) # 透明度の設定
                self.scene.addItem(pixmap_item) # 画像の表示
                self.imageViewer.fitInView(pixmap_item, Qt.KeepAspectRatio)
            except AttributeError:
                # エラーダイアログを表示
                self.show_error_dialog(str("Only image files can be opened"))
                return
        else: return

    # 作業に必要なraw画像を開くスロット.自動でフィルタリングされた画像も生成され、表示される. メニューで ファイル＞Open＞Raw Image から実行される
    @pyqtSlot()
    def openRaw(self):
        # ダイアログ画面の設定
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Raw Files (*.raw)", options=options)
        
        if file_name:
            print(f"Opening file: {file_name}")
            self.MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", "AnotationTool: " + file_name))
            if file_name[-4:] != ".raw":
                # エラーダイアログを表示
                self.show_error_dialog(str("Only '.raw' files can be opened"))
                return

            # CMCインスタンスを作成
            self.cmc = CMC(file_name)
            
            # NumPy配列からQImageに変換
            raw_image = self.cmc.normalize() # 正規化したraw画像
            self.ft_image = self.cmc.fourier_transform() # フーリエ変換のフィルター画像
            height, width = raw_image.shape # 画像サイズ

            for key, image, cb in zip(self.image_dict, (raw_image, self.ft_image, np.zeros_like(raw_image)), (self.checkBox_Raw, self.checkBox_Filtered, self.checkBox_Previous)):
                q_image = QImage(image.copy(), width, height, QImage.Format_Grayscale8) #.copy()しないとエラーが起きる
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image)) # pixmapItemにすることで透明度の設定や表示・非表示が可能
                self.image_dict[key] = pixmap_item # 辞書に画像を登録
                cb.setChecked(True) # チェックボックスの更新
                if key == "previous":
                    pixmap_item.setOpacity(0.3) # 透明度の設定
                self.scene.addItem(pixmap_item) # 画像の表示
                self.imageViewer.fitInView(pixmap_item, Qt.KeepAspectRatio)
        else: return

    def show_error_dialog(self, error_message):
        # エラーダイアログの作成と表示
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText("An error occurred:")
        error_dialog.setInformativeText(error_message)
        error_dialog.exec_()

    # スライダーで透明度変更された際のスロット
    @pyqtSlot()
    def setOpacity(self, value, key):
        if self.image_dict[key]:
            self.image_dict[key].setOpacity(value / 100)

    @pyqtSlot()
    def setPen(self):
        self.show_error_dialog(str("This feature has not yet been implemented"))

    @pyqtSlot()
    def erase(self):
        self.show_error_dialog(str("This feature has not yet been implemented"))

    # ハンドボタンが押された際のスロット
    @pyqtSlot()
    def hand(self):
        if self._hand: # ハンドモードをオフ
            self._hand = False
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
        else:
            self._hand = True # ハンドモードをオン
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)

    # 拡大ボタンが押された際のスロット
    @pyqtSlot()
    def zoom(self, factor = 1.5):
        if 0.95 < self.total_scaling * factor < 50: # 拡大に範囲を設けている
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)

    # 縮小ボタンが押された際のスロット
    @pyqtSlot()
    def unzoom(self, factor = 1/1.5):
        if 0.95 < self.total_scaling * factor < 50: # 縮小に範囲を設けている
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)

    @pyqtSlot()
    def undo(self):
        self._drawing = False
        self.image_dict["previous"].setPixmap(self.undo_image)
    
    # 画像を保存する際のスロット．メニューで ファイル＞Save から実行される
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

    # チェックボックスが押された際のスロット
    @pyqtSlot()
    def toggle_image(self, state, key):
        if self.image_dict[key]:
            if state: # チェックボックスがactiveなら
                self.image_dict[key].setVisible(True)
            else:
                self.image_dict[key].setVisible(False)

    # クリックした位置で描画を行うメソッド
    def draw(self, event):
        # クリックした位置を取得
        pos_scene = self.imageViewer.mapToScene(event.pos())
        x, y = int(pos_scene.x()), int(pos_scene.y())
        
        image = self.image_dict["previous"].pixmap().toImage() # pixmapをqimageに変換
        if 0 <= x < image.width() and 0 <= y < image.height(): # 画像内のクリックのみ
            if event.buttons() & Qt.RightButton: # 右クリック
                image.setPixelColor(x, y, QColor(0, 0, 0)) # クリックしたピクセルを黒くする
            elif event.buttons() & Qt.LeftButton: # 左クリック
                image.setPixelColor(x, y, QColor(225, 147, 56)) # クリックしたピクセルをオレンジに
        
        # キャンバスの画像を QPixmap に変換して表示
        canvas_pixmap = QPixmap.fromImage(image)
        self.image_dict["previous"].setPixmap(canvas_pixmap)

    # タブキーが押されている間の処理
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.toggle_image(True, "previous")
            self.checkBox_Previous.setChecked(True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.toggle_image(False, "previous")
            self.checkBox_Previous.setChecked(False)

    # マウスクリックイベント
    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton: #　中ボタンをクリックした場合はドラッグ
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)
            event = QMouseEvent(QEvent.GraphicsSceneDragMove, event.pos(), Qt.MouseButton.LeftButton, 
                                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            QGraphicsView.mousePressEvent(self.imageViewer, event)

        else: # それ以外はdrawメソッドで処理
            if self.image_dict["previous"] and self._hand == False:
                self._drawing = True
                self.undo_image = QPixmap.fromImage(self.image_dict["previous"].pixmap().toImage().copy())
                self.draw(event)
            else:
                QGraphicsView.mousePressEvent(self.imageViewer, event)

    # マウス移動イベント
    def mouseMoveEvent(self, event):
        if self._drawing: # 描画中にマウスが動いた場合はその途中のピクセルも塗る
            self.draw(event)
        QGraphicsView.mouseMoveEvent(self.imageViewer, event)
        
    # マウスクリックイベント
    def mouseReleaseEvent(self, event):
        if self._hand == False:
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
            if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
                self._drawing = False # 描画終了
        QGraphicsView.mouseReleaseEvent(self.imageViewer, event)        

    # 中ボタン(ホイール)回転イベント
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

    # 画像拡大をアニメーションによってスムーズに行うメソッド
    def scalingTime(self, x):
        factor = 1.0 + float(self.numScheduledScalings) / 300.0
        if 0.95 < self.total_scaling * factor < 50: # 拡大縮小に範囲を設ける
            self.total_scaling *= factor
            self.imageViewer.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # マウス中心に拡大縮小
            self.imageViewer.scale(factor, factor)
        else:
            self.numScheduledScalings = 0
            self.scale_animation.stop()

    # 拡大縮小アニメーション終了
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
ok・Undo機能の追加
ok・以前塗った部分と今回塗った部分で色を分ける
・クリアボタンが欲しい ⇒消しゴムボタンの代わりにするのが良さそう
ok・拡大する時にマウスの位置を中心に拡大縮小を行いたい(現在は画面中央を拡大している)
ok・拡大と縮小に最大値を設ける
ok・ショートカットキーが使えるようにする
ok・透明度の設定だけではわかりづらいからRawとFilteredをグレースケールではなく、色を付けて表示する？
・exe化
ok・周波数フィルタリング⇒輝度値が高い画素をき裂として事前に塗ることで作業量の軽量化
・tabボタンでmasking画像の表示/非表示
"""