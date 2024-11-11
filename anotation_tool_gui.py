import sys
import os
import re
import numpy as np
import cv2
import images_qr
from pathlib import Path
from anotation_tool_ui import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from scipy import ndimage

ORANGE = [225, 147, 56]
RED = [255, 0, 0]
BLACK = [0, 0, 0]


# 配列の正規化(画像として表示することが可能)
def normalize(array, box=None, value_range=None):
    """配列を正規化し，画像として表示することを可能にする

    Args:
        array (ndarray): 正規化したい配列
        box (tuple, optional): 注目領域の指定．この領域の最小最大値によって正規化される．順番は(左上, 右下)，形式は((y,x),(y,x)). Defaults to None.
        range (tuple, optional): 最小最大値の指定．この範囲で正規化される．boxも入力されている場合，そちらが優先される. Defaults to None.

    Returns:
        ndarray: 正規化された画像.
    """
    if box:
        tl, br = box
        _min = array[tl[0] : br[0], tl[1] : br[1]].min()
        _max = array[tl[0] : br[0], tl[1] : br[1]].max()
    elif value_range:
        _min, _max = value_range
    else:
        _min, _max = array.min(), array.max()

    normalized_data = ((array - _min) / (_max - _min)) * 255
    normalized_data = np.clip(normalized_data, 0, 255)

    return normalized_data.astype(np.uint8)


def auto_canny(image, sigma=0.33):
    med = np.median(image)
    th1 = int(max(0, (1 - sigma) * med))
    th2 = int(max(255, (1 + sigma) * med))
    canny = cv2.Canny(image, th1, th2, True)
    return canny


# .raw画像を読み込み、周波数フィルタリングを行い、データを管理するクラス
class CMC:

    @classmethod
    def find_load(cls, fname):
        s = re.search(
            r"(\d+)(UL)?_(\d+)?[a-z]*_?SC", fname
        )  # 正規表現により画像の負荷を取得
        load = s.group(1)
        option = "_" + s.group(2) if s.group(2) else ""
        # deg = s.group(3) if s.group(3) else None

        return load, option

    def __init__(
        self,
        filename,  # .raw形式のデータファイル名
        w=1344,  # 画像データの幅
        h=1344,  # 画像データの高さ
    ):

        self.id, self.option = CMC.find_load(filename)  # 負荷レベルと回転角を取得する
        self.w, self.h = w, h

        # データの読み込み
        with open(filename, "rb") as f:
            rawdata = f.read()
            self.data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
            self.fourier_filter()
            self.extract_edge()

    def extract_edge(self):
        xray_image_laplace_gaussian = ndimage.gaussian_laplace(self.data, sigma=1)
        denoised_data = cv2.fastNlMeansDenoising(
            normalize(xray_image_laplace_gaussian), None, 10, 7, 21
        )
        dst = auto_canny(denoised_data, 0.6)
        kernel = np.ones((3, 3), np.uint8)
        self.edge = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

    # フーリエ変換
    def fourier_filter(self, r=9):
        # 2次元高速フーリエ変換で周波数領域の情報を取り出す
        f_transformed = np.fft.fft2(self.data)

        # 画像の中心に低周波数の成分がくるように並べかえる
        shifted_ft = np.fft.fftshift(f_transformed)
        # フィルターをかける
        shifted_ft[:, self.w // 2 - r : self.w // 2 + r] = 0

        # 元通りに並び替える
        data2invert = np.fft.ifftshift(shifted_ft)
        # 逆フーリエ変換
        self.filtered = normalize(np.abs(np.fft.ifft2(data2invert)))


# ツールのクラス
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
        self.image_dict = {
            k: None for k in ["raw", "filtered", "previous"]
        }  # 3つの画像を保管する辞書. valueの型はQGraphicsPixmapItem
        self.undo_image = None  # 動作直前の画像を保存する変数．Ctrl+Z時に使用
        self.numScheduledScalings = 0  # 拡大縮小をスムーズに行うための変数
        self.total_scaling = 1  # 合計拡大縮小回数．無限に拡大することを防止
        self.change_count = 0  # アノテーションする際の手数をカウント

        # 画像表示用のシーンの準備
        self.scene = QGraphicsScene(self)
        self.imageViewer.setScene(self.scene)
        self.diameter = 1

        # 初期状態設定
        self._hand = False
        self._drawing = False

    # 作業に必要なinputディレクトリとouputディレクトリを開くスロット.自動でフィルタリングされた画像も生成され、表示される. メニューで ファイル＞Open から実行される
    @pyqtSlot()
    def open(self):
        # ダイアログ画面の設定
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.input_folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", options=options
        )

        if self.input_folder:
            print(f"Input folder: {self.input_folder}")
            self.first_load = True  # 初めての読み込みのフラグ
            self.listWidget.clear()  # 読み込み済みのファイルがあればクリア
            for item in self.imageViewer.items():  # すべての画像をクリア
                self.scene.removeItem(item)

            raw_files = [
                file for file in os.listdir(self.input_folder) if file.endswith(".raw")
            ]  # .raw画像の取得
            if not raw_files:  # .raw画像が存在しない場合
                self.show_error_dialog("There are no '.raw' files in that directory")
                return

            # 必要なパラメータをparameters.txtから読み込む
            with open(
                os.path.join(self.input_folder, "parameters.txt"),
                mode="r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    line = line.strip()  # 前後の空白や改行を削除
                    if "=" in line:
                        exec(line)

            # ファイルをリストへ追加
            for i, fname in enumerate(
                sorted(raw_files, key=lambda x: int(CMC.find_load(x)[0]))
            ):
                # base imageを作成
                if i == 1:
                    self.base_edge = CMC(
                        os.path.join(self.input_folder, fname), self.w, self.h
                    ).edge
                item = QListWidgetItem()
                item.setText(fname)
                item.setTextAlignment(Qt.AlignLeading | Qt.AlignVCenter)
                icon = QIcon()
                icon.addPixmap(QPixmap(":/src/raw.png"), QIcon.Normal, QIcon.Off)
                item.setIcon(icon)
                self.listWidget.addItem(item)
        else:
            return

        self.output_folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", options=options
        )
        current_directory = os.getcwd()  # 現在のディレクトリを取得

        if not (
            self.output_folder and Path(self.output_folder) != Path(current_directory)
        ):  # 選択されなかった場合や作業ディレクトリと同じ場合は、現在のディレクトリにフォルダーを自動的に作成
            new_directory = os.path.join(
                current_directory, "anotation_output"
            )  # 新しいフォルダのパスを作成
            os.makedirs(
                new_directory, exist_ok=True
            )  # フォルダを作成（既に存在する場合は無視）
            self.output_folder = new_directory
        self.listWidget.setCurrentRow(0)  # リストの最初のアイテムを選択
        print(f"Output folder: {self.output_folder}")

    # スライダーで透明度が変更された際のスロット
    @pyqtSlot()
    def setOpacity(self, value, key):
        if self.image_dict[key]:
            self.image_dict[key].setOpacity(value / 100)

    # 未実装機能
    @pyqtSlot()
    def setPen(self):
        self.show_error_dialog(str("This feature has not yet been implemented"))

    # 消しゴムボタンが押された際のスロット
    @pyqtSlot()
    def erase(self):
        if self.image_dict["previous"]:
            reply = QMessageBox.question(  # 確認のダイアログ画面
                self,
                "Message",
                "All annotations will be cleared. Are you sure?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.undo_image = QPixmap.fromImage(
                    self.image_dict["previous"].pixmap().toImage().copy()
                )
                q_image = self.image_dict["previous"].pixmap().toImage()
                q_image.fill(Qt.black)  # アノテーションされた部分すべてクリア
                pixmap_item = QPixmap.fromImage(q_image)
                self.image_dict["previous"].setPixmap(pixmap_item)
            else:
                return

    # ハンドボタンが押された際のスロット
    @pyqtSlot()
    def hand(self):
        if self._hand:  # ハンドモードをオフ
            self._hand = False
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
        else:
            self._hand = True  # ハンドモードをオン
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)

    # 拡大ボタンが押された際のスロット
    @pyqtSlot()
    def zoom(self, factor=1.5):
        if 0.95 < self.total_scaling * factor < 50:  # 拡大に範囲を設けている
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)

    # 縮小ボタンが押された際のスロット
    @pyqtSlot()
    def unzoom(self, factor=1 / 1.5):
        if 0.95 < self.total_scaling * factor < 50:  # 縮小に範囲を設けている
            self.total_scaling *= factor
            self.imageViewer.scale(factor, factor)

    # Ctrl+Zが押された際のスロット
    @pyqtSlot()
    def undo(self):
        self._drawing = False
        self.change_count -= 1
        self.image_dict["previous"].setPixmap(self.undo_image)

    # 画像を保存する際のスロット．メニューで ファイル＞Save から実行される
    @pyqtSlot()
    def saveImage(self):
        filePath = os.path.join(
            self.output_folder,
            self.cmc.id + self.cmc.option + "_gt.png",
        )
        img = self.qimage_to_cv(
            self.image_dict["previous"].pixmap().toImage()
        )  # qimageをndarrayに変換
        gray_img = img[:, :, 2].copy()  # 赤色のみを書き出し
        gray_img[gray_img > 0] = 255  # 二値化
        self.change_count = 0

        # 日本語パスに対応した保存
        _, buf = cv2.imencode("*.png", gray_img)
        buf.tofile(filePath)

    # チェックボックスが押された際のスロット
    @pyqtSlot()
    def toggle_image(self, state, key):
        if self.image_dict[key]:
            if state:  # チェックボックスがactiveなら
                self.image_dict[key].setVisible(True)
            else:
                self.image_dict[key].setVisible(False)

    # Previousボタンが押された際のスロット
    @pyqtSlot()
    def prevFile(self):
        if self.closeEvent():
            self.total_scaling = 1  # リセット
            current_row = self.listWidget.currentRow()
            total_items = self.listWidget.count()
            try:
                next_row = (current_row - 1) % total_items
            except ZeroDivisionError:
                return
            self.listWidget.setCurrentRow(next_row)

    # Nextボタンが押された際のスロット
    @pyqtSlot()
    def nextFile(self):
        if self.closeEvent():
            self.total_scaling = 1  # リセット
            current_row = self.listWidget.currentRow()
            total_items = self.listWidget.count()
            try:
                next_row = (current_row + 1) % total_items
            except ZeroDivisionError:
                return
            self.listWidget.setCurrentRow(next_row)

    # Qimageをndarrayに変換するメソッド
    def qimage_to_cv(self, qimage):
        w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
        bytes_ = qimage.bits().asstring(w * h * d // 8)
        arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
        return arr

    # リストのアイテムを読み込むメソッド
    def load(self, current):
        self.closeEvent()  # 保存
        self.scene.clear()
        self.total_scaling = 1  # リセット

        # CMCインスタンスを作成
        # NumPy配列からQImageに変換
        self.cmc = CMC(os.path.join(self.input_folder, current.text()), self.w, self.h)
        raw_image = normalize(
            self.cmc.data, value_range=(self.min, self.max)
        )  # 正規化したraw画像

        self.ft_image = self.cmc.filtered  # フーリエ変換のフィルター画像

        height, width = raw_image.shape  # 画像サイズ

        for key, image, cb in zip(  # キー・画像・チェックボックス
            self.image_dict,
            (raw_image, self.ft_image, np.zeros((height, width, 3), dtype=np.uint8)),
            (self.checkBox_Raw, self.checkBox_Filtered, self.checkBox_Previous),
        ):
            if key == "previous":  # アノテーション画像の場合
                q_image = self.find_previous_png(image, height, width)
            else:  # rawやfiltered画像の処理
                q_image = QImage(
                    image.copy(),
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8,
                )

            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
            self.image_dict[key] = pixmap_item  # 辞書に画像を登録
            if key == "previous":
                pixmap_item.setOpacity(0.3)  # 透明度の設定
            self.scene.addItem(pixmap_item)  # 画像の表示
            if self.first_load:
                cb.setChecked(True)  # チェックボックスの更新
            else:
                pixmap_item.setVisible(cb.isChecked())
            self.imageViewer.fitInView(pixmap_item, Qt.KeepAspectRatio)  # サイズ調整

        self.first_load = False  # 次回のロードからは今のチェックボックス状態に依存

    def find_previous_png(self, image, height, width):
        previous_item = self.listWidget.item(self.listWidget.currentRow() - 1)
        previous_load = CMC.find_load(previous_item.text())[0] if previous_item else ""

        file_list = [
            f"{self.cmc.id}{self.cmc.option}_gt.png",
            f"{previous_load}_gt.png",
        ]

        for i, f_name in enumerate(file_list):
            full_name = self.output_folder + "\\" + f_name
            try:
                # 　現在開いているファイルと名前が一致する画像を開く
                # グレースケールで読み込む
                _img = cv2.imdecode(
                    np.fromfile(
                        full_name,
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
                if i != 0:
                    # 明らかにき裂である画素をオレンジ色で表示
                    ft_image = np.copy(self.ft_image)
                    ft_image[self.cmc.edge < 128] = 0
                    image[ft_image > 20] = ORANGE
                    image[self.base_edge > 0] = BLACK
                    image[self.cmc.data < np.percentile(self.cmc.data, 75)] = BLACK
                image[np.where(_img == 255)] = RED  # 赤色で表示
                break  # 対象画像が見つかった時点で終了
            except (UnboundLocalError, FileNotFoundError):  # ファイルがない場合
                pass

        else:  # アノテーション済みの場合は自動アノテーションしない
            ft_image = np.copy(self.ft_image)
            ft_image[self.cmc.edge < 128] = 0
            image[ft_image > 20] = ORANGE
            image[self.base_edge > 0] = BLACK
            image[self.cmc.data < np.percentile(self.cmc.data, 75)] = BLACK

        q_image = QImage(image.copy(), width, height, width * 3, QImage.Format_RGB888)

        return q_image

    # エラーダイアログを表示するメソッド
    def show_error_dialog(self, error_message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText("An error occurred:")
        error_dialog.setInformativeText(error_message)  # 任意のメッセージ
        error_dialog.exec_()

    # クリックした位置で描画を行うメソッド
    def draw(self, event):
        # クリックした位置を取得
        pos_scene = self.imageViewer.mapToScene(event.pos())
        x, y = int(pos_scene.x()), int(pos_scene.y())

        image = self.image_dict["previous"].pixmap().toImage()  # pixmapをqimageに変換
        if 0 <= x < image.width() and 0 <= y < image.height():  # 画像内のクリックのみ
            self.change_count += 1
            if event.buttons() & Qt.RightButton:  # 右クリック
                if self.diameter > 1:
                    # 削除する円の範囲
                    painter = QPainter(image)
                    painter.setPen(
                        QPen(
                            QColor(0, 0, 0), 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin
                        )
                    )
                    painter.setBrush(QColor(0, 0, 0))

                    # 中心座標(x, y)に直径(diameter)の円を描く
                    painter.drawEllipse(
                        x - self.diameter // 2,
                        y - self.diameter // 2,
                        self.diameter,
                        self.diameter,
                    )
                    painter.end()

                else:
                    image.setPixelColor(
                        x, y, QColor(0, 0, 0)
                    )  # クリックしたピクセルを黒くする
            elif event.buttons() & Qt.LeftButton:  # 左クリック
                image.setPixelColor(
                    x, y, QColor(225, 147, 56)
                )  # クリックしたピクセルをオレンジに

        # キャンバスの画像を QPixmap に変換して表示
        canvas_pixmap = QPixmap.fromImage(image)
        self.image_dict["previous"].setPixmap(canvas_pixmap)

    # ウィンドウサイズが変更された際に呼び出されるメソッド
    def resizeEvent(self, event):
        for img in self.image_dict.values():
            if img:
                self.imageViewer.fitInView(img, Qt.KeepAspectRatio)
        super().resizeEvent(event)

    # 画面を閉じる際や、次のファイルに移る際の処理
    def closeEvent(self, event=None):
        if self.change_count > 0:  # 変更がなければ保存しない
            self.change_count = 0  # リセット
            reply = QMessageBox.question(
                self,
                "Message",
                "Do you want to save changes?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )  # ダイアログ画面

            # 「キャンセル」が押された場合のみ処理を中断するためにFalseを戻り値とする
            if reply == QMessageBox.Yes:
                # 保存処理を行う
                self.saveImage()
                if event:
                    event.accept()
                else:
                    return True
            elif reply == QMessageBox.No:
                if event:
                    event.accept()
                else:
                    return True
            else:
                if event:
                    event.ignore()
                else:
                    return False
        else:
            if not event:
                return True

    # タブキーが押された場合の処理
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.toggle_image(True, "previous")  # 表示
            self.checkBox_Previous.setChecked(True)
        if event.key() == Qt.Key_Q:
            self.toggle_image(True, "filtered")  # 表示
            self.checkBox_Filtered.setChecked(True)

        if event.key() == Qt.Key_B:
            self.prevFile()
        elif event.key() == Qt.Key_N:
            self.nextFile()

        # キーボードの1～3キーで削除領域の直径を変更
        if event.key() == Qt.Key_1:
            self.diameter = 1
        elif event.key() == Qt.Key_2:
            self.diameter = 4
        elif event.key() == Qt.Key_3:
            self.diameter = 8

    # タブキーがリリースされた場合の処理
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.toggle_image(False, "previous")  # 非表示
            self.checkBox_Previous.setChecked(False)
        if event.key() == Qt.Key_Q:
            self.toggle_image(False, "filtered")  # 表示
            self.checkBox_Filtered.setChecked(False)

    # マウスクリックイベント
    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:  # 　中ボタンをクリックした場合はドラッグ
            self.imageViewer.setDragMode(QGraphicsView.ScrollHandDrag)
            event = QMouseEvent(
                QEvent.GraphicsSceneDragMove,
                event.pos(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QGraphicsView.mousePressEvent(self.imageViewer, event)

        else:  # それ以外はdrawメソッドで処理
            if self.image_dict["previous"] and self._hand == False:
                self._drawing = True
                self.undo_image = QPixmap.fromImage(
                    self.image_dict["previous"].pixmap().toImage().copy()
                )
                self.draw(event)
            else:
                QGraphicsView.mousePressEvent(self.imageViewer, event)

    # マウス移動イベント
    def mouseMoveEvent(self, event):
        if self._drawing:  # 描画中にマウスが動いた場合はその途中のピクセルも塗る
            self.draw(event)
        QGraphicsView.mouseMoveEvent(self.imageViewer, event)

    # マウスを離した際のイベント
    def mouseReleaseEvent(self, event):
        if self._hand == False:
            self.imageViewer.setDragMode(QGraphicsView.NoDrag)
            if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
                self._drawing = False  # 描画終了
        QGraphicsView.mouseReleaseEvent(self.imageViewer, event)

    # 中ボタン(ホイール)回転イベント
    def wheelEvent(self, event):
        if self.imageViewer.geometry().contains(self.mapFromGlobal(event.globalPos())):
            numDegrees = event.angleDelta().y() / 8
            numSteps = numDegrees / 15
            self.numScheduledScalings += numSteps
            if self.numScheduledScalings * numSteps < 0:
                self.numScheduledScalings = numSteps
            # アニメーションとすることでスムーズな拡大縮小が可能
            self.scale_animation = QTimeLine(350, self)
            self.scale_animation.setUpdateInterval(20)
            self.scale_animation.valueChanged.connect(self.scalingTime)
            self.scale_animation.finished.connect(self.animFinished)
            self.scale_animation.start()

    # 画像拡大をアニメーションによってスムーズに行うメソッド
    def scalingTime(self, x):
        factor = 1.0 + float(self.numScheduledScalings) / 300.0
        if 0.95 < self.total_scaling * factor < 50:  # 拡大縮小に範囲を設ける
            self.total_scaling *= factor
            self.imageViewer.setTransformationAnchor(
                QGraphicsView.AnchorUnderMouse
            )  # マウス中心に拡大縮小
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


if __name__ == "__main__":
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
ok・クリアボタンが欲しい ⇒消しゴムボタンの代わりにするのが良さそう
ok・拡大する時にマウスの位置を中心に拡大縮小を行いたい(現在は画面中央を拡大している)
ok・拡大と縮小に最大値を設ける
ok・ショートカットキーが使えるようにする
ok・透明度の設定だけではわかりづらいからRawとFilteredをグレースケールではなく、色を付けて表示する？
ok・exe化
ok・周波数フィルタリング⇒輝度値が高い画素をき裂として事前に塗ることで作業量の軽量化
ok・tabボタンでmasking画像の表示/非表示
ok・ディレクトリのファイルを全てまとめて開く(次へボタンを開く)
ok・ファイル一覧を表示する
ok・画面を閉じる際に「保存しますか？」と聞く

5/23～
ok・リストエリアでスクロールした場合も画像の拡大縮小が起きるバグの改善
ok・画像の切り取りを内部では行わず、指示された画像をそのまま読み込む仕様に変更
ok・ULとそうでないものを区別し、previousの読み込みを適切に行う
ok・Qキーでフィルター画像を表示/非表示
ok・チェックボックスや透明度の設定は次の画像になっても保存する
・ゼロ負荷時の画像との差分をとってからフーリエフィルターをかける
・透明化はスライダーのみではなく、数字でも表示する
・「Previous」ではなく、「Canvas」などのより適切な表示にする
ok・筆の大きさを変えられるようにする
ok・電極部分は基準画像からの差分で自動削除を行うようにする
ok・パラメータファイルを読み込むことで手入力の数値を無くす
"""
