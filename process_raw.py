import re
import os
import cv2
import numpy as np


class CMC:

    def __init__(
        self,
        input_file,  # .raw形式のデータファイル名
        w=1344,  # 画像データの幅
        h=1344,  # 画像データの高さ
    ):

        self.file_name = re.match(r".*[\\/](.*\.raw)$", input_file).group(1)
        self.w, self.h = w, h

        # データの読み込み
        with open(input_file, "rb") as rf:
            rawdata = rf.read()
            self.data = np.frombuffer(rawdata, dtype=np.int16).reshape(w, h)

    def crop(self, tl, br, output_folder):
        with open(output_folder + "/" + self.file_name, "wb") as wf:
            wf.write(self.data[tl[1] : br[1], tl[0] : br[0]].tobytes())

    def rotate(self, deg, output_folder):
        with open(output_folder + "/" + self.file_name, "wb") as wf:
            mat = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), int(deg), 1.0)
            rotated_data = cv2.warpAffine(self.data, mat, (self.w, self.h))
            wf.write(rotated_data.tobytes())


if __name__ == "__main__":
    input_folder = "raw/3_ex_cropped/"
    output_folder = "raw/3_ex_cropped/"

    # ex_1
    # tl=(392, 532)  # 切り取り用(左上座標)
    # br=(966, 662)  # 切り取り用(右下座標)
    # w, h = (574, 130)
    # ex_3
    tl = (341, 514)
    br = (976, 702)
    # w, h = (635, 188)

    raw_files = [
        file for file in os.listdir(input_folder) if file.endswith(".raw")
    ]  # .raw画像の取得
    for raw in raw_files:
        cmc = CMC(input_folder + raw)
        # cmc.rotate(15, output_folder)
        # cmc.crop(tl, br, output_folder)
