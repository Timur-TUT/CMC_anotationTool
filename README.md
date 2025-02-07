# Tool for annotating SAXS images of CMCs

This tool reads ".raw" files of CMC's SAXS images and assists in the creation of crack training data for machine learning.

このツールは、CMCの小角散乱画像の".raw "ファイルを読み込み、機械学習用のき裂教師データの作成を支援します。

## Description

This tool automatically loads folders containing raw files. By comparing consecutive images, it becomes easier to distinguish between noise and cracks. Multiple image processing filters are applied to enhance automatic crack detection, reducing manual workload. Additionally, previously annotated areas are retained, preventing redundant work.

## Getting Started

### Dependencies

* PyQt5          | 5.15.7
* numpy          | 1.23
* opencv-python  | 3.4

### Installing

Move to your repository
```
git clone https://github.com/Timur-TUT/CMC_anotationTool.git
```

### Executing program

* Execute anotation_tool_gui.py
```
python anotation_tool_gui.py
```

## How To Use

1. Press `Ctrl + O` to open a directory containing raw images.
2. If necessary, specify a directory to save annotation images. If not specified, an "output" folder is created automatically.
3. Use the `Q` key to check the next image and distinguish between noise and cracks.
4. If needed, adjust the transparency of each image using the slider.
5. Use the left mouse button to annotate crack areas.
6. After finishing the annotation, select the next image. The save dialog will prompt you to confirm image saving.

| User Action    | Operation                             |
|---------------|---------------------------------|
| Left Click    | Annotate cracks                 |
| Right Click   | Erase annotations               |
| Scroll        | Zoom in/out                      |
| Middle Click  | Drag to move the image          |
| `1`, `2`, `3` | Change annotation radius size   |
| `Tab`         | Toggle mask image visibility    |
| `Q`           | Toggle next image visibility    |
| `B`           | Select the previous image       |
| `N`           | Select the next image           |

## Authors

T. A. Khudayberganov  
[@Timur](g212300905@edu.teu.ac.jp)

## Version History

* 1.0
    * Major feature enhancements, including automatic folder loading, multi-image comparison, improved annotation retention, and various image processing filters.

* 0.1
    * Initial Release.