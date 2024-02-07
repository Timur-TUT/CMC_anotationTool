# Tool for annotating SAXS images of CMCs

This tool reads ".raw" files of CMC's SAXS images and assists in the creation of crack training data for machine learning. 

## Description

Raw images are read and frequency filtered to make the boundary between cracks and noise clearer than before, making it easier for the layperson to annotate cracks. In addition, if there are similar images or images to be continuously annotated, those image files can be loaded to improve work efficiency.

## Getting Started

### Dependencies

* PyQt5          | 5.15.7
* numpy          | 1.23
* opencv-python  | 3.4

### Installing

move to your repository
```
git clone https://github.com/Timur-TUT/CMC_anotationTool.git
```

### Executing program

* execute action_tool_gui.py
```
python action_tool_gui.py
```

## How To Use

1. Open ".raw" image (Short Cut: "Ctrl + Shift + O")
2. Open masking image if needed (Short Cut: "Ctrl + O")
3. Use Filtered image for masking
4. When finished, make final checks and minor corrections using Raw images
5. Save image as ".png" file (Short Cut: "Ctrl + S")

| User Action    | Operation        |
|---------------|-------------|
| Left Click    | Draw        |
| Right Click   | Erase       |
| Middle Click  | Drag        |
| Middle Scroll | Zoom/Unzoom |

## Authors

T. A. Khudayberganov
[@Timur](g212300905@edu.teu.ac.jp)

## Version History

* 0.1
    * Initial Release
