# PinPointPrevent

### To Run the program, you will first need.
- Python 3.x (We used python 3.6.5 for this project)
- OpenCV2 (We used version 3.4.1)
- Dlib (We used version 19.13.0)
- Numpy (We used version 1.14.3)

### For graphical version:
```bash
python3 main.py [training-data-directory] [number-of-training-images]
```
"training-data-directory" should be a directory inside the "training" directory

### For command-line version (this version will output a confidence)
```bash
python3 main.py [training-data-directory] [number-of-training-images] -i [path-to-test-data]
```
"training-data-directory" should be a directory inside the "training" directory
"path-to-test-data" should be a path to an image file (png,jpg,etc.)
