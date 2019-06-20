# AutoFaceBlur

Processes a video or image from a camera source or file source, blurs a detected
human face object in the image using a trained Haar cascade classifier while tracking
that object to make sure all the frames are properly blurred. Made with real-time
processing in mind.

## Requirements
_Python 3_ with _NumPy_ and _OpenCV 3.1+_. 

## Installing

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS
	- On Windows, use Anaconda Powershell Prompt as admin
	- On Linux and Mac OS, install the .sh file and use the regular bash
- Install OpenCV
```
$ conda install -c conda-forge opencv
```
- Finally clone the repository
```
$ git clone https://github.com/VictorFSantolim/AutoFaceBlur.git
```

## Usage examples

* Webcam input
```
$ python src/autofaceblur.py -c 0
```
* Image input and output
```
$ python src/autofaceblur.py -i assets/women.jpg -o assets/women_blur.jpg
```
* Video input and output, using the classifier we trained, and showing fps
```
$ python src/autofaceblur.py cascades/myCascade3.xml -v src_video_path -o blurred_video_path --show_fps
```

## Sample output

![Imgur](assets/women_blur.jpg)

## Args and options

`-h` or `--help` show the help message with all args and options and exit\

#### Required: One of the following
`-c CAMERA` or `--camera CAMERA` select camera index CAMERA as the input source\
`-v VIDEO` or `--video VIDEO` select path VIDEO as the video input source\
`-i IMAGE` or `--image IMAGE` select path IMAGE as the image input source

#### Optional
`-s CASCADE` or `--cascade_source CASCADE` Defines CASCADE the trained Haar cascade classifier path. If the argument is absent, uses haarcascade_frontalface_default.xml.\
`-o OUTPUT_FILE` or `--output_file OUTPUT_FILE` Enable output storing, saves the processed result to OUTPUT_FILE\
`-p` or `--hide_processing` Runs processing in background, without displaying the frames being processed\
`--variable_fps` Takes into account the processing time of the frame to track the object, only recommended for camera input.\
`--show_fps` Shows fps in the a window while processing the video.\
`--save_fps` Saves the video fps as it was captured to the video output_file.

## References

[OpenCV Haarascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)\
[YOLOFace](https://github.com/sthanhng/yoloface)
