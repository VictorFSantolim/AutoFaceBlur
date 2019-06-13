# Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

# Python modules
import os
import time
import shutil
import argparse
import tempfile

# Third-party modules
import cv2
import numpy as np

# Own modules
from tracker import faceObject
from tracker import faceTracker

def mainLoop(video_source, cascade_source, show_processing=True, output_file=None, \
        variable_fps=False, save_fps=False, show_fps=True):
    """Main blur processing loop.
    Does the processing in real time, from a cam source, or from a video source,
    identifying an object, tracking it, blurring it, and optionally saving the
    processed output to a file.

    Arguments:
    video_source -- The source of the video, a string for a video file, or an
    integer for a cam source. 

    cascade_source -- The source of the cascade classifier file used to identify
    the object.

    show_processing -- Shows a window while processing the video if True, else
    does it in the background

    output_file -- If not None must be a string to a path where the processed
    video file will be stored

    variable_fps -- Used by tracker, if True averages the FPS for the tracking
    to be smoother.

    save_fps -- If True saves the fps to the output_file

    show_fps -- If True shows the fps in the window if show_processing is also True.
    """
    # Initializes the capture
    cap = cv2.VideoCapture(video_source)

    # Initializes the window
    if show_processing:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    # Initializes the tracker
    tracker = faceTracker()
    
    # Initializes the fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    calculated_fps = fps

    # Gets width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initializes the object classifier
    cascade_classifier = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

    # Initilizes video_writer
    video_out_writer = None
    if output_file is not None:
        video_out_writer = cv2.VideoWriter(output_file, \
                cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    # Timing start
    time_start = time.time()
    # Reads a frame
    ret , frame = cap.read()
    while ret:
        # Makes the frame gray scale (used for haar processing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects the faces using the cascade classifier
        front_faces = cascade_classifier.detectMultiScale(gray, 1.3, 3)
        frameFacesList = list()
        for rect in front_faces:
            (x, y, w, h) = rect
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = faceObject(rect)
            frameFacesList.append(face)

        # Tracks the objects to make sure we don't lose if there is an issue with
        # haar classifier
        tracker.update(frameFacesList)
        if variable_fps:
            registryFacesList = tracker.getFaces(framerate=calculated_fps)
        else:
            registryFacesList = tracker.getFaces(framerate=fps)
        
        # Blurs each of the identified and tracked objects
        for rect in registryFacesList:
            (x, y, w, h) = rect
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            ksize = 2*int(0.15*max(w, h)) + 1
            sub_frame = cv2.GaussianBlur(frame[y:(y+h), x:(x+w)], (ksize, ksize), sigmaX=20)
            frame[y:(y+h), x:(x+w)] = sub_frame

        # Resets these variables for the next loop iteration, prevents some nasty bugs
        del frameFacesList
        del registryFacesList

        # Times the cycle end and updates fps
        time_end = time.time()
        calculated_fps = 1/(time_end - time_start)

        # Saves the video
        if video_out_writer is not None:
            if save_fps:
                out_frame = frame.copy()
                if variable_fps:
                    cv2.putText(out_frame, str(calculated_fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)
                else:
                    cv2.putText(out_frame, str(fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)
                video_out_writer.write(out_frame)
            else:
                video_out_writer.write(frame)

        # Shows while processing
        if show_processing:
            if show_fps:
                # Shows the realtime fps
                cv2.putText(frame, str(calculated_fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)

            # Shows the frame with detected faces
            cv2.imshow("Frame", frame)

            # If Esc(= key 27) is pressed close the window and stops processing.
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        # Timing start
        time_start = time.time()
        # Reads a new frame
        ret , frame = cap.read()

    # Releases the capture and closes the spawned windows
    cap.release()
    if show_processing:
        cv2.destroyAllWindows()
    if video_out_writer is not None:
        video_out_writer.release()

if __name__ == "__main__":
    # Lida com os argumentos da linha de comando
    parser = argparse.ArgumentParser(description="""Processes a video from a camera \
            source or video file source, blurs a detected object in the image using \
            a trained Haar cascade classifier while tracking that object to make sure \
            all the frames are properly blurred.""")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-c", "--camera", help="Integer representing the camera souce index", type=int, default=None)
    input_group.add_argument("-v", "--video", help="Path to the video file that will be processed.", type=str, default=None)

    parser.add_argument("cascade_source", help="Path to the trained haar cascade classifier source file.", type=str)
    parser.add_argument("-o", "--output_file", help="Path where the processed video file will be stored.", type=str, default=None)
    parser.add_argument("-p", "--hide_processing", help="Shows a window while processing the video.", action="store_false")
    parser.add_argument("--variable_fps", help="Takes into account the processing time of the frame\
            to track the object, only recommended for camera input.", action="store_true")
    parser.add_argument("--show_fps", help="Shows fps in the a window while processing the video.", action="store_true")
    parser.add_argument("--save_fps", help="Saves the video fps as it was captured to the video output_file.", action="store_true")

    args = parser.parse_args()

    if args.output_file is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_output_file = os.path.join(tmp_dir, "output_file")
            mainLoop(args.video if args.video else args.camera,\
                    args.cascade_source,\
                    show_processing=args.hide_processing,\
                    output_file=temp_output_file,\
                    variable_fps=args.variable_fps,\
                    save_fps=args.save_fps, show_fps=args.show_fps)
            os.makedirs(os.path.abspath(os.path.dirname(args.output_file)), exist_ok=True)
            shutil.move(temp_output_file, args.output_file)
    else:
        mainLoop(args.video if args.video else args.camera,\
                args.cascade_source,\
                show_processing=args.hide_processing,\
                output_file=args.output_file,\
                variable_fps=args.variable_fps,\
                save_fps=args.save_fps, show_fps=args.show_fps)
