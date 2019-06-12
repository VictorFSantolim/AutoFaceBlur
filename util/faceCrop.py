# Covers all faces anotated in WIDERFace anotationFile with black rectangles, and saves them on outputFolder
from pathlib import Path
import cv2

# Source anotation file name and output images folder names. Must be in the same folder as faceCrop.py
anotationFile = "wider_face_train_bbx_gt.txt"
outputFolder = "saida/"

anotations = open(anotationFile , "r")
while True:
	fullPath = anotations.readline()
	if(fullPath == ''):
		break

	fullPath = fullPath.rstrip()
	folderName , imageName = fullPath.split('/' , 1)

	print("Reading image: " + fullPath)
	image = cv2.imread(fullPath)

	height, width, channels = image.shape

	numFaces = int(anotations.readline())
	print("Num of faces: " + str(numFaces))

	for i in range(numFaces):
		localFaceLine = anotations.readline()
		faceLocation = [int(s) for s in localFaceLine.split() if s.isdigit()]
		print("Face Location " + str(i) + " : " + str(faceLocation))
		x = faceLocation[0]
		y = faceLocation[1]
		w = faceLocation[2]
		h = faceLocation[3]

		# Makes the rectangles larger, so we are sure all head is covered.
		x2 = (x - int(w/2)) if (x - int(w/2))>=0 else 0
		y2 = (y - int(h/2)) if (y - int(h/2))>=0 else 0
		w2 = int(w*1.5) if (int(w*1.5)+x)<width else width-x-1
		h2 = int(h*1.5) if (int(h*1.5)+y)<height else height-y-1

		cv2.rectangle(image, (x2, y2), (x + w2, y + h2), (0,0,0), -1)

	outputPath = outputFolder + imageName
	cv2.imwrite(outputPath,image)

anotations.close()







