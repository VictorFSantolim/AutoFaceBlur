# Mirror horizontaly all images in srcDir, and save them in outDir
from scipy import ndimage
import cv2
import os

# Source and output images folder names. Must be in the same folder as augmentation.py
srcDir = "src"
outDir = "out"

if not os.path.exists(outDir):
    os.mkdir(outDir)

for imageNameExt in os.listdir(srcDir):

	imageName , ext = imageNameExt.rstrip().split('.' , 1)
	image = cv2.imread(os.path.join(srcDir , imageNameExt),0)

	# Just saves the grayscale version of the original image
	cv2.imwrite(os.path.join(outDir , imageNameExt) , image)

	# Rotates grayscale original image counter clock-wise (CCW) and saves
	rotatedImageCCW = ndimage.rotate(image, 10, reshape=False)
	rotatedImageCCWName = imageName + "_rotCCW." + ext
	cv2.imwrite(os.path.join(outDir , rotatedImageCCWName) , rotatedImageCCW)

	# Rotates grayscale original image clock-wise (CW) and saves
	rotatedImageCW = ndimage.rotate(image, -10, reshape=False)
	rotatedImageCWName = imageName + "_rotCW." + ext
	cv2.imwrite(os.path.join(outDir , rotatedImageCWName) , rotatedImageCW)

	# Mirrors horizontaly original grayscale image and saves
	mirrorImage = cv2.flip(image, +1);
	mirrorImageName = imageName + "_mir." + ext
	cv2.imwrite(os.path.join(outDir , mirrorImageName) , mirrorImage)

	# Rotates horizontaly mirrored image counter clock-wise (CCW) and saves
	rotatedMirrorImageCCW = ndimage.rotate(mirrorImage, 10, reshape=False)
	rotatedMirrorImageCCWName = imageName + "_mir_rotCCW." + ext
	cv2.imwrite(os.path.join(outDir , rotatedMirrorImageCCWName) , rotatedMirrorImageCCW)

	# Rotates horizontaly mirrored image clock-wise (CW) and saves
	rotatedMirrorImageCW = ndimage.rotate(mirrorImage, -10, reshape=False)
	rotatedMirrorImageCWName = imageName + "_mir_rotCW." + ext
	cv2.imwrite(os.path.join(outDir , rotatedMirrorImageCWName) , rotatedMirrorImageCW)

	print("Processing image " + imageNameExt)
	
	





