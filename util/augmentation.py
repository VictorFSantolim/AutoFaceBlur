# Mirror horizontaly all images in srcDir, and save them in outDir
import cv2
import os

# Source and output images folder names. Must be in the same folder as augmentation.py
srcDir = "src"
outDir = "out"

if not os.path.exists(outDir):
    os.mkdir(outDir)

for imgNameExt in os.listdir(srcDir):

	image = cv2.imread(os.path.join(srcDir , imgNameExt),0)
	mirrorImg = cv2.flip(image, +1);

	imgName , ext = imgNameExt.rstrip().split('.' , 1)
	mirrorImgName = imgName + "_flip." + ext
	print("Saving mirrored version of "+ imgNameExt +" from "+ srcDir +" as "+ mirrorImgName)
	cv2.imwrite(os.path.join(outDir , mirrorImgName) , mirrorImg)





