import numpy as np,itertools as it, time, os
from datetime import datetime
from PIL import Image
from codeMapping import *
from metrics import *
from initializer import *
from ellipticdisk import createEllipticDisk

# Author: Nikolaos Vouronikos
def calculateBasicValues(blockParams, PR, PB, imageArray, em):
	gridSize,RBWidth,Rxy,Bxy = [],[],[],[]
	qSize,dSize = (4 * blockParams.sipSize),(2 * blockParams.sipSize)
	RED_WIDTH = PR
	RBWidth = [PR, PB]
  	
	RED_RADIOUS_X = math.floor(blockParams.blockHeight / qSize)
	RED_RADIOUS_Y = math.floor(blockParams.blockHeight / qSize)
	Rxy = [RED_RADIOUS_X, RED_RADIOUS_Y]

	BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
	BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
	Bxy = [BLUE_RADIOUS_X, BLUE_RADIOUS_Y]

	gridSize = getGridSize(em, imageArray, Rxy, Bxy, RBWidth)
	return gridSize,RBWidth,Rxy,Bxy

# Author: Vasileios Vouronikos
def mergeCellsToImage(cells, w, h, N) :
	display = np.empty(((h)*N, (w)*N , 3), dtype = np.uint8)
	for i, j in it.product(range(N), range(N)):
		arr = np.array(cells[i*N + j])
		x,y = i*(h), j*(w)
		display[x : x + (h), y : y + (w)] = arr
	return display

# Author: Nikolaos Vouronikos
def writeBestCValuesInFile(optimalCValues, codeTaken, extractionRate, subpath):
	path = os.path.join(subpath, "BestCValues.txt")
	try :
		f = open(path,"w+")
	except :
		print("File cannot be opened")
		exit(1)

	f.write("This file contains optimized c values for the original image.\n\n")
	f.write("Time and Date produced : " + str(datetime.now()) + "\n\n")

	for i in range(len(optimalCValues)):
		f.write("For Block " + str(i + 1) + " the optimized C value is " + str(optimalCValues[i]) + "\n")

	f.write("\n" + "Code we took after final extraction = " + str(codeTaken) + "\n\n")
	f.write("Extraction Rate = " + str(extractionRate) + "%")
	f.close()

# Author: Nikolaos Vouronikos
def writeBasicValuesInFile(gridSize, RBWidth, Rxy, Bxy, subpath):
	path = os.path.join(subpath, "BasicValues.txt")
	try :
		f = open(path,"w+")
	except :
		print("File cannot be opened")
		exit(1)

	f.write(str(gridSize[0]) + "," + str(gridSize[1]) + "\n")
	f.write(str(RBWidth[0]) + "," + str(RBWidth[1]) + "\n")
	f.write(str(Rxy[0]) + "," + str(Rxy[1]) + "\n")
	f.write(str(Bxy[0]) + "," + str(Bxy[1]) + "\n")
	f.close()

# Author: Nikolaos Vouronikos
def writeGridPositionsInFile(gridPositions, subpath):
	path = os.path.join(subpath, "GridPositions.txt")
	try :
		f = open(path,"w+")
	except :
		print("File cannot be opened")
		exit(1)

	for i in range(len(gridPositions)):
		f.write(str(gridPositions[i][0]) + "," + str(gridPositions[i][1]) + "\n")
	f.close()

# Author: Nikolaos Vouronikos
def getCellsFromAttacked(attackedImage, size):
	cells = []
	M,N = attackedImage.size
	channel_array = np.array(attackedImage)
	blockWidth,blockHeight = getBlockDimensions(M, N, size) 
	for r in range(0, (N - blockHeight + 1), blockHeight):
		for c in range(0, (M - blockWidth + 1), blockWidth):
			grid_cell = channel_array[r:r + blockHeight, c:c + blockWidth]
			g_cell = Image.fromarray(grid_cell)
			cells.append(g_cell)
	return cells

# Author: Nikolaos Vouronikos
def getWatermarkedBlock(comCells, index, em, sip, optimalCValue, gridSize, RBWidth, Rxy, Bxy, moves):
	print("Running with c = " + str(optimalCValue))
	g_cell = comCells[index]
	watermarkedBlock = em.getWatermarkedImage(g_cell, sip, len(sip), optimalCValue, 2, 2, gridSize, RBWidth, Rxy, Bxy, moves)
	return g_cell, watermarkedBlock

# Author: Nikolaos Vouronikos
def saveWatermarkedImage(watermarkedImageName, watermarkedImage, dictionary, code, codeSips, extension):
	script_directory = os.path.dirname(os.path.abspath(__file__))
	watermarked_path = os.path.join(script_directory, "watermarked")
	subpath = os.path.join(watermarked_path, watermarkedImageName)

	if not os.path.exists(watermarked_path):
		os.makedirs(watermarked_path)
	if not os.path.exists(subpath):
		os.makedirs(subpath)
	if os.path.exists(os.path.join(subpath, (watermarkedImageName + extension))):
		os.remove(os.path.join(subpath, (watermarkedImageName + extension)))

	watermarkedImage.save((os.path.join(subpath, (watermarkedImageName + extension))), quality = 100)
	mapping_path = os.path.join(subpath, "Code_Mapping.txt")

	try :
		mapping = open(mapping_path,"w+")
	except :
		print("File cannot be opened")
		exit(1)

	for key in dictionary:
		mapping.write(str(key) + "," + str(dictionary[key]) + "\n")

	mapping.close()
	return subpath

# Author: Nikolaos Vouronikos
def calculateElapseTimeAndPrintResults(start, extractionRate):
	# Elapsed time for the algorithm
	print("Extraction percentage = " + str(extractionRate) + "%\n")
	end = time.time()
	secSTR,minSTR = calculateElapseTime(start, end)
	print("Elapsed time = " + str(minSTR) + " mins")
	print("Elapsed time = " + str(secSTR) + " seconds")

# Author: Nikolaos Vouronikos
def calculateElapseTime(start, end):
	elapsedTimeInSeconds = end - start
	elapsedTimeInMinutes = elapsedTimeInSeconds / 60
	secSTR = "{:.3f}".format(elapsedTimeInSeconds)
	minSTR = "{:.3f}".format(elapsedTimeInMinutes)
	return secSTR,minSTR

# Author: Nikolaos Vouronikos
def decodeKey(key1, key2, key3, innerKey):
	if(key1 == innerKey):
		return key1
	elif(key2 == innerKey):
		return key2
	elif(key3 == innerKey):
		return key3
	else:
		return "X"

# Author: Nikolaos Vouronikos
def prepareEmbedding(imagePath, code, mappingMode):
    img = openImage(imagePath)
    M,N = img.size
    imageArray = np.array(img)
    
    code = checkForFullCode(code)							# Check if code is complete or not
    size = sqrt(len(code))								
    mapping = getCodeMapping(mappingMode)					# Get mapping between code digits and watermarks
    codeSips = getSipsFromCode(mapping, code)				# Build watermarks (SiPs) sequence from code sequence using the mapping
    blockWidth,blockHeight = getBlockDimensions(M, N, size)	
    return code, size, mapping, codeSips, blockWidth, blockHeight, imageArray, M, N

# Author: Nikolaos Vouronikos
def isNotExtracted(isExtracted, counter):
    if(isExtracted != True and counter < 5):
        return True
    return False

def getGridSize(em, imageArray, Rxy, Bxy, RBWidth):
	mingrix, mingridy = 1,1
	while True:
		r,g,b = imageArray.split()
		channel_array = np.array(r)
		grid_cell = channel_array[0:mingrix,0:mingridy]
		mag,phase = em.getFFTTransform(grid_cell)
							
		cx = int(grid_cell.shape[0] / 2)
		cy = int(grid_cell.shape[1] / 2)
		red,coord_red = createEllipticDisk(mag,Rxy[0],Rxy[1],RBWidth[0],cx,cy,mingrix,mingridy)
		blue,coord_blue = createEllipticDisk(mag,Bxy[0],Bxy[1],RBWidth[1],cx,cy,mingrix,mingridy)

		if(len(red) == 0 or len(blue) == 0):
			mingrix = mingrix + 1
			mingridy = mingridy + 1
		else:
			break

	return [mingrix, mingridy]