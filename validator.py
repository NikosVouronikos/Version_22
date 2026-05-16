import sys, os, numpy as np
from math import sqrt
from codeMapping import getBlockDimensions,getListFromCode,getSipsFromCode,isNumericString
from PIL import Image
from initializer import openImage,init
from encodeinteger import encodeInteger
from decodesip import decodeSip
from optimizer import printResults
from utilities import decodeKey
from initializer import ExtractionResult

# Author: Nikolaos Vouronikos
def getWatermarkedBlocksInList(code, imagePath):
	watermarkedBlocks = []
	img = openImage(imagePath)
	M,N = img.size
	imageArray = np.array(img)

	size = sqrt(len(code))										
	blockWidth,blockHeight = getBlockDimensions(M, N, size)

	for offsetY in range(0, (N - blockHeight + 1), blockHeight):
		for offsetX in range(0, (M - blockWidth + 1), blockWidth):	
			blockArray = imageArray[offsetY:(offsetY + blockHeight), offsetX:(offsetX + blockWidth)]
			blockImage = Image.fromarray(blockArray)
			watermarkedBlocks.append(blockImage)
	return watermarkedBlocks	

# Author: Nikolaos Vouronikos
def getMapping(code, imagePath):
	mappingPath = os.path.join(os.path.dirname(imagePath), "Code_Mapping.txt")
	mapping = {}

	try:
		f = open(mappingPath, "r")
	except :
		print("File cannot be opened")
		exit(1)

	for i in range(len(code)):
		fileLine = f.readline()
		lineSplit = fileLine.split(",")
		if(isNumericString(lineSplit[1])):
			mapping[int(lineSplit[0])] = int(lineSplit[1])
		else:
			mapping[int(lineSplit[0])] = lineSplit[1].strip()
	return mapping

# Author: Nikolaos Vouronikos
def getInnerSiPs(codeSips):
	innerSiPs = []
	for i in range(len(codeSips)):
		innerKey = codeSips[i]
		innerSiP = encodeInteger(innerKey)
		innerSiPs.append(innerSiP)
	return innerSiPs

# Author: Nikolaos Vouronikos
def getBasicValues(imagePath):
	basicValuesPath = os.path.join(os.path.dirname(imagePath), "BasicValues.txt") 

	try:
		f = open(basicValuesPath, "r")
	except :
		print("File cannot be opened")
		exit(1)

	firstLine = f.readline()
	secondLine = f.readline()
	thirdLine = f.readline()
	fourthLine = f.readline()

	gridSize = [int(firstLine.split(",")[0]), int(firstLine.split(",")[1])]
	RBWidth = [int(secondLine.split(",")[0]), int(secondLine.split(",")[1])]
	Rxy = [int(thirdLine.split(",")[0]), int(thirdLine.split(",")[1])]
	Bxy = [int(fourthLine.split(",")[0]), int(fourthLine.split(",")[1])]

	return gridSize,RBWidth,Rxy,Bxy

# Author: Nikolaos Vouronikos
def getGridPositions(code, imagePath):
	positionsPath = os.path.join(os.path.dirname(imagePath), "GridPositions.txt")
	allPositions = []

	try:
		f = open(positionsPath, "r")
	except :
		print("File cannot be opened")
		exit(1)

	for i in range(len(code)):
		fileLine = f.readline()
		lineSplit = fileLine.split(",")
		position = [int(lineSplit[0]), int(lineSplit[1])]
		allPositions.append(position)
	return allPositions

# Author: Nikolaos Vouronikos
def extract(watermarkedBlocks, codeSips, mapping, innerSips, gridSize, RBWidth, Rxy, Bxy, allGridPositions):
	codeTaken,totalWatermarksExtracted = [],16
	for i in range(len(watermarkedBlocks)):
		watermarkedBlock,sip,bestMove,enable = watermarkedBlocks[i],codeSips[i],allGridPositions[i],1
		isExtracted,key1,key2,key3 = extractSiP(watermarkedBlock, sip, innerSips[i], gridSize, RBWidth, Rxy, Bxy, bestMove)
		decodedKey = decodeKey(key1, key2, key3, sip)
		if(decodedKey == "X"):
			enable = 0
			codeTaken.append(decodedKey)
			totalWatermarksExtracted = totalWatermarksExtracted - 1
		else:
			codeTaken.append(mapping[decodedKey])
		printResults(3, i, enable, 0, [])
	extractionRate = (totalWatermarksExtracted / 16)*100
	extractionResult = ExtractionResult(codeTaken, extractionRate)
	return extractionResult

# Author: Nikolaos Vouronikos
def extractSiP(watermarkedBlock, originalKey, innerSip, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock):
	em,ex = init()
	sip1,sip2,sip3 = ex.getSip(watermarkedBlock, len(innerSip), gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	print("Key which was embeded :", originalKey)
	key1 = decodeSip(sip1)
	key2 = decodeSip(sip2)
	key3 = decodeSip(sip3)
	if(key1 == originalKey or key2 == originalKey or key3 == originalKey) :
		return 1,key1,key2,key3
	return 0,key1,key2,key3

def runValidation(imagePath, code):
    code = getListFromCode(code)
    watermarkedBlocks = getWatermarkedBlocksInList(code, imagePath)
    mapping = getMapping(code, imagePath)
    codeSiPs = getSipsFromCode(mapping, code)
    innerSiPs = getInnerSiPs(codeSiPs)
    gridSize,RBWidth,Rxy,Bxy = getBasicValues(imagePath)
    allGridPositions = getGridPositions(code, imagePath)
    extractionResult = extract(watermarkedBlocks, codeSiPs, mapping, innerSiPs, gridSize, RBWidth, Rxy, Bxy, allGridPositions)
    print(str(extractionResult.codeTaken) + "\n")
    print(str(extractionResult.extractionRate) + "%")

if __name__ == '__main__':
	# Initialization from command line
	# Example: py validator.py watermarked/watermarked_people/watermarked_people.jpg 56728192afd67fca
	imagePath, code = sys.argv[1:3]
	runValidation(imagePath, code)