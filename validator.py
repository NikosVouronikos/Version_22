import sys, os, numpy as np,time
from math import sqrt
from codeMapping import getBlockDimensions,getListFromCode,getSipsFromCode,isNumericString
from PIL import Image
from initializer import openImage,init
from encodeinteger import encodeInteger
from decodesip import decodeSip
from optimizer import printResults
from recossip import recsip
from utilities import calculateElapseTime, calculateElapseTime, decodeKey
from initializer import ExtractionResult
from metrics import *

# Author: Nikolaos Vouronikos
def getWatermarkedBlocksInList(code, imagePath):
	watermarkedBlocks = []
	img = openImage(imagePath)
	M,N = img.size
	imageArray = np.array(img)
									
	blockWidth,blockHeight = getBlockDimensions(M, 3, N, 2)

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
		with open(mappingPath, "r") as f:
			for fileLine in f:
				if not fileLine.strip():
					continue
				lineSplit = fileLine.split(",")
				if len(lineSplit) < 2:
					continue
				key_str = lineSplit[0].strip()
				val_str = lineSplit[1].strip()
				try:
					key = int(key_str)
				except ValueError:
					continue
				if isNumericString(val_str):
					mapping[key] = int(val_str)
				else:
					mapping[key] = val_str
	except Exception:
		print("File cannot be opened")
		exit(1)
	return mapping

# Author: Nikolaos Vouronikos
def getInnerSiP(code):
	innerSiP = encodeInteger(code)
	return innerSiP

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
def getGridPositions(imagePath, division):
	positionsPath = os.path.join(os.path.dirname(imagePath), "GridPositions.txt")
	allPositions = []

	try:
		f = open(positionsPath, "r")
	except :
		print("File cannot be opened")
		exit(1)

	for i in range(division):
		fileLine = f.readline()
		lineSplit = fileLine.split(",")
		position = [int(lineSplit[0]), int(lineSplit[1])]
		allPositions.append(position)
	return allPositions

# Author: Nikolaos Vouronikos
def extract(watermarkedBlocks, innerSip, gridSize, RBWidth, Rxy, Bxy, allGridPositions):
	extractionSuccess,error = 0,0
	codeTaken = None
	for i in range(len(watermarkedBlocks)):
		watermarkedBlock,sip,optimalGridPosition = watermarkedBlocks[i],int(code),allGridPositions[i]
		isExtracted,key1,key2,key3,min_error = extractSiP(watermarkedBlock, sip, innerSip, gridSize, RBWidth, Rxy, Bxy, optimalGridPosition)
		decodedKey = decodeKey(key1, key2, key3, sip)
		if(decodedKey == "X"):
			print("Extraction on attempt " + str(i + 1) + " failed. No key was extracted.")
		else:
			extractionSuccess = 1
			codeTaken = decodedKey
			print("Extraction on attempt " + str(i + 1) + " was successful. Extracted key is : " + str(decodedKey))
		error = error + min_error
	
	ber = (error / (1 * len(innerSip) * 12))
	extractionRate = extractionSuccess*100 if len(watermarkedBlocks) > 0 else 0
	extractionResult = ExtractionResult(codeTaken, extractionRate, ber)
	return extractionResult

# Author: Nikolaos Vouronikos
def extractSiP(watermarkedBlock, originalKey, innerSip, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock,recoEnabled = 1):
	em,ex = init()
	sip1,sip2,sip3 = ex.getSip(watermarkedBlock, len(innerSip), gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
 
	if(recoEnabled):
		r1 = recsip(sip1, innerSip, originalKey)
		r2 = recsip(sip2, innerSip, originalKey)
		r3 = recsip(sip3, innerSip, originalKey)

		error_bits1 = min(SIP_to_BER(sip1, innerSip), SIP_to_BER(r1, innerSip))
		error_bits2 = min(SIP_to_BER(sip2, innerSip), SIP_to_BER(r2, innerSip))
		error_bits3 = min(SIP_to_BER(sip3, innerSip), SIP_to_BER(r3, innerSip))
	else:
		error_bits1 = SIP_to_BER(sip1, innerSip)
		error_bits2 = SIP_to_BER(sip2, innerSip)
		error_bits3 = SIP_to_BER(sip3, innerSip)
	
	min_error = min(error_bits1, error_bits2, error_bits3)
 
	if(recoEnabled):
		if(sip1 == innerSip or r1 == innerSip):
			return 1,decodeSip(innerSip),decodeSip(innerSip),decodeSip(innerSip),min_error
		elif(sip2 == innerSip or r2 == innerSip):
			return 1,decodeSip(innerSip),decodeSip(innerSip),decodeSip(innerSip),min_error
		elif(sip3 == innerSip or r3 == innerSip):
			return 1,decodeSip(innerSip),decodeSip(innerSip),decodeSip(innerSip),min_error
		else:
			return 0,decodeSip(sip1),decodeSip(sip2),decodeSip(sip3),min_error
	else:
		if(sip1 == innerSip):
			return 1,decodeSip(sip1),decodeSip(sip1),decodeSip(sip1),min_error
		elif(sip2 == innerSip):
			return 1,decodeSip(sip2),decodeSip(sip2),decodeSip(sip2),min_error
		elif(sip3 == innerSip):		
			return 1,decodeSip(sip3),decodeSip(sip3),decodeSip(sip3),min_error
		else:
			return 0,decodeSip(sip1),decodeSip(sip2),decodeSip(sip3),min_error

def runValidation(imagePath, code):
    startingPoint = time.time()
    watermarkedBlocks = getWatermarkedBlocksInList(code, imagePath)
    innerSiP = getInnerSiP(int(code))
    gridSize,RBWidth,Rxy,Bxy = getBasicValues(imagePath)
    allGridPositions = getGridPositions(imagePath, 6)
    extractionResult = extract(watermarkedBlocks, innerSiP, gridSize, RBWidth, Rxy, Bxy, allGridPositions)
    end = time.time()
    secSTR,minSTR = calculateElapseTime(startingPoint, end)
    print("Elapsed time = " + str(minSTR) + " mins")
    print("Elapsed time = " + str(secSTR) + " seconds")
    print(str(extractionResult.codeTaken) + "\n")
    print(str(extractionResult.extractionRate) + "%")
    print("BER =", str(extractionResult.ber))
    
if __name__ == '__main__':
	# Initialization from command line
	# Example: py validator.py watermarked/watermarked_people/watermarked_people.jpg 112230765
	imagePath, code = sys.argv[1:3]
	runValidation(imagePath, code)