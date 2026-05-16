import cv2,sys,time,numpy as np
from skimage.metrics import structural_similarity as ssim
from decodesip import decodeSip
from encodeinteger import encodeInteger
from PIL import Image
from initializer import *
from codeMapping import *
from utilities import *
from optimizer import *
from attacker import *
from metrics import *

# Author: Nikolaos Vouronikos
# Description: Extract user's code-sequence from watermarked image
# Output: Extraction Object
def extract(embedResult):
	codeTaken,totalWatermarksExtracted = [],16
	for i in range(len(embedResult.watermarkedBlocks)):
		watermarkedBlock,sip,optimalGridPosition,innerSip,enable = embedResult.watermarkedBlocks[i],embedResult.codeSips[i],embedResult.optimalGridPositionForEachBlock[i],embedResult.innerSips[i],1
		isExtracted,key1,key2,key3 = extractSiP(watermarkedBlock, sip, innerSip, embedResult.gridSize, embedResult.RBWidth, embedResult.Rxy, embedResult.Bxy, optimalGridPosition)
		decodedKey = decodeKey(key1, key2, key3, sip)
		if(decodedKey == "X"):
			enable = 0
			codeTaken.append(decodedKey)
			totalWatermarksExtracted = totalWatermarksExtracted - 1
		else:
			codeTaken.append(embedResult.mapping[decodedKey])
		printResults(3, i, enable, 0, [])
	extractionRate = (totalWatermarksExtracted / 16)*100
	extractionResult = ExtractionResult(codeTaken, extractionRate)
	return extractionResult

# Author: Nikolaos Vouronikos
# Description: Embed user's code-sequence and produce watermarked image
# Output: Embed Object
def embed(code, mode, imagePath, imageName, extension):
	try:
		# Initialize variables
		watermarkedBlocks, innerSips, index, extractionIsPrioritized = [], [], 0, 1
		optimalCValues, gridSize, RBWidth, Rxy, Bxy, optimalGridPositionForEachBlock = [], [], [], [], [], []
		em,ex = init()
		code, size, mapping, codeSips, blockWidth, blockHeight, imageArray, M, N = prepareEmbedding(imagePath, code, 'FIXED')

		for offsetY in range(0, (N - blockHeight + 1), blockHeight):
			for offsetX in range(0, (M - blockWidth + 1), blockWidth):
				innerKey = codeSips[index]																		# innerKey is the integer w (the Watermark)
				print("Embed key : " + str(innerKey) + " in Block " + str(index + 1))
				innerSip = encodeInteger(innerKey)																# innerSip is the 1D permutation of innerKey
				innerSips.append(innerSip)																		# innerSips contains all the SiPs we embedded
				blockProperties = BlockProperties(innerSip, len(innerSip), innerKey, blockWidth, blockHeight)	# Initialize each time BlockProperties object
				blockArray = imageArray[offsetY:(offsetY + blockHeight), offsetX:(offsetX + blockWidth)]		# Initialize and take the block
				blockImage = Image.fromarray(blockArray)														# Construct the block's image
				if(index == 0):
					gridSize, RBWidth, Rxy, Bxy = calculateBasicValues(blockProperties, 2, 2, blockImage, em)

				optimalCValue, watermarkedBlock, optimalGridPosition = findOptimalCValueForBlock(blockProperties, em, code, blockImage, mode, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath, imageName)	# Begin C Optimization for Block
				optimalGridPositionForEachBlock.append(optimalGridPosition), optimalCValues.append(optimalCValue), watermarkedBlocks.append(watermarkedBlock)
				index = index + 1

		watermarkedImageName = "watermarked_" + imageName
		watermarkedImage = mergeCellsToImage(watermarkedBlocks, blockWidth, blockHeight, int(size))								# Merge all watermarked blocks and reconstruct image
		watermarkedImage = Image.fromarray((cv2.resize(np.array(watermarkedImage), (M,N), interpolation = cv2.INTER_LANCZOS4)))	# From the Array go to the Image
		subpath = saveWatermarkedImage(watermarkedImageName, watermarkedImage, mapping, code, codeSips, extension)
		writeBasicValuesInFile(gridSize, RBWidth, Rxy, Bxy, subpath)
		writeGridPositionsInFile(optimalGridPositionForEachBlock, subpath)
		getPSNRAndSSIM(np.array(watermarkedImage), imageArray)
		embedResult = EmbedResult(watermarkedImage,watermarkedBlocks,codeSips,mapping,innerSips,subpath,optimalCValues,gridSize,RBWidth,Rxy,Bxy,optimalGridPositionForEachBlock)
		return embedResult
	except Exception as e:
		print(f"An error occurred: {e}")
		exit(1)

# Author: Vasileios Vouronikos
# Output: Extraction result (0 or 1) and decoded Keys
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

# Author: Nikolaos Vouronikos
# Description: Run C Optimization Algorithm for block (Fast or Full)
# Output: Optimal C Value (Integer), Watermarked Block (PIL.Image.Image), Grid Position ([x,y])
def findOptimalCValueForBlock(blockParams, embedObject, code, g_cell, 
								mode, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath, imageName):
	if(mode == 'FAST'):
		isExtracted,badPositions,counter = False,[],0
		while(isNotExtracted(isExtracted, counter)): # Find random position but if no extraction there try again with different position (max 5 times)
			counter = counter + 1
			randomGridPosition = calculateRandomGridPosition(blockParams, gridSize)
			if(randomGridPosition in badPositions): # Caution not checking the same position (rare but better be sure)
				continue

			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath, randomGridPosition)
			if(isExtracted == False):
				badPositions.append(randomGridPosition)
		return optimalCValue,watermarkedBlock,randomGridPosition
	else:
		optimalCValue,watermarkedBlock,optimalGridPosition = optimizeCValueFull(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath)
		return optimalCValue,watermarkedBlock,optimalGridPosition

# Author: Nikolaos Vouronikos
# Description: Run function starts embed procedure
# Output: None
def run(imagePath, code, mode):
	extension = os.path.splitext(imagePath)[1]
	imageName = (((imagePath.split("/"))[-1]).split(extension))[0]
	startingPoint = time.time()
	code = getListFromCode(code)

	# Run Main Algorithm
	embedResult = embed(code, mode, imagePath, imageName, extension) 	# Embed
	extractionResult = extract(embedResult) 							# Extract (optional)
	codeExtracted = getCodeFromList(extractionResult.codeTaken)
	writeBestCValuesInFile(embedResult.optimalCValues, codeExtracted, extractionResult.extractionRate, embedResult.subpath)
	calculateElapseTimeAndPrintResults(startingPoint, extractionResult.extractionRate)

if __name__ == '__main__':
	# Initialization from command line
	# Example: py rsipw.py testImages/image1.jpg 56728192afd67fca FAST through cmd 
	imagePath, code, mode = sys.argv[1:4]
	run(imagePath, code, mode)