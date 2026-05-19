import random,numpy as np
from metrics import *
from initializer import *
from codeMapping import *
from rsipw import extractSiP

# Author: Nikolaos Vouronikos
# Description: C Optimization Algorithm
def optimizeCValueFast(blockParams, embedObject, code, originalBlock, extractionIsPrioritized, gridSize, 
						RBWidth, Rxy, Bxy, imagePath, gridPositionForEachBlock):

	# enforce minimum C value
	low,high = 40,120
	originalHigh,optimalCValue,isExtractedPrevious,stop,psnr,ssim,isExtracted,completeExtractions = high,high,0,0,0,0,0,[]

	print("Running with c = " + str(low))
	watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, low, 2, 2, 
														gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	psnr,ssim = getPSNRAndSSIM(originalBlock, watermarkedBlock)
	isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams.key, blockParams.sip, code, 
										gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	if(isExtracted == 1):
		return low,watermarkedBlock,isExtracted,psnr,ssim

	while(low <= high):
		print("Running with c = " + str(optimalCValue))
		watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalCValue, 2, 2, 
															gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
		psnr,ssim = getPSNRAndSSIM(originalBlock, watermarkedBlock)
		isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams.key, blockParams.sip, code, 
											gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
		if(stop == 1):
			break
		if(isExtractedPrevious == 1 and isExtracted == 0 and optimalCValue < 7):	# Avoiding dead-end
			optimalCValue = optimalCValuePrevious
			stop = 1
			continue

		isExtractedPrevious = isExtracted 
		optimalCValuePrevious = optimalCValue

		if(hasCValueLowestLimit(optimalCValue)) :	# Avoiding negative values
			break

		if(isSuccessfullExtraction(isExtracted)):
			step = (high - low) // 2
			optimalCValue = high - step
			high = step
			completeExtractions.append(optimalCValuePrevious)
		elif((not isSuccessfullExtraction(isExtracted)) and (optimalCValue >= 25)):
			step = (high - low) // 2
			optimalCValue = high - step
			high = step
		elif((not isSuccessfullExtraction(isExtracted)) and (optimalCValue < 25)) :
			step = (high - low) // 2
			optimalCValue = high + step
			low = high
			high = optimalCValue

		if(optimalCValuePrevious == optimalCValue) :	# Avoiding dead-end
			break

	# Post processing C factor results		
	if((not isSuccessfullExtraction(isExtracted)) and (extractionIsPrioritized == 1)):
		if(completeExtractions != []):
			optimalCValue = optimalCValue + 1
			globalCValue = min(completeExtractions)
			low = optimalCValue
			high = globalCValue
			while(low <= high):
				print("Running with c = " + str(optimalCValue))
				if(optimalCValue > globalCValue):
					optimalCValue = globalCValue
					watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalCValue, 2, 2, 
																		gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
					break

				watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalCValue, 2, 2, 
																	gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams.key, blockParams.sip, code, 
													gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				optimalCValuePrevious = optimalCValue

				if(isSuccessfullExtraction(isExtracted)):
					break
				elif(not isSuccessfullExtraction(isExtracted)):
					low = optimalCValue
					optimalCValue = optimalCValue + 3
				if(optimalCValuePrevious == optimalCValue) :
					break
		else:
			low = optimalCValue + 1
			startCValue = low
			high = 100
			optimalCValue = startCValue
			while(low <= high):
				print("Running with c = " + str(optimalCValue))
				watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalCValue, 2, 2, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams.key, blockParams.sip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				optimalCValuePrevious = optimalCValue
				if(isSuccessfullExtraction(isExtracted)):
					break
				elif(not isSuccessfullExtraction(isExtracted)):
					low = optimalCValue
					optimalCValue = optimalCValue + 3
				if(optimalCValuePrevious == optimalCValue) :
					break
			if(not isSuccessfullExtraction(isExtracted)):
				optimalCValue = 40
				print("Running with c = " + str(optimalCValue))
				watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalCValue, 2, 2, 
																	gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	printResults(2, optimalCValue, 0, 0, [])
	return optimalCValue,watermarkedBlock,isExtracted,psnr,ssim

# Author: Nikolaos Vouronikos
# Description: Running C Optimization Algorithm for each possible grid displacement
def optimizeCValueFull(blockParams, embedObject, code, originalBlock, extractionIsPrioritized, gridSize, 
						RBWidth, Rxy, Bxy, imagePath, step):

    optimalCValuesForGrid, optimalGridPositions, optimalPSNR, optimalSSIM, extractionResults = [], [], [], [], []
    candidatePositions = enumerate_grid_positions(blockParams, gridSize, step)
    countPositions = 0
    for gridPositionForEachBlock in candidatePositions:
        optimalCValue, watermarkedBlock, isExtracted, psnr, ssim = optimizeCValueFast(blockParams, embedObject, code, originalBlock, extractionIsPrioritized,
            																			gridSize, RBWidth, Rxy, Bxy, imagePath, gridPositionForEachBlock)
        optimalCValuesForGrid.append(optimalCValue)
        extractionResults.append(isExtracted)
        optimalPSNR.append(psnr)
        optimalSSIM.append(ssim)
        optimalGridPositions.append(gridPositionForEachBlock)
        printResults(2, optimalCValue, 0, 0, [])
        countPositions = countPositions + 1

    optimalC, maxPSNR, maxSSIM, optimalGridPosition = findOptimalPropertiesForBlock(optimalPSNR, optimalSSIM, extractionResults, optimalGridPositions, optimalCValuesForGrid)
    printResults(1, optimalC, maxPSNR, maxSSIM, optimalGridPosition)
    watermarkedBlock = embedObject.getWatermarkedImage(originalBlock, blockParams.sip, blockParams.sipSize, optimalC, 2, 2,
        												gridSize, RBWidth, Rxy, Bxy, optimalGridPosition)
    return optimalC, watermarkedBlock, optimalGridPosition, countPositions

# Author: Nikolaos Vouronikos
def enumerate_grid_positions(blockParams, gridSize, step):
    positions = []

    maxY = blockParams.blockHeight - (gridSize[0] * blockParams.sipSize)
    maxX = blockParams.blockWidth  - (gridSize[1] * blockParams.sipSize)
    jump = step + 1

    for offsetY in range(0, maxY + 1, jump):
        for offsetX in range(0, maxX + 1, jump):
            positions.append([offsetY, offsetX])

    return positions

# Author: Nikolaos Vouronikos
# Description: Calculate grid position randomly (used for fast version of C Optimization)
def calculateRandomGridPosition(blockParams, gridSize):
	rowSum,columnSum,moveRow,moveColumn = 0,0,[0],[0]
	rowLimit = math.floor(((blockParams.blockHeight) - (gridSize[0] * blockParams.sipSize)) / (gridSize[0]))
	columnLimit = math.floor(((blockParams.blockWidth) - (gridSize[1] * blockParams.sipSize)) / (gridSize[1]))

	for i in range(rowLimit):
		rowSum = rowSum + gridSize[0]
		moveRow.append(rowSum)
	for j in range(columnLimit):
		columnSum = columnSum + gridSize[1]
		moveColumn.append(columnSum)

	rowIndex = random.randint(0, (len(moveRow)) - 1)
	columnIndex = random.randint(0, (len(moveColumn)) - 1)
	offsetY = moveRow[rowIndex]
	offsetX = moveColumn[columnIndex]
	gridPositionForEachBlock = [offsetY,offsetX]
	return gridPositionForEachBlock

# Author: Nikolaos Vouronikos
def getPSNRAndSSIM(originalBlock, watermarkedBlock):
	psnr = PSNR(np.array(originalBlock), np.array(watermarkedBlock))
	ssim = SSIM(np.array(originalBlock), np.array(watermarkedBlock))
	print("PSNR taken " + "{:.5f}".format(psnr))
	print("SSIM taken " + "{:.5f}".format(ssim))
	print("")
	return psnr,ssim

# Author: Nikolaos Vouronikos
def getExtractionResult(watermarkedBlock, compressedImageName, innerKey, sip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock):
	isExtracted,key1,key2,key3 = extractSiP(watermarkedBlock, innerKey, sip, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	print("Result of extraction =",isExtracted)
	print("")
	return isExtracted

# Author: Nikolaos Vouronikos
def findOptimalPropertiesForBlock(optimalPSNR, optimalSSIM, extractionResults, optimalGridPositions, optimalCValuesForGrid):
	maxPSNR,maxSSIM,optimalC,optimalGridPosition,bestIndex = 0,0,30,[0,0],0
	while(True):
		maxPSNRIndex = optimalPSNR.index(max(optimalPSNR))	# Index of maximum psnr found
		maxSSIMIndex = optimalSSIM.index(max(optimalSSIM))	# Index of maximum ssim found
		if(maxPSNRIndex == maxSSIMIndex):
			bestIndex = maxPSNRIndex
			if(isSuccessfullExtraction(extractionResults[bestIndex])):	# Check if with max psnr and ssim we have also successful extraction
				optimalC = optimalCValuesForGrid[bestIndex]
				maxPSNR = optimalPSNR[bestIndex]
				maxSSIM = optimalSSIM[bestIndex]
				optimalGridPosition = optimalGridPositions[bestIndex]
				break
			else:
				# If no extraction remove the associated values from the appropriate lists
				optimalPSNR.pop(bestIndex)
				optimalSSIM.pop(bestIndex)
				optimalCValuesForGrid.pop(bestIndex)
				extractionResults.pop(bestIndex)
				optimalGridPositions.pop(bestIndex)
				continue
		else:
			if(isSuccessfullExtraction(extractionResults[maxPSNRIndex])):	# prioritize psnr over ssim max value
				bestIndex = maxPSNRIndex
			elif(isSuccessfullExtraction(extractionResults[maxSSIMIndex])):
				bestIndex = maxSSIMIndex
			else:
				optimalPSNR.pop(bestIndex)
				optimalSSIM.pop(bestIndex)
				optimalCValuesForGrid.pop(bestIndex)
				extractionResults.pop(bestIndex)
				optimalGridPositions.pop(bestIndex)
				continue
			optimalC = optimalCValuesForGrid[bestIndex]
			maxPSNR = optimalPSNR[bestIndex]
			maxSSIM = optimalSSIM[bestIndex]
			optimalGridPosition = optimalGridPositions[bestIndex]
			break
	return optimalC,maxPSNR,maxSSIM,optimalGridPosition

# Author: Nikolaos Vouronikos
def printResults(mode, c, psnr, ssim, bm):
	if(mode == 1):
		print("Running with c = " + str(c))
		print("PSNR = " + str(psnr))
		print("SSIM = " + str(ssim))
		print(bm)
		print("Finished grid movement")
		print("")
	elif(mode == 2):
		print("Optimal c value for grid = " + str(c))
		print("-----------------------------------")
		print("")
	elif(mode == 3):
		if(psnr != 0):
			print("Complete Extraction in block " + str(c + 1) + "\n")
		else:
			print("Could not extract in block " + str(c + 1) + "\n")
	else:
		print("Extraction percentage = " + str(c) + "%\n")

# Helper functions		
# Author: Nikolaos Vouronikos
def hasCValueLowestLimit(optimalCValue):
	# consider 40 as the minimum allowed C value
	if(optimalCValue <= 40):
		return True
	return False

def isSuccessfullExtraction(isExtracted):
	if(isExtracted == 1):
		return True
	return False

def isCBetterThanCurrent(optimalCValue, currentC):
	if(optimalCValue <= currentC):
		return True
	return False

def isPSNRAndSSIMBetterThanCurrent(optimalPSNR, optimalSSIM, currentPSNR, currentSSIM):
	if(optimalPSNR >= currentPSNR and optimalSSIM >= currentSSIM):
		return True
	return False