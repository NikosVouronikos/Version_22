import cv2,sys,re,time,random,numpy as np,matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from decodesip import decodeSip
from encodeinteger import encodeInteger
from initializer import *
from codeMapping import *
from utilities import *
from optimizer import *
from metrics import *

# Author: Nikolaos Vouronikos
# Description: This script embeds a watermark into an image block and extracts it back.

def extract(watermarkedBlock, innerKey, mapping, innerSip, gridSize, RBWidth, Rxy, Bxy, blockIndex, imagePath, optimalGridPosition):
	isExtracted,key1,key2,key3,ber = extractSiP(watermarkedBlock, innerKey, innerSip, gridSize, RBWidth, Rxy, Bxy, optimalGridPosition)
	decodedKey = decodeKey(key1, key2, key3, innerKey)
	if(decodedKey == "X"):
		return False
	else:
		return True

#watermark -> integer to embed
#imagePath -> path of the input image
#imageName -> if image1.jpg keeps only image1
#blockIndex -> 1,2,...,16
def embed(imagePath, imageName, blockIndex, mode, extension, code):
	try:
		innerSips, index, extractionIsPrioritized = [], 0, 1
		gridSize,RBWidth,Rxy,Bxy = [],[],[],[]
		em,ex = init()
		code, size, mapping, codeSips, blockWidth, blockHeight, imageArray, M, N = prepareEmbedding(imagePath, code, 'FIXED')				
		innerKey = codeSips[blockIndex-1]
		print("Embed key : " + str(innerKey) + " in Block " + str(index + 1))
		innerSip = encodeInteger(innerKey)																	
		blockProperties = BlockProperties(innerSip, len(innerSip), innerKey, blockWidth, blockHeight)									
		offsetX = ((blockIndex - 1) % int(size)) * blockWidth
		offsetY = ((blockIndex - 1) // int(size)) * blockHeight
		blockArray = imageArray[offsetY:(offsetY + blockHeight), offsetX:(offsetX + blockWidth)]	
		blockImage = Image.fromarray(blockArray)
		gridSize, RBWidth, Rxy, Bxy = calculateBasicValues(blockProperties, 2, 2)	
		optimalCValue, watermarkedBlock, optimalGridPosition = findOptimalCValueForBlock(blockProperties, em, code, blockImage, mode, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath, imageName)
		watermarkedBlockImage = Image.fromarray(np.array(watermarkedBlock))
		watermarkedBlockImage.save('watermarked_block_' + imageName + '_' + str(blockIndex) + extension)
		getPSNRAndSSIM(np.array(watermarkedBlockImage), blockArray)
		return watermarkedBlock,optimalCValue,optimalGridPosition,innerKey,innerSip,mapping,gridSize,RBWidth,Rxy,Bxy
	except Exception as e:
		print(f"An error occurred: {e}")
		exit(1)

def findOptimalCValueForBlock(blockParams, embedObject, code, g_cell, 
								mode, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imagePath, imageName):
	if(mode == 'FAST'):
		isExtracted,badPositions,counter = False,[],0
		while(isNotExtracted(isExtracted, counter)): # Find random position but if no extraction there try again with different position (max 5 times)
			counter = counter + 1
			randomGridPosition = calculateRandomGridPosition(blockParams, gridSize)
			if(randomGridPosition in badPositions): # Caution not checking the same position (rare but better be sure)
				continue

			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imageName, imagePath, randomGridPosition)
			if(isExtracted == False):
				badPositions.append(randomGridPosition)
		return optimalCValue,watermarkedBlock,randomGridPosition
	else:
		optimalCValue,watermarkedBlock,optimalGridPosition = optimizeCValueFull(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imageName, imagePath, step)
		return optimalCValue,watermarkedBlock,optimalGridPosition,counter

##############################TEST RUN##############################
if __name__ == '__main__':
	hex_values = [int(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('f') + 1)]
	repeated_lists = [[val] * 16 for val in hex_values]
	mode = 'FULL'
	format = '.png'
	for i in range(1, 21):
		imagePath = 'dataset/image_' + str(i) + format #run 20 images png or jpg
		extension = os.path.splitext(imagePath)[1]
		imageName = (((imagePath.split("/"))[-1]).split(extension))[0]
		#totalWatermarksExtracted = 16
		for j in range(1, 17):
			code = repeated_lists[j-1]
			for i in range(1, 17):
				blockIndex = i # can be 1,2,3,...,16
				watermarkedBlock, optimalCValue, optimalGridPosition, innerKey, innerSip, codeMapping, gridSize, RBWidth, Rxy, Bxy = embed(imagePath, imageName, blockIndex, mode, extension, code) #Embed
				print(optimalCValue, optimalGridPosition)
				result = extract(watermarkedBlock, innerKey, codeMapping, innerSip, gridSize, RBWidth, Rxy, Bxy, blockIndex, imagePath, optimalGridPosition) #Extract
				#if(result == False):
					#totalWatermarksExtracted = totalWatermarksExtracted - 1

	#extractionRate = (totalWatermarksExtracted/16)*100
	#print(str(extractionRate) + '%')