from embed_key import EmbedPermutation
from extract_sip import ExtractPermutation
from PIL import Image

# Author: Nikolaos Vouronikos
# Class: BlockProperties
# sip: The Self-Inverting Permutation for the key (watermark w)
# sipSize: In our scenario this is always equal to 11
# key: Integer w from which the SiP is derived
class BlockProperties:
	def __init__(self, sip, sipSize, key, blockWidth, blockHeight):
		self.sip = sip
		self.sipSize = sipSize
		self.key = key
		self.blockWidth = blockWidth
		self.blockHeight = blockHeight

class EmbedResult:
	def __init__(self, watermarkedImage, watermarkedBlocks, codeSips, mapping, innerSips, subpath, 
					optimalCValues, gridSize, RBWidth, Rxy, Bxy, optimalGridPositionForEachBlock):
		self.watermarkedImage = watermarkedImage
		self.watermarkedBlocks = watermarkedBlocks
		self.codeSips = codeSips
		self.mapping = mapping
		self.innerSips = innerSips
		self.subpath = subpath
		self.optimalCValues = optimalCValues
		self.gridSize = gridSize
		self.RBWidth = RBWidth
		self.Rxy = Rxy
		self.Bxy = Bxy
		self.optimalGridPositionForEachBlock = optimalGridPositionForEachBlock

class ExtractionResult:
	def __init__(self, codeTaken, extractionRate, ber):
		self.codeTaken = codeTaken
		self.extractionRate = extractionRate
		self.ber = ber

# Author: Vasileios Vouronikos
def init():
	w = EmbedPermutation()
	ex = ExtractPermutation()
	return w,ex

# Author: Vasileios Vouronikos
def openImage(path):
	img = Image.open(path)
	if(img.mode == 'P' or img.mode == 'L'):
		img = img.convert('RGB')
	return img