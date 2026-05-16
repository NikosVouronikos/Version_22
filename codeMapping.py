import random,math

# Author: Nikolaos Vouronikos
# Description: Returns block's dimensions according to Image's dimensions
def getBlockDimensions(M, N, size) :
	blockWidth = math.floor((M / size))
	blockHeight = math.floor((N / size))
	return blockWidth,blockHeight

# Author: Nikolaos Vouronikos
# Description: User's code from String to List
def getListFromCode(code):
	listCode = []
	for i in range(len(code)):
		if(isNumericString(code[i])):
			listCode.append(int(code[i]))
		else:
			listCode.append(code[i])
	return listCode

# Author: Nikolaos Vouronikos
def isNumericString(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Author: Nikolaos Vouronikos
# Description: User's code from List to String
def getCodeFromList(codeList):
	str_list = [str(item) for item in codeList]
	delimeter = ""
	code = delimeter.join(str_list)
	return code

# Author: Nikolaos Vouronikos
# Description: Generates a fixed or random mapping between digits [0,f] and watermarks of class 9 [8,15]
def getCodeMapping(mode):
	mapping = {}
	listOfSips = list(range(8,16))						# Class 9 watermarks
	listOfNums = list(range(8))
	if(mode == 'FIXED'):
		for i in range(len(listOfSips)):
			mapping[listOfSips[i]] = listOfNums[i]
	else:
		upperLimit = len(listOfNums) - 1
		for i in range(len(listOfSips)):
			index = random.randint(0, upperLimit)
			mapping[listOfSips[i]] = listOfNums[index]
			listOfNums.remove(listOfNums[index])
			upperLimit = upperLimit - 1
	return mapping

# Author: Nikolaos Vouronikos
# Description: Returns a list of integers w (watermarks) using the generated mapping
def getSipsFromCode(mapping, code):
	codeSips = []
	keyList = list(mapping.keys())
	valuesList = list(mapping.values())
	for i in range(len(code)):
		position = valuesList.index(code[i])
		sip = keyList[position]
		codeSips.append(sip)
	return codeSips

# Author: Nikolaos Vouronikos
# Description: Checks if the code's size is equal to length
# If not then for < length add random digits and for > length keep only the first length digits
def checkForFullCode(code, originalLength):
	diff = originalLength - len(code)
	if(len(code) < originalLength):
		for i in range(diff):
			code.append(random.randint(0,9))
	else:
		code = code[0:originalLength]
	return code
