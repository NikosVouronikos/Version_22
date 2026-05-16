import cv2, numpy as np
from ellipticdisk import createEllipticDisk
from PIL import Image

# Author: Vasileios Vouronikos
class ExtractPermutation:
	def __init__(self):
		return

	def openImage(self, path):
		img = Image.open(path)
		return img

	def getFFTTransform(self, image):
		dft = np.fft.fft2(image, norm = 'ortho')
		fftShift = np.fft.fftshift(dft)
		mag = np.abs(fftShift)
		phase = np.angle(fftShift)
		return mag,phase

	def getIFFTTransform(self, mag, phase):
		real = mag * np.cos(phase)
		imag = mag * np.sin(phase)
		complex_output = np.zeros(mag.shape, complex)

		complex_output.real = real
		complex_output.imag = imag
		back_ishift = np.fft.ifftshift(complex_output)
		img_back = np.fft.ifft2(back_ishift, norm = 'ortho')
		img_back = abs(img_back)
		return img_back 

	def showImage(self, *image_args):
		for image in image_args:
			cv2.imshow('image',image)
			cv2.waitKey(0)
		cv2.waitKey(0)
		return

	def getSip(self, im, SIZE, gridSize, RBWidth, Rxy, Bxy, moves):
		#img = Image.open(img)
		r,g,b = im.split()
		sip1 = self.extractPermutationFromChannel(r, SIZE, gridSize, RBWidth, Rxy, Bxy, moves)
		sip2 = self.extractPermutationFromChannel(g, SIZE, gridSize, RBWidth, Rxy, Bxy, moves)
		sip3 = self.extractPermutationFromChannel(b, SIZE, gridSize, RBWidth, Rxy, Bxy, moves)
		return sip1,sip2,sip3

	def extractPermutationFromChannel(self, channel, SIZE, gridSize, RBWidth, Rxy, Bxy, moves):
		grid_cell_num = 0
		sip = []
		avg = []
		
		# STEP 2: COMPUTE IMAGE SIZE
		M,N = channel.size
		channel_array = np.array(channel)

		# GET THE SIZE OF EACH GRID SHELL
		minAvg = []
		minAll = []

		x,y,i,c = 0,0,0,0

		if(N < M):
			for r in range(moves[0], ((gridSize[1] * SIZE) + moves[0]), (gridSize[1])):
				for c in range(moves[1], ((gridSize[1] * SIZE) + moves[1]), (gridSize[1])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE = 0,0
					
					grid_cell = channel_array[r:r + gridSize[0],c:c + gridSize[1]]
					mag,phase = self.getFFTTransform(grid_cell)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,Rxy[0],Rxy[1],RBWidth[0],cx,cy,gridSize[0],gridSize[1])
					blue,coord_blue = createEllipticDisk(mag,Bxy[0],Bxy[1],RBWidth[1],cx,cy,gridSize[0],gridSize[1])

					if len(red) == 0 or len(blue) == 0:
						raise ValueError(f"Invalid grid cell size {gridSize}: empty annulus at cell ({x},{y}) with moves={moves}, "f"Rxy={Rxy}, Bxy={Bxy}, RBWidth={RBWidth}")

					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)
					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					y += 1
				minAll.append(min(minAvg))
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		
		elif(N > M):
			for r in range(moves[0], ((gridSize[0] * SIZE) + moves[0]), (gridSize[0])):
				for c in range(moves[1], ((gridSize[0] * SIZE) + moves[1]), (gridSize[0])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE = 0,0
					
					grid_cell = channel_array[r:r + gridSize[0],c:c + gridSize[1]]
					mag,phase = self.getFFTTransform(grid_cell)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,Rxy[0],Rxy[1],RBWidth[0],cx,cy,gridSize[0],gridSize[1])
					blue,coord_blue = createEllipticDisk(mag,Bxy[0],Bxy[1],RBWidth[1],cx,cy,gridSize[0],gridSize[1])

					if len(red) == 0 or len(blue) == 0:
						raise ValueError(f"Invalid grid cell size {gridSize}: empty annulus at cell ({x},{y}) with moves={moves}, "f"Rxy={Rxy}, Bxy={Bxy}, RBWidth={RBWidth}")
					
					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)
					c += 1
					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					y += 1
				minAll.append(min(minAvg))
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		else:
			for r in range(moves[0], ((gridSize[0] * SIZE) + moves[0]), (gridSize[0])):
				for c in range(moves[1], ((gridSize[1] * SIZE) + moves[1]), (gridSize[1])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE = 0,0
					
					grid_cell = channel_array[r:r + gridSize[0],c:c + gridSize[1]]
					mag,phase = self.getFFTTransform(grid_cell)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,Rxy[0],Rxy[1],RBWidth[0],cx,cy,gridSize[0],gridSize[1])
					blue,coord_blue = createEllipticDisk(mag,Bxy[0],Bxy[1],RBWidth[1],cx,cy,gridSize[0],gridSize[1])

					if len(red) == 0 or len(blue) == 0:
						raise ValueError(f"Invalid grid cell size {gridSize}: empty annulus at cell ({x},{y}) with moves={moves}, "f"Rxy={Rxy}, Bxy={Bxy}, RBWidth={RBWidth}")
				
					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)
					c += 1
					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					y += 1

				minAll.append(min(minAvg))
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip