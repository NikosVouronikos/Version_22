from ellipticdisk import createEllipticDisk
from PIL import Image
import cv2, numpy as np, itertools as it, math, hashlib

# Author: Vasileios Vouronikos
class EmbedPermutation:
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

	def hash_block(self, b):
		h = hashlib.new('sha256')
		h.update(b.tobytes())
		hashb = h.hexdigest()
		return hashb

	def showImage(self, *image_args):
		for image in image_args:
			cv2.imshow('image', image)
			cv2.waitKey(0)
		cv2.waitKey(0)
		return

	def sortCells(self, mc, nonm, sip_cells):
		cell_list = []
		k,f = 0,0
		for i in range(len(sip_cells)):
			for j in range(len(sip_cells)):
				if(k < len(sip_cells) and sip_cells[k][0] == i and sip_cells[k][1] == j):
					cell_list.append(mc[k])
					k += 1
				else:
					cell_list.append(nonm[f])
					f += 1
		return cell_list

	def getWatermarkedImage(self, im, sip, SIZE, copt, PR, PB, gridSize, RBWidth, Rxy, Bxy, moves):
		r,g,b = im.split()
		im1 = np.array(im)
		M,N = im.size
		n1 = gridSize[0] * SIZE
		k1 = moves[0]
		k2 = gridSize[1] * SIZE + k1
		m1 = moves[1]
		m2 = gridSize[0] * SIZE + m1
		print("Rows offset = (" + str(k1) + "," + str(k2) + ")")
		print("Columns offset = (" + str(m1) + "," + str(m2) + ")")

		r_img = self.embedPermutationToChannel(r, sip, SIZE, copt, PR, PB, gridSize, RBWidth, Rxy, Bxy, moves)
		g_img = self.embedPermutationToChannel(g, sip, SIZE, copt, PR, PB, gridSize, RBWidth, Rxy, Bxy, moves)
		b_img = self.embedPermutationToChannel(b, sip, SIZE, copt, PR, PB, gridSize, RBWidth, Rxy, Bxy, moves)
		
		if(r_img.shape[0] != N or r_img.shape[1] != N): r_img = cv2.resize(r_img, dsize = (n1,n1), interpolation = cv2.INTER_CUBIC)
		if(g_img.shape[0] != N or g_img.shape[1] != N): g_img = cv2.resize(g_img, dsize = (n1,n1), interpolation = cv2.INTER_CUBIC)
		if(b_img.shape[0] != N or b_img.shape[1] != N): b_img = cv2.resize(b_img, dsize = (n1,n1), interpolation = cv2.INTER_CUBIC)

		im1[k1:k2,m1:m2,0] = r_img
		im1[k1:k2,m1:m2,1] = g_img
		im1[k1:k2,m1:m2,2] = b_img

		rgb = Image.fromarray(im1)
		return rgb

	def mergeCellsToImage(self,m,unm,w,h,sip_cells):
		sorted_cells = self.sortCells(m,unm,sip_cells)
		N = len(sip_cells)
		display = np.empty(((w)*N, (h)*N), dtype=np.uint8)
		for i, j in it.product(range(N), range(N)):
			arr = sorted_cells[i*N+j]
			x,y = i*(w), j*(h)
			display[x : x + (w), y : y + (h)] = arr	
		return display

	def modifyMagnitude(self, mag_marked, mag_rest, D_array, sip_cells, gw, gh, copt):
		modified_cells = []
		unmodified_cells = []
		sip_hashes = []

		for i in range(len(mag_rest)):

			MAGNITUDE = mag_rest[i][0]
			PHASE = mag_rest[i][1]
			original_cell = self.getIFFTTransform(MAGNITUDE,PHASE)
			unmodified_cells.append(original_cell)		

		for i in range(len(mag_marked)):
			marked_cells = 0
			MAGNITUDE = mag_marked[i][0]
			PHASE = mag_marked[i][1]
			RED_AN = mag_marked[i][2]
			BLUE_AN = mag_marked[i][3]
			AVG_R = mag_marked[i][4]
			AVG_B = mag_marked[i][5]
			COORD_R = mag_marked[i][6]
			COORD_B = mag_marked[i][7]
			a = np.amax(MAGNITUDE)
			for j in range(gw):
				for k in range(gh):
					for t in range(len(COORD_R)):
						if(j == COORD_R[t][0] and k == COORD_R[t][1]):
							#print("mag ",MAGNITUDE[j,k])
							val = (AVG_B - AVG_R) + (D_array[i] + copt)
							MAGNITUDE[j,k] += val# change magnitude of red region cells
							marked_cells += 1
			original_cell = self.getIFFTTransform(MAGNITUDE,PHASE)
			#sip_hashes.append(self.hash_block(original_cell))
			modified_cells.append(original_cell)
		return modified_cells,unmodified_cells,marked_cells

	def embedPermutationToChannel(self, channel, sip, SIZE, copt, PR, PB, gridSize, RBWidth, Rxy, Bxy, moves):
		grid_cell_num = 0
		sip_cells = []

		# STEP 1: FIND 2D REPRESANTAION OF SIP
		A_matrix = []
		for i in range(0,len(sip)):
			row = []
			for j in range(0,SIZE):
				if(j == sip[i] - 1):
					row.append("*")
					sip_cells.append((i,j))
				else:
					row.append("-")
			A_matrix.append(row)


		# STEP 2: COMPUTE IMAGE SIZE
		M,N = channel.size
		channel_array = np.array(channel)

		# FOR EACH GRID CELL COMPUTE FFT MAGNITUDE AND PHASE:
		# ALSO COMPUTE IMAGINARY RED AND BLUE ANULUS RADIOUSES:

		MaxDRows,MaxD,avg = [],[],[]
		x,y,i = 0,0,0
		mag_red_blue,mag_rest = [],[]

		if(N < M):
			for r in range(moves[0], ((gridSize[1] * SIZE) + moves[0]), (gridSize[1])):
				for c in range(moves[1], ((gridSize[1] * SIZE) + moves[1]), (gridSize[1])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE,D = 0,0,0
					grid_cell = channel_array[r:r + gridSize[0],c:c + gridSize[1]]
					mag,phase = self.getFFTTransform(grid_cell)
					
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)
					
					red,coord_red = createEllipticDisk(mag,Rxy[0],Rxy[1],RBWidth[0],cx,cy,gridSize[0],gridSize[1])
					blue,coord_blue = createEllipticDisk(mag,Bxy[0],Bxy[1],RBWidth[1],cx,cy,gridSize[0],gridSize[1])
     
					if(len(red) == 0 or len(blue) == 0):
						raise ValueError(f"Invalid grid cell size {gridSize}: empty annulus at cell ({x},{y}) with moves={moves}, "f"Rxy={Rxy}, Bxy={Bxy}, RBWidth={RBWidth}")

					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0
					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))

						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1

				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,gridSize[0],gridSize[1],copt)
			img = self.mergeCellsToImage(m,unm,gridSize[0],gridSize[1],sip_cells)
			return img
		elif(N > M):
			for r in range(moves[0], ((gridSize[0] * SIZE) + moves[0]), (gridSize[0])):
				for c in range(moves[1], ((gridSize[0] * SIZE) + moves[1]), (gridSize[0])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE,D = 0,0,0
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

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0

					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))

						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1
				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,gridSize[0],gridSize[1],copt)
			img = self.mergeCellsToImage(m,unm,gridSize[0],gridSize[1],sip_cells)
			return img
		else:
			for r in range(moves[0], ((gridSize[0] * SIZE) + moves[0]), (gridSize[0])):
				for c in range(moves[1], ((gridSize[1] * SIZE) + moves[1]), (gridSize[1])):
					grid_cell_num += 1
					AVG_RED,AVG_BLUE,D = 0,0,0
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

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0

					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))
						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1
				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,gridSize[0],gridSize[1],copt)
			img = self.mergeCellsToImage(m,unm,gridSize[0],gridSize[1],sip_cells)
			return img