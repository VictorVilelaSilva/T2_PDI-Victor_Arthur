import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fp

img = cv2.imread('/content/drive/MyDrive/PDI/T2/pnois1.jpeg', 0) # imagem de entrada

w, h = img.shape

H = np.ones((w, h)) # filtro

mid = h//2
R = 256 #raio do circulo maior
r = 16 # distancia do centro
r2 = 150
r3 = 4

#cv2.circle(H,(h//2,w//2), R, 1, -1) # circulo mais externo (valores dentro do circulo sao iguais a 1)

# remove frequencias dentro do circulo
# as linhas desenhadas possuem dois pixels de largura
#cv2.line(H,(mid-r3,mid-r3),(mid-R,mid+r2), 0, 2)
#cv2.line(H,(mid+r3,mid+r3),(mid+R,mid-r2), 0, 2)
#cv2.line(H, (mid-r, mid), (mid-R,mid), 0, 2)
#cv2.line(H, (mid+r, mid), (mid+R,mid), 0, 2)

# testando retangulos para corrigir o ruido
cv2.rectangle(H,(mid-r2, mid),(mid-r, mid+10), 0, -1) 
cv2.rectangle(H,(mid+r2,mid),(mid+r, mid-10), 0, -1)

# colormap (escala de cinza)
cmap='gray'

plt.figure()
plt.title('Imagem original')
plt.imshow(img, cmap=cmap)
plt.colorbar()

H = fp.fftshift(H) #filtro passa-baixa
F = fp.fft2(img)

#tratando a FFT para os gráficos
Fm = np.absolute(F)
Fm /= Fm.max()
Fm = fp.fftshift(Fm)
Fm = np.log(Fm)

#mostrando a FFT em escala logaritmica
plt.figure()
plt.title('FFT em escala log')
plt.imshow(Fm, cmap=cmap)
plt.colorbar()

# aplicando o filtro
Fg = F*H
plt.figure()
plt.title('FFT filtrada em escala log')
Fga = np.absolute(Fg)
Fga = fp.fftshift(Fga)
Fga = np.log(Fga+1e-6)
plt.imshow(Fga, cmap=cmap)

# obtendo a transformada inversa, que é o sinal original (a imagem) filtrado
f_blurred = fp.ifft2(Fg)
f_blurred = np.absolute(f_blurred)

plt.figure()
plt.title('Imagem filtrada')
plt.imshow(f_blurred, cmap=cmap)
plt.colorbar()

plt.show()

#plt.imshow(circle)
plt.show()