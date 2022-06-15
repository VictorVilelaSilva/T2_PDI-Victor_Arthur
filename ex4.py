from matplotlib import image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def computeKmeans(image, K):
    #transformando a matriz da imagem em uma matriz 
    # em que cada linha é um pixel da imagem
    matrizZ = image.reshape((-1, 3))
    matrizZ = np.float32(matrizZ)

    #definindo critério de parada para o algoritmo de kmeans
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 40, 0.1)
    #definindo o termo K do algoritmo kmeans

    _, labels, centroides = cv2.kmeans(matrizZ, K, None, stop, 40, cv2.KMEANS_RANDOM_CENTERS)

    centroides = np.uint8(centroides)

    imageWCentroidColors = centroides[labels.flatten()]
    finalImg = imageWCentroidColors.reshape((image.shape))

    plt.imshow(finalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('/content/drive/MyDrive/PDI/T2/strawhat.jpg')
plt.imshow(image)
computeKmeans(image, 8)
computeKmeans(image, 256)

#bug no plt.imshow do colab, mostrando a imagem azul. ao executar localmente a saida está correta