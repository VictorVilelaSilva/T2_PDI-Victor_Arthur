from statistics import median
from tkinter import filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import Frame
import cv2
from cv2 import medianBlur
import numpy as np

class GrabCutGUI(Frame):
    def __init__(self, master = None):

        Frame.__init__(self, master)

        self.iniciaUI()

    def iniciaUI(self):
        
        self.master.title("Janela da Imagem Segmentada")
        self.pack()

        self.computaAcoesDoMouse()

        self.imagem = self.carregaImagemASerExibida()

        self.canvas = Canvas(self.master, width = self.imagem.width(), height = self.imagem.height(), cursor = "cross")

        self.canvas.create_image(0, 0, anchor = NW, image = self.imagem)
        self.canvas.image = self.imagem 

        
        self.canvas.pack()

    def computaAcoesDoMouse(self):
        self.startX = None
        self.startY = None
        self.rect   = None
        self.rectangleReady = None
        
        self.master.bind("<ButtonPress-1>", self.callbackBotaoPressionado)
        self.master.bind("<B1-Motion>", self.callbackBotaoPressionadoEmMovimento)
        self.master.bind("<ButtonRelease-1>", self.callbackBotaoSolto)

    def callbackBotaoSolto(self, event):
        if self.rectangleReady:

            windowGrabcut = Toplevel(self.master)
            windowGrabcut.wm_title("Segmentation")
            windowGrabcut.minsize(width = self.imagem.width(), height = self.imagem.height())

            canvasGrabcut = Canvas(windowGrabcut, width = self.imagem.width(), height = self.imagem.height())
            canvasGrabcut.pack()

            mask = np.zeros(self.imagemOpenCV.shape[:2], np.uint8)
            print(mask.shape)
            rectGcut = (int(self.startX), int(self.startY), int(event.x - self.startX), int(event.y - self.startY))
            fundoModel = np.zeros((1, 65), np.float64)
            objModel = np.zeros((1, 65), np.float64)
            blur = cv2.GaussianBlur(self.imagemOpenCV, (9,9), 0)
            cv2.grabCut(self.imagemOpenCV, mask, rectGcut, fundoModel, objModel, 5, cv2.GC_INIT_WITH_RECT)

            maskFinal = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            imgFinal = self.imagemOpenCV * maskFinal[:,:,np.newaxis]
            for x in range(0, self.imagemOpenCV.shape[1]):
                for y in range(0, self.imagemOpenCV.shape[0]):
                    if(maskFinal[y][x] == 0):
                        imgFinal[y][x] = blur[y][x]


            imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)
            imgFinal = Image.fromarray(imgFinal)
            imgFinal = ImageTk.PhotoImage(imgFinal)


            canvasGrabcut.create_image(0, 0, anchor = NW, image = imgFinal)
            canvasGrabcut.image = imgFinal          

    def callbackBotaoPressionadoEmMovimento(self, event):

        currentX = self.canvas.canvasx(event.x)
        currentY = self.canvas.canvasy(event.y)

        self.canvas.coords(self.rect, self.startX, self.startY, currentX, currentY)

        self.rectangleReady = True

    def callbackBotaoPressionado(self, event):

        self.startX = self.canvas.canvasx(event.x)
        self.startY = self.canvas.canvasy(event.y)

        if not self.rect:
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="blue")

    def carregaImagemASerExibida(self):
        caminhoDaImagem = filedialog.askopenfilename()

        
        self.imagemOpenCV = cv2.imread(caminhoDaImagem)

        image = cv2.cvtColor(self.imagemOpenCV, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = ImageTk.PhotoImage(image)

        return image
            


root = Tk()

appcut = GrabCutGUI(master = root)

appcut.mainloop()

