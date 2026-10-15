import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("M01C02_final_mp4-0000_jpg.rf.2zxoE0vPe5oXpVgtlrao.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)                  
filtreNet=np.array([[0,-1,0],[-1, 5, -1],[0, -1, 0]])
img = cv2.filter2D(img, -1, filtreNet)
plt.imshow(img, cmap="gray")
plt.title("CLAHE + filtre Laplacien avec un 5")
plt.axis("off")
plt.show()

cv2.imwrite("image.jpg", img)