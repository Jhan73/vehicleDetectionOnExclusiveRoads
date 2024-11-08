from PIL import Image
from pytesseract import *

pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = Image.open("placa.jpg")

resultado = pytesseract.image_to_string(img)

print("Texto predicho: ", resultado)