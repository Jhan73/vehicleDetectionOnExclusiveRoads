import cv2
import pytesseract

# Configuración de Tesseract para Windows
# Asegúrate de que esta ruta es correcta según tu instalación de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def recognize_plate_text(plate_image_path):
    # Cargar la imagen de la placa
    image = cv2.imread(plate_image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta proporcionada.")
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarización con umbral adaptativo para mejorar el contraste del texto
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configuración de Tesseract para reconocer texto alfanumérico
    # --psm 7: trata la imagen como una sola línea de texto
    # tessedit_char_whitelist: define los caracteres permitidos (letras y números)
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Extraer el texto usando Tesseract OCR
    plate_text = pytesseract.image_to_string(binary_image, config=config)
    
    return plate_text.strip()  # Eliminar espacios en blanco al inicio y final

# Prueba de la función con una imagen de la placa
image_path = "frontoparalelo.png"  # Reemplaza con la ruta de tu imagen
text = recognize_plate_text(image_path)

print("Texto reconocido en la placa:", text)
