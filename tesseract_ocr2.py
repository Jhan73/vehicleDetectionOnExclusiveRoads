import cv2
import pytesseract

# Configuración de Tesseract para Windows
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def recognize_plate_text(plate_image_path):
    # Cargar la imagen de la placa
    image = cv2.imread(plate_image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta proporcionada.")
    
    # Redimensionar la imagen para mejorar la precisión de OCR (ajustar si es necesario)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para suavizar sin perder bordes
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
    
    # Detectar bordes usando Canny
    edged_image = cv2.Canny(gray_image, 30, 200)

    # Encontrar contornos y filtrar por los que podrían ser la placa
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_region = None

    for contour in contours:
        # Aproximar el contorno a un rectángulo
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el contorno tiene 4 lados, es probable que sea la placa
        if len(approx) == 4:
            plate_region = approx
            break

    if plate_region is not None:
        # Crear una máscara para extraer solo la región de la placa
        mask = cv2.drawContours(gray_image.copy(), [plate_region], -1, 255, -1)
        out = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        # Recortar la región de la placa
        x, y, w, h = cv2.boundingRect(plate_region)
        cropped_plate = gray_image[y:y + h, x:x + w]
        
        # Binarización para mejorar el contraste del texto
        _, binary_image = cv2.threshold(cropped_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Configuración de Tesseract para reconocer texto alfanumérico
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Extraer el texto usando Tesseract OCR
        plate_text = pytesseract.image_to_string(binary_image, config=config)
        
        return plate_text.strip()  # Eliminar espacios en blanco al inicio y final
    
    else:
        return "No se detectó una región de placa en la imagen."

# Prueba de la función con una imagen de la placa
image_path = "frontoparalelo.png"  # Reemplaza con la ruta de tu imagen
text = recognize_plate_text(image_path)

print("Texto reconocido en la placa:", text)
