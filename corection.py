import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Definir el modelo de segmentación y cargar los pesos
model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load("model_weights.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Cargar la imagen de prueba
image = cv2.imread("placa.jpg")
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta proporcionada.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Función para predecir la máscara de segmentación
def predict_mask(model, image, device):
    model.eval()
    with torch.no_grad():
        image_resized = cv2.resize(image, (224, 224))
        image_tensor = torch.tensor(image_resized.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)

        # Predicción de la máscara
        pred_mask = model(image_tensor)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float().cpu().numpy().squeeze()
        
        # Redimensionar la máscara a las dimensiones originales de la imagen
        mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0])) * 255
        mask_resized = mask_resized.astype(np.uint8)
    return mask_resized

# Función para encontrar el contorno de la placa
def find_plate_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Función para obtener las esquinas de la placa
def get_plate_corners(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    else:
        raise ValueError("No se encontraron exactamente 4 esquinas en el contorno de la placa.")
    return corners

# Función para ordenar las esquinas en el orden correcto
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Función para corregir la perspectiva de la placa
def correct_perspective(image, corners):
    rect = order_points(corners)
    (width, height) = (440, 140)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# Obtener la máscara de la placa
mask = predict_mask(model, image, device)

# Encontrar el contorno de la placa
contour = find_plate_contours(mask)

# Obtener las esquinas de la placa
try:
    corners = get_plate_corners(contour)
except ValueError as e:
    print(e)

# Corregir la perspectiva usando las esquinas encontradas
warped_plate = correct_perspective(image, corners)

plt.imsave('corregido.png', warped_plate, cmap='gray')

# Mostrar la imagen original, la máscara y la placa corregida
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Máscara de Segmentación")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(warped_plate)
plt.title("Placa Corregida")
plt.axis("off")

plt.show()
