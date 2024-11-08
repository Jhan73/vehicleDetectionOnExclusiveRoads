import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el modelo de segmentación (debe coincidir con la arquitectura usada en el entrenamiento)
model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,  # Salida binaria para segmentación de placas
)

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model = model.to(device)
model.eval()

# Función para procesar y predecir la máscara de una imagen nueva
def predict_mask(image_path):
    # Cargar y preprocesar la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 160))  # Cambiar el tamaño a la entrada que usaste en el entrenamiento
    image = image / 255.0  # Normalizar a [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Cambiar a (canal, alto, ancho)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device)  # Añadir dimensión de batch

    # Hacer la predicción
    with torch.no_grad():
        pred_mask = model(image)
        pred_mask = torch.sigmoid(pred_mask)  # Aplicar función sigmoide para obtener probabilidad
        pred_mask = (pred_mask > 0.5).float()  # Binarizar con umbral de 0.5

    # Convertir la máscara predicha a formato numpy para visualización
    pred_mask = pred_mask.squeeze().cpu().numpy()

    plt.imsave('pred_mask.png', pred_mask, cmap='gray')

    # Mostrar la imagen original y la máscara predicha
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Máscara Predicha")
    plt.axis("off")
    plt.show()

# Prueba el modelo con una nueva imagen
predict_mask("./placa.jpg")
