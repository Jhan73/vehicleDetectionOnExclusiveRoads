import os
import cv2
import torch
from torch.utils.data import Dataset

class PlateSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(224, 224), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size  # Tamaño al que redimensionar las imágenes
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])

        base_name = os.path.splitext(self.images[idx])[0]  # Obtener el nombre base (ejemplo: "image-0001")
        mask_path = os.path.join(self.mask_dir, base_name + ".jpg")
        
        # Cargar la imagen
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen en la ruta: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cargar la máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"No se pudo cargar la máscara en la ruta: {mask_path}")

        # Redimensionar la imagen y la máscara
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Convertir a tensores y aplicar transformaciones
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        else:
            # Si no se usan transformaciones adicionales, convertir a tensor directamente
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float) / 255.0
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0) / 255.0

        return image, mask

import segmentation_models_pytorch as smp

# Crear el modelo UNet con EfficientNet-B3
model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,  # Salida binaria para segmentación de placas
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Parámetros de entrenamiento
num_epochs = 15
batch_size = 8
learning_rate = 0.0001

# Definir dataset y dataloader
train_dataset = PlateSegmentationDataset("./datasets/segmentation/images", "./datasets/segmentation/masks")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Configurar el modelo, optimizador y función de pérdida
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Ciclo de entrenamiento
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks.float())
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model_weights.pth")
