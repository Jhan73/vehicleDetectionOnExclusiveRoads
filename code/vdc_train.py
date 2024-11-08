from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def train(model_size='yolov8n.pt', data_yaml='data.yaml', epochs=50, batch_size=8, img_size=640, save_dir='runs/train'):
    """
    Train the YOLOv8 model.
    
    :param model_size: Pre-trained model size ('yolov8n.pt', 'yolov8s.pt', etc.).
    :param data_yaml: Path to the YAML file with dataset info.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param img_size: Image size for training.
    :param save_dir: Directory where the model and results will be saved.
    """

    model = YOLO(model_size) 

    model.train(
        data=data_yaml,  # Ruta al archivo de configuración YAML
        epochs=epochs,         # Número de épocas de entrenamiento
        batch_size=batch_size,      # Tamaño del lote
        imgsz=img_size,         # Tamaño de la imagen (por ejemplo, 640x640)
        save_dir=save_dir,    # Directory where the model will be saved
        optimizer = 'Adam',
    )

def load_trained_model(weights_path):
    """
    Load a trained YOLOv8 model.
    
    :param weights_path: Path to the .pt file with trained weights.
    :return: Loaded YOLO model.
    """

    model = YOLO(weights_path)
    print(f"Model loaded from {weights_path}")
    return model

def perform_detection(model, image_path):
    """
    Perform object detection on an image.
    
    :param model: Loaded YOLO model.
    :param image_path: Path to the image for detection.
    :return: Detection results.
    """
    results = model(image_path)
    results.show()
    return results


def save_detection_results(results, output_dir='output/'):
    """
    Save the detection results to a specified directory.
    
    :param results: Results from the detection.
    :param output_dir: Directory to save the output images.
    """
    results.save(save_dir=output_dir)
    print(f"Results saved to {output_dir}")

def display_results(image_path):
    """
    Display an image using matplotlib.
    
    :param image_path: Path to the image to display.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def print_detection_details(results, model):
    """
    Print the details of the detected objects.
    
    :param results: Results from the detection.
    :param model: Loaded YOLO model.
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence of detection
            cls = int(box.cls[0])  # Class index of the detected object
            label = model.names[cls]  # Name of the detected class
            print(f'Class: {label}, Confidence: {conf}, Coordinates: ({x1}, {y1}), ({x2}, {y2})')


train()

model = load_trained_model('./runs/detect/train/weights/best.pt')

# Perform detection on a sample image
image_path = 'path/to/sample_image.jpg'
results = perform_detection(model, image_path)

# Save and display the results
save_detection_results(results, output_dir='output/')
display_results(image_path)
# Print detailed information about the detections
print_detection_details(results, model)


