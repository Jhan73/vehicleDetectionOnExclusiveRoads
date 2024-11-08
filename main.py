from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def main():
    # model = YOLO('yolov8n.pt')
    # results = model.train(
    #     data = 'dataset.yaml',
    #     epochs = 50,
    #     imgsz = (900, 1600),
    #     batch = 4,
    #     name = 'piggy-model',
    #     optimizer = 'Adam',
    # )
    model = YOLO('yolov8n.pt')

    image_path = './metropolitano.jpg'

    img = cv2.imread(image_path)

    #Realizar deteccion
    results = model(img)

    # Mostrar resultados
    annotated_img = results[0].plot()  # Anotamos las detecciones en la imagen

    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()




if __name__ == "__main__":

    print("Hola mundo")

