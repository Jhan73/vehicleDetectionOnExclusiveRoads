# Documentación
## Etiquetado en formato YOLO

Cada archivo .txt correspondiente a una imagen debe contener líneas con el formato:
```
<class> <x_center> <y_center> <width> <height>
```
* `<class>`: índice de la clase del objeto.
* `<x_center> <y_center>`: coordenadas normalizadas del centro del cuadro delimitador.
* `<width> <height>`: ancho y alto del cuadro delimitador, normalizados entre 0 y 1.

## Estructura de archivos
1. `datasets/`
En esta carpeta se almacenan los datasets organizados por el tipo de tarea (detección de autos, detección de placas, segmentación). Cada dataset se divide en subcarpetas `train`, `val`, y `test`, lo que facilita el manejo del conjunto de datos durante el entrenamiento y la evaluación.
    * `autos/`: Contiene imágenes etiquetadas para la detección de autos (bounding boxes).
    * `placas/`: Contiene imágenes etiquetadas para la detección de placas vehiculares.
    * `segmentacion/`: Contiene imágenes con etiquetas de segmentación (máscaras).

2. `models/`
Aquí se almacenan los modelos entrenados para cada tarea. Cada tarea (detección de autos, placas y segmentación) tiene su propia carpeta donde se guardan:
    * `checkpoints/`: Guardar los pesos intermedios durante el entrenamiento.
    * `model_autos_final.pth`: El modelo final entrenado.
    * `logs/`: Guardar logs de entrenamiento (pérdidas, precisión, etc.) que puedes usar para monitorear el rendimiento.

3. `scripts/`
Esta carpeta contiene los scripts Python que usarás para entrenar y evaluar los modelos.
    * `entrenamiento_autos.py`: Entrenamiento para el modelo de detección de autos.
    * `entrenamiento_placas.py`: Entrenamiento para el modelo de detección de placas.
    * `entrenamiento_segmentacion.py`: Entrenamiento para el modelo de segmentación.
    * `utils.py`: Funciones auxiliares como carga de datos, preprocesamiento, métricas, etc.
    * `evaluacion.py`: Un script para evaluar el rendimiento de los modelos sobre los datos de validación o test.
    * `inference.py`: Un script que carga un modelo entrenado y realiza inferencia sobre nuevas imágenes.

4. `configs/`
Configuraciones de hiperparámetros para cada modelo. Guardar las configuraciones en archivos `YAML` o `JSON` es útil para mantener el código más limpio y permite ajustes rápidos en los parámetros de entrenamiento.
    * `config_autos.yaml`: Configuración de hiperparámetros (como tasa de aprendizaje, tamaño del lote, número de épocas, etc.) para el modelo de detección de autos.
    * `config_placas.yaml`: Configuración para detección de placas.
    * `config_segmentacion.yaml`: Configuración para segmentación.

5. `resultados/`
    Esta carpeta almacena los resultados finales del entrenamiento y evaluación.

    * `graficas_entrenamiento/`: Gráficas que muestran la evolución de la pérdida, precisión, etc., a lo largo del tiempo.
    * `metricas_finales.txt`: Archivo que guarda las métricas finales (por ejemplo, AP para detección, IoU para segmentación) al final del entrenamiento.