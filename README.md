# Proyecto Redes Neuronales

## Introducción
(Objetivo del entregable, breve descripción del problema)

## Dataset
(Descripción del dataset elegido: tamaño, número de clases, distribución)

## Arquitectura del modelo
- Tipo: Perceptrón Multicapa (MLP)
- Framework: 
- Función de activación: 
- Función de pérdida: 
- Optimizador: 

<!-- ## Código
(Incluir snippet del modelo en Keras)

## Resultados
(Tablas/gráficas de accuracy, precisión, recall, F1)

## Conclusiones
(Breve análisis de resultados y próximos pasos) -->

## 📥 Descarga del dataset

El proyecto utiliza el dataset [Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) de Kaggle.  

Para descargarlo de forma automática, sigue estos pasos:

1. **Instala la API de Kaggle**
   ```bash
   pip install kaggle

2. **Configura tu token de Kaggle**

   - Ve a tu cuenta de Kaggle → **Account** → **Create API Token**.  
   - Se descargará un archivo llamado `kaggle.json`.  
   - Guárdalo en la ruta correspondiente según tu sistema operativo:

     - **Linux/Mac**  
       ```bash
       ~/.kaggle/kaggle.json
       ```

     - **Windows**  
       ```bash
       C:\Users\<tu_usuario>\.kaggle\kaggle.json
       ```

3. **Ejecuta el script de descarga**

   En la raíz del proyecto, corre el siguiente comando:

   ```bash
   python scripts/download_dataset.py