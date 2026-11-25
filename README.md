# Tasador Inteligente de Viviendas — Lima Metropolitana

Aplicación Streamlit que estima el precio de oferta de un departamento en los 43 distritos de Lima Metropolitana a partir de sus atributos físicos (Urbania) e indicadores distritales de criminalidad (MININTER, Mapa del Delito Georreferenciado).

Proyecto del curso Web Analytics (ESAN, 2026-2).

## Contenido

- `streamlit_app.py`: aplicación.
- `modelos_exportados/`: modelo entrenado (Gradient Boosting) y tablas que usa la app.
- `dataset_modelo_lima.csv`: avisos limpios usados para el entrenamiento y para la comparación por distrito.
- `requirements.txt`: versiones de las librerías. Las de scikit-learn, pandas, numpy y joblib deben coincidir con las del entorno donde se entrenó el modelo.

## Ejecución local

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

En share.streamlit.io: Create app, elegir este repositorio, archivo principal `streamlit_app.py` y, en Advanced settings, Python 3.12.
