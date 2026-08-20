# Checkpoint 1: Pipeline Base de Deep Learning

Este repositorio contiene la infraestructura inicial para el entrenamiento y validación de una Red Neuronal Artificial (MLP) desarrollada en PyTorch. El modelo base clasifica datos relacionados con flujos migratorios, evaluando el impacto de la gobernanza y tasas de pobreza.

## Entorno Técnico
- **Framework:** PyTorch (Versión utilizada: 2.x)
- **Dispositivo:** Detección automática (Soporta CPU, CUDA y MPS).
- **Optimizador:** Adam
- **Función de Pérdida:** Binary Cross Entropy (BCELoss)
- **Learning Rate:** `0.01` (Seleccionado para permitir una convergencia rápida y estable en este dataset inicial, evitando saltos muy grandes en los gradientes).

## Interpretación del Entrenamiento (Curva de Pérdida)
Durante la ejecución de 50 épocas, se observó lo siguiente:
- **Reducción de Loss:** La pérdida de entrenamiento (Train Loss) disminuyó consistentemente, indicando que el modelo ajustó sus pesos correctamente a través del Backpropagation.
- **Validación y Overfitting:** La pérdida de validación (Val Loss) acompañó la tendencia a la baja junto con un aumento en el *Validation Accuracy*, lo que indica que el modelo logra generalizar sin sobreajustarse (overfitting) a los datos de entrenamiento.

## Estructura del Repositorio
- `src/pipeline_base.py`: Contiene el ciclo de vida completo del modelo (creación, dataloaders, loop de entrenamiento y loop de validación).
- `requirements.txt`: Dependencias necesarias (`torch`, `numpy`, `scikit-learn`).