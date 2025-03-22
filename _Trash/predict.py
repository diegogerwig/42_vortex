import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from joblib import load
from datetime import datetime
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.viz import plot_raw

# Configuración
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eeg_predict')

# Ignorar warnings
warnings.filterwarnings('ignore')

# Directorio para modelos
MODELS_DIR = '../models'

def load_model(model_path=None):
    """
    Carga un modelo previamente entrenado.
    
    Args:
        model_path (str): Ruta al archivo del modelo. Si es None, busca el modelo más reciente.
        
    Returns:
        tuple: (pipeline, model_info)
    """
    import glob
    
    if model_path is None or '*' in model_path:
        # Buscar el modelo más reciente
        if model_path and '*' in model_path:
            # Usar glob para expandir el comodín
            model_files = glob.glob(model_path)
        else:
            # Buscar en el directorio de modelos
            model_files = glob.glob(os.path.join(MODELS_DIR, '*.joblib'))
            
        if not model_files:
            raise FileNotFoundError("No se encontraron modelos en el directorio de modelos")
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = model_files[0]
        
        # Buscar archivo de información correspondiente
        info_path = model_path.replace('.joblib', '_info.json')
        if not os.path.exists(info_path):
            logger.warning(f"No se encontró archivo de información para el modelo {model_path}")
            info = None
        else:
            with open(info_path, 'r') as f:
                info = json.load(f)
    else:
        # Usar el modelo especificado
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en la ruta {model_path}")
        
        # Buscar archivo de información correspondiente
        info_path = model_path.replace('.joblib', '_info.json')
        if not os.path.exists(info_path):
            logger.warning(f"No se encontró archivo de información para el modelo {model_path}")
            info = None
        else:
            with open(info_path, 'r') as f:
                info = json.load(f)
    
    # Cargar el modelo
    logger.info(f"Cargando modelo desde {model_path}")
    pipeline = load(model_path)
    
    if info:
        logger.info(f"Modelo: {info.get('pipeline_config', 'Desconocido')}")
        logger.info(f"Accuracy CV: {info.get('cv_accuracy', 'Desconocido')}")
    
    return pipeline, info

def load_specific_subject(subject_id, run_id):
    """
    Carga los datos EEG de un sujeto específico.
    
    Args:
        subject_id (int): ID del sujeto
        run_id (int): ID del run
        
    Returns:
        tuple: (raw_data, task_type, paradigm)
    """
    logger.info(f"Cargando datos del sujeto {subject_id}, run {run_id}")
    
    # Cargar los datos usando la función de MNE
    raw_files = eegbci.load_data(subject_id, [run_id])
    
    if not raw_files:
        raise ValueError(f"No se encontraron archivos para el sujeto {subject_id}, run {run_id}")
    
    # Leer el archivo EDF
    raw_data = read_raw_edf(raw_files[0], preload=True)
    
    # Estandarizar nombres de canales al sistema internacional 10-20
    eegbci.standardize(raw_data)
    
    # Configurar montaje EEG
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_data.set_montage(montage)
    
    # Determinar el tipo de tarea y paradigma
    if run_id in [3, 5, 7, 9, 11, 13]:
        task_type = "motor_execution"
    else:
        task_type = "motor_imagery"
    
    if run_id in [3, 4, 7, 8, 11, 12]:
        paradigm = "left_right_hand"
    else:  # runs 5, 6, 9, 10, 13, 14
        paradigm = "hands_feet"
    
    # Guardar metadatos
    metadata = {
        'subject': subject_id,
        'task_type': task_type,
        'paradigm': paradigm,
        'run': run_id
    }
    
    raw_data.metadata = metadata
    
    return raw_data, task_type, paradigm

# Versión simplificada de preprocess_data para ser compatible con la nueva versión
def preprocess_data(raw_data, params=None):
    """
    Aplica preprocesamiento a los datos EEG crudos.
    
    Args:
        raw_data (mne.io.Raw): Datos EEG crudos
        params (dict): Parámetros de preprocesamiento
        
    Returns:
        tuple: (X, y, epochs, event_id)
    """
    # Parámetros por defecto
    default_params = {
        'low_cutoff': 4,
        'high_cutoff': 40,
        'apply_notch': True,
        'tmin': 0.0,
        'tmax': 4.0,
        'baseline': (0, 0),
        'baseline_correction': True,
        'exclude_rest': True
    }
    
    # Usar parámetros proporcionados o por defecto
    if params is None:
        params = default_params
    
    # Crear copia para no modificar los datos originales
    filter_data = raw_data.copy()
    
    # Aplicar filtro pasa banda
    logger.info(f"Aplicando filtro pasa banda ({params['low_cutoff']}-{params['high_cutoff']} Hz)...")
    filter_data.filter(params['low_cutoff'], params['high_cutoff'], fir_design='firwin')
    
    # Aplicar filtro notch si es necesario
    if params['apply_notch']:
        logger.info("Aplicando filtro notch a 60Hz...")
        filter_data.notch_filter(freqs=[60], fir_design='firwin')
    
    # Extraer eventos de las anotaciones
    events, event_id = mne.events_from_annotations(filter_data)
    
    # Mapear IDs de eventos a nombres más descriptivos
    metadata = getattr(filter_data, 'metadata', {})
    paradigm = metadata.get('paradigm', '')
    
    if paradigm == 'left_right_hand':
        new_event_id = {
            'rest': event_id.get('T0', 0),
            'left_hand': event_id.get('T1', 0),
            'right_hand': event_id.get('T2', 0)
        }
    else:  # hands_feet
        new_event_id = {
            'rest': event_id.get('T0', 0),
            'both_hands': event_id.get('T1', 0),
            'both_feet': event_id.get('T2', 0)
        }
    
    # Eliminar eventos con valor 0 (no encontrados)
    new_event_id = {k: v for k, v in new_event_id.items() if v != 0}
    
    logger.info(f"Mapeo de eventos: {new_event_id}")
    
    # Excluir eventos de descanso si está activado
    if params.get('exclude_rest', True) and 'rest' in new_event_id:
        logger.info("Excluyendo eventos de descanso (REST)...")
        event_id_no_rest = {k: v for k, v in new_event_id.items() if k != 'rest'}
        
        # Verificar que quedan eventos después de excluir 'rest'
        if not event_id_no_rest:
            logger.warning("No quedan eventos después de excluir 'rest'. Se usarán todos los eventos.")
        else:
            new_event_id = event_id_no_rest
    
    # Crear épocas
    epochs = mne.Epochs(
        filter_data,
        events,
        event_id=new_event_id,
        tmin=params['tmin'],
        tmax=params['tmax'],
        baseline=(0, 0) if params.get('baseline_correction', True) else None,
        preload=True
    )
    
    logger.info(f"Creadas {len(epochs)} épocas con {len(epochs.ch_names)} canales")
    
    # Extraer características para ML
    X = epochs.get_data()  # Forma: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # Etiquetas
    
    # Reshape para ML (aplanar características)
    n_epochs, n_channels, n_times = X.shape
    X_flat = X.reshape(n_epochs, n_channels * n_times)
    
    logger.info(f"Datos extraídos: X shape {X_flat.shape}, y shape {y.shape}")
    
    return X_flat, y, epochs, new_event_id

def predict_eeg(raw_data, pipeline, show_raw=False, preprocessing_params=None):
    """
    Realiza predicciones sobre datos EEG.
    
    Args:
        raw_data (mne.io.Raw): Datos EEG crudos
        pipeline (sklearn.pipeline.Pipeline): Pipeline entrenado
        show_raw (bool): Si se muestra la visualización de datos crudos
        preprocessing_params (dict): Parámetros para el preprocesamiento
        
    Returns:
        dict: Resultados de la predicción
    """
    if preprocessing_params is None:
        preprocessing_params = {
            'low_cutoff': 4,
            'high_cutoff': 40,
            'apply_notch': True,
            'tmin': 0.0,
            'tmax': 4.0,
            'baseline': (0, 0),
            'baseline_correction': True,
            'exclude_rest': True
        }
    
    # Visualizar datos crudos si se solicita
    if show_raw:
        fig = plot_raw(raw_data, title='Datos EEG crudos', show=False)
        plt.tight_layout()
        plt.show()
    
    # Preprocesar datos
    logger.info("Preprocesando datos...")
    X, y, epochs, event_id = preprocess_data(raw_data, preprocessing_params)
    
    # Realizar predicción
    logger.info("Realizando predicción...")
    start_time = datetime.now()
    y_pred = pipeline.predict(X)
    predict_time = (datetime.now() - start_time).total_seconds()
    
    # Si existe ground truth, calcular métricas
    if y is not None and len(y) > 0:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Tiempo de predicción: {predict_time:.4f} segundos")
        
        # Mapear IDs numéricos a nombres de clases
        id_to_class = {v: k for k, v in event_id.items()}
        class_names = [id_to_class.get(c, f"Clase {c}") for c in sorted(np.unique(y))]
        
        # Generar reporte
        report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                   yticklabels=class_names, cbar=False)
        plt.title(f"Matriz de Confusión\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}")
        plt.ylabel('Clase real')
        plt.xlabel('Clase predicha')
        plt.tight_layout()
        plt.show()
        
        results = {
            'X': X,
            'y_true': y,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predict_time': predict_time,
            'class_mapping': id_to_class,
            'epochs': epochs
        }
    else:
        # Si no hay ground truth, solo devolver predicciones
        results = {
            'X': X,
            'y_pred': y_pred,
            'predict_time': predict_time,
            'epochs': epochs
        }
    
    return results

def visualize_predictions_over_time(predict_results):
    """
    Visualiza las predicciones sobre el tiempo.
    
    Args:
        predict_results (dict): Resultados de la predicción
    """
    # Obtener datos
    epochs = predict_results['epochs']
    y_pred = predict_results['y_pred']
    
    if 'y_true' in predict_results:
        y_true = predict_results['y_true']
        class_mapping = predict_results['class_mapping']
        
        # Mapear clases a nombres
        class_names = {v: k for k, v in class_mapping.items()}
        y_pred_names = [class_names.get(y, f"Clase {y}") for y in y_pred]
        y_true_names = [class_names.get(y, f"Clase {y}") for y in y_true]
        
        # Crear dataframe con predicciones por época
        df = pd.DataFrame({
            'Tiempo (s)': epochs.times,
            'Predicción': y_pred_names[0],
            'Clase real': y_true_names[0]
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(epochs.times, [1 if p == r else 0 for p, r in zip(y_pred, y_true)], 'g-', label='Predicción correcta')
        plt.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
        plt.title('Predicciones a lo largo del tiempo')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Predicción correcta (1) / incorrecta (0)')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        # Solo visualizar predicciones sin ground truth
        unique_predictions = np.unique(y_pred)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_predictions)))
        
        plt.figure(figsize=(12, 6))
        for i, pred_class in enumerate(unique_predictions):
            mask = y_pred == pred_class
            plt.scatter(np.arange(len(y_pred))[mask], np.ones(np.sum(mask))*i, 
                       label=f'Clase {pred_class}', color=colors[i], alpha=0.7)
        
        plt.title('Predicciones por época')
        plt.xlabel('Número de época')
        plt.ylabel('Clase predicha')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_feature_importance(pipeline, feature_names=None):
    """
    Visualiza la importancia de características para modelos que lo soporten.
    
    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline entrenado
        feature_names (list): Nombres de las características
    """
    # Intentar obtener el estimador final
    if hasattr(pipeline, 'named_steps'):
        if 'classifier' in pipeline.named_steps:
            estimator = pipeline.named_steps['classifier']
            
            # Si es GridSearchCV, obtener el mejor estimador
            if hasattr(estimator, 'best_estimator_'):
                estimator = estimator.best_estimator_
            
            # Verificar si el modelo soporta importancia de características
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                
                # Si no se proporcionan nombres, usar índices
                if feature_names is None:
                    feature_names = [f'Característica {i}' for i in range(len(importances))]
                
                # Ordenar por importancia
                indices = np.argsort(importances)[::-1]
                
                # Tomar las 20 características más importantes
                top_n = 20
                indices = indices[:top_n]
                
                plt.figure(figsize=(10, 8))
                plt.title('Importancia de características')
                plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Importancia relativa')
                plt.tight_layout()
                plt.show()
            else:
                logger.warning("El modelo no soporta visualización de importancia de características")
        else:
            logger.warning("No se encontró un clasificador en el pipeline")
    else:
        logger.warning("El objeto proporcionado no es un pipeline válido")

def batch_predict(pipeline, data_directory, subject_ids=None, runs=None):
    """
    Realiza predicciones en lote para múltiples sujetos y runs.
    
    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline entrenado
        data_directory (str): Directorio de datos
        subject_ids (list): Lista de IDs de sujetos, si es None, usa 1-5
        runs (list): Lista de runs, si es None, usa runs predeterminados
        
    Returns:
        dict: Resultados de las predicciones
    """
    if subject_ids is None:
        subject_ids = list(range(1, 6))  # Sujetos 1-5 por defecto
    
    if runs is None:
        # Usar un conjunto de runs motor_execution e imagery
        runs = [4, 8, 12]  # Imagery izquierda/derecha, ambas manos/pies
    
    results = {}
    
    for subject in subject_ids:
        subject_results = {}
        
        for run in runs:
            try:
                # Cargar datos
                raw_data, task_type, paradigm = load_specific_subject(subject, run)
                
                # Realizar predicción
                prediction = predict_eeg(raw_data, pipeline, show_raw=False)
                
                # Guardar resultados
                subject_results[f'run_{run}'] = {
                    'task_type': task_type,
                    'paradigm': paradigm,
                    'accuracy': prediction.get('accuracy', None),
                    'f1_score': prediction.get('f1_score', None),
                    'predict_time': prediction['predict_time']
                }
                
                logger.info(f"Sujeto {subject}, Run {run}: Predicción completada")
                
            except Exception as e:
                logger.error(f"Error en sujeto {subject}, run {run}: {str(e)}")
                subject_results[f'run_{run}'] = {'error': str(e)}
        
        results[f'subject_{subject}'] = subject_results
    
    # Generar resumen
    summary = {}
    for subject_key, subject_data in results.items():
        for run_key, run_data in subject_data.items():
            if 'accuracy' in run_data and run_data['accuracy'] is not None:
                if run_data['task_type'] not in summary:
                    summary[run_data['task_type']] = []
                
                summary[run_data['task_type']].append({
                    'subject': subject_key,
                    'run': run_key,
                    'paradigm': run_data['paradigm'],
                    'accuracy': run_data['accuracy'],
                    'f1_score': run_data['f1_score']
                })
    
    # Calcular promedios
    for task_type, task_data in summary.items():
        avg_acc = np.mean([item['accuracy'] for item in task_data])
        avg_f1 = np.mean([item['f1_score'] for item in task_data])
        
        logger.info(f"Promedio para {task_type}: Accuracy = {avg_acc:.4f}, F1 = {avg_f1:.4f}")
    
    return results, summary
