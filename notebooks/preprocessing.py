import numpy as np
import pandas as pd
import random
import os
import logging
import warnings
from datetime import datetime

# Librerías para EEG
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws

# Procesamiento de señales
from scipy import signal

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eeg_preprocessing')

# Ignorar warnings
warnings.filterwarnings('ignore')

# Parámetros de preprocesamiento
PREPROCESSING_PARAMS = {
    'low_cutoff': 8,      # Hz - Incluimos ondas theta (4-8 Hz)
    'high_cutoff': 40,    # Hz - Incluimos hasta gama bajo (30-45 Hz)
    'apply_notch': True,  # Filtro notch para ruido de línea eléctrica
    'tmin': -1.0,         # Tiempo inicial para épocas (segundos)
    'tmax': 4.0,          # Tiempo final para épocas (segundos)
    'csp_components': 6   # Componentes CSP a utilizar
}

# Definir los cuatro grupos de experimentos
EXPERIMENT_GROUPS = {
    'motor_execution_left_right': {
        'runs': [3, 7, 11],
        'task_type': 'motor_execution',
        'paradigm': 'left_right_hand',
        'description': 'Motor execution: left vs right hand'
    },
    'motor_imagery_left_right': {
        'runs': [4, 8, 12],
        'task_type': 'motor_imagery',
        'paradigm': 'left_right_hand',
        'description': 'Motor imagery: left vs right hand'
    },
    'motor_execution_hands_feet': {
        'runs': [5, 9, 13],
        'task_type': 'motor_execution',
        'paradigm': 'hands_feet',
        'description': 'Motor execution: hands vs feet'
    },
    'motor_imagery_hands_feet': {
        'runs': [6, 10, 14],
        'task_type': 'motor_imagery',
        'paradigm': 'hands_feet',
        'description': 'Motor imagery: hands vs feet'
    }
}

# Función para cargar un sujeto específico con un experimento específico
def load_specific_subject(subject_id, run_id):
    """
    Carga datos EEG de un sujeto y run específicos.
    
    Args:
        subject_id (int): ID del sujeto a cargar
        run_id (int): ID del run a cargar
        
    Returns:
        tuple: (raw_data, subject_id, run_id, task_type, paradigm)
    """
    logger.info(f"Cargando sujeto: {subject_id}, run: {run_id}")
    
    # Determinar el tipo de tarea y paradigma según el run
    if run_id in [3, 7, 11]:
        task_type = "motor_execution"
        paradigm = "left_right_hand"
    elif run_id in [4, 8, 12]:
        task_type = "motor_imagery"
        paradigm = "left_right_hand"
    elif run_id in [5, 9, 13]:
        task_type = "motor_execution"
        paradigm = "hands_feet"
    elif run_id in [6, 10, 14]:
        task_type = "motor_imagery"
        paradigm = "hands_feet"
    else:
        raise ValueError(f"Run ID {run_id} no válido")
    
    logger.info(f"Tipo de tarea: {task_type}, paradigma: {paradigm}")
    
    # Cargar los datos usando la función de MNE
    raw_files = eegbci.load_data(subject_id, [run_id])
    
    if not raw_files:
        raise ValueError(f"No se encontraron archivos para el sujeto {subject_id}, run {run_id}")
    
    # Leer y concatenar los archivos EDF
    raws = [read_raw_edf(f, preload=True) for f in raw_files]
    raw_data = concatenate_raws(raws)
    
    # Estandarizar nombres de canales al sistema internacional 10-20
    eegbci.standardize(raw_data)
    
    # Configurar montaje EEG
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_data.set_montage(montage)
    
    # Guardar metadatos del sujeto
    raw_data.info['subject_info'] = {'his_id': str(subject_id)}
    
    # Guardar metadatos adicionales como atributo
    metadata = {
        'subject': subject_id,
        'task_type': task_type,
        'paradigm': paradigm,
        'run': run_id
    }
    
    raw_data.metadata = metadata
    
    return raw_data, subject_id, run_id, task_type, paradigm

# Función para preprocesar datos
def preprocess_data(raw_data, low_cutoff=4, high_cutoff=45, apply_notch=True, tmin=-1.0, tmax=4.0):
    """
    Aplica preprocesamiento a los datos EEG crudos.
    """
    # Crear copia para no modificar los datos originales
    filter_data = raw_data.copy()
    
    # Aplicar filtro pasa banda
    logger.info(f"Aplicando filtro pasa banda ({low_cutoff}-{high_cutoff} Hz)...")
    filter_data.filter(low_cutoff, high_cutoff, fir_design='firwin')
    
    # Aplicar filtro notch si es necesario
    if apply_notch:
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
    
    # Crear épocas
    epochs = mne.Epochs(
        filter_data,
        events,
        event_id=new_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
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

# Función para cargar múltiples sujetos con un grupo de experimento específico
def load_subjects_for_experiment(num_subjects=10, experiment_group=None, random_seed=None):
    """
    Carga datos EEG de múltiples sujetos para un grupo de experimento específico.
    
    Args:
        num_subjects (int): Número de sujetos a cargar
        experiment_group (str): Grupo de experimento a utilizar, si es None se elige aleatoriamente
        random_seed (int): Semilla para reproducibilidad
        
    Returns:
        list: Lista con información de EEG
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Si no se especifica un grupo, elegir uno aleatoriamente
    if experiment_group is None:
        experiment_group = random.choice(list(EXPERIMENT_GROUPS.keys()))
    
    # Obtener configuración del grupo de experimento
    group_config = EXPERIMENT_GROUPS[experiment_group]
    runs = group_config['runs']
    task_type = group_config['task_type']
    paradigm = group_config['paradigm']
    
    print(f"Grupo de experimento seleccionado: {group_config['description']}")
    print(f"Ejecutando experimentos con runs: {runs}")
    print(f"Tipo de tarea: {task_type}, Paradigma: {paradigm}")
    
    # Seleccionar sujetos aleatorios
    available_subjects = list(range(1, 110))  # Sujetos 1-109
    selected_subjects = random.sample(available_subjects, num_subjects)
    
    print(f"\nSujetos seleccionados: {selected_subjects}")
    
    eeg_data = []
    total_eegs = 0
    
    # Para cada sujeto, cargar los 3 runs del grupo de experimento
    for subject in selected_subjects:
        for run in runs:
            print(f"\nCargando EEG #{total_eegs+1}:")
            print(f"Sujeto: {subject}, Run: {run}")
            
            # Intentar cargar y preprocesar, con reintentos en caso de error
            max_attempts = 5
            success = False
            
            for attempt in range(max_attempts):
                try:
                    # Cargar datos específicos
                    raw_data, subj_id, run_id, task_type, paradigm = load_specific_subject(subject, run)
                    
                    # Preprocesar datos
                    X, y, epochs, event_id = preprocess_data(
                        raw_data, 
                        PREPROCESSING_PARAMS['low_cutoff'],
                        PREPROCESSING_PARAMS['high_cutoff'],
                        PREPROCESSING_PARAMS['apply_notch'],
                        PREPROCESSING_PARAMS['tmin'],
                        PREPROCESSING_PARAMS['tmax']
                    )
                    
                    # Guardar información relevante
                    eeg_info = {
                        'subject': subject,
                        'run': run,
                        'task_type': task_type,
                        'paradigm': paradigm,
                        'X': X,
                        'y': y,
                        'epochs': epochs,
                        'event_id': event_id,
                        'class_counts': {k: np.sum(y == v) for k, v in event_id.items()},
                        'experiment_group': experiment_group
                    }
                    
                    eeg_data.append(eeg_info)
                    total_eegs += 1
                    
                    # Resumen
                    print(f"  Sujeto: {subject}, Run: {run}, Tarea: {task_type}, Paradigma: {paradigm}")
                    print(f"  Forma de datos: {X.shape}")
                    print(f"  Clases: {list(event_id.keys())}")
                    print(f"  Distribución de clases: {eeg_info['class_counts']}")
                    
                    success = True
                    break
                    
                except Exception as e:
                    print(f"Error al cargar el sujeto {subject}, run {run}: {str(e)}")
                    print(f"Reintentando ({attempt+1}/{max_attempts})...")
                    
                    if attempt == max_attempts - 1:
                        print(f"No se pudo cargar el EEG para el sujeto {subject}, run {run} después de {max_attempts} intentos")
                        # Continuar con el siguiente run o sujeto
            
            if not success:
                print(f"Saltando al siguiente EEG...")
    
    print(f"\nCargados {len(eeg_data)} EEGs exitosamente de {len(selected_subjects)} sujetos.")
    return eeg_data

# Función para guardar metadatos de los sujetos
def save_metadata(eeg_data, filename='eeg_metadata.csv'):
    """Guarda los metadatos de los sujetos en un archivo CSV en el directorio ../models"""
    # Asegurar que el directorio ../models existe
    models_dir = os.path.join('..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Directorio {models_dir} creado")
    
    # Construir la ruta completa del archivo
    file_path = os.path.join(models_dir, filename)
    
    metadata = []
    for info in eeg_data:
        meta = {
            'subject': info['subject'],
            'run': info['run'],
            'task_type': info['task_type'],
            'paradigm': info['paradigm'],
            'experiment_group': info.get('experiment_group', ''),
            'num_samples': info['X'].shape[0],
            'num_features': info['X'].shape[1],
            'classes': ','.join(info['event_id'].keys()),
            'class_distribution': str(info['class_counts'])
        }
        metadata.append(meta)
    
    df = pd.DataFrame(metadata)
    df.to_csv(file_path, index=False)
    logger.info(f"Metadatos guardados en {file_path}")
    return df

# Función para normalizar etiquetas entre diferentes paradigmas
def normalize_labels(eeg_data):
    """
    Normaliza las etiquetas para garantizar compatibilidad entre paradigmas
    1 -> rest, 2 -> clase1 (left_hand/both_hands), 3 -> clase2 (right_hand/both_feet)
    """
    for info in eeg_data:
        mapping = {}
        event_id = info['event_id']
        
        # Asignar nuevos valores según el paradigma
        if 'rest' in event_id:
            mapping[event_id['rest']] = 1
        
        if 'left_hand' in event_id:
            mapping[event_id['left_hand']] = 2
        elif 'both_hands' in event_id:
            mapping[event_id['both_hands']] = 2
            
        if 'right_hand' in event_id:
            mapping[event_id['right_hand']] = 3
        elif 'both_feet' in event_id:
            mapping[event_id['both_feet']] = 3
        
        # Aplicar mapeo a las etiquetas
        y_original = info['y'].copy()
        for old_label, new_label in mapping.items():
            info['y'][info['y'] == old_label] = new_label
        
        # Actualizar metadata
        info['original_event_id'] = info['event_id'].copy()
        info['event_id'] = {'rest': 1, 'clase1': 2, 'clase2': 3}
        info['class_counts'] = {k: np.sum(info['y'] == v) for k, v in info['event_id'].items()}
        
    return eeg_data

# Función para guardar los datos procesados
# Función para guardar los datos procesados
def save_processed_data(eeg_data, base_filename='eeg_dataset'):
    """
    Guarda los datos procesados en archivos numpy en el directorio ../models
    y crea un único archivo JSON con metadatos completos para facilitar cargas posteriores
    
    Args:
        eeg_data (list): Lista de datos EEG procesados
        base_filename (str): Nombre base para los archivos
    
    Returns:
        dict: Rutas de los archivos guardados e información del conjunto de datos
    """
    import json
    
    # Asegurar que el directorio ../models existe
    models_dir = os.path.join('..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Directorio {models_dir} creado")
    
    # Crear subdirectorio para este conjunto de datos con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(models_dir, f"{base_filename}_{timestamp}")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        logger.info(f"Directorio {dataset_dir} creado")
    
    # Extraer información del grupo de experimento
    experiment_group = eeg_data[0].get('experiment_group', 'unknown_group')
    
    # Preparar datos para guardar
    all_X = []
    all_y = []
    all_subjects = []
    all_runs = []
    
    # Preparar metadatos detallados
    subjects_metadata = []
    
    for info in eeg_data:
        X = info['X']
        y = info['y']
        subject = info['subject']
        run = info['run']
        
        # Agregar a las listas para los arrays principales
        n_samples = X.shape[0]
        all_X.append(X)
        all_y.append(y)
        all_subjects.extend([subject] * n_samples)
        all_runs.extend([run] * n_samples)
        
        # Recopilar metadatos para este sujeto/run
        subject_meta = {
            'subject': subject,
            'run': run,
            'task_type': info['task_type'],
            'paradigm': info['paradigm'],
            'experiment_group': info.get('experiment_group', ''),
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'classes': list(info['event_id'].keys()),
            'class_counts': {k: int(np.sum(y == v)) for k, v in info['event_id'].items()},
            'sample_indices': {
                'start': len(all_subjects) - n_samples,
                'end': len(all_subjects)
            }
        }
        
        subjects_metadata.append(subject_meta)
    
    # Concatenar todos los datos
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    subjects_all = np.array(all_subjects)
    runs_all = np.array(all_runs)
    
    # Guardar archivos de datos
    np.save(os.path.join(dataset_dir, 'X_all.npy'), X_all)
    np.save(os.path.join(dataset_dir, 'y_all.npy'), y_all)
    np.save(os.path.join(dataset_dir, 'subjects_all.npy'), subjects_all)
    np.save(os.path.join(dataset_dir, 'runs_all.npy'), runs_all)
    
    # Crear configuración completa
    dataset_config = {
        'dataset_info': {
            'timestamp': timestamp,
            'experiment_group': experiment_group,
            'n_subjects': len(set(subjects_all)),
            'n_runs': len(set(runs_all)),
            'n_samples': X_all.shape[0],
            'n_features': X_all.shape[1],
            'class_distribution': {
                'rest': int(np.sum(y_all == 1)),
                'clase1': int(np.sum(y_all == 2)),
                'clase2': int(np.sum(y_all == 3))
            }
        },
        'preprocessing_params': PREPROCESSING_PARAMS,
        'subjects_data': subjects_metadata,
        'experiment_group_info': EXPERIMENT_GROUPS[experiment_group]
    }
    
    # Guardar configuración como JSON (único archivo de metadatos)
    metadata_file = os.path.join(dataset_dir, 'dataset_info.json')
    with open(metadata_file, 'w') as f:
        json.dump(dataset_config, f, indent=4)
    
    logger.info(f"Datos y metadatos completos guardados en {dataset_dir}")
    
    # Retornar información sobre los archivos guardados
    return {
        'dataset_dir': dataset_dir,
        'files': {
            'X': os.path.join(dataset_dir, 'X_all.npy'),
            'y': os.path.join(dataset_dir, 'y_all.npy'),
            'subjects': os.path.join(dataset_dir, 'subjects_all.npy'),
            'runs': os.path.join(dataset_dir, 'runs_all.npy'),
            'metadata': metadata_file
        },
        'config': dataset_config
    }

# Función para limpiar la memoria de los datos grandes
def clean_memory(eeg_data):
    """
    Limpia la memoria eliminando los datos grandes dentro de eeg_data
    
    Args:
        eeg_data (list): Lista de datos EEG procesados
    
    Returns:
        list: Lista de datos EEG sin los arrays grandes
    """
    import gc
    
    # Crear una copia ligera de los datos
    light_data = []
    
    for info in eeg_data:
        # Guardar dimensiones de los datos
        X_shape = info['X'].shape
        y_shape = info['y'].shape
        
        # Crear una versión ligera sin los arrays grandes
        light_info = {k: v for k, v in info.items() if k not in ['X', 'y', 'epochs']}
        light_info['X_shape'] = X_shape
        light_info['y_shape'] = y_shape
        
        light_data.append(light_info)
    
    # Limpiar memoria
    eeg_data.clear()
    gc.collect()
    
    logger.info("Memoria limpiada. Arrays grandes eliminados de eeg_data")
    return light_data
