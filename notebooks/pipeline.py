import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from datetime import datetime
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin

# Librerías para EEG
import mne
from mne.decoding import CSP

# Procesamiento de señales
from scipy import signal

# Preprocesamiento y ML
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Configuración
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eeg_pipeline')

# Ignorar warnings
warnings.filterwarnings('ignore')

# Directorio para modelos
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuración del experimento
CV_FOLDS = 5
RANDOM_SEED = 42

# Implementación más robusta del CSP transformer
class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False, n_channels=None):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.n_channels = n_channels
        self.csp = None
        
    def fit(self, X, y):
        # Determinar automáticamente el número de canales si no se especifica
        n_trials, n_features = X.shape
        
        # Calcular el número de canales automáticamente si no se proporciona
        if self.n_channels is None:
            # Intenta detectar un número de canales que resulte en una división exacta
            possible_channels = [64, 32, 24, 22, 21, 19, 16, 14, 8, 4, 2]
            for n_ch in possible_channels:
                if n_features % n_ch == 0:
                    self.n_channels = n_ch
                    logger.info(f"Se ha detectado automáticamente {n_ch} canales")
                    break
            
            # Si no se puede determinar, utilizar un enfoque de extracción de características simple
            if self.n_channels is None:
                logger.warning(f"No se pudo determinar automáticamente el número de canales. "
                               f"Por favor, especifique n_channels manualmente.")
                logger.warning("Usando PCA como alternativa a CSP")
                
                # Utilizar PCA como alternativa a CSP
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=min(n_features, 20))
                self.pca.fit(X)
                self.using_pca = True
                return self
        
        # Calcular número de tiempos basado en canales
        n_times = n_features // self.n_channels
        
        # Si el número de tiempos es 0, algo está mal
        if n_times == 0:
            logger.warning(f"El número de características ({n_features}) es menor que el número de canales ({self.n_channels})")
            logger.warning("Usando PCA como alternativa a CSP")
            
            # Utilizar PCA como alternativa a CSP
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=min(n_features, 20))
            self.pca.fit(X)
            self.using_pca = True
            return self
        
        # Configurar CSP
        self.csp = CSP(n_components=self.n_components, reg=self.reg, log=self.log, norm_trace=self.norm_trace)
        
        # Reshape para CSP (n_trials, n_channels, n_times)
        try:
            X_reshaped = X.reshape(n_trials, self.n_channels, n_times)
            self.csp.fit(X_reshaped, y)
            self.using_pca = False
        except ValueError as e:
            logger.warning(f"Error al reshape para CSP: {str(e)}")
            logger.warning("Usando PCA como alternativa a CSP")
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=min(n_features, 20))
            self.pca.fit(X)
            self.using_pca = True
        
        return self
    
    def transform(self, X):
        # Si estamos usando PCA en lugar de CSP
        if hasattr(self, 'using_pca') and self.using_pca:
            return self.pca.transform(X)
        
        # Reshape para CSP (n_trials, n_channels, n_times)
        n_trials, n_features = X.shape
        n_times = n_features // self.n_channels
        
        try:
            X_reshaped = X.reshape(n_trials, self.n_channels, n_times)
            return self.csp.transform(X_reshaped)
        except Exception as e:
            logger.error(f"Error en CSP transform: {str(e)}")
            # En caso de error, devolver características simples
            return np.column_stack([
                np.mean(X, axis=1),
                np.std(X, axis=1),
                np.max(X, axis=1),
                np.min(X, axis=1)
            ])

# Versión robusta del transformador de características frecuenciales
class FrequencyBandsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=160, n_channels=None, bands=None):
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.bands = bands or {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha_low': (8, 10),
            'alpha_high': (10, 13),
            'beta_low': (13, 16),
            'beta_mid': (16, 20),
            'beta_high': (20, 30),
            'gamma_low': (30, 45)
        }
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Detectar dimensiones
        n_trials, n_features = X.shape
        
        # Determinar número de canales si no se proporcionó
        if self.n_channels is None:
            # Intentar encontrar un número de canales que divida exactamente n_features
            possible_channels = [64, 32, 24, 22, 21, 19, 16, 14, 8, 4, 2]
            for n_ch in possible_channels:
                if n_features % n_ch == 0:
                    self.n_channels = n_ch
                    logger.info(f"Se han detectado automáticamente {n_ch} canales")
                    break
            
            # Si no se puede determinar, usar características simples
            if self.n_channels is None:
                logger.warning(f"No se pudo determinar el número de canales para {n_features} características")
                logger.warning("Usando características estadísticas básicas")
                return self._extract_basic_features(X)
        
        # Calcular número de muestras temporales
        n_times = n_features // self.n_channels
        
        # Si n_times es 0, usar características básicas
        if n_times == 0:
            logger.warning(f"Las características ({n_features}) son menos que los canales ({self.n_channels})")
            logger.warning("Usando características estadísticas básicas")
            return self._extract_basic_features(X)
        
        # Intentar reshape y PSD
        try:
            X_reshaped = X.reshape(n_trials, self.n_channels, n_times)
            return self._extract_frequency_features(X_reshaped)
        except Exception as e:
            logger.error(f"Error al extraer características frecuenciales: {str(e)}")
            return self._extract_basic_features(X)
    
    def _extract_basic_features(self, X):
        """Extrae características estadísticas básicas cuando no es posible el análisis frecuencial"""
        features = []
        for trial in X:
            # Características por trial
            trial_features = [
                np.mean(trial),               # Media
                np.std(trial),                # Desviación estándar
                np.min(trial),                # Mínimo
                np.max(trial),                # Máximo
                np.median(trial),             # Mediana
                np.percentile(trial, 25),     # Q1
                np.percentile(trial, 75),     # Q3
                np.percentile(trial, 75) - np.percentile(trial, 25),  # IQR
                np.sum(np.abs(np.diff(trial))),  # Complejidad (suma de diferencias)
                np.mean(np.abs(trial)),       # Media absoluta
                np.sqrt(np.mean(np.square(trial))),  # RMS
                np.sum(trial > np.mean(trial)),  # Tiempo sobre la media
                (np.max(trial) - np.min(trial))  # Rango
            ]
            features.append(trial_features)
        return np.array(features)
    
    def _extract_frequency_features(self, X_reshaped):
        """Extrae características frecuenciales si es posible el análisis de Fourier"""
        features = []
        
        for trial in X_reshaped:
            # Para cada canal, calcular PSD usando scipy
            trial_features = []
            
            for channel in trial:
                # Ajustar tamaño de segmento para welch
                nperseg = min(256, len(channel))
                if nperseg < 4:  # Si es demasiado pequeño para análisis frecuencial
                    # Características temporales simples
                    simple_features = [
                        np.mean(channel), np.std(channel), 
                        np.min(channel), np.max(channel)
                    ] * 8  # Repetir para tener mismo número que las bandas
                    trial_features.extend(simple_features)
                    continue
                
                # Calcular PSD
                freqs, psd = signal.welch(channel, fs=self.sfreq, nperseg=nperseg, nfft=max(256, nperseg*2))
                
                # Extraer características de bandas de frecuencia
                band_features = []
                for fmin, fmax in self.bands.values():
                    # Encontrar índices de frecuencia para la banda actual
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    
                    # Extraer características de esta banda
                    if np.any(idx_band):
                        band_psd = psd[idx_band]
                        band_features.extend([
                            np.mean(band_psd),       # Potencia media
                            np.std(band_psd),        # Desviación estándar
                            np.max(band_psd),        # Potencia máxima
                            np.sum(band_psd)         # Potencia total
                        ])
                    else:
                        band_features.extend([0, 0, 0, 0])
                
                # Características temporales adicionales
                temporal_features = [
                    np.std(channel),           # Desviación estándar
                    np.max(np.abs(channel)),   # Amplitud máxima
                    np.mean(np.abs(channel)),  # Amplitud media
                    np.percentile(channel, 75) - np.percentile(channel, 25)  # IQR
                ]
                
                trial_features.extend(band_features + temporal_features)
            
            features.append(trial_features)
        
        # Normalizar tamaños si hay inconsistencias
        feature_lengths = [len(f) for f in features]
        if len(set(feature_lengths)) > 1:
            max_len = max(feature_lengths)
            for i, feat in enumerate(features):
                if len(feat) < max_len:
                    # Rellenar con ceros si faltan características
                    features[i] = np.pad(feat, (0, max_len - len(feat)), 'constant')
        
        return np.array(features)

# Función para crear pipeline avanzado, más robusto
def create_advanced_pipeline(config='csp_svm', csp_components=6):
    """
    Crea un pipeline avanzado para clasificación EEG.
    
    Args:
        config (str): Configuración del pipeline
        csp_components (int): Número de componentes CSP
        
    Returns:
        Pipeline: Pipeline de scikit-learn configurado
    """
    # Estrategia de validación cruzada
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    if config == 'csp_svm':
        # Pipeline con CSP y SVM optimizado
        return Pipeline([
            ('scaler', RobustScaler()),
            ('csp', CSPTransformer(n_components=csp_components, n_channels=None)),
            ('classifier', GridSearchCV(
                SVC(probability=True, class_weight='balanced'),
                param_grid={
                    'C': [1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01],
                    'kernel': ['rbf']
                },
                cv=cv, scoring='accuracy', n_jobs=-1
            ))
        ])
    
    elif config == 'freq_rf':
        # Pipeline con características de frecuencia y Random Forest
        return Pipeline([
            ('scaler', StandardScaler()),
            ('freq_features', FrequencyBandsTransformer()),
            ('selector', SelectKBest(f_classif, k=50)),
            ('classifier', RandomForestClassifier(n_estimators=500, max_depth=None, 
                                                min_samples_split=2, bootstrap=True,
                                                class_weight='balanced', random_state=RANDOM_SEED,
                                                n_jobs=-1))
        ])
    
    elif config == 'csp_freq_rf':
        # Pipeline combinando CSP y características de frecuencia
        return Pipeline([
            ('scaler', StandardScaler()),
            ('csp', CSPTransformer(n_components=csp_components, n_channels=None)),
            ('freq_features', FrequencyBandsTransformer()),
            ('pca', PCA(n_components=30)),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=None,
                                               class_weight='balanced', random_state=RANDOM_SEED,
                                               n_jobs=-1))
        ])
    
    elif config == 'pca_mlp':
        # Pipeline con PCA y MLP con early stopping
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)),
            ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                        solver='adam', alpha=0.0001, batch_size='auto',
                                        learning_rate='adaptive', max_iter=300,
                                        early_stopping=True, validation_fraction=0.2,
                                        random_state=RANDOM_SEED))
        ])
    
    else:
        raise ValueError(f"Configuración desconocida: {config}")

# Función para comparar pipelines
def compare_pipelines(eeg_data, configs=None):
    """
    Compara diferentes configuraciones de pipelines.
    
    Args:
        eeg_data (list): Lista de información EEG
        configs (list): Lista de configuraciones de pipeline a probar
        
    Returns:
        dict: Resultados de la comparación
    """
    if configs is None:
        configs = ['csp_svm', 'freq_rf', 'csp_freq_rf', 'pca_mlp']
    
    results = {}
    
    # Preparar datos combinados
    X_combined = np.vstack([info['X'] for info in eeg_data])
    y_combined = np.concatenate([info['y'] for info in eeg_data])
    
    # Normalizar etiquetas si es necesario
    unique_labels = np.unique(y_combined)
    if len(unique_labels) > 3:  # Si hay más de 3 clases diferentes
        print("Encontradas más de 3 clases distintas. Normalizando etiquetas...")
        label_map = {}
        for i, label in enumerate(unique_labels):
            label_map[label] = i + 1
        
        # Aplicar mapeo
        for info in eeg_data:
            for old_label, new_label in label_map.items():
                info['y'][info['y'] == old_label] = new_label
        
        # Actualizar datos combinados
        y_combined = np.concatenate([info['y'] for info in eeg_data])
    
    print(f"Datos combinados: X shape {X_combined.shape}, y shape {y_combined.shape}")
    print(f"Clases: {np.unique(y_combined)}")
    print(f"Distribución de clases: {np.bincount(y_combined.astype(int))}")
    
    print("\nComparando configuraciones de pipelines con validación cruzada...")
    
    for config in configs:
        print(f"\nEvaluando pipeline: '{config}'")
        pipeline = create_advanced_pipeline(config)
        
        # Configurar validación cruzada
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        # Métricas a evaluar
        accuracy_scores = []
        f1_scores = []
        training_times = []
        prediction_times = []
        
        # Validación cruzada
        fold = 1
        for train_idx, test_idx in skf.split(X_combined, y_combined):
            print(f"  Fold {fold}/{CV_FOLDS}...")
            X_train, X_test = X_combined[train_idx], X_combined[test_idx]
            y_train, y_test = y_combined[train_idx], y_combined[test_idx]
            
            # Entrenar
            start_time = datetime.now()
            pipeline.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()
            
            # Predecir
            start_time = datetime.now()
            y_pred = pipeline.predict(X_test)
            predict_time = (datetime.now() - start_time).total_seconds()
            
            # Calcular métricas
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Guardar resultados
            accuracy_scores.append(acc)
            f1_scores.append(f1)
            training_times.append(train_time)
            prediction_times.append(predict_time)
            
            print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, Time: {train_time:.2f}s")
            fold += 1
        
        # Calcular promedios y desviaciones
        scores = {}
        scores['accuracy'] = np.mean(accuracy_scores)
        scores['accuracy_std'] = np.std(accuracy_scores)
        scores['f1_weighted'] = np.mean(f1_scores)
        scores['f1_weighted_std'] = np.std(f1_scores)
        scores['training_time'] = np.mean(training_times)
        scores['prediction_time'] = np.mean(prediction_times)
        
        # Mostrar resultados finales
        print(f"  Accuracy CV: {scores['accuracy']:.4f} ± {scores['accuracy_std']:.4f}")
        print(f"  F1 Score CV: {scores['f1_weighted']:.4f} ± {scores['f1_weighted_std']:.4f}")
        print(f"  Tiempo promedio entrenamiento: {scores['training_time']:.2f}s")
        
        # Guardar resultados
        results[config] = scores
    
    # Identificar mejor configuración
    best_config = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nMejor configuración: '{best_config}' con accuracy {results[best_config]['accuracy']:.4f}")
    
    return {
        'results': results,
        'best_config': best_config
    }

def hold_one_out_experiment(eeg_data, pipeline_config='csp_svm'):
    """
    Experimento hold-one-out con pipeline avanzado.
    
    Args:
        eeg_data (list): Lista de información EEG
        pipeline_config (str): Configuración del pipeline a usar
        
    Returns:
        dict: Resultados del experimento
    """
    n_eegs = len(eeg_data)
    results = []
    
    print(f"Iniciando experimento hold-one-out con pipeline '{pipeline_config}'\n")
    
    for i in range(n_eegs):
        print(f"Iteración {i+1}/{n_eegs} - Excluyendo Sujeto {eeg_data[i]['subject']}")
        
        # Separar datos de test y entrenamiento
        test_data = eeg_data[i]
        train_data = [eeg_data[j] for j in range(n_eegs) if j != i]
        
        # Verificar compatibilidad de clases
        test_classes = set(test_data['event_id'].keys())
        all_compatible = True
        
        for train_item in train_data:
            train_classes = set(train_item['event_id'].keys())
            if train_classes != test_classes:
                all_compatible = False
                break
        
        if not all_compatible:
            print("  ⚠️ Advertencia: Las clases en los datos de entrenamiento no coinciden con las clases de test")
            print("  ⚠️ Normalizando etiquetas para garantizar compatibilidad")
            
            # Normalizar etiquetas para asegurar compatibilidad
            # 1 -> rest, 2 -> clase1 (left_hand/both_hands), 3 -> clase2 (right_hand/both_feet)
            
            # Mapeo para test_data
            test_mapping = {}
            for idx, key in enumerate(['rest', 
                                      'left_hand' if 'left_hand' in test_data['event_id'] else 'both_hands',
                                      'right_hand' if 'right_hand' in test_data['event_id'] else 'both_feet']):
                if key in test_data['event_id']:
                    test_mapping[test_data['event_id'][key]] = idx + 1
            
            # Aplicar mapeo a datos de test
            y_test_original = test_data['y'].copy()
            for old_label, new_label in test_mapping.items():
                test_data['y'][test_data['y'] == old_label] = new_label
            
            # Aplicar mapeo a datos de entrenamiento
            for train_item in train_data:
                train_mapping = {}
                for idx, key in enumerate(['rest', 
                                          'left_hand' if 'left_hand' in train_item['event_id'] else 'both_hands',
                                          'right_hand' if 'right_hand' in train_item['event_id'] else 'both_feet']):
                    if key in train_item['event_id']:
                        train_mapping[train_item['event_id'][key]] = idx + 1
                
                # Aplicar mapeo
                for old_label, new_label in train_mapping.items():
                    train_item['y'][train_item['y'] == old_label] = new_label
        
        # Combinar datos de entrenamiento
        X_train_combined = np.vstack([item['X'] for item in train_data])
        y_train_combined = np.concatenate([item['y'] for item in train_data])
        
        # Datos de test
        X_test = test_data['X']
        y_test = test_data['y']
        
        print(f"  Datos de entrenamiento: {X_train_combined.shape}, Datos de test: {X_test.shape}")
        print(f"  Clases en train: {np.unique(y_train_combined)}, Clases en test: {np.unique(y_test)}")
        
        # Crear y entrenar pipeline avanzado
        pipeline = create_advanced_pipeline(pipeline_config)
        
        print("  Entrenando modelo...")
        start_time = datetime.now()
        pipeline.fit(X_train_combined, y_train_combined)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Predecir en datos de test
        print("  Evaluando en datos de test...")
        start_time = datetime.now()
        y_pred = pipeline.predict(X_test)
        predict_time = (datetime.now() - start_time).total_seconds()
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # Mapear IDs numéricos a nombres de clases
        if all_compatible:
            id_to_class = {v: k for k, v in test_data['event_id'].items()}
        else:
            # Usar mapeo genérico si se normalizaron las etiquetas
            id_to_class = {1: 'rest', 2: 'clase1', 3: 'clase2'}
        
        class_names = [id_to_class.get(c, f"Clase {c}") for c in sorted(np.unique(y_test))]
        
        # Generar reporte
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Guardar resultados de esta iteración
        iter_results = {
            'subject': test_data['subject'],
            'paradigm': test_data['paradigm'],
            'task_type': test_data['task_type'],
            'accuracy': acc,
            'f1_score': f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_test,
            'y_pred': y_pred,
            'class_mapping': id_to_class,
            'pipeline': pipeline  # Guardar el pipeline entrenado
        }
        
        results.append(iter_results)
        
        print(f"  Resultados: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
        print(f"  Tiempo: Entrenamiento = {train_time:.2f}s, Predicción = {predict_time:.2f}s\n")
    
    # Calcular métricas promedio
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])
    
    print(f"Resultados finales del experimento:")
    print(f"  Promedio Accuracy: {avg_accuracy:.4f}")
    print(f"  Promedio F1 Score: {avg_f1:.4f}")
    
    return {
        'iterations': results,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        'pipeline_config': pipeline_config
    }

def plot_confusion_matrices(experiment_results):
    """
    Visualiza las matrices de confusión para cada iteración del experimento.
    """
    iterations = experiment_results['iterations']
    n_eegs = len(iterations)
    
    # Crear rejilla para gráficos
    n_cols = 3
    n_rows = (n_eegs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows > 1:
        axes = axes.flatten()
    
    for i, result in enumerate(iterations):
        if n_eegs == 1:
            ax = axes
        else:
            ax = axes[i]
            
        cm = result['confusion_matrix']
        
        # Obtener nombres de clases
        class_mapping = result['class_mapping']
        class_names = [class_mapping.get(idx, f"Clase {idx}") for idx in sorted(class_mapping.keys())]
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                  yticklabels=class_names, ax=ax, cbar=False)
        
        # Configurar título y etiquetas
        ax.set_title(f"Sujeto {result['subject']} - {result['paradigm']}\nAcc: {result['accuracy']:.3f}")
        ax.set_ylabel('Clase real')
        ax.set_xlabel('Clase predicha')
    
    # Ocultar ejes no utilizados
    if n_eegs < len(axes):
        for j in range(n_eegs, len(axes)):
            axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Matrices de Confusión por Sujeto', y=1.02, fontsize=16)
    return fig

def plot_subject_performances(experiment_results):
    """
    Visualiza el rendimiento para cada sujeto en el experimento.
    """
    iterations = experiment_results['iterations']
    
    # Extraer datos para visualización
    subjects = [r['subject'] for r in iterations]
    accuracies = [r['accuracy'] for r in iterations]
    f1_scores = [r['f1_score'] for r in iterations]
    task_types = [r['task_type'] for r in iterations]
    paradigms = [r['paradigm'] for r in iterations]
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Sujeto': subjects,
        'Accuracy': accuracies,
        'F1 Score': f1_scores,
        'Tipo de Tarea': task_types,
        'Paradigma': paradigms
    })
    
    # Ordenar por accuracy
    df_sorted = df.sort_values('Accuracy', ascending=False)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Barras para accuracy y F1
    x = np.arange(len(df_sorted))
    width = 0.35
    
    ax.bar(x - width/2, df_sorted['Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x + width/2, df_sorted['F1 Score'], width, label='F1 Score', color='#2ecc71')
    
    # Configurar etiquetas de eje X
    labels = [f"S{s}\n({p[:1]}{'E' if t == 'motor_execution' else 'I'})" 
             for s, p, t in zip(df_sorted['Sujeto'], df_sorted['Paradigma'], df_sorted['Tipo de Tarea'])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Añadir etiquetas
    for i, acc in enumerate(df_sorted['Accuracy']):
        ax.text(i - width/2, acc + 0.01, f"{acc:.3f}", ha='center')
    
    for i, f1 in enumerate(df_sorted['F1 Score']):
        ax.text(i + width/2, f1 + 0.01, f"{f1:.3f}", ha='center')
    
    # Configurar gráfico
    ax.set_ylabel('Puntuación')
    ax.set_title('Rendimiento por Sujeto')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir línea para promedio
    avg_acc = experiment_results['avg_accuracy']
    ax.axhline(y=avg_acc, linestyle='--', color='#e74c3c', alpha=0.7)
    ax.text(len(df_sorted)-1, avg_acc + 0.02, f"Promedio: {avg_acc:.3f}", ha='right', color='#e74c3c')
    
    # Añadir leyenda para las abreviaturas
    legend_text = "Abreviaturas:\nL = left_right_hand, H = hands_feet\nE = motor_execution, I = motor_imagery"
    ax.text(0.02, -0.15, legend_text, transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def pipeline_comparison_chart(pipeline_results):
    """
    Crea un gráfico de barras comparando diferentes pipelines.
    """
    # Extraer resultados
    configs = list(pipeline_results['results'].keys())
    accuracies = [pipeline_results['results'][c]['accuracy'] for c in configs]
    f1_scores = [pipeline_results['results'][c]['f1_weighted'] for c in configs]
    acc_std = [pipeline_results['results'][c]['accuracy_std'] for c in configs]
    
    # Ordenar por accuracy
    sorted_indices = np.argsort(accuracies)[::-1]  # Orden descendente
    configs = [configs[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    acc_std = [acc_std[i] for i in sorted_indices]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Barras con error
    ax.bar(x - width/2, accuracies, width, yerr=acc_std, 
          label='Accuracy', color='#3498db', capsize=5)
    ax.bar(x + width/2, f1_scores, width, 
          label='F1 Score', color='#2ecc71')
    
    # Añadir etiquetas de valor
    for i, acc in enumerate(accuracies):
        ax.text(i - width/2, acc + acc_std[i] + 0.01, f"{acc:.3f}", ha='center')
    
    for i, f1 in enumerate(f1_scores):
        ax.text(i + width/2, f1 + 0.01, f"{f1:.3f}", ha='center')
    
    # Configurar gráfico
    ax.set_ylabel('Puntuación')
    ax.set_title('Comparación de Pipelines')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Resaltar el mejor pipeline
    best_config = pipeline_results['best_config']
    best_idx = configs.index(best_config)
    ax.get_xticklabels()[best_idx].set_color('red')
    ax.get_xticklabels()[best_idx].set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def train_and_save_model(eeg_data, pipeline_config='csp_svm', model_name=None):
    """
    Entrena un modelo con todos los datos y lo guarda
    
    Args:
        eeg_data (list): Lista de información EEG
        pipeline_config (str): Configuración del pipeline
        model_name (str): Nombre base para el modelo
        
    Returns:
        tuple: (pipeline, model_info)
    """
    # Preparar datos combinados
    X_combined = np.vstack([info['X'] for info in eeg_data])
    y_combined = np.concatenate([info['y'] for info in eeg_data])
    
    # Crear pipeline
    pipeline = create_advanced_pipeline(pipeline_config)
    
    # Entrenar modelo
    print(f"Entrenando modelo final con pipeline '{pipeline_config}'...")
    start_time = datetime.now()
    pipeline.fit(X_combined, y_combined)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Modelo entrenado en {train_time:.2f} segundos")
    
    # Evaluar modelo con validación cruzada
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_combined, y_combined, cv=cv, scoring='accuracy')
    
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Guardar modelo
    if model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'eeg_model_{timestamp}'
    
    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    dump(pipeline, model_path)
    
    # Preparar información del modelo
    subjects = [info['subject'] for info in eeg_data]
    paradigms = [info['paradigm'] for info in eeg_data]
    task_types = [info['task_type'] for info in eeg_data]
    
    model_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_config': pipeline_config,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'subjects': subjects,
        'paradigms': paradigms,
        'task_types': task_types,
        'classes': sorted([str(c) for c in np.unique(y_combined)]),
        'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y_combined, return_counts=True))},
        'feature_shape': X_combined.shape,
        'training_time': train_time,
        'model_file': model_path
    }
    
    # Guardar información del modelo
    info_path = os.path.join(MODELS_DIR, f'{model_name}_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Modelo guardado en: {model_path}")
    print(f"Información guardada en: {info_path}")
    
    return pipeline, model_info
