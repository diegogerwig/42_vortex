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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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

# Función para filtrar eventos de reposo (rest)
def filter_rest_events(eeg_data):
    """
    Filtra los eventos de 'rest' (clase 1) de los datos EEG.
    
    Args:
        eeg_data (list): Lista de información EEG
        
    Returns:
        list: Lista de información EEG sin eventos de 'rest'
    """
    filtered_eeg_data = []
    
    for subject_data in eeg_data:
        # Identificar índices que no son eventos de reposo (clase != 1)
        non_rest_idx = np.where(subject_data['y'] != 1)[0]
        
        if len(non_rest_idx) == 0:
            logger.warning(f"Sujeto {subject_data['subject']} no tiene eventos distintos a 'rest'")
            continue
        
        # Crear un nuevo diccionario con los datos filtrados
        filtered_data = {
            'subject': subject_data['subject'],
            'run': subject_data.get('run', 0),
            'task_type': subject_data['task_type'],
            'paradigm': subject_data['paradigm'],
            'experiment_group': subject_data.get('experiment_group', 'unknown'),
            'X': subject_data['X'][non_rest_idx],
            'y': subject_data['y'][non_rest_idx],
        }
        
        # Ajustar las etiquetas para que sean 0 y 1 (en vez de 2 y 3)
        old_labels = np.unique(filtered_data['y'])
        label_map = {old_label: i for i, old_label in enumerate(old_labels)}
        
        filtered_data['y'] = np.array([label_map[y] for y in filtered_data['y']])
        filtered_data['event_id'] = {k: label_map.get(v, v) for k, v in subject_data.get('event_id', {}).items() 
                                    if v != 1}  # Excluir 'rest'
        
        # Si existe class_counts, ajustarlo
        if 'class_counts' in subject_data:
            filtered_counts = {str(label_map.get(int(k), k)): v 
                              for k, v in subject_data['class_counts'].items() 
                              if int(k) != 1}
            filtered_data['class_counts'] = filtered_counts
        
        # Añadir a la lista de datos filtrados
        filtered_eeg_data.append(filtered_data)
        
        logger.info(f"Sujeto {subject_data['subject']}: {len(non_rest_idx)} eventos no-rest encontrados")
        logger.info(f"Nuevas etiquetas: {np.unique(filtered_data['y'])}")
    
    logger.info(f"Total de {len(filtered_eeg_data)} sujetos después de filtrar eventos 'rest'")
    return filtered_eeg_data

# Implementación más robusta del CSP transformer
class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=0.1, log=True, norm_trace=False, n_channels=None):
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

# Versión mejorada del transformador de características frecuenciales
class FrequencyBandsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=160, n_channels=None, bands=None):
        self.sfreq = sfreq
        self.n_channels = n_channels
        # Bandas más específicas para motor imagery
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

# Características de asimetría para motor imagery
class AsymmetryFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_channels=None):
        self.n_channels = n_channels
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_trials, n_features = X.shape
        
        # Determinar automáticamente el número de canales
        if self.n_channels is None:
            possible_channels = [64, 32, 24, 22, 21, 19, 16, 14, 8, 4, 2]
            for n_ch in possible_channels:
                if n_features % n_ch == 0:
                    self.n_channels = n_ch
                    break
            
            if self.n_channels is None:
                # Si no se puede determinar, devolver características simples
                return self._extract_basic_features(X)
        
        n_times = n_features // self.n_channels
        
        try:
            X_reshaped = X.reshape(n_trials, self.n_channels, n_times)
            return self._extract_asymmetry_features(X_reshaped)
        except Exception as e:
            logger.error(f"Error al extraer características de asimetría: {str(e)}")
            return self._extract_basic_features(X)
    
    def _extract_basic_features(self, X):
        """Extrae características básicas cuando no es posible el análisis de asimetría"""
        return np.column_stack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.max(X, axis=1),
            np.min(X, axis=1)
        ])
    
    def _extract_asymmetry_features(self, X_reshaped):
        """Extrae características de asimetría entre hemisferios"""
        n_trials, n_channels, n_times = X_reshaped.shape
        features = []
        
        for trial in X_reshaped:
            trial_features = []
            
            # Asumimos un montaje estándar donde los canales están agrupados por hemisferios
            # Esto es una simplificación - en la práctica se necesitaría conocer el montaje exacto
            
            # Para un montaje típico de 64 canales, podemos comparar:
            # C3 vs C4 (canales motores centrales)
            # P3 vs P4 (canales parietales)
            # O1 vs O2 (canales occipitales)
            
            # Simplificación: comparamos canales equidistantes del centro
            half_channels = n_channels // 2
            
            for i in range(min(6, half_channels)):  # Limitamos a 6 pares o menos
                left_ch = trial[i]
                right_ch = trial[n_channels - i - 1]
                
                # Calcular asimetría en el dominio del tiempo
                asymmetry_time = left_ch - right_ch
                
                # Características de asimetría temporal
                trial_features.extend([
                    np.mean(asymmetry_time),        # Media de la asimetría
                    np.std(asymmetry_time),         # Variabilidad de la asimetría
                    np.max(np.abs(asymmetry_time))  # Máxima asimetría absoluta
                ])
                
                # Asimetría en frecuencia (simplificada)
                # Espectro de potencia para ambos canales
                left_psd = np.abs(np.fft.fft(left_ch)[:n_times//2])**2
                right_psd = np.abs(np.fft.fft(right_ch)[:n_times//2])**2
                
                # Calcular asimetría espectral (diferencia logarítmica)
                with np.errstate(divide='ignore', invalid='ignore'):
                    asymmetry_freq = np.log(left_psd) - np.log(right_psd)
                    asymmetry_freq = np.nan_to_num(asymmetry_freq)  # Manejar inf/nan
                
                # Características de asimetría frecuencial
                trial_features.extend([
                    np.mean(asymmetry_freq),        # Media de asimetría espectral
                    np.std(asymmetry_freq),         # Variabilidad de asimetría espectral
                    np.sum(asymmetry_freq > 0),     # Predominancia izquierda vs derecha
                    np.sum(asymmetry_freq < 0)      # Predominancia derecha vs izquierda
                ])
            
            features.append(trial_features)
        
        # Normalizar tamaños si hay inconsistencias
        feature_lengths = [len(f) for f in features]
        if len(set(feature_lengths)) > 1:
            max_len = max(feature_lengths)
            for i, feat in enumerate(features):
                if len(feat) < max_len:
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

# Función para crear pipeline para clasificación binaria (sin rest)
def create_binary_pipeline(config='csp_svm', csp_components=6):
    """
    Crea un pipeline para clasificación binaria de EEG (sin eventos 'rest').
    
    Args:
        config (str): Configuración del pipeline
        csp_components (int): Número de componentes CSP
        
    Returns:
        Pipeline: Pipeline de scikit-learn configurado
    """
    # Estrategia de validación cruzada
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    if config == 'ensemble':
        # Pipeline ensemble mejorado
        return Pipeline([
            ('scaler', RobustScaler()),
            ('csp', CSPTransformer(n_components=csp_components, reg=0.1, n_channels=None)),
            ('classifier', VotingClassifier(
                estimators=[
                    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')),
                    ('rf', RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced', random_state=RANDOM_SEED)),
                    ('gbm', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_SEED))
                ],
                voting='soft'
            ))
        ])
    
    elif config == 'csp_svm':
        # Pipeline con CSP y SVM optimizado
        return Pipeline([
            ('scaler', RobustScaler()),
            ('csp', CSPTransformer(n_components=csp_components, reg=0.1, n_channels=None)),
            ('classifier', GridSearchCV(
                SVC(probability=True, class_weight='balanced'),
                param_grid={
                    'C': [1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.001],
                    'kernel': ['rbf']
                },
                cv=cv, scoring='accuracy', n_jobs=-1
            ))
        ])
    
    elif config == 'csp_freq_rf':
        # Pipeline combinando CSP y características de frecuencia con RF
        return Pipeline([
            ('scaler', StandardScaler()),
            ('csp', CSPTransformer(n_components=csp_components, reg=0.1, n_channels=None)),
            ('freq_features', FrequencyBandsTransformer(sfreq=160)),
            ('pca', PCA(n_components=40)),
            ('classifier', RandomForestClassifier(
                n_estimators=1000,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                criterion='entropy'
            ))
        ])
    
    elif config == 'asymmetry_rf':
        # Pipeline con características de asimetría (específico para motor imagery)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('asymmetry', AsymmetryFeaturesTransformer()),
            ('selector', SelectKBest(f_classif, k=40)),
            ('classifier', RandomForestClassifier(
                n_estimators=800, 
                max_depth=None,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ])
    
    elif config == 'pca_mlp':
        # Pipeline con PCA y MLP mejorado
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=RANDOM_SEED
            ))
        ])
    
    elif config == 'enhanced_csp_rf':
        # Pipeline combinando CSP, características frecuenciales y de asimetría
        return Pipeline([
            ('scaler', RobustScaler()),
            ('features', FeatureUnion([
                ('csp', CSPTransformer(n_components=csp_components, reg=0.1, n_channels=None)),
                ('freq', FrequencyBandsTransformer(sfreq=160)),
                ('asymmetry', AsymmetryFeaturesTransformer())
            ])),
            ('selector', SelectKBest(f_classif, k=60)),
            ('classifier', RandomForestClassifier(
                n_estimators=1200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='auto',
                bootstrap=True,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
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

# Función para comparar pipelines binarios
def compare_binary_pipelines(eeg_data, configs=None):
    """
    Compara diferentes configuraciones de pipelines binarios.
    
    Args:
        eeg_data (list): Lista de información EEG (sin eventos 'rest')
        configs (list): Lista de configuraciones de pipeline a probar
        
    Returns:
        dict: Resultados de la comparación
    """
    if configs is None:
        configs = ['csp_svm', 'csp_freq_rf', 'asymmetry_rf', 'pca_mlp', 'ensemble', 'enhanced_csp_rf']
    
    results = {}
    
    # Preparar datos combinados
    X_combined = np.vstack([info['X'] for info in eeg_data])
    y_combined = np.concatenate([info['y'] for info in eeg_data])
    
    print(f"Datos combinados: X shape {X_combined.shape}, y shape {y_combined.shape}")
    print(f"Clases: {np.unique(y_combined)}")
    print(f"Distribución de clases: {np.bincount(y_combined.astype(int))}")
    
    print("\nComparando configuraciones de pipelines con validación cruzada...")
    
    for config in configs:
        print(f"\nEvaluando pipeline: '{config}'")
        pipeline = create_binary_pipeline(config)
        
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

def binary_hold_one_out_experiment(eeg_data, pipeline_config='enhanced_csp_rf'):
    """
    Experimento hold-one-out para clasificación binaria.
    
    Args:
        eeg_data (list): Lista de información EEG (sin eventos 'rest')
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
        test_classes = set(np.unique(test_data['y']))
        all_compatible = True
        
        for train_item in train_data:
            train_classes = set(np.unique(train_item['y']))
            if train_classes != test_classes:
                all_compatible = False
                break
        
        if not all_compatible:
            print("  ⚠️ Advertencia: Las clases en los datos de entrenamiento no coinciden con las clases de test")
            print("  ⚠️ Normalizando etiquetas para garantizar compatibilidad")
            
            # Normalización simple para clasificación binaria (0, 1)
            for idx, item in enumerate(train_data):
                unique_labels = np.unique(item['y'])
                if len(unique_labels) > 0:
                    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                    train_data[idx]['y'] = np.array([label_map[y] for y in item['y']])
            
            # Normalizar datos de test también
            unique_labels = np.unique(test_data['y'])
            if len(unique_labels) > 0:
                label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                test_data['y'] = np.array([label_map[y] for y in test_data['y']])
        
        # Combinar datos de entrenamiento
        X_train_combined = np.vstack([item['X'] for item in train_data])
        y_train_combined = np.concatenate([item['y'] for item in train_data])
        
        # Datos de test
        X_test = test_data['X']
        y_test = test_data['y']
        
        print(f"  Datos de entrenamiento: {X_train_combined.shape}, Datos de test: {X_test.shape}")
        print(f"  Clases en train: {np.unique(y_train_combined)}, Clases en test: {np.unique(y_test)}")
        
        # Crear y entrenar pipeline para clasificación binaria
        pipeline = create_binary_pipeline(pipeline_config)
        
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
        
        # Obtener nombres de clases simplificados para clasificación binaria
        class_names = ["Clase 1", "Clase 2"]
        
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
            'class_names': class_names,
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

def plot_binary_confusion_matrices(experiment_results):
    """
    Visualiza las matrices de confusión para cada iteración del experimento binario.
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
        class_names = result.get('class_names', ["Clase 1", "Clase 2"])
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names, 
                    ax=ax, cbar=False)
        
        # Configurar título y etiquetas
        ax.set_title(f"Sujeto {result['subject']} - {result['paradigm']}\nAcc: {result['accuracy']:.3f}")
        ax.set_ylabel('Clase real')
        ax.set_xlabel('Clase predicha')
    
    # Ocultar ejes no utilizados
    if n_eegs < len(axes):
        for j in range(n_eegs, len(axes)):
            axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Matrices de Confusión por Sujeto (Clasificación Binaria)', y=1.02, fontsize=16)
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

def plot_binary_subject_performances(experiment_results):
    """
    Visualiza el rendimiento para cada sujeto en el experimento de clasificación binaria.
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
    
    '# Configurar gráfico
    ax.set_ylabel('Puntuación')
    ax.set_title('Rendimiento por Sujeto (Clasificación Binaria)')
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

def binary_pipeline_comparison_chart(pipeline_results):
    """
    Crea un gráfico de barras comparando diferentes pipelines para clasificación binaria.
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    ax.set_title('Comparación de Pipelines (Clasificación Binaria)')
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

def train_and_save_binary_model(eeg_data, pipeline_config='enhanced_csp_rf', model_name=None):
    """
    Entrena un modelo binario con todos los datos y lo guarda
    
    Args:
        eeg_data (list): Lista de información EEG (sin eventos 'rest')
        pipeline_config (str): Configuración del pipeline
        model_name (str): Nombre base para el modelo
        
    Returns:
        tuple: (pipeline, model_info)
    """
    # Preparar datos combinados
    X_combined = np.vstack([info['X'] for info in eeg_data])
    y_combined = np.concatenate([info['y'] for info in eeg_data])
    
    # Crear pipeline para clasificación binaria
    pipeline = create_binary_pipeline(pipeline_config)
    
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
        model_name = f'eeg_binary_model_{timestamp}'
    
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
        'model_file': model_path,
        'binary_classification': True
    }
    
    # Guardar información del modelo
    info_path = os.path.join(MODELS_DIR, f'{model_name}_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Modelo guardado en: {model_path}")
    print(f"Información guardada en: {info_path}")
    
    return pipeline, model_info

# Script principal para ejecutar el análisis binario completo
def main_binary_classification():
    """
    Script principal para ejecutar análisis de clasificación binaria (sin 'rest')
    """
    # Paso 1: Encontrar el directorio del dataset más reciente
    def find_latest_dataset_dir(base_dir='../models'):
        """Encuentra el directorio de dataset más reciente"""
        dataset_dirs = [d for d in os.listdir(base_dir) if d.startswith('eeg_dataset_')]
        if not dataset_dirs:
            raise ValueError(f"No se encontraron datasets en {base_dir}")
        
        # Ordenar por timestamp (parte del nombre)
        latest_dir = sorted(dataset_dirs)[-1]
        return os.path.join(base_dir, latest_dir)

    # Función para cargar el dataset desde archivos
    def load_dataset_from_files(dataset_dir):
        """
        Carga un dataset guardado previamente
        
        Args:
            dataset_dir (str): Ruta al directorio del dataset
            
        Returns:
            dict: Datos cargados
        """
        print(f"Cargando dataset desde: {dataset_dir}")
        
        # Cargar arrays
        X_all = np.load(os.path.join(dataset_dir, 'X_all.npy'))
        y_all = np.load(os.path.join(dataset_dir, 'y_all.npy'))
        subjects_all = np.load(os.path.join(dataset_dir, 'subjects_all.npy'))
        runs_all = np.load(os.path.join(dataset_dir, 'runs_all.npy'))
        
        # Cargar configuración
        with open(os.path.join(dataset_dir, 'dataset_info.json'), 'r') as f:
            config = json.load(f)
        
        # Reconstruir datos en formato de lista de diccionarios para compatibilidad
        eeg_data = []
        
        for subject_info in config['subjects_data']:
            subject_id = subject_info['subject']
            run_id = subject_info['run']
            
            # Identificar índices para este sujeto/run
            start_idx = subject_info['sample_indices']['start']
            end_idx = subject_info['sample_indices']['end']
            
            # Extraer datos para este sujeto
            X_subject = X_all[start_idx:end_idx]
            y_subject = y_all[start_idx:end_idx]
            
            # Crear diccionario con información
            subject_data = {
                'subject': subject_id,
                'run': run_id,
                'task_type': subject_info['task_type'],
                'paradigm': subject_info['paradigm'],
                'experiment_group': subject_info['experiment_group'],
                'X': X_subject,
                'y': y_subject,
                'event_id': {'rest': 1, 'clase1': 2, 'clase2': 3},
                'class_counts': subject_info['class_counts']
            }
            
            eeg_data.append(subject_data)
        
        print(f"Dataset cargado exitosamente: {len(eeg_data)} registros EEG")
        print(f"Grupo de experimento: {config['dataset_info']['experiment_group']}")
        print(f"Total de muestras: {X_all.shape[0]}, Características: {X_all.shape[1]}")
        
        return {
            'eeg_data': eeg_data,
            'config': config,
            'X_all': X_all,
            'y_all': y_all,
            'subjects_all': subjects_all,
            'runs_all': runs_all
        }

    # Paso 2: Cargar el dataset y filtrar eventos 'rest'
    latest_dataset_dir = find_latest_dataset_dir()
    print(f"Usando el dataset más reciente: {latest_dataset_dir}")

    dataset = load_dataset_from_files(latest_dataset_dir)
    eeg_data = dataset['eeg_data']

    # Filtrar eventos de reposo (rest)
    print("\n=== Filtrando eventos 'rest' para clasificación binaria ===")
    binary_eeg_data = filter_rest_events(eeg_data)

    # Paso 3: Comparar diferentes pipelines de clasificación binaria
    print("\n=== Comparación de pipelines para clasificación binaria ===")
    pipeline_configs = ['csp_svm', 'csp_freq_rf', 'asymmetry_rf', 'pca_mlp', 'ensemble', 'enhanced_csp_rf']
    binary_comparison_results = compare_binary_pipelines(binary_eeg_data, configs=pipeline_configs)

    # Paso 4: Visualizar comparación de pipelines
    fig = binary_pipeline_comparison_chart(binary_comparison_results)
    plt.savefig(os.path.join(MODELS_DIR, 'binary_pipeline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Paso 5: Evaluar el mejor pipeline con hold-one-out cross-validation
    best_binary_pipeline = binary_comparison_results['best_config']
    print(f"\n=== Evaluación del mejor pipeline ({best_binary_pipeline}) con hold-one-out ===")
    binary_holdout_results = binary_hold_one_out_experiment(binary_eeg_data, pipeline_config=best_binary_pipeline)

    # Paso 6: Visualizar resultados por sujeto
    fig_perf = plot_binary_subject_performances(binary_holdout_results)
    plt.savefig(os.path.join(MODELS_DIR, 'binary_subject_performances.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_perf)

    fig_cm = plot_binary_confusion_matrices(binary_holdout_results)
    plt.savefig(os.path.join(MODELS_DIR, 'binary_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_cm)

    # Paso 7: Entrenar y guardar el modelo final
    print("\n=== Entrenamiento del modelo final para clasificación binaria ===")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'eeg_binary_model_{dataset["config"]["dataset_info"]["experiment_group"]}_{timestamp}'
    final_model, model_info = train_and_save_binary_model(binary_eeg_data, pipeline_config=best_binary_pipeline, model_name=model_name)

    print("\n=== Resumen de la evaluación para clasificación binaria ===")
    print(f"Mejor pipeline: {best_binary_pipeline}")
    print(f"Accuracy CV: {binary_comparison_results['results'][best_binary_pipeline]['accuracy']:.4f} ± {binary_comparison_results['results'][best_binary_pipeline]['accuracy_std']:.4f}")
    print(f"Hold-one-out Accuracy: {binary_holdout_results['avg_accuracy']:.4f}")
    print(f"Modelo final guardado como: {model_info['model_file']}")
    print(f"Gráficos de resultados guardados en el directorio '{MODELS_DIR}'")
    
    return {
        'binary_eeg_data': binary_eeg_data,
        'comparison_results': binary_comparison_results,
        'holdout_results': binary_holdout_results,
        'model_info': model_info,
        'final_model': final_model
    }

