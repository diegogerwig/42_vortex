import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
from mne.decoding import SPoC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import linalg
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import concurrent.futures
import json
import hashlib
import time
# Configuración global
matplotlib.use('TkAgg')
mne.set_log_level("CRITICAL")
DATA_SAMPLE_PATH = "../data/"

# Enumeraciones
SUBJECTS = [1,2,3,4,5,6]
MODES = ["train", "predict", "all"]
TRANSFORMERS = ["FAST_CSP", "CSP", "SPoC"]

EXPERIMENTS = {
    "hands_vs_feet__action": ['do/hands', 'do/feet'],
    "hands_vs_feet__imagery": ['imagine/hands', 'imagine/feet'],
    "imagery_vs_action__hands": ['do/hands', 'imagine/hands'],
    "imagery_vs_action__feets": ['do/feet', 'imagine/feet'],
}

EXPERIMENTS_IDS = {
    'action': [5, 9, 13],
    'imagery': [6, 10, 14]
}

# Implementación de CSP
class CSP(BaseEstimator, TransformerMixin):
    """
    CSP implementation based on MNE implementation
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.n_classes = None
        self.mean = None
        self.std = None

    def calculate_cov_(self, X, y):
        """Calculate the covariance matrices for each class."""
        _, n_channels, _ = X.shape
        covs = []

        for l in self.n_classes:
            lX = X[np.where(y == l)]
            lX = lX.transpose([1, 0, 2])
            lX = lX.reshape(n_channels, -1)
            covs.append(np.cov(lX))

        return np.asarray(covs)

    def calculate_eig_(self, covs):
        """Calculate eigenvalues and eigenvectors for pairwise combinations of covariance matrices."""
        eigenvalues, eigenvectors = [], []

        for idx, cov in enumerate(covs):
            for iidx, compCov in enumerate(covs):
                if idx < iidx:
                    eigVals, eigVects = linalg.eig(cov, cov + compCov)
                    sorted_indices = np.argsort(np.abs(eigVals - 0.5))[::-1]
                    eigenvalues.append(eigVals[sorted_indices])
                    eigenvectors.append(eigVects[:, sorted_indices])

        return eigenvalues, eigenvectors

    def pick_filters(self, eigenvectors):
        """Select CSP filters based on the sorted eigenvectors."""
        filters = []

        for EigVects in eigenvectors:
            if filters == []:
                filters = EigVects[:, :self.n_components]
            else:
                filters = np.concatenate([filters, EigVects[:, :self.n_components]], axis=1)

        self.filters = filters.T

    def fit(self, X, y):
        self.n_classes = np.unique(y)

        if len(self.n_classes) < 2:
            raise ValueError("n_classes must be >= 2")

        covs = self.calculate_cov_(X, y)
        eigenvalues, eigenvectors = self.calculate_eig_(covs)
        self.pick_filters(eigenvectors)

        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)
        X -= self.mean
        X /= self.std
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Funciones para procesamiento de datos
def fetch_data(subjNumber):
    run_execution = [5, 9, 13]  # (open and close both fists or both feet)
    run_imagery =  [6, 10, 14]  # (imagine opening and closing both fists or both feet)

    raw_files = []

    for i, j in zip(run_execution, run_imagery):
        try:
            # Datos de ejecución
            raw_files_execution = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                                eegbci.load_data(subjNumber, i, DATA_SAMPLE_PATH)]
            raw_execution = concatenate_raws(raw_files_execution)

            # Datos de imaginación
            raw_files_imagery = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                                eegbci.load_data(subjNumber, j, DATA_SAMPLE_PATH)]
            raw_imagery = concatenate_raws(raw_files_imagery)

            # Anotaciones para ejecución
            events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1, T1=2, T2=3))
            mapping = {1: 'rest', 2: 'do/feet', 3: 'do/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
                orig_time=raw_execution.info['meas_date'])
            raw_execution.set_annotations(annot_from_events)

            # Anotaciones para imaginación
            events, _ = mne.events_from_annotations(raw_imagery, event_id=dict(T0=1, T1=2, T2=3))
            mapping = {1: 'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw_imagery.info['sfreq'],
                orig_time=raw_imagery.info['meas_date'])
            raw_imagery.set_annotations(annot_from_events)

            raw_files.append(raw_execution)
            raw_files.append(raw_imagery)
        except Exception as e:
            print(f"Error al procesar subject {subjNumber}, run {i}/{j}: {str(e)}")
            # Si no hay datos, intenta descargarlos
            try:
                print(f"Intentando descargar datos para subject {subjNumber}...")
                eegbci.load_data(subjNumber, i, DATA_SAMPLE_PATH, force_update=True)
                eegbci.load_data(subjNumber, j, DATA_SAMPLE_PATH, force_update=True)
                print(f"Datos descargados. Por favor, vuelve a ejecutar el script.")
            except Exception as e2:
                print(f"Error al descargar datos: {str(e2)}")
            
    if not raw_files:
        raise ValueError(f"No se pudieron obtener datos para el subject {subjNumber}")
            
    raw = concatenate_raws(raw_files)

    event, event_dict = events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    return [raw, event, event_dict, picks]

def prepare_data(raw, plotIt=False):
    eegbci.standardize(raw)
    montage = make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')

    if plotIt:
        montage = raw.get_montage()
        p = montage.plot()
        p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})

    return raw

def filter_data(raw, plotIt=False):
    raw.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    if plotIt:
        p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        plt.show()
    return raw

def filter_eye_artifacts(raw, picks, method, plotIt=False):
    raw_corrected = raw.copy()
    n_components = 20

    ica = ICA(n_components=n_components, method=method, fit_params=None, random_state=97)
    ica.fit(raw_corrected, picks=picks)

    [eog_indicies, scores] = ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
    ica.exclude.extend(eog_indicies)
    ica.apply(raw_corrected, n_pca_components=n_components, exclude=ica.exclude)

    if plotIt:
        ica.plot_components()
        ica.plot_scores(scores, exclude=eog_indicies)
        plt.show()

    return raw_corrected

def fetch_events(data_filtered, tmin=-1., tmax=4.):
    events, event_ids = events_from_annotations(data_filtered)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs, picks

def pre_process_data(subjectID, experiments):
    [raw, event, event_dict, picks] = fetch_data(subjectID)
    raw_prepared = prepare_data(raw)
    raw_filtered = filter_data(raw_prepared)
    labels, epochs, picks = fetch_events(raw_filtered)

    # Extraer solo las épocas correspondientes a las etiquetas seleccionadas
    selected_epochs = epochs[experiments]
    X = selected_epochs.get_data()
    y = selected_epochs.events[:, -1] - 1

    return [X, y, epochs]

# Funciones para entrenamiento
def pipeline_creation(X, y, transformer1, transformer2=None, transformer3=None):
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = LDA(solver='lsqr', shrinkage='auto')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    final_result = []

    pipeline1 = make_pipeline(transformer1, lda)
    scores1 = cross_val_score(pipeline1, X, y, cv=cv, n_jobs=1)
    final_result.append(('LDA ', pipeline1, scores1))
    
    if transformer2:
        pipeline2 = make_pipeline(transformer2, log_reg)
        scores2 = cross_val_score(pipeline2, X, y, cv=cv, n_jobs=1)
        final_result.append(('LOGR', pipeline2, scores2))
    
    if transformer3:
        pipeline3 = make_pipeline(transformer3, rfc)
        scores3 = cross_val_score(pipeline3, X, y, cv=cv, n_jobs=1)
        final_result.append(('RFC', pipeline3, scores3))

    return final_result

def save_pipeline(pipe, epochs_data_train, labels, subjectID, experiment_name):
    pipe = pipe.fit(epochs_data_train, labels)
    fileName = f"../data/models/model_subject_{subjectID}_{experiment_name}.joblib"
    dump(pipe, fileName)
    return

def train_data(X, y, transformer="CSP", run_all_pipelines=False):
    if transformer == "CSP":
        from mne.decoding import CSP as MNE_CSP
        # using CSP transformers from MNE
        csp1 = MNE_CSP()

        if run_all_pipelines:
            csp2 = MNE_CSP()
            csp3 = MNE_CSP()
            return pipeline_creation(X, y, csp1, csp2, csp3)
        return pipeline_creation(X, y, csp1)

    elif transformer == "FAST_CSP":
        # using custom CSP transformers
        csp1 = CSP()

        if run_all_pipelines:
            csp2 = CSP()
            csp3 = CSP()
            return pipeline_creation(X, y, csp1, csp2, csp3)
        return pipeline_creation(X, y, csp1)

    elif transformer == "SPoC":
        # using Spoc transformers
        Spoc1 = SPoC(n_components=15, reg='oas', log=True, rank='full')

        if run_all_pipelines:
            Spoc2 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            Spoc3 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            return pipeline_creation(X, y, Spoc1, Spoc2, Spoc3)
        return pipeline_creation(X, y, Spoc1)
    else:
        raise ValueError(f"Unknown transformer, please enter valid one.")

# Funciones para predicción
def predict(X, y, subjectId, experiment_name, log=False):
    PREDICT_MODEL = f"../data/models/model_subject_{subjectId}_{experiment_name}.joblib"
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    scores = []
    if log:
        print("epoch_nb =  [prediction]    [truth]    equal?")
        print("---------------------------------------------")
    for n in range(X.shape[0]):
        pred = clf.predict(X[n:n + 1, :, :])[0]
        truth = y[n:n + 1][0]
        if log:
            print(f"epoch_{n:2} =      [{pred}]           [{truth}]      {'' if pred == truth else False}")
        scores.append(1 - np.abs(pred - y[n:n + 1][0]))
    return np.mean(scores).round(3)

# Funciones para manejo de argumentos
def get_config():
    """Retorna la configuración por defecto"""
    config = {
        'SUBJECTS': [1],
        'MODE': 'all',
        'TRANSFORMER': 'FAST_CSP',
        'EXPERIMENT': 'imagery_vs_action__feets'
    }
    return config

# Funciones principales
def hash_list_secure(my_list):
    sorted_tuple = tuple(sorted(my_list))
    hash_object = hashlib.sha256(str(sorted_tuple).encode())
    return hash_object.hexdigest()

def process_subject(subjectID, args, isSingleSubject=False):
    start_time_inner = time.time()
    [X, y, epochs] = pre_process_data(subjectID, EXPERIMENTS[args['EXPERIMENT']])

    result_inner = [0, 0]
    output = []
    output.append(f"----------------------------------------------[Subject {subjectID}]")
    stats = {
        'subject_id': subjectID,
        'pipelines': [],
        'cross_val_score': 0,
        'accuracy': 0
    }
    
    if args['MODE'] == "train" or args['MODE'] == "all":
        pipelines = train_data(X=X, y=y, transformer=args['TRANSFORMER'], run_all_pipelines=True)
        best_pipeline = {'cross_val_score': -1}

        for pipel in pipelines:
            cross_val_score = pipel[2].mean()
            pipeline_name = pipel[0]
            pipeline = pipel[1]
            output.append(f":--- [S{subjectID}] {pipeline_name} cross_val_score : {cross_val_score.round(2)}")

            if cross_val_score > best_pipeline['cross_val_score']:
                best_pipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            stats['pipelines'].append((pipeline_name, cross_val_score))
        
        save_pipeline(best_pipeline['pipeline'], X, y, subjectID, args['EXPERIMENT'])
        result_inner[0] = best_pipeline['cross_val_score']
        stats['cross_val_score'] = result_inner[0]

    if args['MODE'] == "predict" or args['MODE'] == "all":
        prediction_result = predict(X, y, subjectID, args['EXPERIMENT'], isSingleSubject)
        output.append(
            f":--- [S{subjectID}] Prediction accurracy: {'{:.2%}'.format(prediction_result).rstrip('0').rstrip('.')}")
        result_inner[1] = prediction_result
        stats['accuracy'] = result_inner[1]

    end_time_inner = time.time()
    time_cost_inner = end_time_inner - start_time_inner
    stats['time_cost'] = time_cost_inner
    print(*output, sep="\n")
    print(f":--- [S{subjectID}] time cost: {round(stats['time_cost'], 2)} seconds")
    return stats

def calculate_all_means(cross_val_scores, accuracy_scores, final_stats):
    print("\n----------------------------[Mean Scores for all subjects]----------------------------")
    if len(cross_val_scores) > 1:
        print(f":--- Mean cross_val : {np.mean(cross_val_scores).round(2)}")
        final_stats['mean_cross_val_score'] = np.mean(cross_val_scores)
    if len(accuracy_scores) > 1:
        print(f":--- Mean accuracy  : {np.mean(accuracy_scores).round(2)}")
        final_stats['mean_accuracy'] = np.mean(accuracy_scores)

def dump_result_to_json(final_stats, args):
    import os
    
    # Crear directorio si no existe
    os.makedirs("../data/results", exist_ok=True)
    
    results_filename = \
        f"../data/results/results-{args['MODE']}-{args['EXPERIMENT']}-{time.time()}-{args['TRANSFORMER']}-{final_stats['subjects_hash']}.json"

    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=4)

    print(f"Los resultados del entrenamiento/predicción se han guardado en:\n[{results_filename}]")

def ensure_directories():
    """Asegura que los directorios necesarios existan"""
    import os
    
    # Directorios necesarios
    dirs = [
        "../data/models",
        "../data/results",
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def main():
    start_time = time.time()
    
    # Asegurar que los directorios necesarios existan
    ensure_directories()
    
    # Usar configuración predeterminada
    args = get_config()
    print("Configuración:", args)

    print(
        f"Experiment in study: ({EXPERIMENTS[args['EXPERIMENT']][0]}) <--VS--> ({EXPERIMENTS[args['EXPERIMENT']][1]})")
    CALC_MEAN_FOR_ALL = True if len(args['SUBJECTS']) > 1 else False

    cross_val_scores = []
    accuracy_scores = []
    final_stats = {
        'subjects_hash': "all" if len(args['SUBJECTS']) == 109 else ''.join(map(str, args['SUBJECTS'])),
        'config': args,
        'events': EXPERIMENTS[args['EXPERIMENT']],
        'subjects': [],
        'time_unit': "seconds"
    }

    for subjectID in args['SUBJECTS']:
        result = process_subject(subjectID, args, isSingleSubject=not CALC_MEAN_FOR_ALL)
        if args['MODE'] == "train" or args['MODE'] == "all":
            cross_val_scores.append(result['cross_val_score'])

        if args['MODE'] == "predict" or args['MODE'] == "all":
            accuracy_scores.append(result['accuracy'])

        final_stats['subjects'].append(result)

    if CALC_MEAN_FOR_ALL:
        calculate_all_means(cross_val_scores, accuracy_scores, final_stats)

    final_stats['time_cost'] = time.time() - start_time
    print(f":--- Time Elapsed for all : {round(final_stats['time_cost'], 2)}")

    dump_result_to_json(final_stats, args)

if __name__ == "__main__":
    main()