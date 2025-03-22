# Importar las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import random
from datetime import datetime
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Importar funciones de los módulos
from predict import load_model, load_specific_subject, predict_eeg, visualize_predictions_over_time
from preprocessing import preprocess_data

# Parámetros de preprocesamiento
PREPROCESSING_PARAMS = {
    'low_cutoff': 8,      
    'high_cutoff': 40,    
    'apply_notch': True,  
    'tmin': -1.0,         
    'tmax': 4.0,          
}

# Configuración de los grupos de experimentos
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

def predict_new_subjects(num_subjects=6, model_path=None, save_results=True):
    """
    Realiza predicciones sobre nuevos sujetos utilizando un experimento aleatorio para cada uno.
    
    Args:
        num_subjects (int): Número de sujetos a predecir
        model_path (str): Ruta al modelo a utilizar
        save_results (bool): Si se guardan los resultados en un archivo
        
    Returns:
        dict: Resultados detallados y resumen
    """
    # Cargar el modelo
    print(f"Cargando modelo para predicción...")
    pipeline, model_info = load_model(model_path)
    
    # Seleccionar sujetos aleatorios
    # Evitar los sujetos que ya fueron utilizados para entrenamiento
    used_subjects = []
    if model_info and 'subjects' in model_info:
        used_subjects = model_info['subjects']
    
    # Todos los posibles sujetos
    all_subjects = list(range(1, 110))  # Sujetos 1-109
    
    # Filtrar sujetos ya utilizados
    available_subjects = [s for s in all_subjects if s not in used_subjects]
    
    # Seleccionar sujetos aleatorios
    if len(available_subjects) < num_subjects:
        print(f"Advertencia: Solo hay {len(available_subjects)} sujetos disponibles. Se usarán todos.")
        selected_subjects = available_subjects
    else:
        selected_subjects = random.sample(available_subjects, num_subjects)
    
    print(f"Sujetos seleccionados para predicción: {selected_subjects}")
    
    # Mantener resultados
    results = {}
    summary_data = []
    
    # Para cada sujeto, seleccionar un grupo de experimento aleatorio
    for i, subject_id in enumerate(selected_subjects):
        print(f"\nProcesando sujeto {i+1}/{len(selected_subjects)} (ID={subject_id}):")
        
        # Seleccionar un grupo de experimento aleatorio
        experiment_group = random.choice(list(EXPERIMENT_GROUPS.keys()))
        group_config = EXPERIMENT_GROUPS[experiment_group]
        
        # Seleccionar un run aleatorio de este grupo
        run_id = random.choice(group_config['runs'])
        
        print(f"  Experimento: {group_config['description']}")
        print(f"  Run seleccionado: {run_id}")
        
        try:
            # Cargar datos del sujeto
            raw_data, task_type, paradigm = load_specific_subject(subject_id, run_id)
            
            # Preprocesar datos
            print(f"  Preprocesando datos...")
            X, y, epochs, event_id = preprocess_data(
                raw_data,
                PREPROCESSING_PARAMS['low_cutoff'],
                PREPROCESSING_PARAMS['high_cutoff'],
                PREPROCESSING_PARAMS['apply_notch'],
                PREPROCESSING_PARAMS['tmin'],
                PREPROCESSING_PARAMS['tmax']
            )
            
            # Realizar predicción
            print(f"  Realizando predicción...")
            start_time = datetime.now()
            y_pred = pipeline.predict(X)
            predict_time = (datetime.now() - start_time).total_seconds()
            
            # Calcular métricas
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            cm = confusion_matrix(y, y_pred)
            
            print(f"  Resultados: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            print(f"  Tiempo de predicción: {predict_time:.2f} segundos")
            
            # Mapear clases
            id_to_class = {v: k for k, v in event_id.items()}
            class_names = [id_to_class.get(c, f"Clase {c}") for c in sorted(np.unique(y))]
            
            # Generar reporte
            report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
            
            # Guardar resultados
            subject_result = {
                'subject_id': subject_id,
                'experiment_group': experiment_group,
                'run_id': run_id,
                'task_type': task_type,
                'paradigm': paradigm,
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'predict_time': predict_time,
                'y_true': y.tolist(),
                'y_pred': y_pred.tolist(),
                'class_mapping': id_to_class
            }
            
            results[f'subject_{subject_id}'] = subject_result
            
            # Agregar a los datos de resumen
            summary_data.append({
                'subject_id': subject_id,
                'experiment_group': experiment_group,
                'run_id': run_id,
                'accuracy': accuracy,
                'f1_score': f1,
                'task_type': task_type,
                'paradigm': paradigm
            })
            
        except Exception as e:
            print(f"  Error procesando sujeto {subject_id}: {str(e)}")
            results[f'subject_{subject_id}'] = {'error': str(e)}
    
    # Crear DataFrame de resumen
    summary_df = pd.DataFrame(summary_data)
    
    # Calcular promedio general
    avg_accuracy = summary_df['accuracy'].mean()
    avg_f1 = summary_df['f1_score'].mean()
    
    print("\n===== RESUMEN DE PREDICCIONES =====")
    print(f"Total de sujetos evaluados: {len(summary_df)}")
    print(f"Accuracy promedio: {avg_accuracy:.4f}")
    print(f"F1 Score promedio: {avg_f1:.4f}")
    print("\nResultados por sujeto:")
    
    # Mostrar tabla de resultados individuales
    summary_table = summary_df[['subject_id', 'experiment_group', 'run_id', 'accuracy', 'f1_score']]
    print(summary_table)
    
    # Guardar resultados si se solicita
    if save_results:
        # Crear directorio de resultados si no existe
        results_dir = os.path.join('..', 'models', 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Nombre de archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'prediction_results_{timestamp}.json')
        
        # Guardar resultados completos
        with open(results_file, 'w') as f:
            # Convertir clases numpy a tipos nativos de Python para serialización JSON
            serializable_results = {
                'timestamp': timestamp,
                'num_subjects': len(selected_subjects),
                'selected_subjects': selected_subjects,
                'avg_accuracy': float(avg_accuracy),
                'avg_f1_score': float(avg_f1),
                'results': results,
                'model_info': model_info
            }
            json.dump(serializable_results, f, indent=4)
        
        # Guardar resumen como CSV
        summary_file = os.path.join(results_dir, f'prediction_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nResultados guardados en:")
        print(f"- Detallados: {results_file}")
        print(f"- Resumen: {summary_file}")
    
    # Visualizar resultados en un gráfico
    plt.figure(figsize=(10, 6))
    
    # Ordenar por accuracy
    summary_df_sorted = summary_df.sort_values('accuracy', ascending=False)
    
    # Crear gráfico de barras
    sns.barplot(x='subject_id', y='accuracy', hue='experiment_group', data=summary_df_sorted)
    
    # Agregar línea para el promedio
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', label=f'Promedio: {avg_accuracy:.4f}')
    
    # Ajustar gráfico
    plt.title('Accuracy por Sujeto')
    plt.xlabel('ID del Sujeto')
    plt.ylabel('Accuracy')
    plt.legend(title='Grupo de Experimento')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar gráfico si se guardan resultados
    if save_results:
        plt_file = os.path.join(results_dir, f'prediction_accuracy_{timestamp}.png')
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        print(f"- Gráfico: {plt_file}")
    
    plt.show()
    
    return {
        'results': results,
        'summary': summary_df,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1
    }
