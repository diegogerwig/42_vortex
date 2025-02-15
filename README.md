# 🧠 TOTAL PERSPECTIVE VORTEX 

This subject aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms.
Using a subject’s EEG reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B in a t0 to tn timeframe.


## 🎯 Goals

- Process EEG datas (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline object from scikit-learn
- Classify a data stream in "real time"


## 📊 Dataset 

EEG Motor Movement/Imagery Dataset

[Link of dataset](https://physionet.org/content/eegmmidb/1.0.0/)


## 💫 Dimensionality reduction algorithm 

### CSP (Common Spatial Patterns)

Common Spatial Patterns (CSP) algorithm is employed to extract discriminative spatial patterns from EEG data, enhancing the classification of different states.

The goal of the CSP algorithm is to find spatial filters that maximize the variance of one class of signals while minimizing the variance of another class. This is achieved by finding a transformation matrix that, when applied to the raw EEG signals, results in new signals (features) that have the highest variance for one class and the lowest variance for the other.

### ICA (Independent Component Analysis)

Independent Component Analysis (ICA) is a statistical and computational technique used to separate a multivariate signal into its constituent independent subcomponents. It's particularly useful when the observed signals are a linear mixture of various independent sources, making it a powerful tool in signal processing, neuroscience, image processing, and other fields.