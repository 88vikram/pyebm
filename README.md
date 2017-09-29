# pyEBM - A toolbox for Event Based Models

The event-based model (EBM) for data-driven disease progression modeling estimates the sequence in which biomarkers for a disease become abnormal. This helps in understanding the dynamics of disease progression and facilitates early diagnosis by staging patients on a disease progression timeline. A more accurate and scalable EBM algorithm (Discriminative EBM) was introduced in [2]. 

Call EBM.Control to find the central ordering in a few biomarkers using method [1]
Call DEBM.Control to find the central ordering in a few biomarkers using method [2]

EBM and its variants typically consists of 2 steps. 

Step 1: Mixture Model to figure out biomarker distributions in Normal and Abnormal classes

Step 2: Estimating a mean ordering of biomarkers.

This toolbox supports 3 different Gaussian mixture models.
1. Algorithm proposed in [3] by Alexandra Young et. al.
2. Algorithm proposed in [2] by Vikram Venkatraghavan et. al.
3. Algorithm proposed in [4] by Vikram Venkatraghavan et. al.

## Required Libraries

Python 2.7.x, numpy 1.11 or higher, pandas 0.19, sklearn 0.18, scipy 0.18, seaborn 0.7, statsmodels 0.6

## Explanation of Inputs:

### DataIn:
 String to the CSV File where the data is stored. This can also be a Pandas dataframe with necessary data. The CSV file or the dataframe must contain the following fields: PTID (Patient ID), Diagnosis (Clinical Label), Biomarkers, Confounding Factors, EXAMDATE (Date of Examination). See ADNI_7.csv for example.
### (optional) MethodOptions:
Named Tuple with any or all of the following fields:

*   MixtureModel - Choose the mixture model algorithm (Options: 'vv1'[2],'vv2'(default)[4], 'ay'[3])
*   Bootstrap - Number of iterations in the bootstrapping [default - Turned Off].
*   Staging - Choose the patient staging algorithm (Options 'ml' for most likely stage[1,2,3],'exp' for expected stage[4])

### (optional) VerboseOptions:
Named Tuple with any or all of the following fields:

*   Distributions - plots biomarker distributions [default - Turned Off]
*   Ordering - plots the central ordering as a positional variance diagram [default - Turned Off].
*   PlotOrder - positional variance diagram has mean positions along the main diagonal [default - Turned Off]. This is used only when Ordering is Turned on.
*   WriteBootstrapData - String which specifies the location and name of the files to save the data used in different bootstrap iterations. [default - Turned Off]

### (optional) Factors:
Confounding Factors used for correcting the biomarkers. By Default, it is Age, Sex, ICV (intra-cranial volume)

### (optional) Labels:
Clinical list of labels in the dataset. By Default, it is CN, MCI, AD.

### References:

[1] Fonteijn, H.M., et. al., ‘[An event-based model for disease progression and its application in familial Alzheimer's disease and Huntington's disease](https://doi.org/10.1016/j.neuroimage.2012.01.062)’, NeuroImage 60(3), 1880–1889 (2012).

[2] Venkatraghavan V., et. al., ‘[A Discriminative Event Based Model for Alzheimer's Disease Progression Modeling](https://arxiv.org/abs/1702.06408)’, IPMI (2017).

[3] Young, A.L., et. al.: ‘[A data-driven model of biomarker changes in sporadic Alzheimer’s disease](https://doi.org/10.1093/brain/awu176)’, Brain 137(9), 2564–2577 (2014).

[4] Venkatraghavan V., et. al., ‘A Discriminative Event Based Model for Alzheimer's Disease Progression Modeling’, Manuscript under preparation.
