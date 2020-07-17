# pyebm - A toolbox for Event Based Models

If you would like to use the algorithms, install the toolbox using pip:

- `pip install pyebm`

The examples to use the toolbox can be found in the jupyter notebook at: [link](https://github.com/88vikram/pyebm/tree/master/notebooks)

The event-based model (EBM) for data-driven disease progression modeling estimates the sequence in which biomarkers for a disease become abnormal. This helps in understanding the dynamics of disease progression and facilitates early diagnosis by staging patients on a disease progression timeline. A more accurate and scalable EBM algorithm (Discriminative EBM) was introduced in [1,2]. DEBM was also one of the winners of the TADPOLE prediction challenge (TADPOLE). For more details of the prediction challenge, read Razvan Marinescu et al. [5].

Call debm.fit to find the central ordering in a few biomarkers using method [1,2]

Call ebm.fit to find the central ordering in a few biomarkers using method [3]

EBM and its variants typically consists of 2 steps. 

Step 1: Mixture Model to figure out biomarker distributions in Normal and Abnormal classes

Step 2: Estimating a mean ordering of biomarkers.

This toolbox supports 3 different Gaussian mixture models.
1. Algorithm proposed in [2] by Vikram Venkatraghavan et. al. NeuroImage (2019) (default, and shown to be more stable in [2])
2. Algorithm proposed in [1] by Vikram Venkatraghavan et. al. IPMI (2017)
3. Algorithm proposed in [4] by Alexandra Young et. al. Brain (2014)

## Required Libraries

Python 3.5+ (or higher), numpy 1.16+, pandas 0.24+, sklearn 0.20+, scipy 1.2+, seaborn 0.9+, statsmodels 0.9+

## Explanation of Inputs:

### DataIn:
 String to the CSV File where the data is stored. This can also be a Pandas dataframe with necessary data. The CSV file or the dataframe must contain the following fields: PTID (Patient ID), Diagnosis (Clinical Label), Biomarkers, Confounding Factors, EXAMDATE (Date of Examination). See Data_7.csv for example.
### (optional) MethodOptions:
Named Tuple with any or all of the following fields:

*   MixtureModel - Choose the mixture model algorithm (Options: 'GMMvv1'[1],'GMMvv2'(default)[2], 'GMMay'[4]) 
*   Bootstrap - Number of iterations in the bootstrapping [default - Turned Off].
*   PatientStaging - Choose the patient staging algorithm, with a two element list consisting of ['exp'/'ml','p'/'l']. The first element in the list chooses 'ml' for most likely stage[1,3,4] or 'exp' for expected stage[2]. The second element in the list chooses 'l' for likelihood[1,3,4] or 'p' for posterior probability[2]. (['exp','p'] was shown to be the preferred choice in [2])
*   (Only in ebm.fit) NStartpoints, Niterations and N_MCMC are algorithm specific parameters for EBM method.

### (optional) VerboseOptions:
Named Tuple with any or all of the following fields:

*   Distributions - plots biomarker distributions [default - Turned Off]
*   Ordering - plots the central ordering as a positional variance diagram [default - Turned Off].
*   PlotOrder - positional variance diagram has mean positions along the main diagonal [default - Turned Off]. This is used only when Ordering is Turned on.
*   WriteBootstrapData - String which specifies the location and name of the files to save the data used in different bootstrap iterations. [default - Turned Off]
*   PatientStaging - plots the patient stages of subjects in different classes. [default - Turned Off]

### (optional) Factors:
Confounding Factors used for correcting the biomarkers. By Default, it is Age, Sex, ICV (intra-cranial volume)

### (optional) Labels:
Clinical list of labels in the dataset. By Default, it is CN, MCI, AD.

### (optional) DataTest:
If given, DataTest will be used as a test-set to evaluate the disease progression model obtained using DataIn.

## Explanation of Outputs:

### ModelOutput:
A stucture with the following fields:
* BiomarkerList - List of Biomarkers used in EBM
* BiomarkerParameters - Mixture Model parameters for the biomarkers
* CentralOrderings - Central Ordering in different boostrap iterations. When bootstrapping is turned off, this gives the central ordering for the entire dataset.
* MeanCentralOrdering - Mean Central Ordering among different bootstrap iterations. When bootstrapping is turned off, this is the same as CentralOrderings.
* EventCenters - Event centers which determins how close the events are to each other.

### SubjTrainAll:
A list where each element is a pandas dataframe corresponding to different bootstrap iterations. 
Each dataframe consists of the the following fields :
* PTID - patient identifiers used in training
* Ordering - Subject-wise orderings of the subjects used for training the model
* Weights - Probabilistic weights for the each position in the subject-wise ordering
* Stages - Staging of each subject in the training dataset.

### SubjTestAll:
A list where each element is a pandas dataframe corresponding to different bootstrap iterations. 
Each dataframe consists of the the following fields:
* PTID - patient identifiers used in testing
* Ordering - Subject-wise orderings of the subjects used for testing the model
* Weights - Probabilistic weights for the each position in the subject-wise ordering
* Stages - Staging of each subject in the testing dataset.

### References:

[1] Venkatraghavan V., et al., ‘[A Discriminative Event Based Model for Alzheimer's Disease Progression Modeling](https://arxiv.org/abs/1702.06408)’, IPMI (2017).

[2] Venkatraghavan V., et al., ‘[Disease Progression Timeline Estimation for Alzheimer's Disease using Discriminative Event Based Modeling](https://doi.org/10.1016/j.neuroimage.2018.11.024)’, NeuroImage 186, 518 - 532 (2019).

[3] Fonteijn, H.M., et al., ‘[An event-based model for disease progression and its application in familial Alzheimer's disease and Huntington's disease](https://doi.org/10.1016/j.neuroimage.2012.01.062)’, NeuroImage 60(3), 1880–1889 (2012).

[4] Young, A.L., et al.: ‘[A data-driven model of biomarker changes in sporadic Alzheimer’s disease](https://doi.org/10.1093/brain/awu176)’, Brain 137(9), 2564–2577 (2014).

[5] Marinescu, R. et al.: ‘[The Alzheimer's Disease Prediction Of Longitudinal Evolution (TADPOLE) Challenge: Results after 1 Year Follow-up](https://arxiv.org/abs/2002.03419). 

### Contact:

v.venkatraghavan@erasmusmc.nl
