# *=========================================================================
# *
# *  Copyright Erasmus MC Rotterdam and contributors
# *
# *  Licensed under the GNU GENERAL PUBLIC LICENSE Version 3;
# *  you may not use this file except in compliance with the License.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *=========================================================================*/

from __future__ import print_function
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import datetime
import time
    
def pdReadData(str_data,flag_JointFit=False,Labels=['CN','MCI','AD']):

    if type(str_data) is str:
        Data = pd.read_csv(str_data)
    elif len(str_data)==0:
        Data=[]; UniqueSubjIDs=[]
        return Data,UniqueSubjIDs
    else:    
        Data = str_data
    UniqueSubjIDs = pd.Series.unique(Data['PTID'])
    labels = np.zeros(len(Data['Diagnosis']),dtype=int)
    
    for i in range(len(Labels)):
        idx_label=Data['Diagnosis'] == Labels[i]; 
        idx_label=np.where(idx_label.values);
        idx_label = idx_label[0];
        if i==0:
            label_i=1;
        elif i==len(Labels)-1:
            label_i=3;
        else:
            label_i=2;
        labels[idx_label]=label_i;
    idx_selectedsubjects = np.logical_not(labels == 0)
    labels_selected=labels[np.logical_not(labels == 0)]
    Data=Data[idx_selectedsubjects]
    Data=Data.drop('Diagnosis',axis=1)
    Data=Data.assign(Diagnosis=pd.Series(labels_selected,Data.index))
    
    return Data,UniqueSubjIDs

def CorrectConfounders(DataTrain,DataTest,Factors=['Age','Sex','ICV'],flag_correct=1,Groups=[]):

    flag_test=1;
    droplist = ['PTID','Diagnosis','EXAMDATE']
    GroupValues = []
    if flag_correct==0 or len(Factors)==0:
        DataTrain=DataTrain.drop(Factors,axis=1)
        DataBiomarkers=DataTrain.copy()
        H = list(DataBiomarkers)
        for j in droplist:
            if any(j in f for f in H):
                DataBiomarkers=DataBiomarkers.drop(j,axis=1)
        BiomarkersList=list(DataBiomarkers)
        if len(DataTest)>0:
            DataTest = DataTest.drop(Factors,axis=1)
    else:
        ## Change categorical value to numerical value
        if len(DataTest)==0:
            flag_test=0;
            DataTest=DataTrain.copy()

        if any('Sex' in f for f in Factors):
            count=-1;
            for Data in [DataTrain,DataTest]:
                count=count+1;
                sex = np.zeros(len(Data['Diagnosis']),dtype=int)
                idx_male=Data['Sex'] == 'Male'; idx_male=np.where(idx_male); idx_male = idx_male[0];
                idx_female=Data['Sex'] == 'Female'; idx_female=np.where(idx_female); idx_female = idx_female[0];
                sex[idx_male]=1; sex[idx_female]=0;
                Data=Data.drop('Sex',axis=1)
                Data=Data.assign(Sex=pd.Series(sex,Data.index))
                
                if count==0:
                    DataTrain = Data.copy()
                else:
                    DataTest = Data.copy()
        
        ## Separate the list of biomarkers from confounders and meta data
        count=-1;
        for Data in [DataTrain,DataTest]:
            count=count+1;
            idx = Data['Diagnosis']==1
            DataBiomarkers=Data
            DataBiomarkers=DataBiomarkers.drop(Factors,axis=1)
            H = list(DataBiomarkers)
            for j in droplist:
                if any(j in f for f in H):
                    DataBiomarkers=DataBiomarkers.drop(j,axis=1)
            for j in Groups:
                GroupValues.append([])
                if any(j in f for f in H):
                    GroupValues[-1].append(DataBiomarkers[j])
                    DataBiomarkers=DataBiomarkers.drop(j,axis=1)
            BiomarkersList=list(DataBiomarkers)
            BiomarkersListnew=[]
            for i in range(len(BiomarkersList)):
                BiomarkersListnew.append(BiomarkersList[i].replace(' ','_'))
                BiomarkersListnew[i]=BiomarkersListnew[i].replace('-','_')
            
            for i in range(len(BiomarkersList)):
                Data=Data.rename(columns={BiomarkersList[i]:BiomarkersListnew[i]})
            ## Contruct the formula for regression. Also compute the mean value of the confounding factors for correction
            if count==0: # Do it only for training set
                str_confounders=''
                mean_confval = np.zeros(len(Factors))
                for j in range(len(Factors)):
                    str_confounders = str_confounders + '+' + Factors[j] 
                    mean_confval[j]=np.nanmean(Data[Factors[j]].values)
                str_confounders=str_confounders[1:]
            
                ## Multiple linear regression
                betalist=[]
                for i in range(len(BiomarkersList)):
                    str_formula = BiomarkersListnew[i] + '~' + str_confounders
                    result = sm.ols(formula=str_formula, data=Data[idx]).fit()
                    betalist.append(result.params)
            
            ## Correction for the confounding factors
            Deviation=(Data[Factors] - mean_confval)
            Deviation[np.isnan(Deviation)]=0
            for i in range(len(BiomarkersList)):
                betai=betalist[i].values
                betai_slopes = betai[1:]
                CorrectionFactor = np.dot(Deviation.values,betai_slopes)
                Data[BiomarkersListnew[i]] = Data[BiomarkersListnew[i]] - CorrectionFactor
            Data=Data.drop(Factors,axis=1)
            Data=Data.drop(Groups,axis=1)
            for i in range(len(BiomarkersList)):
                Data=Data.rename(columns={BiomarkersListnew[i]:BiomarkersList[i]})
            if count==0:
                DataTrain = Data.copy()
            else:
                DataTest = Data.copy()
                
    if flag_test==0:
        DataTest=[]
        if len(Groups)==0:
            GroupValues=[]
        else:
            GroupValues=GroupValues.pop()
    return DataTrain,DataTest,BiomarkersList,GroupValues

def pd2mat(pdData,BiomarkersList,flag_JointFit):
    # Convert arrays from pandas dataframe format to the matrices (which are used in DEBM algorithms)
    num_events = len(BiomarkersList);
    if flag_JointFit==0:
        num_feats = 1
    num_subjects=pdData.shape[0]
    matData = np.zeros((num_subjects,num_events,num_feats))
    for i in range(num_events):
        matData[:,i,0]=pdData[BiomarkersList[i]].values
    return matData

def ExamDate_str2num(ExamDateSeries):
    
    timestamp = np.zeros(len(ExamDateSeries))
    for i in range(len(ExamDateSeries)):
        stre=ExamDateSeries.values[i]
        if len(stre)>5:
            timestamp[i]=time.mktime(datetime.datetime.strptime(stre, "%Y-%m-%d").timetuple())
        else:
            timestamp[i]=np.nan
    TimestampSeries=pd.Series(timestamp)
    return TimestampSeries     
