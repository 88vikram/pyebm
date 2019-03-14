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
from pyebm.mixture_model import do_classification as dc 
from pyebm import core_utilities as cu
import numpy as np

def fit(DataIn,MethodOptions=False,VerboseOptions=False,Factors=None,Labels = None,DataTest=[],Groups=[]):
    
    if Factors == None:
        Factors = ['Age','Sex','ICV']
    if Labels == None:
        Labels=['CN','MCI','AD']
    
    data_CN_raw,data_MCI_raw,data_AD_raw,ptid_CN_raw,ptid_MCI_raw,ptid_AD_raw,Data_all,pdData_all, \
                    Data_test_all,pdDataTest_all,GroupValues_cn,GroupValues_ad,GroupValues_mci,GroupValues_test,BiomarkersList, DMO, DVO=cu.parse_inputs(DataIn,\
                    MethodOptions, VerboseOptions,Factors,Labels,DataTest,Groups,'debm')
    
    data_CN_raw_list,data_AD_raw_list,data_all_list,ptid_all_list,GroupValues_list=cu.bootstrap_data_prep(data_CN_raw, \
                    data_MCI_raw,data_AD_raw,ptid_CN_raw,ptid_MCI_raw,ptid_AD_raw,Data_all,pdData_all,DMO,DVO,BiomarkersList,GroupValues_cn,GroupValues_ad,GroupValues_mci)
    
    pi0_all=[];  
    params_opt_all=[]; 
    event_centers_all=[]; 
    SubjTrainAll = []; 
    SubjTestAll = []
    for i in range(len(data_AD_raw_list)): ## For each bootstrapped iteration
        
        if len(data_AD_raw_list)>1:
            print([i],end=""),
        data_AD_raw = data_AD_raw_list[i];
        data_CN_raw = data_CN_raw_list[i];
        Data_all =  data_all_list[i];   
        GroupValues =    [GroupValues_list[i]] ;
        BiomarkerParams,p_yes,p_no,likeli_post,likeli_pre=cu.do_mixturemodel(DMO,data_AD_raw,data_CN_raw,Data_all,Groups,GroupValues)   
        pi0,event_centers=cu.find_central_ordering(Data_all,p_yes,BiomarkerParams,Groups,GroupValues,DMO,'debm')
        subj_stages=cu.do_patient_staging(pi0,event_centers,DMO,p_yes,p_no,likeli_post,likeli_pre,Groups,GroupValues)
        SubjTrain=cu.compile_subject_data(ptid_all_list[i],p_yes,subj_stages)
        SubjTest=[];
        subj_stages_test=[];
        if len(Data_test_all)>0:
            p_yes_test,p_no_test,likeli_post_test,likeli_pre_test=dc.Classify(Data_test_all,BiomarkerParams,DMO);                
            subj_stages_test=cu.do_patient_staging(pi0,event_centers,DMO,p_yes_test,p_no_test,likeli_post_test,likeli_pre_test,Groups,GroupValues_test)
            SubjTest=cu.compile_subject_data(pdDataTest_all['PTID'],p_yes_test,subj_stages_test)
            
        pi0_all.append(pi0)
        params_opt_all.append(BiomarkerParams)
        event_centers_all.append(event_centers)
        SubjTrainAll.append(SubjTrain)
        SubjTestAll.append(SubjTest)
        
    pi0_mean, evn, evn_full = cu.get_mean_ordering(pi0_all,event_centers_all, data_AD_raw_list, Groups,GroupValues)
    
    ModelOutput=cu.compile_model_output(BiomarkersList,pi0_mean,pi0_all,event_centers_all,params_opt_all)
    cu.show_outputs(Data_all, Data_test_all, pdData_all, Labels, pdDataTest_all, subj_stages, subj_stages_test, BiomarkerParams,\
                 evn_full, evn, BiomarkersList,pi0_all,pi0_mean,DMO,DVO,Groups,GroupValues,'debm')                                  

    return ModelOutput,SubjTrainAll,SubjTestAll