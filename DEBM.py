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

def Control(DataIn,MethodOptions=False,VerboseOptions=False,Factors=['Age','Sex','ICV'],Labels=['CN','MCI','AD']):
    
    from collections import namedtuple
    import pandas as pd
    ## Default Options for the function call
    DMO = namedtuple('DefaultMethodOptions','MixtureModel Bootstrap Staging')
    DVO = namedtuple('DefaultVerboseOptions','Distributions Ordering PlotOrder WriteBootstrapData')
    DMO.JointFit = 0; DMO.MixtureModel='vv1'; DMO.Bootstrap=0; DMO.Staging='c';
    DVO.Distributions = 0; DVO.Ordering=0; DVO.PlotOrder=0; DVO.WriteBootstrapData=0;
    
    ## Replacing the Default Options with Options given by the user
    if type(MethodOptions)!= bool:
        for fld in MethodOptions._fields:
            setattr(DMO,fld,getattr(MethodOptions, fld))
    
    if type(VerboseOptions)!= bool:
        for fld in VerboseOptions._fields:
            setattr(DVO,fld,getattr(VerboseOptions, fld))
    
    import corealgo as ca                 
    import util
    import weighted_mallows as wma
    import MixtureModel as mm
    import numpy as np
    import time
    from sklearn.utils import resample
    
    ## Data Preparation for DEBM
    t0 = time.time()
    pdData_all,UniqueSubjIDs = util.pdReadData(DataIn,0,Labels=Labels)
    pdData_all,BiomarkersList= util.CorrectConfounders(pdData_all,Factors)

    num_events=len(BiomarkersList);
    idx_CN=pdData_all['Diagnosis']==1; idx_AD=pdData_all['Diagnosis']==3; idx_MCI=pdData_all['Diagnosis']==2;
    Data_all=util.pd2mat(pdData_all,BiomarkersList,DMO.JointFit)
    idx_CN=np.where(idx_CN); idx_CN = idx_CN[0];
    idx_AD=np.where(idx_AD); idx_AD = idx_AD[0];
    idx_MCI=np.where(idx_MCI); idx_MCI = idx_MCI[0];
    data_AD_raw = Data_all[idx_AD,:,:]; data_CN_raw = Data_all[idx_CN,:,:]; data_MCI_raw=Data_all[idx_MCI,:,:]
    
    ## Data Preparation for Bootstrapping
    pi0_all=[]; data_AD_raw_list=[]; params_opt_all=[]; event_centers_all=[]; p_yes_all=[];
    data_CN_raw_list = []; data_all_list = [];
    for i in range(DMO.Bootstrap):
        bs_data_AD_raw=resample(data_AD_raw,random_state=i);
        bs_data_CN_raw=resample(data_CN_raw,random_state=i);
        bs_data_MCI_raw=resample(data_MCI_raw,random_state=i);
        bs_data_all=np.concatenate((bs_data_AD_raw,bs_data_CN_raw,bs_data_MCI_raw))  
        labels_AD = np.zeros(len(bs_data_AD_raw))+3
        labels_CN = np.zeros(len(bs_data_CN_raw))+1
        labels_MCI = np.zeros(len(bs_data_MCI_raw))+2
        labels_all=np.concatenate((labels_AD,labels_CN,labels_MCI))
        if type(DVO.WriteBootstrapData)==str:
            str_out=DVO.WriteBootstrapData+'_'+str(i)+'.csv'
            Dbs=pd.DataFrame(bs_data_all[:,:,0],columns=BiomarkersList)
            Dbs['Diagnosis']=labels_all
            Dbs.to_csv(str_out,index=False)
        data_AD_raw_list.append(bs_data_AD_raw)
        data_CN_raw_list.append(bs_data_CN_raw)
        data_all_list.append(bs_data_all)
    if DMO.Bootstrap==False:
        data_AD_raw_list.append(data_AD_raw)
        data_CN_raw_list.append(data_CN_raw)
        data_all_list.append(Data_all)
    
    for i in range(len(data_AD_raw_list)): ## For each bootstrapped iteration
        ## Reject possibly wrongly labeled data 
        if len(data_AD_raw_list)>1:
            print [i],
        data_AD_raw = data_AD_raw_list[i];
        data_CN_raw = data_CN_raw_list[i];
        Data_all =  data_all_list[i];                             
        Data_AD_pruned,Data_CN_pruned,params_raw,params_pruned=ca.Reject(data_AD_raw,data_CN_raw);

        if DMO.MixtureModel=='vv1':
            params_opt,bnds_all = ca.GMM_Control(Data_all,Data_CN_pruned,Data_AD_pruned,params_pruned,type_opt=1);
        elif DMO.MixtureModel=='ay':
            params_opt=mm.GMM_AY(Data_all,data_AD_raw,data_CN_raw)

        ## Get Posterior Probabilities                                       
        p_yes,p_no,likeli_post,likeli_pre=ca.Classify(Data_all,params_opt);                                                  
        ## Probabilistic Kendall's Tau based Generalized Mallows Model                                                   
        pi0,event_centers = wma.weighted_mallows.fitMallows(p_yes,params_opt);
        
        pi0_all.append(pi0)
        params_opt_all.append(params_opt)
        event_centers_all.append(event_centers)
        p_yes_all.append(p_yes)
        
        
    ## Get Mean Ordering of all the bootstrapped iterations.
    if len(data_AD_raw_list)>1:
        wts=np.arange(num_events,0,-1)
        wts_all=np.tile(wts,(len(data_AD_raw_list),1)).tolist()
        (pi0_mean,bestscore,scores) = wma.weighted_mallows.consensus(num_events,pi0_all,wts_all,[]);   
    else:
        pi0_mean=pi0_all[0]
        
        
    t1=time.time();
    total_time=t1-t0;
                
    ## Visualize Results
    if DVO.Ordering==1:
        util.VisualizeOrdering(BiomarkersList, pi0_all,pi0_mean,DVO.PlotOrder);
    if DVO.Distributions==1:
        params_all=[params_opt];
        util.VisualizeBiomarkerDistribution(Data_all,params_all,BiomarkersList);
                                           
    if DMO.Bootstrap==False:
        pi0_all=pi0_all[0]

    return pi0_mean,pi0_all,params_opt_all,BiomarkersList,event_centers_all,p_yes_all
