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
import numpy as np
from sklearn.utils import resample
import pandas as pd 
from pyebm.mixture_model import gaussian_mixture_model as gmm  
from pyebm.mixture_model import do_classification as dc  
from pyebm.central_ordering import generalized_mallows as gm
from pyebm.inputs_outputs import prepare_outputs as po
from pyebm.inputs_outputs import visualize
from pyebm.inputs_outputs import prepare_inputs as pi
from pyebm.central_ordering import data_likelihood as dl
from collections import namedtuple
import copy as cp
import sys
from numpy import matlib

def parse_inputs(DataIn,MethodOptions,VerboseOptions,Factors,Labels,DataTest,Groups,maskidx,algo_type):
    
    if algo_type=='debm':
        DMO = namedtuple('DefaultMethodOptions','MixtureModel Bootstrap PatientStaging')
        DMO.JointFit = 0; DMO.MixtureModel='GMMvv2'; DMO.Bootstrap=0; DMO.PatientStaging=['exp','p']; 
        
    elif algo_type=='ebm':
        DMO = namedtuple('DefaultMethodOptions',' MixtureModel Bootstrap PatientStaging NStartpoints Niterations N_MCMC')
        DMO.JointFit=0; DMO.MixtureModel='GMMvv2'; DMO.Bootstrap=0; DMO.PatientStaging=['exp','p']; 
        DMO.NStartpoints=10; DMO.Niterations=1000; DMO.N_MCMC=10000;
    DVO = namedtuple('DefaultVerboseOptions','Distributions Ordering PlotOrder WriteBootstrapData PatientStaging')
    DVO.Distributions = 0; DVO.Ordering=0; DVO.PlotOrder=1; DVO.WriteBootstrapData=0; DVO.PatientStaging = 0
    
    ## Replacing the Default Options with Options given by the user
    if type(MethodOptions)!= bool:
        for fld in MethodOptions._fields:
            setattr(DMO,fld,getattr(MethodOptions, fld))
    
    if type(VerboseOptions)!= bool:
        for fld in VerboseOptions._fields:
            setattr(DVO,fld,getattr(VerboseOptions, fld))
            
    ## Data Preparation for DEBM
    pdData_all,UniqueSubjIDs = pi.pdReadData(DataIn,0,Labels=Labels)
    pdDataTest_all,UniqueTestSubjIDs = pi.pdReadData(DataTest,0,Labels=Labels)
    pdData_all,pdDataTest_all,BiomarkersList,GroupValues= pi.CorrectConfounders(pdData_all,pdDataTest_all,Factors=Factors,Groups=Groups)
    idx_CN=pdData_all['Diagnosis'].values==1; idx_AD=pdData_all['Diagnosis'].values==3; idx_MCI=pdData_all['Diagnosis'].values==2;
    
    Data_all=pi.pd2mat(pdData_all,BiomarkersList,0)
    
    if len(maskidx) == 0:
        maskidx = np.ones(Data_all.shape[0],bool)
        
    if len(pdDataTest_all)>0:
        Data_test_all=pi.pd2mat(pdDataTest_all,BiomarkersList,0)
        if len(Groups)>0:
            GroupValues_test=[GroupValues[1]]
        else:
            GroupValues_test=[]
    else:
        Data_test_all=[]
        GroupValues_test=[]
    #idx_CN=np.where(idx_CN); idx_CN = idx_CN[0];
    #idx_AD=np.where(idx_AD); idx_AD = idx_AD[0];
    #idx_MCI=np.where(idx_MCI); idx_MCI = idx_MCI[0];
    data_AD_raw = Data_all[idx_AD,:,:]; data_CN_raw = Data_all[idx_CN,:,:]; data_MCI_raw=Data_all[idx_MCI,:,:]
    maskidx_AD = maskidx[idx_AD]
    maskidx_CN = maskidx[idx_CN]
    maskidx_MCI = maskidx[idx_MCI]
    
    if len(Groups)==0:
        GroupValues_ad=[];GroupValues_cn=[];GroupValues_mci=[];
    else:
        GroupValues_ad=GroupValues[0][idx_AD]
        GroupValues_cn=GroupValues[0][idx_CN]
        GroupValues_mci=GroupValues[0][idx_MCI]
        #data_AD_raw = data_AD_raw[GroupValues_ad==2]
        #data_CN_raw = data_CN_raw[GroupValues_cn==2]

    ptid_AD_raw = pdData_all.loc[idx_AD,'PTID']; ptid_CN_raw = pdData_all.loc[idx_CN,'PTID']; ptid_MCI_raw = pdData_all.loc[idx_MCI,'PTID']
    return data_CN_raw,data_MCI_raw,data_AD_raw,ptid_CN_raw,ptid_MCI_raw,ptid_AD_raw,Data_all,pdData_all,Data_test_all, pdDataTest_all,GroupValues_cn,GroupValues_ad,GroupValues_mci,GroupValues_test,BiomarkersList, DMO, DVO, maskidx, maskidx_CN, maskidx_AD, maskidx_MCI

def bootstrap_data_prep(data_CN_raw,data_MCI_raw,data_AD_raw,ptid_CN_raw,ptid_MCI_raw,ptid_AD_raw,Data_all, pdData_all,DMO,DVO,BiomarkersList,Groups,GroupValues_cn,GroupValues_ad,GroupValues_mci,maskidx, maskidx_CN, maskidx_MCI, maskidx_AD):
    
    data_AD_raw_list = []
    data_CN_raw_list = []; data_all_list = []; ptid_all_list=[]
    GroupValues_list = []
    maskidx_list = []
    for i in range(DMO.Bootstrap):
        bs_data_AD_raw=resample(data_AD_raw,random_state=i);
        bs_ptid_AD_raw=resample(ptid_AD_raw,random_state=i);
        bs_maskidx_AD = resample(maskidx_AD,random_state=i);
        bs_data_CN_raw=resample(data_CN_raw,random_state=i);
        bs_ptid_CN_raw=resample(ptid_CN_raw,random_state=i);
        bs_maskidx_CN = resample(maskidx_CN,random_state=i);
        if len(GroupValues_ad)>0:
            bs_GroupValues_ad=resample(GroupValues_ad,random_state=i);
            bs_GroupValues_cn=resample(GroupValues_cn,random_state=i);    
        else:
            bs_GroupValues_ad=[]; bs_GroupValues_cn=[]; bs_GroupValues_mci = [];
        if len(data_MCI_raw)>0:
            bs_data_MCI_raw=resample(data_MCI_raw,random_state=i);
            bs_ptid_MCI_raw=resample(ptid_MCI_raw,random_state=i);  
            bs_maskidx_MCI = resample(maskidx_MCI,random_state=i);
            if len(GroupValues_mci)>0:
                bs_GroupValues_mci=resample(GroupValues_mci,random_state=i);                   
            bs_data_all=np.concatenate((bs_data_AD_raw,bs_data_CN_raw,bs_data_MCI_raw))  
            bs_ptid_all=np.concatenate((bs_ptid_AD_raw,bs_ptid_CN_raw,bs_ptid_MCI_raw)) 
            bs_GroupValues = np.concatenate((bs_GroupValues_ad,bs_GroupValues_cn,bs_GroupValues_mci))
            bs_maskidx = np.concatenate((bs_maskidx_AD,bs_maskidx_CN,bs_maskidx_MCI)) 
        else:
            bs_data_all=np.concatenate((bs_data_AD_raw,bs_data_CN_raw))  
            bs_ptid_all=np.concatenate((bs_ptid_AD_raw,bs_ptid_CN_raw)) 
            bs_GroupValues = np.concatenate((bs_GroupValues_ad,bs_GroupValues_cn))
            bs_maskidx = np.concatenate((bs_maskidx_AD,bs_maskidx_CN)) 
            
        labels_AD = np.zeros(len(bs_data_AD_raw))+3
        labels_CN = np.zeros(len(bs_data_CN_raw))+1
        if len(data_MCI_raw)>0:
            labels_MCI = np.zeros(len(bs_data_MCI_raw))+2
            labels_all=np.concatenate((labels_AD,labels_CN,labels_MCI))
        else:
            labels_all=np.concatenate((labels_AD,labels_CN))
        if type(DVO.WriteBootstrapData)==str:
            str_out=DVO.WriteBootstrapData+'_'+str(i)+'.csv'
            Dbs=pd.DataFrame(bs_data_all[:,:,0],columns=BiomarkersList)
            Dbs['Diagnosis']=labels_all
            Dbs.to_csv(str_out,index=False)
        data_AD_raw_list.append(bs_data_AD_raw)
        data_CN_raw_list.append(bs_data_CN_raw)
        data_all_list.append(bs_data_all)
        maskidx_list.append(bs_maskidx)
        ptid_all_list.append(bs_ptid_all)
        GroupValues_list.append(bs_GroupValues)
    if DMO.Bootstrap==False:
        data_AD_raw_list.append(data_AD_raw)
        data_CN_raw_list.append(data_CN_raw)
        data_all_list.append(Data_all)
        ptid_all_list.append(pdData_all['PTID'])
        maskidx_list.append(maskidx)
        if len(Groups)>0:
            GroupValues_list.append(pdData_all[Groups[0]])
        else:
            GroupValues_list=[0]
        
    return data_CN_raw_list,data_AD_raw_list,data_all_list,ptid_all_list,GroupValues_list,maskidx_list
    
def do_mixturemodel(DMO,data_AD_raw,data_CN_raw,Data_all,Groups,GroupValues,GroupValues_cn,GroupValues_ad,HyperParams=1,flag_init_together=1,only_init=0):
        BiomarkerParams = namedtuple('BiomarkerParams','Control Disease Mixing')
        if DMO.MixtureModel[:3]=='GMM':
            if len(Groups)==0 or flag_init_together==1:
                Data_AD_pruned,Data_CN_pruned,params_raw,params_pruned=dc.Reject(data_AD_raw,data_CN_raw);
                
            else:
                params_pruned = []
                gval=np.unique(GroupValues[0])
                idx_valid=~np.isnan(gval)
                gval=gval[idx_valid]
                tot_grp=[]
                for g in gval:
                    tot_grp.append(np.sum(GroupValues[0]==g))
                gmax = gval[np.argmax(tot_grp)]
                for g in gval:
                    data_AD_raw_gr = data_AD_raw[GroupValues_ad==gmax]
                    data_CN_raw_gr = data_CN_raw[GroupValues_cn==gmax]
                    Data_AD_pruned,Data_CN_pruned,params_raw,params_pr_gr=dc.Reject(data_AD_raw_gr,data_CN_raw_gr);
                    params_pruned.append(params_pr_gr)
            #Data_AD_pruned,Data_CN_pruned,params_raw,params_pruned=dc.Reject(data_AD_raw,data_CN_raw);
            Ncni = []; Nadi = []
            for i in range(Data_all.shape[1]):
                Ncni.append(Data_CN_pruned[i].shape[0])
                Nadi.append(Data_AD_pruned[i].shape[0])
            ## Bias Correction to get an unbiased estimate                                      
            if DMO.MixtureModel=='GMMvv2': #or DMO.MixtureModel=='GMMvv3'                 
                if len(Groups)==0:
                    params_opt = params_pruned
                    mixes_old=params_opt[:,4,0]; flag_stop = 0;
                    BiomarkerParams.Mixing=[];
                    count=0
                    while flag_stop==0:
                        count+=1
                        params_optmix,bnds_all = gmm.GMM_Control(Data_all,Ncni,Nadi,Groups,
                                                                 GroupValues,params_opt,itvl=0.001);
                        if only_init==0:
                            params_opt,bnds_all=gmm.GMM_Control(Data_all,Ncni,Nadi,
                                                            Groups,GroupValues,params_optmix,type_opt=3,
                                                            params_pruned=params_pruned,
                                                            Mixing=BiomarkerParams.Mixing);
                        else:
                            params_opt = params_optmix
                            flag_stop=1
                        mixes = params_opt[:,4,0] 
                        if np.mean(np.abs(mixes-mixes_old))<10**-2:
                            flag_stop=1;
                        mixes_old = np.copy(mixes)                
                else:
                    
                    flag_stop = 0;
                    gval=np.unique(GroupValues[0])
                    idx_valid=~np.isnan(gval)
                    gval=gval[idx_valid]
                    mixes_old=[];
                    
                    params_opt_all = []
                    for k in range(len(gval)):
                        if type(params_pruned) == list:
                            mixes_old.append(params_pruned[k][:,4,0]) 
                            params_opt_all.append(params_pruned[k])
                        else:
                            mixes_old.append(params_pruned[:,4,0]) 
                            params_opt_all.append(params_pruned)
                        
                    mixes=mixes_old
                    while flag_stop==0:
                        BiomarkerParams.Mixing = []
                        count=-1
                        for g in gval: # estimation of mixing parameters.
                            idx=GroupValues[0]==g
                            count=count+1
                            params_opt = np.copy(params_opt_all[count])
                            params_opt[:,4,0] = mixes[count]
                            params_opt,bnds_all = gmm.GMM_Control(Data_all[idx,:],Ncni,Nadi,Groups,GroupValues,
                                                                  [params_opt],itvl=0.001)                                     
                            params_opt_all[count] = np.copy(params_opt)                                 
                            BiomarkerParams.Mixing.append( params_opt[:,4,0] )
                        if only_init==0:
                            params_opt,bnds_all=gmm.GMM_Control(Data_all,Ncni,Nadi,
                                                                Groups,GroupValues,params_opt_all,type_opt=3,
                                                                params_pruned=params_pruned,
                                                                Mixing=BiomarkerParams.Mixing,HyperParams=HyperParams); # Gaussian parameters
                        else:
                            flag_stop=1
                            params_opt = params_opt_all
                        mixes = BiomarkerParams.Mixing
                        if np.mean(np.abs(mixes[0]-mixes_old[0]))<10**-2:
                            flag_stop=1;
                        mixes_old = np.copy(mixes)
                        
            elif DMO.MixtureModel=='GMMvv1':
                params_opt,bnds_all = gmm.GMM_Control(Data_all,Ncni,Nadi,params_pruned,type_opt=1);
            elif DMO.MixtureModel=='GMMay':
                params_opt=gmm.GMM_AY(Data_all,data_AD_raw,data_CN_raw)
            if len(Groups)==0:  
                BiomarkerParams.Mixing = params_opt[:,4,0]
            if type(params_opt)!=list:
                BiomarkerParams.Control = params_opt[:,0:2,0]
                BiomarkerParams.Disease = params_opt[:,2:4,0]
            else:
                BiomarkerParams.Control = [];     BiomarkerParams.Disease = []
                for k in range(len(params_opt)):
                    BiomarkerParams.Control.append(params_opt[k][:,:2,0])
                    BiomarkerParams.Disease.append(params_opt[k][:,2:4,0])
            p_yes,p_no,likeli_post,likeli_pre=dc.Classify(Data_all,BiomarkerParams,DMO,Groups,GroupValues); 
        return BiomarkerParams,p_yes,p_no,likeli_post,likeli_pre
            
def find_central_ordering(Data_all,p_yes,BiomarkerParams,Groups,GroupValues,DMO,algo_type,maskidx):
    indv_heterogeneity = np.zeros(Data_all.shape[0])
    if len(Groups)==0:     
        if algo_type == 'debm':                                           
            pi0,event_centers,indv_heterogeneity = gm.weighted_mallows.fitMallows(p_yes[maskidx,:],BiomarkerParams.Mixing);
        elif algo_type == 'ebm': 
            mix = np.copy(BiomarkerParams.Mixing);
            BiomarkerParams.Mixing=0.5+np.zeros(len(mix));
            pi0,event_centers=dl.MCMC(Data_all[:,:,0],BiomarkerParams,DMO);    
            BiomarkerParams.Mixing=mix
    else:
        gval=np.unique(GroupValues[0])
        idx_valid=~np.isnan(gval)
        gval=gval[idx_valid]
        
        event_centers=[]; pi0=[]; count_gr=-1
        for g in gval:
            idx=GroupValues[0]==g
            pygr=p_yes[np.logical_and(idx,maskidx),:]
            Dgr = Data_all[np.logical_and(idx,maskidx),:,0]
            count_gr = count_gr+1
            if algo_type == 'debm':
                pi0_gr,ecgr,indv_hetgr = gm.weighted_mallows.fitMallows(pygr,BiomarkerParams.Mixing[count_gr]);
                
                ecgr = ecgr - np.min(ecgr) + 0.1
                ecgr = 0.8* ((ecgr - 0.1) / (np.max(ecgr) - 0.1)) + 0.1
                
            elif algo_type == 'ebm': 
                mix = np.copy(BiomarkerParams.Mixing[count_gr]);
                BiomarkerParams.Mixing[count_gr]=0.5+np.zeros(len(mix));
                pi0_gr,ecgr=dl.MCMC(Dgr,BiomarkerParams,DMO);    
                BiomarkerParams.Mixing[count_gr]=mix
            pi0.append(pi0_gr)
            event_centers.append(ecgr)
            indv_heterogeneity[np.logical_and(idx,maskidx)]=indv_hetgr
    return pi0,event_centers,indv_heterogeneity
    
def do_patient_staging(pi0,event_centers,DMO,p_yes,p_no,likeli_post,likeli_pre,Groups,GroupValues):
        if DMO.PatientStaging[1]=='p':
            Y = p_yes;
            N = 1- p_yes;
        else:
            Y = likeli_post;
            N = likeli_pre;
        
        if len(Groups)==0:
            subj_stages=patient_staging(pi0,event_centers,Y,N,DMO.PatientStaging);
        else:
            gval=np.unique(GroupValues[0])
            idx_valid=~np.isnan(gval)
            gval=gval[idx_valid]
            subj_stages=np.zeros(Y.shape[0]); count_gr=-1;
            for g in gval:
                count_gr = count_gr+1;
                idx=GroupValues[0]==g
                Ygr=Y[idx,:]
                Ngr=N[idx,:]
                subj_stages_gr=patient_staging(pi0[count_gr],event_centers[count_gr],Ygr,Ngr,DMO.PatientStaging);
                subj_stages[idx]=subj_stages_gr
        return subj_stages
    
def compile_subject_data(ptidi,p_yes,subj_stages):
    SubjTrain=pd.DataFrame(columns=['PTID', 'Orderings','Weights','Stages'])
    SubjTrain['PTID'] = ptidi
    so_list,weights_list = po.Prob2ListAndWeights(p_yes); 
    SubjTrain['Orderings'] = so_list
    SubjTrain['Weights'] = weights_list
    SubjTrain['Stages'] = subj_stages
    return SubjTrain
    
def compile_model_output(BiomarkersList,pi0_mean,pi0_all,event_centers_all,params_opt_all):
    ModelOutput = namedtuple('ModelOutput','MeanCentralOrdering EventCenters CentralOrderings BiomarkerList BiomarkerParameters')
    ModelOutput.BiomarkerList = BiomarkersList; ModelOutput.MeanCentralOrdering=pi0_mean;
    ModelOutput.EventCenters = event_centers_all; ModelOutput.CentralOrderings=pi0_all;
    ModelOutput.BiomarkerParameters = params_opt_all
    return ModelOutput
    
def get_mean_ordering(pi0_all,event_centers_all, data_AD_raw_list, Groups,GroupValues):
    event_centers_ordered = cp.deepcopy(event_centers_all)
    gval=np.unique(GroupValues[0])
    idx_valid=~np.isnan(gval)
    gval=gval[idx_valid]    
    if len(Groups)==0:
        for i in range(len(event_centers_all)):
            count=-1
            for x in pi0_all[i]:
                count=count+1;
                event_centers_ordered[i][x] = cp.deepcopy(event_centers_all[i][count])
        
        evn = np.asarray(event_centers_ordered)
        evn_full = np.mean(evn,axis=0)
        if len(data_AD_raw_list)>1:
            pi0_mean = list(np.argsort(evn_full))
        else:
            pi0_mean=pi0_all[0]
    else:
        for g in range(len(gval)):
            for i in range(len(event_centers_all)):
                count=-1
                for x in pi0_all[i][g]:
                    count=count+1;
                    event_centers_ordered[i][g][x] = cp.deepcopy(event_centers_all[i][g][count])
            
        evn = np.asarray(event_centers_ordered)
        evn_full = np.mean(evn,axis=0)
        pi0_mean=[]
        for g in range(len(gval)):
            if len(data_AD_raw_list)>1:
                pi0_mean.append(list(np.argsort(evn_full[g,:])))
            else:
                pi0_mean.append(pi0_all[0][g])
            
    return pi0_mean, evn, evn_full
    
def show_outputs(Data_all, Data_test_all, pdData_all, Labels, pdDataTest_all, subj_stages, subj_stages_test, BiomarkerParams,\
                 evn_full, evn, BiomarkersList,pi0_all,pi0_mean,DMO,DVO,Groups,GroupValues,algo_type):
        ## Visualize Results
    if DVO.Ordering==1:
        if len(Groups)==0:
            visualize.Ordering(BiomarkersList, pi0_all,pi0_mean,DVO.PlotOrder);
            if algo_type=='debm':
                visualize.EventCenters(BiomarkersList,pi0_mean, evn_full, evn);
        else:
            gval=np.unique(GroupValues[0])
            idx_valid=~np.isnan(gval)
            gval=gval[idx_valid]
            for g in range(len(gval)):
                sys.stdout.flush()
                print(Groups[0],'=',gval[g])
                pi0_all_g=[]
                for i in range(len(pi0_all)):
                    pi0_all_g.append(pi0_all[i][g])
                visualize.Ordering(BiomarkersList, pi0_all_g,pi0_mean[g],DVO.PlotOrder);
                if algo_type=='debm':
                    visualize.EventCenters(BiomarkersList,pi0_mean[g], evn_full[g,:], evn[:,g,:]);
                
    if DVO.Distributions==1:
        if len(Groups)==0:
            params_all=[BiomarkerParams];
            visualize.BiomarkerDistribution(Data_all,params_all,BiomarkersList);
        else:
            BiomarkerParamsGr = namedtuple('BiomarkerParams','Control Disease Mixing')

            gval=np.unique(GroupValues[0])
            idx_valid=~np.isnan(gval)
            gval=gval[idx_valid]
            countgr = -1;
            for g in gval:
                idxgr=GroupValues[0]==g
                countgr = countgr + 1
                BiomarkerParamsGr.Control = BiomarkerParams.Control[countgr]
                BiomarkerParamsGr.Disease = BiomarkerParams.Disease[countgr]
                BiomarkerParamsGr.Mixing = BiomarkerParams.Mixing[countgr]
                params_all=[BiomarkerParamsGr];
                visualize.BiomarkerDistribution(Data_all[idxgr,:,:],params_all,BiomarkersList);
            
    if DVO.PatientStaging==1:
        print ('Estimated patient stages of subjects in training set')
        visualize.Staging(subj_stages,pdData_all['Diagnosis'],Labels)
        if len(Data_test_all)>0:
            print ('Estimated patient stages of subjects in test set')
            visualize.Staging(subj_stages_test,pdDataTest_all['Diagnosis'],Labels)
    return
 
def patient_staging(pi0,event_centers,likeli_post,likeli_pre,type_staging):
    
    L_yes=np.divide(likeli_post,likeli_post+likeli_pre+1e-10)
    
    L_no = 1 - L_yes
    event_centers_pad=np.insert(event_centers,0,0)
    event_centers_pad=np.append(event_centers_pad,1)
    pk_s=np.diff(event_centers_pad)
    pk_s[:]=1;
    
    m=L_yes.shape
    prob_stage = np.zeros((m[0],m[1]+1))
    p_no_perm = L_no[:,pi0];
    p_yes_perm = L_yes[:,pi0];
    for j in range(m[1]+1):
        prob_stage[:,j]=pk_s[j]*np.multiply(np.nanprod(p_yes_perm[:,:j],axis=1),np.nanprod(p_no_perm[:,j:],axis=1))

    all_stages_rep2=matlib.repmat(event_centers_pad[:-1],m[0],1)
    
    if type_staging[0]=='exp':
        subj_stages = np.zeros(prob_stage.shape[0])
        for i in range(prob_stage.shape[0]):
            idx_nan=np.isnan(p_yes_perm[i,:])
            pr=prob_stage[i,1:]
            ev = event_centers_pad[1:-1]
            subj_stages[i]=np.mean(np.multiply(np.append(prob_stage[i,0],pr[~idx_nan]),np.append(event_centers_pad[0],ev[~idx_nan])))/(np.mean(np.append(prob_stage[i,0],pr[~idx_nan])))
    elif type_staging[0]=='ml':
        subj_stages=np.argmax(prob_stage,axis=1)
    
    return subj_stages