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
import scipy.optimize as opt
import scipy.stats


def GMM_Control(Data_all,Ncni,Nadi,Groups,GroupValues,params_nobias,type_opt=1,itvl=1.96,HyperParams=1,params_pruned=[],Mixing=[]):

    Nevents=Data_all.shape[1];
    Nfeats=Data_all.shape[2]
    params = np.zeros((Nevents,5,Nfeats));                   
    
    if len(Groups)==0:
        for i in range(Nevents):
            for j in range(Nfeats):
                params[i,:,j]=params_nobias[i,:,j];
        params_all = [params]
    else:
        params_all = params_nobias
    
    for i in range(Nevents):
        bnds_all=[]
        paramsik = []
        Dalli=Data_all[:,i,:];
        Dalli_valid_c = Dalli
        for k in range(len(params_all)):
            params = params_all[k]
            if type(params_pruned)== list:
                if len(params_pruned)>0:
                    pp = params_pruned[k]
                else:
                    pp = []
            else:
                pp=params_pruned
            paramsik.append(params[i,:,:])
            bnds=np.zeros(((Nfeats*4)+1,2));
            if type_opt==1:
                bnds[-1,0]=0.01;
                bnds[-1,1]=0.99;            
                for j in range(Nfeats):
                    bnds[(j*4)+0,0] = params[i,0,j] - itvl*(params[i,1,j]/np.sqrt(Ncni[i]))
                    bnds[(j*4)+0,1] = params[i,0,j] + itvl*(params[i,1,j]/np.sqrt(Ncni[i]))
                    bnds[(j*4)+2,0] = params[i,2,j] - itvl*(params[i,3,j]/np.sqrt(Nadi[i]))
                    bnds[(j*4)+2,1] = params[i,2,j] + itvl*(params[i,3,j]/np.sqrt(Nadi[i]))
                            
                    bnds[(j*4)+1,0] = params[i,1,j] - itvl*(params[i,1,j]/np.sqrt(Ncni[i]-2))
                    bnds[(j*4)+1,1] = params[i,1,j] + itvl*(params[i,1,j]/np.sqrt(Ncni[i]-2))
                    bnds[(j*4)+3,0] = params[i,3,j] - itvl*(params[i,3,j]/np.sqrt(Nadi[i]-2))
                    bnds[(j*4)+3,1] = params[i,3,j] + itvl*(params[i,3,j]/np.sqrt(Nadi[i]-2))
            else:
                bnds[-1,0]=params_all[k][i,4,0];
                bnds[-1,1]=params_all[k][i,4,0];
                if type_opt==2:
                    for j in range(Nfeats):
                        bnds[(j*4)+0,0] = np.min([params[i,0,j],params[i,2,j]])
                        bnds[(j*4)+0,1] = np.max([params[i,0,j],params[i,2,j]])
                        bnds[(j*4)+2,0] = np.min([params[i,0,j],params[i,2,j]])
                        bnds[(j*4)+2,1] = np.max([params[i,0,j],params[i,2,j]])
                                
                        bnds[(j*4)+1,0] = params[i,1,j]
                        bnds[(j*4)+1,1] = params[i,1,j]*2
                        bnds[(j*4)+3,0] = params[i,3,j]
                        bnds[(j*4)+3,1] = params[i,3,j]*2
                else:
                    for j in range(Nfeats):
                        bnds[(j*4)+0,0] = np.min([pp[i,0,j],pp[i,2,j]])
                        bnds[(j*4)+0,1] = np.max([pp[i,0,j],pp[i,2,j]])
                        bnds[(j*4)+2,0] = np.min([pp[i,0,j],pp[i,2,j]])
                        bnds[(j*4)+2,1] = np.max([pp[i,0,j],pp[i,2,j]])
                                
                        bnds[(j*4)+1,0] = pp[i,1,j]
                        bnds[(j*4)+1,1] = pp[i,1,j]*2
                        bnds[(j*4)+3,0] = pp[i,3,j]
                        bnds[(j*4)+3,1] = pp[i,3,j]*2
            bnds_all.append(bnds)
        if len(params_all)==1:
            params[i,:,:]=GMM(Dalli_valid_c,Nfeats,params[i,:,:],bnds,Groups,GroupValues,[])
        else:
            params_grps_i=coGMM(Dalli_valid_c,Nfeats,paramsik,bnds_all,Groups,GroupValues,HyperParams)
            for k in range(len(params_all)):
                params_all[k][i,:,0] = params_grps_i[:,k]
    if len(params_all)==1:
        params_all = params
    return params_all,bnds_all

def coGMM(Data,Nfeats,paramsik,bnds_all,Groups,GroupValues,HyperParams):
    tup_arg=(Data,Groups,GroupValues,HyperParams);
    params=np.asarray(paramsik).flatten()
    bnds=np.asarray(bnds_all).flatten()
    lb = bnds[::2]
    ub = bnds[1::2]
    bnds = np.transpose(np.asarray([lb,ub]))
    res=opt.minimize(calculate_objectivefunction_cogmm,params,args=(tup_arg),method='SLSQP', options={'disp': False,'maxiter': 600}, bounds=bnds)
    gval=np.unique(GroupValues[0])
    idx_valid=~np.isnan(gval)
    gval=gval[idx_valid]
    params_groups = np.zeros((5,len(gval)))
    for j in range(len(gval)):
        params_groups[0,j]=res.x[(j*5)+0]
        params_groups[1,j]=res.x[(j*5)+1]
        params_groups[2,j]=res.x[(j*5)+2]
        params_groups[3,j]=res.x[(j*5)+3]
        params_groups[4,j]=res.x[(j*5)+4]
    return params_groups
    
def calculate_objectivefunction_cogmm(param,data,Groups,GroupValues,Hyperparams):
    gval=np.unique(GroupValues[0])
    idx_valid=~np.isnan(gval)
    gval=gval[idx_valid]  
    obj1 = 0; obj2=0;
    for g in range(len(gval)):
        paramsg = param[(5*g):(5*g+5)]
        idx=GroupValues[0]==gval[g]
        data_g=data[idx]
        lg=calculate_likelihood_gmm(paramsg,data_g,[],[],[])
        obj1 = obj1 + lg
    return obj1

def GMM(Data,Nfeats,params,bnds,Groups,GroupValues,Mixing):

        idx = bnds[:,1]-bnds[:,0]<=0.01
        bnds[idx,1] = bnds[idx,0] + 0.01;
        tup_arg=(Data,Groups,GroupValues,Mixing);
        try:
            p=np.zeros(((Nfeats*4)+1));
            for j in range(Nfeats):
                p[(j*4)+0]=params[0,j]
                p[(j*4)+1]=params[1,j]
                p[(j*4)+2]=params[2,j]
                p[(j*4)+3]=params[3,j]
            p[-1]=params[4,0]
            if Nfeats==1:
                res=opt.minimize(calculate_likelihood_gmm,p,args=(tup_arg),method='SLSQP', options={'disp': False,'maxiter': 600}, bounds=bnds)
            if max(np.isnan(res.x))!=1: # In case of convergence to a nan value
                p[:]=res.x[:]
                for j in range(Nfeats):
                    params[0,j]=p[(j*4)+0]
                    params[1,j]=p[(j*4)+1]
                    params[2,j]=p[(j*4)+2]
                    params[3,j]=p[(j*4)+3]
                params[4,:]=p[-1]
                    
        except ValueError:
            print('Warning: Error in Gaussian Mixture Model')
        return params

def calculate_likelihood_gmm(param,data,Groups,GroupValues,Mixing):
    if len(Mixing)==0:
        param_mix=param[4];
        norm_pre=scipy.stats.norm(loc=param[0], scale=param[1]);
        norm_post=scipy.stats.norm(loc=param[2],scale=param[3]);                                            #uniform. loc=min boarder, scale=max-min
        invalid_indices=np.isnan(data);
        valid_indices=np.logical_not(invalid_indices)
        likeli_pre=norm_pre.pdf(data[valid_indices]);
        likeli_post=norm_post.pdf(data[valid_indices]);
        
        likeli=np.multiply(param_mix,likeli_pre) + np.multiply(1-param_mix,likeli_post) + 1e-100;
        loglikeli=-np.sum(np.log(likeli));
    else:
        gval=np.unique(GroupValues[0])
        idx_valid=~np.isnan(gval)
        gval=gval[idx_valid]
        loglikeli_list=[]
        
        norm_pre=scipy.stats.norm(loc=param[0], scale=param[1]);
        norm_post=scipy.stats.norm(loc=param[2],scale=param[3]);                                            #uniform. loc=min boarder, scale=max-min

        for g in range(len(gval)):
            param_mix=Mixing[g];
            idx=GroupValues[0]==gval[g]
            data_g=data[idx]
            invalid_indices=np.isnan(data_g);
            valid_indices=np.logical_not(invalid_indices)
            data_g = data_g[valid_indices]
            likeli_pre=norm_pre.pdf(data_g);
            likeli_post=norm_post.pdf(data_g);
            likeli=np.multiply(param_mix,likeli_pre) + np.multiply(1-param_mix,likeli_post) + 1e-100;
            loglikeli=-np.sum(np.log(likeli)) ;  
            
            ##### Including weighted sum
            #loglikeli = np.divide(loglikeli,len(data_g)) 
            ##### Including weighted sum
            
            loglikeli_list.append(loglikeli);
            
        loglikeli=np.sum(loglikeli_list)
    return loglikeli

def GMM_AY(Data_all,data_AD_raw,data_CN_raw):
    
    Neve = data_AD_raw.shape[1]
    params = np.zeros((Neve,5,1));
    for i in range(Neve):
        Dcni=data_CN_raw[:,i,0];
        params[i,0,0]=np.nanmean(Dcni);
        params[i,1,0] = np.nanstd(Dcni)+0.001; # Useful when standard deviation is 0
        Dalli=Data_all[:,i];
        Dadi=data_AD_raw[:,i,0];
        params[i,2,0]=np.nanmean(Dadi);
        params[i,3,0] = np.nanstd(Dadi)+0.001;
        params[i,4,0] = 0.5 # initialization with equal likelihood
        bnds=np.zeros((5,2));
        #bnds[:,1]=(np.max(Dalli_valid_c),2*params[i,1] ,np.max(Dalli_valid_c),2*params[i,3],1);
        event_sign = params[i,0]<params[i,2];
        if event_sign==1:
            bnds[:,0] = (np.nanmin(Dalli),0,params[i,2,0],0,0);
            bnds[:,1]=(params[i,0,0],params[i,1,0] ,np.nanmax(Dalli),params[i,3,0],1);
        else:
            bnds[:,0] = (params[i,0,0],0,np.nanmin(Dalli),0,0);
            bnds[:,1]=(np.nanmax(Dalli),params[i,1,0] ,params[i,2,0],params[i,3,0],1);
        idx = bnds[:,1]-bnds[:,0]<=0.001
        bnds[idx,1] = bnds[idx,0] + 0.001; # Upper bound should be greater 
        tup_arg=(Dalli[:,0],[],0,[])
        try:
            res=opt.least_squares(calculate_likelihood_gmm,params[i,:,0],args=(tup_arg),method='trf', bounds=np.transpose(bnds))
            if max(np.isnan(res.x))!=1: # In case of convergence to a nan value
                params[i,:,0]=res.x
        except ValueError:
            print('Warning: Error in Gaussian Mixture Model')
            
    return params
    
def calculate_prob_mm(Data,params,val_invalid=0.5):
    # Works for single dimensional features
    m=np.shape(Data);
    p_yes=np.zeros(m);
    likeli_pre_all=np.zeros(m);
    likeli_post_all=np.zeros(m);                
    for i in range(0,m[1]):
        r=1-params[i,4]
        Datai=Data[:,i];
        paramsi=params[i,:];
        p=np.zeros(np.shape(Datai));
        invalid_indices=np.isnan(Datai);
        p[invalid_indices]=val_invalid;
        valid_indices=np.logical_not(invalid_indices)
        Datai_valid=Datai[valid_indices];
        norm_pre=scipy.stats.norm(loc=paramsi[0], scale=paramsi[1]);
        likeli_pre=norm_pre.pdf(Datai_valid);
        norm_post=scipy.stats.norm(loc=paramsi[2],scale=paramsi[3]);
        likeli_post=norm_post.pdf(Datai_valid);
        
        p[valid_indices]=np.divide(likeli_post*r,(likeli_post*r)+(1-r)*likeli_pre+1e-100);
        likeli_pre_all[valid_indices,i]=likeli_pre
        likeli_post_all[valid_indices,i]=likeli_post
        likeli_pre_all[invalid_indices,i] = 0.5
        likeli_post_all[invalid_indices,i] = 0.5
        p_yes[:,i]=p;
    p_no=1-p_yes;
    return p_yes,p_no,likeli_pre_all,likeli_post_all
