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
from scipy.stats import multivariate_normal
import scipy.optimize as opt
import scipy.stats


def GMM_Control(Data_all,Ncni,Nadi,params_nobias,type_opt=1,itvl=1.96,params_pruned=[]):

    Nevents=Data_all.shape[1];
    Nfeats=Data_all.shape[2]
    params = np.zeros((Nevents,5,Nfeats));                   
    bnds_all=[]
    for i in range(Nevents):
        for j in range(Nfeats):
            params[i,:,j]=params_nobias[i,:,j]; 

    for i in range(Nevents):
        Dalli=Data_all[:,i,:];
        Dalli_valid_c = Dalli
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
            bnds[-1,0]=params_nobias[i,4,0];
            bnds[-1,1]=params_nobias[i,4,0];
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
                    bnds[(j*4)+0,0] = np.min([params_pruned[i,0,j],params_pruned[i,2,j]])
                    bnds[(j*4)+0,1] = np.max([params_pruned[i,0,j],params_pruned[i,2,j]])
                    bnds[(j*4)+2,0] = np.min([params_pruned[i,0,j],params_pruned[i,2,j]])
                    bnds[(j*4)+2,1] = np.max([params_pruned[i,0,j],params_pruned[i,2,j]])
                            
                    bnds[(j*4)+1,0] = params_pruned[i,1,j]
                    bnds[(j*4)+1,1] = params_pruned[i,1,j]*2
                    bnds[(j*4)+3,0] = params_pruned[i,3,j]
                    bnds[(j*4)+3,1] = params_pruned[i,3,j]*2
        bnds_all.append(bnds)
        params[i,:,:]=GMM(Dalli_valid_c,Nfeats,params[i,:,:],bnds)
    
    return params,bnds_all

def GMM(Data,Nfeats,params,bnds):

        idx = bnds[:,1]-bnds[:,0]<=0.00001
        bnds[idx,1] = bnds[idx,0] + 0.00001; # Upper bound should be greater 
        tup_arg=(Data,0);
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
            else:
                res=opt.minimize(calculate_likelihood_gmmN,p,args=(tup_arg),method='SLSQP', options={'disp': False,'maxiter': 600}, bounds=bnds, tol=1e-12)
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

def calculate_likelihood_gmm(param,data,dummy):
    # The dummy argument is a hack to overcome how single element tuple is handled 
    # differently in windows and linux. While it remains a single element tuple in windows,
    # in linux, the data type changes to that of the tuple element.
    
    param_mix=param[4];
    norm_pre=scipy.stats.norm(loc=param[0], scale=param[1]);
    norm_post=scipy.stats.norm(loc=param[2],scale=param[3]);
    invalid_indices=np.isnan(data);
    valid_indices=np.logical_not(invalid_indices)
    likeli_pre=norm_pre.pdf(data[valid_indices]);
    likeli_post=norm_post.pdf(data[valid_indices]);
    
    likeli=np.multiply(param_mix,likeli_pre) + np.multiply(1-param_mix,likeli_post) + 1e-100;
    loglikeli=-np.sum(np.log(likeli)) ;
    
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
        tup_arg=(Dalli[:,0],0);
        
        try:
            res=opt.least_squares(calculate_likelihood_gmm,params[i,:,0],args=(tup_arg),method='trf', bounds=np.transpose(bnds))
            if max(np.isnan(res.x))!=1: # In case of convergence to a nan value
                params[i,:,0]=res.x
        except ValueError:
            print('Warning: Error in Gaussian Mixture Model')
            
    return params

def calculate_likelihood_gmmN(param,data,dummy):
    # The dummy argument is a hack to overcome how single element tuple is handled 
    # differently in windows and linux. While it remains a single element tuple in windows,
    # in linux, the data type changes to that of the tuple element.
    
    m=np.shape(data)
        
    Nfeats=m[1]
    param_mix=param[-1];
    cov_n = np.zeros((Nfeats,Nfeats))
    mean_n=np.zeros(Nfeats)
    for j in range(Nfeats):
        cov_n[j,j]=param[(j*4)+1]**2.
        mean_n[j]=param[(j*4)+0]
    invalid_indices=np.isnan(data[:,0]);
    valid_indices=np.logical_not(invalid_indices)
    likeli_pre=multivariate_normal.pdf(data[valid_indices,:],mean=mean_n, cov=cov_n);
    
    cov_d = np.zeros((Nfeats,Nfeats))
    mean_d=np.zeros(Nfeats)
    for j in range(Nfeats):
        cov_d[j,j]=param[(j*4)+3]**2.
        mean_d[j]=param[(j*4)+2]
        
    likeli_post=multivariate_normal.pdf(data[valid_indices,:],mean=mean_d, cov=cov_d);
    
    likeli=np.multiply(param_mix,likeli_pre) + np.multiply(1-param_mix,likeli_post) + 1e-100;
    loglikeli=-np.sum(np.log(likeli)) ;
    return loglikeli
    
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

def calculate_prob_mmN(Data,params,val_invalid=0.5):
    # Works for N dimensional features
    
    m=np.shape(Data);
    Nfeats=m[2]
    p_yes=np.zeros((m[0],m[1]));
    likeli_pre_all=np.zeros((m[0],m[1]));
    likeli_post_all=np.zeros((m[0],m[1])); 
    for i in range(0,m[1]):
        r=1-params[i,4,0]
        Datai=Data[:,i,:];
                      
        p=np.zeros(m[0]);
        invalid_indices=np.isnan(Datai[:,0])
        p[invalid_indices]=val_invalid;
        valid_indices=np.logical_not(invalid_indices);
        Datai_valid=Datai[valid_indices,:];
        # This is where the modification ends           
        cov_n = np.zeros((Nfeats,Nfeats))
        mean_n=np.zeros(Nfeats)
        for j in range(Nfeats):
            cov_n[j,j]=params[i,1,j]**2.
            mean_n[j]=params[i,0,j]
        
        likeli_pre=multivariate_normal.pdf(Datai_valid,mean=mean_n, cov=cov_n);
        
        cov_d = np.zeros((Nfeats,Nfeats))
        mean_d=np.zeros(Nfeats)
        for j in range(Nfeats):
            cov_d[j,j]=params[i,3,j]**2.
            mean_d[j]=params[i,2,j]
        
        likeli_post=multivariate_normal.pdf(Datai_valid,mean=mean_d, cov=cov_d);
        
        p[valid_indices]=np.divide(likeli_post*r,(likeli_post*r)+(1-r)*likeli_pre+1e-100);
        likeli_pre_all[valid_indices,i]=likeli_pre
        likeli_post_all[valid_indices,i]=likeli_post
        likeli_pre_all[invalid_indices,i] = 0.5
        likeli_post_all[invalid_indices, i] = 0.5
        p_yes[:,i]=p;
    p_no=1-p_yes;
    return p_yes,p_no,likeli_pre,likeli_post
