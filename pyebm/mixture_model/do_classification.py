
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
from pyebm.mixture_model.gaussian_mixture_model import calculate_prob_mmN
from pyebm.mixture_model.gaussian_mixture_model import calculate_prob_mm

def Reject(data_AD,data_CN):
        
    m=np.shape(data_AD)
    Data_AD=[]; Data_CN=[];
    params_raw = np.zeros((m[1],5,m[2]));   
    params_pruned = np.zeros((m[1],5,m[2]));           
    for i in range(m[1]):
        for j in range(m[2]):
            Dcni=np.reshape(data_CN[:,i,j],[-1,1]);
            cn_mean=np.nanmean(Dcni);
            cn_std = np.nanstd(Dcni)+0.001;
            params_raw[i,0,j]=cn_mean; params_raw[i,1,j]=cn_std;
                      
            Dadi=np.reshape(data_AD[:,i,j],[-1,1]);
            ad_mean=np.nanmean(Dadi);
            ad_std = np.nanstd(Dadi)+0.001;
            params_raw[i,2,j] = ad_mean; params_raw[i,3,j] = ad_std; 
            params_raw[i,4,j]=0.5;
        
        p = np.zeros((1,5,m[2]))
        p[0,:,:]=params_raw[i,:,:]
        
        mcn=np.shape(data_CN)
        Dcni_valid_c = np.zeros((mcn[0],1,mcn[2]))
        Dcni_valid_c[:,0,:]=data_CN[:,i,:]
        if m[2]==1:
            py,pn,likeli_pre,likeli_post=calculate_prob_mm(Dcni_valid_c[:,:,0],p,val_invalid=np.nan);
        else:
            py,pn,likeli_pre,likeli_post=calculate_prob_mmN(Dcni_valid_c,p,val_invalid=np.nan);
        idx_in_cn=np.where(py<=0.5)
        
        mad=np.shape(data_AD)
        Dadi_valid_c = np.zeros((mad[0],1,mad[2]))
        Dadi_valid_c[:,0,:]=data_AD[:,i,:]
        if m[2]==1:
            py,pn,likeli_pre,likeli_post=calculate_prob_mm(Dadi_valid_c[:,:,0],p,val_invalid=np.nan);
        else:
            py,pn,likeli_pre,likeli_post=calculate_prob_mmN(Dadi_valid_c,p,val_invalid=np.nan);
        idx_in_ad=np.where(py>0.5)

        Data_CN.append(Dcni_valid_c[idx_in_cn])
        Data_AD.append(Dadi_valid_c[idx_in_ad])
        for j in range(m[2]):
            params_pruned[i,0,j]=np.nanmean(Data_CN[i][:,j]); params_pruned[i,1,j]=np.nanstd(Data_CN[i][:,j])+0.001;
            params_pruned[i,2,j]=np.nanmean(Data_AD[i][:,j]); params_pruned[i,3,j]=np.nanstd(Data_AD[i][:,j])+0.001;
            params_pruned[i,4,j]=params_raw[i,4,j];

    return Data_AD,Data_CN,params_raw,params_pruned
    
def Classify(Data4Classification,BiomarkerParams,DMO,Groups = None ,GroupValues = None):
    if Groups == None:
        Groups = []
        
    if DMO.MixtureModel[:3]=='GMM':
         Nfeats=Data4Classification.shape[2]
         
         if Nfeats==1:
             params = np.zeros((Data4Classification.shape[1],5))
             params[:,:2] = BiomarkerParams.Control
             params[:,2:4] = BiomarkerParams.Disease
         else:
             params = np.zeros((Data4Classification.shape[1],5,Nfeats))
             params[:,:2,:] = BiomarkerParams.Control
             params[:,2:4,:] = BiomarkerParams.Disease
         if len(Groups)==0:
            if Nfeats==1:
                params[:,4] = BiomarkerParams.Mixing
                p_yes,p_no,likeli_pre,likeli_post=calculate_prob_mm(Data4Classification[:,:,0],params,val_invalid=np.nan);
            else:
                params[:,4,0] = BiomarkerParams.Mixing
                p_yes,p_no,likeli_pre,likeli_post=calculate_prob_mmN(Data4Classification,params,val_invalid=np.nan);
         else:
            p_yes = np.zeros((Data4Classification.shape[0],Data4Classification.shape[1]))
            p_no = np.zeros((Data4Classification.shape[0],Data4Classification.shape[1]))
            likeli_post = np.zeros((Data4Classification.shape[0],Data4Classification.shape[1]))
            likeli_pre = np.zeros((Data4Classification.shape[0],Data4Classification.shape[1]))
            gval=np.unique(GroupValues[0])
            idx_valid=~np.isnan(gval)
            gval=gval[idx_valid]
            count=-1
            for g in gval:
                count=count+1
                idx=GroupValues[0]==g
                params[:,4] = BiomarkerParams.Mixing[count]
                if Nfeats==1:
                    p_yes_gr,p_no_gr,likeli_pre_gr,likeli_post_gr=calculate_prob_mm(Data4Classification[idx,:,0],params,val_invalid=np.nan);
                else:
                    p_yes_gr,p_no_gr,likeli_pre_gr,likeli_post_gr=calculate_prob_mmN(Data4Classification[idx,:,:],params,val_invalid=np.nan);
                p_yes[idx,:]=p_yes_gr
                p_no[idx,:]=p_no_gr
                likeli_post[idx,:]=likeli_post_gr
                likeli_pre[idx,:]=likeli_post_gr
    return p_yes,p_no,likeli_post,likeli_pre