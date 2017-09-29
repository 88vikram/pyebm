
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

def Reject(data_AD,data_CN):

    import numpy as np
    from MixtureModel import calculate_prob_mmN
    from MixtureModel import calculate_prob_mm
                  
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

def GMM_Control(Data_all,Data_CN_pruned,Data_AD_pruned,params_nobias,type_opt=1,itvl=1.96,params_pruned=[]):

    import numpy as np
    Nevents=len(Data_CN_pruned);
    Nfeats=Data_CN_pruned[0].shape[1]
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
            Ncni = Data_CN_pruned[i].shape[0]
            Nadi= Data_AD_pruned[i].shape[0]              
            for j in range(Nfeats):
                bnds[(j*4)+0,0] = params[i,0,j] - itvl*(params[i,1,j]/np.sqrt(Ncni))
                bnds[(j*4)+0,1] = params[i,0,j] + itvl*(params[i,1,j]/np.sqrt(Ncni))
                bnds[(j*4)+2,0] = params[i,2,j] - itvl*(params[i,3,j]/np.sqrt(Nadi))
                bnds[(j*4)+2,1] = params[i,2,j] + itvl*(params[i,3,j]/np.sqrt(Nadi))
                        
                bnds[(j*4)+1,0] = params[i,1,j] - itvl*(params[i,1,j]/np.sqrt(Ncni-2))
                bnds[(j*4)+1,1] = params[i,1,j] + itvl*(params[i,1,j]/np.sqrt(Ncni-2))
                bnds[(j*4)+3,0] = params[i,3,j] - itvl*(params[i,3,j]/np.sqrt(Nadi-2))
                bnds[(j*4)+3,1] = params[i,3,j] + itvl*(params[i,3,j]/np.sqrt(Nadi-2))

        bnds_all.append(bnds)
        params[i,:,:]=GMM(Dalli_valid_c,Nfeats,params[i,:,:],bnds)
    
    return params,bnds_all

def GMM(Data,Nfeats,params,bnds):
        from MixtureModel import calculate_likelihood_gmmN
        from MixtureModel import calculate_likelihood_gmm
        import scipy.optimize as opt
        import numpy as np
        idx = bnds[:,1]-bnds[:,0]<=0.001
        bnds[idx,1] = bnds[idx,0] + 0.001; # Upper bound should be greater 
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
                res=opt.minimize(calculate_likelihood_gmmN,p,args=(tup_arg),method='SLSQP', options={'disp': False,'maxiter': 600}, bounds=bnds)
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
    
def Classify(Data4Classification,params):

    from MixtureModel import calculate_prob_mmN
    from MixtureModel import calculate_prob_mm
    import numpy as np
    
    Nfeats=Data4Classification.shape[2]
    if Nfeats==1:
        p_yes,p_no,likeli_pre,likeli_post=calculate_prob_mm(Data4Classification[:,:,0],params,val_invalid=np.nan);
    else:
        p_yes,p_no,likeli_pre,likeli_post=calculate_prob_mmN(Data4Classification,params,val_invalid=np.nan);

    return p_yes,p_no,likeli_post,likeli_pre

def adhoc(Data,params,n_startpoints,n_iterations,mix):
    import numpy as np
    import copy

    import MixtureModel as mm
    m1=np.shape(Data)[1];           
    p_yes,p_no,likeli_pre_all,likeli_post_all=mm.calculate_prob_mm(Data,params);
    ml_ordering_mat = np.zeros((n_startpoints,m1));
    samples_likelihood_mat = np.zeros(n_startpoints);

    for startpoint in range (0,n_startpoints):
        this_samples_ordering = np.zeros((n_iterations,m1));
        this_samples_likelihood = np.zeros((n_iterations,1));
    
        seq_init = np.random.permutation(m1);
        this_samples_ordering[0,:] = seq_init;
        
        for i in range(0,n_iterations):
            if (i>0):
                move_event_from = int( np.ceil(m1*np.random.rand())-1);
                move_event_to = int ( np.ceil(m1*np.random.rand())-1);
                current_ordering = copy.copy(this_samples_ordering[i-1,:]);
                temp = current_ordering[move_event_from];
                current_ordering[move_event_from] = current_ordering[move_event_to];
                current_ordering[move_event_to]=temp;
                this_samples_ordering[i,:] = copy.copy(current_ordering);
           
            S = this_samples_ordering[i,:];
            this_samples_likelihood[i],p_prob_k,pk=objfn_likelihood(S,p_yes,p_no,mix);
            if (i>0):
                ratio = np.exp(this_samples_likelihood[i]-this_samples_likelihood[i-1]);
                if (ratio<1):
                    this_samples_likelihood[i] = copy.copy(this_samples_likelihood[i-1]);
                    this_samples_ordering[i,:] = copy.copy(this_samples_ordering[i-1,:]);

        perm_index = np.argmax(this_samples_likelihood);
        ml_ordering = this_samples_ordering[perm_index,:];    
        ml_ordering_mat[startpoint,:] = ml_ordering;
        
        samples_likelihood_mat[startpoint] = this_samples_likelihood[perm_index,0];
    
    max_like_ix = np.argmax(samples_likelihood_mat);
    obj_fn = samples_likelihood_mat[max_like_ix];
    ml_ordering = ml_ordering_mat[max_like_ix,:]
    return ml_ordering,obj_fn,samples_likelihood_mat
    
def MCMC(Data,params,n_mcmciterations,mix,n_startpoints,n_iterations):
    import numpy as np
    import copy
    import random
    
    import MixtureModel as mm
    m1=np.shape(Data)[1];
    p_yes,p_no,likeli_pre_all,likeli_post_all=mm.calculate_prob_mm(Data,params);
    
    seq_init,obj_fn,samples_likelihood_mat=adhoc(Data,params,n_startpoints,n_iterations,mix);
    this_samples_ordering = np.zeros((n_mcmciterations,m1));
    this_samples_likelihood = np.zeros((n_mcmciterations,1));
    this_samples_ordering[0,:]=copy.copy(seq_init);
    
    for i in range(0,n_mcmciterations):
            if (i>0):
                move_event_from = int(np.ceil(m1*np.random.rand())-1);
                move_event_to = int(np.ceil(m1*np.random.rand())-1);
                current_ordering = copy.copy(this_samples_ordering[i-1,:]);
                temp = current_ordering[move_event_from];
                current_ordering[move_event_from] = current_ordering[move_event_to];
                current_ordering[move_event_to]=temp;
                this_samples_ordering[i,:] = copy.copy(current_ordering);
           
            S = this_samples_ordering[i,:];
            this_samples_likelihood[i],p_prob_k,pk=objfn_likelihood(S,p_yes,p_no,mix);
            if (i>0):
                ratio = np.exp(this_samples_likelihood[i]-this_samples_likelihood[i-1]);
                if (ratio<random.random()):
                    this_samples_likelihood[i] = copy.copy(this_samples_likelihood[i-1]);
                    this_samples_ordering[i,:] = copy.copy(this_samples_ordering[i-1,:]);
    
    perm_index = np.argmax(this_samples_likelihood);
    ml_ordering = this_samples_ordering[perm_index,:];  
    opt_likelihood,p_prob_k,pk=objfn_likelihood(ml_ordering,p_yes,p_no,mix);
    this_samples_likelihood = np.zeros(m1-1)                                  
    for i in range(m1-1):
        this_ordering = np.copy(ml_ordering)
        a=this_ordering[i]; b=this_ordering[i+1];
        this_ordering[i]=b; this_ordering[i+1]=a;
        this_samples_likelihood[i],p_prob_k,pk=objfn_likelihood(this_ordering,p_yes,p_no,mix);
    ordering_distances=-this_samples_likelihood + opt_likelihood
    event_centers = np.cumsum(ordering_distances)
    event_centers = event_centers/np.max(event_centers)   
    pi0=list(ml_ordering.astype(int))
    event_centers = np.insert(event_centers,0,0)
    return pi0,event_centers
    
def objfn_likelihood(S,p_yes,p_no,mix):
    import numpy as np
    m=np.shape(p_yes);    
    k=m[1]+1;
    abnormmix=1-mix

    S=S.astype(int);
    p_perm_k = np.zeros((m[0],k));
    P_yes_perm = p_yes[:,S];
    P_no_perm = p_no[:,S];
    mixperm = mix[S]
    abnormmixperm = abnormmix[S]
    pk = np.zeros(k)
    psk = np.zeros(k)
    for j in range(0,k):
        mi1 = np.prod(abnormmixperm[:j])
        mi2 = np.prod(mixperm[j:])
        psk[j] = mi1*mi2
    ps = np.sum(psk)
    pk=psk/ps
    for j in range(0,k):
        p1=np.prod(P_yes_perm[:,0:j],axis=1);
        p2=np.prod(P_no_perm[:,j:k],axis=1);
        p_perm_k[:,j] = np.multiply(p1,p2);
     
    pxjs = np.sum(np.multiply(pk,p_perm_k),1)+1e-250
    likelihood = np.log(ps) + np.sum(np.log(pxjs));
                           
    return likelihood,p_perm_k,pk
