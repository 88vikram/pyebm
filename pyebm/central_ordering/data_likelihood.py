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
import copy
import random

from pyebm.mixture_model import gaussian_mixture_model as gmm
from pyebm.mixture_model import do_classification as dc 

def adhoc(Data,params,n_startpoints,n_iterations,mix,DMO):
    

    m1=np.shape(Data)[1];
    if DMO.MixtureModel[:3]=='GMM':
        params_opt = np.zeros((len(params.Mixing),5,1))
        params_opt[:,:2,0] = params.Control
        params_opt[:,2:4,0] = params.Disease
        params_opt[:,4,0] = params.Mixing
        p_yes,p_no,likeli_pre_all,likeli_post_all=gmm.calculate_prob_mm(Data,params_opt);
        
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
    
def MCMC(Data,params,DMO):
    
    n_mcmciterations = DMO.N_MCMC
    n_startpoints = DMO.NStartpoints
    n_iterations = DMO.Niterations
    mix = params.Mixing
    m1=np.shape(Data)[1];
    if DMO.MixtureModel[:3]=='GMM':                    
        params_opt = np.zeros((len(params.Mixing),5,1))
        params_opt[:,:2,0] = params.Control
        params_opt[:,2:4,0] = params.Disease
        params_opt[:,4,0] = params.Mixing
        p_yes,p_no,likeli_pre_all,likeli_post_all=gmm.calculate_prob_mm(Data,params_opt);
    
    seq_init,obj_fn,samples_likelihood_mat=adhoc(Data,params,n_startpoints,n_iterations,mix,DMO);
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

