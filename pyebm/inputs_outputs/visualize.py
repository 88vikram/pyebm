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
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats 
import pandas as pd
import seaborn as sns
from matplotlib import rc
    
def BiomarkerDistribution(Data_all,params_all,BiomarkersList):

    m=np.shape(Data_all)
    n=len(params_all)
    fig, ax = plt.subplots(int(np.ceil(m[1]/3)), 3, figsize=(13, 4*np.ceil(m[1]/3)))
    for i in range(m[1]):
            Dalli=Data_all[:,i,0];
            valid_data=np.logical_not(np.isnan(Dalli))
            Dallis=Dalli[valid_data].reshape(-1,1); Dallis=Dallis[:,0];
            x_grid=np.linspace(np.min(Dallis),np.max(Dallis),1000)
            for j in range(n):
                i1 = int(i/3);
                j1 = np.remainder(i,3)
                paramsij=params_all[j];
                norm_pre=scipy.stats.norm(loc=paramsij.Control[i,0], scale=paramsij.Control[i,1]);
                norm_post=scipy.stats.norm(loc=paramsij.Disease[i,0],scale=paramsij.Disease[i,1]);
                likeli_pre=norm_pre.pdf(x_grid)
                likeli_post=norm_post.pdf(x_grid)
                h=np.histogram(Dallis,50)                          
                maxh=np.nanmax(h[0])
                ax[i1,j1].hist(Dallis,50, fc='blue', histtype='stepfilled', alpha=0.3, normed=False)
                #ylim=ax[i,j].get_ylim()
                likeli_pre=likeli_pre*(paramsij.Mixing[i]);
                likeli_post=likeli_post*(1-paramsij.Mixing[i]);
                likeli_tot=likeli_pre+likeli_post;
                likeli_tot_corres = np.zeros(len(h[1])-1)
                bin_size=h[1][1]-h[1][0]
                for k in range(len(h[1])-1):
                    bin_loc=h[1][k]+bin_size
                    idx=np.argmin(np.abs(x_grid-bin_loc))
                    likeli_tot_corres[k] = likeli_tot[idx]
                
                max_scaling=maxh/np.max(likeli_tot);
                
                scaling_opt=1; opt_score=np.inf
                if max_scaling>1:
                    scale_range=np.arange(1,max_scaling+1,max_scaling/1000.)
                else:
                    scale_range=np.arange(1,(10/max_scaling)+1,max_scaling/1000.)
                    scale_range=np.reciprocal(scale_range)
                
                for s in scale_range:
                    l2norm=(likeli_tot_corres*s - h[0])**2
                    idx_nonzero=h[0]>0
                    l2norm=l2norm[idx_nonzero]
                    score=np.sum(l2norm)
                    if score < opt_score:
                        opt_score=score
                        scaling_opt=s;
                likeli_pre=likeli_pre*scaling_opt;
                likeli_post=likeli_post*scaling_opt;
                likeli_tot=likeli_pre+likeli_post;
                
                ax[i1,j1].plot(x_grid,likeli_pre, color='green', alpha=0.5, lw=3)
                ax[i1,j1].plot(x_grid,likeli_post, color='red', alpha=0.5, lw=3)
                ax[i1,j1].plot(x_grid,likeli_tot, color='black', alpha=0.5, lw=3)
                
                ax[i1,j1].set_title(BiomarkersList[i])
    for j in range(1,n):
        plt.setp([a.get_yticklabels() for a in fig.axes[j::n]], visible=False);
    plt.show()
    
def Ordering(labels, pi0_all,pi0_mean, plotorder):

    columns = ['Features', 'Event Position', 'Count']
    datapivot = pd.DataFrame(columns = columns)
    for i in range(len(labels)):
        bb = [item.index(i) for item in pi0_all]
        for j in range(len(labels)):
            cc = pd.DataFrame([[bb.count(j),j, labels[i]]], index = [j], columns = ['Count','Event Position','Features'])
            datapivot = datapivot.append(cc)
    datapivot = datapivot.pivot("Features", "Event Position", "Count")
    if plotorder == True:
        newindex = []
        for i in range(len(list(pi0_mean))):
            aa = labels[pi0_mean[i]]
            newindex.append(aa)
        datapivot = datapivot.reindex(newindex)        
    xticks = np.arange(len(labels)) + 1
    datapivot = datapivot[datapivot.columns].astype(float)
    fig, ax = plt.subplots(1,1,figsize=(7, 7))
    heatmap = sns.heatmap(datapivot, cmap = 'binary', xticklabels=xticks, vmin=0, vmax=len(pi0_all),ax=ax)
    plt.sca(ax)
    plt.title('Positional variance diagram of the central ordering')
    plt.yticks(rotation=0) 
    plt.show()
    
def EventCenters(labels, pi0_mean, evn_full, evn):
    
    fig, ax = plt.subplots(1,1,figsize=(7, 7))
    for i in range(len(labels)):
        (_, caps, _)=ax.errorbar(evn_full[i],len(labels)-1-pi0_mean.index(i),xerr=np.std(evn[:,i],axis=0),fmt='--o',color='b',alpha=0.7,barsabove=True,capsize=1)
        for cap in caps:
            cap.set_markeredgewidth(5)
    ax.set_yticks(np.arange(0,len(labels)-1+.1,1))
    ax.set_yticklabels([labels[x] for x in pi0_mean[::-1]],rotation=45)
    #ax.get_yaxis().set_visible(False)
    ax.set_xlim([0,1])
    ax.set_xticks(np.arange(0,1.,0.1))
    #ax.set_xticklabels(np.arange(0,1.,0.1))
    ax.grid(b=True, axis='x', color='k', linestyle='--',alpha=0.5,which='major')
    ax.set_xlabel('Disease Stage')
    ax.set_ylim([-0.5,0.5+len(labels)-1])
    ax.set_title('Event Center Variance Diagram')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
def Staging(subj_stages,Diagnosis,Labels):
    
    if np.max(subj_stages)>1:
        nb=np.max(subj_stages)+2
        freq,binc=np.histogram(subj_stages,bins=np.arange(np.max(subj_stages)+1.01))
    else:
        nb=50;
        freq,binc=np.histogram(subj_stages,bins=nb)
        
    freq = (1.*freq)/len(subj_stages)
    maxfreq=np.max(freq)
    
    idx_cn = np.where(Diagnosis==1); idx_cn = idx_cn[0]
    idx_ad = np.where(Diagnosis==np.max(Diagnosis)); idx_ad = idx_ad[0]
    idx_mci = np.where(np.logical_and(Diagnosis > 1, Diagnosis < np.max(Diagnosis))); idx_mci = idx_mci[0]
    
    if len(Labels)>np.max(Diagnosis):
        strc = '';
        for j in range(1, len(Labels)-1):
            strc = strc+Labels[j]+','
        for j in range(1, len(Labels)-1):
            del Labels[1]
        strc = strc[:-1]
        Labels.insert(1,strc)
        
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-whitegrid')
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('mathtext', fontset='stixsans');
    c = ['#4daf4a','#377eb8','#e41a1c']
    count=-1;
    freq_all=[]; maxfreq_ml = 0;
    if np.max(subj_stages)>1:
        for idx in [idx_cn,idx_mci,idx_ad]:
            if len(idx)>0:
                count=count+1;
            freq,binc=np.histogram(subj_stages[idx],bins=binc)
            freq = (1.*freq)/len(subj_stages)
            maxfreq_ml = np.max([np.max(freq),maxfreq_ml])
            bw=2/(nb)
            ax.bar(binc[:-1]+count*bw,freq,width=bw,color=c[count],label=Labels[count],zorder=3-count)
        ax.set_xlim([-bw,(count+1)*bw+np.max([1,np.max(subj_stages)])])
        ax.set_ylim([0,maxfreq_ml])
    else:
        for idx in [idx_cn,idx_mci,idx_ad]:
            if len(idx)>0:
                count=count+1;
            freq,binc=np.histogram(subj_stages[idx],bins=binc)
            freq = (1.*freq)/len(subj_stages)
            if count>0:
                freq=freq+freq_all[count-1]
            freq_all.append(freq)
            bw=1/(2.*nb)
            ax.bar(binc[:-1],freq,width=bw,color=c[count],label=Labels[count],zorder=3-count)
        ax.set_xlim([-bw,bw+np.max([1,np.max(subj_stages)])])
        ax.set_ylim([0,maxfreq])
    if np.max(subj_stages)<1:
        ax.set_xticks(np.arange(0,1.05,0.1))
    else:
        ax.set_xticks(np.arange(0,np.max(subj_stages)+0.05,1))
        
    ax.set_yticks(np.arange(0,maxfreq,0.1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Estimated Disease Stage',fontsize=16)
    ax.set_ylabel('Frequency of occurrences',fontsize=16)
    ax.legend(fontsize=16)
    plt.title('Patient Staging',fontsize=16)
    plt.show()
