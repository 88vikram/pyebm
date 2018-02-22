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

def Prob2ListAndWeights(p_yes):
    import numpy as np
    SubjectwiseWeights = [];
    SubjectwiseOrdering = []
    for i in range(0,np.shape(p_yes)[0]):
        weights_reverse=np.sort(p_yes[i,:])
        valid_indices=np.logical_not(np.isnan(weights_reverse))
        weights_reverse = weights_reverse[valid_indices]
        SubjectwiseWeights.append(weights_reverse[::-1].tolist());
        ordering_reverse=np.argsort(p_yes[i,:]);
        ordering_reverse = ordering_reverse[valid_indices].astype(int)
        SubjectwiseOrdering.append(ordering_reverse[::-1].tolist());
    
    return SubjectwiseOrdering,SubjectwiseWeights

def perminv(sigma):
	result = sigma[:];
	for i in range(len(sigma)):
		result[sigma[i]] = i;
	return result;

def adjswap(pi,i):
	pic = pi[:]
	(pic[i],pic[i+1])=(pic[i+1],pic[i])
	return pic;

def pdReadData(str_data,flag_JointFit=False,Labels=['CN','MCI','AD']):
    import pandas as pd
    import numpy as np
    
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

def CorrectConfounders(DataTrain,DataTest,Factors=['Age','Sex','ICV'],flag_correct=1):
    import statsmodels.formula.api as sm
    import numpy as np
    import pandas as pd
    flag_test=1;
    droplist = ['PTID','Diagnosis','EXAMDATE']
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
                Data=Data.assign(Sex=pd.Series(sex))
                if count==0:
                    DataTrain = Data.copy()
                else:
                    DataTest = Data.copy()
        
        ## Separate the list of biomarkers from confounders and meta data
        count=-1;
        for Data in [DataTrain,DataTest]:
            count=count+1;
            DataBiomarkers=Data
            DataBiomarkers=DataBiomarkers.drop(Factors,axis=1)
            H = list(DataBiomarkers)
            for j in droplist:
                if any(j in f for f in H):
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
                    result = sm.ols(formula=str_formula, data=Data).fit()
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
            
            for i in range(len(BiomarkersList)):
                Data=Data.rename(columns={BiomarkersListnew[i]:BiomarkersList[i]})
            if count==0:
                DataTrain = Data.copy()
            else:
                DataTest = Data.copy()
                
    if flag_test==0:
        DataTest=[]
    return DataTrain,DataTest,BiomarkersList

def pd2mat(pdData,BiomarkersList,flag_JointFit):
    # Convert arrays from pandas dataframe format to the matrices (which are used in DEBM algorithms)
    import numpy as np
    num_events = len(BiomarkersList);
    if flag_JointFit==0:
        num_feats = 1
    num_subjects=pdData.shape[0]
    matData = np.zeros((num_subjects,num_events,num_feats))
    for i in range(num_events):
        matData[:,i,0]=pdData[BiomarkersList[i]].values
    return matData

def ExamDate_str2num(ExamDateSeries):
    import datetime
    import time
    import pandas as pd
    import numpy as np
    timestamp = np.zeros(len(ExamDateSeries))
    for i in range(len(ExamDateSeries)):
        stre=ExamDateSeries.values[i]
        if len(stre)>5:
            timestamp[i]=time.mktime(datetime.datetime.strptime(stre, "%Y-%m-%d").timetuple())
        else:
            timestamp[i]=np.nan
    TimestampSeries=pd.Series(timestamp)
    return TimestampSeries

def VisualizeBiomarkerDistribution(Data_all,params_all,BiomarkersList):

    from matplotlib import pyplot as plt
    import numpy as np
    import scipy.stats 
    
    m=np.shape(Data_all)
    n=len(params_all)
    fig, ax = plt.subplots(int(round(1+m[1]/3)), 3, figsize=(13, 4*(1+m[1]/3)))
    for i in range(m[1]):
            Dalli=Data_all[:,i,0];
            valid_data=np.logical_not(np.isnan(Dalli))
            Dallis=Dalli[valid_data].reshape(-1,1); Dallis=Dallis[:,0];
            x_grid=np.linspace(np.min(Dallis),np.max(Dallis),1000)
            for j in range(n):
                i1 = int(i/3);
                j1 = np.remainder(i,3)
                paramsij=params_all[j][i,:];
                norm_pre=scipy.stats.norm(loc=paramsij[0], scale=paramsij[1]);
                norm_post=scipy.stats.norm(loc=paramsij[2],scale=paramsij[3]);
                h=np.histogram(Dallis,50)                          
                maxh=np.nanmax(h[0])
                ax[i1,j1].hist(Dallis,50, fc='blue', histtype='stepfilled', alpha=0.3, normed=False)
                #ylim=ax[i,j].get_ylim()
                likeli_pre=norm_pre.pdf(x_grid)*(paramsij[4]);
                likeli_post=norm_post.pdf(x_grid)*(1-paramsij[4]);
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
                    scale_range=np.arange(1,(1/max_scaling)+1,max_scaling/1000.)
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
    
def VisualizeOrdering(labels, pi0_all,pi0_mean, plotorder):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
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
    heatmap = sns.heatmap(datapivot, cmap = 'binary', xticklabels=xticks, vmin=0, vmax=len(pi0_all))
    fig = heatmap.get_figure()
    plt.title('Positional variance diagram of the central ordering')
    plt.yticks(rotation=0) 
    plt.show()
    
    
def VisualizeStaging(subj_stages,Diagnosis,Labels):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    if np.max(subj_stages)>1:
        nb=np.max(subj_stages)+2
        freq,binc=np.histogram(subj_stages,bins=np.arange(np.max(subj_stages)+1.01))
    else:
        nb=50;
        freq,binc=np.histogram(subj_stages,bins=nb)
        
    freq = (1.*freq)/len(subj_stages)
    maxfreq=np.max(freq)
    
    idx_cn = np.where(Diagnosis==1); idx_cn = idx_cn[0]
    idx_ad = np.where(Diagnosis==len(Labels)); idx_ad = idx_ad[0]
    idx_mci = np.where(np.logical_and(Diagnosis > 1, Diagnosis < len(Labels))); idx_mci = idx_mci[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-whitegrid')
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('mathtext', fontset='stixsans');
    c = ['#4daf4a','#377eb8','#e41a1c']
    count=-1;
    freq_all=[]
    for idx in [idx_cn,idx_mci,idx_ad]:
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
        ax.set_xticklabels(np.arange(0,1.05,0.1),fontsize=14)
    else:
        ax.set_xticks(np.arange(0,np.max(subj_stages)+0.05,1))
        ax.set_xticklabels(np.arange(0,np.max(subj_stages)+0.05,1),fontsize=14)
        
    ax.set_yticks(np.arange(0,maxfreq,0.1))
    ax.set_yticklabels(np.arange(0,maxfreq,0.1),fontsize=14)
    ax.set_xlabel('Estimated Disease State',fontsize=16)
    ax.set_ylabel('Frequency of occurrences',fontsize=16)
    ax.legend(fontsize=16)
    plt.title('Patient Staging',fontsize=16)
    plt.show()
        