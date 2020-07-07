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

from pyebm.inputs_outputs import prepare_outputs as po
import numpy 
import copy
from math import exp

def adjswap(pi,i):
	pic = pi[:]
	(pic[i],pic[i+1])=(pic[i+1],pic[i])
	return pic; 

class weighted_mallows:
    def __init__(self, phi, sigma0):
        self.phi = phi[:]
        self.sigma0 = sigma0[:]

    @classmethod 
    def fitMallows(h,p_yes,mixing_params):
          p_yes_padded = numpy.concatenate((numpy.zeros((p_yes.shape[0],1)),p_yes,numpy.ones((p_yes.shape[0],1))),axis=1)
          so_list,weights_list = po.Prob2ListAndWeights(p_yes_padded);    
          num_events = len(mixing_params)
          pi0_init = 1+numpy.argsort(mixing_params);
          pi0_init=numpy.insert(pi0_init,0,num_events+1)
          pi0_init=numpy.append(pi0_init,0)
          pi0,bestscore,scores,indv_heterogeneity,indv_distance = weighted_mallows.consensus(num_events+2,so_list,weights_list,pi0_init)
          ordering_distances=scores - bestscore
          event_centers = numpy.cumsum(ordering_distances)
          event_centers = event_centers/numpy.max(event_centers)

          so_list,weights_list = po.Prob2ListAndWeights(p_yes_padded); 
          sig0=weighted_mallows.fitPhi(pi0,so_list,weights_list)
          
          indv_mahalanobis=numpy.divide(indv_distance,sig0)
          indv_heterogeneity_alt=numpy.nansum(indv_mahalanobis,axis=1)
          idx0=pi0.index(0);
          del pi0[idx0]
          idx_last=pi0.index(num_events+1);
          del pi0[idx_last]
          pi0[:] = [int(x - 1) for x in pi0];
          event_centers = event_centers[:-1]
          return pi0,event_centers,indv_heterogeneity
          
    @classmethod
    def consensus(h,n,D,prob,pi0_init,flag_only_init=None):
        if flag_only_init==None:
            flag_only_init=0
        maxSeqEvals = 10000
        sig0=list(pi0_init);
        bestscore,indv_heterogeneity,indv_distance = weighted_mallows.totalconsensus(sig0,D,prob)
        if flag_only_init==1:
            scores=[]
            return sig0,bestscore,scores,indv_heterogeneity,indv_distance
        sig_list=(maxSeqEvals+n-1)*[0]; count = 0 ;
        sig_list[count]=sig0; 
        while True:      
            scores = (n-1)*[0] 
            indv_heterogeneity_all = []
            indv_distance_all = []
            for i in range(n-1):
                count=count+1;
                sig = adjswap(sig0,i)
                sig_list[count]=sig
                scores[i],ih,ihi = weighted_mallows.totalconsensus(sig,D,prob)
                indv_heterogeneity_all.append(ih)
                indv_distance_all.append(ihi)
            bestidx = scores.index(min(scores))
            bestsig = adjswap(sig0,bestidx)
            if bestscore > scores[bestidx]:
                sig0 = bestsig[:]
                bestscore = scores[bestidx]
                indv_heterogeneity = indv_heterogeneity_all[bestidx]
                indv_distance = indv_distance_all[bestidx]
            if bestscore <= scores[bestidx] or count>=maxSeqEvals:
                break
        return sig0,bestscore,scores,indv_heterogeneity,indv_distance

    @classmethod
    def fitPhi(h,pi0,D,weights_list):
        n = len(pi0)
        theta = numpy.array((n-1)*[0.0])
        Vbar = numpy.array((n-1)*[0.0])
        cnt=-1
        for x in D:
            cnt=cnt+1
            w=copy.copy(weights_list[cnt])
            Vbar += numpy.array(weighted_mallows.__Ikendall(pi0,x,w))
       # m = sum([D[x] for x in D])
        Vbar /= cnt
        for j in range(n-1):
            theta[j] = weighted_mallows.__solveFVeqn(Vbar[j],n,j+1) 
			#print(FVeqn(Vbar[j],n,j+1,phi[j]))
        return list(numpy.exp(-theta))

    @classmethod
    def __solveFVeqn(h,Vbar,n,j):
        (lb,ub) = (-1.0,1.0)
        fub = weighted_mallows.__FVeqn(Vbar,n,j,ub)
        flb = weighted_mallows.__FVeqn(Vbar,n,j,lb)
        while fub > 0.0:
            ub *= 2
            fub = weighted_mallows.__FVeqn(Vbar,n,j,ub)
        while flb < 0.0:
            lb *= 2
            flb = weighted_mallows.__FVeqn(Vbar,n,j,lb)
        MAXITER = 20; TOL = 1e-4; 
        itnum = 0
        while itnum < MAXITER:
            midpt = (ub+lb)/2.0
            fmid = weighted_mallows.__FVeqn(Vbar,n,j,midpt)
            if fmid == 0 or abs(ub-lb)/2.0<TOL:
                return midpt
            itnum += 1
            if fmid*fub > 0:
                ub = midpt; fub = fmid
            else:	
                lb = midpt
        return midpt
		
    @classmethod
    def __FVeqn(h,Vbar,n,j,theta):
        if theta == 0:
            return .5*(n-j)-Vbar
        if n*theta > 200:
            return -Vbar
        return 1.0/(exp(theta)-1)-(n-j+1)/(exp((n-j+1)*theta)-1)-Vbar            

    @classmethod
    def totalconsensus(h,pi0,D,prob):    
        score = (len(D))*[0]
        score_indv = numpy.zeros((len(D),len(pi0)-1)) + numpy.nan
        for i in range(len(D)):
            s=copy.copy(D[i]);
            p=copy.copy(prob[i]);
            pi0c=copy.copy(pi0);
            pi0c,p_new=weighted_mallows.__removeAbsentEvents(pi0c,s,p); # for NaN events and non-events
            si = weighted_mallows.__Ikendall(pi0c,s,p_new);
            idx = list(numpy.sort(pi0c))
            score_indv[i,idx[:-1]]=si
            score[i]= numpy.nansum(score_indv[i,:])
        tscore=numpy.mean(score)
        return tscore,score,score_indv


    @classmethod
    def __removeAbsentEvents(h,pi0c,seq,p):
        pi0c_new=[];
        removed=[]
        p_new=p;  
        #p_new = [x if x>0.5 else 0 for x in p]  
        for j in range(len(pi0c)):
            e=pi0c[j];
            if weighted_mallows.__find(seq,e) !=-1 :
                pi0c_new.append(e);
        return pi0c_new,p_new

    # pi0 and pi1 are assumed to be written in inverse notation
    @classmethod
    def __kendall(h,Ordering1,Ordering2,p):
        n=len(Ordering1);
        weighted_distance = numpy.array((n-1)*[0.0]); 
        for i in range(0,n-1):    
            e1=Ordering1[i];
            idx_e2=weighted_mallows.__find(Ordering2,e1);
            if idx_e2>i:     
                Ordering2.insert(i, Ordering2[idx_e2]);
                Ordering2 = Ordering2[:idx_e2+1] + Ordering2[idx_e2+2 :];
                pn=numpy.asarray(p);
                #wd=sum(pn[i]-pn[i+1:idx_e2+1]); 
                dp=pn[i:idx_e2]-pn[idx_e2]
                wd=sum(dp);
                weighted_distance[i]=wd;
                p.insert(i, p[idx_e2]);
                p = p[:idx_e2+1] + p[idx_e2+2 :];
        #denom = n*(n-1)/2;
        return sum(weighted_distance)
    
    @classmethod
    def __Ikendall(h,Ordering1,Ordering2,p):
        n=len(Ordering1);
        weighted_distance = numpy.array((n-1)*[0.0]); 
        for i in range(0,n-1):    
            e1=Ordering1[i];
            idx_e2=weighted_mallows.__find(Ordering2,e1);
            if idx_e2>i:     
                Ordering2.insert(i, Ordering2[idx_e2]);
                Ordering2 = Ordering2[:idx_e2+1] + Ordering2[idx_e2+2 :];
                pn=numpy.asarray(p);
                #wd=sum(pn[i]-pn[i+1:idx_e2+1]); 
                dp=pn[i:idx_e2]-pn[idx_e2]
                wd=sum(dp);
                weighted_distance[i]=wd;
                p.insert(i, p[idx_e2]);
                p = p[:idx_e2+1] + p[idx_e2+2 :];
        #denom = n*(n-1)/2;
        return weighted_distance
  
    @classmethod
    def __find(h,pi,val):
        n=len(pi);
        for i in range(0,n):
            if pi[i]==val:
                return i                
        return -1
          