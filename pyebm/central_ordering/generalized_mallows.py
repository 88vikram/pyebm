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
          pi0,bestscore,scores = weighted_mallows.consensus(num_events+2,so_list,weights_list,pi0_init)
          ordering_distances=scores - bestscore
          event_centers = numpy.cumsum(ordering_distances)
          event_centers = event_centers/numpy.max(event_centers)
          idx0=pi0.index(0);
          del pi0[idx0]
          idx_last=pi0.index(num_events+1);
          del pi0[idx_last]
          pi0[:] = [int(x - 1) for x in pi0];
          event_centers = event_centers[:-1]
          return pi0,event_centers
          
    # here D is assumed to be in inverse form (vertical bar form)
    # and represented as a dictionary indexed by tuples
    @classmethod
    def consensus(h,n,D,prob,pi0_init):
        maxSeqEvals = 10000
        sig0=list(pi0_init);
        bestscore = weighted_mallows.totalconsensus(sig0,D,prob)
        sig_list=(maxSeqEvals+n-1)*[0]; count = 0 ;
        sig_list[count]=sig0; 
        while True:      
            scores = (n-1)*[0] 
            for i in range(n-1):
                count=count+1;
                sig = adjswap(sig0,i)
                sig_list[count]=sig
                scores[i] = weighted_mallows.totalconsensus(sig,D,prob)
            bestidx = scores.index(min(scores))
            bestsig = adjswap(sig0,bestidx)
            if bestscore > scores[bestidx]:
                sig0 = bestsig[:]
                bestscore = scores[bestidx] 
            if bestscore <= scores[bestidx] or count>=maxSeqEvals:
                break
        return sig0,bestscore,scores
            

    @classmethod
    def totalconsensus(h,pi0,D,prob):    
        score = (len(D))*[0]
        for i in range(0,len(D)):
            s=copy.copy(D[i]);
            p=copy.copy(prob[i]);
            pi0c=copy.copy(pi0);
            pi0c,p_new=weighted_mallows.__removeAbsentEvents(pi0c,s,p); # for NaN events and non-events
            score[i]= weighted_mallows.__kendall(pi0c,s,p);
        
        tscore=numpy.mean(score)
        return tscore


    @classmethod
    def __removeAbsentEvents(h,pi0c,seq,p):
        pi0c_new=[];
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
    def __find(h,pi,val):
        n=len(pi);
        for i in range(0,n):
            if pi[i]==val:
                return i                
        return -1
          
    # pis here is assumed to be written in inverse notation
    # this is the insertion code (the interleaving at each stage of the mallows model)