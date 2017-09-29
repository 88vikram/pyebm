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

## An example function call

import DEBM
pi0_mean,pi0_all,params_opt_all,BiomarkersList,event_centers_all,p_yes_all=DEBM.Control('ADNI_7.csv')
print [BiomarkersList[x] for x in pi0_mean]

import EBM
pi0_mean,pi0_all,params_opt_all,BiomarkersList,event_centers_all=EBM.Control('ADNI_7.csv')
print [BiomarkersList[x] for x in pi0_mean]

## Another Example with Visual Biomarker Distributions as output

from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap')
MO.Bootstrap=0; MO.MixtureModel='vv1';
VO = namedtuple('VerboseOptions','Distributions')
VO.Distributions=1; 
pi0_mean,pi0_all,params_opt_all,BiomarkersList,event_centers_all,p_yes_all=DEBM.Control('ADNI_7.csv',MethodOptions=MO,VerboseOptions=VO)

## Another Example with bootstrapping and visual output
from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap')
MO.Bootstrap=100; MO.MixtureModel='vv1';
VO = namedtuple('VerboseOptions','Ordering PlotOrder Distributions')
VO.Ordering=1; VO.PlotOrder=1; VO.Distributions=0; 
pi0_mean,pi0_all,params_opt_all,BiomarkersList,event_centers_all,p_yes_all=DEBM.Control('ADNI_7.csv',MethodOptions=MO,VerboseOptions=VO)

print [BiomarkersList[x] for x in pi0_mean]
