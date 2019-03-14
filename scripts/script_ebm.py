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

## Simplest function call (with default parameters)
from __future__ import print_function

import os
from pyebm import ebm

ModelOutput,SubjTrain,SubjTest=ebm.fit('./resources/Data_7.csv')
print([ModelOutput.BiomarkerList[x] for x in ModelOutput.MeanCentralOrdering])

## Example with Visual Biomarker Distributions as output

from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap')
MO.Bootstrap=0; MO.MixtureModel='GMMvv2';
VO = namedtuple('VerboseOptions','Distributions')
VO.Distributions=1; 
ModelOutput,SubjTrain,SubjTest=ebm.fit('./resources/Data_7.csv',MethodOptions=MO,VerboseOptions=VO)

## Example with bootstrapping and visual output

from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap')
MO.Bootstrap=5; MO.MixtureModel='GMMvv2';
VO = namedtuple('VerboseOptions','Ordering PlotOrder Distributions')
VO.Ordering=1; VO.PlotOrder=1; VO.Distributions=0; 
ModelOutput,SubjTrain,SubjTest=ebm.fit('./resources/Data_7.csv',MethodOptions=MO,VerboseOptions=VO)

print([ModelOutput.BiomarkerList[x] for x in ModelOutput.MeanCentralOrdering])

## Example with Patient Staging and visual output. Also, a pandas dataframe can be sent as an input instead of CSV
from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap PatientStaging')
MO.Bootstrap=0; MO.MixtureModel='GMMvv2'; MO.PatientStaging=['ml','l']
VO = namedtuple('VerboseOptions','Distributions PatientStaging')
VO.PatientStaging=1; VO.Distributions=0; 
import pandas as pd
D=pd.read_csv('./resources/Data_7.csv')
ModelOutput,SubjTrain,SubjTest=ebm.fit(D,MethodOptions=MO,VerboseOptions=VO)


from collections import namedtuple
MO = namedtuple('MethodOptions','MixtureModel Bootstrap PatientStaging')
MO.Bootstrap=0; MO.MixtureModel='GMMvv2'; MO.PatientStaging=['exp','p']
VO = namedtuple('VerboseOptions','Distributions PatientStaging')
VO.PatientStaging=1; VO.Distributions=0; 
import pandas as pd
D=pd.read_csv('./resources/Data_7.csv')
ModelOutput,SubjTrain,SubjTest=ebm.fit(D,MethodOptions=MO,VerboseOptions=VO)
