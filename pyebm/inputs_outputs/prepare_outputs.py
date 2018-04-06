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

def Prob2ListAndWeights(p_yes):
    
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
