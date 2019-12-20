# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .pig20 import Pig20Dataset as pig20
from .atrw import ATRWDataset as atrw
from .pig15 import PIGDataset as pig15
from .pig17 import PIG17Dataset as pig17
from .pig_univ import PigUnivDataset as pig_univ 
from .atrw_univ import ATRWUnivDataset as atrw_univ
from .iccv2019_univ import ICCV2019UnivDataset as iccv2019_univ