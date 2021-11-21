# **************************************************************************
# *
# *  Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from pwem.protocols import ProtClassify3D
from pyworkflow.constants import BETA

from cistem import Plugin


class CistemProt3DClassification(ProtClassify3D):
    """ Classify particles into 3d classes """
    _label = '3d classification'
    _devStatus = BETA

    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        # form.addParam()
        pass

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions -----------------------------
    def firstStep(self):
        pass

    def createOutputStep(self):
        pass

    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        pass

    def _summary(self):
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions -----------------------------
