# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
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

import os

from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.constants import BETA, SCIPION_DEBUG_NOCLEAN
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.convert.headers import getFileFormat, MRC
from pwem.emlib.image import ImageHandler, DT_FLOAT

from .program_ctffind import ProgramCtffind

from tomo.objects import CTFTomo
from tomo.protocols import ProtTsEstimateCTF


class CistemProtTsCtffind(ProtTsEstimateCTF):
    """ CTF estimation on a set of tilt series using CTFFIND4. """
    _label = 'tilt-series ctffind4'
    _devStatus = BETA

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL
        self.usePowerSpectra = False
        self.inputIsStack = True

    def _createFilenameTemplates(self):
        """ Centralize how files are called. """
        myDict = {'output_psd': self._getExtraPath('gain_estimate.bin'),
                  'output_ctf': self._getTmpPath(''),
                  'output_extra': self._getExtraPath()
                  }
        self._updateFilenamesDict(myDict)

    # -------------------------- DEFINE param functions -----------------------
    def _initialize(self):
        ProtTsEstimateCTF._initialize(self)
        self._ctfProgram = ProgramCtffind(self)
        self._createFilenameTemplates()

    def _defineProcessParams(self, form):
        form.addParam('recalculate', params.BooleanParam, default=False,
                      condition='recalculate',
                      label="Do recalculate ctf?")
        form.addParam('continueRun', params.PointerParam, allowsNull=True,
                      condition='recalculate', label="Input previous run",
                      pointerClass='ProtTsCtffind')
        form.addHidden('sqliteFile', params.FileParam,
                       condition='recalculate',
                       allowsNull=True)
        # ctffind resamples input mics automatically
        form.addHidden('ctfDownFactor', params.FloatParam, default=1.)
        ProgramCtffind.defineProcessParams(form)

    # --------------------------- STEPS functions -----------------------------
    def processTiltSeriesStep(self, tsId):
        """ Step called for a given tilt series. """
        tsFn = self._tsDict.getTiList(tsId)[0].getFileName()

        # Link input TS stack as mrcs
        workingDir = self._getTmpPath(tsId)
        pwutils.makePath(workingDir)
        tsFnMrc = os.path.join(workingDir, pwutils.replaceBaseExt(tsFn, 'mrcs'))

        self._convertInputTs(tsFn, tsFnMrc)
        self._estimateCtf(workingDir, tsFnMrc, tsId)
        self._parseOutput(workingDir, tsId, tsFnMrc)

        if not pwutils.envVarOn(SCIPION_DEBUG_NOCLEAN):
            pwutils.cleanPath(workingDir)

        self._tsDict.setFinished(tsId)

    def _convertInputTs(self, tsFn, tsFnMrc):
        if getFileFormat(tsFn) == MRC:
            pwutils.createAbsLink(os.path.abspath(tsFn), tsFnMrc)
        else:
            ih = ImageHandler()
            ih.convert(tsFn, tsFnMrc, DT_FLOAT)

    def _estimateCtf(self, workingDir, tsFn, tsId, *args):
        try:
            outputLog = os.path.join(workingDir, 'output-log.txt')
            outputPsd = os.path.join(workingDir, self.getPsdName(tsFn))

            program, args = self._ctfProgram.getCommand(
                micFn=tsFn,
                powerSpectraPix=None,
                ctffindOut=outputLog,
                ctffindPSD=outputPsd)

            self.runJob(program, args)

            # Move files we want to keep
            pwutils.makePath(self._getExtraPath(tsId))
            pwutils.moveFile(outputPsd, self._getExtraPath(tsId))
            pwutils.moveFile(outputPsd.replace('.mrcs', '_avrot.txt'),
                             self._getExtraPath(tsId))
        except:
            print("ERROR: Ctffind has failed for %s" % tsFn)

    def _parseOutput(self, workingDir, tsId, tsFnMrc):
        outputCtfs = os.path.join(workingDir, self.getCtfOutput(tsFnMrc))
        print(outputCtfs)


        raise Exception('DEBUG')

        for ti in self._tsDict.getTiList(tsId):
            ti.setCTF(self.getCtf(outputCtfs, ti))

    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        errors = []

        if self.lowRes.get() > 50:
            errors.append("Minimum resolution cannot be > 50A.")

        valueStep = round(self.stepPhaseShift.get(), 2)
        valueMin = round(self.minPhaseShift.get(), 2)
        valueMax = round(self.maxPhaseShift.get(), 2)

        if not (self.minPhaseShift < self.maxPhaseShift and
                valueStep <= (valueMax - valueMin) and
                0. <= valueMax <= 180.):
            errors.append('Wrong values for phase shift search.')

        return errors

    def _citations(self):
        return ["Mindell2003", "Rohou2015"]

    # --------------------------- UTILS functions -----------------------------
    def _doInsertTiltImageSteps(self):
        return False

    def getPsdName(self, tsFn):
        return pwutils.removeBaseExt(tsFn) + '_PSD.mrcs'

    def getCtfOutput(self, tsFn):
        return pwutils.removeBaseExt(tsFn) + '_PSD.txt'

    def getCtf(self, outputLog, ti):
        """ Parse the CTF object estimated for this Tilt-Image. """
        psd = self.getPsdName(tsFn)
        outCtf = self._getTmpPath(psd.replace('.mrc', '.txt'))
        ctfModel = self._ctfProgram.parseOutputAsCtf(outCtf,
                                                     psdFile=self._getExtraPath(ti.getTsId(), psd))
        ctfTomo = CTFTomo.ctfModelToCtfTomo(ctfModel)

        ti.setCTF(ctfTomo)
