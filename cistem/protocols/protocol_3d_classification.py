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

from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import EnumParam, MultiPointerParam, PointerParam, FloatParam, IntParam, BooleanParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pwem.protocols import ProtClassify3D
from pyworkflow.constants import BETA

class CistemProt3DClassification(ProtClassify3D):
    """ Classify particles into 3d classes """
    _label = '3d classification'
    _devStatus = BETA

    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        myDict = {
            'parameters': 'c3d.par',
        }
        self._updateFilenamesDict(myDict)
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        # TODO add help
        form.addSection(label='Input')
        form.addParam('input_particles', PointerParam, pointerClass='SetOfParticles', label='Input particles',
                        help='Particles to be classified')
        form.addParam('input_initialVolumes', MultiPointerParam, pointerClass='Volume', label='Initial volumes',
                        help='Initial volumes that serve as a reference for the classification. '
                        'Classification will output a disctinct class for each of these volumes')


        form.addSection(label='Refinement')
        form.addParam('cycleCount', IntParam, label='Cycle Count',
                        help='Number of refinement cycles to be executed',
                        default=1, validators=[GE(1)])
        form.addParam('refinement_type', EnumParam, choices=['Local', 'Global'], label='Refinement type',
                        default=0)
        group = form.addGroup('Refinement parameters', help='Parameters to be refined',
                                expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_psi', BooleanParam, label='ψ',
                        default=True, expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_theta', BooleanParam, label='θ',
                        default=True, expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_phi', BooleanParam, label='φ',
                        default=True, expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_xShift', BooleanParam, label='X shift',
                        default=True, expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_yShift', BooleanParam, label='Y shift',
                        default=True, expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_lowResLimit', FloatParam, label='Low resolution limit (Å)',
                        default=225.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_outerMaskRadius', FloatParam, label='Outer mask radius (Å)',
                        default=97.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_innerMaskRadius', FloatParam, label='Inner mask radius (Å)',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_signedCcResLimit', FloatParam, label='Signed CC resolution limit (Å)',
                        default=0.0, expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_usedPercentage', FloatParam, label='Percentage used (%)',
                        default=100.0, validators=[Range(0, 100)], expertLevel=LEVEL_ADVANCED)
        group = form.addGroup('Global search', expertLevel=LEVEL_ADVANCED,
                                condition='refinement_type == 1')
        group.addParam('refine_globalMaskRadius', FloatParam, label='Global mask radius (Å)',
                        default=120.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_numResults', IntParam, label='Number of results to refine',
                        default=20, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_refineInputParameters', BooleanParam, label='Refine input parameters',
                        default=True, expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_angularStep', FloatParam, label='Angular search step (º)',
                        default=35.26, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_xRange', FloatParam, label='Search range in X (Å)',
                        default=22.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_yRange', FloatParam, label='Search range in Y (Å)',
                        default=22.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)


        form.addSection(label='Classification')
        form.addParam('classification_hiresLimit', FloatParam, label='High Resolution Limit (Å)',
                        default=30.0, validators=[GT(0)])
        form.addParam('classification_enableFocus', BooleanParam, label='Enable focused classification',
                        default=False, expertLevel=LEVEL_ADVANCED)
        group = form.addGroup('Focus sphere', 
                                condition='classification_enableFocus is True',
                                help='Sphere on which the classification will be focussed', 
                                expertLevel=LEVEL_ADVANCED)
        group.addParam('classification_sphereX', FloatParam, label='Center X (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED)
        group.addParam('classification_sphereY', FloatParam, label='Center Y (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED)
        group.addParam('classification_sphereZ', FloatParam, label='Center Z (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED)
        group.addParam('classification_radius', FloatParam, label='Radius (Å)',
                        default=10.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)


        form.addSection(label='CTF refinement')
        form.addParam('ctf_enable', BooleanParam, label='Refine CTF',
                        default=False, expertLevel=LEVEL_ADVANCED)
        form.addParam('ctf_range', FloatParam, label='Defocus search range (Å)',
                        condition='ctf_enable is True',
                        default=500.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('ctf_step', FloatParam, label='Defocus search step (Å)',
                        condition='ctf_enable is True',
                        default=50.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)


        form.addSection(label='Reconstruction')
        form.addParam('reconstruction_score2weight', FloatParam, label='Score to weight constant (Å²)',
                        default=2.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_adjustScore4Defocus', BooleanParam, label='Adjust score for defocus',
                        default=True, expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_scoreThreshold', FloatParam, label='Score threshold',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_resLimit', FloatParam, label='Resolution limit (Å²)',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_enableAutoCrop', BooleanParam, label='Enable auto cropping images',
                        default=False, expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_enableLikelihoodBlurring', BooleanParam, label='Enable likelihood blurring',
                        default=False, expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_smoothingFactor', FloatParam, label='Smoothing factor',
                        condition='reconstruction_enableLikelihoodBlurring is True',
                        default=10.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)


        form.addSection(label='Masking')
        form.addParam('masking_enableAuto', BooleanParam, label='Use auto masking',
                        default=True)
        form.addParam('masking_volume', PointerParam, pointerClass='VolumeMask', label='Mask',
                        help='3D mask used to crop the output volume',
                        condition='masking_enableAuto is False', default=None)
        form.addParam('masking_edgeWidth', FloatParam, label='Edge width (Å)',
                        condition='masking_volume is not None', 
                        default=10.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('masking_outsideWeight', FloatParam, label='Outside weight',
                        condition='masking_volume is not None', 
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('masking_enableLpfOutsideMask', BooleanParam, label='Use a LPF outside the mask',
                        condition='masking_volume is not None', 
                        default=False, expertLevel=LEVEL_ADVANCED)
        form.addParam('masking_lpfResolution', FloatParam, label='LPF resolution (Å)',
                        condition='masking_enableLpfOutsideMask is True', 
                        default=20.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # Shorthands for variables
        nCycles = self.cycleCount.get()
        
        # Perform refine, reconstruct and merge steps repeatedly
        for i in range(nCycles):
            self._insertFunctionStep('refineStep', i)
            self._insertFunctionStep('reconstructStep', i)
            self._insertFunctionStep('mergeStep', i)

        # Generate the output
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -----------------------------
    def refineStep(self, i):
        pass

    def reconstructStep(self, i):
        pass

    def mergeStep(self, i):
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
