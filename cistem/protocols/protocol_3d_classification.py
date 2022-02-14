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

from pwem.constants import ALIGN_PROJ
from pwem.emlib.image.image_handler import ImageHandler
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import EnumParam, MultiPointerParam, PointerParam, FloatParam, IntParam, BooleanParam, StringParam, ProtocolClassParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pyworkflow.constants import BETA
from pyworkflow.utils.path import copyFile, makePath, createLink, cleanPattern, moveFile

from pwem.protocols import ProtClassify3D
from pwem.objects.data import CTFModel, Transform

from cistem import Plugin
from cistem.convert import FullFrealignParFile, FrealignStatisticsFile, matrixFromGeometry, boolToYN

import math
import numpy as np

class CistemProt3DClassification(ProtClassify3D):
    """ Classify particles into 3d classes """
    _label = '3d classification'
    _devStatus = BETA

    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        classFmt='c%(cls)02d'
        iterFmt='i%(iter)02d'
        jobFmt='j%(job)02d'
        reconstructionNameFmt='%(rec)s'
        parityFmt='%(par)s'
        seedFmt='%(seed)s'

        myDict = {
            'input_particles': f'Inputs/input_particles.mrcs',
            'input_parameters': f'Inputs/input_parameters_{classFmt}.par',
            'input_volume': f'Inputs/input_volume_{classFmt}.mrc',
            'input_statistics': f'Inputs/input_statistics_{classFmt}.txt',
            'refine3d_input_parameters': f'Refine3D/Parameters/input_parameters_{iterFmt}_{classFmt}.par',
            'refine3d_input_reconstruction': f'Refine3D/Reconstructions/input_reconstruction_{iterFmt}_{classFmt}.mrc',
            'refine3d_input_statistics': f'Refine3D/Statistics/input_statistics_{iterFmt}_{classFmt}.txt',
            'refine3d_output_matching_projections': f'Refine3D/Projections/output_matching_projections_{iterFmt}_{classFmt}_{jobFmt}.mrc',
            'refine3d_output_parameters': f'Refine3D/Parameters/output_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'refine3d_output_shifts': f'Refine3D/Parameters/output_shifts_{iterFmt}_{classFmt}_{jobFmt}.par',
            'classify_output_parameters': f'Classify/Parameters/classify_output_{iterFmt}.par',
            'classify_output_class_parameters': f'Classify/Parameters/output_class_parameters_{iterFmt}_{classFmt}.par',
            'reconstruct3d_input_parameters': f'Reconstruct3D/Parameters/input_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'reconstruct3d_input_reconstruction': f'Reconstruct3D/Reconstructions/input_reconstruction_{iterFmt}_{classFmt}.mrc',
            'reconstruct3d_output_reconstruction': f'Reconstruct3D/Reconstructions/output_reconstruction_{iterFmt}_{classFmt}_{reconstructionNameFmt}_{jobFmt}.mrc',
            'reconstruct3d_output_statistics': f'Reconstruct3D/Statistics/output_statistics_{iterFmt}_{classFmt}_{jobFmt}.txt',
            'reconstruct3d_output_dump': f'Reconstruct3D/Dumps/output_dump_{iterFmt}_{classFmt}_{parityFmt}_{jobFmt}.dmp',
            'merge3d_input_dump': f'Merge3D/Dumps/input_dump_{parityFmt}_{seedFmt}.dmp',
            'merge3d_output_reconstruction': f'Merge3D/Reconstructions/output_reconstruction_{iterFmt}_{classFmt}_{reconstructionNameFmt}.mrc',
            'merge3d_output_statistics': f'Merge3D/Statistics/output_resolution_statistics_{iterFmt}_{classFmt}.txt',
            'output_volume': f'Reconstructions/output_volume_{iterFmt}_{classFmt}.mrc',
            'output_statistics': f'Statistics/output_statistics_{iterFmt}_{classFmt}.txt',
            'output_classification': f'Classifications/output_classification_{iterFmt}.par',
            'output_refinement': f'Refinements/output_refinement_{iterFmt}_{classFmt}.par' 
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('input_referenceRefinement', PointerParam, pointerClass='CistemProt3DClassification', label='Reference refinement',
                        allowsNull=True,
                        help='Local refinement requires a previous refinement as a reference for angular assignment. This field is '
                        'used to specify that previous refinement. When this field is not specified and local refinement is used, '
                        'the angular assignment contained by the particles themselves will be used for all classes')
        form.addParam('input_referenceIteration', IntParam, label='Reference refinement iteration',
                        default=-1, condition='input_referenceRefinement is not None',
                        help='Iteration from the reference refinement used as a reference. 0 is first, 1 second and so on. '
                        'Negative values can be used to reference backwards, -1 referring to the last one, -2 to penultimate '
                        'and so on')
        form.addParam('input_particles', PointerParam, pointerClass='SetOfParticles', label='Input particles',
                        allowsNull=True, condition='input_referenceRefinement is None',
                        help='Particles to be classified')
        form.addParam('input_initialVolumes', MultiPointerParam, pointerClass='Volume', label='Initial volumes',
                        allowsNull=True, condition='input_referenceRefinement is None',
                        help='Initial volumes that serve as a reference for the classification. '
                        'Classification will output a disctinct class for each of these volumes')
        form.addParam('input_molecularMass', FloatParam, label='Molecular mass (kDa)',
                        default=100.0, validators=[GT(0)],
                        help='Estimated molecular mass of the proteins. In general this should be '
                        'the molecular mass of the coherent parts (e.g. micelle would not be '
                        'included)')
        form.addParam('input_isWhite', BooleanParam, label='Is white protein',
                        default=True, 
                        help='Specifies if the input particles are in \'negative\'. '
                        'By default Scipion uses white proteins')

        form.addSection(label='Refinement')
        form.addParam('cycleCount', IntParam, label='Cycle Count',
                        default=1, validators=[GE(1)],
                        help='The number of refinement cycles to run. For a global search, '
                        'one is usually sufficient, possibly followed by another one at a '
                        'later stage in the refinement if the user suspects that the initial '
                        'reference was limited in quality such that a significant number of '
                        'particles were misaligned. For local refinement of a single class, '
                        'typically 3 to 5 cycles are sufficient, possibly followed by another '
                        'local refinement at increased resolution (see below). If multiple '
                        'classes are refined, between 30 and 50 cycles should be run to ensure '
                        'convergence of the classes.')
        form.addParam('refine_type', EnumParam, choices=['Local', 'Global'], label='Refinement type',
                        default=0,
                        help='If no starting parameters from a previous refinement are available, '
                        'they have to be determined in a global search (slow); '
                        'otherwise it is usually sufficient to perform local refinement (fast).')
        form.addParam('refine_symmetry', StringParam, label='Symmetry',
                        default='c1',
                        help='Symmetry of the proteins to be refined')
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
                        default=225.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='The data used for refinement is usually bandpass-limited '
                        'to exclude spurious low-resolution features in the particle '
                        'background (set by the low-resolution limit) and high-resolution '
                        'noise (set by the high-resolution limit). It is good practice '
                        'to set the low-resolution limit to 2.5x the approximate '
                        'particle mask radius. The high-resolution limit should remain '
                        'significantly below the resolution of the reference used for '
                        'refinement to enable unbiased resolution estimation using the '
                        'Fourier Shell Correlation curve.')
        form.addParam('refine_highResLimit', FloatParam, label='High resolution limit (Å)',
                        default=8.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='The data used for refinement is usually bandpass-limited '
                        'to exclude spurious low-resolution features in the particle '
                        'background (set by the low-resolution limit) and high-resolution '
                        'noise (set by the high-resolution limit). It is good practice '
                        'to set the low-resolution limit to 2.5x the approximate '
                        'particle mask radius. The high-resolution limit should remain '
                        'significantly below the resolution of the reference used for '
                        'refinement to enable unbiased resolution estimation using the '
                        'Fourier Shell Correlation curve.')
        form.addParam('refine_outerMaskRadius', FloatParam, label='Outer mask radius (Å)',
                        default=97.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The radius of the circular mask applied to the input images '
                        'before refinement starts. This mask should be sufficiently large '
                        'to include the largest dimension of the particle. When a global '
                        'search is performed, the radius should be set to include the expected '
                        'area containing the particle. This area is usually larger than '
                        'the area defined by the largest dimension of the particle because '
                        'particles may not be precisely centered.')
        form.addParam('refine_innerMaskRadius', FloatParam, label='Inner mask radius (Å)',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The radius of the circular mask applied to the input images '
                        'before refinement starts. This mask should be sufficiently large '
                        'to include the largest dimension of the particle. When a global '
                        'search is performed, the radius should be set to include the expected '
                        'area containing the particle. This area is usually larger than '
                        'the area defined by the largest dimension of the particle because '
                        'particles may not be precisely centered.')
        form.addParam('refine_signedCcResLimit', FloatParam, label='Signed CC resolution limit (Å)',
                        default=0.0, expertLevel=LEVEL_ADVANCED,
                        help='Particle alignment is done by maximizing a correlation coefficient '
                        'with the reference. The user has the option to maximize the unsigned '
                        'correlation coefficient instead (starting at the limit set here) to '
                        'reduce overfitting (Stewart and Grigorieff, 2004). Overfitting is also '
                        'reduced by appropriate weighting of the data and this is usually '
                        'sufficient to achieve good refinement results. The limit set here should '
                        'therefore be set to 0.0 to maximize the signed correlation at all '
                        'resolutions, unless there is evidence that there is overfitting. '
                        '(This feature was formerly known as “FBOOST”.)')
        form.addParam('refine_usedPercentage', FloatParam, label='Percentage used (%)',
                        default=100.0, validators=[Range(0, 100)], expertLevel=LEVEL_ADVANCED,
                        help='Percentage of the input particles to be used. If not 100%, only a random '
                        'subset of particles will be used')
        group = form.addGroup('Global search', expertLevel=LEVEL_ADVANCED,
                                condition='refine_type == 1')
        group.addParam('refine_globalMaskRadius', FloatParam, label='Global mask radius (Å)',
                        default=120.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        group.addParam('refine_numResults', IntParam, label='Number of results to refine',
                        default=20, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='For a global search, an angular grid search is performed and '
                        'the alignment parameters for the N best matching projections are '
                        'then refined further in a local refinement. Only the set of '
                        'parameters yielding the best score (correlation coefficient) is '
                        'kept. Increasing N will increase the chances of finding the correct '
                        'particle orientations but will slow down the search. A value '
                        'of 20 is recommended.')
        group.addParam('refine_refineInputParameters', BooleanParam, label='Refine input parameters',
                        default=True, expertLevel=LEVEL_ADVANCED,
                        help=' In addition to the N best sets of parameter values found during '
                        'the grid search, the input set of parameters is also locally refined. '
                        'Switching this off can help reduce over-fitting that may have biased '
                        'the input parameters.')
        group.addParam('refine_angularStep', FloatParam, label='Angular search step (º)',
                        default=35.26, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='he angular step used to generate the search grid for the global '
                        'search. An appropriate value is suggested by default (depending on '
                        'particle size and high-resolution limit) but smaller values can be '
                        'tried if the user suspects that the search misses orientations found '
                        'in the particle dataset. The smaller the value, the finer the search '
                        'grid and the slower the search.')
        group.addParam('refine_xRange', FloatParam, label='Search range in X (Å)',
                        default=22.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The global search can be limited in the X and Y directions '
                        '(measured from the box center) to ensure that only particles close '
                        'to the box center are found. This is useful when the particle density '
                        'is high and particles end up close to each other. In this case, it is '
                        'usually still possible to align all particles in a cluster of particles '
                        '(assuming they do not significantly overlap). The values provided here '
                        'for the search range should be set to exclude the possibility that the '
                        'same particle is selected twice and counted as two different particles.')
        group.addParam('refine_yRange', FloatParam, label='Search range in Y (Å)',
                        default=22.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The global search can be limited in the X and Y directions '
                        '(measured from the box center) to ensure that only particles close '
                        'to the box center are found. This is useful when the particle density '
                        'is high and particles end up close to each other. In this case, it is '
                        'usually still possible to align all particles in a cluster of particles '
                        '(assuming they do not significantly overlap). The values provided here '
                        'for the search range should be set to exclude the possibility that the '
                        'same particle is selected twice and counted as two different particles.')


        form.addSection(label='Classification')
        form.addParam('classification_criteria', EnumParam, choices=['Occupancy', 'Score'], label='Criteria',
                        default=1, 
                        help='Criteria used for determining the particles belonging to a class. In '
                        'our testing, using “Score” leads to better results.\n'
                        '-Occupancy: Occupancy values are updated for each particle-class '
                        'combination according to the log(p) value of the particle and '
                        ' the average occupancy of the class. Then a particle is considered '
                        ' to belong to the class where it obtains the highest occupancy\n'
                        '-Score: The class with the highest score parameter is selected '
                        'for each particle')
        form.addParam('classification_resLimit', FloatParam, label='Resolution Limit (Å)',
                        default=30.0, validators=[GE(0)],
                        help='The limit set here is analogous to the high-resolution limit '
                        'set for refinement. It cannot exceed the refinement limit. Setting '
                        'it to a lower resolution may increase the useful SNR for '
                        'classification and lead to better separation of particles with '
                        'different structural features. However, at lower resolution the '
                        'classification may also become less sensitive to heterogeneity '
                        'represented by smaller structural features.')
        form.addParam('classification_enableFocus', BooleanParam, label='Enable focused classification',
                        default=False, expertLevel=LEVEL_ADVANCED,
                        help='Classification can be performed based on structural variability '
                        'in a defined region of the particle. This is useful when there are '
                        'multiple regions that have uncorrelated structural variability. '
                        'Using focused classification, each of these regions can be classified '
                        'in turn. The focus feature can also be used to reduce noise from other '
                        'parts of the images and increase the useful SNR for classification. '
                        'The focus region is defined by a sphere with coordinates and radius '
                        'in the following four inputs. (This feature was formerly known as '
                        '“focus_mask”.)')
        group = form.addGroup('Focus sphere', 
                                condition='classification_enableFocus is True',
                                help='Sphere on which the classification will be focussed', 
                                expertLevel=LEVEL_ADVANCED)
        group.addParam('classification_sphereX', FloatParam, label='Center X (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED,
                        help='Spherical region inside the particle that contains '
                        'the structural variability to focus on')
        group.addParam('classification_sphereY', FloatParam, label='Center Y (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED,
                        help='Spherical region inside the particle that contains '
                        'the structural variability to focus on')
        group.addParam('classification_sphereZ', FloatParam, label='Center Z (Å)',
                        default=0, expertLevel=LEVEL_ADVANCED,
                        help='Spherical region inside the particle that contains '
                        'the structural variability to focus on')
        group.addParam('classification_sphereRadius', FloatParam, label='Radius (Å)',
                        default=10.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='Spherical region inside the particle that contains '
                        'the structural variability to focus on')


        form.addSection(label='CTF refinement')
        form.addParam('ctf_enable', BooleanParam, label='Refine CTF',
                        default=False, expertLevel=LEVEL_ADVANCED,
                        help=' Should the CTF be refined as well? This is only '
                        'recommended for high-resolution data that yield '
                        'reconstructions of better than 4 Å resolution, and for '
                        'particles of sufficient molecular mass (500 kDa and higher).')
        form.addParam('ctf_range', FloatParam, label='Defocus search range (Å)',
                        condition='ctf_enable is True',
                        default=500.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The range of defocus values to search over for each '
                        'particle. A search with the step size given in the next input '
                        'will be performed starting at the defocus values determined in '
                        'the previous refinement cycle minus the search range, up to '
                        'values plus the search range. The search steps will be applied '
                        'to both defocus values, keeping the estimated astigmatism constant.')
        form.addParam('ctf_step', FloatParam, label='Defocus search step (Å)',
                        condition='ctf_enable is True',
                        default=50.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='The search step for the defocus search.')


        form.addSection(label='Reconstruction')
        form.addParam('reconstruction_score2weight', FloatParam, label='Score to weight constant (Å²)',
                        default=2.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED,
                        help='The particles inserted into a reconstruction will be '
                        'weighted according to their scores. The weighting function '
                        'is akin to a B-factor, attenuating high-resolution signal '
                        'of particles with lower scores more strongly than of particles '
                        'with higher scores. The B-factor applied to each particle '
                        'prior to insertion into the reconstruction is calculated '
                        'as B = (score - average score) * constant * 0.25. Users '
                        'are encouraged to calculate reconstructions with different '
                        'values to find a value that produces the highest resolution. '
                        'Values between 0 and 10 are reasonable (0 will disable weighting).')
        form.addParam('reconstruction_adjustScore4Defocus', BooleanParam, label='Adjust score for defocus',
                        default=True, expertLevel=LEVEL_ADVANCED,
                        help='Scores sometimes depend on the amount of image defocus. '
                        'A larger defocus amplifies low-resolution features in the '
                        'image and this may lead to higher particle scores compared '
                        'to particles from an image with a small defocus. Adjusting '
                        'the scores for this difference makes sure that particles '
                        'with smaller defocus are not systematically downweighted '
                        'by the above B-factor weighting.')
        form.addParam('reconstruction_scoreThreshold', FloatParam, label='Score threshold',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='Particles with a score lower than the threshold will be '
                        'excluded from the reconstruction. This provides a way to '
                        'exclude particles that may score low because of misalignment '
                        'or damage. A value = 0 will select all particles; 0 < value <= 1 '
                        'will be interpreted as a percentage; value > 1 will be '
                        'interpreted as a fixed score threshold.')
        form.addParam('reconstruction_resLimit', FloatParam, label='Resolution limit (Å²)',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                        help='The reconstruction calculation can be accelerated by '
                        'limiting its resolution. It is important to make sure that '
                        'the resolution limit entered here is higher than the '
                        'resolution used for refinement in the following cycle.')
        form.addParam('reconstruction_enableAutoCrop', BooleanParam, label='Enable auto cropping images',
                        default=False, expertLevel=LEVEL_ADVANCED,
                        help='The reconstruction calculation can also be accelerated '
                        'by cropping the boxes containing the particles. Cropping '
                        'will slightly reduce the overall quality of the reconstruction '
                        'due to increased aliasing effects and should not be used when '
                        'finalizing refinement. However, during refinement, cropping '
                        'can greatly increase the speed of reconstruction without '
                        'noticeable impact on the refinement results.')
        form.addParam('reconstruction_enableLikelihoodBlurring', BooleanParam, label='Enable likelihood blurring',
                        default=False, expertLevel=LEVEL_ADVANCED)
        form.addParam('reconstruction_smoothingFactor', FloatParam, label='Smoothing factor',
                        condition='reconstruction_enableLikelihoodBlurring is True',
                        default=1.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)


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

        form.addParallelSection()


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # Shorthands for variables
        nCycles = self._getCycleCount()
        nClasses = self._getClassCount()
        nParticles = self._getParticleCount()
        nWorkers = max(max(int(self.numberOfMpi), int(self.numberOfThreads))-1, 1)
        self.workDistribution = self._distributeWork(nParticles, nWorkers)
        nBlocks = len(self.workDistribution)
        
        # Initialize required files for the first iteration
        self._insertFunctionStep('convertInputStep', nClasses)

        # Execute the pipeline in multiple depending on the parallelization strategy
        prerequisites = []
        if nBlocks > 1:
            prerequisites = self._insertMultiBlockSteps(nCycles, nClasses, nBlocks)
        else:
            prerequisites = self._insertMonoBlockSteps(nCycles, nClasses)

        # Generate the output
        self._insertFunctionStep('createOutputStep', prerequisites=prerequisites)

    # --------------------------- STEPS functions -----------------------------
    
    def convertInputStep(self, nClasses):
        self._createWorkingDir()
        self._createInputParticleStack()
        self._createInputParameters(nClasses)
        self._createInputInitialVolumes(nClasses)
        self._createInputStatistics(nClasses)
    
    def refineStep(self, iter, cls, job):
        # Shorthands for variables
        refine3d = Plugin.getProgram('refine3d')

        # Execute the refine3d program
        self._prepareRefine(iter, cls, job)
        args = self._getRefineArgTemplate() % self._getRefineArgDictionary(iter, cls, job)
        self.runJob(refine3d, args, cwd=self._getTmpPath(), env=Plugin.getEnviron())

    def classifyStep(self, nClasses, nJobs, iter):
        # Merge refinements
        self._mergeRefinementParameters(nClasses, nJobs, iter)

        # Read the refinement data performed by the previous step(s)
        refinement = self._readRefinementParameters(nClasses, iter)

        # Perform the classification according to the selected criteria
        criteria = 'score' if self.classification_criteria.get() == 1 else 'occupancy'
        if criteria == 'occupancy':
            self._updateOccupancyValues(refinement)
        classification = self._classifyRefinement(refinement, criteria)

        # Write the results to disk
        self._writeClassification(iter, classification)
        self._writeClassParameters(iter, refinement, classification)

        # Copy output classification
        copyFile(
            self._getTmpPath(self._getFileName('classify_output_parameters', iter=iter)),
            self._getExtraPath(self._getFileName('output_classification', iter=iter))
        )

    def reconstructStep(self, iter, cls, job):
        # Shorthands for variables
        reconstruct3d = Plugin.getProgram('reconstruct3d')

        # Execute the reconstruct3d program
        self._prepareReconstruct(iter, cls, job)
        args = self._getReconstructArgTemplate() % self._getReconstructArgDictionary(iter, cls, job)        
        self.runJob(reconstruct3d, args, cwd=self._getTmpPath(), env=Plugin.getEnviron())

    def mergeStep(self, nJobs, iter, cls):
        merge3d = Plugin.getProgram('merge3d')

        # Execute the merge3d program
        self._prepareMerge(nJobs, iter, cls)
        argTemplate = self._getMergeArgTemplate()
        args = argTemplate % self._getMergeArgDictionary(nJobs, iter, cls)
        self.runJob(merge3d, args, cwd=self._getTmpPath(), env=Plugin.getEnviron())

        # Copy the output volume and statistics
        copyFile(
            self._getTmpPath(self._getFileName('merge3d_output_reconstruction', iter=iter, cls=cls, rec='filtered')),
            self._getExtraPath(self._getFileName('output_volume', iter=iter, cls=cls))
        )
        copyFile(
            self._getTmpPath(self._getFileName('merge3d_output_statistics', iter=iter, cls=cls)),
            self._getExtraPath(self._getFileName('output_statistics', iter=iter, cls=cls))
        )

    def reconstructAllStep(self, iter, cls):
        # Shorthands for variables
        reconstruct3d = Plugin.getProgram('reconstruct3d')

        # Execute the reconstruct3d program
        self._prepareReconstruct(iter, cls, 0)
        args = self._getReconstructArgTemplate() % self._getReconstructArgDictionary(iter, cls, 0, True)        
        self.runJob(reconstruct3d, args, cwd=self._getTmpPath(), env=Plugin.getEnviron())

        # Copy the output volume and statistics
        copyFile(
            self._getTmpPath(self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=0, rec='filtered')),
            self._getExtraPath(self._getFileName('output_volume', iter=iter, cls=cls))
        )
        copyFile(
            self._getTmpPath(self._getFileName('reconstruct3d_output_statistics', iter=iter, cls=cls, job=0)),
            self._getExtraPath(self._getFileName('output_statistics', iter=iter, cls=cls))
        )

    def createOutputStep(self):
        # Create a SetOfClasses3D
        classes = self._createOutput3dClasses(self._getLastIter())
        self._defineOutputs(outputClasses=classes)

        # Create a SetOfVolumes and define its relations
        volumes = self._createOutputVolumes(classes)
        self._defineOutputs(outputVolumes=volumes)

        # Define source relations
        sources = self._getSourceParameters()
        for source in sources:
            self._defineSourceRelation(source, classes)
            self._defineSourceRelation(source, volumes)


    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        result = []

        if self.classification_resLimit.get() < self.refine_highResLimit.get():
            result.append('Classification resolution limit can not exceed refinement high resolution limit')

        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            if not refRefinement.isFinished():
                result.append('Reference refinement must have finished')
        else:
            if self.input_particles.get() is None:
                result.append('Input particles must be specified when reference refinement is not')
            if len(self.input_initialVolumes) == 0:
                result.append('Input initial volumes must be specified when reference refinement is not')

        return result

    def _summary(self):
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions -----------------------------

    def _getReferenceRefinement(self):
        return self.input_referenceRefinement.get()

    def _getInputParticlesParam(self):
        refRefinement = self._getReferenceRefinement()
        return self.input_particles if refRefinement is None else refRefinement._getInputParticlesParam()

    def _getInputParticles(self):
        return self._getInputParticlesParam().get()

    def _getSourceParameters(self):
        result = []

        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            result = [self.input_referenceRefinement]
        else:
            result = [self.input_particles, self.input_initialVolumes]

        return result

    def _getCycleCount(self):
        return self.cycleCount.get()

    def _getIter(self, i):
        n = self._getCycleCount()
        i %= n # Cycle in (-n, n)
        i += n # Ensure that the number is positive
        i %= n # Cycle in [0, n)
        return i

    def _getLastIter(self):
        return self._getIter(-1)

    def _getParticleCount(self):
        return len(self._getInputParticles())

    def _getClassCount(self):
        refRefinement = self._getReferenceRefinement()
        return len(self.input_initialVolumes) if refRefinement is None else refRefinement._getClassCount()

    def _distributeWork(self, n, m):
        """ Given n items, it distributes it into at most m similarly sized groups. It returns a list 
        with the [first, last) elements of each group as the tuple (first, last)"""

        # Obtain the quotient and the reminder of dividing n by m
        q, r = divmod(n, m)

        # Distribute the work as evenly as possible, groups of the same size
        # as the quotient and the reminder spread across the first group. If 
        # m>n, q=0, so avoid adding zeros at the end of the array
        result = []
        first = 0
        for _ in range(r):
            last = first + q + 1
            result.append((first, last))
            first = last

        if q > 0:
            for _ in range(m-r): # always r < m
                last = first + q
                result.append((first, last))
                first = last


        # Ensure that all the work has been distributed correctly
        assert(result[0][0] == 0) # Start at 0
        assert(result[-1][1] == n) # End at n
        assert(len(result) <= m) # At most m groups

        return result

    def _insertMonoBlockSteps(self, nCycles, nClasses):
        # Perform refine, reconstruct and merge steps repeatedly
        reconstructSteps = [len(self._steps)]*nClasses # Initialize to the previous step
        for i in range(nCycles):
            # Refine all classes
            refineSteps = [0]*nClasses
            for j in range(nClasses):
                prerequisites = [reconstructSteps[j]] # Depend on the creation of the initial volume
                self._insertFunctionStep('refineStep', i, j, 0, prerequisites=prerequisites)
                refineSteps[j] = len(self._steps)

            # Classify according to the result of the refinement
            prerequisites = refineSteps # Depend on all previous refinements referring to this job
            self._insertFunctionStep('classifyStep', nClasses, 1, i, prerequisites=prerequisites)
            classifySteps = [len(self._steps)]

            # Reconstruct each class
            reconstructSteps = [0]*nClasses
            for j in range(nClasses):
                prerequisites = classifySteps # Depend on the classification
                self._insertFunctionStep('reconstructAllStep', i, j, prerequisites=prerequisites)
                reconstructSteps[j] = len(self._steps)

        return reconstructSteps # Return the completion of all reconstruct steps

    def _insertMultiBlockSteps(self, nCycles, nClasses, nJobs):
        # Perform refine, reconstruct and merge steps repeatedly
        mergeSteps = [len(self._steps)]*nClasses # Initialize to the previous step
        for i in range(nCycles):
            # Refine all classes
            refineSteps = [[0]*nJobs for _ in range(nClasses)]
            for j in range(nClasses):
                for k in range(nJobs):
                    prerequisites = [mergeSteps[j]] # Depend on the creation of the initial volume
                    self._insertFunctionStep('refineStep', i, j, k, prerequisites=prerequisites)
                    refineSteps[j][k] = len(self._steps)

            # Classify according to the result of the refinement
            prerequisites = [item for sublist in refineSteps for item in sublist] # All refinements
            self._insertFunctionStep('classifyStep', nClasses, nJobs, i, prerequisites=prerequisites)
            classifySteps = [len(self._steps)]

            # Reconstruct each class
            for j in range(nClasses):
                reconstructSteps = [0]*nJobs
                for k in range(nJobs):
                    prerequisites = classifySteps # Depend on the classification
                    self._insertFunctionStep('reconstructStep', i, j, k, prerequisites=prerequisites)
                    reconstructSteps[k] = len(self._steps)
                
                prerequisites=reconstructSteps # Depend on all partial reconstructions of this class
                self._insertFunctionStep('mergeStep', nJobs, i, j, prerequisites=prerequisites)
                mergeSteps[j] = len(self._steps)

        return mergeSteps # Return the completion of all merge steps


    def _createWorkingDir(self):
        tmpDirectories = [
            'Inputs',
            'Refine3D/Parameters',
            'Refine3D/Reconstructions',
            'Refine3D/Statistics',
            'Refine3D/Projections',
            'Classify/Parameters',
            'Reconstruct3D/Parameters',
            'Reconstruct3D/Reconstructions',
            'Reconstruct3D/Statistics',
            'Reconstruct3D/Dumps',
            'Merge3D/Reconstructions',
            'Merge3D/Statistics',
            'Merge3D/Dumps',
        ]

        extraDirectories = [
            'Inputs',
            'Reconstructions',
            'Statistics',
            'Classifications',
            'Refinements'
        ]
        
        for d in tmpDirectories:
            makePath(self._getTmpPath(d))

        for d in extraDirectories:
            makePath(self._getExtraPath(d))

    def _createInputParticleStack(self):
        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            # Create a link to the reference refinement's particles
            createLink(
                refRefinement._getExtraPath(refRefinement._getFileName('input_particles')),
                self._getExtraPath(self._getFileName('input_particles'))
            )

        else:
            # Create a stack of particles from the input particles
            particles = self._getInputParticles()
            path = self._getExtraPath(self._getFileName('input_particles'))
            particles.writeStack(path)

        # Create a link for the input particles in tmp (required by programs)
        createLink(
            self._getExtraPath(self._getFileName('input_particles')),
            self._getTmpPath(self._getFileName('input_particles'))
        )

    def _createInputParameters(self, nClasses):
        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            # Link last refinement parameters from the reference refinement
            for cls in range(nClasses):
                createLink(
                    refRefinement._getExtraPath(refRefinement._getFileName('output_refinement', iter=refRefinement._getIter(self.input_referenceIteration.get()), cls=cls)),
                    self._getExtraPath(self._getFileName('input_parameters', cls=cls))
                )

        else:
            # Create a single parameter file from input particles and replicate it for the rest of the classes
            path = self._getExtraPath(self._getFileName('input_parameters', cls=0))
            particles = self._getInputParticles()

            with FullFrealignParFile(path, 'w') as f:
                f.writeHeader()
                for i, particle in enumerate(particles):
                    f.writeParticle(particle, id=i+1, occupancy=100.0/nClasses) # Use its 1-based index as the mic_id
        
            for cls in range(1, nClasses):
                createLink(
                    self._getExtraPath(self._getFileName('input_parameters', cls=0)),
                    self._getExtraPath(self._getFileName('input_parameters', cls=cls))
                )

    def _createInputInitialVolumes(self, nClasses):
        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            # Link last reconstruction output volume from the reference refinement
            for cls in range(nClasses):
                createLink(
                    refRefinement._getExtraPath(refRefinement._getFileName('output_volume', iter=refRefinement._getIter(self.input_referenceIteration.get()), cls=cls)),
                    self._getExtraPath(self._getFileName('input_volume', cls=cls))
                )
        else:
            # Write the input parameters
            ih = ImageHandler()
            for i in range(nClasses):
                path = self._getExtraPath(self._getFileName('input_volume', cls=i))
                ih.convert(self.input_initialVolumes[i].get(), path)

    def _createInputStatistics(self, nClasses):
        refRefinement = self._getReferenceRefinement()
        if refRefinement is not None:
            # Link last reconstruction statistics from the reference refinement
            for cls in range(nClasses):
                createLink(
                    refRefinement._getExtraPath(refRefinement._getFileName('output_statistics', iter=refRefinement._getIter(self.input_referenceIteration.get()), cls=cls)),
                    self._getExtraPath(self._getFileName('input_statistics', cls=cls))
                )
        else:
            # Create an empty file as it wont be used and replicate it
            with open(self._getExtraPath(self._getFileName('input_statistics', cls=0)), 'w') as f:
                f.write("C NO STATISTICS")

            # Link to /dev/null. TODO maybe recover info from 
            for cls in range(1, nClasses):
                createLink(
                    self._getExtraPath(self._getFileName('input_statistics', cls=0)),
                    self._getExtraPath(self._getFileName('input_statistics', cls=cls))
                )

    def _prepareRefine(self, iter, cls, job):
        if iter == 0:
            # For the first step use the input volume as the input reconstruction
            createLink(
                self._getExtraPath(self._getFileName('input_parameters', cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_parameters', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('input_volume', cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('input_statistics', cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_statistics', iter=iter, cls=cls))
            )
        else:
            # Use the statistics, parameters and volume from the previous reconstruction
            createLink(
                self._getExtraPath(self._getFileName('output_refinement', iter=iter-1, cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_parameters', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('output_volume', iter=iter-1, cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('output_statistics', iter=iter-1, cls=cls)),
                self._getTmpPath(self._getFileName('refine3d_input_statistics', iter=iter, cls=cls))
            )


    def _getRefineArgTemplate(self):
        return """ << eof
%(input_particle_images)s
%(input_parameter_file)s
%(input_reconstruction)s
%(input_reconstruction_statistics)s
%(use_statistics)s
%(output_matching_projections)s
%(output_parameter_file)s
%(output_shift_file)s
%(my_symmetry)s
%(first_particle)d
%(last_particle)d
%(percent_used)f
%(pixel_size)f
%(voltage_kV)f
%(spherical_aberration_mm)f
%(amplitude_contrast)f
%(molecular_mass_kDa)f
%(inner_mask_radius)f
%(outer_mask_radius)f
%(low_resolution_limit)f
%(high_resolution_limit)f
%(signed_CC_limit)f
%(classification_resolution_limit)f
%(mask_radius_search)f
%(high_resolution_limit_search)f
%(angular_step)f
%(best_parameters_to_keep)d
%(max_search_x)f
%(max_search_y)f
%(mask_center_2d_x)f
%(mask_center_2d_y)f
%(mask_center_2d_z)f
%(mask_radius_2d)f
%(defocus_search_range)f
%(defocus_step)f
%(padding)f
%(global_search)s
%(local_refinement)s
%(refine_psi)s
%(refine_theta)s
%(refine_phi)s
%(refine_x)s
%(refine_y)s
%(calculate_matching_projections)s
%(apply_2D_masking)s
%(ctf_refinement)s
%(normalize_particles)s
%(invert_contrast)s
%(exclude_blank_edges)s
%(normalize_input_3d)s
%(threshold_input_3d)s
eof
"""

    def _getRefineArgDictionary(self, iter, cls, job):
        inputParticles = self._getInputParticles()
        acquisition = inputParticles.getAcquisition()
        block = self.workDistribution[job]
        useStatistics = (self._getReferenceRefinement() is not None) or (iter > 0)

        return {
            'input_particle_images': self._getFileName('input_particles'),
            'input_parameter_file': self._getFileName('refine3d_input_parameters', iter=iter, cls=cls),
            'input_reconstruction': self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls),
            'input_reconstruction_statistics': self._getFileName('refine3d_input_statistics', iter=iter, cls=cls),
            'use_statistics': boolToYN(useStatistics),
            'output_matching_projections': self._getFileName('refine3d_output_matching_projections', iter=iter, cls=cls, job=job),
            'output_parameter_file': self._getFileName('refine3d_output_parameters', iter=iter, cls=cls, job=job),
            'output_shift_file': self._getFileName('refine3d_output_shifts', iter=iter, cls=cls, job=job),
            'my_symmetry': self.refine_symmetry.get(),
            'first_particle': block[0] + 1,
            'last_particle': block[1],
            'percent_used': self.refine_usedPercentage.get() / 100.0,
            'pixel_size': 1.0/inputParticles.getSamplingRate(), # Pixel size is the inverse of the sampling rate
            'voltage_kV': acquisition.getVoltage(), # Already in kV
            'spherical_aberration_mm': acquisition.getSphericalAberration(), # Already in mm
            'amplitude_contrast': acquisition.getAmplitudeContrast(),
            'molecular_mass_kDa': self.input_molecularMass.get(),
            'inner_mask_radius': self.refine_innerMaskRadius.get(),
            'outer_mask_radius': self.refine_outerMaskRadius.get(),
            'low_resolution_limit': self.refine_lowResLimit.get(),
            'high_resolution_limit': self.refine_highResLimit.get(),
            'signed_CC_limit': self.refine_signedCcResLimit.get(),
            'classification_resolution_limit': self.classification_resLimit.get(),
            'mask_radius_search': self.refine_globalMaskRadius.get(),
            'high_resolution_limit_search': self.refine_highResLimit.get(),
            'angular_step': self.refine_angularStep.get(),
            'best_parameters_to_keep': self.refine_numResults.get(),
            'max_search_x': self.refine_xRange.get(),
            'max_search_y': self.refine_yRange.get(),
            'mask_center_2d_x': self.classification_sphereX.get(),
            'mask_center_2d_y': self.classification_sphereY.get(),
            'mask_center_2d_z': self.classification_sphereZ.get(),
            'mask_radius_2d': self.classification_sphereRadius.get(),
            'defocus_search_range': self.ctf_range.get(),
            'defocus_step': self.ctf_step.get(),
            'padding': 1.0,
            'global_search': boolToYN(self.refine_type.get() == 1),
            'local_refinement': boolToYN(self.refine_type.get() == 0),
            'refine_psi': boolToYN(self.refine_psi.get()),
            'refine_theta': boolToYN(self.refine_theta.get()),
            'refine_phi': boolToYN(self.refine_phi.get()),
            'refine_x': boolToYN(self.refine_xShift.get()),
            'refine_y': boolToYN(self.refine_yShift.get()),
            'calculate_matching_projections': boolToYN(False),
            'apply_2D_masking': boolToYN(self.classification_enableFocus.get()),
            'ctf_refinement': boolToYN(self.ctf_enable.get()),
            'normalize_particles': boolToYN(True),
            'invert_contrast': boolToYN(self.input_isWhite.get()),
            'exclude_blank_edges': boolToYN(False), 
            'normalize_input_3d': boolToYN(not self.reconstruction_enableLikelihoodBlurring.get()),
            'threshold_input_3d': boolToYN(True),
        }

    def _mergeRefinementParameters(self, nClasses, nJobs, iter):
        for cls in range(nClasses):
            outputPath = self._getExtraPath(self._getFileName('output_refinement', iter=iter, cls=cls))
            with FullFrealignParFile(outputPath, 'w') as output:
                output.writeHeader()

                for job in range(nJobs):
                    inputPath = self._getTmpPath(self._getFileName('refine3d_output_parameters', iter=iter, cls=cls, job=job))
                    with FullFrealignParFile(inputPath, 'r') as input:
                        for row in input:
                            output.writeRow(row)

    def _readRefinementParameters(self, nClasses, iter):
        result = [None] * nClasses

        # Read all the putput parameters
        for cls in range(nClasses):
            path = self._getExtraPath(self._getFileName('output_refinement', iter=iter, cls=cls))
            with FullFrealignParFile(path, 'r') as f:
                result[cls] = list(f)

        # Ensure that everything was read
        assert(len(result) == nClasses)
        for i in range(1, len(result)):
            assert(len(result[i]) == len(result[0]))
        return result

    def _calculateRefinementParameterAverage(self, refinement, parameter):
        sum = 0
        for row in refinement:
            sum += row[parameter]
        return sum / len(refinement)

    def _updateOccupancyValues(self, refinement):
        nClasses = len(refinement)
        nParticles = len(refinement[0])
        averageOccupancies = [self._calculateRefinementParameterAverage(cls, 'occupancy') for cls in refinement]

        for i in range(nParticles):
            # Find the greatest logP value
            maxLogP = refinement[0][i]['log_p']
            for j in range(1, nClasses):
                value = refinement[j][i]['log_p']
                if value > maxLogP:
                    maxLogP = value

            # Calculate the difference of the actual logP value respect the maximum one
            deltaLogP = [maxLogP - refinement[j][i]['log_p'] for j in range(nClasses)]

            # Calculate the occupancies
            occupancies = [averageOccupancies[j] * math.exp(-deltaLogP[j]) if deltaLogP[j] < 10 else 0 for j in range(nClasses)] # TODO maybe skip the if-else part and always use exp
            occupancies = [x/sum(occupancies) for x in occupancies]

            # Average sigma 
            averageSigma = 0
            for j in range(nClasses):
                averageSigma += refinement[j][i]['sigma']*occupancies[j]

            # Write the results
            for j in range(nClasses):
                refinement[j][i]['sigma'] = averageSigma
                refinement[j][i]['occupancy'] = occupancies[j]*100

    def _classifyRefinement(self, refinement, criteria):
        result = []
        
        # Perform the classification
        for i in range(len(refinement[0])):
            # Determine the best class
            bestCls = 0
            for cls in range(1, len(refinement)):
                if refinement[cls][i][criteria] > refinement[bestCls][i][criteria]:
                    bestCls = cls

            # Store it
            row = refinement[bestCls][i].copy()
            row['film'] = bestCls
            result.append(row)

        assert(len(result) == len(refinement[0]))
        return result

    def _writeClassification(self, iter, classification):
        path = self._getTmpPath(self._getFileName('classify_output_parameters', iter=iter))
        with FullFrealignParFile(path, 'w') as f:
            f.writeHeader()
            for row in classification:
                f.writeRow(row)

    def _writeClassParameters(self, iter, refinement, classification):
        for cls, _ in enumerate(refinement):
            path = self._getTmpPath(self._getFileName('classify_output_class_parameters', iter=iter, cls=cls))
            with FullFrealignParFile(path, 'w') as f:
                f.writeHeader()
                for refinementRow, classificationRow in zip(refinement[cls], classification):
                    row = refinementRow.copy()
                    row['film'] = 1 if classificationRow['film'] == cls else -1 # Selectively enable
                    row['magnification'] = 0.0
                    row['change'] = 0.0
                    f.writeRow(row)

    def _prepareReconstruct(self, iter, cls, job):
        if iter == 0:
            # For the first step use the input volume as the input reconstruction
            createLink(
                self._getExtraPath(self._getFileName('input_volume', cls=cls)),
                self._getTmpPath(self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls))
            )
        else:
            # Use the previous reconstruction
            createLink(
                self._getExtraPath(self._getFileName('output_volume', iter=iter-1, cls=cls)),
                self._getTmpPath(self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls))
            )

        # Use the parameters provided by the previous step
        createLink(
            self._getTmpPath(self._getFileName('classify_output_class_parameters', iter=iter, cls=cls)),
            self._getTmpPath(self._getFileName('reconstruct3d_input_parameters', iter=iter, cls=cls, job=job))
        )

    def _getReconstructArgTemplate(self):
        return """ << eof
%(input_particle_stack)s
%(input_parameter_file)s
%(input_reconstruction)s
%(output_reconstruction_1)s
%(output_reconstruction_2)s
%(output_reconstruction_filtered)s
%(output_resolution_statistics)s
%(my_symmetry)s
%(first_particle)d
%(last_particle)d
%(pixel_size)f
%(voltage_kV)f
%(spherical_aberration_mm)f
%(amplitude_contrast)f
%(molecular_mass_kDa)f
%(inner_mask_radius)f
%(outer_mask_radius)f
%(resolution_limit_rec)f
%(resolution_limit_ref)f
%(score_weight_conversion)f
%(score_threshold)f
%(smoothing_factor)f
%(padding)f
%(normalize_particles)s
%(adjust_scores)s
%(invert_contrast)s
%(exclude_blank_edges)s
%(crop_images)s
%(split_even_odd)s
%(center_mass)s
%(use_input_reconstruction)s
%(threshold_input_3d)s
%(dump_arrays)s
%(dump_file_1)s
%(dump_file_2)s
eof
"""

    def _getReconstructArgDictionary(self, iter, cls, job, all=False):
        inputParticles = self._getInputParticles()
        acquisition = inputParticles.getAcquisition()
        block = self.workDistribution[job]

        return {
            'input_particle_stack': self._getFileName('input_particles'),
            'input_parameter_file': self._getFileName('reconstruct3d_input_parameters', iter=iter, cls=cls, job=job),
            'input_reconstruction': self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls),
            'output_reconstruction_1': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='1'),
            'output_reconstruction_2': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='2'),
            'output_reconstruction_filtered': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='filtered'),
            'output_resolution_statistics': self._getFileName('reconstruct3d_output_statistics', iter=iter, cls=cls, job=job),
            'my_symmetry': self.refine_symmetry.get(),
            'first_particle': block[0] + 1,
            'last_particle': block[1],
            'pixel_size': 1.0/inputParticles.getSamplingRate(), # Pixel size is the inverse of the sampling rate
            'voltage_kV': acquisition.getVoltage(), # Already in kV
            'spherical_aberration_mm': acquisition.getSphericalAberration(), # Already in mm
            'amplitude_contrast': acquisition.getAmplitudeContrast(),
            'molecular_mass_kDa': self.input_molecularMass.get(),
            'inner_mask_radius': self.refine_innerMaskRadius.get(),
            'outer_mask_radius': self.refine_outerMaskRadius.get(),
            'resolution_limit_rec': self.reconstruction_resLimit.get(),
            'resolution_limit_ref': self.refine_highResLimit.get(),
            'score_weight_conversion': self.reconstruction_score2weight.get(),
            'score_threshold': self.reconstruction_scoreThreshold.get(),
            'smoothing_factor': self.reconstruction_smoothingFactor.get(),
            'padding': 1.0,
            'normalize_particles': boolToYN(False), # Does not work if True
            'invert_contrast': boolToYN(self.input_isWhite.get()),
            'exclude_blank_edges': boolToYN(False),
            'adjust_scores': boolToYN(self.reconstruction_adjustScore4Defocus.get()),
            'crop_images': boolToYN(self.reconstruction_enableAutoCrop.get()),
            'split_even_odd': boolToYN(False),
            'center_mass': boolToYN(False),
            'use_input_reconstruction': boolToYN(self.reconstruction_enableLikelihoodBlurring.get()),
            'threshold_input_3d': boolToYN(True),
            'dump_arrays': boolToYN(not all),
            'dump_file_1': self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=job, par='odd'),
            'dump_file_2': self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=job, par='evn'),
        }

    def _prepareMerge(self, nJobs, iter, cls):
        # Link the input seed files
        parities = ['evn', 'odd']
        for parity in parities:
            for i in range(nJobs):
                createLink(
                    self._getTmpPath(self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=i, par=parity)),
                    self._getTmpPath(self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par=parity, seed=str(i+1)))
                )

    def _getMergeArgTemplate(self):
        return """ << eof
%(output_reconstruction_1)s
%(output_reconstruction_2)s
%(output_reconstruction_filtered)s
%(output_resolution_statistics)s
%(molecular_mass_kDa)f
%(inner_mask_radius)f
%(outer_mask_radius)f
%(dump_file_seed_1)s
%(dump_file_seed_2)s
%(number_of_dump_files)d
eof
"""

    def _getMergeArgDictionary(self, nJobs, iter, cls):
        return {
            'output_reconstruction_1': self._getFileName('merge3d_output_reconstruction', iter=iter, cls=cls, rec='1'),
            'output_reconstruction_2': self._getFileName('merge3d_output_reconstruction', iter=iter, cls=cls, rec='2'),
            'output_reconstruction_filtered': self._getFileName('merge3d_output_reconstruction', iter=iter, cls=cls, rec='filtered'),
            'output_resolution_statistics': self._getFileName('merge3d_output_statistics', iter=iter, cls=cls),
            'molecular_mass_kDa': self.input_molecularMass.get(),
            'inner_mask_radius': self.refine_innerMaskRadius.get(),
            'outer_mask_radius': self.refine_outerMaskRadius.get(),
            'dump_file_seed_1': self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par='odd', seed=''),
            'dump_file_seed_2': self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par='evn', seed=''),
            'number_of_dump_files': nJobs,
        }

    def _fillClasses(self, clsSet, nClasses, iteration):
        """ Create the SetOfClasses3D from a given iteration. """
        classLoader = CistemProt3DClassification.ClassesLoader(
            [self._getExtraPath(self._getFileName('output_volume', iter=iteration, cls=cls)) for cls in range(nClasses)],
            [self._getExtraPath(self._getFileName('output_statistics', iter=iteration, cls=cls)) for cls in range(nClasses)],
            self._getExtraPath(self._getFileName('output_classification', iter=iteration)),
            ALIGN_PROJ
        )
        classLoader.fillClasses(clsSet)

    def _createOutput3dClasses(self, iter):
        nClasses = self._getClassCount()
        classes = self._createSetOfClasses3D(self._getInputParticles())
        self._fillClasses(classes, nClasses, iter)
        return classes

    def _createOutputVolumes(self, classes):
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(classes.getImages().getSamplingRate())
        for cls in classes:
            vol = cls.getRepresentative()
            vol.setObjId(cls.getObjId())
            volumes.append(vol)
        
        return volumes

    class ClassesLoader:
        """ Helper class to read classes information from parameter files produced
        by Cistem classification runs.
        """
        def __init__(self, repPaths, statisticsPaths, particleDataPath, alignment):
            self._classRepresentatives = repPaths
            self._classStatistics = [FrealignStatisticsFile(p, 'r') for p in statisticsPaths]
            self._particleData = FullFrealignParFile(particleDataPath, 'r')
            self._alignment = alignment

        def fillClasses(self, clsSet):
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                updateClassCallback=self._updateClass,
                                itemDataIterator=iter(self._particleData),
                                doClone=False)

        def _updateParticle(self, item, row):
            self._updateClassId(item, row)
            self._updateCtf(item, row)
            self._updateTransform(item, row)

            #if getattr(self, '__updatingFirst', True):
            #    self._reader.createExtraLabels(item, row, PARTICLE_EXTRA_LABELS)
            #    self.__updatingFirst = False
            #else:
            #    self._reader.setExtraLabels(item, row)

        def _updateClassId(self, item, row):
            item.setClassId(row['film']+1)

        def _updateCtf(self, item, row):
            if not item.hasCTF():
                item.setCTF(CTFModel())
            ctf = item.getCTF()
            ctf.setStandardDefocus(row['defocus_u'], row['defocus_v'], row['defocus_angle'])

        def _updateTransform(self, item, row):
            matrix = matrixFromGeometry(
                np.array([row['shift_x'], row['shift_y'], 0]),
                np.array([row['psi'], row['theta'], row['phi']])
            )

            if not item.hasTransform():
                item.setTransform(Transform())
            transform = item.getTransform()
            transform.setMatrix(matrix)

        def _updateClass(self, item):
            classId = item.getObjId()
            item.setAlignment(self._alignment)
            item.getRepresentative().setLocation(self._classRepresentatives[classId-1])
