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

from pwem.emlib.image.image_handler import ImageHandler
from pwem.objects.data import Particle, SetOfParticles
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import EnumParam, MultiPointerParam, PointerParam, FloatParam, IntParam, BooleanParam, StringParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pyworkflow.constants import BETA
from pyworkflow.utils.path import makePath, createLink, cleanPattern, moveFile

from pwem.protocols import ProtClassify3D

from cistem import Plugin
from cistem.convert import FullFrealignParFile, boolToYN

import math
import multiprocessing as mp
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
            'input_particles': f'input_particles_{jobFmt}.mrcs',
            'input_parameters': f'input_parameters_{jobFmt}.par',
            'input_volume': f'input_volume_{classFmt}.mrc',
            'refine3d_input_parameters': f'Refine3D/Parameters/input_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'refine3d_input_reconstruction': f'Refine3D/Reconstructions/input_reconstruction_{iterFmt}_{classFmt}.mrc',
            'refine3d_input_statistics': f'Refine3D/Statistics/input_statistics_{iterFmt}_{classFmt}.txt',
            'refine3d_output_matching_projections': f'Refine3D/Projections/output_matching_projections_{iterFmt}_{classFmt}_{jobFmt}.mrc',
            'refine3d_output_parameters': f'Refine3D/Parameters/output_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'refine3d_output_shifts': f'Refine3D/Parameters/output_shifts_{iterFmt}_{classFmt}_{jobFmt}.par',
            'classify_output_parameters': f'Classify/Parameters/output_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'reconstruct3d_input_parameters': f'Reconstruct3D/Parameters/input_parameters_{iterFmt}_{classFmt}_{jobFmt}.par',
            'reconstruct3d_input_reconstruction': f'Reconstruct3D/Reconstructions/input_reconstruction_{iterFmt}_{classFmt}.mrc',
            'reconstruct3d_output_reconstruction': f'Reconstruct3D/Reconstructions/output_reconstruction_{iterFmt}_{classFmt}_{reconstructionNameFmt}_{jobFmt}.mrc',
            'reconstruct3d_output_statistics': f'Reconstruct3D/Statistics/output_statistics_{iterFmt}_{classFmt}_{jobFmt}.txt',
            'reconstruct3d_output_dump': f'Reconstruct3D/Dumps/output_dump_{iterFmt}_{classFmt}_{parityFmt}_{jobFmt}.dmp',
            'merge3d_input_dump': f'Merge3D/Dumps/input_dump_{parityFmt}_{seedFmt}.dmp',
            'merge3d_output_reconstruction': f'Merge3D/Reconstructions/output_reconstruction_{iterFmt}_{classFmt}_{reconstructionNameFmt}.mrc',
            'merge3d_output_resolution_statistics': f'Merge3D/Statistics/output_resolution_statistics_{iterFmt}_{classFmt}.txt',
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
        form.addParam('input_molecularMass', FloatParam, label='Molecular mass (kDa)',
                        default=100.0, validators=[GT(0)]) # TODO default


        form.addSection(label='Refinement')
        form.addParam('cycleCount', IntParam, label='Cycle Count',
                        help='Number of refinement cycles to be executed',
                        default=1, validators=[GE(1)])
        form.addParam('refine_type', EnumParam, choices=['Local', 'Global'], label='Refinement type',
                        default=0)
        form.addParam('refine_symmetry', StringParam, label='Symmetry',
                        default='c1')
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
        form.addParam('refine_highResLimit', FloatParam, label='High resolution limit (Å)',
                        default=8.0, validators=[GT(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_outerMaskRadius', FloatParam, label='Outer mask radius (Å)',
                        default=97.5, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_innerMaskRadius', FloatParam, label='Inner mask radius (Å)',
                        default=0.0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_signedCcResLimit', FloatParam, label='Signed CC resolution limit (Å)',
                        default=0.0, expertLevel=LEVEL_ADVANCED)
        form.addParam('refine_usedPercentage', FloatParam, label='Percentage used (%)',
                        default=100.0, validators=[Range(0, 100)], expertLevel=LEVEL_ADVANCED)
        group = form.addGroup('Global search', expertLevel=LEVEL_ADVANCED,
                                condition='refine_type == 1')
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
        form.addParam('classification_resLimit', FloatParam, label='Resolution Limit (Å)',
                        help='Resolution limit for classification. Use 0.0 for maximum',
                        default=30.0, validators=[GE(0)])
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
        group.addParam('classification_sphereRadius', FloatParam, label='Radius (Å)',
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

        form.addParallelSection()


    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # Shorthands for variables
        nCycles = self.cycleCount.get()
        nClasses = len(self.input_initialVolumes)
        nParticles = len(self.input_particles.get())
        nWorkers = int(self.numberOfMpi)*int(self.numberOfThreads)
        workDistribution = distribute_work(nParticles, nWorkers)
        nJobs = len(workDistribution)
        
        # Initialize required files for the first iteration
        self._insertFunctionStep('convertInputStep', workDistribution)

        # Perform refine, reconstruct and merge steps repeatedly
        mergeSteps = [len(self._steps)]*nClasses # Initialize to the convertInputStep, where initial volumes are created
        for i in range(nCycles):
            # Refine all classes
            refineSteps = [[0]*nJobs for _ in range(nClasses)]
            for j in range(nClasses):
                for k in range(nJobs):
                    prerequisites = [mergeSteps[j]] # Depend on the creation of the initial volume
                    self._insertFunctionStep('refineStep', i, j, k, prerequisites=prerequisites)
                    refineSteps[j][k] = len(self._steps)

            # Classify according to the result of the refinement
            classifySteps = [0]*nJobs
            for j in range(nJobs):
                prerequisites = [refineSteps[k][j] for k in range(nClasses)] # Depend on all previous refinements referring to this job
                self._insertFunctionStep('classifyStep', nClasses, i, j, prerequisites=prerequisites)
                classifySteps[j] = len(self._steps)

            # Reconstruct each class
            for j in range(nClasses):
                reconstructSteps = [0]*nJobs
                for k in range(nJobs):
                    prerequisites = [classifySteps[k]] # Depend on the classification referring to this job
                    self._insertFunctionStep('reconstructStep', i, j, k, prerequisites=prerequisites)
                    reconstructSteps[k] = len(self._steps)
                
                prerequisites=reconstructSteps # Depend on all partial reconstructions of this class
                self._insertFunctionStep('mergeStep', nJobs, i, j, prerequisites=prerequisites)
                mergeSteps[j] = len(self._steps)

        # Generate the output
        prerequisites = mergeSteps # Depend on the creation of all initial volumes
        self._insertFunctionStep('createOutputStep', prerequisites=prerequisites)

    # --------------------------- STEPS functions -----------------------------
    
    def convertInputStep(self, workDistribution):
        self._createWorkingDir()

        # Distribute particles according to the work sizes
        particles = self.input_particles.get()
        particleGroups = self._distributeParticles(particles, workDistribution)

        # Write particle stacks and parameter files for each group
        for job, particleGroup in enumerate(particleGroups):
            self._createInputParticleStack(job, particleGroup)
            self._createInputParameters(job, particleGroup)

        self._createInitialVolumes()

    def refineStep(self, iter, cls, job):
        # Shorthands for variables
        refine3d = Plugin.getProgram('refine3d')

        # Execute the refine3d program
        self._prepareRefine(iter, cls, job)
        args = self._getRefineArgTemplate() % self._getRefineArgDictionary(iter, cls, job)
        self.runJob(refine3d, args, cwd=self._getExtraPath(), env=Plugin.getEnviron())

    def classifyStep(self, nClasses, iter, job):
        refinement = self._readRefinementParameters(nClasses, iter, job)

        # Perform the classification
        for i in range(len(refinement[0])):
            # Determine the best class
            bestCls = 0
            maxScore = 0.0
            for cls in range(len(refinement)):
                score = refinement[cls][i]['score']
                if score >= maxScore:
                    (bestCls, maxScore) = (cls, score)

            # Modify the refinement to only include the best class
            for cls in range(len(refinement)):
                refinement[cls][i]['film'] = 1 if cls == bestCls else 0 # Used as include

        # Write the results to disk
        for cls, particles in enumerate(refinement):
            path = self._getExtraPath(self._getFileName('classify_output_parameters', iter=iter, cls=cls, job=job))
            with FullFrealignParFile(path, 'w') as f:
                f.writeHeader()
                for particle in particles:
                    f.writeRow(particle)

    def reconstructStep(self, iter, cls, job):
        # Shorthands for variables
        reconstruct3d = Plugin.getProgram('reconstruct3d')

        # Execute the reconstruct3d program
        self._prepareReconstruct(iter, cls, job)
        args = self._getReconstructArgTemplate() % self._getReconstructArgDictionary(iter, cls, job)        
        self.runJob(reconstruct3d, args, cwd=self._getExtraPath(), env=Plugin.getEnviron())

    def mergeStep(self, nJobs, iter, cls):
        merge3d = Plugin.getProgram('merge3d')

        # Execute the merge3d program
        self._prepareMerge(nJobs, iter, cls)
        argTemplate = self._getMergeArgTemplate()
        args = argTemplate % self._getMergeArgDictionary(nJobs, iter, cls)
        self.runJob(merge3d, args, cwd=self._getExtraPath(), env=Plugin.getEnviron())

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




    def _createWorkingDir(self):
        directories = [
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
        
        for d in directories:
            path = self._getExtraPath(d)                    
            makePath(path)

    def _distributeParticles(self, particles, workDistribution):
        # Cistem requires input particles to be ordered by micrograph id
        particleIter = iter(particles.iterItems(orderBy=['_micId', 'id'], 
                                                direction='ASC' ))

        # Separate particles into groups
        groups = []
        for jobSize in workDistribution:
            # Create a new set of particles at the back, configuring it as the source one
            group = self._createSetOfParticles()
            group.copyInfo(particles)

            # Append as many particles as requested to the group
            for _ in range(jobSize):
                particle = next(particleIter, None)
                assert(particle is not None)
                group.append(particle)

            # Write the new group to the result
            groups.append(group)

        # Ensure we have consumed all particles
        assert(next(particleIter, None) is None)

        return groups


    def _createInputParticleStack(self, job, particles):
        path = self._getExtraPath(self._getFileName('input_particles', job=job))
        particles.writeStack(path) # They should be already ordered by micId

    def _createInputParameters(self, job, particles):
        path = self._getExtraPath(self._getFileName('input_parameters', job=job))

        with FullFrealignParFile(path, 'w') as f:
            f.writeHeaderAndParticleSet(particles)

    def _createInitialVolumes(self):
        initialVolumes = self.input_initialVolumes

        ih = ImageHandler()
        for i, vol in enumerate(initialVolumes):
            path = self._getExtraPath(self._getFileName('input_volume', cls=i))
            ih.convert(vol.get(), path)


    def _prepareRefine(self, iter, cls, job):
        if iter == 0:
            # For the first step use the input volume as the input reconstruction
            createLink(
                self._getExtraPath(self._getFileName('input_volume', cls=cls)),
                self._getExtraPath(self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('input_parameters', job=job)),
                self._getExtraPath(self._getFileName('refine3d_input_parameters', iter=iter, cls=cls, job=job))
            )
        else:
            # Use the statistics, parameters and volume from the previous reconstruction
            createLink(
                self._getExtraPath(self._getFileName('merge3d_output_reconstruction', iter=iter-1, cls=cls, rec='filtered')),
                self._getExtraPath(self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls))
            )
            createLink(
                self._getExtraPath(self._getFileName('refine3d_output_parameters', iter=iter-1, cls=cls, job=job)),
                self._getExtraPath(self._getFileName('refine3d_input_parameters', iter=iter, cls=cls, job=job))
            )
            createLink(
                self._getExtraPath(self._getFileName('merge3d_output_resolution_statistics', iter=iter-1, cls=cls)),
                self._getExtraPath(self._getFileName('refine3d_input_statistics', iter=iter, cls=cls))
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
        inputParticles = self.input_particles.get()
        acquisition = inputParticles.getAcquisition()

        return {
            'input_particle_images': self._getFileName('input_particles', job=job),
            'input_parameter_file': self._getFileName('refine3d_input_parameters', iter=iter, cls=cls, job=job),
            'input_reconstruction': self._getFileName('refine3d_input_reconstruction', iter=iter, cls=cls),
            'input_reconstruction_statistics': self._getFileName('refine3d_input_statistics', iter=iter, cls=cls),
            'use_statistics': boolToYN(iter > 0), # Do not use statistics at the first iteration (they do not exist)
            'output_matching_projections': self._getFileName('refine3d_output_matching_projections', iter=iter, cls=cls, job=job),
            'output_parameter_file': self._getFileName('refine3d_output_parameters', iter=iter, cls=cls, job=job),
            'output_shift_file': self._getFileName('refine3d_output_shifts', iter=iter, cls=cls, job=job),
            'my_symmetry': self.refine_symmetry.get(),
            'first_particle': 0,
            'last_particle': 0,
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
            'calculate_matching_projections': boolToYN(False), # TODO determine
            'apply_2D_masking': boolToYN(self.classification_enableFocus.get()),
            'ctf_refinement': boolToYN(self.ctf_enable.get()),
            'normalize_particles': boolToYN(True),
            'invert_contrast': boolToYN(False), # TODO determine
            'exclude_blank_edges': boolToYN(False), 
            'normalize_input_3d': boolToYN(True), # TODO determine
            'threshold_input_3d': boolToYN(True),
        }

    def _readRefinementParameters(self, nClasses, iter, job):
        parameters = [[] for _ in range(nClasses)]

        # Read all the putput parameters
        for cls in range(nClasses):
            path = self._getExtraPath(self._getFileName('refine3d_output_parameters', iter=iter, cls=cls, job=job))

            # Read an entire file
            with FullFrealignParFile(path, 'r') as f:
                for row in f:
                    parameters[cls].append(row)

        # Ensure that everything was read
        for i in range(1, len(parameters)):
            assert(len(parameters[i]) == len(parameters[0]))

        return parameters

    def _prepareReconstruct(self, iter, cls, job):
        if iter == 0:
            # For the first step use the input volume as the input reconstruction
            createLink(
                self._getExtraPath(self._getFileName('input_volume', cls=cls)),
                self._getExtraPath(self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls))
            )
        else:
            # Use the previous reconstruction
            createLink(
                self._getExtraPath(self._getFileName('merge3d_output_reconstruction', iter=iter-1, cls=cls, rec='filtered')),
                self._getExtraPath(self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls))
            )

        # Use the parameters provided by the previous step
        createLink(
            self._getExtraPath(self._getFileName('refine3d_output_parameters', iter=iter, cls=cls, job=job)),
            self._getExtraPath(self._getFileName('reconstruct3d_input_parameters', iter=iter, cls=cls, job=job))
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

    def _getReconstructArgDictionary(self, iter, cls, job):
        inputParticles = self.input_particles.get()
        acquisition = inputParticles.getAcquisition()

        return {
            'input_particle_stack': self._getFileName('input_particles', job=job),
            'input_parameter_file': self._getFileName('reconstruct3d_input_parameters', iter=iter, cls=cls, job=job),
            'input_reconstruction': self._getFileName('reconstruct3d_input_reconstruction', iter=iter, cls=cls),
            'output_reconstruction_1': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='1'),
            'output_reconstruction_2': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='2'),
            'output_reconstruction_filtered': self._getFileName('reconstruct3d_output_reconstruction', iter=iter, cls=cls, job=job, rec='filtered'),
            'output_resolution_statistics': self._getFileName('reconstruct3d_output_statistics', iter=iter, cls=cls, job=job),
            'my_symmetry': self.refine_symmetry.get(),
            'first_particle': 0,
            'last_particle': 0,
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
            'normalize_particles': boolToYN(True),
            'invert_contrast': boolToYN(False), # TODO determine
            'exclude_blank_edges': boolToYN(False),
            'adjust_scores': boolToYN(self.reconstruction_adjustScore4Defocus.get()),
            'crop_images': boolToYN(self.reconstruction_enableAutoCrop.get()),
            'split_even_odd': boolToYN(True), # TODO does not work if false
            'center_mass': boolToYN(False),
            'use_input_reconstruction': boolToYN(False), # TODO determine
            'threshold_input_3d': boolToYN(True),
            'dump_arrays': boolToYN(True),
            'dump_file_1': self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=job, par='odd'),
            'dump_file_2': self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=job, par='evn'),
        }

    def _prepareMerge(self, nJobs, iter, cls):
        # Link the input seed files
        parities = ['evn', 'odd']
        for parity in parities:
            for i in range(nJobs):
                src = self._getExtraPath(self._getFileName('reconstruct3d_output_dump', iter=iter, cls=cls, job=i, par=parity))
                dst = self._getExtraPath(self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par=parity, seed=str(i+1)))
                createLink(src, dst)

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
            'output_resolution_statistics': self._getFileName('merge3d_output_resolution_statistics', iter=iter, cls=cls),
            'molecular_mass_kDa': self.input_molecularMass.get(),
            'inner_mask_radius': self.refine_innerMaskRadius.get(),
            'outer_mask_radius': self.refine_outerMaskRadius.get(),
            'dump_file_seed_1': self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par='odd', seed=''),
            'dump_file_seed_2': self._getFileName('merge3d_input_dump', iter=iter, cls=cls, par='evn', seed=''),
            'number_of_dump_files': nJobs,
        }


def distribute_work(n, m):
    """ Given an n, it distributes it into at most m similarly sized groups. It returns a list 
    with the amount elements in each group"""

    # Obtain the quotient and the reminder of dividing n by m
    q, r = divmod(n, m)

    # Distribute the work as evenly as possible, groups of the same size
    # as the quotient and the reminder spread across the first group. If 
    # m>n, q=0, so avoid adding zeros at the end of the array
    if q > 0:
        result = [q+1]*r + [q]*(m-r) # Always r<m
    else:
        result = [1]*r

    # Ensure that all the work has been distributed correctly
    assert(np.sum(result) == n)
    assert(len(result) == m)

    return result