# Based on:
# https://github.com/scipion-em/scipion-em-relion/blob/9aa81ef5950766eea97d0041c860ee17560bb612/relion/convert/__init__.py#L95

from pwem.objects.data import CTFModel, Transform
import numpy as np

class ClassesLoader:
    """ Helper class to read classes information from star files produced
    by Cistem classification runs (2D or 3D).
    """
    def __init__(self, repPaths, statisticsPaths, particleDataPaths, alignment):
        from . import FullFrealignParFile, FrealignStatisticsFile
        self._classRepresentatives = repPaths
        self._classStatistics = [FrealignStatisticsFile(p, 'r') for p in statisticsPaths]
        self._particleData = [[FullFrealignParFile(p, 'r') for p in c] for c in particleDataPaths]
        self._alignment = alignment

    def fillClasses(self, clsSet):
        clsSet.classifyItems(updateItemCallback=self._updateParticle,
                             updateClassCallback=self._updateClass,
                             itemDataIterator=self._buildParticleIter(),
                             doClone=False)

    def _updateParticle(self, item, row):
        row = self._processRow(row) # Select only the active class
        self._updateClassId(item, row)
        self._updateCtf(item, row)
        self._updateTransform(item, row)

        #if getattr(self, '__updatingFirst', True):
        #    self._reader.createExtraLabels(item, row, PARTICLE_EXTRA_LABELS)
        #    self.__updatingFirst = False
        #else:
        #    self._reader.setExtraLabels(item, row)

    def _updateClassId(self, item, row):
        item.setClassId(row['class_id']+1)

    def _updateCtf(self, item, row):
        if not item.hasCTF():
            item.setCTF(CTFModel())
        ctf = item.getCTF()
        ctf.setStandardDefocus(row['defocus_u'], row['defocus_v'], row['defocus_angle'])

    def _updateTransform(self, item, row):
        from .convert import matrixFromGeometry
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

    def _buildParticleIter(self):
        from itertools import chain
        clsIterators = [chain.from_iterable(c) for c in self._particleData] # Chain all the paths for a given class
        return iter(zip(*clsIterators))

    def _processRow(self, row):
        # Selects the active class in the row
        result = None

        for i, subrow in enumerate(row):
            if(subrow['film'] == 1): # used as include
                assert(result is None) # Only 1 be set
                result = subrow
                result['class_id'] = i

        assert(result is not None) # Should be set
        return result