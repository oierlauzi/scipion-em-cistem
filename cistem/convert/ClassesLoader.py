# Based on:
# https://github.com/scipion-em/scipion-em-relion/blob/9aa81ef5950766eea97d0041c860ee17560bb612/relion/convert/__init__.py#L95

from pwem.objects.data import CTFModel, Transform
import numpy as np

class ClassesLoader:
    """ Helper class to read classes information from star files produced
    by Cistem classification runs (2D or 3D).
    """
    def __init__(self, repPaths, statisticsPaths, particleDataPath, alignment):
        from . import FullFrealignParFile, FrealignStatisticsFile
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