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