# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca) [1]
# *
# * [1] Department of Anatomy and Cell Biology, McGill University
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
import numpy as np
from itertools import izip

from pyworkflow.em.data import Coordinate, SetOfClasses2D, SetOfAverages
from pyworkflow.em import ImageHandler
from pyworkflow.utils.path import replaceBaseExt, join, exists


def rowToCtfModel(ctfRow, ctfModel):
    defocusU = float(ctfRow.get('DF1'))
    defocusV = float(ctfRow.get('DF2'))
    defocusAngle = float(ctfRow.get('ANGAST'))
    ctfModel.setStandardDefocus(defocusU, defocusV, defocusAngle)


def parseCtffind4Output(filename):
    """ Retrieve defocus U, V and angle from the
    output file of the ctffind4 execution.
    """
    result = None
    if os.path.exists(filename):
        f = open(filename)
        for line in f:
            if not line.startswith("#"):
                result = tuple(map(float, line.split()[1:]))
                # Stop reading. In ctffind4-4.0.15 output file has additional lines.
                break
        f.close()
    return result


def setWrongDefocus(ctfModel):
    ctfModel.setDefocusU(-999)
    ctfModel.setDefocusV(-1)
    ctfModel.setDefocusAngle(-999)
    
    
def readCtfModel(ctfModel, filename):
    result = parseCtffind4Output(filename)
    if result is None:
        setWrongDefocus(ctfModel)
        ctfFit, ctfResolution, ctfPhaseShift = -999, -999, -999
    else:
        defocusU, defocusV, defocusAngle, ctfPhaseShift, ctfFit, ctfResolution = result
        ctfModel.setStandardDefocus(defocusU, defocusV, defocusAngle)
    ctfModel.setFitQuality(ctfFit)
    ctfModel.setResolution(ctfResolution)

    # Avoid creation of phaseShift
    ctfPhaseShiftDeg = np.rad2deg(ctfPhaseShift)
    if ctfPhaseShiftDeg != 0:
        ctfModel.setPhaseShift(ctfPhaseShiftDeg)


def readShiftsMovieAlignment(shiftFn):
    f = open(shiftFn, 'r')
    xshifts = []
    yshifts = []

    for line in f:
        l = line.strip()
        if l.startswith('image #'):
            parts = l.split()
            xshifts.append(float(parts[-2].rstrip(',')))
            yshifts.append(float(parts[-1]))
    f.close()
    return xshifts, yshifts


def writeShiftsMovieAlignment(movie, shiftsFn, s0, sN):
    movieAlignment = movie.getAlignment()
    shiftListX, shiftListY = movieAlignment.getShifts()

    # Generating metadata for global shifts
    a0, aN = movieAlignment.getRange()
    alFrame = a0

    if s0 < a0:
        diff = a0 - s0
        initShifts = "0.0000 " * diff
    else:
        initShifts = ""

    if sN > aN:
        diff = sN - aN
        finalShifts = "0.0000 " * diff
    else:
        finalShifts = ""

    shiftsX = ""
    shiftsY = ""
    for shiftX, shiftY in izip(shiftListX, shiftListY):
        if alFrame >= s0 and alFrame <= sN:
            shiftsX = shiftsX + "%0.4f " % shiftX
            shiftsY = shiftsY + "%0.4f " % shiftY
        alFrame += 1

    f = open(shiftsFn, 'w')
    shifts = (initShifts + shiftsX + " " + finalShifts + "\n"
              + initShifts + shiftsY + " " + finalShifts)
    f.write(shifts)
    f.close()


def readSetOfCoordinates(workDir, micSet, coordSet, highRes):
    """ Read from cisTEM .plt files. """
    for mic in micSet:
        micCoordFn = join(workDir, replaceBaseExt(mic.getFileName(), 'plt'))
        readCoordinates(mic, micCoordFn, coordSet, highRes)


def readCoordinates(mic, fn, coordsSet, highRes):
    if exists(fn):
        with open(fn, 'r') as f:
            for line in f:
                values = line.strip().split()
                x_ang, y_ang = float(values[0]), float(values[1])
                newPix, _ = _findNewPixelSize(mic.getXDim(), mic.getYDim(),
                                              mic.getSamplingRate(), highRes)
                x = int((x_ang - 1) * mic.getSamplingRate() / newPix)
                y = int((y_ang - 1) * mic.getSamplingRate() / newPix)
                coord = Coordinate()
                coord.setPosition(x, y)
                coord.setMicrograph(mic)
                coordsSet.append(coord)
        f.close()


def writeReferences(inputSet, outputFn):
    """
    Write references star and stack files from SetOfAverages or SetOfClasses2D.
    Params:
        inputSet: the input SetOfParticles to be converted
        outputFn: where to write the output files.
    """
    ih = ImageHandler()

    def _convert(item, i):
        index = i + 1
        ih.convert(item, (index, outputFn))
        item.setLocation(index, outputFn)

    if isinstance(inputSet, SetOfAverages):
        for i, img in enumerate(inputSet):
            _convert(img, i)
    elif isinstance(inputSet, SetOfClasses2D):
        for i, rep in enumerate(inputSet.iterRepresentatives()):
            _convert(rep, i)
    else:
        raise Exception('Invalid object type: %s' % type(inputSet))


def _findNewPixelSize(micX, micY, orig_pixel_size, high_res):
    """ Reimplemented from core/functions.cpp of cisTEM. """
    # First we look for a nice factorizable micrograph dimension
    # which gives approximately the desired pixel size
    if micX == micY:
        wanted_int = int(micX * orig_pixel_size / high_res * 2.0)
        newMicX = _returnClosestFactorizedUpper(wanted_int, 5, True)
        newMicY = newMicX
    elif micX > micY:
        wanted_int = int(micY * orig_pixel_size / high_res * 2.0)
        newMicY = _returnClosestFactorizedUpper(wanted_int, 5, True)
        newMicX = _myroundint(float(newMicY) / float(micY) * float(micX))
    else:
        wanted_int = int(micX * orig_pixel_size / high_res * 2.0)
        newMicX = _returnClosestFactorizedUpper(wanted_int, 5, True)
        newMicY = _myroundint(float(newMicX) / float(micX) * float(micY))

    # calculate new pixel size
    new_pixX = orig_pixel_size * float(micX) / float(newMicX)
    new_pixY = orig_pixel_size * float(micY) / float(newMicY)

    return new_pixX, new_pixY


def _myroundint(a):
    """ Reimplemented from core/functions.h of cisTEM. """
    return int(a + 0.5) if a > 0 else int(a - 0.5)


def _returnClosestFactorizedUpper(wanted_int, largest_factor, enforce_even=False):
    """ Reimplemented from core/functions.cpp of cisTEM. """
    number = 0
    remainder = wanted_int
    if enforce_even:
        temp_int = wanted_int
        if (temp_int % 2) != 0:
            temp_int += 1
        for number in range(temp_int, 10000 * wanted_int, 2):
            remainder = number
            for factor in range(2, largest_factor+1, 1):
                if remainder == 1:
                    break
                else:
                    temp_int = remainder % factor
                    while temp_int == 0:
                        remainder /= factor
                        temp_int = remainder % factor
            if remainder == 1:
                break
    else:
        raise Exception('Not implemented!')

    return number
