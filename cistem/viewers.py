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
from io import open

from pyworkflow.protocol.params import LabelParam
from pyworkflow.utils import removeExt, cleanPath
from pyworkflow.viewer import DESKTOP_TKINTER, Viewer
from pyworkflow.gui.project import ProjectWindow
from pwem.viewers import CtfView, EmPlotter, MicrographsView, EmProtocolViewer
import pwem.viewers.showj as showj
from pwem.objects import SetOfMovies

from .protocols import CistemProtCTFFind, CistemProtUnblur


def createCtfPlot(ctfSet, ctfId):
    ctfModel = ctfSet[ctfId]
    psdFn = ctfModel.getPsdFile()
    fn = removeExt(psdFn) + "_avrot.txt"
    gridsize = [1, 1]
    xplotter = EmPlotter(x=gridsize[0], y=gridsize[1],
                         windowTitle='CTFFind results')
    plot_title = getPlotSubtitle(ctfModel)
    a = xplotter.createSubPlot(plot_title, 'Spacial frequency (1/A)',
                               'Amplitude (or cross-correlation)',
                               yformat=False)
    
    legendName = ['Amplitude spectrum',
                  'CTF Fit',
                  'Quality of fit']
    for i in [2, 3, 4]:
        _plotCurve(a, i, fn)
    xplotter.showLegend(legendName)
    a.grid(True)
    xplotter.show()


def getPlotSubtitle(ctf):
    ang = u"\u212B"
    deg = u"\u00b0"
    def1, def2, angle = ctf.getDefocus()
    phSh = ctf.getPhaseShift()
    score = ctf.getFitQuality()
    res = ctf.getResolution()

    title = "Def1: %d %s | Def2: %d %s | Angle: %0.1f%s | " % (
        def1, ang, def2, ang, angle, deg)

    if phSh is not None:
        title += "Phase shift: %0.2f %s | " % (phSh, deg)

    title += "Fit: %0.1f %s | Score: %0.3f" % (res, ang, score)

    return title


OBJCMD_CTFFIND4 = "CTFFind plot results"

ProjectWindow.registerObjectCommand(OBJCMD_CTFFIND4, createCtfPlot)


class CtffindViewer(Viewer):
    """ Specific way to visualize SetOfCtf. """
    _environments = [DESKTOP_TKINTER]
    _targets = [CistemProtCTFFind]

    def _visualize(self, prot, **kwargs):
        outputCTF = getattr(prot, 'outputCTF', None)

        if outputCTF is not None:
            ctfView = CtfView(self._project, outputCTF)
            viewParams = ctfView.getViewParams()
            viewParams[showj.OBJCMDS] = "'%s'" % OBJCMD_CTFFIND4
            return [ctfView]
        else:
            return [self.infoMessage("The output SetOfCTFs has not been "
                                     "produced", "Missing output")]


def _plotCurve(a, i, fn):
    freqs = _getValues(fn, 0)
    curv = _getValues(fn, i)
    a.plot(freqs, curv)
    a.set_ylim([-0.1, 1.1])


def _getValues(fn, row):
    f = open(fn)
    values = []
    i = 0
    for line in f:
        if not line.startswith("#"):
            if i == row:
                values = line.split()
                break
            i += 1
    f.close()
    return values


class ProtUnblurViewer(EmProtocolViewer):
    _targets = [CistemProtUnblur]
    _environments = [DESKTOP_TKINTER]

    _label = 'viewer unblur'

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('doShowMics', LabelParam,
                      label="Show aligned micrographs?", default=True,
                      help="Show the output aligned micrographs.")
        form.addParam('doShowMicsDW', LabelParam,
                      label="Show aligned DOSE-WEIGHTED micrographs?",
                      default=True,
                      help="Show the output aligned dose-weighted "
                           "micrographs.")
        form.addParam('doShowMovies', LabelParam,
                      label="Show output movies?", default=True,
                      help="Show the output movies with alignment "
                           "information.")
        form.addParam('doShowFailedMovies', LabelParam,
                      label="Show FAILED movies?", default=True,
                      help="Create a set of failed movies "
                           "and display it.")

    def _getVisualizeDict(self):
        self._errors = []
        visualizeDict = {'doShowMics': self._viewParam,
                         'doShowMicsDW': self._viewParam,
                         'doShowMovies': self._viewParam,
                         'doShowFailedMovies': self._viewParam
                         }
        return visualizeDict

    def _viewParam(self, param=None):
        labelsDef = 'enabled id _filename _samplingRate '
        labelsDef += '_acquisition._dosePerFrame _acquisition._doseInitial '
        viewParamsDef = {showj.MODE: showj.MODE_MD,
                         showj.ORDER: labelsDef,
                         showj.VISIBLE: labelsDef,
                         showj.RENDER: None
                         }
        if param == 'doShowMics':
            if getattr(self.protocol, 'outputMicrographs', None) is not None:
                return [MicrographsView(self.getProject(),
                                        self.protocol.outputMicrographs)]
            else:
                return [self.errorMessage('No output micrographs found!',
                                          title="Visualization error")]

        elif param == 'doShowMicsDW':
            if getattr(self.protocol, 'outputMicrographsDoseWeighted', None) is not None:
                return [MicrographsView(self.getProject(),
                                        self.protocol.outputMicrographsDoseWeighted)]
            else:
                return [self.errorMessage('No output dose-weighted micrographs found!',
                                          title="Visualization error")]

        elif param == 'doShowMovies':
            if getattr(self.protocol, 'outputMovies', None) is not None:
                output = self.protocol.outputMovies
                return [self.objectView(output, viewParams=viewParamsDef)]
            else:
                return [self.errorMessage('No output movies found!',
                                          title="Visualization error")]

        elif param == 'doShowFailedMovies':
            self.failedList = self.protocol._readFailedList()
            if not self.failedList:
                return [self.errorMessage('No failed movies found!',
                                          title="Visualization error")]
            else:
                sqliteFn = self.protocol._getPath('movies_failed.sqlite')
                self.createFailedMoviesSqlite(sqliteFn)
                return [self.objectView(sqliteFn, viewParams=viewParamsDef)]

    def createFailedMoviesSqlite(self, path):
        inputMovies = self.protocol.inputMovies.get()
        cleanPath(path)
        movieSet = SetOfMovies(filename=path)
        movieSet.copyInfo(inputMovies)
        movieSet.copyItems(inputMovies,
                           updateItemCallback=self._findFailedMovies)

        movieSet.write()
        movieSet.close()

        return movieSet

    def _findFailedMovies(self, item, row):
        if item.getObjId() not in self.failedList:
            setattr(item, "_appendItem", False)
