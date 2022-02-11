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

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam, BooleanParam
from pyworkflow.gui.plotter import getHexColorList
from pwem.viewers import TableView, Classes3DView, EmPlotter

from cistem.protocols.protocol_3d_classification import CistemProt3DClassification

from cistem.convert.FrealignParFile import FullFrealignParFile

import numpy as np
import matplotlib.pyplot as plt

class Cistem3DClassificationViewer(ProtocolViewer):
    _label = 'viewer classification 3d'
    _targets = [CistemProt3DClassification]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Classification')
        form.addParam('iteration', IntParam, label='Iteration',
                        default=-1,
                        help='Iteration to be shown. 0 is first, 1 second and '
                        'so on. Negative numbers refer backwards so that -1'
                        'represents the last one, -2 the penultimate and so on')
        form.addParam('displayClasses', LabelParam, label='Classification')
        form.addParam('displayScores', LabelParam, label='Scores')
        form.addParam('displayScoreImage', LabelParam, label='Score Image')

        form.addSection(label='Convergence')
        form.addParam('considerReferences', BooleanParam, label='Consider previous refinement(s)',
                        default=True,
                        help='If a reference refinement is specified for this protocol, this option '
                        'specifies if previous refinement chain is included in the plot')
        form.addParam('displayRefinementScores', LabelParam, label='Refinement scores')
        form.addParam('displayClassificationScores', LabelParam, label='Classification scores')
        form.addParam('displayClassSizeDistribution', LabelParam, label='Class size distribution')
        form.addParam('displayClassDistributionImage', LabelParam, label='Class distribution image')

    def _getVisualizeDict(self):
        return {
            'displayClasses': self._displayClasses,
            'displayScores': self._displayScores,
            'displayScoreImage': self._displayScoreImage,
            'displayRefinementScores': self._displayRefinementScores,
            'displayClassificationScores': self._displayClassificationScores,
            'displayClassSizeDistribution': self._displayClassSizeDistribution,
            'displayClassDistributionImage': self._displayClassDistributionImage
        }
    
    # --------------------------- DEFINE display functions ----------------------
    def _displayClasses(self, e):
        classes = self.protocol._createOutput3dClasses(self.protocol._getClassCount(), self._getIteration())
        return [Classes3DView(self._project, classes.strId(), classes.getFileName())]
    
    def _displayScores(self, e):
        iter = self._getIteration()
        nClasses = self.protocol._getClassCount()

        classification = self._readClassification(iter)
        scores = self._readRefinementColumn(nClasses, iter, 'score')
        return self._showClassificationTable(scores, classification, 'Classification')
    
    def _displayScoreImage(self, e):
        iter = self._getIteration()
        nClasses = self.protocol._getClassCount()
        scores = self._readRefinementColumn(nClasses, iter, 'score')

        # Plot scores as an image
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(np.array(scores).T, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Class number')
        ax.set_ylabel('Particle index')
        ax.set_title('Score')

        return [fig]

    def _displayRefinementScores(self, e):
        scoreStatistics = self._readProtocolChainRefinementStatistics('score')
        
        plot = EmPlotter(y=self.protocol._getClassCount()) # A column for each class

        for cls in range(self.protocol._getClassCount()):
            maximums = [stat[cls]['max'] for stat in scoreStatistics]
            minimums = [stat[cls]['min'] for stat in scoreStatistics]
            averages = [stat[cls]['avg'] for stat in scoreStatistics]
            iterations = list(range(len(averages)))

            # Plot the results
            plot.createSubPlot(f'Score convergence class {cls}', 'Iteration', 'Score')
            plot.plotData(iterations, minimums, 'red')
            plot.plotData(iterations, averages, 'green')
            plot.plotData(iterations, maximums, 'blue')
            plot.showLegend(['min', 'avg', 'max'])

        return [plot]

    def _displayClassificationScores(self, e):
        scoreStatistics = self._readProtocolChainClassificationStatistics('score')
        maximums = [stat['max'] for stat in scoreStatistics]
        minimums = [stat['min'] for stat in scoreStatistics]
        averages = [stat['avg'] for stat in scoreStatistics]
        iterations = list(range(len(averages)))
        
        # Plot the results
        plot = EmPlotter()
        plot.createSubPlot('Score convergence', 'Iteration', 'Score')
        plot.plotData(iterations, minimums, 'red')
        plot.plotData(iterations, averages, 'green')
        plot.plotData(iterations, maximums, 'blue')
        plot.showLegend(['min', 'avg', 'max'])

        return [plot]

    def _displayClassSizeDistribution(self, e):
        classificationSizes = self._readProtocolChainClassificationSizes()
        colors = getHexColorList(self.protocol._getClassCount())
        iterations = list(range(len(classificationSizes)))
        nParticles = self.protocol._getParticleCount()
        labels = [f'Class {i}' for i in range(len(colors))]
        
        plot = EmPlotter(x=2)

        # Plot the absolute sizes
        plot.createSubPlot('Class absolute sizes', 'Iteration', 'Particle Count')
        for cls, color in enumerate(colors):
            sizes = [x[cls] for x in classificationSizes]
            plot.plotData(iterations, sizes, color)
        plot.showLegend(labels)

        # Plot the relative sizes
        plot.createSubPlot('Class relative sizes', 'Iteration', 'Particle %')
        for cls, color in enumerate(colors):
            sizes = [x[cls]/nParticles*100.0 for x in classificationSizes]
            plot.plotData(iterations, sizes, color)
        plot.showLegend(labels)

        return [plot]

    def _displayClassDistributionImage(self, e):
        classification = self._readProtocolChainClassification()

        # Plot classification as an image
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(np.array(classification).T, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Particle index')
        ax.set_title('Particle classification evolution')
        
        return [fig]

    # --------------------------- UTILS functions -----------------------------
    def _getIteration(self):
        return self.protocol._getIter(self.iteration.get())

    def _readClassification(self, iter):
        result = []

        path = self.protocol._getExtraPath(self.protocol._getFileName('output_classification', iter=iter))
        with FullFrealignParFile(path, 'r') as f:
            for row in f:
                result.append(row['film'])

        return result

    def _readRefinementColumn(self, nClasses, iter, column):
        result = []

        for cls in range(nClasses):
            current = []

            path = self.protocol._getExtraPath(self.protocol._getFileName('output_refinement', iter=iter, cls=cls))
            with FullFrealignParFile(path, 'r') as f:
                for row in f:
                    current.append(row[column])

            result.append(current)

        for i in range(1, len(result)):
            assert(len(result[0]) == len(result[i]))

        return result

    def _showClassificationTable(self, scores, classification, title=None):
        # Ensemble the header
        header = tuple([f'Class {i} score' for i in range(len(scores))] + ['Classification'])

        data = list(zip(*(scores + [classification])))
        return [TableView(header, data, None, title)]

    def _getProtocolChain(self):
        result = [self.protocol]

        # Recursively obtain the reference refinements if requested
        if self.considerReferences.get():
            while result[-1]._getReferenceRefinement() is not None:
                result.append(result[-1]._getReferenceRefinement())

            # First refinement in the front:
            result.reverse()

        return result

    def _readProtocolChainRefinementStatistics(self, col):
        result = []

        protocols = self._getProtocolChain()
        for protocol in protocols:
            for iter in range(protocol._getCycleCount()):
                result.append([])
                for cls in range(protocol._getClassCount()):
                    path = protocol._getExtraPath(protocol._getFileName('output_refinement', iter=iter, cls=cls))
                    with FullFrealignParFile(path, 'r') as f:
                        result[-1].append({
                            'max': f.getMax(col),
                            'min': f.getMin(col),
                            'avg': f.getAverage(col)
                        })

        return result

    def _readProtocolChainClassificationStatistics(self, col):
        result = []

        protocols = self._getProtocolChain()
        for protocol in protocols:
            for iter in range(protocol._getCycleCount()):
                path = protocol._getExtraPath(protocol._getFileName('output_classification', iter=iter))
                with FullFrealignParFile(path, 'r') as f:
                    result.append({
                        'max': f.getMax(col),
                        'min': f.getMin(col),
                        'avg': f.getAverage(col)
                    })

        return result

    def _readProtocolChainClassification(self):
        result = []

        protocols = self._getProtocolChain()
        for protocol in protocols:
            # Get the classification column
            for iter in range(protocol._getCycleCount()):
                path = protocol._getExtraPath(protocol._getFileName('output_classification', iter=iter))
                with FullFrealignParFile(path, 'r') as f:
                    result.append([row['film'] for row in f])

        return result

    def _readProtocolChainClassificationSizes(self):
        result = []

        nClasses = self.protocol._getClassCount()
        classifications = self._readProtocolChainClassification()

        for iteration in classifications:
            data = [0] * nClasses
            for particleCls in iteration:
                data[particleCls] += 1
            result.append(data)

        return result