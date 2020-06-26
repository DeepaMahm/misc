"""
- computing quantification results of a network model
- created by Deepa
- 25/06/2020
"""
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import numpy as np
import pandas as pd

from pprint import pprint
from vmtk import vmtkscripts
from vtk.util.numpy_support import vtk_to_numpy
from collections import OrderedDict


def extract_segment_lengths(networkExtraction):
    """
    Generates model output containing segment lengths in CellData
    :param newtork:
    :return:
    """
    network = networkExtraction.Network
    centerlinesGeometry = vmtkscripts.vmtkCenterlineGeometry()
    centerlinesGeometry.Centerlines = network
    centerlinesGeometry.Execute()

    surfacewriter = vmtkscripts.vmtkSurfaceWriter()
    surfacewriter.Surface = centerlinesGeometry.Centerlines
    surfacewriter.OutputFileName = "vesseltree_network_length.vtp"
    surfacewriter.Execute()


def extract_segment_radius(networkExtraction):
    """
    Generates model output containing segment radii in CellData
    :param newtork:
    :return:
    """
    surfacewriter = vmtkscripts.vmtkSurfaceWriter()
    surfacewriter.Surface = networkExtraction.GraphLayout
    surfacewriter.OutputFileName = "vesseltree_network_raidus.vtp"
    surfacewriter.Execute()


def get_coordinates(networkExtraction):
    """
    get the x,y,z coordinates of head and tail nodes in graph
    :param vtkarray:
    :param graph:
    :return: df containing xyz coordinates
    """
    graph = networkExtraction.GraphLayout

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(graph)  # set the data to processed
    cleaner.Update()  # execute the algorithm
    cleanedGraph = cleaner.GetOutput()

    cleanedGraph_dsa = dsa.WrapDataObject(graph)

    start = []
    end = []

    for edgeId in range(cleanedGraph.GetNumberOfCells()):
        edge = cleanedGraph.GetCell(edgeId)  # this will be a vtkLine, which only has two points

        edgeNode0 = edge.GetPointId(0)
        edgeNode1 = edge.GetPointId(1)
        start.append(edgeNode0)
        end.append(edgeNode1)

    pos = vtk_to_numpy(cleanedGraph_dsa.Points)
    nbranch = cleanedGraph.GetNumberOfCells()
    pos = pos.reshape(nbranch, 2, 3)

    xyz = OrderedDict()
    for i in range(len(end)):
        xyz[f'{start[i]}'] = pos[i][0]
        xyz[f'{end[i]}'] = pos[i][1]

    pos = pd.DataFrame(xyz)
    pos = pos.transpose()
    pos.columns = ['xpos', 'ypos', 'zpos']
    return pos


if __name__ == '__main__':

    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = 'vesseltree.stl'
    reader.Execute()

    # network extraction
    networkExtraction = vmtkscripts.vmtkNetworkExtraction()
    networkExtraction.Surface = reader.Surface
    networkExtraction.Execute()

    # generate ployData: lengths
    extract_segment_lengths(networkExtraction=networkExtraction)
    # generate ployData: radii
    extract_segment_radius(networkExtraction=networkExtraction)
    # get coordinates of terminal/free ends and N-furcation points in network
    pos =get_coordinates(networkExtraction=networkExtraction)
