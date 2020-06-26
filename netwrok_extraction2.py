"""
- computing quantification results of a network model
- created by Deepa Maheshvare
- Indian Institute of Science
- 25/06/2020
"""

import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import pandas as pd

from pprint import pprint
from vmtk import vmtkscripts
from vtk.util.numpy_support import vtk_to_numpy
from collections import OrderedDict
from typing import List


def write_ployData_output(surface, file: str) -> None:
    """
    write polyData output: for visualization in paraview
    :param surface:
    :param file:
    :return:
    """
    surfacewriter = vmtkscripts.vmtkSurfaceWriter()
    surfacewriter.Surface = surface
    surfacewriter.OutputFileName = f"network_{file}.vtp"
    surfacewriter.Execute()


def extract_segment_lengths(networkExtraction) -> List:
    """
    Generates model output containing segment lengths in CellData
    :param newtork:
    :return:
    """
    network = networkExtraction.Network

    centerlinesGeometry = vmtkscripts.vmtkCenterlineGeometry()
    centerlinesGeometry.Centerlines = network
    centerlinesGeometry.Execute()
    centerline = centerlinesGeometry.Centerlines

    #  output
    write_ployData_output(surface=centerline, file="length")

    # table data
    centerline_dsa = dsa.WrapDataObject(centerline)
    length = []
    for edgeId in range(dsa.WrapDataObject(network).GetNumberOfCells()):
        length.append(centerline_dsa.CellData['Length'][edgeId])

    return length


def extract_segment_radius(networkExtraction) -> List:
    """
    Generates model output containing segment radii in CellData
    :param newtork:
    :return:
    """
    graph = networkExtraction.GraphLayout

    #  output
    write_ployData_output(surface=graph, file="radius")

    # table data
    graph_dsa = dsa.WrapDataObject(graph)
    radius = []
    for edgeId in range(dsa.WrapDataObject(graph).GetNumberOfCells()):
        radius.append(graph_dsa.CellData['Radius'][edgeId])

    return radius


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
    length = extract_segment_lengths(networkExtraction=networkExtraction)
    pprint(length)

    # generate ployData: radii
    radius = extract_segment_radius(networkExtraction=networkExtraction)
    pprint(radius)

    # get coordinates of terminal/free ends and N-furcation points in network
    pos = get_coordinates(networkExtraction=networkExtraction)
    pprint(pos)
