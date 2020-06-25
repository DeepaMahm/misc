"""
- computing quantification results of a network model
- created by Deepa
- 25/06/2020
"""
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import numpy as np
import pandas as pd

from vmtk import pypes
from pprint import pprint
from vmtk import vmtkscripts
from vtk.util.numpy_support import vtk_to_numpy
from collections import OrderedDict


def get_coordinates(vtkarray, df):
    """
    get the x,y,z coordinates of head and tail nodes in graph
    :param vtkarray:
    :param graph:
    :return: df containing xyz coordinates
    """
    start = df['start']
    end = df['end']

    pos = vtk_to_numpy(vtkarray.Points)

    pos = pos.reshape(nbranch, 2, 3)

    xyz = OrderedDict()
    l = []
    for i in range(len(end)):
        xyz[f'{start[i]}'] = pos[i][0]
        xyz[f'{end[i]}'] = pos[i][1]
        l.append(np.linalg.norm(pos[i][0] - pos[i][1]))

    pos = pd.DataFrame(xyz)
    pos = pos.transpose()
    pos.columns = ['xpos', 'ypos', 'zpos']
    df['length'] = l
    return pos, df


if __name__ == '__main__':

    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = 'vesseltree.stl'
    reader.Execute()

    # network extraction
    networkExtraction = vmtkscripts.vmtkNetworkExtraction()
    networkExtraction.Surface = reader.Surface
    networkExtraction.Execute()

    network = networkExtraction.Network
    graph = networkExtraction.GraphLayout

    network_dsa = dsa.WrapDataObject(network)
    pprint(network_dsa.GetNumberOfCells())
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(graph)  # set the data to processed
    cleaner.Update()  # execute the algorithm
    cleanedGraph = cleaner.GetOutput()

    cleanedGraph_dsa = dsa.WrapDataObject(graph)

    pprint(cleanedGraph_dsa.GetNumberOfCells())
    start = []
    end = []
    radius = []

    nbranch = cleanedGraph.GetNumberOfCells()
    print(nbranch)

    for edgeId in range(cleanedGraph.GetNumberOfCells()):
        edge = cleanedGraph.GetCell(edgeId)  # this will be a vtkLine, which only has two points

        edgeNode0 = edge.GetPointId(0)
        edgeNode1 = edge.GetPointId(1)
        start.append(edgeNode0)
        end.append(edgeNode1)

        radius.append((cleanedGraph_dsa.CellData['Radius'][edgeId]))

    df = pd.DataFrame({'start': start, 'end': end, 'radius': radius})
    [pos, df] = get_coordinates(vtkarray=cleanedGraph_dsa, df=df)
    pprint(pos)
    pprint(df)
