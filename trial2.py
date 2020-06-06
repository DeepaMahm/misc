"""
- Deepa 6/6/2020
"""

import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import networkx as nx

from vmtk import pypes
from vmtk import vmtkscripts
from vtk.util.numpy_support import vtk_to_numpy

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
    tail = []
    head = []
    radius = []

    for edgeId in range(cleanedGraph.GetNumberOfCells()):
        edge = cleanedGraph.GetCell(edgeId)  # this will be a vtkLine, which only has two points

        edgeNode0 = edge.GetPointId(0)
        edgeNode1 = edge.GetPointId(1)
        tail.append(edgeNode0)
        head.append(edgeNode1)

        radius.append((cleanedGraph_dsa.CellData['Radius'][edgeId]))

    
