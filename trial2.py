import os
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vmtk import pypes
from pprint import pprint
from vmtk import vmtkscripts
from vtk.util.numpy_support import vtk_to_numpy
from collections import OrderedDict
from matpancreas.settings_model import RESULTS_DIR
from matpancreas.utils.io_utils_py import dataframes_to_xlsx


def write_output(d):
    f_path = os.path.join(RESULTS_DIR, 'imaging', 'data.xlsx')
    dataframes_to_xlsx(f_path=f_path, d=d, startrow=0)


def get_coordinates(vtkarray, graph):
    """
    get the x,y,z coordinates of head and tail nodes in graph
    :param vtkarray:
    :param graph:
    :return: df containing xyz coordinates
    """
    tail = graph['tail']
    head = graph['head']
    pos = vtk_to_numpy(vtkarray.Points)

    pos = pos.reshape(14, 2, 3)

    xyz = OrderedDict()
    l = []
    for i in range(len(head)):
        xyz[f'{tail[i]}'] = pos[i][0]
        xyz[f'{head[i]}'] = pos[i][1]
        l.append(np.linalg.norm(pos[i][0] - pos[i][1]))

    df = pd.DataFrame(xyz)
    df = df.transpose()
    df.columns = ['xpos', 'ypos', 'zpos']
    graph['length'] = l
    return df, graph


if __name__ == '__main__':

    # ref google groups: vmtknetworkextraction generate an empty network
    # ref: http://www.vmtk.org/tutorials/PypesBasic.html
    # ref: https://blog.kitware.com/improved-vtk-numpy-integration-part-2/
    # ref: http://www.vmtk.org/tutorials/WorkingWithNumpyArrays.html
    # ref: VMTK mailing list:  Centerline length

    # command
    # vmtkcenterlines -ifile vesseltree.stl --pipe vmtkbranchextractor -ofile vesseltree_clsp.vtp

    """
    vmtkCommand = '''vmtkcenterlines -ifile segment1.stl -endpoints 1
                     --pipe vmtkcenterlineresampling -length 0.1 -ofile vesseltree_centerline.vtp'''

    p = pypes.Pype()
    p.SetArgumentsString(vmtkCommand)
    p.ParseArguments()
    p.Execute()

    vmtkCommand = '''vmtkcenterlinegeometry -ifile vesseltree_centerline.vtp -ofile vesseltree_centerline.vtp'''

    p = pypes.Pype()
    p.SetArgumentsString(vmtkCommand)
    p.ParseArguments()
    p.Execute()

    centerlineReader = vmtkscripts.vmtkSurfaceReader()
    centerlineReader.InputFileName = 'vesseltree_centerline.vtp'
    centerlineReader.Execute()

    # centerlines
    # ref: https://mail.google.com/mail/u/0/?tab=rm&ogbl#search/evan/FMfcgxwHMsSvqTVwpztSNlcSFTvfBnvb
    clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
    clNumpyAdaptor.Centerlines = centerlineReader.Surface
    clNumpyAdaptor.Execute()

    numpyCenterlines = clNumpyAdaptor.ArrayDict

    pprint(numpyCenterlines.keys())
    pprint(numpyCenterlines['PointData'].keys())
    pprint(numpyCenterlines['CellData'].keys())
    pprint(numpyCenterlines['PointData']['MaximumInscribedSphereRadius'])

    for cellDataKey in numpyCenterlines['CellData']:
        if cellDataKey == 'CellPointIds':
            pass
        else:
            print('Shape of ', cellDataKey, ' = ', numpyCenterlines['CellData'][cellDataKey].shape)

    exit()
    """
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

    g = pd.DataFrame({'tail': tail, 'head': head, 'radius': radius})
    # [pos, g] = get_coordinates(vtkarray=cleanedGraph_dsa, graph=g)
    write_output(d={'graph': g}) #, 'pos': pos


    """

    ed_ls = [(x, y) for x, y in zip(tail, head)]
    G = nx.OrderedGraph()
    G.add_edges_from(ed_ls)
    nx.draw(G)
    pprint(nx.incidence_matrix(G, oriented=True).toarray())
    plt.savefig('fig.png', bbox_inches='tight')

    pprint(network_dsa.PointData.keys())
    pprint(network_dsa.CellData.keys())
    pprint(network_dsa.PointData['Radius'])
    pprint(network_dsa.CellData['Topology'])

    pprint(graph_dsa.PointData.keys())
    pprint(graph_dsa.CellData.keys())
    pprint(graph_dsa.CellData['Radius'])

    """
