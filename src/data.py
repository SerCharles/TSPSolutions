import os
import re
import xml.dom.minidom
import json
import glob
import numpy as np

def read_tsp_data(base_name):
    """Read one tsp file

    Args:
        base_name [str]: [the base name of the file, a280 for example] 
        
    Returns:
        N [int]: [the number of nodes]
        graph [numpy double array], [N * N]: [the graph distances]
        best_result [double]: [the best result of the problem]
    """
    #base info
    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, 'data'))
    file_name = os.path.join(base_path, base_name + '.xml')
    best_name = os.path.join(base_path, 'optimal.json')
    pattern = re.compile(r'\d+')
    N = int(pattern.findall(base_name)[0])
    graph = np.zeros((N, N), dtype=np.float64)
    
    #read xml
    root = xml.dom.minidom.parse(file_name).documentElement.getElementsByTagName('graph')[0]
    vertexs = root.getElementsByTagName('vertex')
    for start in range(N):
        vertex = vertexs[start]
        edges = vertex.getElementsByTagName('edge')
        for edge in edges:
            end = int(edge.firstChild.data)
            cost = edge.getAttribute('cost')
            graph[start][end] = cost  

    #read optimal
    f = open(best_name, 'r', encoding='utf8')
    json_data = json.load(f)
    best_result = json_data[base_name]
    f.close()
    return N, graph, best_result
