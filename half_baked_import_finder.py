import ast
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt

file = open('import_test.py')
lines = file.readlines()
imports = []

Import = namedtuple("Import", ["module", "name", "alias"])

def get_imports(path, list_folders, list_files):
    with open(path, encoding ='utf8') as fh:
        # lines = fh.readlines()
        root = ast.parse(fh.read(), path)
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            for i in node.names:
                if 'skimage' in i.name:
                    temp = i.name.split('.')
                    if len(temp)>1:
#                                print(n.name.split('.')[1]+'/__init__')
                            yield i.name.split('.')[1]+'/__init__'
                        
        elif isinstance(node, ast.ImportFrom):
            if str(node.module).split('.')[0] == 'skimage':
                for n in node.names:
                    if len(str(node.module).split('.')) > 1:
                        if str(node.module).split('.')[1] not in list_folders:
#                                print(str(node.module).split('.')[1])
                            yield str(node.module).split('.')[1]
                        else:
                            if len(str(node.module).split('.'))==1:
#                                    print(str(node.name) + '/__init__')
                                yield str(node.name) + '/__init__'
                            elif len(str(node.module).split('.'))==2:
#                                    print(n.name)
                                yield n.name 
                            elif len(str(node.module).split('.'))>2:
#                                    print(str(node.module).split('.')[2])
                                yield str(node.module).split('.')[2]
                    else:
                        yield '__init__'
                        
                
            else:
                if node.module == None:
                    for n in node.names:
                        if n.name in list_folders:
#                                print(n.name + '/__init__')
                            yield n.name + '/__init__'
                        else:
#                                print('/__init__')
                            yield '__init__'
                else:
                    temp = str(node.module).split('.')
                    if temp[0] in list_folders:
                        if len(temp)==1:
                            for n in node.names:
#                                    print(n.name.split('.')[0])
                                yield n.name.split('.')[0]
                        else:
#                                print(temp[1])
                            yield temp[1]
                    elif temp[0] in list_files:
#                            print(temp[0])
                        yield temp[0]


"""
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split('.')
        else:
            continue

        for n in node.names:
            yield Import(module, n.name.split('.'), n.asname)
""" 


  
import os
start = 'scikit-image/skimage'
g = nx.DiGraph()

list_folders = []
list_files = []
flag = True #no pyx
for i in os.listdir(start):
    if '.py' not in i and i != '__pycache__':
        list_folders.append(i)

for i in list_folders:
    for file in os.listdir(start+'/'+i):
        if file[-3:] == '.py':
            list_files.append(file[:-3])
        elif file[-4:] == '.pyx':
            list_files.append(file[:-4])
weight = 1

for i in os.listdir(start):
    if '.py' not in i and i != '__pycache__':
        for file in os.listdir(start+'/'+i):
            if file[-3:] == '.py': #or  file[-4:] == '.pyx':
                for reference in get_imports(start+'/'+i+'/'+file, list_folders, list_files):
                    if ((reference[:2]=='__' or reference[0] != '_') and (file[0]!='_' or file[:2]=='__')) or flag:
                        if file == '__init__.py':
                            name = i+'/__init__'
                        elif file[-3:] == '.py':
                            name = file[:-3]
                        elif file[-4:] == '.pyx':
                            name = file[:-4]
                        else:
                            name = 'error'
                        g.add_edge(name, reference, weight=weight)
                    else:
                        pass

print(nx.dag_longest_path(g))

#nx.draw_networkx(g, node_size=30)
#plt.draw()
#plt.show()
arr = []
path = dict(nx.all_pairs_shortest_path(g))
for start in path:
    for end in path[start]:
        if len(path[start][end]) > 2:
            arr.append((len(path[start][end]), path[start][end]))

nx.write_gml(g,'tree.txt')

printable = '\n'.join([str(i[1]) for i in sorted(arr)[::-1]])
file = open('paths.txt', 'w+')
file.write(printable)
file.close()

# print(printable)
"""
                if len(reference[0])>0 and reference[0][0] == 'skimage':
                    if len(reference[0]) == 1:
                        folders[key].files[file].folder_references += reference[1]
                    elif len(reference[0]) == 2:
                        folders[key].files[file].file_references += reference[1]
                        folders[key].files[file].folder_references.append(references[0][1])
                    else:
                        folders[key].files[file].file_references.append(references[0][2])
                        folders[key].files[file].folder_references.append(references[0][1])
"""
