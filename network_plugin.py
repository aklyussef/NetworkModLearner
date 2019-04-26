import os
import sys
from collections import defaultdict

def _open_gz(path, mode):
    import gzip
    return gzip.open(path, mode=mode)


def _open_bz2(path, mode):
    import bz2
    return bz2.BZ2File(path, mode=mode)

mode = 'r'

# To handle new extensions, define a function accepting a `path` and `mode`.
# Then add the extension to _dispatch_dict.
_dispatch_dict = defaultdict(lambda: open)
_dispatch_dict['.gz'] = _open_gz
_dispatch_dict['.bz2'] = _open_bz2
_dispatch_dict['.gzip'] = _open_gz

def is_string(tstr):
    return isinstance(tstr,str)

comment_predecessors = ['*','#']
def read_file(filepath):
    """ This function reads network files
    Parameters
    ----------
    Takes filepath as argument
    Returns
    -------
    :return List of Tuples representing nodes and the edges between them
    or None if file doesn't exist or isn't a network file
    """
    #check if file exists
    if( not os.path.isfile(filepath)):
        return None
    #Ignore lines starting with #,* looks like first two values are nodes so far
    ext = os.path.splitext(filepath)[1]
    fobj = _dispatch_dict[ext](filepath, mode=mode)
    return fobj

def get_edge_tuples(fobj):
    edge_tuples = []
    linelist = fobj.readlines()
    for line in linelist:
        if(isinstance(line,bytes)):
            line = line.decode("utf-8")
        if any(line.startswith(x) for x in comment_predecessors):
            continue
        nodes = line.split()
        try:
            (node1,node2) = nodes[0],nodes[1]
        except:
            print('bad line is: {}'.format(line))
            print('line split elements are \'{}\''.format(nodes))
            exit(1)
        edge_tuples.append((node1,node2))
    return edge_tuples
