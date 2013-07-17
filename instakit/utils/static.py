import os
import os.path
from os.path import join, abspath
root = join(os.path.dirname(__file__), '..', 'data')
listfiles = lambda *pth: os.listdir(join(root, *pth))
path = lambda *pth: abspath(join(root, *pth))
