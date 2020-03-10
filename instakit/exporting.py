# -*- coding: utf-8 -*-
from __future__ import print_function

import os

from clu.exporting import ExporterBase

# The “basepath” is the directory enclosing the package root:
basepath = os.path.dirname(
           os.path.dirname(__file__))

class Exporter(ExporterBase, basepath=basepath, appname="instakit"):
    pass

exporter = Exporter(path=__file__)
export = exporter.decorator()

export(Exporter)

# Assign the modules’ `__all__` and `__dir__` using the exporter:
__all__, __dir__ = exporter.all_and_dir()
