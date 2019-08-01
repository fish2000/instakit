#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import argparse
import enum
import importlib
import inspect
import types

from pprint import pprint

class Parameter(object):
    """ A placeholder object, used for the moment in the inline tests """
    pass

QUALIFIER = '.'

def dotpath_join(base, *addenda):
    """ Join dotpath elements together as one, á la os.path.join(…) """
    for addendum in addenda:
        if not base.endswith(QUALIFIER):
            base += QUALIFIER
        if addendum.startswith(QUALIFIER):
            if len(addendum) == 1:
                raise ValueError('operand too short: %s' % addendum)
            addendum = addendum[1:]
        base += addendum
    # N.B. this might be overthinking it -- 
    # maybe we *want* to allow dotpaths
    # that happen to start and/or end with dots?
    if base.endswith(QUALIFIER):
        return base[:-1]
    return base

def qualified_import(qualified):
    """ Import a qualified thing-name.
        e.g. 'instakit.processors.halftone.FloydSteinberg'
    """
    if QUALIFIER not in qualified:
        raise ValueError("qualified_import() needs a qualified name "
                         "(got %s)" % qualified)
    head = qualified.split(QUALIFIER)[-1]
    tail = qualified.replace("%s%s" % (QUALIFIER, head), '')
    module = importlib.import_module(tail)
    cls = getattr(module, head)
    print("Qualified Import: %s" % qualified)
    return cls

def qualified_name_tuple(cls):
    """ Get the module name and the thing-name for a class.
        e.g. ('instakit.processors.halftone', 'FloydSteinberg')
    """
    mod_name = getattr(cls, '__module__')
    cls_name = getattr(cls, '__qualname__',
               getattr(cls, '__name__'))
    return mod_name, cls_name

def qualified_name(cls):
    """ Get a qualified thing-name for a class.
        e.g. 'instakit.processors.halftone.FloydSteinberg'
    """
    mod_name, cls_name = qualified_name_tuple(cls)
    out = "%s%s%s" % (mod_name, QUALIFIER, cls_name)
    print("Qualified Name: %s" % out)
    return out

class Nothing(object):
    """ Placeholder singleton, signifying nothing """
    __slots__ = tuple()
    def __new__(cls, *a, **k):
        return Nothing

def check_parameter_default(param_default):
    """ Filter result values coming from inspect.signature(…) """
    if param_default == inspect._empty:
        return Nothing
    return param_default

def default_arguments(cls):
    """ Get a dictionary of the keyword arguments with provided defaults,
        as furnished by a given classes’ “__init__” function.
    """
    try:
        signature = inspect.signature(cls)
    except (ValueError, TypeError) as exc:
        m, n = qualified_name_tuple(cls)
        qn = "%s%sSlow%s" % (m.replace('ext.', ''), QUALIFIER, n) # WTF HAX
        NonCythonCls = qualified_import(qn)
        if qualified_name(NonCythonCls) != qualified_name(cls):
            return default_arguments(NonCythonCls)
        else:
            raise exc
    if len(signature.parameters) < 1:
        return {}
    return { parameter.name : check_parameter_default(parameter.default) \
                                for parameter \
                                in signature.parameters.values() }

def is_enum(cls):
    """ Predicate function to ascertain whether a class is an Enum. """
    return enum.Enum in cls.__mro__

def enum_choices(cls):
    """ Return a list of the names of the given Enum class members. """
    return [choice.name for choice in cls]

FILE_ARGUMENT_NAMES = ('path', 'pth', 'file')

def add_argparser(subparsers, cls):
    """ Add a subparser -- an instance of “argparse.ArgumentParser” --
        with arguments and defaults matching the keyword arguments and
        defaults provided by the given class (q.v. “default_arguments(…)”
        definition supra.)
    """
    qualname = qualified_name(cls)
    cls_help = getattr(cls, '__doc__', None) or "help for %s" % qualname
    parser = subparsers.add_parser(qualname, help=cls_help)
    if is_enum(cls): # Deal with enums
        argument_name = cls.__name__.lower()
        add_argument_args = dict(choices=enum_choices(cls),
                                 type=str,
                                 help='help for enum %s' % argument_name)
        parser.add_argument(argument_name,
                          **add_argument_args)
    else: # Deal with __init__ signature
        for argument_name, argument_value in default_arguments(cls).items():
            argument_type = type(argument_value)
            argument_required = False
            add_argument_args = dict(help='help for argument %s' % argument_name)
            if argument_value is not Nothing:
                add_argument_args.update({ 'default' : argument_value })
            else:
                add_argument_args.update({ 'type' : argument_name in FILE_ARGUMENT_NAMES \
                                                and argparse.FileType('rb') \
                                                 or str })
                argument_required = True
            if argument_type is bool:
                add_argument_args.update({ 'action' : 'store_true' })
            elif argument_type is type(None):
                add_argument_args.update({ 'type' : str })
            elif is_enum(argument_type):
                add_argument_args.update({ 'choices' : enum_choices(argument_type),
                                              'type' : str })
            argument_template = argument_required and '%s' or '--%s'
            parser.add_argument(argument_template % argument_name,
                              **add_argument_args)
    return parser

functype = types.FunctionType

def get_processors_from(module_name):
    """ Memoized processor-extraction function """
    from instakit.utils.static import asset
    if not hasattr(get_processors_from, 'cache'):
        get_processors_from.cache = {}
    if module_name not in get_processors_from.cache:
        processors = []
        module = importlib.import_module(module_name)
        print("Module: %s (%s)" % (module.__name__,
                    asset.relative(module.__file__)))
        for thing in (getattr(module, name) for name in dir(module)):
            if hasattr(thing, 'process'):
                print("Found thing: %s" % thing)
                if module.__name__ in thing.__module__:
                    if thing not in processors:
                        if type(getattr(thing, 'process')) is functype:
                            processors.append(thing)
        get_processors_from.cache[module_name] = tuple(processors)
    return get_processors_from.cache[module_name] 


def test():
    
    # Test “qualified_import()”:
    print("Testing “qualified_import()”…")
    
    class_name = 'instakit.processors.halftone.SlowFloydSteinberg'
    ImportedFloydSteinberg = qualified_import(class_name)
    assert ImportedFloydSteinberg.__name__ == 'SlowFloydSteinberg'
    assert ImportedFloydSteinberg.__qualname__ == 'SlowFloydSteinberg'
    assert ImportedFloydSteinberg.__module__ == 'instakit.processors.halftone'
    
    class_name = 'instakit.processors.halftone.Atkinson' # TWIST!!
    ImportedAtkinson = qualified_import(class_name)
    assert ImportedAtkinson.__name__ == 'Atkinson'
    assert ImportedAtkinson.__qualname__ == 'Atkinson'
    assert ImportedAtkinson.__module__ == 'instakit.processors.ext.halftone'
    
    print("Success!")
    print()
    
    # Test “qualified_name()”:
    print("Testing “qualified_name()”…")
    
    class_name = qualified_name(Parameter)
    assert class_name == '__main__.Parameter'
    
    class_name = qualified_name(ImportedFloydSteinberg)
    assert class_name == 'instakit.processors.halftone.SlowFloydSteinberg'
    
    class_name = qualified_name(ImportedAtkinson)
    assert class_name == 'instakit.processors.ext.halftone.Atkinson'
    
    print("Success!")
    print()
    
    # Test “Nothing”:
    print("Testing “Nothing”…")
    
    assert type(Nothing) == type
    assert Nothing() == Nothing
    
    print("Success!")
    print()
    
    # Test “default_arguments()”:
    print("Testing “default_arguments()”…")
    
    default_args = default_arguments(ImportedFloydSteinberg)
    assert default_args == dict(threshold=128.0)
    
    slow_atkinson = 'instakit.processors.halftone.SlowAtkinson'
    default_args = default_arguments(qualified_import(slow_atkinson))
    assert default_args == dict(threshold=128.0)
    
    noise = 'instakit.processors.noise.GaussianNoise'
    default_args = default_arguments(qualified_import(noise))
    assert default_args == dict()
    
    contrast = 'instakit.processors.adjust.Contrast'
    default_args = default_arguments(qualified_import(contrast))
    assert default_args == dict(value=1.0)
    
    unsharp_mask = 'instakit.processors.blur.UnsharpMask'
    default_args = default_arguments(qualified_import(unsharp_mask))
    assert default_args == dict(radius=2,
                                percent=150,
                                threshold=3)
    
    curves = 'instakit.processors.curves'
    curveset = dotpath_join(curves, 'CurveSet')
    interpolate_mode = dotpath_join(curves, 'InterpolateMode')
    ImportedInterpolateMode = qualified_import(interpolate_mode)
    default_args = default_arguments(qualified_import(curveset))
    LAGRANGE = ImportedInterpolateMode.LAGRANGE
    assert default_args == dict(path=Nothing,
                                interpolation_mode=LAGRANGE)
    
    print("Success!")
    print()
    
    # Test “is_enum()”:
    print("Testing “is_enum()”…")
    
    assert is_enum(ImportedInterpolateMode)
    assert not is_enum(Parameter)
    assert not is_enum(ImportedFloydSteinberg)
    assert not is_enum(ImportedAtkinson)
    
    mode = 'instakit.utils.mode.Mode'
    assert is_enum(qualified_import(mode))
    
    interpolate_mode = 'instakit.processors.curves.InterpolateMode'
    assert is_enum(qualified_import(interpolate_mode))
    
    noise_mode = 'instakit.processors.noise.NoiseMode'
    assert is_enum(qualified_import(noise_mode))
    
    print("Success!")
    print()
    
    # Test “add_argparser()”:
    print("Testing “add_argparser()”…")
    
    parser = argparse.ArgumentParser(prog='instaprocess',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="print verbose messages to STDOUT")
    
    processor_names = ('adjust', 'blur', 'curves', 'halftone', 'noise', 'squarecrop')
    utility_names = ('colortype', 'gcr', 'kernels', 'lutmap',
                     'misc', 'mode', 'ndarrays', 'pipeline', 'static', 'stats')
    
    module_names = []
    module_names.extend(['instakit.processors.%s' % name for name in processor_names])
    module_names.extend(['instakit.utils.%s' % name for name in utility_names])
    
    processors = {}
    
    for module_name in module_names:
        processors[module_name] = get_processors_from(module_name)
    
    subparsers = parser.add_subparsers(help="subcommands for instakit processors")
    for processor_tuple in processors.values():
        for processor in processor_tuple:
            add_argparser(subparsers, processor)
    
    pprint(processors, indent=4)
    
    print()
    ns = parser.parse_args(['-h'])
    print(ns)
    
    print()
    ns = parser.parse_args(['instakit.utils.mode.Mode', '--help'])
    print(ns)
    
    print("Success!")
    print()


if __name__ == '__main__':
    test()