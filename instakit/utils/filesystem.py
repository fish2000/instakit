#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import abc
import collections
import collections.abc
import contextlib
import os
import re
import sys

try:
    from scandir import scandir, walk
except ImportError:
    from os import scandir, walk

from functools import wraps
from tempfile import _TemporaryFileWrapper as TemporaryFileWrapperBase

from instakit.utils.misc import memoize, stringify, suffix_searcher, u8bytes, u8str

__all__ = ('DEFAULT_PATH',
           'DEFAULT_PREFIX',
           'DEFAULT_ENCODING',
           'DEFAULT_TIMEOUT',
           'ExecutionError', 'FilesystemError',
           'script_path', 'which', 'back_tick',
           'rm_rf', 'temporary',
           'TemporaryName',
           'Directory',
           'cd', 'wd',
           'TemporaryDirectory', 'Intermediate',
           'NamedTemporaryFile')

__dir__ = lambda: list(__all__)

DEFAULT_PATH = ":".join(filter(os.path.exists, ("/usr/local/bin",
                                                "/bin",  "/usr/bin",
                                                "/sbin", "/usr/sbin")))

DEFAULT_ENCODING = 'latin-1'
DEFAULT_TIMEOUT = 60 # seconds
DEFAULT_PREFIX = "yo-dogg-"

class ExecutionError(Exception):
    """ An error during the execution of a shell command """
    pass

class FilesystemError(Exception):
    """ An error that occurred while mucking about with the filesystem """
    pass

def script_path():
    """ Return the path to the embedded scripts directory. """
    return os.path.join(
           os.path.dirname(__file__), 'scripts')

def which(binary_name, pathvar=None):
    """ Deduces the path corresponding to an executable name,
        as per the UNIX command `which`. Optionally takes an
        override for the $PATH environment variable.
        Always returns a string - an empty one for those
        executables that cannot be found.
    """
    from distutils.spawn import find_executable
    if not hasattr(which, 'pathvar'):
        which.pathvar = os.getenv("PATH", DEFAULT_PATH)
    return find_executable(binary_name, pathvar or which.pathvar) or ""

def back_tick(command,  as_str=True,
                       ret_err=False,
                     raise_err=None, **kwargs):
    """ Run command `command`, return stdout -- or (stdout, stderr) if `ret_err`.
        Roughly equivalent to ``check_output`` in Python 2.7.
        
        Parameters
        ----------
        command : str / list / tuple
            Command to execute. Can be passed as a single string (e.g "ls -la")
            or a tuple or list composed of the commands’ individual tokens (like
            ["ls", "-la"]).
        as_str : bool, optional
            Whether or not the values returned from ``proc.communicate()`` should
            be unicode-decoded as bytestrings (using the specified encoding, which
            defaults to Latin-1) before `back_tick(…)` returns. Default is True.
        ret_err : bool, optional
            If True, the return value is (stdout, stderr). If False, it is stdout.
            In either case `stdout` and `stderr` are strings containing output
            from the commands’ execution. Default is False.
        raise_err : None or bool, optional
            If True, raise instakit.utils.filesystem.errors.ExecutionError when
            calling the function results in a non-zero return code.
            If None, it is set to True if `ret_err` is False,
                                  False if `ret_err` is True.
            Default is None (exception-raising behavior depends on the `ret_err`
            value).
        encoding : str, optional
            The name of the encoding to use when decoding the command output per
            the `as_str` value. Default is “latin-1”.
        directory : str / Directory / path-like, optional
            The directory in which to execute the command. Default is None (in
            which case the process working directory, unchanged, will be used).
        verbose : bool, optional
            Whether or not debug information should be spewed to `sys.stderr`.
            Default is False.
        timeout : int, optional
            Number of seconds to wait for the executed command to complete before
            forcibly killing the subprocess. Default is 60.
        
        Returns
        -------
        out : str / tuple
            If `ret_err` is False, return stripped string containing stdout from
            `command`.  If `ret_err` is True, return tuple of (stdout, stderr)
            where ``stdout`` is the stripped stdout, and ``stderr`` is the
            stripped stderr.
        
        Raises
        ------
        A `instakit.utils.filesystem.errors.ExecutionError` will raise if the
        executed command returns with any non-zero exit status, and `raise_err`
        is set to True.
        
    """
    # Step 1: Prepare for battle:
    import subprocess, shlex
    verbose = bool(kwargs.pop('verbose',  False))
    timeout =  int(kwargs.pop('timeout',  DEFAULT_TIMEOUT))
    encoding = str(kwargs.pop('encoding', DEFAULT_ENCODING))
    raise_err = raise_err is not None and raise_err or bool(not ret_err)
    issequence = isinstance(command, (list, tuple))
    command_str = issequence and " ".join(command) or u8str(command).strip()
    directory = 'directory' in kwargs and os.fspath(kwargs.pop('directory')) or None
    # Step 2: DO IT DOUG:
    if not issequence:
        command = shlex.split(command)
    if verbose:
        print("EXECUTING:", file=sys.stdout)
        print("`{}`".format(command_str),
                            file=sys.stdout)
        print("",           file=sys.stdout)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                           cwd=directory,
                                         shell=False)
    try:
        output, errors = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output, errors = process.communicate(timeout=None)
    returncode = process.returncode
    # Step 3: Analyze the return code:
    if returncode is None:
        process.terminate()
        raise ExecutionError('`{}` terminated without exiting cleanly'.format(command_str))
    if raise_err and returncode != 0:
        raise ExecutionError('`{}` exited with status {}, error: “{}”'.format(command_str,
                                   returncode,
                                   u8str(errors).strip()))
    # Step 4: Tidy the output and return it:
    if verbose:
        if returncode != 0:
            print("",                           file=sys.stderr)
            print("NONZERO RETURN STATUS: {}".format(returncode),
                                                file=sys.stderr)
            print("",                           file=sys.stderr)
        if len(u8str(output.strip())) > 0:
            print("")
            print("OUTPUT:",                            file=sys.stdout)
            print("`{}`".format(u8str(output).strip()), file=sys.stdout)
            print("",                                   file=sys.stdout)
        if len(u8str(errors.strip())) > 0:
            print("",                                   file=sys.stderr)
            print("ERRORS:",                            file=sys.stderr)
            print("`{}`".format(u8str(errors).strip()), file=sys.stderr)
            print("",                                   file=sys.stderr)
    output = output.strip()
    if ret_err:
        errors = errors.strip()
        return (as_str and output.decode(encoding) or output), \
               (as_str and errors.decode(encoding) or errors)
    return (as_str and output.decode(encoding) or output)

def rm_rf(pth):
    """ rm_rf() does what `rm -rf` does – so, for the love of fuck,
        BE FUCKING CAREFUL WITH IT.
    """
    if not pth:
        raise ExecutionError(
            "Can’t rm -rf without something to rm arr-effedly")
    pth = os.fspath(pth)
    try:
        if os.path.isfile(pth) or os.path.islink(pth):
            os.unlink(pth)
        elif os.path.isdir(pth):
            subdirs = []
            for path, dirs, files in walk(pth, followlinks=True):
                for tf in files:
                    os.unlink(os.path.join(path, tf))
                subdirs.extend([os.path.join(path, td) for td in dirs])
            for subdir in reversed(subdirs):
                os.rmdir(subdir)
            os.rmdir(pth)
        return True
    except (OSError, IOError):
        pass
    return False

def temporary(suffix=None, prefix=None, parent=None, **kwargs):
    """ Wrapper around `tempfile.mktemp()` that allows full overriding of the
        prefix and suffix by the caller -- that is to say, no random elements
        are used in the returned filename if both a prefix and a suffix are
        supplied.
        
        To avoid problems, the function will throw a FilesystemError if it is
        called with arguments that result in the computation of a filename
        that already exists.
    """
    from tempfile import mktemp, gettempdir
    directory = os.fspath(kwargs.pop('dir', parent) or gettempdir())
    tempmade = mktemp(prefix=prefix, suffix=suffix, dir=directory)
    tempsplit = os.path.splitext(os.path.basename(tempmade))
    if not suffix:
        suffix = tempsplit[1][1:]
    if not prefix or kwargs.pop('randomized', False):
        prefix, _ = os.path.splitext(tempsplit[0]) # WTF, HAX!
    fullpth = os.path.join(directory, "%s%s" % (prefix, suffix))
    if os.path.exists(fullpth):
        raise FilesystemError("temporary(): file exists: %s" % fullpth)
    return fullpth


class TypeLocker(abc.ABCMeta):
    
    """ instakit.utils.filesystem.TypeLocker is a metaclass that does two
        things with the types for whom it is designated as meta:
        
        1) It keeps an index of those types in a dictionary member of
           the `TypeLocker` metaclass itself; and
        
        2) During class creation – the call to `TypeLocker.__new__(…)` –
           it installs a class method called “directory(…)” that will,
           when invoked, always return a new Directory instance that has
           been initialized with the one provided argument “pth” (if one
           was passed).
        
        … The point of this is to allow any of the classes throughout the
        instakit.utils.filesystem module regardless of where they are defined
        or from whom they inherit, to make use of cheaply-constructed Directory
        instances wherever convenient.
        
        Because the “directory(…)” method installed by TypeLocker performs
        a lazy-lookup of the Directory class, using its own type index dict,
        the order of definition does not matter i.e. the TemporaryName class
        (q.v. definition immediately sub.) can use Directories despite its
        definition occuring before Directory – in fact TemporaryName itself
        is utilized within at least one Directory method – sans any issues.
    """
    
    # The metaclass-internal dictionary of generated classes:
    types = collections.OrderedDict()
    
    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        """ Maintain declaration order in class members: """
        return collections.OrderedDict()
    
    def __new__(metacls, name, bases, attributes, **kwargs):
        """ All classes are initialized with a “directory(…)”
            class method, lazily returning an instance of the
            instakit.utils.filesystem.Directory(…) class, per
            the arguments:
        """
        # Always replace the “directory” method anew:
        attributes['directory'] = staticmethod(
                                      lambda pth=None: \
                                      metacls.types['Directory'](pth=pth))
        cls = super(TypeLocker, metacls).__new__(metacls, name,
                                                          bases,
                                                          dict(attributes),
                                                        **kwargs)
        metacls.types[name] = cls
        os.PathLike.register(cls)
        return cls

class TemporaryFileWrapper(TemporaryFileWrapperBase,
                           contextlib.AbstractContextManager,
                           os.PathLike,
                           metaclass=TypeLocker):
    
    """ Local subclass of `tempfile._TemporaryFileWrapper`.
        
        We also inherit from both `contextlib.AbstractContextManager`
        and the `os.PathLike` abstract bases -- the latter requires
        that we implement an __fspath__(…) method (q.v. implementation,
        sub.) -- and additionally, `filesystem.TypeLocker` is named as
        the metaclass (q.v. metaclass __new__(…) implementation supra.)
        to cache its type and register it as an os.PathLike subclass.
        
        … Basically a better deal than the original ancestor, like
        all-around. Plus it does not have a name prefixed with an
        underscore, which if it’s not your implementation dogg that
        can be a bit lexically irritating.
    """
    
    def __fspath__(self):
        return self.name

@memoize
def TemporaryNamedFile(tempth, mode='wb', buffer_size=-1, delete=True):
    """ Variation on ``tempfile.NamedTemporaryFile(…)``, for use within
        `filesystem.TemporaryName()` – q.v. class definition sub.
        
        Parameters
        ----------
        tempth : str / bytes / descriptor / filename-ish
            File name, path, or descriptor to open.
        mode : str / bytes, optional
            String-like symbolic explication of mode with which to open
            the file -- q.v. ``io.open(…)`` or ``__builtins__.open(…)``
            supra.
        buffer_size : int, optional
            Integer indicating buffer size to use during file reading
            and/or writing. Default value is -1 (which indicates that
            reads and writes should be unbuffered).
        delete : bool, optional
            Boolean value indicating whether to delete the wrapped
            file upon scope exit or interpreter shutdown (whichever
            happens first). Default is True.
        
        Returns
        -------
            A ``instakit.utils.filesystem.TemporaryFileWrapper`` object,
            initialized and ready to be used, as per its counterpart(s),
            ``tempfile.NamedTemporaryFile``, and
            `filesystem.NamedTemporaryFile`.
        
        Raises
        ------
            A `instakit.utils.filesystem.FilesystemError`, corresponding to
            any errors that may be raised during its own internal calls to
            ``os.open(…)`` and ``os.fdopen(…)``
        
    """
    from tempfile import _bin_openflags, _text_openflags
    
    if 'b' in mode:
        flags = _bin_openflags
    else:
        flags = _text_openflags
    if os.name == 'nt' and delete:
        flags |= os.O_TEMPORARY
    
    descriptor = 0
    filehandle = None
    path = None
    
    try:
        path = os.fspath(tempth)
        descriptor = os.open(path, flags)
        filehandle = os.fdopen(descriptor, mode, buffer_size)
        return TemporaryFileWrapper(filehandle, path, delete)
    except BaseException as base_exception:
        try:
            rm_rf(path)
        except ExecutionError:
            pass
        if descriptor > 0:
            os.close(descriptor)
        raise FilesystemError(str(base_exception))

class TemporaryName(collections.abc.Hashable,
                    contextlib.AbstractContextManager,
                    os.PathLike,
                    metaclass=TypeLocker):
    
    """ This is like NamedTemporaryFile without any of the actual stuff;
        it just makes a file name -- YOU have to make shit happen with it.
        But: should you cause such scatalogical events to transpire, this
        class (when invoked as a context manager) will clean it up for you.
        Unless you say not to. Really it's your call dogg I could give AF
    """
    
    fields = ('name', 'exists',
              'destroy', 'prefix', 'suffix', 'parent')
    
    def __init__(self, prefix=None, suffix="tmp",
                       parent=None,
                     **kwargs):
        """ Initialize a new TemporaryName object.
            
            All parameters are optional; you may specify “prefix”, “suffix”,
            and “dir” (alternatively as “parent” which I think reads better)
            as per `tempfile.mktemp(…)`. Suffixes may omit the leading period
            without confusing things. 
        """
        randomized = kwargs.pop('randomized', False)
        if not prefix:
            prefix = DEFAULT_PREFIX
            randomized = True
        if suffix:
            if not suffix.startswith(os.extsep):
                suffix = "%s%s" % (os.extsep, suffix)
        else:
            suffix = "%stmp" % os.extsep
        if parent is None:
            parent = kwargs.pop('dir', None)
        if parent:
            parent = os.fspath(parent)
        self._name = temporary(prefix=prefix, suffix=suffix,
                                              parent=parent,
                                              randomized=randomized)
        self._destroy = True
        self._parent = parent
        self.prefix = prefix
        self.suffix = suffix
    
    @property
    def name(self):
        """ The temporary file path (which initially does not exist). """
        return self._name
    
    @property
    def basename(self):
        """ The basename (aka the filename) of the temporary file path. """
        return os.path.basename(self._name)
    
    @property
    def dirname(self):
        """ The dirname (aka the enclosing directory) of the temporary file. """
        return self.parent()
    
    @property
    def exists(self):
        """ Whether or not there is anything existant at the temporary file path.
            
            Note that this property will be true for directories created therein,
            as well as FIFOs or /dev entries, or any of the other zany filesystem
            possibilities you and the POSIX standard can imagine, in addition to
            regular files.
        """
        return os.path.exists(self._name)
    
    @property
    def destroy(self):
        """ Whether or not this TemporaryName instance should destroy any file
            that should happen to exist at its temporary file path (as per its
            “name” attribute) on scope exit.
        """
        return self._destroy
    
    @property
    def filehandle(self):
        """ Access a TemporaryNamedFile instance, opened and ready to read and write,
            for this TemporaryName instances’ temporary file path.
            
            Accessing this property delegates the responsibility for destroying
            the TemporaryName file contents to the TemporaryNamedFile object --
            saving the TemporaryNamedFile in, like, a variable somewhere and then
            letting the original TemporaryName go out of scope will keep the file
            alive and unclosed, for example.
        """
        return TemporaryNamedFile(self.do_not_destroy())
    
    def split(self):
        """ Return (dirname, basename) e.g. for /yo/dogg/i/heard/youlike,
            you get back (Directory("/yo/dogg/i/heard"), "youlike")
        """
        return self.dirname, self.basename
    
    def copy(self, destination):
        """ Copy the file (if one exists) at the instances’ file path
            to a new destination.
        """
        if not destination:
            raise FilesystemError("Copying requires a place to which to copy")
        import shutil
        if self.exists:
            return shutil.copy2(self._name, os.fspath(destination))
        return False
    
    def do_not_destroy(self):
        """ Mark this TemporaryName instance as one that should not be automatically
            destroyed upon the scope exit for the instance.
            
            This function returns the temporary file path, and may be called more
            than once without further side effects.
        """
        self._destroy = False
        return self._name
    
    def parent(self):
        """ Sugar for `os.path.abspath(os.path.join(self.name, os.pardir))`
            which, if you are curious, gets you the parent directory of
            the instances’ target filename, wrapped in a Directory
            instance.
        """
        return self.directory(os.path.abspath(
                              os.path.join(self.name,
                                           os.pardir)))
    
    def close(self):
        """ Destroys any existing file at this instances’ file path. """
        if self.exists:
            return rm_rf(self._name)
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self.destroy:
            self.close()
        return exc_type is None
    
    def to_string(self):
        """ Stringify the TemporaryName instance. """
        return stringify(self, type(self).fields)
    
    def __repr__(self):
        return stringify(self, type(self).fields)
    
    def __str__(self):
        if self.exists:
            return os.path.realpath(self._name)
        return self._name
    
    def __bytes__(self):
        return u8bytes(str(self))
    
    def __fspath__(self):
        return self._name
    
    def __bool__(self):
        return self.exists
    
    def __eq__(self, other):
        try:
            return os.path.samefile(self._name,
                                    os.fspath(other))
        except FileNotFoundError:
            return False
    
    def __ne__(self, other):
        try:
            return not os.path.samefile(self._name,
                                        os.fspath(other))
        except FileNotFoundError:
            return True
    
    def __hash__(self):
        return hash((self._name, self.exists))

non_dotfile_match = re.compile(r"^[^\.]").match
non_dotfile_matcher = lambda p: non_dotfile_match(p.name) # type: ignore

class Directory(collections.abc.Hashable,
                collections.abc.Mapping,
                collections.abc.Sized,
                contextlib.AbstractContextManager,
                os.PathLike,
                metaclass=TypeLocker):
    
    """ A context-managed directory: change in on enter, change back out
        on exit. Plus a few convenience functions for listing and whatnot.
    """
    
    fields = ('name', 'old', 'new', 'exists',
              'will_change',        'did_change',
              'will_change_back',   'did_change_back')
    
    zip_suffix = "%szip" % os.extsep
    
    def __init__(self, pth=None):
        """ Initialize a new Directory object.
            
            There is only one parameter, “pth” -- the target path for the Directory
            object instance. When the Directory is initialized as a context manager,
            the process working directory will change to this directory (provided
            that it’s a different path than the current working directory, according
            to `os.path.samefile(…)`).
            
            The “pth” parameter is optional, in which case the instance uses the
            process working directory as its target, and no change-of-directory
            calls will be issued. Values for “pth” can be string-like, or existing
            Directory instances -- either will work.
            
            There are two decendant classes of Directory (q.v. definitions below)
            that enforce stipulations for the “pth” parameter: the `cd` class 
            requires a target path to be provided (and therefore will nearly always
            change the working directory when invoked as a context manager). Its
            sibling class `wd` forbids the naming of a “pth” value, thereby always
            initializing itself with the current working directory as its target,
            and fundamentally avoids issuing any directory-change calls.
        """
        if pth is not None:
            self.target = u8str(os.fspath(pth))
        else:
            self.target = os.getcwd()
    
    @property
    def name(self):
        """ The instances’ target directory path. """
        return getattr(self, 'target', None) or \
               getattr(self, 'new')
    
    @property
    def basename(self):
        """ The basename (aka the name of the directory, like as opposed to the
            entire fucking absolute path) of the target directory.
        """
        return os.path.basename(self.name)
    
    @property
    def dirname(self):
        """ The dirname (aka the path of the enclosing directory) of the target
            directory, wrapped in a new Directory instance.
        """
        return self.parent()
    
    @property
    def exists(self):
        """ Whether or not the instances’ target path exists as a directory. """
        return os.path.isdir(self.name)
    
    @property
    def initialized(self):
        """ Whether or not the instance has been “initialized” -- as in, the
            `target` instance value has been set (q.v. `ctx_initialize(…)`
            help sub.) as it stands immediately after `__init__(…)` has run.
        """
        return hasattr(self, 'target')
    
    @property
    def targets_set(self):
        """ Whether or not the instance has had targets set (the `new` and `old`
            instance values, q.v. `ctx_set_targets(…)` help sub.) and is ready
            for context-managed use.
        """
        return hasattr(self, 'old') and hasattr(self, 'new')
    
    @property
    def prepared(self):
        """ Whether or not the instance has been internally prepared for use
            (q.v. `ctx_prepare()` help sub.) and is in a valid state.
        """
        return hasattr(self, 'will_change') and \
               hasattr(self, 'will_change_back') and \
               hasattr(self, 'did_change') and \
               hasattr(self, 'did_change_back')
    
    def split(self):
        """ Return a two-tuple containing `(dirname, basename)` – like e.g.
            for `/yo/dogg/i/heard/youlike`, your return value will be like
            `(Directory("/yo/dogg/i/heard"), "youlike")`
        """
        return self.dirname, self.basename
    
    def ctx_initialize(self):
        """ Restores the instance to the freshly-allocated state -- with one
            notable exception: if it had been previously prepared (through a
            call to `instance.ctx_prepare()`) and thus has a “new” attribute
            filled in with a target path, `ctx_initialize()` will preserve
            the contents of that attribute in the value of the `self.target` 
            instance member.
            
            The call deletes all other instance attributes from the internal
            mapping of the instance in question, leaving it in a state ready
            for either context-managed reëntry, or for reuse in an unmanaged
            fashion *provided* one firstly calls `instance.ctx_set_targets()`
            or `instance.ctx_prepare()` in order to reconfigure (the minimal
            subset of, or the full complement of) the member-variable values
            needed by the internal workings of a Directory instance.
        """
        if self.targets_set:
            self.target = u8str(self.new or self.old)
            del self.old
            del self.new
        if self.prepared:
            del self.will_change
            del self.will_change_back
            del self.did_change
            del self.did_change_back
        return self
    
    def ctx_set_targets(self, old=None):
        """ Sets the “self.old” and “self.new” instance variable values,
            using the value of `self.target` and an (optional) string-like
            argument to use as the value for “self.old”.
            
            One shouldn’t generally call this or have a need to call this --
            although one can manually invoke `instance.ctx_set_targets(…)`
            to reconfigure a Directory instance to use it again after it has
            been re-initialized after a call to `instance.ctx_initialize()`
            (q.v. `ctx_initialize()` help supra.) in cases where it isn’t
            going to be used as part of a managed context; that is to say,
            outside of a `with` statement.
            
            (Within a `with` statement, the call issued upon scope entry to
            `Directory.__enter__(self)` will internally make a call to
            `Directory.ctx_prepare(self)` (q.v. doctext help sub.) which
            that will call `Directory.ctx_set_targets(self, …)` itself.)
        """
        if not self.initialized:
            if old is None:
                old = os.getcwd()
            setattr(self, 'old', old)
            setattr(self, 'new', old)
            return self
        setattr(self, 'old', old is not None and old or self.target)
        setattr(self, 'new', self.target)
        del self.target
        return self
    
    def ctx_prepare(self):
        """ Prepares the member values of the Directory instance according
            to a requisite `self.target` directory-path value; the primary
            logic performed by this function determines whether or not it
            is necessary to switch the process working directory while the
            Directory instance is actively being used as a context manager
            in the scope of a `while` block.
            
            The reason this is done herein is to minimize the number of
            calls to potentially expensive system-call-wrapping functions
            such as `os.getcwd()`, `os.path.samefile(…)`, and especially
            `os.chdir(…)` -- which the use of the latter affects the state
            of the process issuing the call in a global fashion, and can
            cause invincibly undebuggable behavioral oddities to crop up
            in a variety of circumstances. 
        """
        self.ctx_set_targets(old=os.getcwd())
        if os.path.isdir(self.new):
            self.will_change = not os.path.samefile(self.old,
                                                    self.new)
        else:
            self.will_change = False
        self.did_change = False
        self.will_change_back = self.will_change
        self.did_change_back = False
        return self
    
    def __enter__(self):
        self.ctx_prepare()
        if self.will_change and self.exists:
            os.chdir(self.new)
            self.did_change = os.path.samefile(self.new,
                                               os.getcwd())
            self.will_change_back = self.did_change
        return self
    
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        # N.B. return False to throw, True to supress:
        if self.will_change_back and os.path.isdir(self.old):
            os.chdir(self.old)
            self.did_change_back = os.path.samefile(self.old,
                                                    os.getcwd())
            if self.did_change_back:
                # return to pristine state:
                self.ctx_initialize()
                return exc_type is None
        # return to pristine state:
        self.ctx_initialize()
        return False
    
    def realpath(self, pth=None):
        """ Sugar for calling os.path.realpath(self.name) """
        return u8str(
            os.path.realpath(
            os.fspath(pth or self.name)))
    
    def ls(self, pth=None, suffix=None):
        """ List files -- defaults to the process’ current working directory.
            As per the UNIX custom, files whose name begins with a dot are
            omitted.
            
            Specify an optional “suffix” parameter to filter the list by a
            particular file suffix (leading dots unnecessary but unharmful).
        """
        files = (str(direntry.name) \
                 for direntry in filter(non_dotfile_matcher,
                                scandir(self.realpath(pth))))
        if not suffix:
            return files
        return filter(suffix_searcher(suffix), files)
    
    def ls_la(self, pth=None, suffix=None):
        """ List all files, including files whose name starts with a dot.
            The default is to use the process’ current working directory.
            
            Specify an optional “suffix” parameter to filter the list by a
            particular file suffix (leading dots unnecessary but unharmful).
            
            (Technically speaking, `ls_la()` is a misnomer for this method,
            as it does not provide any extended meta-info like you get if
            you use the “-l” flag when invoking the `ls` command -- I just
            like calling it that because “ls -la” was one of the first shell
            commands I ever learned, and it reads better than `ls_a()` which
            I think looks awkward and goofy.)
        """
        files = (str(direntry.name) \
                 for direntry in scandir(self.realpath(pth)))
        if not suffix:
            return files
        return filter(suffix_searcher(suffix), files)
    
    def subpath(self, subpth, whence=None, requisite=False):
        """ Returns the path to a subpath of the instances’ target path. """
        fullpth = os.path.join(os.fspath(whence or self.name),
                               os.fspath(subpth))
        return (os.path.exists(fullpth) or not requisite) and fullpth or None
    
    def subdirectory(self, subdir, whence=None):
        """ Returns the path to a subpath of the instances’ target path --
            much like Directory.subpath(…) -- as an instance of Directory.
        """
        pth = self.subpath(subdir, whence, requisite=False)
        if os.path.isfile(pth):
            raise FilesystemError("file exists at subdirectory path: %s" % pth)
        if os.path.islink(pth):
            raise FilesystemError("symlink exists at subdirectory path: %s" % pth)
        if os.path.ismount(pth):
            raise FilesystemError("mountpoint exists at subdirectory path: %s" % pth)
        return self.directory(pth)
    
    def makedirs(self, pth=None):
        """ Creates any parts of the target directory path that don’t
            already exist, á la the `mkdir -p` shell command.
        """
        try:
            os.makedirs(os.path.abspath(
                        os.path.join(self.name,
                        os.fspath(pth or os.curdir))),
                        exist_ok=False)
        except OSError as os_error:
            raise FilesystemError(str(os_error))
        return self
    
    def walk(self, followlinks=False):
        """ Sugar for calling X.walk(self.name), where X is either
            `scandir` (in the case of python 2.7) or `os` (for
            python 3 and thereafter).
        """
        return walk(self.name, followlinks=followlinks)
    
    def parent(self):
        """ Sugar for `os.path.abspath(os.path.join(self.name, os.pardir))`
            which, if you are curious, gets you the parent directory of
            the instances’ target directory, wrapped in a Directory
            instance.
        """
        return self.directory(os.path.abspath(
                              os.path.join(self.name,
                                           os.pardir)))
    
    def copy_all(self, destination):
        """ Copy the entire temporary directory tree, all contents
            included, to a new destination path. The destination must not
            already exist, and `copy_all(…)` will not overwrite existant
            directories. Like, if you have yourself an instance of Directory,
            `directory`, and you want to copy it to `/home/me/myshit`,
            `/home/me` should already exist but `/home/me/myshit` should not,
            as the `myshit` subdirectory will be created when you invoke the
            `directory.copy_all('/home/me/myshit')` call.
            
            Does that make sense to you? Try it, you’ll get a FilesystemError
            if it evidently did not make sense to you.
            
            The destination path may be specified using a string-like, or
            with a Directory object. Internally, this method uses
            `shutil.copytree(…)` to tell the filesystem what to copy where.
        """
        import shutil
        whereto = self.directory(pth=destination)
        if whereto.exists or os.path.isfile(whereto.name) \
                          or os.path.islink(whereto.name):
            raise FilesystemError(
                "copy_all() destination exists: %s" % whereto.name)
        if self.exists:
            return shutil.copytree(self.name, whereto.name)
        return False
    
    def zip_archive(self, zpth=None, zmode=None):
        """ Recursively descends through the target directory, stowing all
            that it finds into a zipfile at the specified path.
            
            Use the optional “zmode” parameter to specify the compression
            algorithm, as per the constants found in the `zipfile` module;
            the default value is `zipfile.ZIP_DEFLATED`.
        """
        import zipfile
        if zpth is None:
            raise FilesystemError("Need to specify a zip-archive file path")
        zpth = os.fspath(zpth)
        if not zpth.lower().endswith(self.zip_suffix):
            zpth += self.zip_suffix
        if os.path.exists(zpth):
            if os.path.isdir(zpth):
                raise FilesystemError("Can't overwrite a directory: %s" % zpth)
            raise FilesystemError("File path for zip-archive already exists")
        if not zmode:
            zmode = zipfile.ZIP_DEFLATED
        with TemporaryName(prefix="ziparchive-",
                           suffix=self.zip_suffix[1:]) as ztmp:
            with zipfile.ZipFile(ztmp.name, "w", zmode) as ziphandle:
                relparent = lambda p: os.path.relpath(p, os.fspath(self.parent()))
                for root, dirs, files in self.walk(followlinks=True):
                    ziphandle.write(root, relparent(root)) # add directory
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        if os.path.isfile(filepath): # regular files only
                            arcname = os.path.join(relparent(root), filename)
                            ziphandle.write(filepath, arcname) # add regular file
            ztmp.copy(zpth)
        return self.realpath(zpth)
    
    def close(self):
        """ Stub method -- always returns True: """
        return True
    
    def to_string(self):
        """ Stringify the Directory instance. """
        return stringify(self, type(self).fields)
    
    def __repr__(self):
        return stringify(self, type(self).fields)
    
    def __str__(self):
        if self.exists:
            return self.realpath()
        return self.name
    
    def __bytes__(self):
        return u8bytes(str(self))
    
    def __fspath__(self):
        return self.name
    
    def __bool__(self):
        return self.exists
    
    def __iter__(self):
        return scandir(self.realpath())
    
    def __len__(self):
        return len(list(self))
    
    def __getitem__(self, filename):
        pth = self.subpath(filename, requisite=True)
        if not pth:
            raise KeyError(
                "file not found: %s" % os.fspath(filename))
        return pth
    
    def __contains__(self, filename):
        return self.subpath(filename, requisite=True) is not None
    
    def __eq__(self, other):
        try:
            return os.path.samefile(self.name,
                                    os.fspath(other))
        except FileNotFoundError:
            return False
    
    def __ne__(self, other):
        try:
            return not os.path.samefile(self.name,
                                        os.fspath(other))
        except FileNotFoundError:
            return True
    
    def __hash__(self):
        return hash((self.name, self.exists))
    
    def __call__(self, *args, **kwargs):
        return self

class cd(Directory):
    
    def __init__(self, pth):
        """ Change to a new directory (the target path `pth` must be specified).
        """
        super(cd, self).__init__(pth)


class wd(Directory):
    
    def __init__(self):
        """ Initialize a Directory instance for the current working directory.
        """
        super(wd, self).__init__(pth=None)


class TemporaryDirectory(Directory):
    
    """ It's funny how this code looks, like, 99 percent exactly like the above
        TemporaryName class -- shit just works out that way. But this actually
        creates the directory in question; much like filesystem::TemporaryDirectory
        from libimread, this class wraps tempfile.mkdtemp() and can be used as
        a context manager (the C++ orig used RAII).
    """
    
    fields = ('name', 'old', 'new', 'exists',
              'destroy', 'prefix',  'suffix', 'parent',
              'will_change',        'did_change',
              'will_change_back',   'did_change_back')
    
    def __init__(self, prefix="TemporaryDirectory-", suffix="",
                                                     parent=None,
                                                     change=True,
                                                   **kwargs):
        """ Initialize a new TemporaryDirectory object.
            
            All parameters are optional; you may specify “prefix”, “suffix”,
            and “dir” (alternatively as “parent” which I think reads better)
            as per `tempfile.mkdtemp(…)`. Suffixes may omit the leading period
            without confusing things. 
            
            The boolean “change” parameter determines whether or not the
            process working directory will be changed to the newly created
            temporary directory; it defaults to `True`.
        """
        from tempfile import mkdtemp
        if suffix:
            if not suffix.startswith(os.extsep):
                suffix = "%s%s" % (os.extsep, suffix)
        if parent is None:
            parent = kwargs.pop('dir', None)
        if parent:
            parent = os.fspath(parent)
        self._name = mkdtemp(prefix=prefix, suffix=suffix, dir=parent)
        self._destroy = True
        self._parent = parent
        self.prefix = prefix
        self.suffix = suffix
        self.change = change
        super(TemporaryDirectory, self).__init__(self._name)
    
    @property
    def name(self):
        """ The temporary directory pathname. """
        return self._name
    
    @property
    def exists(self):
        """ Whether or not the temporary directory exists. """
        return os.path.isdir(self._name)
    
    @property
    def destroy(self):
        """ Whether or not the TemporaryDirectory instance has been marked
            for automatic deletion upon scope exit (q.v __exit__(…) method
            definition sub.)
        """
        return self._destroy
    
    @wraps(Directory.ctx_prepare)
    def ctx_prepare(self):
        change = super(TemporaryDirectory, self).ctx_prepare().change
        self.will_change = self.will_change_back = bool(self.will_change and change)
        return self
    
    def close(self):
        """ Delete the directory pointed to by the TemporaryDirectory
            instance, and everything it contains. USE WITH CAUTION.
        """
        out = super(TemporaryDirectory, self).close()
        if self.exists:
            return rm_rf(self.name) and out
        return False
    
    def do_not_destroy(self):
        """ Mark this TemporaryDirectory instance as one that should not
            be automatically destroyed upon the scope exit for the instance.
            
            This function returns the temporary directory path, and may
            be called more than once without further side effects.
        """
        self._destroy = False
        return self.name
    
    def __enter__(self):
        if not self.exists:
            self.makedirs()
        super(TemporaryDirectory, self).__enter__()
        return self
    
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        out = super(TemporaryDirectory, self).__exit__(exc_type, exc_val, exc_tb)
        if self.destroy:
            out &= self.close()
        return out


class Intermediate(TemporaryDirectory, Directory):
    
    """ instakit.utils.filesystem.Intermediate isn’t a class, per se – rather,
        it is a class factory proxy that normally constructs a new Directory
        instance in leu of itself, except for when it it is constructed without
        a `pth` argument, in which case, it falls back to the construction of
        a new TemporaryDirectory instance instead.
    """
    
    def __new__(cls, pth=None):
        """ The constructor simply delegates to the creation of either a new
            Directory or a new TemporaryDirectory.
        """
        if pth is not None:
            return Directory(pth=pth)
        return TemporaryDirectory(prefix="%s-" % cls.__name__,
                                  change=False)
    
    def __init__(self, pth=None):
        """ The initializer explicitly does nothing, as it will always be called
            on an already-initialized instance of some other class.
        """
        pass

def NamedTemporaryFile(mode='w+b', buffer_size=-1,
                       suffix="tmp", prefix=DEFAULT_PREFIX,
                       directory=None,
                       delete=True):
    """ Variation on tempfile.NamedTemporaryFile(…), such that suffixes
        are passed WITHOUT specifying the period in front (versus the
        standard library version which makes you pass suffixes WITH
        the fucking period, ugh).
    """
    from tempfile import (gettempdir, _bin_openflags,
                                      _text_openflags,
                                      _mkstemp_inner)
    
    parent = Directory(pth=directory or gettempdir())
    
    if suffix:
        if not suffix.startswith(os.extsep):
            suffix = "%s%s" % (os.extsep, suffix)
    else:
        suffix = "%stmp" % os.extsep
    
    if 'b' in mode:
        flags = _bin_openflags
    else:
        flags = _text_openflags
    if os.name == 'nt' and delete:
        flags |= os.O_TEMPORARY
    
    (descriptor, name) = _mkstemp_inner(parent.name, prefix,
                                                     suffix, flags,
                                             u8bytes(suffix))
    try:
        filehandle = os.fdopen(descriptor, mode, buffer_size)
        return TemporaryFileWrapper(filehandle, name, delete)
    except BaseException as base_exception:
        rm_rf(name)
        if descriptor > 0:
            os.close(descriptor)
        raise FilesystemError(str(base_exception))

del TemporaryFileWrapperBase

def test():
    
    """ Run the inline tests for the instakit.utils.filesystem module. """
    
    # Simple inline tests for “TemporaryName”, “cd” and “cwd”,
    # and “TemporaryDirectory”:
    from tempfile import gettempdir
    
    initial = os.getcwd()
    tfp = None
    tdp = None
    
    with TemporaryName(prefix="test-temporaryname-",
                       randomized=True) as tfn:
        print("* Testing TemporaryName file instance: %s" % tfn.name)
        assert os.path.samefile(os.getcwd(),            initial)
        assert gettempdir() in tfn.name
        assert tfn.prefix == "test-temporaryname-"
        assert tfn.suffix == ".tmp"
        assert not tfn._parent
        assert tfn.prefix in tfn.name
        assert tfn.suffix in tfn.name
        assert tfn.prefix in os.fspath(tfn)
        assert tfn.suffix in os.fspath(tfn)
        assert tfn.destroy
        assert type(tfn.directory(os.path.basename(tfn.name))) == Directory
        assert isinstance(tfn,                          TemporaryName)
        assert isinstance(tfn,                          collections.abc.Hashable)
        assert isinstance(tfn,                          contextlib.AbstractContextManager)
        assert isinstance(tfn,                          os.PathLike)
        p = tfn.parent()
        assert os.path.samefile(os.fspath(p),           gettempdir())
        # The next four asserts will be “not” asserts while
        # the TemporaryName has not been written to:
        assert not tfn.exists
        assert not os.path.isfile(os.fspath(tfn))
        assert not tfn.basename in tfn.dirname
        assert not tfn.basename in p
        # Here we write something to the TemporaryName:
        with open(os.fspath(tfn), mode="w") as handle:
            handle.write("yo dogg")
        # Now we repeat the above four asserts,
        # as positive assertions:
        assert tfn.exists
        assert os.path.isfile(os.fspath(tfn))
        assert tfn.basename in tfn.dirname
        assert tfn.basename in p
        # Stash the TemporaryName’s path to later assert
        # that it no longer exists - that it has been correctly
        # deleted on scope exit:
        tfp = tfn.name
        assert os.path.exists(tfp)
        print("* TemporaryName file object tests completed OK")
        print("")
    
    # Confirm that the TemporaryName has been deleted:
    assert not os.path.exists(tfp)
    
    with wd() as cwd:
        print("* Testing working-directory instance: %s" % cwd.name)
        assert os.path.samefile(os.getcwd(),           cwd.new)
        assert os.path.samefile(os.getcwd(),           cwd.old)
        assert os.path.samefile(os.getcwd(),           os.fspath(cwd))
        assert os.path.samefile(cwd.new,               cwd.old)
        assert os.path.samefile(cwd.new,               initial)
        assert os.path.samefile(cwd.old,               initial)
        assert os.path.samefile(cwd.new,               os.fspath(cwd))
        assert os.path.samefile(cwd.old,               os.fspath(cwd))
        assert os.path.samefile(os.fspath(cwd),        initial)
        assert not cwd.subdirectory('yodogg').exists
        # assert cwd.subdirectory('yodogg').makedirs().exists
        assert not cwd.will_change
        assert not cwd.did_change
        assert not cwd.will_change_back
        assert not cwd.did_change_back
        assert type(cwd.directory(cwd.new)) == Directory
        assert isinstance(cwd,                         wd)
        assert isinstance(cwd,                         Directory)
        assert isinstance(cwd,                         collections.abc.Hashable)
        assert isinstance(cwd,                         collections.abc.Mapping)
        assert isinstance(cwd,                         collections.abc.Sized)
        assert isinstance(cwd,                         contextlib.AbstractContextManager)
        assert isinstance(cwd,                         os.PathLike)
        assert os.path.isdir(os.fspath(cwd))
        assert not 'yodogg' in cwd
        assert cwd.basename in cwd.dirname
        # print(", ".join(list(cwd.ls())))
        print("* Working-directory object tests completed OK")
        print("")
    
    with cd(gettempdir()) as tmp:
        print("* Testing directory-change instance: %s" % tmp.name)
        assert os.path.samefile(os.getcwd(),          gettempdir())
        assert os.path.samefile(os.getcwd(),          tmp.new)
        assert os.path.samefile(gettempdir(),         tmp.new)
        assert os.path.samefile(os.getcwd(),          os.fspath(tmp))
        assert os.path.samefile(gettempdir(),         os.fspath(tmp))
        assert not os.path.samefile(os.getcwd(),      initial)
        assert not os.path.samefile(tmp.new,          initial)
        assert not os.path.samefile(os.fspath(tmp),   initial)
        assert os.path.samefile(tmp.old,              initial)
        # assert not tmp.subdirectory('yodogg').exists
        # assert tmp.subdirectory('yodogg').makedirs().exists
        assert tmp.will_change
        assert tmp.did_change
        assert tmp.will_change_back
        assert not tmp.did_change_back
        assert type(tmp.directory(tmp.new)) == Directory
        assert isinstance(tmp,                        cd)
        assert isinstance(tmp,                        Directory)
        assert isinstance(tmp,                        collections.abc.Hashable)
        assert isinstance(tmp,                        collections.abc.Mapping)
        assert isinstance(tmp,                        collections.abc.Sized)
        assert isinstance(tmp,                        contextlib.AbstractContextManager)
        assert isinstance(tmp,                        os.PathLike)
        assert os.path.isdir(os.fspath(tmp))
        assert tmp.basename in tmp.dirname
        print("* Directory-change object tests completed OK")
        print("")
    
    with TemporaryDirectory(prefix="test-temporarydirectory-") as ttd:
        print("* Testing TemporaryDirectory instance: %s" % ttd.name)
        # assert os.path.commonpath((os.getcwd(), gettempdir())) == gettempdir()
        # print(os.path.commonpath((os.getcwd(), gettempdir())))
        assert gettempdir() in ttd.name
        assert gettempdir() in ttd.new
        assert gettempdir() in os.fspath(ttd)
        assert initial not in ttd.name
        assert initial not in ttd.new
        assert initial not in os.fspath(ttd)
        assert initial in ttd.old
        assert not ttd.subdirectory('yodogg').exists
        assert ttd.subdirectory('yodogg').makedirs().exists
        assert 'yodogg' in ttd
        assert ttd.prefix == "test-temporarydirectory-"
        assert not ttd.suffix
        assert not ttd._parent
        assert ttd.prefix in ttd.name
        assert ttd.exists
        assert ttd.destroy
        assert ttd.will_change
        assert ttd.did_change
        assert ttd.will_change_back
        assert not ttd.did_change_back
        assert type(ttd.directory(ttd.new)) == Directory
        assert isinstance(ttd,                          TemporaryDirectory)
        assert isinstance(ttd,                          Directory)
        assert isinstance(ttd,                          collections.abc.Hashable)
        assert isinstance(ttd,                          collections.abc.Mapping)
        assert isinstance(ttd,                          collections.abc.Sized)
        assert isinstance(ttd,                          contextlib.AbstractContextManager)
        assert isinstance(ttd,                          os.PathLike)
        p = ttd.parent()
        assert os.path.samefile(os.fspath(p),           gettempdir())
        assert os.path.isdir(os.fspath(ttd))
        assert os.path.isdir(os.fspath(p))
        assert ttd.basename in p
        assert ttd.basename in ttd.dirname
        assert ttd.dirname == p
        # Stash the TemporaryDirectory’s path as a Directory
        # instance, to later assert that it no longer exists –
        # that it has been correctly deleted on scope exit:
        tdp = Directory(ttd)
        assert tdp.exists
        print("* TemporaryDirectory object tests completed OK")
        print("")
    
    # Confirm that the TemporaryDirectory has been deleted:
    assert not tdp.exists

if __name__ == '__main__':
    test()
