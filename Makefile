
clean: clean-pyc clean-cython

distclean: clean-all-pyc clean-all-cython

clean-pyc:
	find . -name \*.pyc -print -delete

clean-all-pyc:
	find . -name \*.pyc -print -delete

clean-cython:
	find . -name \*.so -print -delete

clean-all-cython:
	find . -name \*.so -print -delete
	find . -name \*.c -print -delete

cython:
	python setup.py build_ext --inplace

.PHONY: distclean clean cython
