
clean: clean-pyc clean-cython

distclean: clean-pyc clean-cython clean-build-artifacts

rebuild: distclean cython

dist: rebuild sdist twine-upload

upload: bump dist

bigupload: bigbump dist

clean-pyc:
	find . -name \*.pyc -print -delete

clean-cython:
	find . -name \*.so -print -delete

clean-build-artifacts:
	rm -rf build dist instakit.egg-info

cython:
	python setup.py build_ext --inplace

sdist:
	python setup.py sdist

twine-upload:
	twine upload -s --repository-url=https://upload.pypi.org/legacy/ dist/*

bump:
	bumpversion --verbose patch

bigbump:
	bumpversion --verbose minor

check:
	check-manifest -v
	python setup.py check -m -s
	travis lint .travis.yml

.PHONY: clean-pyc clean-cython clean-build-artifacts
.PHONY: clean distclean rebuild dist upload bigupload
.PHONY: cython sdist twine-upload bump bigbump check

