
clean: clean-pyc clean-cython

distclean: clean-all-pyc clean-cython clean-build-artifacts

dist: cython distclean upload

upload: bump sdist twine-upload

bigupload: bigbump sdist twine-upload

clean-pyc:
	find . -name \*.pyc -print -delete

clean-all-pyc:
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
	twine upload --repository-url=https://upload.pypi.org/legacy/ dist/*

bump:
	bumpversion --verbose patch

bigbump:
	bumpversion --verbose minor


.PHONY: bigbump bump clean distclean dist cython upload
