
from os.path import join
#from django.contrib.staticfiles.finders import AppDirectoriesFinder
from django.contrib.staticfiles.storage import AppStaticStorage

#storage = AppDirectoriesFinder().storages.get('instakit')
storage = AppStaticStorage('instakit')
listfiles = lambda *pth: storage.listdir(join('instakit', *pth))[-1]
path = lambda *pth: storage.path(join('instakit', *pth))
