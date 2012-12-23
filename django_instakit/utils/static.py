
from os.path import join
#from django.contrib.staticfiles.finders import AppDirectoriesFinder
from django.contrib.staticfiles.storage import AppStaticStorage

#storage = AppDirectoriesFinder().storages.get('django_instakit')
storage = AppStaticStorage('django_instakit')
listfiles = lambda *pth: storage.listdir(join('django_instakit', *pth))[-1]
path = lambda *pth: storage.path(join('django_instakit', *pth))