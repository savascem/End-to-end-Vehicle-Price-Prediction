from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from base.views import result
from . import settings

urlpatterns = [
    path('', include('base.urls')),
    path('admin/', admin.site.urls),
    path('graph/', result, name='graph'),
]
