# urls.py

from django.urls import path
from .views import analyze_skin_api

urlpatterns = [
    path('api/analyze-skin/', analyze_skin_api, name='analyze_skin_api'),
    # other URL patterns
]
