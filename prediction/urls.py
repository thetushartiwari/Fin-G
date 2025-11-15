from django.urls import path
from .views import analyze_user_investment ,index
from .views import success_stories

urlpatterns = [
    path('', index, name="index"),  
    path('analyze_user_investment/', analyze_user_investment, name="analyze_user_investment"),
    path('success-stories/', success_stories, name="success_stories")
]

