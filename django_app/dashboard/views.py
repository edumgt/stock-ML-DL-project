from os import getenv

from django.http import JsonResponse
from django.shortcuts import render


def home(request):
    return render(
        request,
        "dashboard/index.html",
        {
            "api_base_url": getenv("FLASK_API_BASE_URL", "http://127.0.0.1:5000"),
            "airflow_ui_url": getenv("AIRFLOW_UI_URL", "http://127.0.0.1:8080"),
        },
    )


def health(request):
    return JsonResponse({"status": "ok", "service": "django-web"})
