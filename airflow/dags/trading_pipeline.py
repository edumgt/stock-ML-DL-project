from datetime import datetime
from os import getenv

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator


API_BASE = getenv("FLASK_API_BASE_URL", "http://flask-api:5000")


def post_json(endpoint, payload):
    response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def crawl_market():
    return post_json(
        "/api/webapp/crawl",
        {"ticker": "005930", "market": "kospi", "pages": 10},
    )


def run_ml_signal():
    return post_json(
        "/api/webapp/ml-predict",
        {
            "ticker": "005930",
            "source": "naver",
            "pages": 30,
            "period": "3y",
            "model_type": "rf",
            "forward_days": 5,
            "threshold": 0.01,
        },
    )


def run_forecast():
    return post_json(
        "/api/webapp/stock-forecast",
        {
            "ticker": "005930",
            "source": "naver",
            "pages": 20,
            "period": "1y",
        },
    )


with DAG(
    dag_id="stock_market_pipeline",
    description="Daily crawling and forecast pipeline via Flask API",
    start_date=datetime(2026, 5, 1),
    schedule="0 18 * * 1-5",
    catchup=False,
    tags=["stocks", "ml", "forecast"],
) as dag:
    crawl_task = PythonOperator(
        task_id="crawl_market_data",
        python_callable=crawl_market,
    )

    ml_task = PythonOperator(
        task_id="run_ml_signal",
        python_callable=run_ml_signal,
    )

    forecast_task = PythonOperator(
        task_id="run_daily_forecast",
        python_callable=run_forecast,
    )

    crawl_task >> ml_task >> forecast_task
