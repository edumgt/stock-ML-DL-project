from os import getenv

from fastapi import HTTPException
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from api.routers import mongodb_crud, webapp


def _json_error(message, status_code=400):
    return jsonify({"detail": message}), status_code


def _call(model_cls, handler, *args, payload=None, **kwargs):
    try:
        req = model_cls(**(payload or {}))
        return handler(*args, req, **kwargs), 200
    except ValidationError as exc:
        return {"detail": exc.errors()}, 422
    except HTTPException as exc:
        return {"detail": exc.detail}, exc.status_code


def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    cors_origins = [origin.strip() for origin in getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()]
    CORS(app, resources={r"/*": {"origins": cors_origins}})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "flask-api"})

    @app.post("/api/webapp/crawl")
    def crawl():
        data, status = _call(webapp.CrawlReq, webapp.crawl, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.post("/api/webapp/cluster")
    def cluster():
        data, status = _call(webapp.ClusterReq, webapp.cluster, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.post("/api/webapp/ml-predict")
    def ml_predict():
        data, status = _call(webapp.MLPredictReq, webapp.ml_predict, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.post("/api/webapp/dl-predict")
    def dl_predict():
        data, status = _call(webapp.DLPredictReq, webapp.dl_predict, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.post("/api/webapp/stock-forecast")
    def stock_forecast():
        data, status = _call(webapp.StockForecastReq, webapp.stock_forecast, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.get("/api/mongo/health")
    def mongo_health():
        try:
            return jsonify(mongodb_crud.health())
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.post("/api/mongo/users")
    def create_user():
        data, status = _call(mongodb_crud.UserCreateReq, mongodb_crud.create_user, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.post("/api/mongo/auth/login")
    def mongo_login():
        data, status = _call(mongodb_crud.LoginReq, mongodb_crud.login, payload=request.get_json(silent=True))
        return jsonify(data), status

    @app.get("/api/mongo/users")
    def list_users():
        try:
            limit = int(request.args.get("limit", 50))
            return jsonify(mongodb_crud.list_users(limit=limit))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.get("/api/mongo/crawls")
    def list_crawls():
        try:
            limit = int(request.args.get("limit", 50))
            return jsonify(mongodb_crud.list_crawls(limit=limit))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.get("/api/mongo/analyses")
    def list_analyses():
        try:
            limit = int(request.args.get("limit", 50))
            return jsonify(mongodb_crud.list_analyses(limit=limit))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.delete("/api/mongo/users/<user_id>")
    def delete_user(user_id):
        try:
            return jsonify(mongodb_crud.delete_user(user_id))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.delete("/api/mongo/crawls/<crawl_id>")
    def delete_crawl(crawl_id):
        try:
            return jsonify(mongodb_crud.delete_crawl(crawl_id))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    @app.delete("/api/mongo/analyses/<analysis_id>")
    def delete_analysis(analysis_id):
        try:
            return jsonify(mongodb_crud.delete_analysis(analysis_id))
        except HTTPException as exc:
            return _json_error(exc.detail, exc.status_code)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(getenv("PORT", "5000")), debug=True)
