"""MongoDB 저장소 유틸."""

from __future__ import annotations

from datetime import datetime, timezone
from os import getenv
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_doc(payload: dict[str, Any]) -> dict[str, Any]:
    doc = dict(payload)
    doc["created_at"] = _now_utc()
    doc["updated_at"] = _now_utc()
    return doc


def _serialize(value: Any):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


class MongoRepository:
    def __init__(self):
        self._client = MongoClient(getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self._db = self._client[getenv("MONGODB_DB_NAME", "stock_mldl")]

    @property
    def users(self) -> Collection:
        return self._db["login_users"]

    @property
    def crawls(self) -> Collection:
        return self._db["crawl_data"]

    @property
    def analyses(self) -> Collection:
        return self._db["analysis_data"]

    def ping(self) -> bool:
        self._client.admin.command("ping")
        return True

    @staticmethod
    def parse_id(value: str) -> ObjectId:
        return ObjectId(value)

    @staticmethod
    def serialize_doc(doc: dict[str, Any] | None):
        if not doc:
            return None
        result = _serialize(doc)
        if "_id" in result:
            result["id"] = result.pop("_id")
        return result

    @staticmethod
    def normalize_many(cursor):
        return [MongoRepository.serialize_doc(d) for d in cursor]

    def insert_one(self, collection: Collection, payload: dict[str, Any]) -> str:
        res = collection.insert_one(_to_doc(payload))
        return str(res.inserted_id)

    def update_one(self, collection: Collection, doc_id: str, payload: dict[str, Any]) -> bool:
        updates = dict(payload)
        updates["updated_at"] = _now_utc()
        res = collection.update_one({"_id": self.parse_id(doc_id)}, {"$set": updates})
        return res.matched_count == 1

    def get_one(self, collection: Collection, doc_id: str):
        return self.serialize_doc(collection.find_one({"_id": self.parse_id(doc_id)}))

    def get_many(self, collection: Collection, limit: int = 50):
        docs = collection.find().sort("created_at", -1).limit(limit)
        return self.normalize_many(docs)

    def delete_one(self, collection: Collection, doc_id: str) -> bool:
        res = collection.delete_one({"_id": self.parse_id(doc_id)})
        return res.deleted_count == 1


repo = MongoRepository()

__all__ = ["PyMongoError", "repo"]
