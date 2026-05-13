"""MongoDB CRUD 라우터."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import Any

from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.mongodb_store import PyMongoError, repo

router = APIRouter(prefix="/api/mongo", tags=["MongoDB CRUD"])
logger = logging.getLogger(__name__)


class UserCreateReq(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    password: str = Field(min_length=4, max_length=200)
    role: str = Field(default="user", max_length=30)


class UserUpdateReq(BaseModel):
    password: str | None = Field(default=None, min_length=4, max_length=200)
    role: str | None = Field(default=None, max_length=30)
    active: bool | None = None


class LoginReq(BaseModel):
    username: str
    password: str


class CrawlCreateReq(BaseModel):
    ticker: str
    market: str
    pages: int = Field(default=5, ge=1, le=120)
    ohlcv_rows: int = 0
    latest_ohlcv: dict[str, Any] | None = None
    stock_info: dict[str, Any] | None = None
    market_sample: list[dict[str, Any]] = Field(default_factory=list)


class CrawlUpdateReq(BaseModel):
    market: str | None = None
    pages: int | None = Field(default=None, ge=1, le=120)
    ohlcv_rows: int | None = None
    latest_ohlcv: dict[str, Any] | None = None
    stock_info: dict[str, Any] | None = None
    market_sample: list[dict[str, Any]] | None = None


class AnalysisCreateReq(BaseModel):
    analysis_type: str = Field(description="ml | dl | cluster | timeseries 등")
    ticker: str | None = None
    tickers: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class AnalysisUpdateReq(BaseModel):
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    memo: str | None = None


PBKDF2_ROUNDS = 600000
DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$600000$7c6f3bc5a0ec01d1cd0e9f8f6f2c40b6$"
    "5ef95915c8a7a8520bd6d54d52d35f9a194a60e9e7d2c69cdcfba5044a778803"
)


def _hash_password(password: str) -> str:
    salt_hex = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ROUNDS,
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt_hex}${digest}"


def _verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, rounds, salt_hex, expected = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(rounds),
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _guard():
    try:
        repo.ping()
    except PyMongoError as e:
        logger.warning("MongoDB 연결 실패: %s", e)
        raise HTTPException(status_code=503, detail="MongoDB 연결 실패")


def _handle_id_error():
    raise HTTPException(status_code=400, detail="잘못된 id 형식입니다.")


@router.get("/health")
def health():
    _guard()
    return {"status": "ok", "db": "mongodb"}


@router.post("/users")
def create_user(req: UserCreateReq):
    _guard()
    exists = repo.users.find_one({"username": req.username})
    if exists:
        raise HTTPException(status_code=409, detail="이미 존재하는 username입니다.")
    user_id = repo.insert_one(
        repo.users,
        {
            "username": req.username,
            "password_hash": _hash_password(req.password),
            "role": req.role,
            "active": True,
        },
    )
    return {"id": user_id}


@router.post("/auth/login")
def login(req: LoginReq):
    _guard()
    user = repo.users.find_one({"username": req.username, "active": {"$ne": False}})
    stored_hash = user.get("password_hash") if user else DUMMY_PASSWORD_HASH
    valid = _verify_password(req.password, stored_hash)
    if not user or not valid:
        raise HTTPException(status_code=401, detail="로그인 실패")
    user_doc = repo.serialize_doc(user)
    user_doc.pop("password_hash", None)
    return {"message": "로그인 성공", "user": user_doc}


@router.get("/users")
def list_users(limit: int = Query(default=50, ge=1, le=200)):
    _guard()
    docs = repo.get_many(repo.users, limit=limit)
    for doc in docs:
        doc.pop("password_hash", None)
    return docs


@router.get("/users/{user_id}")
def get_user(user_id: str):
    _guard()
    try:
        doc = repo.get_one(repo.users, user_id)
    except InvalidId:
        _handle_id_error()
    if not doc:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    doc.pop("password_hash", None)
    return doc


@router.put("/users/{user_id}")
def update_user(user_id: str, req: UserUpdateReq):
    _guard()
    updates = req.model_dump(exclude_none=True)
    if "password" in updates:
        updates["password_hash"] = _hash_password(updates.pop("password"))
    if not updates:
        raise HTTPException(status_code=400, detail="수정할 필드가 없습니다.")
    try:
        ok = repo.update_one(repo.users, user_id, updates)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"status": "ok"}


@router.delete("/users/{user_id}")
def delete_user(user_id: str):
    _guard()
    try:
        ok = repo.delete_one(repo.users, user_id)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"status": "ok"}


@router.post("/crawls")
def create_crawl(req: CrawlCreateReq):
    _guard()
    doc_id = repo.insert_one(repo.crawls, req.model_dump())
    return {"id": doc_id}


@router.get("/crawls")
def list_crawls(limit: int = Query(default=50, ge=1, le=200)):
    _guard()
    return repo.get_many(repo.crawls, limit=limit)


@router.get("/crawls/{crawl_id}")
def get_crawl(crawl_id: str):
    _guard()
    try:
        doc = repo.get_one(repo.crawls, crawl_id)
    except InvalidId:
        _handle_id_error()
    if not doc:
        raise HTTPException(status_code=404, detail="크롤링 기록을 찾을 수 없습니다.")
    return doc


@router.put("/crawls/{crawl_id}")
def update_crawl(crawl_id: str, req: CrawlUpdateReq):
    _guard()
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="수정할 필드가 없습니다.")
    try:
        ok = repo.update_one(repo.crawls, crawl_id, updates)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="크롤링 기록을 찾을 수 없습니다.")
    return {"status": "ok"}


@router.delete("/crawls/{crawl_id}")
def delete_crawl(crawl_id: str):
    _guard()
    try:
        ok = repo.delete_one(repo.crawls, crawl_id)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="크롤링 기록을 찾을 수 없습니다.")
    return {"status": "ok"}


@router.post("/analyses")
def create_analysis(req: AnalysisCreateReq):
    _guard()
    doc_id = repo.insert_one(repo.analyses, req.model_dump())
    return {"id": doc_id}


@router.get("/analyses")
def list_analyses(limit: int = Query(default=50, ge=1, le=200)):
    _guard()
    return repo.get_many(repo.analyses, limit=limit)


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    _guard()
    try:
        doc = repo.get_one(repo.analyses, analysis_id)
    except InvalidId:
        _handle_id_error()
    if not doc:
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return doc


@router.put("/analyses/{analysis_id}")
def update_analysis(analysis_id: str, req: AnalysisUpdateReq):
    _guard()
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="수정할 필드가 없습니다.")
    try:
        ok = repo.update_one(repo.analyses, analysis_id, updates)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return {"status": "ok"}


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    _guard()
    try:
        ok = repo.delete_one(repo.analyses, analysis_id)
    except InvalidId:
        _handle_id_error()
    if not ok:
        raise HTTPException(status_code=404, detail="분석 기록을 찾을 수 없습니다.")
    return {"status": "ok"}
