"""
api/auth.py
-----------
Minimal, secure authentication layer for Automated MLSecOps.

Design decisions:
  - Separate SQLite database (auth.db) — never touches mlflow.db or any existing data.
  - Passwords hashed with bcrypt via passlib; never stored in plain text.
  - JWT access tokens (HS256) with configurable expiry; secret loaded from env-var.
  - Router mounts at /auth — zero changes to existing routes.
  - Optional: provides a FastAPI dependency `get_current_user` that other routes
    can import to enforce ownership in future iterations.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Passlib / JWT imports with graceful fallback message ──────────────────────
try:
    import bcrypt as _bcrypt
    from jose import JWTError, jwt
except ImportError as _e:  # noqa: F841
    raise ImportError(
        "Auth dependencies missing. Run: py -m pip install bcrypt 'python-jose[cryptography]>=3.3'"
    ) from _e

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AUTH_DB_PATH = _PROJECT_ROOT / "auth.db"

_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
if not _SECRET_KEY:
    raise RuntimeError("AUTH_SECRET_KEY must be set")

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours

# ─────────────────────────────────────────────────────────────────────────────
# Database setup (separate auth.db)
# ─────────────────────────────────────────────────────────────────────────────

_engine = create_engine(
    f"sqlite:///{_AUTH_DB_PATH}",
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


class _Base(DeclarativeBase):
    pass


class UserDB(_Base):
    """Minimal users table — only what is needed for auth."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(254), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    full_name = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def _create_tables() -> None:
    """Create auth tables if they do not exist (safe, idempotent)."""
    _Base.metadata.create_all(bind=_engine)


def _get_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing (bcrypt directly — avoids passlib 1.7 / bcrypt 4.x compat issues)
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_access_token(user_id: int, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT; raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or len(v.split("@")[0]) == 0 or len(v.split("@")[1]) == 0:
            raise ValueError("Invalid email address format")
        return v

    @field_validator("password")
    @classmethod
    def _pw_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    user_id: int
    email: str
    full_name: Optional[str] = None
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency — inject current user
# ─────────────────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(_get_db),
) -> UserDB:
    """
    FastAPI dependency.  Import and add to any route that requires an
    authenticated user:

        @app.get("/protected")
        def protected(user: UserDB = Depends(get_current_user)):
            ...
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    user_id = int(payload.get("sub", 0))
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(_get_db)):
    """Register a new user account."""
    existing = db.query(UserDB).filter(UserDB.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    user = UserDB(
        email=req.email,
        password_hash=_hash_password(req.password),
        full_name=req.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = _create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(_get_db)):
    """Authenticate with email + password; returns a JWT access token."""
    user = db.query(UserDB).filter(UserDB.email == req.email).first()
    if user is None or not _verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = _create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: UserDB = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserResponse(
        user_id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
    )


@router.post("/verify-token")
def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
):
    """Lightweight token verification endpoint for the Streamlit frontend."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No token")
    payload = _decode_token(credentials.credentials)
    return {"valid": True, "user_id": payload.get("sub"), "email": payload.get("email")}
