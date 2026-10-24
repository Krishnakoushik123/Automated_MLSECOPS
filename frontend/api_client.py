import os
import requests
import time
import streamlit as st
from typing import Dict, Any, List, Optional

class APIClient:
    def __init__(self, base_url: str = None):
<<<<<<< HEAD
        self.base_url = base_url or os.getenv("API_URL", "http://localhost:8000")
        self._token: Optional[str] = None  # JWT set after login

    def set_token(self, token: str) -> None:
        """Store the JWT received after login/register."""
        self._token = token

    def clear_token(self) -> None:
        """Clear the stored token (logout)."""
        self._token = None

    def _auth_headers(self) -> Dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}
=======
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.getenv("API_URL"):
            self.base_url = os.getenv("API_URL").rstrip("/")
        elif "API_URL" in st.secrets:
            self.base_url = st.secrets["API_URL"].rstrip("/")
        else:
            self.base_url = "http://localhost:8000"

>>>>>>> 796b41cb3b0875121c6c37f47ced37dbca9337d8
        
    def _get(self, endpoint: str) -> Optional[Any]:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=self._auth_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"APIClient GET {endpoint} Error: {e}")
            return None

    def _post(self, endpoint: str, data: Dict = None, files: Dict = None) -> Optional[Any]:
        try:
            if files:
                response = requests.post(f"{self.base_url}{endpoint}", files=files, headers=self._auth_headers(), timeout=30)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data, headers=self._auth_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"APIClient POST {endpoint} Error: {e}")
            return None

    def get_health(self) -> Optional[Dict[str, Any]]:
        return self._get("/health")

    def get_experiments(self) -> List[Dict[str, Any]]:
        res = self._get("/experiments")
        if res and "experiments" in res:
            return res["experiments"]
        return []

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._get(f"/experiments/{experiment_id}")

    def run_pipeline(self, target_column: str = "target", opt_metric: str = "f1", models_to_train: List[str] = None, task_type: str = "classification") -> Optional[Dict[str, Any]]:
        if models_to_train is None:
            models_to_train = ["logistic_regression", "random_forest", "gradient_boosting"]
        data = {
            "target_column": target_column,
            "opt_metric": opt_metric,
            "models_to_train": models_to_train,
            "task_type": task_type
        }
        return self._post("/experiments/run", data=data)

    def upload_dataset(self, file_name: str, file_bytes: bytes) -> Optional[Dict[str, Any]]:
        files = {"file": (file_name, file_bytes, "application/octet-stream")}
        return self._post("/data/upload", files=files)

    def get_metrics(self) -> Optional[str]:
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"APIClient GET /metrics Error: {e}")
            return None

    def get_monitoring_stats(self) -> Optional[Dict[str, Any]]:
        return self._get("/monitoring/stats")

    def load_model(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._post(f"/experiments/{experiment_id}/model/load")

    def get_model_download_url(self, experiment_id: str) -> str:
        return f"{self.base_url}/experiments/{experiment_id}/model/download"

    def submit_ground_truth(self, indices: list, labels: list) -> Optional[Dict[str, Any]]:
        return self._post("/monitoring/ground_truth", data={"prediction_indices": indices, "ground_truth": labels})

    # ── Auth endpoints ────────────────────────────────────────────────────────

    def register(self, email: str, password: str, full_name: str = None) -> Optional[Dict[str, Any]]:
        """Register a new user. Returns token payload or error dict."""
        try:
            payload = {"email": email, "password": password}
            if full_name:
                payload["full_name"] = full_name
            response = requests.post(
                f"{self.base_url}/auth/register", json=payload, timeout=10
            )
            if response.status_code == 409:
                return {"error": "Email already registered"}
            if not response.ok:
                try:
                    err_json = response.json()
                    detail = err_json.get("detail")
                    if isinstance(detail, list) and len(detail) > 0:
                        detail = detail[0].get("msg", str(detail))
                    return {"error": detail or f"Registration failed ({response.status_code})"}
                except Exception:
                    return {"error": f"Registration failed ({response.status_code})"}
            data = response.json()
            if "access_token" in data:
                self.set_token(data["access_token"])
            return data
        except Exception as e:
            print(f"APIClient register error: {e}")
            return None

    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Login with email+password. Returns token payload or error dict."""
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"email": email, "password": password},
                timeout=10,
            )
            if response.status_code == 401:
                return {"error": "Invalid email or password"}
            if not response.ok:
                try:
                    err_json = response.json()
                    detail = err_json.get("detail")
                    if isinstance(detail, list) and len(detail) > 0:
                        detail = detail[0].get("msg", str(detail))
                    return {"error": detail or f"Login failed ({response.status_code})"}
                except Exception:
                    return {"error": f"Login failed ({response.status_code})"}
            data = response.json()
            if "access_token" in data:
                self.set_token(data["access_token"])
            return data
        except Exception as e:
            print(f"APIClient login error: {e}")
            return None

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Fetch the authenticated user's profile."""
        return self._get("/auth/me")

    def verify_stored_token(self) -> bool:
        """Verify the stored token is still valid. Returns True if valid."""
        if not self._token:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/auth/verify-token",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=5,
            )
            return response.status_code == 200 and response.json().get("valid", False)
        except Exception:
            return False
