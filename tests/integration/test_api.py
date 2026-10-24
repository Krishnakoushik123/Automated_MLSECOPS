"""
tests/integration/test_api.py
------------------------------
Integration tests for FastAPI inference & security API endpoints:
  - GET  /health
  - GET  /metrics
  - GET  /security/audit
  - POST /predict
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"mlsecops_" in response.content

    def test_security_audit_endpoint(self):
        response = client.get("/security/audit")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data

    def test_predict_endpoint(self):
        payload = {
            "features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 0.0,
                          0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 0.0]]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "latency_ms" in data

    def test_unsupervised_experiment_invalid_algo(self):
        csv_content = "f1,f2,f3\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
        files = {"file": ("data.csv", csv_content, "text/csv")}
        response = client.post(
            "/experiments/run/unsupervised",
            data={"algorithm": "invalid_algo", "task_type": "clustering"},
            files=files,
        )
        assert response.status_code == 400

    def test_unsupervised_experiment_queue(self):
        import io, numpy as np, pandas as pd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(4)])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        files = {"file": ("cluster_data.csv", buf.getvalue(), "text/csv")}
        response = client.post(
            "/experiments/run/unsupervised",
            data={"algorithm": "kmeans", "task_type": "clustering", "n_clusters": "3"},
            files=files,
        )
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert data["status"] == "QUEUED"
        assert data["algorithm"] == "kmeans"

        # Check status endpoint
        exp_id = data["experiment_id"]
        status_res = client.get(f"/experiments/unsupervised/{exp_id}")
        assert status_res.status_code == 200
        assert status_res.json()["experiment_id"] == exp_id

    def test_unsupervised_experiment_not_found(self):
        response = client.get("/experiments/unsupervised/nonexistent_id_9999")
        assert response.status_code == 404

    def test_image_experiment_invalid_format(self):
        files = {"file": ("test.txt", b"not a zip file", "text/plain")}
        response = client.post(
            "/experiments/run/image",
            data={"task": "image_classification"},
            files=files,
        )
        assert response.status_code == 400

    def test_image_experiment_not_found(self):
        response = client.get("/experiments/image/nonexistent_img_9999")
        assert response.status_code == 404

