"""API tests (CLAUDE.md Section 6): health, schema, valid predict, missing field
400, negative value 400, batch with 3 rows, model loads with pinned versions."""
import importlib.metadata as md
import io
import sys
from pathlib import Path

import pytest

# backend/ on path so `import app` / `import predictor` work from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from predictor import Predictor  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.fixture(scope="module")
def features():
    return Predictor().features


@pytest.fixture(scope="module")
def medians():
    schema = Predictor().schema
    return {f["name"]: f["median"] for f in schema["features"]}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert isinstance(body["version"], str)


def test_schema(client, features):
    r = client.get("/api/schema")
    assert r.status_code == 200
    body = r.get_json()
    assert body["order"] == features
    assert len(body["features"]) == len(features)


def test_model_info(client, features):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"]["name"]
    assert body["features"] == features


def test_predict_valid(client, medians):
    r = client.post("/api/predict", json={"metrics": medians})
    assert r.status_code == 200
    body = r.get_json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["risk_tier"] in ("low", "elevated", "critical")
    assert isinstance(body["threshold_flags"], list)
    assert len(body["top_contributors"]) >= 1


def test_predict_missing_field_400(client, medians):
    bad = dict(medians)
    bad.pop(next(iter(bad)))  # drop one required feature
    r = client.post("/api/predict", json={"metrics": bad})
    assert r.status_code == 400
    assert r.get_json()["error"] == "validation_error"


def test_predict_negative_400(client, medians):
    bad = dict(medians)
    bad[next(iter(bad))] = -5
    r = client.post("/api/predict", json={"metrics": bad})
    assert r.status_code == 400
    body = r.get_json()
    assert body["error"] == "validation_error"
    assert "must be a non-negative number" in next(iter(body["fields"].values()))


def test_predict_ignores_unknown_keys(client, medians):
    payload = dict(medians)
    payload["NotAFeature"] = 1
    r = client.post("/api/predict", json={"metrics": payload})
    assert r.status_code == 200
    assert "NotAFeature" in r.get_json().get("ignored", [])


def test_batch_three_rows(client, features, medians):
    header = ",".join(features)
    row = ",".join(str(medians[f]) for f in features)
    csv = "\n".join([header, row, row, row])
    data = {"file": (io.BytesIO(csv.encode()), "rows.csv")}
    r = client.post("/api/predict/batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 3
    probs = [x["probability"] for x in body["results"]]
    assert probs == sorted(probs, reverse=True)  # sorted by probability desc


def test_batch_missing_column_400(client, features, medians):
    cols = features[:-1]  # drop one column
    header = ",".join(cols)
    row = ",".join(str(medians[f]) for f in cols)
    data = {"file": (io.BytesIO(f"{header}\n{row}".encode()), "bad.csv")}
    r = client.post("/api/predict/batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["error"] == "validation_error"


def test_model_loads_with_pinned_versions():
    """Load-critical libs must match ml/requirements pins so model.joblib deserializes."""
    p = Predictor()
    assert p.model is not None
    expected = {"numpy": "2.5.4", "scikit-learn": "1.9.1",
                "xgboost": "3.4.1", "imbalanced-learn": "0.14.2"}
    for lib, ver in expected.items():
        assert md.version(lib) == ver, f"{lib} {md.version(lib)} != pinned {ver}"
