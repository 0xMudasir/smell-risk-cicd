"""Phase 6 - Export artifacts to the app.

Copies the RQ JSON results from ml/artifacts/ to frontend/src/assets/results/ and
the model files from training to backend/model/, so the frontend and API both
read exactly what the pipeline produced (Rule 2: never invent numbers).

Run:  python -m ml.src.export_artifacts
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 6.")


if __name__ == "__main__":
    main()
