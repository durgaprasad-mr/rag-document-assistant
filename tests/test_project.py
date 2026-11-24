from pathlib import Path


def test_project_layout():
    assert Path("backend/app.py").exists()
    assert Path("backend/ingest.py").exists()
    assert Path("backend/rag_pipeline.py").exists()


def test_required_directories_exist():
    assert Path("frontend").is_dir()
    assert Path("data").is_dir()
