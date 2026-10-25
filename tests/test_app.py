from io import BytesIO
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_empty_state(monkeypatch):
    monkeypatch.delenv("WISATA_DATA_PATH", raising=False)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "menunggu indeks BGE" in app.info[0].value
    assert not app.metric


def test_app_loads_dataset_and_filters_season(raw_reviews, monkeypatch, tmp_path):
    buf = BytesIO()
    raw_reviews.to_excel(buf, index=False)
    dataset = tmp_path / "reviews.xlsx"
    dataset.write_bytes(buf.getvalue())
    monkeypatch.setenv("WISATA_DATA_PATH", str(dataset))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert [item.value for item in app.metric] == ["5", "2", "2", "2"]
    app.radio[0].set_value("Seluruh dataset (audit)").run()
    assert len(app.dataframe[0].value) == 2
    app.selectbox(key="catalog_season").select("Hujan").run()
    assert not app.exception
    assert app.dataframe[0].value["destinasi"].tolist() == ["Museum Contoh"]
    app.selectbox(key="catalog_category").select("pantai").run()
    assert app.dataframe[0].value.empty


def test_app_handles_bad_dataset_without_traceback(monkeypatch, tmp_path):
    path = tmp_path / "invalid.xlsx"
    path.write_bytes(b"invalid")
    monkeypatch.setenv("WISATA_DATA_PATH", str(path))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "workbook" in app.error[0].value
