from sebrain.api import create_app


def test_fastapi_boundary_exposes_real_brain_routes(tmp_path):
    app = create_app(
        datasets_dir=tmp_path / "datasets",
        data_dir=tmp_path / ".sebrain",
    )
    paths = {route.path for route in app.routes}
    assert {"/health", "/knowledge", "/analyze", "/ask"}.issubset(paths)
