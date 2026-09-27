import importlib


def test_all_engines_import():
    for number in range(1, 37):
        importlib.import_module(f"sebrain.c{number:02d}")


def test_public_facade_imports():
    from sebrain import SEBrain, KnowledgeFabricLoader, RAGPipeline, BrainDatasetBridge
    assert SEBrain is not None
    assert KnowledgeFabricLoader is not None
    assert RAGPipeline is not None
    assert BrainDatasetBridge is not None
