from sebrain.c30 import CrossLanguageReasoner, LanguageId, ConceptKind


def test_c30_detector_ignores_comment_only_lines():
    r = CrossLanguageReasoner()
    source = """
# class Fake:
# async def fake():
# import fake
"""
    assert r.detect(source, language=LanguageId.PYTHON) == []


def test_c30_detector_ignores_comment_only_lines_for_cpp():
    r = CrossLanguageReasoner()
    source = """
// class Fake {}
// namespace fake {}
// std::thread worker;
"""
    assert r.detect(source, language=LanguageId.CPP) == []


def test_c30_detector_still_detects_real_code_after_comments():
    r = CrossLanguageReasoner()
    source = """
# class Fake:
class Real:
    pass
"""
    detections = r.detect(source, language=LanguageId.PYTHON)
    assert any(d.concept is ConceptKind.CLASS and d.line == 3 for d in detections)
