import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
APP_DIR = os.path.join(ROOT, "app")

pkg = types.ModuleType("app")
pkg.__path__ = [APP_DIR]
sys.modules.setdefault("app", pkg)

spec = importlib.util.spec_from_file_location("app.main", os.path.join(APP_DIR, "main.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["app.main"] = module
spec.loader.exec_module(module)
_analyze_language = module._analyze_language


class LanguageAnalysisTests(unittest.TestCase):
    def test_indonesian_slang_is_detected_as_indonesian(self):
        tags, ratios = _analyze_language("gue mau coba nih")
        self.assertEqual(tags.get("IND"), "100%")
        self.assertEqual(ratios.get("IND"), 1.0)

    def test_english_slang_is_detected_as_english(self):
        tags, ratios = _analyze_language("yo dude please help me")
        self.assertEqual(tags.get("EN"), "100%")
        self.assertEqual(ratios.get("EN"), 1.0)

    def test_arabic_transliteration_is_detected_as_arabic(self):
        tags, ratios = _analyze_language("assalamualaikum habibi")
        self.assertEqual(tags.get("AR"), "100%")
        self.assertEqual(ratios.get("AR"), 1.0)


if __name__ == "__main__":
    unittest.main()
