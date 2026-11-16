import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from topic_modeling import clean_tweet


class TestCleanTweet(unittest.TestCase):
    def test_removes_rt_prefix_and_mentions_and_urls(self):
        raw = "RT @usuario1: Mirá esto @amigo https://t.co/abc12345 increíble!"
        self.assertEqual(clean_tweet(raw), "Mirá esto increíble!")

    def test_removes_http_and_www_urls(self):
        raw = "Visita http://ejemplo.com y también www.ejemplo.org/ruta ahora"
        self.assertEqual(clean_tweet(raw), "Visita y también ahora")

    def test_normalizes_whitespace(self):
        raw = "  Hola   mundo \n\t esto   es   un tweet  "
        self.assertEqual(clean_tweet(raw), "Hola mundo esto es un tweet")

    def test_handles_non_string_and_empty(self):
        self.assertEqual(clean_tweet(""), "")
        self.assertEqual(clean_tweet(None), "")


if __name__ == "__main__":
    unittest.main()
