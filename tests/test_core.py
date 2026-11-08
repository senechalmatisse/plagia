"""Tests de la couche core : segmentation, empreintes, graines, registres, E/S, types, configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plagia.core.cache import DiskCache
from plagia.core.config import ExperimentConfig, ProtocolConfig, load_config
from plagia.core.errors import ConfigurationError, MissingDependencyError
from plagia.core.hashing import hash64, sha256_file, sha256_text, stable_hash, stable_json
from plagia.core.io import append_jsonl, read_json, read_jsonl, read_yaml, write_json, write_jsonl, write_yaml
from plagia.core.lexicon import function_words
from plagia.core.registry import Registry
from plagia.core.seeds import derive_seed, make_random, make_rng
from plagia.core.text import (
    count_words,
    detokenize,
    is_punctuation,
    normalize_whitespace,
    paragraph_spans,
    sentence_spans,
    sentences,
    tokenize,
    tokenize_with_offsets,
    word_spans,
    words,
)
from plagia.core.types import Corpus, Document, Passage, PassagePair
from plagia.core.versions import code_version, environment_info, git_revision, package_versions


class TestText:
    def test_sentences_respect_abbreviations_and_paragraphs(self) -> None:
        text = "Mr. Smith met the U.S. President! Was it J. Edgar? Yes.\n\nA title without dot\n\nEnd."
        assert sentences(text) == [
            "Mr. Smith met the U.S. President!",
            "Was it J. Edgar?",
            "Yes.",
            "A title without dot",
            "End.",
        ]

    def test_sentence_spans_are_exact_offsets(self) -> None:
        text = "  First one.  Second one!  "
        spans = sentence_spans(text)
        assert [text[a:b] for a, b in spans] == ["First one.", "Second one!"]

    def test_tokens_and_offsets(self) -> None:
        text = "It's 3.5% of GDP, isn't it?"
        assert tokenize(text) == ["It's", "3.5", "%", "of", "GDP", ",", "isn't", "it", "?"]
        for token, start, end in tokenize_with_offsets(text):
            assert text[start:end] == token
        assert words(text) == ["It's", "3.5", "of", "GDP", "isn't", "it"]
        assert count_words(text) == 6
        assert [text[a:b] for a, b in word_spans(text)] == words(text)

    def test_helpers(self) -> None:
        assert is_punctuation("?!") and not is_punctuation("a?")
        assert normalize_whitespace("a  b\r\n\n\n\nc ") == "a b\n\nc"
        assert detokenize(["Hello", ",", "world", "(", "yes", ")", "."]) == "Hello, world (yes)."
        assert paragraph_spans("a\n\n  \n b") == [(0, 1), (7, 8)]
        assert "the" in function_words("en") and "le" in function_words("fr")


class TestHashingAndSeeds:
    def test_stable_hash_is_order_independent(self) -> None:
        assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
        assert stable_json({"p": Path("x/y"), "s": {2, 1}}) == '{"p":"x/y","s":[1,2]}'
        assert sha256_text("abc").startswith("ba7816bf")
        assert hash64("shingle") == hash64("shingle") != hash64("other")

    def test_file_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_bytes(b"abc")
        assert sha256_file(path) == sha256_text("abc")

    def test_seeds_are_deterministic_and_independent(self) -> None:
        assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)
        assert derive_seed(42, "a") != derive_seed(42, "b") != derive_seed(43, "a")
        assert make_rng(1, "x").integers(0, 10**6) == make_rng(1, "x").integers(0, 10**6)
        assert make_random(1, "x").random() == make_random(1, "x").random()
        assert 0 <= derive_seed(7) < 2**32


class TestRegistry:
    def test_register_create_and_errors(self) -> None:
        registry: Registry[object] = Registry("composant")

        @registry.register("thing")
        class Thing:
            def __init__(self, value: int = 1) -> None:
                self.value = value

        assert "thing" in registry and len(registry) == 1 and list(registry) == ["thing"]
        assert registry.create("thing", value=3).value == 3  # type: ignore[attr-defined]
        with pytest.raises(ConfigurationError, match="Disponibles"):
            registry.get("missing")
        with pytest.raises(ConfigurationError, match="déjà"):
            registry.add("thing", Thing)
        with pytest.raises(ConfigurationError, match=r"Paramètres invalides"):
            registry.create("thing", nope=1)


class TestIO:
    def test_jsonl_yaml_json_roundtrip(self, tmp_path: Path) -> None:
        records = [{"a": 1, "é": "ü"}, {"a": 2}]
        assert write_jsonl(tmp_path / "x.jsonl", records) == 2
        append_jsonl(tmp_path / "x.jsonl", {"a": 3})
        assert [r["a"] for r in read_jsonl(tmp_path / "x.jsonl")] == [1, 2, 3]
        assert b"\r\n" not in (tmp_path / "x.jsonl").read_bytes()
        write_yaml(tmp_path / "c.yaml", {"k": [1, 2]})
        assert read_yaml(tmp_path / "c.yaml") == {"k": [1, 2]}
        write_json(tmp_path / "c.json", {"b": 1, "a": np.int64(2)})
        assert read_json(tmp_path / "c.json") == {"a": 2, "b": 1}

    def test_invalid_jsonl_line(self, tmp_path: Path) -> None:
        (tmp_path / "bad.jsonl").write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            list(read_jsonl(tmp_path / "bad.jsonl"))

    def test_disk_cache(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path, "test")
        calls = []

        def compute() -> list[int]:
            calls.append(1)
            return [1, 2, 3]

        assert cache.get_or_compute({"k": 1}, compute) == [1, 2, 3]
        assert cache.get_or_compute({"k": 1}, compute) == [1, 2, 3]
        assert len(calls) == 1 and {"k": 1} in cache
        cache.path(cache.key({"k": 1})).write_bytes(b"corrupted")
        assert cache.get_or_compute({"k": 1}, compute) == [1, 2, 3]
        cache.clear()
        assert {"k": 1} not in cache
        assert DiskCache(tmp_path, enabled=False).get_or_compute(1, compute) == [1, 2, 3]


class TestTypes:
    def test_document_roundtrip_and_group(self) -> None:
        doc = Document("d1", "text", author="A", work="w", meta={"x": 1})
        assert Document.from_dict(doc.to_dict()) == doc
        assert doc.group == "w" and Document("d2", "t").group == "d2"
        assert doc.replace(split="train").split == "train"
        assert doc.sha256 == sha256_text("text")

    def test_passage_overlap(self) -> None:
        a, b = Passage("d", 0, 10), Passage("d", 5, 10)
        assert a.end == 10 and a.overlap(b) == 5 and a.overlaps(b)
        assert a.overlap(Passage("e", 0, 10)) == 0
        assert a.text("0123456789abc") == "0123456789"
        assert Passage.from_dict(a.to_dict()) == a
        with pytest.raises(ValueError):
            Passage("d", -1, 2)

    def test_passage_pair_roundtrip(self) -> None:
        pair = PassagePair(Passage("s", 1, 2), Passage("t", 3, 4), 0.5, "O1", {"size": 50})
        again = PassagePair.from_dict(pair.to_dict())
        assert again.suspicious == pair.suspicious and again.source == pair.source and again.meta["size"] == 50
        assert PassagePair.from_dict(PassagePair(Passage("s", 0, 1)).to_dict()).source is None

    def test_corpus_operations(self) -> None:
        docs = [Document("a", "x", author="A", split="train"), Document("b", "y", author="B", split="test")]
        pair = PassagePair(Passage("b", 0, 1), Passage("a", 0, 1))
        corpus = Corpus(docs, name="c", cases=[pair], pairs=[("b", "a")])
        assert len(corpus) == 2 and "a" in corpus and docs[0] in corpus and corpus.get("b").author == "B"
        assert corpus.ids == ["a", "b"] and corpus.texts == ["x", "y"] and corpus.labels == ["A", "B"]
        assert corpus.authors == ["A", "B"] and corpus.groups == ["a", "b"]
        assert corpus.split_counts() == {"train": 1, "test": 1}
        test = corpus.split("test")
        assert test.ids == ["b"] and len(test.cases) == 1 and test.pairs == [("b", "a")]
        assert corpus.split("train").cases == []
        assert isinstance(corpus[0:1], Corpus) and corpus[0].id == "a"
        assert "Corpus(" in repr(corpus)
        assert corpus.map(lambda d: d.replace(text=d.text.upper())).texts == ["X", "Y"]
        with pytest.raises(ValueError, match="dupliqué"):
            Corpus([docs[0], docs[0]])
        with pytest.raises(KeyError):
            corpus.get("zzz")


class TestConfig:
    def test_task_section_is_required(self) -> None:
        with pytest.raises(ValueError, match="obligatoire"):
            ExperimentConfig(name="x", task="attribution")

    def test_unknown_key_and_extra_section_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("name: x\ntask: clustering\nunknown: 1\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalide"):
            load_config(path)
        path.write_text("- a\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="dictionnaire"):
            load_config(path)

    def test_seeds_and_fingerprint(self) -> None:
        assert ProtocolConfig(seeds=3).seed_list(10) == [10, 11, 12]
        assert ProtocolConfig(seeds=[5, 9]).seed_list(10) == [5, 9]
        with pytest.raises(ConfigurationError):
            ProtocolConfig(seeds=0).seed_list(1)
        data = {
            "name": "x",
            "task": "clustering",
            "clustering": {
                "corpora": {"c": {"path": "p"}},
                "representations": {"r": {"blocks": [{"type": "tfidf"}]}},
                "algorithms": {"k": {"type": "kmeans"}},
            },
        }
        config = ExperimentConfig.model_validate(data)
        assert config.fingerprint() == ExperimentConfig.model_validate(data).fingerprint()
        assert config.with_overrides(n_jobs=4).fingerprint() == config.fingerprint()
        assert config.run_dir == Path("results/x") and config.section is config.clustering
        with pytest.raises(ValueError, match="inattendue"):
            ExperimentConfig.model_validate(
                {
                    **data,
                    "attribution": {
                        "corpus": "p",
                        "representations": {"r": {"blocks": [{"type": "bow"}]}},
                        "classifiers": {"c": {"type": "naive_bayes"}},
                    },
                }
            )


class TestVersions:
    def test_environment_information(self) -> None:
        versions = package_versions(["numpy", "surely-not-installed-package"])
        assert versions["numpy"] and versions["surely-not-installed-package"] is None
        info = environment_info({"extra": 1})
        assert info["extra"] == 1 and "python" in info and "git" in info
        assert set(git_revision()) == {"commit", "dirty"}
        assert code_version().startswith("0.1.0")

    def test_missing_dependency_message(self) -> None:
        error = MissingDependencyError("gensim", "Doc2Vec", "uv sync")
        assert "gensim" in str(error) and error.feature == "Doc2Vec"
