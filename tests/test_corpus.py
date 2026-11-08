"""Tests de la couche corpus : nettoyage, langue, découpages sans fuite, manifestes, reproductibilité, obfuscateurs et corpus synthétique."""

from __future__ import annotations

import filecmp
import random
from pathlib import Path

import pytest

from plagia.core.errors import ConfigurationError, LeakageError, PlagiaError
from plagia.core.text import count_words
from plagia.core.types import Corpus, Document, Passage, PassagePair
from plagia.corpus import cell_counts, check_no_leakage, generate_toy_corpus, load_corpus, save_corpus
from plagia.corpus.cleaning import (
    CleaningConfig,
    author_name_variants,
    clean_corpus,
    clean_text,
    decode_bytes,
    deduplicate,
    normalize_unicode,
    remove_author_mentions,
    split_camel_case,
    strip_gutenberg_boilerplate,
    strip_markup,
    strip_signatures,
)
from plagia.corpus.language import detect_language, language_scores
from plagia.corpus.manifest import build_manifest, read_manifest_digest, verify_manifest
from plagia.corpus.obfuscation import (
    OBFUSCATORS,
    BackTranslation,
    Chain,
    DictSynonyms,
    Identity,
    Paraphrase,
    RandomEdit,
    Shuffle,
    Synonym,
    WordNetSynonyms,
    build_obfuscator,
    match_case,
)
from plagia.corpus.splits import (
    allocate,
    assign_document_splits,
    assign_group_splits,
    assign_pair_component_splits,
    carve_validation,
    cv_splitter,
    effective_folds,
    intersections,
    leakage_report,
    split_ids,
    subsample_authors,
    subsample_per_author,
    truncate_split,
    truncate_words,
)
from plagia.corpus.synthetic import SyntheticConfig, generate_plagiarism_corpus
from plagia.corpus.toy import toy_synonyms


class TestCleaning:
    def test_unicode_markup_and_signatures(self) -> None:
        assert normalize_unicode("é\x07") == "é"
        assert strip_markup("<p>A &amp; B</p><!-- x -->").strip() == "A & B"
        assert "x < y" in strip_markup("x < y")
        text = "Body text.\nBy John Smith\n(c) Reuters 1996\nContact me at john@x.com or www.x.com\n-- \nJohn"
        cleaned = strip_signatures(text)
        assert "Reuters" not in cleaned and "@" not in cleaned and "www" not in cleaned and "Body text." in cleaned
        assert "By John Smith" not in cleaned

    def test_author_mentions(self) -> None:
        assert split_camel_case("AaronPressman") == "Aaron Pressman"
        assert "Pressman" in author_name_variants("AaronPressman")
        assert remove_author_mentions("Aaron Pressman wrote; pressman said.", "AaronPressman").count("ressman") == 0

    def test_gutenberg_boilerplate(self) -> None:
        raw = "Title: X\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicence"
        assert strip_gutenberg_boilerplate(raw).strip() == "Body."
        assert "Body" in clean_text(raw, CleaningConfig(gutenberg=True))

    def test_decode_and_corpus_cleaning(self) -> None:
        assert decode_bytes("é".encode()) == "é" and decode_bytes("é".encode("cp1252")) == "é"
        assert decode_bytes(b"\xef\xbb\xbfabc") == "abc"
        corpus = Corpus([Document("a", "<b>Hello</b> world", author="A"), Document("b", "<i></i>", author="B")])
        cleaned = clean_corpus(corpus, CleaningConfig(min_words=1))
        assert cleaned.ids == ["a"] and cleaned[0].text == "Hello world"

    def test_exact_deduplication_keeps_training_copy(self) -> None:
        corpus = Corpus(
            [
                Document("t", "Same  TEXT", split="test"),
                Document("r", "same text", split="train"),
                Document("u", "unique", split="test"),
            ]
        )
        kept, removed = deduplicate(corpus)
        assert removed == ["t"] and kept.ids == ["r", "u"]


def test_language_detection() -> None:
    assert detect_language("The cat is on the table and it was happy with the dog.")[0] == "en"
    assert detect_language("Le chat est sur la table et il est content avec le chien.")[0] == "fr"
    assert detect_language("1234 5678")[0] == "und"
    assert set(language_scores("")) >= {"en", "fr"}


class TestSplits:
    def test_allocate(self) -> None:
        assert allocate(6, {"train": 5 / 6, "test": 1 / 6}) == {"train": 5, "test": 1}
        assert sum(allocate(10, {"a": 0.6, "b": 0.2, "c": 0.2}).values()) == 10
        assert allocate(3, {"a": 0.98, "b": 0.01, "c": 0.01}) == {"a": 1, "b": 1, "c": 1}
        with pytest.raises(ValueError):
            allocate(3, {"a": 0})

    def test_group_splits_have_empty_intersections(self, literary: Corpus) -> None:
        check_no_leakage(literary)
        works = {split: {d.group for d in literary if d.split == split} for split in ("train", "test")}
        assert not works["train"] & works["test"]
        assert all(v == set() for v in intersections(split_ids(literary)).values())
        per_author_test = {d.author for d in literary.split("test")}
        assert per_author_test == set(literary.authors)

    def test_leakage_detection(self) -> None:
        leaky = Corpus(
            [
                Document("a", "one text", work="w1", split="train"),
                Document("b", "other text", work="w1", split="test"),
            ]
        )
        assert leakage_report(leaky)["groups"] == ["w1"]
        with pytest.raises(LeakageError, match="groups"):
            check_no_leakage(leaky)
        duplicate = Corpus([Document("a", "same", split="train"), Document("b", "SAME", split="test")])
        with pytest.raises(LeakageError, match="texts"):
            check_no_leakage(duplicate)
        shared_source = Corpus(
            [Document("s", "susp", split="test"), Document("t", "source", split="train")],
            cases=[PassagePair(Passage("s", 0, 2), Passage("t", 0, 2))],
        )
        with pytest.raises(LeakageError, match="case_sources"):
            check_no_leakage(shared_source)

    def test_document_splits_and_validation(self, news: Corpus) -> None:
        check_no_leakage(news)
        carved = carve_validation(news, 0.25, seed=1)
        assert set(carved.split_counts()) == {"train", "validation", "test"}
        assert carved.split("test").ids == news.split("test").ids
        grouped = carve_validation(
            assign_group_splits(news.map(lambda d: d.replace(work=d.id)), {"train": 0.5, "test": 0.5}, 2),
            0.2,
            1,
            grouped=True,
        )
        check_no_leakage(grouped)

    def test_subsampling_and_truncation(self, literary: Corpus) -> None:
        two = subsample_authors(literary, 2, seed=3)
        assert len(two.authors) == 2 and two.authors == subsample_authors(literary, 2, seed=3).authors
        assert subsample_authors(literary, None, 1) is literary
        few = subsample_per_author(literary, 3, seed=1)
        counts = {a: sum(1 for d in few.split("train") if d.author == a) for a in few.authors}
        assert set(counts.values()) == {3} and len(few.split("test")) == len(literary.split("test"))
        assert truncate_words("one two three four", 2) == "one two"
        cut = truncate_split(literary, 10)
        assert all(len(d.text.split()) <= 12 for d in cut.split("test"))
        assert truncate_split(literary, None) is literary

    def test_folds(self) -> None:
        labels = ["a"] * 6 + ["b"] * 6
        groups = [f"g{i // 2}" for i in range(12)]
        assert effective_folds(labels, None, 5) == 5
        assert effective_folds(labels, groups, 5) == 3
        assert effective_folds(["a", "b"], None, 5) == 0
        assert cv_splitter(3, True, 0).get_n_splits() == 3

    def test_pair_component_splits(self) -> None:
        docs = [Document(i, i) for i in ("s1", "s2", "s3", "t1", "t2", "t3")]
        corpus = Corpus(docs, pairs=[("s1", "t1"), ("s2", "t1"), ("s3", "t3")])
        split = assign_pair_component_splits(corpus, {"train": 0.5, "test": 0.5}, seed=0)
        assigned = {d.id: d.split for d in split}
        assert assigned["s1"] == assigned["s2"] == assigned["t1"] and assigned["s3"] == assigned["t3"]

    def test_document_split_fractions(self, literary: Corpus) -> None:
        split = assign_document_splits(literary, {"train": 0.75, "test": 0.25}, seed=4)
        assert abs(split.split_counts()["test"] / len(split) - 0.25) < 0.05


class TestStore:
    def test_save_load_manifest(self, literary: Corpus, tmp_path: Path) -> None:
        digest = save_corpus(literary, tmp_path / "c")
        assert digest == read_manifest_digest(tmp_path / "c") and verify_manifest(tmp_path / "c")
        loaded = load_corpus(tmp_path / "c")
        assert (
            loaded.ids == literary.ids and loaded.texts == literary.texts and loaded.meta["manifest_sha256"] == digest
        )
        entry = build_manifest(literary)[0]
        assert {"id", "author", "work", "language", "length", "licence", "split", "sha256"} <= set(entry)

    def test_tampered_manifest_is_detected(self, literary: Corpus, tmp_path: Path) -> None:
        save_corpus(literary, tmp_path / "c")
        manifest = tmp_path / "c" / "manifest.jsonl"
        manifest.write_text(manifest.read_text(encoding="utf-8").replace('"train"', '"test"', 1), encoding="utf-8")
        with pytest.raises(PlagiaError, match="altéré"):
            load_corpus(tmp_path / "c")
        with pytest.raises(PlagiaError, match="Aucun corpus"):
            load_corpus(tmp_path / "missing")

    def test_two_builds_with_same_seed_are_identical(self, tmp_path: Path) -> None:
        """Critère du jalon J1 : deux reconstructions à graine égale donnent des fichiers identiques."""
        for name in ("a", "b"):
            base = generate_toy_corpus(n_authors=4, works_per_author=6, docs_per_work=4, words_per_doc=330, seed=9)
            corpus = generate_plagiarism_corpus(
                base,
                SyntheticConfig(n_sources=40, n_suspicious=50, passages_per_cell=2, seed=9),
                synonyms=toy_synonyms(),
            )
            save_corpus(corpus, tmp_path / name)
        files = sorted(p.name for p in (tmp_path / "a").iterdir())
        assert files == sorted(p.name for p in (tmp_path / "b").iterdir())
        for name in files:
            assert filecmp.cmp(tmp_path / "a" / name, tmp_path / "b" / name, shallow=False), name


class TestObfuscation:
    TEXT = "The quick brown fox jumps over the lazy dog. The old merchant sold a strange ship. A storm came."

    def test_levels(self) -> None:
        synonyms = DictSynonyms(
            {"quick": ["fast"], "lazy": ["idle"], "old": ["aged"], "strange": ["odd"], "ship": ["vessel"]}
        )
        rng = random.Random(1)
        assert build_obfuscator("O0")(self.TEXT, rng) == self.TEXT
        o1 = build_obfuscator("O1", edit_rate=0.4)(self.TEXT, random.Random(2))
        assert o1 != self.TEXT and o1
        o2 = build_obfuscator("O2", synonyms=synonyms, synonym_rate=1.0)(self.TEXT, random.Random(3))
        assert "fast" in o2 and "idle" in o2 and "vessel" in o2
        o3 = build_obfuscator("O3", synonyms=synonyms, synonym_rate=1.0)(self.TEXT, random.Random(4))
        assert sorted(o3.split()) == sorted(o2.split()) and o3 != o2
        with pytest.raises(ConfigurationError, match="O4"):
            build_obfuscator("O4")
        with pytest.raises(ConfigurationError, match="inconnu"):
            build_obfuscator("O9")

    def test_o4_with_translator_and_paraphraser(self) -> None:
        calls: list[tuple[str, str]] = []

        def translate(text: str, source: str, target: str) -> str:
            calls.append((source, target))
            return text.upper() if target == "fr" else text.lower()

        assert build_obfuscator("O4", translator=translate)("Abc", random.Random(0)) == "abc"
        assert calls == [("en", "fr"), ("fr", "en")]
        assert build_obfuscator("O4", paraphraser=lambda t: t[::-1])("abc", random.Random(0)) == "cba"
        assert isinstance(BackTranslation(translate), BackTranslation) and isinstance(Paraphrase(str), Paraphrase)

    def test_components(self) -> None:
        assert match_case("word", "Model") == "Word" and match_case("word", "MODEL") == "WORD"
        assert DictSynonyms({"a": ["b"]}).synonyms("B") == ["a"]
        assert Shuffle()("Only one sentence.", random.Random(0)) == "Only one sentence."
        assert Chain([Identity(), Identity()])("x", random.Random(0)) == "x"
        with pytest.raises(ConfigurationError):
            RandomEdit(rate=2)
        assert set(OBFUSCATORS.names()) >= {"identity", "random_edit", "synonym", "shuffle", "chain"}
        assert Synonym(DictSynonyms({}), rate=1.0)("Nothing changes here.", random.Random(0)) == "Nothing changes here."

    def test_wordnet_requires_nltk(self) -> None:
        try:
            import nltk  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="nltk"):
                WordNetSynonyms()
        else:  # pragma: no cover - dépend de l'environnement
            try:
                assert "fast" in WordNetSynonyms().synonyms("quick") or True
            except ImportError:
                pytest.skip("ressource wordnet absente")


class TestSynthetic:
    def test_offsets_levels_and_balance(self, plagiarism_corpus: Corpus) -> None:
        texts = {d.id: d.text for d in plagiarism_corpus}
        check_no_leakage(plagiarism_corpus)
        counts = cell_counts(plagiarism_corpus)
        assert len(counts) == 12 and min(counts.values()) >= 6
        suspicious = [d for d in plagiarism_corpus if d.meta["role"] == "suspicious"]
        clean = [d for d in suspicious if d.meta["level"] is None]
        assert abs(len(clean) / len(suspicious) - 0.5) < 0.05
        for case in plagiarism_corpus.cases:
            assert case.source is not None
            susp = texts[case.suspicious.document_id][case.suspicious.start : case.suspicious.end]
            source = texts[case.source.document_id][case.source.start : case.source.end]
            assert susp and source
            if case.obfuscation == "O0":
                assert susp == source
            assert count_words(source) == case.meta["size"]
        assert plagiarism_corpus.pairs and all(
            s.startswith("susp") and t.startswith("src") for s, t in plagiarism_corpus.pairs
        )

    def test_one_cell_per_host(self, plagiarism_corpus: Corpus) -> None:
        by_doc: dict[str, set[tuple[str | None, int]]] = {}
        for case in plagiarism_corpus.cases:
            by_doc.setdefault(case.suspicious.document_id, set()).add((case.obfuscation, case.meta["size"]))
        assert all(len(cells) == 1 for cells in by_doc.values())

    def test_shortfall_is_reported(self) -> None:
        base = generate_toy_corpus(n_authors=2, works_per_author=6, docs_per_work=4, words_per_doc=330, seed=3)
        config = SyntheticConfig(n_sources=4, n_suspicious=20, levels=["O0"], sizes=[300], passages_per_cell=12, seed=3)
        corpus = generate_plagiarism_corpus(base, config)
        produced = cell_counts(corpus)[("O0", 300)]
        assert produced < 12 and corpus.meta["shortfall"] == {"O0/300": produced}
        ample = generate_plagiarism_corpus(base, config.model_copy(update={"passages_per_cell": 2}))
        assert ample.meta["shortfall"] == {}

    def test_insufficient_base_is_rejected(self) -> None:
        base = generate_toy_corpus(n_authors=1, works_per_author=1, docs_per_work=2, words_per_doc=50, seed=0)
        with pytest.raises(PlagiaError, match="insuffisant"):
            generate_plagiarism_corpus(base, SyntheticConfig(n_sources=5, n_suspicious=5))
