"""Tests des chargeurs : C50 (répertoire, zip, zip imbriqué), Gutenberg (client simulé), PAN (XML, paires, export) et AIDA CoNLL-YAGO."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from plagia.core.errors import ExternalServiceError, PlagiaError
from plagia.core.types import Passage, PassagePair
from plagia.corpus.loaders.aida import load_aida, read_aida_tsv, read_conll2003
from plagia.corpus.loaders.c50 import download_c50, load_c50
from plagia.corpus.loaders.gutenberg import (
    GutenbergAuthor,
    GutenbergClient,
    GutenbergConfig,
    GutenbergWork,
    body_paragraphs,
    build_gutenberg_corpus,
    make_excerpts,
    mirror_path,
    parse_header,
    titles_match,
)
from plagia.corpus.loaders.pan import combine_pan, load_pan, parse_pan_xml, write_pan_detections
from plagia.corpus.splits import check_no_leakage

C50_FILES = {
    "C50train/AaronPressman/1.txt": "Aaron Pressman reports on the Senate bill. Lawmakers voted.",
    "C50train/AaronPressman/2.txt": "The FCC ruled today on telecom rules.",
    "C50train/AlanCrosby/1.txt": "Prague stocks rose sharply on Monday.",
    "C50test/AaronPressman/3.txt": "Congress debated encryption exports.",
    "C50test/AlanCrosby/2.txt": "The Czech crown weakened slightly.",
}


def _write_c50_dir(root: Path) -> None:
    for name, text in C50_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class TestC50:
    def test_directory(self, tmp_path: Path) -> None:
        _write_c50_dir(tmp_path)
        corpus = load_c50(tmp_path)
        assert len(corpus) == 5 and corpus.authors == ["AaronPressman", "AlanCrosby"]
        assert corpus.split_counts() == {"train": 3, "test": 2}
        first = corpus.get("c50/train/AaronPressman/1")
        assert "Pressman" not in first.text and first.licence == "CC BY 4.0"

    def test_zip_and_nested_zip(self, tmp_path: Path) -> None:
        inner = _zip_bytes({k: v.encode() for k, v in C50_FILES.items()})
        (tmp_path / "flat.zip").write_bytes(inner)
        (tmp_path / "nested.zip").write_bytes(_zip_bytes({"C50.zip": inner}))
        assert len(load_c50(tmp_path / "flat.zip")) == 5
        assert len(load_c50(tmp_path / "nested.zip")) == 5

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(PlagiaError, match="introuvable"):
            load_c50(tmp_path / "missing")
        (tmp_path / "empty").mkdir()
        with pytest.raises(PlagiaError, match="Aucun fichier"):
            load_c50(tmp_path / "empty")
        existing = tmp_path / "reuter50_50.zip"
        existing.write_bytes(b"x")
        assert download_c50(tmp_path) == existing
        with pytest.raises(ExternalServiceError):
            download_c50(tmp_path / "other.zip", url="http://127.0.0.1:9/none.zip", timeout=2)


def _book(title: str, author: str, paragraphs: int = 60) -> bytes:
    body = "\n\n".join(
        f"Paragraph {i} of {title}. " + " ".join(f"word{j} sentence goes on." for j in range(40))
        for i in range(paragraphs)
    )
    text = (
        f"Title: {title}\nAuthor: {author}\nLanguage: English\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\nCHAPTER I\n\n{body}\n"
        f"*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\nLicence text"
    )
    return text.encode("utf-8")


class TestGutenberg:
    def test_helpers(self) -> None:
        assert mirror_path(1342) == "1/3/4" and mirror_path(5) == "0"
        assert titles_match("The Time Machine", "Time Machine: An Invention")
        assert not titles_match("Emma", "Persuasion")
        assert parse_header("Title: Emma\nAuthor: Jane Austen\n")["author"] == "Jane Austen"
        paragraphs = body_paragraphs("CHAPTER I\n\nShort.\n\nA real paragraph with enough words here.")
        assert paragraphs == ["A real paragraph with enough words here."]

    def test_excerpts_respect_bounds(self) -> None:
        paragraphs = [" ".join(["Word is here."] * 70) for _ in range(30)]
        excerpts = make_excerpts(paragraphs, min_words=500, max_words=1000)
        assert excerpts and all(500 <= len(e.split()) <= 1000 for e in excerpts)

    def test_build_with_fake_mirror_and_catalog(self, tmp_path: Path) -> None:
        books = {
            1: ("First Book", "Ann Writer"),
            2: ("Second Book", "Ann Writer"),
            3: ("Other Tale", "Bob Author"),
            4: ("Last Tale", "Bob Author"),
        }
        catalog = 'Text#,Type,Issued,Title,Language,Authors\n4,Text,2000,Last Tale,en,"Author, Bob"\n'
        requested: list[str] = []

        def fetch(url: str) -> bytes:
            requested.append(url)
            if url.endswith("pg_catalog.csv"):
                return catalog.encode()
            book_id = int(url.rsplit("/", 1)[-1].split("-")[0].split(".")[0])
            if url.endswith("-0.txt") and book_id != 2:
                raise OSError("absent")
            return _book(*books[book_id])

        config = GutenbergConfig(
            authors=[
                GutenbergAuthor(
                    name="Ann Writer",
                    works=[GutenbergWork(title="First Book", id=1), GutenbergWork(title="Second Book", id=2)],
                ),
                GutenbergAuthor(
                    name="Bob Author", works=[GutenbergWork(title="Other Tale", id=3), GutenbergWork(title="Last Tale")]
                ),
            ],
            delay_seconds=0,
            excerpts_per_work=3,
            min_words=200,
            max_words=400,
        )
        client = GutenbergClient(tmp_path / "raw", config, fetcher=fetch)
        corpus = build_gutenberg_corpus(config, tmp_path / "raw", client)
        assert set(corpus.authors) == {"ann_writer", "bob_author"}
        assert all(1 <= len(d.text.split()) <= 400 for d in corpus)
        assert {d.split for d in corpus if d.author == "ann_writer"} == {"train", "test"}
        assert all("PROJECT GUTENBERG" not in d.text for d in corpus)
        assert any(u.endswith("pg_catalog.csv") for u in requested)
        assert (tmp_path / "raw" / "1.txt").exists()
        before = len(requested)
        client.fetch(1)
        assert len(requested) == before  # lu dans le cache local

    def test_fetch_failure(self, tmp_path: Path) -> None:
        config = GutenbergConfig(
            authors=[
                GutenbergAuthor(name="A B", works=[GutenbergWork(title="X", id=1), GutenbergWork(title="Y", id=2)])
            ]
            * 2,
            delay_seconds=0,
        )

        def fail(url: str) -> bytes:
            raise OSError("down")

        with pytest.raises(ExternalServiceError):
            GutenbergClient(tmp_path, config, fetcher=fail).fetch(9)
        with pytest.raises(ExternalServiceError):
            GutenbergClient(tmp_path, config, fetcher=fail).catalog()


def _pan_tree(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "susp").mkdir()
    (root / "02-no-obfuscation").mkdir()
    (root / "01-no-plagiarism").mkdir()
    (root / "src" / "source-document00001.txt").write_bytes("Source text with é\r\nand more words here.".encode())
    (root / "susp" / "suspicious-document00001.txt").write_text("Intro. Source text with é and more.", encoding="utf-8")
    (root / "susp" / "suspicious-document00002.txt").write_text("Clean text.", encoding="utf-8")
    (root / "02-no-obfuscation" / "suspicious-document00001-source-document00001.xml").write_text(
        '<document reference="suspicious-document00001.txt">'
        '<feature name="plagiarism" obfuscation="none" this_offset="7" this_length="18" '
        'source_reference="source-document00001.txt" source_offset="0" source_length="18" type="artificial"/>'
        '<feature name="about" />'
        "</document>",
        encoding="utf-8",
    )
    (root / "01-no-plagiarism" / "suspicious-document00002-source-document00001.xml").write_text(
        '<document reference="suspicious-document00002.txt"></document>', encoding="utf-8"
    )
    (root / "pairs").write_text(
        "suspicious-document00001.txt source-document00001.txt\nsuspicious-document00002.txt source-document00001.txt\n",
        encoding="utf-8",
    )
    (root / "02-no-obfuscation" / "broken.xml").write_text("<document", encoding="utf-8")


class TestPan:
    def test_load(self, tmp_path: Path) -> None:
        _pan_tree(tmp_path)
        corpus = load_pan(tmp_path, split="train")
        assert len(corpus) == 3 and len(corpus.cases) == 1 and len(corpus.pairs) == 2
        case = corpus.cases[0]
        assert case.obfuscation == "none" and case.source == Passage("source-document00001", 0, 18)
        source = corpus.get("source-document00001")
        assert "\r\n" in source.text and source.meta["role"] == "source"  # aucune normalisation : décalages valides
        susp = corpus.get("suspicious-document00001")
        assert susp.text[case.suspicious.start : case.suspicious.end] == "Source text with é"
        with pytest.raises(PlagiaError):
            load_pan(tmp_path / "nothing")

    def test_combine_train_and_test_without_leakage(self, tmp_path: Path) -> None:
        _pan_tree(tmp_path / "train")
        _pan_tree(tmp_path / "test")
        (tmp_path / "test" / "susp" / "suspicious-document00002.txt").write_text("Other text.", encoding="utf-8")
        train, test = load_pan(tmp_path / "train", split="train"), load_pan(tmp_path / "test", split="test")
        corpus = combine_pan(train, test, name="pan")
        # textes communs (source et suspect 1) retirés de l'entraînement ; suspect 2 renommé
        assert corpus.split("train").ids == ["suspicious-document00002"]
        assert corpus.meta["dropped_from_train"] == 2 and corpus.meta["test_prefix"] == "test/"
        assert set(corpus.split("test").ids) == {
            "test/source-document00001",
            "test/suspicious-document00001",
            "test/suspicious-document00002",
        }
        assert [c.suspicious.document_id for c in corpus.cases] == ["test/suspicious-document00001"]
        assert corpus.cases[0].source is not None and corpus.cases[0].source.document_id == "test/source-document00001"
        assert ("test/suspicious-document00002", "test/source-document00001") in corpus.pairs
        assert ("suspicious-document00002", "source-document00001") not in corpus.pairs
        check_no_leakage(corpus)
        assert write_pan_detections(corpus.cases, tmp_path / "out") == 1
        assert (tmp_path / "out" / "suspicious-document00001-source-document00001.xml").exists()
        separate = combine_pan(load_pan(tmp_path / "train", split="train"), load_pan(tmp_path / "train", split="test"))
        assert separate.meta["dropped_from_train"] == 3 and not separate.split("train").ids

    def test_write_detections_roundtrip(self, tmp_path: Path) -> None:
        detection = PassagePair(Passage("suspicious-document00001", 7, 18), Passage("source-document00001", 0, 18))
        assert write_pan_detections([detection, PassagePair(Passage("x", 0, 1))], tmp_path / "out") == 1
        parsed = parse_pan_xml(
            tmp_path / "out" / "suspicious-document00001-source-document00001.xml", "detected-plagiarism"
        )
        assert parsed[0].suspicious == detection.suspicious and parsed[0].source == detection.source


AIDA_TSV = """-DOCSTART- (1 EU)
EU\tB\tEU\t--NME--
rejects
German\tB\tGerman\tGermany\thttp://en.wikipedia.org/wiki/Germany\t11867\t/m/0345h
call
.

Peter\tB\tPeter Blackburn\tPeter_Blackburn\thttp://en.wikipedia.org/wiki/Peter_Blackburn\t1\t/m/x
Blackburn\tI\tPeter Blackburn\tPeter_Blackburn\thttp://en.wikipedia.org/wiki/Peter_Blackburn\t1\t/m/x

-DOCSTART- (947testa CRICKET)
Leicestershire\tB\tLeicestershire\tLeicestershire_County_Cricket_Club\thttp://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club\t2\t/m/y
won
"""

CONLL = """-DOCSTART- -X- -X- O

EU NNP B-NP I-ORG
rejects VBZ B-VP O
German JJ B-NP I-MISC
call NN I-NP O
. . O O

Peter NNP B-NP I-PER
Blackburn NNP I-NP I-PER

-DOCSTART- -X- -X- O

Leicestershire NNP B-NP B-ORG
won VBD B-VP O
"""


class TestAida:
    def test_reader_and_merge(self, tmp_path: Path) -> None:
        (tmp_path / "aida.tsv").write_text(AIDA_TSV, encoding="utf-8")
        (tmp_path / "conll.txt").write_text(CONLL, encoding="utf-8")
        documents = read_aida_tsv(tmp_path / "aida.tsv")
        assert [d.split for d in documents] == ["train", "validation"]
        assert [(m.text, m.entity) for m in documents[0].mentions] == [
            ("EU", None),
            ("German", "Germany"),
            ("Peter Blackburn", "Peter_Blackburn"),
        ]
        conll = read_conll2003(tmp_path / "conll.txt")
        assert [(m.text, m.label) for m in conll[0].mentions] == [
            ("EU", "ORG"),
            ("German", "MISC"),
            ("Peter Blackburn", "PER"),
        ]
        corpus = load_aida(tmp_path / "aida.tsv", tmp_path / "conll.txt")
        doc = corpus[0]
        gold = doc.meta["gold_entities"]
        assert [g["label"] for g in gold] == ["ORG", "MISC", "PER"]
        assert all(doc.text[g["start"] : g["end"]] == g["text"] for g in gold)
        assert "\n" in doc.text  # une phrase par ligne

    def test_merge_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "aida.tsv").write_text(AIDA_TSV, encoding="utf-8")
        (tmp_path / "conll.txt").write_text(CONLL.split("-DOCSTART- -X- -X- O\n\nLeicestershire")[0], encoding="utf-8")
        corpus = load_aida(tmp_path / "aida.tsv", tmp_path / "conll.txt")
        assert all(g["label"] is None for g in corpus[0].meta["gold_entities"])
