"""Tests de la couche annotation : documents annotés, schéma commun, annotateur
heuristique, cache (proxy, hors ligne), Spotlight, bases de connaissances, Stanford,
spaCy (objets simulés), prétraitement et qualité des annotateurs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from plagia.annotation import (
    AnnotatedDocument,
    AnnotationCache,
    CachedAnnotator,
    CacheMissError,
    DbpediaKnowledgeBase,
    Entity,
    HeuristicAnnotator,
    Preprocessor,
    RegexAnnotator,
    SpacyPipeline,
    TypeHierarchy,
    YagoKnowledgeBase,
    annotate_documents,
    annotation_versions,
    build_annotator,
    normalize_label,
    stopword_list,
)
from plagia.annotation.document import as_annotated, token_sentences
from plagia.annotation.quality import linking_quality, ner_quality, prf, resource_id
from plagia.annotation.schema import kind_from_types, ptb_to_upos
from plagia.annotation.spacy_engine import from_spacy_doc
from plagia.annotation.spotlight import SpotlightLinker, merge_links, normalize_type, parse_spotlight
from plagia.annotation.stanford import (
    SEPARATOR,
    StanfordNer,
    StanfordPipeline,
    StanfordPosTagger,
    align_tokens,
    group_entities,
    parse_tsv,
)
from plagia.core.config import AnnotationConfig
from plagia.core.errors import ConfigurationError, ExternalServiceError, MissingDependencyError, PlagiaError
from plagia.core.types import Document

TEXT = "Mr. John Smith met Acme Corp in London on 12 March 1996. Shares rose 5% to $12.5 million."


class TestDocument:
    def test_from_text_roundtrip_and_truncate(self) -> None:
        doc = AnnotatedDocument.from_text("d", "One two three. Four five six.")
        assert doc.tokens[:3] == ("One", "two", "three") and len(doc.sentences) == 2
        assert AnnotatedDocument.from_dict(doc.to_dict()) == doc
        short = doc.truncate_words(4)
        assert short.text == "One two three. Four" and short.sentences[-1][1] == len(short.tokens)
        assert doc.truncate_words(100) is doc
        assert as_annotated("text").text == "text" and as_annotated(doc) is doc
        assert token_sentences([0, 4, 8], [(0, 7), (8, 9)]) == ((0, 2), (2, 3))
        with pytest.raises(ValueError):
            AnnotatedDocument("x", "a", ("a",), ())

    def test_entity_types_at_depth(self) -> None:
        entity = Entity(
            "Zidane", 0, 6, "PERSON", "PER", "uri", ("0:owl:Thing", "1:dbo:Agent", "2:dbo:Person", "3:dbo:Athlete")
        )
        assert entity.types_at_depth(2) == ("dbo:Person",) and entity.types_at_depth(9) == ("dbo:Athlete",)
        assert Entity.from_dict(entity.to_dict()) == entity
        assert Entity("x", 0, 1, "MISC", "MISC").types_at_depth(1) == ()


class TestSchema:
    def test_mappings(self) -> None:
        assert normalize_label("GPE", "ontonotes") == "LOC" and normalize_label("MONEY", "ontonotes") == "NUM"
        assert normalize_label("B-PER", "conll") == "PER" and normalize_label("TIME", "stanford7") == "DATE"
        assert normalize_label("ORGANIZATION", "stanford3") == "ORG" and normalize_label("WEIRD", "ontonotes") == "MISC"
        assert (
            kind_from_types(["dbo:Person"]) == "PER"
            and kind_from_types(["2:dbo:Place"]) == "LOC"
            and kind_from_types([]) == "MISC"
        )
        assert ptb_to_upos("NNP") == "PROPN" and ptb_to_upos("??") == "X"
        with pytest.raises(KeyError):
            normalize_label("PER", "unknown")


class TestHeuristic:
    def test_entities_and_pos(self) -> None:
        doc = HeuristicAnnotator().annotate("d", TEXT)
        kinds = {(e.text, e.kind) for e in doc.entities}
        assert {("Mr. John Smith", "PER"), ("Acme Corp", "ORG"), ("London", "LOC"), ("12 March 1996", "DATE")} <= kinds
        assert ("5%", "NUM") in kinds and doc.pos is not None
        assert doc.pos[doc.tokens.index("in")] == "ADP" and doc.pos[doc.tokens.index(".")] == "PUNCT"
        assert doc.engine == "heuristic" and doc.lemmas is not None

    def test_regex_annotator_and_registry(self) -> None:
        doc = RegexAnnotator().annotate("d", "Hello world.")
        assert doc.pos is None and doc.entities == ()
        assert build_annotator(AnnotationConfig(engine="none")).name == "regex"
        assert HeuristicAnnotator().recognize("Mr. Brown went to Paris.")[0].kind == "PER"


class CountingAnnotator(HeuristicAnnotator):
    def __init__(self) -> None:
        self.calls = 0

    def process(self, texts: Any) -> list[AnnotatedDocument]:
        self.calls += len(texts)
        return super().process(texts)


class TestCache:
    def test_proxy_computes_once_then_runs_offline(self, tmp_path: Path) -> None:
        cache = AnnotationCache(tmp_path / "cache.sqlite")
        annotator = CountingAnnotator()
        items = [("a", TEXT), ("b", "Other text in Paris.")]
        first = CachedAnnotator(annotator, cache).annotate_many(items)
        again = CachedAnnotator(annotator, cache, offline=True).annotate_many(items)
        assert annotator.calls == 2 and first == again and again[0].id == "a"
        with pytest.raises(CacheMissError):
            CachedAnnotator(annotator, cache, offline=True).annotate_many([("c", "never seen")])
        assert cache.stats()["annotations"] == {"heuristic": 2}
        cache.put_response("k", "svc", "{}")
        assert cache.get_response("k") == "{}" and cache.get_response("zz") is None
        cache.close()

    def test_cache_is_picklable(self, tmp_path: Path) -> None:
        import pickle

        with AnnotationCache(tmp_path / "c.sqlite") as cache:
            cache.put_response("k", "s", "v")
            with pickle.loads(pickle.dumps(cache)) as clone:
                assert clone.get_response("k") == "v"

    def test_pipeline_with_cache(self, tmp_path: Path) -> None:
        config = AnnotationConfig(engine="heuristic", cache_path=tmp_path / "c.sqlite")
        documents = [Document("a", TEXT), Document("b", "Hello Paris.")]
        annotated = annotate_documents(documents, config)
        assert annotate_documents(documents, config, offline=True) == annotated
        assert annotate_documents(documents, AnnotationConfig(engine="none"))[0].entities == ()
        assert annotation_versions(config)["annotator"] == "heuristic"
        assert "annotator_error" in annotation_versions(AnnotationConfig(engine="stanford"))


SPOTLIGHT_RESPONSE = {
    "Resources": [
        {
            "@URI": "http://dbpedia.org/resource/London",
            "@support": "100",
            "@types": "Schema:Place,DBpedia:Place,DBpedia:City",
            "@surfaceForm": "London",
            "@offset": "31",
            "@similarityScore": "0.99",
        },
        {
            "@URI": "http://dbpedia.org/resource/Shares",
            "@support": "5",
            "@types": "",
            "@surfaceForm": "Shares",
            "@offset": "57",
            "@similarityScore": "0.4",
        },
    ]
}


class TestSpotlightAndKnowledgeBase:
    def test_parse_and_merge(self) -> None:
        assert (
            normalize_type("DBpedia:Person") == "dbo:Person"
            and normalize_type("Http://xmlns.com/foaf/0.1/Person") == "foaf:Person"
        )
        mentions = parse_spotlight(SPOTLIGHT_RESPONSE)
        assert mentions[0].types == ("schema:Place", "dbo:Place", "dbo:City") and mentions[0].end == 37
        assert parse_spotlight({"Resources": SPOTLIGHT_RESPONSE["Resources"][0]})[0].uri.endswith("London")
        doc = HeuristicAnnotator().annotate("d", TEXT)
        merged = merge_links(doc, mentions)
        london = next(e for e in merged.entities if e.text == "London")
        assert london.uri.endswith("London") and "raw:dbo:City" in london.types
        assert any(e.label == "SPOTLIGHT" for e in merged.entities)

    def test_linker_uses_cache_and_offline(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def post(url: str, data: bytes, headers: Any) -> bytes:
            calls.append(url)
            return json.dumps(SPOTLIGHT_RESPONSE).encode()

        cache = AnnotationCache(tmp_path / "c.sqlite")
        linker = SpotlightLinker("http://spot/rest/", cache=cache, http_post=post)
        assert len(linker.link(TEXT)) == 2 and len(linker.link(TEXT)) == 2
        assert calls == ["http://spot/rest/annotate"]
        offline = SpotlightLinker("http://spot/rest", cache=cache, offline=True)
        assert offline.link(TEXT) and linker.link_document(HeuristicAnnotator().annotate("d", TEXT)).entities
        with pytest.raises(CacheMissError):
            offline.link("unseen")

        def down(url: str, data: bytes, headers: Any) -> bytes:
            raise OSError("refused")

        with pytest.raises(ExternalServiceError):
            SpotlightLinker(http_post=down).link("x")
        cache.close()

    def test_hierarchy_readers(self, tmp_path: Path) -> None:
        nt = tmp_path / "onto.nt"
        sub = "<http://www.w3.org/2000/01/rdf-schema#subClassOf>"
        nt.write_text(
            f"<http://dbpedia.org/ontology/Person> {sub} <http://dbpedia.org/ontology/Agent> .\n"
            f"<http://dbpedia.org/ontology/Agent> {sub} <http://www.w3.org/2002/07/owl#Thing> .\n"
            f"<http://dbpedia.org/ontology/City> {sub} <http://dbpedia.org/ontology/Place> .\n"
            "<http://dbpedia.org/ontology/City> <http://www.w3.org/2000/01/rdf-schema#label> <x> .\n",
            encoding="utf-8",
        )
        hierarchy = TypeHierarchy.load(nt)
        assert hierarchy.path_to_root("dbo:Person") == ["owl:Thing", "dbo:Agent", "dbo:Person"]
        assert hierarchy.depth("dbo:City") == 2 and hierarchy.at_depth("dbo:Person", 1) == "dbo:Agent"
        assert hierarchy.ancestors("dbo:Person") == ["dbo:Agent", "owl:Thing"] and "dbo:Place" in hierarchy
        tsv = tmp_path / "tax.tsv"
        tsv.write_text("# commentaire\nschema:Person\tschema:Thing\n", encoding="utf-8")
        assert TypeHierarchy.load(tsv, root="schema:Thing").depth("schema:Person") == 1
        try:
            import rdflib  # noqa: F401
        except ImportError:
            with pytest.raises(MissingDependencyError):
                TypeHierarchy.load(tmp_path / "onto.ttl")

    def test_dbpedia_and_yago_typing(self, tmp_path: Path) -> None:
        hierarchy = TypeHierarchy.from_edges(
            [("dbo:City", "dbo:Place"), ("dbo:Place", "owl:Thing"), ("schema:Person", "owl:Thing")]
        )
        doc = merge_links(HeuristicAnnotator().annotate("d", TEXT), parse_spotlight(SPOTLIGHT_RESPONSE))
        typed = DbpediaKnowledgeBase(hierarchy).type_document(doc)
        london = next(e for e in typed.entities if e.text == "London")
        assert london.types_at_depth(1) == ("dbo:Place",) and london.types_at_depth(2) == ("dbo:City",)
        answer = {"results": {"bindings": [{"t": {"value": "http://schema.org/Person"}}]}}
        requests: list[str] = []

        def get(url: str) -> bytes:
            requests.append(url)
            return json.dumps(answer).encode()

        cache = AnnotationCache(tmp_path / "c.sqlite")
        yago = YagoKnowledgeBase(hierarchy, cache=cache, http_get=get)
        entity = Entity("Bowie", 0, 5, "PERSON", "PER", uri="http://dbpedia.org/resource/David_Bowie")
        assert yago.types_of(entity) == ["schema:Person"] and yago.types_of(entity) == ["schema:Person"]
        assert len(requests) == 1 and YagoKnowledgeBase.yago_iri(entity.uri or "").endswith("/David_Bowie")
        assert yago.type_entity(entity).types == ("0:owl:Thing", "1:schema:Person")
        assert yago.types_of(Entity("x", 0, 1, "MISC", "MISC")) == []
        with pytest.raises(CacheMissError):
            YagoKnowledgeBase(hierarchy, cache=cache, offline=True).types_of(
                Entity("y", 0, 1, "P", "PER", uri="http://dbpedia.org/resource/Other")
            )
        cache.close()


class TestStanford:
    def test_parsing_alignment_and_grouping(self) -> None:
        output = f"John\tPERSON\nSmith\tPERSON\n-LRB-\tO\n\n{SEPARATOR}\tO\n\nParis\tLOCATION\n"
        documents = parse_tsv(output)
        assert len(documents) == 2 and documents[1] == [[("Paris", "LOCATION")]]
        offsets = align_tokens("John Smith (x)", ["John", "Smith", "-LRB-", "x", "-RRB-"])
        assert offsets == [(0, 4), (5, 10), (11, 12), (12, 13), (13, 14)]
        assert align_tokens("“quoted”", ["``", "quoted", "''"]) == [(0, 1), (1, 7), (7, 8)]
        entities = group_entities(
            "John Smith in Paris", [(0, 4), (5, 10), (11, 13), (14, 19)], ["B-PER", "I-PER", "O", "B-LOC"], "conll"
        )
        assert [(e.text, e.kind) for e in entities] == [("John Smith", "PER"), ("Paris", "LOC")]

    def test_subprocess_adapters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        texts = ["John Smith visited Paris.", "IBM hired Mary."]

        def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
            if "CRFClassifier" in " ".join(command):
                out = f"John\tPERSON\nSmith\tPERSON\nvisited\tO\nParis\tLOCATION\n.\tO\n\n{SEPARATOR}\tO\n\nIBM\tORGANIZATION\nhired\tO\nMary\tPERSON\n.\tO\n"
            else:
                out = f"John\tNNP\nSmith\tNNP\nvisited\tVBD\nParis\tNNP\n.\t.\n\n{SEPARATOR}\tNNP\n\nIBM\tNNP\nhired\tVBD\nMary\tNNP\n.\t.\n"
            return SimpleNamespace(stdout=out, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ner = StanfordNer("stanford-ner-4.2.0.jar", "english.all.3class.distsim.crf.ser.gz")
        pos = StanfordPosTagger("stanford-postagger.jar", "english-left3words-distsim.tagger")
        pipeline = StanfordPipeline(ner, pos)
        documents = pipeline.annotate_many([("a", texts[0]), ("b", texts[1])])
        assert [(e.text, e.kind) for e in documents[0].entities] == [("John Smith", "PER"), ("Paris", "LOC")]
        assert documents[1].pos == ("PROPN", "VERB", "PROPN", "PUNCT") and documents[1].tags[1] == "VBD"
        assert ner.scheme == "stanford3" and "stanford-ner" in pipeline.version and pipeline.params()["ner"]

    def test_subprocess_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(command: list[str], **kwargs: Any) -> None:
            raise FileNotFoundError("java")

        monkeypatch.setattr(subprocess, "run", missing)
        with pytest.raises(MissingDependencyError, match="java"):
            StanfordNer("x.jar", "m.crf.ser.gz").process(["a"])

        def failing(command: list[str], **kwargs: Any) -> None:
            raise subprocess.CalledProcessError(1, command, stderr="boom")

        monkeypatch.setattr(subprocess, "run", failing)
        with pytest.raises(PlagiaError, match="boom"):
            StanfordNer("x.jar", "m.crf.ser.gz").process(["a"])

        monkeypatch.setattr(subprocess, "run", lambda c, **k: SimpleNamespace(stdout="a\tO\n", stderr=""))
        with pytest.raises(PlagiaError, match="documents en sortie"):
            StanfordNer("x.jar", "m.crf.ser.gz").process(["a", "b"])
        with pytest.raises(ConfigurationError):
            build_annotator(AnnotationConfig(engine="stanford"))


class FakeToken(SimpleNamespace):
    pass


def _fake_spacy_doc() -> SimpleNamespace:
    words = [
        ("Paris", "PROPN", "NNP", "Paris", 0),
        (" ", "SPACE", "_SP", " ", 5),
        ("is", "AUX", "VBZ", "be", 6),
        ("big", "ADJ", "JJ", "big", 9),
        (".", "PUNCT", ".", ".", 12),
    ]
    tokens = [
        FakeToken(text=w, pos_=p, tag_=t, lemma_=lem, idx=i, i=k, is_space=w.isspace())
        for k, (w, p, t, lem, i) in enumerate(words)
    ]
    entity = SimpleNamespace(text="Paris", start_char=0, end_char=5, label_="GPE", kb_id_="")
    return SimpleNamespace(
        text="Paris  is big.",
        has_annotation=lambda name: True,
        sents=[tokens],
        ents=[entity],
        __iter__=None,
        tokens=tokens,
    )


def test_from_spacy_doc_conversion() -> None:
    fake = _fake_spacy_doc()

    class Doc:
        def __init__(self, inner: SimpleNamespace) -> None:
            self.__dict__.update(inner.__dict__)

        def __iter__(self) -> Any:
            return iter(self.tokens)

    doc = from_spacy_doc("d", Doc(fake))
    assert doc.tokens == ("Paris", "is", "big", ".") and doc.pos == ("PROPN", "AUX", "ADJ", "PUNCT")
    assert doc.sentences == ((0, 4),) and doc.entities[0].kind == "LOC" and doc.lemmas[1] == "be"


def test_spacy_pipeline_reports_missing_dependency() -> None:
    pipeline = SpacyPipeline("en_core_web_sm")
    assert pipeline.scheme == "ontonotes" and SpacyPipeline("fr_core_news_sm").scheme == "wikiner"
    assert pipeline.params()["model"] == "en_core_web_sm" and "_nlp" in pipeline.__getstate__()
    try:
        import spacy  # noqa: F401
    except ImportError:
        with pytest.raises(MissingDependencyError):
            _ = pipeline.version


@pytest.mark.spacy
def test_spacy_end_to_end() -> None:  # pragma: no cover - nécessite spaCy et un modèle
    pytest.importorskip("spacy")
    pipeline = SpacyPipeline("en_core_web_sm")
    try:
        doc = pipeline.annotate("d", TEXT)
    except MissingDependencyError:
        pytest.skip("modèle spaCy absent")
    assert doc.pos is not None and any(e.kind == "PER" for e in doc.entities)
    raw = pipeline.process([TEXT])[0]
    assert pipeline.convert("d", TEXT, pipeline.deserialize(pipeline.serialize(raw))).tokens == doc.tokens


class TestPreprocessor:
    def test_steps(self) -> None:
        doc = AnnotatedDocument.from_text("d", "The Cats, the dogs and 42 birds!")
        assert Preprocessor(stopwords_source="sklearn").tokens(doc) == [
            "the",
            "cats",
            ",",
            "the",
            "dogs",
            "and",
            "42",
            "birds",
            "!",
        ]
        assert Preprocessor(punctuation=False, stopwords=False, stopwords_source="sklearn").tokens(doc) == [
            "cats",
            "dogs",
            "42",
            "birds",
        ]
        assert Preprocessor(numbers="mask", punctuation=False, stopwords_source="builtin").tokens(doc)[-2] == "<num>"
        assert "42" not in Preprocessor(numbers="drop").tokens(doc)
        assert Preprocessor(lowercase=False).tokens(doc)[1] == "Cats"
        assert Preprocessor(min_length=4, punctuation=False).tokens(doc) == ["cats", "dogs", "birds"]
        assert len(Preprocessor().sentences(AnnotatedDocument.from_text("d", "One two. Three four."))) == 2
        assert Preprocessor().params()["normalization"] == "none"

    def test_lemma_and_stem_requirements(self) -> None:
        doc = AnnotatedDocument.from_text("d", "Running dogs")
        with pytest.raises(ConfigurationError, match="spaCy"):
            Preprocessor(normalization="lemma").tokens(doc)
        lemmatized = doc.replace(lemmas=("run", "dog"))
        assert Preprocessor(normalization="lemma").tokens(lemmatized) == ["run", "dog"]
        with pytest.raises(ConfigurationError):
            Preprocessor(normalization="other")  # type: ignore[arg-type]
        assert "the" in stopword_list("en", "builtin") and "the" in stopword_list("en", "sklearn")
        with pytest.raises(ConfigurationError):
            stopword_list("fr", "sklearn")
        try:
            import nltk  # noqa: F401
        except ImportError:
            with pytest.raises(MissingDependencyError):
                Preprocessor(normalization="stem").tokens(doc)
            with pytest.raises(MissingDependencyError):
                stopword_list("en", "nltk")


class TestQuality:
    def test_ner_and_linking_scores(self) -> None:
        doc = HeuristicAnnotator().annotate("d", "John Smith met Acme Corp in London.")
        gold = [
            [
                {"start": 0, "end": 10, "label": "PER", "uri": "http://en.wikipedia.org/wiki/John_Smith"},
                {"start": 15, "end": 24, "label": "ORG"},
                {"start": 28, "end": 34, "label": "LOC"},
            ]
        ]
        scores = ner_quality(gold, [doc])
        assert scores["overall"]["f1"] == 1.0 and scores["by_type"]["PER"]["recall"] == 1.0
        linked = doc.replace(
            entities=tuple(
                Entity(
                    e.text,
                    e.start,
                    e.end,
                    e.label,
                    e.kind,
                    uri="http://dbpedia.org/resource/John_Smith" if e.kind == "PER" else None,
                )
                for e in doc.entities
            )
        )
        assert linking_quality(gold, [linked])["f1"] == 1.0
        assert resource_id("http://en.wikipedia.org/wiki/S%C3%A3o_Paulo") == "São_Paulo" and resource_id(None) is None
        assert prf(0, 0, 0)["f1"] == 0.0
