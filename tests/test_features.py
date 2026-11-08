"""Tests de la couche features : blocs, représentation pondérée, masquage, ablations et absence de fuite (vocabulaire et IDF ajustés sur l'entraînement uniquement)."""

from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.core.config import BlockConfig, RepresentationConfig
from plagia.core.errors import ConfigurationError, MissingDependencyError
from plagia.features import (
    FEATURE_BLOCKS,
    BagOfWords,
    CharNgrams,
    Doc2VecEmbedding,
    EntityFeatures,
    Lsa,
    Masker,
    PosNgrams,
    SentenceEmbeddings,
    Stylometry,
    Tfidf,
    TypeFeatures,
    WordVectors,
    ablate,
    ablation_representations,
    block_family,
    block_slices,
    build_representation,
    representation_families,
)
from plagia.features.base import MinMaxScalerSafe, StandardScalerSafe, make_scaler, ngrams
from plagia.features.masking import replace_token_ranges
from plagia.features.shingles import char_ngram_set, hashed_shingles, positioned_shingles, shingle_set, union_size
from plagia.features.stylometry import mattr, yule_k
from plagia.models.naive_bayes import NaiveBayes


def _docs(annotated: dict[str, AnnotatedDocument], n: int = 20) -> list[AnnotatedDocument]:
    return list(annotated.values())[:n]


class TestLexicalBlocks:
    def test_bow_tfidf_char(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        bow = BagOfWords(stopwords_source="builtin").fit(docs)
        X = bow.transform(docs)
        assert sparse.isspmatrix_csr(X) and X.shape[0] == 20 and X.min() >= 0
        assert X.shape[1] == len(bow.get_feature_names_out())
        binary = BagOfWords(binary=True, stopwords_source="builtin").fit_transform(docs)
        assert set(np.unique(binary.data)) == {1.0}
        tfidf = Tfidf(ngram_range=[1, 2], stopwords_source="builtin").fit_transform(docs)
        assert np.allclose(sparse.linalg.norm(tfidf, axis=1), 1.0)
        assert any(
            " " in name
            for name in Tfidf(ngram_range=(1, 2), stopwords_source="builtin").fit(docs).get_feature_names_out()
        )
        chars = CharNgrams(ngram_range=(2, 3), min_df=1).fit(docs)
        assert chars.transform(docs).shape[1] == len(chars.get_feature_names_out())
        assert str(chars.get_feature_names_out()[0]).startswith("c[")
        assert BagOfWords().fit_transform(["plain text input"]).shape == (1, 3)

    def test_vocabulary_and_idf_fitted_on_training_only(self) -> None:
        """Absence de fuite : un mot propre au test n'entre ni dans le vocabulaire ni dans l'IDF."""
        train = ["alpha beta gamma", "alpha delta", "beta gamma"]
        test = ["zeta zeta alpha"]
        block = Tfidf(ngram_range=(1, 1), stopwords_source="builtin").fit(train)
        assert "zeta" not in set(block.get_feature_names_out())
        idf_before = block.vectorizer_.idf_.copy()
        block.transform(test)
        assert np.array_equal(idf_before, block.vectorizer_.idf_)
        chars = CharNgrams(ngram_range=(4, 4), min_df=1).fit(train)
        assert "c[zeta]" not in set(chars.get_feature_names_out())

    def test_blocks_refit_inside_each_fold(self, annotated: dict[str, AnnotatedDocument]) -> None:
        """Dans GridSearchCV, chaque pli réajuste la représentation sur son seul entraînement."""
        docs = list(annotated.values())
        labels = [d.id.split("-")[0] for d in docs]
        sizes: list[int] = []

        class Recording(BagOfWords):
            def fit(self, X, y=None):  # type: ignore[no-untyped-def]
                sizes.append(len(X))
                return super().fit(X, y)

            def fit_transform(self, X, y=None, **params):  # type: ignore[no-untyped-def]
                sizes.append(len(X))
                return super().fit_transform(X, y, **params)

        pipeline = Pipeline([("features", Recording(stopwords_source="builtin")), ("clf", NaiveBayes())])
        search = GridSearchCV(pipeline, {"clf__alpha": [1.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=0))
        search.fit(docs, labels)
        assert sorted(sizes)[:3] == [2 * len(docs) // 3] * 3 and max(sizes) == len(docs)


class TestStylometryAndPos:
    def test_stylometry(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        block = Stylometry().fit(docs)
        X = block.transform(docs).toarray()
        assert X.shape == (20, len(block.get_feature_names_out())) and X.min() >= 0 and X.max() <= 1
        assert "fw:the" in block.get_feature_names_out() and "rich:yule_k" in block.get_feature_names_out()
        raw = Stylometry(scale="none", function_words=False, punctuation=False).fit(docs).transform(docs).toarray()
        assert raw.shape[1] == 4 + 15 + 7
        standard = Stylometry(scale="standard").fit(docs).transform(docs).toarray()
        assert abs(standard.mean()) < 0.1
        with pytest.raises(ValueError):
            make_scaler("weird")

    def test_richness_measures(self) -> None:
        assert yule_k(Counter()) == 0.0
        assert yule_k(Counter({"a": 1, "b": 1})) == 0.0
        assert mattr([], 5) == 0.0 and mattr(["a", "b", "a"], 5) == pytest.approx(2 / 3)
        assert mattr(["a", "b"] * 10, 4) == pytest.approx(0.5)

    def test_pos_ngrams(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        block = PosNgrams(ngram_range=(1, 2)).fit(docs)
        names = set(block.get_feature_names_out())
        assert "pos[DET NOUN]" in names and block.transform(docs).shape[0] == 20
        with pytest.raises(ConfigurationError, match="annotateur grammatical"):
            PosNgrams().fit(["no tags here"])
        tagged = docs[0].replace(tags=tuple("NN" for _ in docs[0].tokens))
        assert "pos[NN]" in set(PosNgrams(tagset="fine", ngram_range=(1, 1)).fit([tagged]).get_feature_names_out())


class TestEntityBlocks:
    def test_entity_features(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        block = EntityFeatures().fit(docs)
        X = block.transform(docs)
        names = list(block.get_feature_names_out())
        assert X.shape == (20, len(names)) and "ent_density:PER" in names
        assert any(n.startswith("ent[") for n in names) and any(n.startswith("ent_seq[") for n in names)
        empty = EntityFeatures(counts=False, proportions=False).fit(["no entity"])
        assert empty.transform(["no entity"]).shape == (1, 0)
        uri_doc = docs[0].replace(entities=(Entity("Paris", 0, 5, "GPE", "LOC", uri="dbr:Paris"),))
        assert (
            "ent[dbr:Paris]"
            in EntityFeatures(level="uri", counts=False, proportions=False, type_ngrams=False)
            .fit([uri_doc])
            .get_feature_names_out()
        )

    def test_type_features(self) -> None:
        typed = AnnotatedDocument.from_text("d", "Zidane played.").replace(
            entities=(
                Entity("Zidane", 0, 6, "PERSON", "PER", "u", ("0:owl:Thing", "1:dbo:Agent", "2:dbo:Person")),
                Entity("played", 7, 13, "X", "MISC"),
            ),
        )
        names = set(TypeFeatures(depth=1).fit([typed]).get_feature_names_out())
        assert names == {"type[dbo:Agent]", "type[kind:MISC]"}
        assert TypeFeatures(fallback="none").fit(["nothing"]).transform(["x"]).shape == (1, 0)
        assert TypeFeatures(fallback="none").fit(["nothing"]).get_feature_names_out().size == 0


class TestMasking:
    def test_modes(self, annotated: dict[str, AnnotatedDocument]) -> None:
        doc = AnnotatedDocument.from_text("d", "John Smith visited Paris yesterday.").replace(
            entities=(Entity("John Smith", 0, 10, "PERSON", "PER"), Entity("Paris", 19, 24, "GPE", "LOC")),
            pos=("PROPN", "PROPN", "VERB", "PROPN", "NOUN", "PUNCT"),
        )
        masked = Masker("entities").fit([doc]).transform([doc])[0]
        assert masked.text == "PER visited LOC yesterday." and masked.tokens[0] == "PER"
        assert Masker("entities", placeholder="star").fit([doc]).mask(doc).text == "* visited * yesterday."
        assert Masker("propn").fit([doc]).mask(doc).text == "PROPN PROPN visited PROPN yesterday."
        content = Masker("content").fit([doc]).mask(doc)
        assert content.text == "**** ***** ******* ***** *********." and len(content.tokens) == len(doc.tokens)
        assert Masker("content", style="single").fit([doc]).mask(doc).text.startswith("* *")
        top = Masker("content", keep_top_k=1).fit([doc, doc])
        assert len(top.keep_) == 1
        assert Masker("none").fit([doc]).mask(doc) is doc
        with pytest.raises(ConfigurationError):
            Masker("bad").fit([doc])
        with pytest.raises(ConfigurationError):
            Masker("propn").fit([doc]).mask(AnnotatedDocument.from_text("x", "No tags"))
        assert Masker().get_feature_names_out(["x"]) == ["x"]

    def test_replace_token_ranges_keeps_spacing(self) -> None:
        doc = AnnotatedDocument.from_text("d", "A  b\n\nc d.")
        out = replace_token_ranges(doc, {1: (3, "X", "PROPN")})
        assert out.text == "A  X d." and out.tokens == ("A", "X", "d", ".")


class TestRepresentation:
    def test_weighted_union_and_slices(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        rep = RepresentationConfig(
            blocks=[
                BlockConfig(type="bow", params={"stopwords_source": "builtin"}),
                BlockConfig(type="bow", weight=3.0, params={"stopwords_source": "builtin"}),
                BlockConfig(type="tfidf", mask="entities", params={"stopwords_source": "builtin"}),
            ]
        )
        union = build_representation(rep)
        X = union.fit_transform(docs)
        slices = block_slices(union)
        assert list(slices) == ["bow", "bow_2", "tfidf"]
        first, second = X[:, slices["bow"]].toarray(), X[:, slices["bow_2"]].toarray()
        assert np.allclose(second, 3 * first)
        assert len(union.get_feature_names_out()) == X.shape[1]
        clone(union)
        pickle.loads(pickle.dumps(union))
        with pytest.raises(ConfigurationError):
            build_representation([])

    def test_families_and_ablations(self) -> None:
        rep = RepresentationConfig(
            blocks=[BlockConfig(type="char_ngrams"), BlockConfig(type="stylometry"), BlockConfig(type="entities")]
        )
        assert representation_families(rep) == ["chars", "stylometry", "entities"]
        assert block_family("tfidf") == "words" and block_family("unknown") == "unknown"
        variants = ablation_representations(rep, ["entities"])
        assert set(variants) == {"only_entities", "without_entities"}
        assert [b.type for b in variants["without_entities"].blocks] == ["char_ngrams", "stylometry"]
        with pytest.raises(ConfigurationError):
            ablate(rep, "words", "only")
        with pytest.raises(ConfigurationError):
            ablate(rep, "chars", "sideways")
        assert set(FEATURE_BLOCKS.names()) >= {
            "bow",
            "tfidf",
            "char_ngrams",
            "stylometry",
            "pos_ngrams",
            "entities",
            "types",
            "lsa",
        }


class TestSemanticBlocks:
    def test_lsa(self, annotated: dict[str, AnnotatedDocument]) -> None:
        docs = _docs(annotated)
        block = Lsa(n_components=5).fit(docs)
        X = block.transform(docs).toarray()
        assert X.shape == (20, 5) and np.allclose(np.linalg.norm(X, axis=1), 1.0)
        assert block.get_feature_names_out()[0] == "lsa0"

    def test_optional_backends_report_missing_dependencies(self) -> None:
        for block in (Doc2VecEmbedding(), WordVectors()):
            try:
                import gensim  # noqa: F401
            except ImportError:
                with pytest.raises(MissingDependencyError):
                    block.fit(["a b c"])
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            with pytest.raises(MissingDependencyError):
                SentenceEmbeddings().fit(["a"])


@pytest.mark.slow
def test_gensim_blocks_when_available(annotated: dict[str, AnnotatedDocument]) -> None:  # pragma: no cover
    pytest.importorskip("gensim")
    docs = _docs(annotated)
    assert Doc2VecEmbedding(vector_size=8, epochs=2).fit(docs).transform(docs).shape == (20, 8)
    assert WordVectors(vector_size=8, epochs=2).fit(docs).transform(docs).shape == (20, 8)


class TestShinglesAndScalers:
    def test_shingles(self) -> None:
        text = "The cat sat on the mat. The cat sat."
        assert "the cat sat" in shingle_set(text, 3) and len(char_ngram_set("abcd", 2)) == 3
        positioned = positioned_shingles(text, 3)
        assert text[positioned[0][1] : positioned[0][2]] == "The cat sat"
        assert hashed_shingles(["a", "b", "c"], 2).dtype == np.uint64
        assert shingle_set("a b c", 1) == {"a", "b", "c"} and union_size([{"a"}, {"a", "b"}]) == 2
        assert ngrams(["a", "b", "c"], (1, 2)) == ["a", "b", "c", "a b", "b c"]

    def test_scalers(self) -> None:
        values = np.array([[0.0, 5.0], [2.0, 5.0]])
        assert np.allclose(MinMaxScalerSafe().fit(values).transform(np.array([[4.0, 5.0]])), [[1.0, 0.0]])
        assert np.allclose(StandardScalerSafe().fit(values).transform(values)[:, 1], 0.0)
        assert make_scaler("none") is None
