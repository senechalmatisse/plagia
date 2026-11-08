"""Moteur spaCy : segmentation, lemmes, étiquettes grammaticales, entités.

Les documents spaCy sont sérialisés au format `DocBin` dans le cache ; le modèle n'est chargé qu'à la première utilisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from plagia.annotation.base import ANNOTATORS, Annotator
from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.annotation.schema import normalize_label
from plagia.core.errors import MissingDependencyError

_NON_ENGLISH = ("fr_", "de_", "es_", "it_", "pt_", "nl_")


def from_spacy_doc(doc_id: str, doc: Any, scheme: str = "ontonotes", engine: str = "spacy") -> AnnotatedDocument:
    """Convertit un ``spacy.tokens.Doc`` (les jetons d'espacement sont ignorés)."""
    kept = [t for t in doc if not t.is_space]
    index = {t.i: k for k, t in enumerate(kept)}
    sentences: list[tuple[int, int]] = []
    if doc.has_annotation("SENT_START"):
        for sent in doc.sents:
            members = [index[t.i] for t in sent if t.i in index]
            if members:
                sentences.append((members[0], members[-1] + 1))
    entities = tuple(
        Entity(
            text=ent.text,
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            kind=normalize_label(ent.label_, scheme),
            uri=ent.kb_id_ or None,
        )
        for ent in doc.ents
    )
    has_pos = doc.has_annotation("POS")
    has_tag = doc.has_annotation("TAG")
    has_lemma = doc.has_annotation("LEMMA")
    return AnnotatedDocument(
        id=doc_id,
        text=doc.text,
        tokens=tuple(t.text for t in kept),
        offsets=tuple((t.idx, t.idx + len(t.text)) for t in kept),
        pos=tuple(t.pos_ for t in kept) if has_pos else None,
        tags=tuple(t.tag_ for t in kept) if has_tag else None,
        lemmas=tuple(t.lemma_ for t in kept) if has_lemma else None,
        sentences=tuple(sentences),
        entities=entities,
        engine=engine,
    )


class SpacyPipeline(Annotator):
    """Pipeline spaCy chargé à la demande (``en_core_web_sm``, ``_md``, ``_trf``, ``fr_core_news_sm``…)."""

    payload_format = "docbin"

    def __init__(
        self,
        model: str = "en_core_web_sm",
        batch_size: int = 64,
        n_process: int = 1,
        disable: Sequence[str] = (),
        scheme: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.n_process = n_process
        self.disable = tuple(disable)
        self.scheme = scheme or ("wikiner" if model.startswith(_NON_ENGLISH) else "ontonotes")
        self.name = f"spacy:{model}"
        self._nlp: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state

    @property
    def nlp(self) -> Any:
        """Pipeline spaCy (chargé à la première utilisation)."""
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise MissingDependencyError("spacy", "annotation spaCy", "Lancez « uv sync ».") from exc
            try:
                self._nlp = spacy.load(self.model, disable=list(self.disable))
            except OSError as exc:
                raise MissingDependencyError(
                    self.model,
                    "annotation spaCy",
                    f"Lancez « plagia setup --spacy {self.model} » ou « uv sync --group models ».",
                ) from exc
        return self._nlp

    @property
    def version(self) -> str:
        nlp = self.nlp  # lève MissingDependencyError si spaCy ou le modèle manquent
        import spacy

        return f"spacy-{spacy.__version__}/{self.model}-{nlp.meta.get('version', '?')}"

    def params(self) -> dict[str, Any]:
        return {"model": self.model, "disable": sorted(self.disable)}

    def process(self, texts: Sequence[str]) -> list[Any]:
        return list(self.nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_process))

    def serialize(self, raw: Any) -> bytes:
        from spacy.tokens import DocBin

        payload: bytes = DocBin(docs=[raw]).to_bytes()
        return payload

    def deserialize(self, payload: bytes) -> Any:
        from spacy.tokens import DocBin

        return next(iter(DocBin().from_bytes(payload).get_docs(self.nlp.vocab)))

    def convert(self, doc_id: str, text: str, raw: Any) -> AnnotatedDocument:
        return from_spacy_doc(doc_id, raw, self.scheme, engine=self.name)


ANNOTATORS.add("spacy", SpacyPipeline)
