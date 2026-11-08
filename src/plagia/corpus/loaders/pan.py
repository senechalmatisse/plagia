"""Corpus PAN d'alignement de textes : PAN 2013, PAN 2014 et PAN-PC-11.

Le chargeur parcourt un répertoire extrait : textes `suspicious-documentXXXXX.txt` et `source-documentXXXXX.txt`, annotations XML (`<feature name="plagiarism" …/>`) et
fichiers `pairs`. Les textes sont lus SANS normalisation (ni Unicode, ni fins de ligne) : les décalages de caractères des annotations PAN restent valides.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from xml.sax.saxutils import quoteattr

from plagia.core.errors import PlagiaError
from plagia.core.io import atomic_write_text, ensure_dir
from plagia.core.logging import get_logger
from plagia.core.types import Corpus, Document, Passage, PassagePair

log = get_logger(__name__)

PAN_LICENCE = "PAN (usage de recherche)"


def _doc_id(reference: str) -> str:
    return Path(reference).name.removesuffix(".txt")


def _role(name: str) -> str | None:
    if name.startswith("suspicious-document"):
        return "suspicious"
    if name.startswith("source-document"):
        return "source"
    return None


def _subcorpus_obfuscation(folder: str) -> str | None:
    # « 02-no-obfuscation » → « none », « 03-random-obfuscation » → « random »
    parts = folder.split("-", 1)
    if len(parts) == 2 and parts[0].isdigit():
        label = parts[1].removesuffix("-obfuscation")
        return "none" if label in {"no", "no-obfuscation"} else label
    return None


def parse_pan_xml(path: str | Path, feature_name: str = "plagiarism") -> list[PassagePair]:
    """Cas (ou détections) d'un fichier XML PAN."""
    path = Path(path)
    root = ET.parse(path).getroot()
    reference = root.get("reference")
    if not reference:
        return []
    susp_id = _doc_id(reference)
    folder_label = _subcorpus_obfuscation(path.parent.name)
    pairs = []
    for feature in root.iter("feature"):
        if feature.get("name") != feature_name:
            continue
        source_ref = feature.get("source_reference")
        source = None
        if source_ref:
            source = Passage(
                _doc_id(source_ref), int(feature.get("source_offset", 0)), int(feature.get("source_length", 0))
            )
        pairs.append(
            PassagePair(
                suspicious=Passage(susp_id, int(feature.get("this_offset", 0)), int(feature.get("this_length", 0))),
                source=source,
                obfuscation=feature.get("obfuscation") or folder_label,
                meta={"subcorpus": path.parent.name, "type": feature.get("type")},
            )
        )
    return pairs


def load_pan(root: str | Path, split: str = "test", name: str | None = None) -> Corpus:
    """Charge un corpus PAN d'alignement de textes extrait dans `root`."""
    root = Path(root)
    if not root.is_dir():
        raise PlagiaError(f"Répertoire PAN introuvable : {root}")
    texts: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if _role(path.name):
            texts.setdefault(path.name.removesuffix(".txt"), path)
    if not texts:
        raise PlagiaError(f"Aucun document suspect ou source trouvé dans {root}.")
    cases: list[PassagePair] = []
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            cases.extend(parse_pan_xml(xml_path))
        except ET.ParseError as exc:
            log.warning("XML illisible ignoré : %s (%s)", xml_path, exc)
    pairs: dict[tuple[str, str], None] = {}
    for pairs_file in sorted(root.rglob("pairs")):
        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                pairs[(_doc_id(fields[0]), _doc_id(fields[1]))] = None
    if not pairs:
        for case in cases:
            if case.source is not None:
                pairs[(case.suspicious.document_id, case.source.document_id)] = None
    documents = [
        Document(
            id=doc_id,
            text=path.read_bytes().decode("utf-8", errors="replace"),
            author=None,
            work=doc_id,
            language="en",
            licence=PAN_LICENCE,
            split=split,
            meta={"role": _role(path.name)},
        )
        for doc_id, path in sorted(texts.items())
    ]
    corpus_name = name or root.name
    log.info("PAN « %s » : %d documents, %d cas, %d paires.", corpus_name, len(documents), len(cases), len(pairs))
    return Corpus(
        documents, name=corpus_name, cases=cases, pairs=list(pairs), meta={"source": "PAN", "licence": PAN_LICENCE}
    )


def combine_pan(train: Corpus, test: Corpus, name: str | None = None) -> Corpus:
    """Réunit un corpus PAN d'entraînement et un corpus de test sans fuite.

    Les deux corpus PAN numérotent leurs documents de la même façon. Un document du test
    identique (même texte) à un document de l'entraînement est retiré de l'entraînement,
    avec ses cas et ses paires : le test reste intact. Si des identifiants restent communs
    (textes différents), ceux du test sont préfixés par `test/` ; l'export au format PAN
    retire ce préfixe.
    """
    test_texts = {d.sha256 for d in test}
    dropped = {d.id for d in train if d.sha256 in test_texts}
    if dropped:
        log.warning("%d document(s) de l'entraînement présents dans le test : retirés de l'entraînement.", len(dropped))

    def kept(case: PassagePair) -> bool:
        return case.suspicious.document_id not in dropped and (
            case.source is None or case.source.document_id not in dropped
        )

    train_docs = [d for d in train if d.id not in dropped]
    train_ids = {d.id for d in train_docs}
    prefix = "test/" if any(d.id in train_ids for d in test) else ""

    def rename(doc_id: str) -> str:
        return prefix + doc_id

    def renamed(case: PassagePair) -> PassagePair:
        source = case.source
        if source is not None:
            source = Passage(rename(source.document_id), source.start, source.length)
        suspicious = Passage(rename(case.suspicious.document_id), case.suspicious.start, case.suspicious.length)
        return PassagePair(
            suspicious=suspicious, source=source, score=case.score, obfuscation=case.obfuscation, meta=dict(case.meta)
        )

    return Corpus(
        [*train_docs, *(d.replace(id=rename(d.id), work=rename(d.group)) for d in test)],
        name=name or train.name,
        cases=[*(c for c in train.cases if kept(c)), *(renamed(c) for c in test.cases)],
        pairs=[
            *((s, t) for s, t in train.pairs if s not in dropped and t not in dropped),
            *((rename(s), rename(t)) for s, t in test.pairs),
        ],
        meta={**train.meta, "dropped_from_train": len(dropped), "test_prefix": prefix},
    )


def pan_name(doc_id: str) -> str:
    """Nom de fichier PAN d'un document (sans préfixe de jeu)."""
    return doc_id.rsplit("/", 1)[-1]


def write_pan_detections(detections: Iterable[PassagePair], directory: str | Path) -> int:
    """Écrit les détections au format XML de PAN (un fichier par paire suspect-source)."""
    directory = ensure_dir(directory)
    grouped: dict[tuple[str, str], list[PassagePair]] = {}
    for det in detections:
        if det.source is None:
            continue
        grouped.setdefault((det.suspicious.document_id, det.source.document_id), []).append(det)
    for (susp_id, src_id), items in sorted(grouped.items()):
        susp, src = pan_name(susp_id), pan_name(src_id)
        lines = [f"<document reference={quoteattr(susp + '.txt')}>"]
        for det in sorted(items, key=lambda d: d.suspicious.start):
            assert det.source is not None
            lines.append(
                '<feature name="detected-plagiarism" '
                f'this_offset="{det.suspicious.start}" this_length="{det.suspicious.length}" '
                f"source_reference={quoteattr(src + '.txt')} "
                f'source_offset="{det.source.start}" source_length="{det.source.length}"/>'
            )
        lines.append("</document>")
        atomic_write_text(directory / f"{susp}-{src}.xml", "\n".join(lines) + "\n")
    return len(grouped)
