"""Stockage des corpus en JSON Lines avec manifeste (format pivot du projet).

Disposition d'un répertoire de corpus :

* `documents.jsonl` : un document par ligne (texte et métadonnées) ;
* `manifest.jsonl` et `manifest.sha256` : manifeste et son empreinte (F1.7) ;
* `cases.jsonl` : cas de plagiat avec décalages de caractères (corpus de plagiat) ;
* `pairs.jsonl` : paires (suspect, source) à aligner (format PAN) ;
* `corpus.json` : nom, graine, paramètres de construction, effectifs, empreinte.

Un corpus n'est jamais modifié en place : un corpus nettoyé est une nouvelle version écrite dans un autre répertoire, avec sa propre empreinte.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from plagia.core.errors import CorpusNotFoundError, PlagiaError
from plagia.core.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from plagia.core.text import count_words
from plagia.core.types import Corpus, Document, PassagePair
from plagia.corpus.manifest import read_manifest_digest, verify_manifest, write_manifest

DOCUMENTS = "documents.jsonl"
CASES = "cases.jsonl"
PAIRS = "pairs.jsonl"
INFO = "corpus.json"


def corpus_summary(corpus: Corpus) -> dict[str, Any]:
    """Effectifs : documents, auteurs, œuvres, jeux, longueurs en mots, cas par obfuscation."""
    lengths = sorted(count_words(d.text) for d in corpus)
    words = (
        {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "mean": round(sum(lengths) / len(lengths), 1),
            "max": lengths[-1],
        }
        if lengths
        else {}
    )
    return {
        "documents": len(corpus),
        "authors": len(corpus.authors),
        "works": len({d.group for d in corpus}),
        "splits": dict(sorted(corpus.split_counts().items())),
        "languages": dict(sorted(Counter(d.language for d in corpus).items())),
        "words": words,
        "cases": len(corpus.cases),
        "cases_by_obfuscation": dict(sorted(Counter(c.obfuscation or "none" for c in corpus.cases).items())),
        "pairs": len(corpus.pairs),
    }


def save_corpus(corpus: Corpus, directory: str | Path) -> str:
    """Écrit le corpus ; renvoie l'empreinte SHA-256 du manifeste."""
    directory = ensure_dir(directory)
    write_jsonl(directory / DOCUMENTS, (d.to_dict() for d in corpus))
    digest = write_manifest(corpus, directory)
    if corpus.cases:
        write_jsonl(directory / CASES, (c.to_dict() for c in corpus.cases))
    else:
        (directory / CASES).unlink(missing_ok=True)
    if corpus.pairs:
        write_jsonl(directory / PAIRS, ({"suspicious": s, "source": t} for s, t in corpus.pairs))
    else:
        (directory / PAIRS).unlink(missing_ok=True)
    write_json(
        directory / INFO,
        {"name": corpus.name, "manifest_sha256": digest, "summary": corpus_summary(corpus), "meta": corpus.meta},
    )
    return digest


def load_corpus(directory: str | Path, verify: bool = True) -> Corpus:
    """Charge un corpus ; vérifie le manifeste si `verify`."""
    directory = Path(directory)
    if not (directory / DOCUMENTS).exists():
        raise CorpusNotFoundError(directory)
    if verify and (directory / "manifest.sha256").exists() and not verify_manifest(directory):
        raise PlagiaError(f"Manifeste altéré dans {directory} : empreinte différente.")
    info = read_json(directory / INFO) if (directory / INFO).exists() else {}
    documents = [Document.from_dict(r) for r in read_jsonl(directory / DOCUMENTS)]
    cases = [PassagePair.from_dict(r) for r in read_jsonl(directory / CASES)] if (directory / CASES).exists() else []
    pairs = (
        [(r["suspicious"], r["source"]) for r in read_jsonl(directory / PAIRS)] if (directory / PAIRS).exists() else []
    )
    meta = dict(info.get("meta") or {})
    if (directory / "manifest.sha256").exists():
        meta["manifest_sha256"] = read_manifest_digest(directory)
    return Corpus(documents, name=info.get("name", directory.name), cases=cases, pairs=pairs, meta=meta)


def corpus_fingerprint(directory: str | Path) -> str:
    """Empreinte du manifeste d'un corpus enregistré."""
    return read_manifest_digest(directory)
