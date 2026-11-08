"""Corpus de presse Reuters C50 : 50 auteurs, 50 textes d'entraînement et 50 de test par auteur, sans recouvrement (UCI, licence CC BY 4.0).

L'archive UCI contient `C50train/<Auteur>/<fichier>.txt` et `C50test/...`, parfois dans une archive imbriquée : le chargeur lit les deux dispositions, ainsi qu'un répertoire déjà extrait.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

from plagia.core.errors import ExternalServiceError, PlagiaError
from plagia.core.io import ensure_dir
from plagia.core.logging import get_logger
from plagia.core.types import Corpus, Document
from plagia.corpus.cleaning import CleaningConfig, clean_text, decode_bytes

log = get_logger(__name__)

C50_URL = "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip"
C50_LICENCE = "CC BY 4.0"
_MEMBER_RE = re.compile(r"(?:^|/)C50(train|test)/([^/]+)/([^/]+\.txt)$", re.I)


def download_c50(destination: str | Path, url: str = C50_URL, timeout: int = 120) -> Path:
    """Télécharge l'archive C50 (si absente) ; renvoie son chemin."""
    target = Path(destination)
    if target.is_dir():
        target = target / "reuter50_50.zip"
    if target.exists():
        return target
    ensure_dir(target.parent)
    log.info("Téléchargement de C50 depuis %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            target.write_bytes(response.read())
    except OSError as exc:
        raise ExternalServiceError(f"Téléchargement de C50 impossible : {exc}") from exc
    return target


def _iter_zip(archive: zipfile.ZipFile) -> Iterator[tuple[str, str, str, bytes]]:
    found = False
    for name in sorted(archive.namelist()):
        match = _MEMBER_RE.search(name)
        if match:
            found = True
            yield match.group(1).lower(), match.group(2), match.group(3), archive.read(name)
    if not found:  # archive imbriquée (C50.zip dans l'archive UCI)
        for name in sorted(archive.namelist()):
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
                    yield from _iter_zip(inner)


def _iter_directory(root: Path) -> Iterator[tuple[str, str, str, bytes]]:
    for path in sorted(root.rglob("*.txt")):
        match = _MEMBER_RE.search(path.relative_to(root).as_posix())
        if match:
            yield match.group(1).lower(), match.group(2), match.group(3), path.read_bytes()


def load_c50(source: str | Path, cleaning: CleaningConfig | None = CleaningConfig()) -> Corpus:
    """Charge C50 depuis une archive zip ou un répertoire extrait.

    Identifiants stables `c50/<jeu>/<auteur>/<fichier>` ; le jeu (`train`/`test`)
    est celui fourni par UCI. Les mentions du nom de l'auteur sont retirées du texte.
    """
    source = Path(source)
    if not source.exists():
        raise PlagiaError(f"Source C50 introuvable : {source} (voir « plagia corpus c50 --download »).")
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            members = list(_iter_zip(archive))
    else:
        members = list(_iter_directory(source))
    if not members:
        raise PlagiaError(f"Aucun fichier C50train/C50test trouvé dans {source}.")
    order = {"train": 0, "test": 1}
    members.sort(key=lambda m: (order[m[0]], m[1], m[2]))
    documents = []
    for split, author, filename, data in members:
        text = decode_bytes(data)
        if cleaning is not None:
            text = clean_text(text, cleaning, author)
        documents.append(
            Document(
                id=f"c50/{split}/{author}/{filename.removesuffix('.txt')}",
                text=text,
                author=author,
                work=None,
                language="en",
                licence=C50_LICENCE,
                split=split,
                meta={"source": "Reuters C50 (UCI)"},
            )
        )
    log.info("C50 : %d documents, %d auteurs.", len(documents), len({d.author for d in documents}))
    return Corpus(documents, name="c50", meta={"source": "UCI Reuter_50_50", "licence": C50_LICENCE})
