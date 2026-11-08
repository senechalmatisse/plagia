"""Ligne de commande `plagia`.

Sous-commandes : `setup`, `corpus`, `annotate`, `run`, `attribute`, `detect`, cluster`, `report`, `demo` (démonstrateur) et `bench` (contrôle de performance).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plagia import __version__
from plagia.core.config import AnnotationConfig, BlockConfig, ModelConfig, RepresentationConfig, load_config
from plagia.core.errors import CorpusNotFoundError, MissingDependencyError, PlagiaError
from plagia.core.logging import configure_logging, get_logger

log = get_logger("plagia")

DEFAULT_REPRESENTATION = RepresentationConfig(
    blocks=[
        BlockConfig(type="char_ngrams", params={"ngram_range": [3, 5]}),
        BlockConfig(type="stylometry", weight=0.5),
    ]
)


# setup
def cmd_setup(args: argparse.Namespace) -> int:
    """Installe les ressources NLTK et les modèles spaCy ; vérifie Java et Spotlight."""
    status: dict[str, Any] = {}
    if not args.check_only:
        try:
            import nltk

            for resource in args.nltk:
                status[f"nltk:{resource}"] = bool(nltk.download(resource, quiet=True))
        except ImportError:
            status["nltk"] = "absent (uv sync)"
    import importlib.util

    for model in args.spacy:
        if importlib.util.find_spec(model) is not None:
            status[f"spacy:{model}"] = "ok"
        elif args.check_only:
            status[f"spacy:{model}"] = "absent (uv sync : groupe models)"
        else:
            try:
                from spacy.cli import download

                download(model)
                status[f"spacy:{model}"] = "installé"
            except (ImportError, SystemExit) as exc:
                status[f"spacy:{model}"] = f"échec ({exc}) ; installez-le avec « uv sync » (groupe models)"
    for module in ("spacy", "nltk", "gensim", "sentence_transformers", "umap", "networkx", "rdflib"):
        try:
            __import__(module)
            status[f"module:{module}"] = "ok"
        except ImportError:
            status[f"module:{module}"] = "absent"
    java = shutil.which("java")
    if java:
        version = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)
        lines = [line for line in (version.stderr + version.stdout).splitlines() if "version" in line]
        status["java"] = lines[0].strip() if lines else java
    else:
        status["java"] = "absent (Stanford facultatif)"
    if args.spotlight_url:
        try:
            with urllib.request.urlopen(f"{args.spotlight_url.rstrip('/')}/annotate?text=Berlin", timeout=10):
                status["spotlight"] = "joignable"
        except OSError as exc:
            status["spotlight"] = f"injoignable ({exc})"
    for key, value in status.items():
        print(f"{key:32s} {value}")
    return 0


# corpus
def _export_corpus(args: argparse.Namespace) -> int:
    """Écrit des documents d'un corpus en fichiers .txt, avec un index CSV (exemples, essais)."""
    import csv
    import re
    from collections import Counter

    from plagia.corpus import load_corpus

    corpus = load_corpus(args.path)
    passages = Counter(c.suspicious.document_id for c in corpus.cases)
    levels: dict[str, set[str]] = {}
    for case in corpus.cases:
        levels.setdefault(case.suspicious.document_id, set()).add(case.obfuscation or "")
    wanted = set(args.ids) if args.ids else None
    if wanted is not None and (unknown := wanted - set(corpus.ids)):
        raise PlagiaError("Identifiant(s) absent(s) du corpus : " + ", ".join(sorted(unknown)))
    selected = [
        d
        for d in corpus
        if (wanted is None or d.id in wanted)
        and (args.split is None or d.split == args.split)
        and (args.role is None or d.meta.get("role") == args.role)
        and (not args.with_plagiarism or passages[d.id] > 0)
    ][: args.limit]
    if not selected:
        raise PlagiaError("Aucun document ne correspond à ces critères.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in selected:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", document.id) + ".txt"
        (out / name).write_text(document.text, encoding="utf-8")
        rows.append(
            {
                "fichier": name,
                "id": document.id,
                "auteur": document.author or "",
                "jeu": document.split or "",
                "role": document.meta.get("role", ""),
                "passages_plagies": passages[document.id],
                "obfuscations": " ".join(sorted(levels.get(document.id, set()))),
            }
        )
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} document(s) écrit(s) dans {out} ; index : {out / 'index.csv'}")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    """Construit, vérifie ou nettoie un corpus."""
    from plagia.corpus import (
        CleaningConfig,
        SyntheticConfig,
        assign_group_splits,
        carve_validation,
        check_no_leakage,
        clean_corpus,
        deduplicate,
        generate_plagiarism_corpus,
        generate_toy_corpus,
        load_corpus,
        save_corpus,
        toy_synonyms,
    )
    from plagia.corpus.store import corpus_summary

    action = args.corpus_command
    if action == "export":
        return _export_corpus(args)
    if action == "info":
        corpus = load_corpus(args.path)
        print(json.dumps(corpus_summary(corpus), indent=2, ensure_ascii=False))
        check_no_leakage(corpus)
        print("Manifeste vérifié ; aucune fuite entre jeux.")
        return 0
    if action == "c50":
        from plagia.corpus.loaders.c50 import download_c50, load_c50

        source = Path(args.source)
        if args.download:
            source = download_c50(source)
        corpus = load_c50(source)
        corpus, removed = deduplicate(corpus)
        print(f"{len(removed)} doublon(s) exact(s) supprimé(s).")
        corpus = carve_validation(corpus, args.validation, args.seed) if args.validation else corpus
    elif action == "gutenberg":
        from plagia.corpus.loaders.gutenberg import GutenbergConfig, build_gutenberg_corpus

        corpus, removed = deduplicate(build_gutenberg_corpus(GutenbergConfig.from_yaml(args.config), args.raw))
        print(f"{len(removed)} doublon(s) exact(s) supprimé(s).")
    elif action == "pan":
        from plagia.corpus.loaders.pan import combine_pan, load_pan
        from plagia.corpus.splits import assign_pair_component_splits

        corpus = load_pan(args.source, split=args.split, name=args.name)
        if args.test:
            corpus = combine_pan(corpus, load_pan(args.test, split="test"), name=args.name or "pan")
        if args.validation:
            train = corpus.split("train")
            carved = assign_pair_component_splits(
                train, {"train": 1 - args.validation, "validation": args.validation}, args.seed
            )
            moved = {d.id: d.split for d in carved}
            corpus = corpus.map(lambda d: d.replace(split=moved.get(d.id, d.split)))
    elif action == "aida":
        from plagia.corpus.loaders.aida import load_aida

        corpus = load_aida(args.tsv, args.conll)
    elif action == "synthetic":
        base = load_corpus(args.base)
        settings: dict[str, Any] = {"seed": args.seed}
        if args.config:
            from plagia.core.io import read_yaml

            settings.update(read_yaml(args.config) or {})
        for key in ("n_sources", "n_suspicious", "passages_per_cell"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        synonyms = toy_synonyms() if args.synonyms == "toy" else None
        corpus = generate_plagiarism_corpus(
            base, SyntheticConfig(**settings), synonyms=synonyms, name=args.name or "synthetic"
        )
    elif action == "toy":
        corpus = _toy_corpus(
            args, generate_toy_corpus, assign_group_splits, SyntheticConfig, generate_plagiarism_corpus, toy_synonyms
        )
    elif action == "clean":
        corpus = clean_corpus(
            load_corpus(args.source),
            CleaningConfig(signatures=not args.keep_signatures, author_mentions=not args.keep_author_mentions),
        )
        corpus, removed = deduplicate(corpus)
        print(f"{len(removed)} doublon(s) exact(s) supprimé(s).")
    else:  # pragma: no cover - argparse garantit l'action
        raise PlagiaError(f"Action inconnue : {action}")
    check_no_leakage(corpus)
    if corpus.cases:
        from plagia.corpus.synthetic import cell_counts

        cells = ", ".join(f"{level}/{size}={n}" for (level, size), n in cell_counts(corpus).items())
        print(f"Passages par cellule (niveau/taille) : {cells}")
        if corpus.meta.get("shortfall"):
            log.warning("Cellules sous l'objectif : %s", corpus.meta["shortfall"])
    digest = save_corpus(corpus, args.out)
    print(
        json.dumps(
            {"out": str(args.out), "manifest_sha256": digest, **corpus_summary(corpus)}, indent=2, ensure_ascii=False
        )
    )
    return 0


def _toy_corpus(
    args: argparse.Namespace, generate: Any, splits: Any, synthetic_config: Any, synthetic: Any, synonyms: Any
) -> Any:
    if args.kind == "plagiarism":
        # Sources assez longues pour des passages de 300 mots ; assez de documents pour les deux rôles.
        per_work = max(args.docs_per_work, -(-(args.sources + args.suspicious) // (6 * args.authors)) + 1)
        base = generate(
            n_authors=args.authors,
            works_per_author=6,
            docs_per_work=per_work,
            words_per_doc=max(args.words, 350),
            seed=args.seed,
            style_strength=args.style_strength,
        )
        config = synthetic_config(
            n_sources=args.sources, n_suspicious=args.suspicious, passages_per_cell=args.per_cell, seed=args.seed
        )
        return synthetic(base, config, synonyms=synonyms(), name="toy_plagiarism")
    corpus = generate(
        n_authors=args.authors,
        works_per_author=6,
        docs_per_work=args.docs_per_work,
        words_per_doc=args.words,
        seed=args.seed,
        name=f"toy_{args.kind}",
        style_strength=args.style_strength,
    )
    if args.kind == "literary":
        return splits(corpus, {"train": 5 / 6, "test": 1 / 6}, args.seed)
    keyed = corpus.map(lambda d: d.replace(work=None))
    from plagia.corpus import assign_document_splits

    return assign_document_splits(keyed, {"train": 0.5, "test": 0.5}, args.seed)


# annotate
def cmd_annotate(args: argparse.Namespace) -> int:
    """Annote un corpus et remplit le cache (aucun appel externe en mode --offline)."""
    from plagia.annotation.cache import AnnotationCache
    from plagia.annotation.pipeline import annotate_documents
    from plagia.corpus import load_corpus

    if args.config:
        config = load_config(args.config).annotation
    else:
        config = AnnotationConfig(
            engine=args.engine,
            model=args.model,
            cache_path=Path(args.cache),
            linker="spotlight" if args.spotlight_url else "none",
            spotlight_url=args.spotlight_url or "http://localhost:2222/rest",
        )
    corpus = load_corpus(args.corpus)
    start = time.perf_counter()
    with AnnotationCache(config.cache_path) as cache:
        documents = annotate_documents(list(corpus), config, cache=cache, offline=args.offline)
        stats = cache.stats()
    entities = sum(len(d.entities) for d in documents)
    if args.evaluate:
        from plagia.annotation.quality import linking_quality, ner_quality

        gold = [list(d.meta.get("gold_entities", [])) for d in corpus]
        if not any(gold):
            raise PlagiaError("--evaluate : aucune entite de reference (meta.gold_entities), corpus AIDA attendu.")
        quality = {"ner": ner_quality(gold, documents), "linking": linking_quality(gold, documents)}
        print(json.dumps(quality, indent=2, ensure_ascii=False))
    elapsed = time.perf_counter() - start
    print(f"{len(documents)} documents annotés ({entities} entités) en {elapsed:.1f} s ; cache : {stats}")
    return 0


# run / report
def cmd_run(args: argparse.Namespace) -> int:
    """Exécute une expérience décrite par un fichier YAML."""
    from plagia.experiments import Plan, Runner

    config = load_config(args.config)
    if args.output_dir:
        config = config.with_overrides(output_dir=args.output_dir)
    if args.dry_run:
        plan = Plan.from_config(config)
        print(json.dumps(plan.summary(), indent=2))
        for condition in plan.conditions[: args.show]:
            print(("[base] " if condition.baseline else "       ") + condition.id)
        return 0
    store = Runner(config, n_jobs=args.n_jobs, resume=not args.no_resume, limit=args.limit).run()
    print(f"Résultats : {store.directory / 'results.csv'}")
    if args.report:
        from plagia.reporting import generate_report

        print(f"Rapport : {generate_report(store.directory)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Régénère le rapport d'une expérience à partir de ses résultats."""
    from plagia.reporting import generate_report

    path = generate_report(args.results, args.format, args.out)
    print(f"Rapport : {path}")
    return 0


# attribute
def _annotate_texts(items: list[tuple[str, str]], engine: str, cache_path: str | None) -> list[Any]:
    from plagia.annotation.cache import AnnotationCache
    from plagia.annotation.pipeline import annotate_documents
    from plagia.core.types import Document

    cache_file = cache_path or "data/cache/annotations.sqlite"
    config = AnnotationConfig.model_validate({"engine": engine, "cache_path": cache_file})
    cache = AnnotationCache(config.cache_path) if engine != "none" else None
    try:
        return annotate_documents([Document(i, t) for i, t in items], config, cache=cache)
    finally:
        if cache is not None:
            cache.close()


def cmd_attribute(args: argparse.Namespace) -> int:
    """Entraîne un modèle d'attribution ou classe des textes avec un modèle enregistré."""
    from plagia.corpus import load_corpus
    from plagia.tasks.attribution import AuthorshipAttributor

    if args.train:
        corpus = load_corpus(args.train).split("train")
        representation, classifier = DEFAULT_REPRESENTATION, ModelConfig(type="linear_svm")
        if args.config:
            section = load_config(args.config).attribution
            if section is None:
                raise PlagiaError("La configuration doit contenir une section « attribution ».")
            representation = section.representations[args.representation or next(iter(section.representations))]
            classifier = section.classifiers[args.classifier or next(iter(section.classifiers))]
        documents = _annotate_texts([(d.id, d.text) for d in corpus], args.engine, args.cache)
        model = AuthorshipAttributor(representation, classifier).fit(documents, corpus.labels)
        model.engine = args.engine  # type: ignore[attr-defined]
        path = model.save(args.save or "results/models/attribution.joblib")
        print(f"Modèle entraîné sur {len(corpus)} textes ({len(corpus.authors)} auteurs) : {path}")
    if args.files:
        if not (args.model or args.train):
            raise PlagiaError("Indiquez --model (modèle enregistré) ou --train (corpus).")
        _existing_files(args.files)
        model = AuthorshipAttributor.load(args.model) if args.model else model
        engine = getattr(model, "engine", args.engine)
        items = [(str(p), Path(p).read_text(encoding="utf-8")) for p in args.files]
        documents = _annotate_texts(items, engine, args.cache)
        for (name, _), ranking in zip(items, model.rank(documents, args.top_k), strict=True):
            ranked = ", ".join(f"{author} ({score:.2f})" for author, score in ranking)
            print(f"{name}: {ranked}")
    return 0


# detect
def _load_sources(path: str) -> list[Any]:
    from plagia.core.types import Document
    from plagia.corpus import load_corpus

    source = Path(path)
    if (source / "documents.jsonl").exists():
        corpus = load_corpus(source)
        sources = [d for d in corpus if d.meta.get("role", "source") == "source"]
        return sources or list(corpus)
    if not source.exists():
        raise CorpusNotFoundError(source)
    files = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    documents = [Document(p.stem, p.read_text(encoding="utf-8")) for p in files]
    if not documents:
        raise PlagiaError(f"Aucune source dans {source} : ni corpus (documents.jsonl) ni fichiers .txt.")
    return documents


def _existing_files(paths: Sequence[str]) -> list[Path]:
    """Vérifie les fichiers d'entrée avant tout calcul coûteux."""
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise PlagiaError("Fichier(s) introuvable(s) : " + ", ".join(missing))
    return [Path(p) for p in paths]


def cmd_detect(args: argparse.Namespace) -> int:
    """Détecte les passages repris dans des fichiers suspects et produit un rapport HTML."""
    import joblib

    from plagia.core.io import write_json
    from plagia.reporting import render_similarity_report
    from plagia.tasks.plagiarism import (
        CandidateRetriever,
        Extender,
        ExtrinsicDetector,
        Filter,
        IntrinsicDetector,
        SeedMatcher,
        TextAligner,
        build_report,
    )

    _existing_files(args.files)
    sources = _load_sources(args.sources)
    source_docs = _annotate_texts([(d.id, d.text) for d in sources], args.engine, args.cache)
    pair_classifier = joblib.load(args.pair_model) if args.pair_model else None
    aligner = TextAligner(
        SeedMatcher(ngram_size=args.ngram_size),
        Extender(min_density=args.min_density),
        Filter(args.min_length),
        verifier=pair_classifier.predict_proba if pair_classifier else None,
        threshold=pair_classifier.threshold if pair_classifier else 0.5,
    )
    detector = ExtrinsicDetector(CandidateRetriever(args.retrieval, k=args.k), aligner).fit(source_docs)
    texts = {d.id: d.text for d in sources}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        text = Path(path).read_text(encoding="utf-8")
        document = _annotate_texts([(Path(path).stem, text)], args.engine, args.cache)[0]
        detections = detector.detect(document)
        if args.intrinsic:
            detections += IntrinsicDetector(args.window, args.window // 2).detect(document)
        report = build_report(document.id, text, detections)
        stem = Path(path).stem
        (out_dir / f"{stem}.html").write_text(render_similarity_report(report, text, texts), encoding="utf-8")
        write_json(out_dir / f"{stem}.json", report.to_dict())
        summary = f"score {report.score:.2f}, couverture {100 * report.coverage:.1f} %, {len(detections)} passage(s)"
        print(f"{path}: {summary} → {out_dir / (stem + '.html')}")
    return 0


# demo
def cmd_demo(args: argparse.Namespace) -> int:
    """Démonstrateur : page HTML pour un texte (--text) ou application Streamlit (--serve)."""
    import importlib.util

    if not (args.model or args.sources):
        raise PlagiaError("Indiquez --model (modèle d'attribution), --sources (collection de sources), ou les deux.")
    if args.model and not Path(args.model).is_file():
        raise PlagiaError(
            f"Modèle introuvable : {args.model}. "
            f"Entraînez-le avec « plagia attribute --train <corpus> --save {args.model} »."
        )
    if args.sources and not Path(args.sources).exists():
        raise CorpusNotFoundError(args.sources)
    if args.serve:
        if importlib.util.find_spec("streamlit") is None:
            raise MissingDependencyError("streamlit", "démonstrateur interactif", "Lancez « uv sync --extra demo ».")
        forwarded = [f"--top-k={args.top_k}"]
        for option in ("model", "sources", "engine", "cache"):
            if getattr(args, option):
                forwarded.append(f"--{option}={getattr(args, option)}")
        app = Path(__file__).with_name("demo_app.py")
        command = [sys.executable, "-m", "streamlit", "run", str(app), "--browser.gatherUsageStats=false"]
        return subprocess.call([*command, "--", *forwarded])
    if not args.text:
        raise PlagiaError("Indiquez --text (page HTML pour un fichier) ou --serve (application Streamlit).")
    from plagia.reporting.demo import DemoAnalyzer

    _existing_files([args.text])
    sources = _load_sources(args.sources) if args.sources else []
    analyzer = DemoAnalyzer.from_paths(args.model, sources, engine=args.engine, cache_path=args.cache, top_k=args.top_k)
    text_path = Path(args.text)
    result = analyzer.analyze(text_path.read_text(encoding="utf-8"), text_path.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_html(), encoding="utf-8")
    if result.ranking:
        print("Auteurs : " + ", ".join(f"{author} ({score:.2f})" for author, score in result.ranking))
    if result.report is not None:
        print(f"Couverture {100 * result.report.coverage:.1f} %, {len(result.report.detections)} passage(s)")
    print(f"Page : {out}")
    return 0


# cluster
def cmd_cluster(args: argparse.Namespace) -> int:
    """Regroupe les documents d'un corpus et évalue contre les auteurs (et les œuvres)."""
    from plagia.core.io import write_json
    from plagia.corpus import load_corpus
    from plagia.evaluation import clustering_metrics
    from plagia.features import build_representation
    from plagia.models import build_clusterer
    from plagia.tasks.clustering import characterize_clusters, choose_k

    corpus = load_corpus(args.corpus)
    if args.split:
        corpus = corpus.split(args.split)
    documents = _annotate_texts([(d.id, d.text) for d in corpus], args.engine, args.cache)
    blocks = [BlockConfig(type=b) for b in args.blocks]
    union = build_representation(blocks)
    X = union.fit_transform(documents)
    if args.k == "true":
        k = len(corpus.authors)
    elif args.k == "auto":
        k = choose_k(X, range(2, min(30, len(corpus) - 1)), args.dimensions)["best_silhouette"] or 2
    else:
        k = int(args.k)
    clusterer = build_clusterer(args.algorithm, n_clusters=k, n_components=args.dimensions)
    labels = clusterer.fit_predict(X)
    result = {"k": k, "author": clustering_metrics(corpus.labels, labels, clusterer.embedding_)}
    if any(d.work for d in corpus):
        result["work"] = clustering_metrics(corpus.groups, labels)
    result["clusters"] = {
        str(k): [f for f, _ in v[:10]]
        for k, v in characterize_clusters(X, labels, list(union.get_feature_names_out()), 10).items()
    }
    print(json.dumps({k: v for k, v in result.items() if k != "clusters"}, indent=2, ensure_ascii=False))
    if args.out:
        write_json(
            Path(args.out) / "clusters.json",
            {**result, "labels": {d.id: int(c) for d, c in zip(corpus, labels, strict=True)}},
        )
    return 0


# bench
def cmd_bench(args: argparse.Namespace) -> int:
    """Contrôle de performance : TF-IDF de mots et de caractères sur un corpus synthétique."""
    from plagia.corpus.toy import generate_toy_corpus
    from plagia.features import CharNgrams, Tfidf

    per_author = max(1, -(-args.documents // (6 * args.authors)))
    corpus = generate_toy_corpus(
        n_authors=args.authors, works_per_author=6, docs_per_work=per_author, words_per_doc=args.words, seed=0
    )
    texts = corpus.texts[: args.documents]
    start = time.perf_counter()
    words = Tfidf(stopwords_source="builtin").fit_transform(texts)
    middle = time.perf_counter()
    chars = CharNgrams().fit_transform(texts)
    end = time.perf_counter()
    print(
        json.dumps(
            {
                "documents": len(texts),
                "words_per_document": args.words,
                "tfidf_words_s": round(middle - start, 2),
                "tfidf_chars_s": round(end - middle, 2),
                "total_s": round(end - start, 2),
                "word_features": words.shape[1],
                "char_features": chars.shape[1],
                "objective_s": 120,
            },
            indent=2,
        )
    )
    return 0


# parser
def build_parser() -> argparse.ArgumentParser:
    """Analyseur des arguments de `plagia`."""
    parser = argparse.ArgumentParser(prog="plagia", description="Analyse d'identité d'auteurs et détection de plagiat.")
    parser.add_argument("--version", action="version", version=f"plagia {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="journal détaillé")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="ressources NLTK, modèles spaCy, vérification de Java et de Spotlight")
    setup.add_argument("--spacy", nargs="*", default=["en_core_web_sm"], help="modèles spaCy à installer")
    setup.add_argument(
        "--nltk", nargs="*", default=["punkt_tab", "stopwords", "wordnet", "omw-1.4"], help="ressources NLTK"
    )
    setup.add_argument("--spotlight-url", default=None, help="point d'accès Spotlight à tester")
    setup.add_argument("--check-only", action="store_true", help="vérifier sans rien installer")
    setup.set_defaults(func=cmd_setup)

    corpus = sub.add_parser("corpus", help="construction et vérification des corpus")
    csub = corpus.add_subparsers(dest="corpus_command", required=True)
    info = csub.add_parser("info", help="effectifs, manifeste et absence de fuite")
    info.add_argument("path")
    c50 = csub.add_parser("c50", help="Reuters C50 (UCI)")
    c50.add_argument("--source", required=True, help="archive zip ou répertoire extrait")
    c50.add_argument("--download", action="store_true", help="télécharger l'archive UCI vers --source")
    c50.add_argument("--validation", type=float, default=0.0, help="part de l'entraînement réservée à la validation")
    gut = csub.add_parser("gutenberg", help="corpus littéraire Project Gutenberg")
    gut.add_argument("--config", required=True)
    gut.add_argument("--raw", default="data/raw/gutenberg", help="cache des livres téléchargés")
    pan = csub.add_parser("pan", help="corpus PAN d'alignement de textes (2013, 2014, PAN-PC-11)")
    pan.add_argument("--source", required=True)
    pan.add_argument(
        "--split", default="test", help="jeu des documents de --source (train pour un corpus d'entraînement)"
    )
    pan.add_argument("--test", default=None, help="corpus PAN de test à ajouter (jeu test)")
    pan.add_argument("--validation", type=float, default=0.0, help="part de l'entraînement mise en validation")
    pan.add_argument("--name", default=None)
    aida = csub.add_parser("aida", help="AIDA CoNLL-YAGO (fichier fusionné)")
    aida.add_argument("--tsv", required=True)
    aida.add_argument("--conll", default=None, help="fichier CoNLL 2003 pour les types d'entités")
    syn = csub.add_parser("synthetic", help="corpus de plagiat synthétique")
    syn.add_argument("--base", required=True, help="corpus de base (répertoire)")
    syn.add_argument("--config", default=None, help="paramètres YAML (SyntheticConfig)")
    syn.add_argument("--n-sources", dest="n_sources", type=int, default=None)
    syn.add_argument("--n-suspicious", dest="n_suspicious", type=int, default=None)
    syn.add_argument("--passages-per-cell", dest="passages_per_cell", type=int, default=None)
    syn.add_argument("--synonyms", choices=["wordnet", "toy"], default="wordnet")
    syn.add_argument("--name", default=None)
    toy = csub.add_parser("toy", help="corpus jouet hors ligne (démonstrations, tests)")
    toy.add_argument("--kind", choices=["literary", "news", "plagiarism"], default="literary")
    toy.add_argument("--authors", type=int, default=8)
    toy.add_argument("--docs-per-work", dest="docs_per_work", type=int, default=8)
    # 150 mots : assez court pour que l'attribution ne sature pas (démonstrations calibrées ainsi).
    toy.add_argument("--words", type=int, default=150, help="longueur minimale des textes (literary, news)")
    toy.add_argument("--sources", type=int, default=200)
    toy.add_argument("--suspicious", type=int, default=240)
    toy.add_argument("--per-cell", dest="per_cell", type=int, default=15)
    toy.add_argument(
        "--style-strength", dest="style_strength", type=float, default=0.3, help="écart des styles d'auteurs (0 à 1)"
    )
    export = csub.add_parser("export", help="exporte des documents en fichiers .txt (exemples, essais)")
    export.add_argument("path", help="répertoire du corpus")
    export.add_argument("--out", required=True, help="répertoire de sortie")
    export.add_argument("--split", default=None, help="train, validation ou test")
    export.add_argument("--role", choices=["source", "suspicious"], default=None, help="corpus de plagiat")
    export.add_argument("--ids", nargs="+", default=None, help="identifiants à exporter")
    export.add_argument(
        "--with-plagiarism", dest="with_plagiarism", action="store_true", help="seulement les textes avec plagiat"
    )
    export.add_argument("--limit", type=int, default=None)
    clean = csub.add_parser("clean", help="nouvelle version nettoyée d'un corpus")
    clean.add_argument("--source", required=True)
    clean.add_argument("--keep-signatures", action="store_true")
    clean.add_argument("--keep-author-mentions", action="store_true")
    for p in (c50, gut, pan, aida, syn, toy, clean):
        p.add_argument("--out", required=True, help="répertoire du corpus produit")
        p.add_argument("--seed", type=int, default=42)
    corpus.set_defaults(func=cmd_corpus)

    annotate = sub.add_parser("annotate", help="annotation d'un corpus (cache SQLite)")
    annotate.add_argument("corpus")
    annotate.add_argument("--config", default=None, help="configuration d'expérience (section annotation)")
    annotate.add_argument("--engine", choices=["spacy", "stanford", "heuristic", "none"], default="spacy")
    annotate.add_argument("--model", default="en_core_web_sm")
    annotate.add_argument("--cache", default="data/cache/annotations.sqlite")
    annotate.add_argument("--spotlight-url", default=None)
    annotate.add_argument("--offline", action="store_true", help="échouer plutôt que d'appeler un outil")
    annotate.add_argument(
        "--evaluate", action="store_true", help="qualité de la reconnaissance et du liage (AIDA, F3.7)"
    )
    annotate.set_defaults(func=cmd_annotate)

    run = sub.add_parser("run", help="exécuter une expérience (YAML)")
    run.add_argument("config")
    run.add_argument("--n-jobs", type=int, default=None)
    run.add_argument("--no-resume", action="store_true", help="tout recalculer")
    run.add_argument("--limit", type=int, default=None, help="nombre maximal d'exécutions")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--dry-run", action="store_true", help="afficher le plan sans exécuter")
    run.add_argument("--show", type=int, default=20, help="conditions affichées avec --dry-run")
    run.add_argument("--report", action="store_true", help="générer le rapport à la fin")
    run.set_defaults(func=cmd_run)

    attribute = sub.add_parser("attribute", help="attribution d'auteur de textes")
    attribute.add_argument("files", nargs="*")
    attribute.add_argument("--model", default=None, help="modèle enregistré (joblib)")
    attribute.add_argument("--train", default=None, help="corpus d'entraînement (jeu train)")
    attribute.add_argument("--save", default=None, help="chemin du modèle entraîné")
    attribute.add_argument("--config", default=None, help="configuration d'attribution")
    attribute.add_argument("--representation", default=None)
    attribute.add_argument("--classifier", default=None)
    attribute.add_argument("--top-k", type=int, default=3)
    attribute.add_argument("--engine", choices=["spacy", "heuristic", "none"], default="none")
    attribute.add_argument("--cache", default=None)
    attribute.set_defaults(func=cmd_attribute)

    detect = sub.add_parser("detect", help="détection de passages repris")
    detect.add_argument("files", nargs="+", help="fichiers suspects (.txt)")
    detect.add_argument("--sources", required=True, help="corpus ou répertoire de fichiers .txt sources")
    detect.add_argument("--out", default="results/detect")
    detect.add_argument("--pair-model", default=None, help="classificateur de paires enregistré (joblib)")
    detect.add_argument("--retrieval", choices=["tfidf", "shingles", "minhash"], default="tfidf")
    detect.add_argument("--k", type=int, default=10)
    detect.add_argument("--ngram-size", type=int, default=4)
    detect.add_argument("--min-density", type=float, default=1.0)
    detect.add_argument("--min-length", type=int, default=150)
    detect.add_argument("--intrinsic", action="store_true", help="ajouter la détection intrinsèque")
    detect.add_argument("--window", type=int, default=500)
    detect.add_argument("--engine", choices=["spacy", "heuristic", "none"], default="none")
    detect.add_argument("--cache", default=None)
    detect.set_defaults(func=cmd_detect)

    cluster = sub.add_parser("cluster", help="regroupement de documents")
    cluster.add_argument("corpus")
    cluster.add_argument("--split", default=None)
    cluster.add_argument("--blocks", nargs="+", default=["tfidf"])
    cluster.add_argument("--algorithm", choices=["kmeans", "agglomerative", "dbscan", "hdbscan"], default="kmeans")
    cluster.add_argument("--k", default="true", help="« true » (nombre d'auteurs), « auto » (silhouette) ou entier")
    cluster.add_argument("--dimensions", type=int, default=100)
    cluster.add_argument("--engine", choices=["spacy", "heuristic", "none"], default="none")
    cluster.add_argument("--cache", default=None)
    cluster.add_argument("--out", default=None)
    cluster.set_defaults(func=cmd_cluster)

    demo = sub.add_parser("demo", help="démonstrateur : auteurs probables et passages repris (F9.5)")
    demo.add_argument("--model", default=None, help="modèle d'attribution enregistré (plagia attribute --save)")
    demo.add_argument("--sources", default=None, help="corpus ou répertoire de fichiers .txt sources")
    demo.add_argument("--text", default=None, help="fichier texte à analyser (page HTML autonome)")
    demo.add_argument("--out", default="results/demo/demo.html")
    demo.add_argument("--serve", action="store_true", help="application Streamlit (uv sync --extra demo)")
    demo.add_argument("--engine", choices=["spacy", "heuristic", "none"], default=None)
    demo.add_argument("--cache", default=None)
    demo.add_argument("--top-k", type=int, default=5)
    demo.set_defaults(func=cmd_demo)

    report = sub.add_parser("report", help="rapport d'évaluation d'une expérience")
    report.add_argument("results", help="répertoire de résultats de l'expérience")
    report.add_argument("--format", choices=["html", "md"], default="html")
    report.add_argument("--out", default=None)
    report.set_defaults(func=cmd_report)

    bench = sub.add_parser("bench", help="contrôle de performance des représentations TF-IDF")
    bench.add_argument("--documents", type=int, default=5000)
    bench.add_argument("--words", type=int, default=500)
    bench.add_argument("--authors", type=int, default=50)
    bench.set_defaults(func=cmd_bench)
    return parser


# Commandes de construction des corpus attendus par les configurations du dépôt.
CORPUS_RECIPES = {
    "c50": "plagia corpus c50 --source data/raw/c50.zip --download --validation 0.2 --out data/corpora/c50",
    "literary": "plagia corpus gutenberg --config configs/corpus/gutenberg.yaml --out data/corpora/literary",
    "synthetic": (
        "plagia corpus synthetic --base data/corpora/c50 --config configs/corpus/synthetic.yaml "
        "--out data/corpora/synthetic (après C50)"
    ),
    "pan13": (
        "plagia corpus pan --source data/raw/pan13-train --split train --test data/raw/pan13-test "
        "--validation 0.2 --name pan13 --out data/corpora/pan13"
    ),
    "aida": (
        "plagia corpus aida --tsv data/raw/AIDA-YAGO2-dataset.tsv --conll data/raw/conll2003.txt "
        "--out data/corpora/aida"
    ),
    "toy_literary": "plagia corpus toy --kind literary --out data/corpora/toy_literary",
    "toy_news": "plagia corpus toy --kind news --out data/corpora/toy_news",
    "toy_plagiarism": "plagia corpus toy --kind plagiarism --out data/corpora/toy_plagiarism",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée de la commande `plagia`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO")
    try:
        code: int = args.func(args)
        return code
    except CorpusNotFoundError as exc:
        log.error("%s", exc)
        log.error(
            "Construisez-le d'abord : %s", CORPUS_RECIPES.get(exc.path.name, "plagia corpus … (voir data/README.md)")
        )
        return 2
    except PlagiaError as exc:
        log.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("Fichier introuvable : %s", exc.filename or exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
