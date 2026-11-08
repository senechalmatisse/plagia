"""Typage par bases de connaissances : ontologie DBpedia et taxonomie YAGO.

Le type d'une entité liée s'obtient dans la hiérarchie de classes à une profondeur réglable. Après typage, `Entity.types` contient tous les ancêtres des types de
l'entité sous la forme `"profondeur:classe"` (`"0:owl:Thing"`, `"1:dbo:Agent"`, `"2:dbo:Person"`) : les caractéristiques choisissent la profondeur sans avoir besoin
de la hiérarchie.

Hiérarchies lisibles : TSV `enfant<TAB>parent`, N-Triples (`rdfs:subClassOf`, analyseur intégré) et tout format RDF via rdflib (Turtle, RDF/XML de l'ontologie
DBpedia, taxonomie YAGO 4.6). Les types des entités YAGO sont obtenus par SPARQL (réponses mises en cache) ; les faits de YAGO ne sont pas chargés.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from plagia.annotation.cache import AnnotationCache, CacheMissError
from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.core.errors import ExternalServiceError, MissingDependencyError

RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
PREFIXES = {
    "http://dbpedia.org/ontology/": "dbo",
    "http://schema.org/": "schema",
    "https://schema.org/": "schema",
    "http://yago-knowledge.org/resource/": "yago",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://www.wikidata.org/entity/": "wikidata",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#": "dul",
}
_NT_RE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$")

HttpGet = Callable[[str], bytes]


def compact(iri: str) -> str:
    """IRI complète → nom préfixé (``http://dbpedia.org/ontology/Person`` → ``dbo:Person``)."""
    for base, prefix in PREFIXES.items():
        if iri.startswith(base):
            return f"{prefix}:{iri[len(base) :]}"
    return iri


class TypeHierarchy:
    """Hiérarchie de classes (graphe enfant → parents, éventuellement multiple)."""

    def __init__(self, parents: Mapping[str, Iterable[str]], root: str = "owl:Thing") -> None:
        self.parents: dict[str, list[str]] = {c: sorted(set(p)) for c, p in parents.items()}
        self.root = root
        self.classes: set[str] = {root, *self.parents, *(p for ps in self.parents.values() for p in ps)}
        self._paths: dict[str, list[str]] = {}

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]], root: str = "owl:Thing") -> TypeHierarchy:
        """Construit la hiérarchie depuis des couples ``(enfant, parent)``."""
        table: dict[str, set[str]] = defaultdict(set)
        for child, parent in edges:
            if child != parent:
                table[compact(child)].add(compact(parent))
        return cls(table, root)

    @classmethod
    def from_tsv(cls, path: str | Path, root: str = "owl:Thing") -> TypeHierarchy:
        """Fichier `enfant<TAB>parent` (lignes `#` ignorées)."""
        edges = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                child, parent = line.split("\t")[:2]
                edges.append((child.strip(), parent.strip()))
        return cls.from_edges(edges, root)

    @classmethod
    def from_ntriples(cls, path: str | Path, predicate: str = RDFS_SUBCLASS, root: str = "owl:Thing") -> TypeHierarchy:
        """Triplets N-Triples `<enfant> rdfs:subClassOf <parent>` (analyseur intégré)."""
        edges = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                match = _NT_RE.match(line.strip())
                if match and match.group(2) == predicate:
                    edges.append((match.group(1), match.group(3)))
        return cls.from_edges(edges, root)

    @classmethod
    def from_rdf(cls, path: str | Path, rdf_format: str | None = None, root: str = "owl:Thing") -> TypeHierarchy:
        """Tout format RDF lisible par rdflib (Turtle, RDF/XML, N3…)."""
        try:
            from rdflib import Graph, URIRef
            from rdflib.namespace import RDFS
        except ImportError as exc:
            raise MissingDependencyError("rdflib", "lecture d'ontologies RDF", "Lancez « uv sync ».") from exc
        graph = Graph()
        graph.parse(str(path), format=rdf_format)
        edges = [
            (str(s), str(o))
            for s, o in graph.subject_objects(RDFS.subClassOf)
            if isinstance(s, URIRef) and isinstance(o, URIRef)
        ]
        return cls.from_edges(edges, root)

    @classmethod
    def load(cls, path: str | Path, root: str = "owl:Thing") -> TypeHierarchy:
        """Choisit le lecteur selon l'extension (`.tsv`, `.nt`, autres via rdflib)."""
        suffix = Path(path).suffix.lower()
        if suffix in {".tsv", ".txt"}:
            return cls.from_tsv(path, root=root)
        if suffix == ".nt":
            return cls.from_ntriples(path, root=root)
        return cls.from_rdf(path, root=root)

    def __contains__(self, cls: object) -> bool:
        return cls in self.classes

    def path_to_root(self, cls: str) -> list[str]:
        """Plus court chemin de la racine jusqu'à `cls` (inclus)."""
        if cls in self._paths:
            return self._paths[cls]
        previous: dict[str, str | None] = {cls: None}
        queue: deque[str] = deque([cls])
        end = cls
        while queue:
            node = queue.popleft()
            parents = self.parents.get(node, [])
            if not parents or node == self.root:
                end = node
                break
            for parent in parents:
                if parent not in previous:
                    previous[parent] = node
                    queue.append(parent)
        path = [end]
        while (child := previous[path[-1]]) is not None:
            path.append(child)
        if path[0] != self.root:
            path.insert(0, self.root)
        self._paths[cls] = path
        return path

    def depth(self, cls: str) -> int:
        """Profondeur de `cls` (0 pour la racine)."""
        return len(self.path_to_root(cls)) - 1

    def ancestors(self, cls: str) -> list[str]:
        """Tous les ancêtres de `cls` (parcours en largeur)."""
        seen: list[str] = []
        queue = deque(self.parents.get(cls, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.append(node)
            queue.extend(self.parents.get(node, []))
        return seen

    def at_depth(self, cls: str, depth: int) -> str:
        """Ancêtre de `cls` à la profondeur `depth` (`cls` s'il est moins profond)."""
        path = self.path_to_root(cls)
        return path[min(depth, len(path) - 1)]

    def typed_path(self, classes: Iterable[str]) -> tuple[str, ...]:
        """Classes et ancêtres sous la forme `"profondeur:classe"` (triés)."""
        result: set[str] = set()
        for cls in classes:
            for depth, node in enumerate(self.path_to_root(cls)):
                result.add(f"{depth}:{node}")
        return tuple(sorted(result, key=lambda s: (int(s.partition(":")[0]), s)))


class KnowledgeBase(ABC):
    """Base de connaissances : types d'une entité liée et hiérarchie de classes."""

    name = "kb"

    def __init__(self, hierarchy: TypeHierarchy) -> None:
        self.hierarchy = hierarchy

    @abstractmethod
    def types_of(self, entity: Entity) -> list[str]:
        """Classes (noms préfixés) de l'entité."""

    def type_entity(self, entity: Entity) -> Entity:
        """Entité dont ``types`` contient ses classes et leurs ancêtres avec leur profondeur."""
        classes = [c for c in self.types_of(entity) if c in self.hierarchy]
        if not classes:
            return Entity(
                entity.text, entity.start, entity.end, entity.label, entity.kind, entity.uri, (), entity.score
            )
        return Entity(
            entity.text,
            entity.start,
            entity.end,
            entity.label,
            entity.kind,
            entity.uri,
            self.hierarchy.typed_path(classes),
            entity.score,
        )

    def type_document(self, doc: AnnotatedDocument) -> AnnotatedDocument:
        """Type toutes les entités liées du document."""
        return doc.replace(entities=tuple(self.type_entity(e) if e.uri else e for e in doc.entities))


class DbpediaKnowledgeBase(KnowledgeBase):
    """Ontologie DBpedia : types fournis par Spotlight (``raw:dbo:…``), hiérarchie locale."""

    name = "dbpedia"

    def types_of(self, entity: Entity) -> list[str]:
        return [t.removeprefix("raw:") for t in entity.types if t.startswith("raw:dbo:")]


def _default_get(timeout: int) -> HttpGet:
    def get(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: bytes = response.read()
            return body

    return get


class YagoKnowledgeBase(KnowledgeBase):
    """Taxonomie YAGO 4.6 : types des entités par SPARQL (mis en cache), hiérarchie locale.

    L'URI DBpedia `http://dbpedia.org/resource/X` est rapprochée de la ressource YAGO
    `http://yago-knowledge.org/resource/X` (même titre Wikipedia).
    """

    name = "yago"

    def __init__(
        self,
        hierarchy: TypeHierarchy,
        endpoint: str = "https://yago-knowledge.org/sparql/query",
        cache: AnnotationCache | None = None,
        offline: bool = False,
        timeout: int = 60,
        http_get: HttpGet | None = None,
    ) -> None:
        super().__init__(hierarchy)
        self.endpoint = endpoint
        self.cache = cache
        self.offline = offline
        self.http_get = http_get or _default_get(timeout)

    @staticmethod
    def yago_iri(uri: str) -> str:
        """IRI YAGO correspondant à une URI DBpedia."""
        return "http://yago-knowledge.org/resource/" + uri.rstrip("/").rsplit("/", 1)[-1]

    def types_of(self, entity: Entity) -> list[str]:
        if not entity.uri:
            return []
        iri = self.yago_iri(entity.uri)
        query = f"SELECT DISTINCT ?t WHERE {{ <{iri}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t }}"
        key = AnnotationCache.key(query, "yago-sparql", self.endpoint, {})
        payload = self.cache.get_response(key) if self.cache else None
        if payload is None:
            if self.offline:
                raise CacheMissError("Réponse SPARQL absente du cache (mode hors ligne).")
            url = f"{self.endpoint}?{urllib.parse.urlencode({'query': query})}"
            try:
                payload = self.http_get(url).decode("utf-8")
            except OSError as exc:
                raise ExternalServiceError(f"Point d'accès SPARQL injoignable : {exc}") from exc
            if self.cache:
                self.cache.put_response(key, "yago-sparql", payload)
        bindings = json.loads(payload).get("results", {}).get("bindings", [])
        return sorted({compact(b["t"]["value"]) for b in bindings if "t" in b})
