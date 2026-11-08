"""Corpus jouet multi-auteurs généré hors ligne (démonstrations, tests, intégration continue).

Chaque auteur reçoit un style tiré de la graine (connecteurs favoris, longueur et structure des phrases, ponctuation, adjectifs et verbes préférés) et des entités qui
lui sont propres (personnes, lieux, organisations), à la manière d'un journaliste qui couvre ses propres sociétés. Chaque auteur écrit plusieurs œuvres, chacune centrée
sur un sujet : on peut ainsi reproduire en miniature le biais de sujet (H2, H5).

Ce corpus n'est PAS un substitut aux corpus C50 et Gutenberg : il permet seulement d'exécuter toute la chaîne sans téléchargement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from plagia.core.seeds import make_random
from plagia.core.text import count_words
from plagia.core.types import Corpus, Document
from plagia.corpus.obfuscation import DictSynonyms

TOPICS: dict[str, list[str]] = {
    "sea": "ship harbour captain sailor storm voyage island wave coast anchor tide cargo".split(),
    "trade": "merchant market price contract bank profit debt ledger warehouse customer shop coin".split(),
    "war": "soldier battle army general sword fortress siege regiment enemy victory camp banner".split(),
    "law": "judge court lawyer trial witness verdict crime prison letter will estate inheritance".split(),
    "science": (
        "doctor experiment machine laboratory theory engine instrument discovery patient fever telescope chemist"
    ).split(),
    "country": "farmer village field horse garden orchard harvest cottage river meadow church mill".split(),
    "city": "street carriage office clerk newspaper theatre crowd lamp bridge factory station square".split(),
    "family": "mother daughter son cousin widow servant marriage wedding house parlour dinner sister".split(),
}
ADJECTIVES = (
    "old young quiet dark bright strange cold warm grave gentle proud poor rich silent small "
    "great ancient curious weary honest cruel pale solemn merry anxious eager narrow wide "
    "bitter sweet heavy faithful restless distant noble humble"
).split()
TRANSITIVE = (
    "watched visited admired feared followed found left met remembered described carried "
    "defended praised questioned studied sold bought built crossed opened promised refused "
    "received sent wrote ignored welcomed betrayed trusted guarded"
).split()
INTRANSITIVE = "waited smiled hesitated laughed wept sighed paused returned departed slept wandered listened".split()
ADVERBS = (
    "slowly quickly silently gravely gently suddenly certainly carefully eagerly warmly "
    "coldly openly firmly bitterly calmly patiently"
).split()
CONNECTIVES = (
    "However Moreover Indeed Meanwhile Still Therefore Yet Then Nevertheless Thus Besides "
    "Accordingly Perhaps Naturally Afterwards"
).split()
SUBORDINATORS = "because although while when since whereas though until as".split()
COORDINATORS = "and but or yet so".split()
PREPOSITIONS = "in near beyond across beside behind toward within upon along among".split()
DETERMINERS = "the a this that every each his her their some".split()
TITLES = ["Mr.", "Mrs.", "Miss", "Dr.", "Captain", "Lord", "Lady", "Colonel", "Sir"]
FIRST_NAMES = (
    "Arthur Edmund Clara Henry Lucy Thomas Margaret Walter Agnes Robert Eleanor Frederick "
    "Harriet George Charlotte William Beatrice Oliver Matilda Samuel Isabel James Ada Victor "
    "Emma Hugh Florence Lewis Rose Albert"
).split()
LAST_NAMES = (
    "Ashby Thornton Mallory Whitcombe Fairfax Pembroke Harlow Kendrick Ravensworth Bellamy "
    "Carrow Dunmore Everett Grantham Hollis Langley Merriman Norcott Penrose Quinton "
    "Rutherford Stanhope Talbot Underhill Vance Westbrook Yardley Ainsworth Blackwood Crane"
).split()
PLACES = (
    "Fenwick Kingsbridge Ravenscar Ellesmere Holloway Westford Brightwater Ashcombe Stonehaven "
    "Northfield Greywater Lindham Oakridge Thornbury Wexley Carden Duncliffe Halford Marlow Eastmoor"
).split()
ORG_SUFFIXES = ["Company", "Bank", "Society", "Railway", "Mills", "Gazette", "Institute", "Club"]

TOY_SYNONYMS: dict[str, list[str]] = {
    "old": ["aged", "elderly"],
    "quiet": ["calm", "still"],
    "dark": ["dim", "gloomy"],
    "strange": ["odd", "peculiar"],
    "cold": ["chilly", "frigid"],
    "warm": ["mild", "tepid"],
    "grave": ["serious"],
    "gentle": ["tender", "kindly"],
    "proud": ["haughty", "vain"],
    "poor": ["needy", "destitute"],
    "rich": ["wealthy", "affluent"],
    "silent": ["mute", "noiseless"],
    "small": ["little", "tiny"],
    "great": ["large", "vast"],
    "curious": ["inquisitive"],
    "weary": ["tired", "fatigued"],
    "honest": ["sincere", "truthful"],
    "cruel": ["harsh", "brutal"],
    "pale": ["wan", "pallid"],
    "merry": ["cheerful", "jolly"],
    "anxious": ["uneasy", "worried"],
    "eager": ["keen", "avid"],
    "bright": ["brilliant", "radiant"],
    "young": ["youthful"],
    "watched": ["observed", "viewed"],
    "admired": ["esteemed", "respected"],
    "feared": ["dreaded"],
    "followed": ["pursued", "trailed"],
    "found": ["discovered", "located"],
    "met": ["encountered"],
    "remembered": ["recalled", "recollected"],
    "described": ["depicted", "portrayed"],
    "carried": ["bore", "conveyed"],
    "defended": ["protected", "shielded"],
    "questioned": ["interrogated", "queried"],
    "studied": ["examined", "analysed"],
    "bought": ["purchased", "acquired"],
    "built": ["constructed", "erected"],
    "crossed": ["traversed"],
    "refused": ["declined", "rejected"],
    "received": ["obtained"],
    "sent": ["dispatched", "forwarded"],
    "wrote": ["penned", "composed"],
    "slowly": ["gradually", "leisurely"],
    "quickly": ["rapidly", "swiftly"],
    "silently": ["noiselessly"],
    "suddenly": ["abruptly", "unexpectedly"],
    "carefully": ["cautiously", "attentively"],
    "ship": ["vessel", "boat"],
    "harbour": ["port", "haven"],
    "storm": ["tempest", "gale"],
    "voyage": ["journey", "crossing"],
    "merchant": ["trader", "dealer"],
    "market": ["bazaar"],
    "price": ["cost", "charge"],
    "soldier": ["trooper", "warrior"],
    "battle": ["fight", "combat"],
    "army": ["troops"],
    "judge": ["magistrate"],
    "lawyer": ["attorney", "counsel"],
    "trial": ["hearing"],
    "crime": ["offence", "felony"],
    "doctor": ["physician", "surgeon"],
    "machine": ["device", "apparatus"],
    "discovery": ["finding", "revelation"],
    "farmer": ["peasant", "grower"],
    "village": ["hamlet"],
    "field": ["pasture"],
    "house": ["home", "dwelling"],
    "street": ["road", "avenue"],
    "crowd": ["throng", "multitude"],
    "servant": ["maid", "attendant"],
    "letter": ["note", "message"],
    "estate": ["property", "manor"],
    "river": ["stream", "brook"],
    "waited": ["lingered"],
    "smiled": ["grinned"],
    "laughed": ["chuckled"],
    "wept": ["cried", "sobbed"],
}


def toy_synonyms() -> DictSynonyms:
    """Fournisseur de synonymes couvrant le vocabulaire du corpus jouet (obfuscation O2 hors ligne)."""
    return DictSynonyms(TOY_SYNONYMS)


def _weighted(rng: random.Random, items: list[str], k: int) -> dict[str, float]:
    chosen = rng.sample(items, min(k, len(items)))
    return {item: rng.uniform(1.0, 4.0) for item in chosen}


def _pick(rng: random.Random, items: list[str], favourites: dict[str, float], bias: float) -> str:
    if favourites and rng.random() < bias:
        names = list(favourites)
        return rng.choices(names, weights=[favourites[n] for n in names])[0]
    return rng.choice(items)


@dataclass
class AuthorStyle:
    """Paramètres stylistiques et thématiques d'un auteur du corpus jouet."""

    name: str
    connective_prob: float
    subordinate_prob: float
    coordinate_prob: float
    semicolon_prob: float
    dash_prob: float
    exclaim_prob: float
    question_prob: float
    adjective_prob: float
    adverb_prob: float
    entity_prob: float
    title_prob: float
    bias: float
    connectives: dict[str, float]
    adjectives: dict[str, float]
    verbs: dict[str, float]
    prepositions: dict[str, float]
    determiners: dict[str, float]
    topics: list[str]
    persons: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    organisations: list[str] = field(default_factory=list)

    @classmethod
    def sample(cls, name: str, rng: random.Random, strength: float = 1.0) -> AuthorStyle:
        """Tire un style au hasard ; `strength` (0 à 1) règle l'écart au style moyen."""

        def draw(low: float, high: float) -> float:
            middle = (low + high) / 2
            return middle + strength * (rng.uniform(low, high) - middle)

        persons = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(5)]
        places = rng.sample(PLACES, 3)
        organisations = [f"{rng.choice(PLACES)} {rng.choice(ORG_SUFFIXES)}" for _ in range(2)]
        return cls(
            name=name,
            connective_prob=draw(0.05, 0.4),
            subordinate_prob=draw(0.1, 0.5),
            coordinate_prob=draw(0.1, 0.45),
            semicolon_prob=draw(0.0, 0.5),
            dash_prob=draw(0.0, 0.25),
            exclaim_prob=draw(0.0, 0.12),
            question_prob=draw(0.0, 0.08),
            adjective_prob=draw(0.2, 0.75),
            adverb_prob=draw(0.05, 0.45),
            entity_prob=draw(0.2, 0.55),
            title_prob=draw(0.1, 0.9),
            bias=strength * rng.uniform(0.45, 0.8),
            connectives=_weighted(rng, CONNECTIVES, 3),
            adjectives=_weighted(rng, ADJECTIVES, 8),
            verbs=_weighted(rng, TRANSITIVE, 8),
            prepositions=_weighted(rng, PREPOSITIONS, 3),
            determiners=_weighted(rng, DETERMINERS, 3),
            topics=rng.sample(sorted(TOPICS), 2),
            persons=persons,
            places=places,
            organisations=organisations,
        )


class _Writer:
    def __init__(self, style: AuthorStyle, nouns: list[str], protagonists: list[str], rng: random.Random) -> None:
        self.s = style
        self.nouns = nouns
        self.protagonists = protagonists
        self.rng = rng

    def entity(self) -> str:
        rng, s = self.rng, self.s
        kind = rng.random()
        if kind < 0.55:
            person = rng.choice(self.protagonists + s.persons)
            if rng.random() < s.title_prob:
                person = f"{rng.choice(TITLES)} {person.split()[-1]}"
            return person
        if kind < 0.8:
            return rng.choice(s.places)
        return f"the {rng.choice(s.organisations)}"

    def noun_phrase(self) -> str:
        rng, s = self.rng, self.s
        if rng.random() < s.entity_prob * 0.6:
            return self.entity()
        det = _pick(rng, DETERMINERS, s.determiners, s.bias)
        noun = rng.choice(self.nouns)
        if rng.random() < s.adjective_prob:
            adjective = _pick(rng, ADJECTIVES, s.adjectives, s.bias)
            if det == "a" and adjective[0] in "aeiou":
                det = "an"
            return f"{det} {adjective} {noun}"
        if det == "a" and noun[0] in "aeiou":
            det = "an"
        return f"{det} {noun}"

    def prepositional(self) -> str:
        rng, s = self.rng, self.s
        prep = _pick(rng, PREPOSITIONS, s.prepositions, s.bias)
        target = rng.choice(s.places) if rng.random() < s.entity_prob * 0.5 else self.noun_phrase()
        return f"{prep} {target}"

    def clause(self) -> str:
        rng, s = self.rng, self.s
        subject = self.noun_phrase()
        adverb = f" {_pick(rng, ADVERBS, {}, 0)}" if rng.random() < s.adverb_prob else ""
        form = rng.random()
        if form < 0.6:
            verb = _pick(rng, TRANSITIVE, s.verbs, s.bias)
            core = f"{subject}{adverb} {verb} {self.noun_phrase()}"
        elif form < 0.8:
            core = f"{subject} was{adverb} {_pick(rng, ADJECTIVES, s.adjectives, s.bias)}"
        else:
            core = f"{subject} {rng.choice(INTRANSITIVE)}{adverb}"
        if rng.random() < 0.45:
            core += f" {self.prepositional()}"
        if rng.random() < s.dash_prob:
            core += f" — {self.noun_phrase()} —"
        return core

    def sentence(self) -> str:
        rng, s = self.rng, self.s
        text = self.clause()
        if rng.random() < s.subordinate_prob:
            text += f", {rng.choice(SUBORDINATORS)} {self.clause()}"
        elif rng.random() < s.coordinate_prob:
            if rng.random() < s.semicolon_prob:
                text += f"; {self.clause()}"
            else:
                text += f", {rng.choice(COORDINATORS)} {self.clause()}"
        if rng.random() < s.connective_prob:
            text = f"{_pick(rng, CONNECTIVES, s.connectives, s.bias)}, {text}"
        text = text.rstrip(" —")
        end = "."
        draw = rng.random()
        if draw < s.exclaim_prob:
            end = "!"
        elif draw < s.exclaim_prob + s.question_prob:
            end = "?"
        return text[0].upper() + text[1:] + end


def generate_toy_corpus(
    n_authors: int = 5,
    works_per_author: int = 6,
    docs_per_work: int = 8,
    words_per_doc: int = 300,
    seed: int = 0,
    name: str = "toy",
    style_strength: float = 1.0,
) -> Corpus:
    """Génère un corpus jouet déterministe (auteur, œuvre et sujet connus pour chaque texte).

    `style_strength` (0 à 1) rapproche les styles des auteurs du style moyen : plus il
    est faible, plus l'attribution est difficile.
    """
    documents: list[Document] = []
    used_names: set[str] = set()
    for a in range(n_authors):
        rng = make_random(seed, "toy", "author", a)
        author = f"author{a:02d}"
        style = AuthorStyle.sample(author, rng, style_strength)
        # Évite que deux auteurs partagent exactement les mêmes protagonistes.
        style.persons = [p for p in style.persons if p not in used_names] or style.persons
        used_names.update(style.persons)
        for w in range(works_per_author):
            work_rng = make_random(seed, "toy", "work", a, w)
            topic = style.topics[w % len(style.topics)]
            extra = work_rng.choice(sorted(TOPICS))
            nouns = TOPICS[topic] + TOPICS[extra][:4]
            protagonists = [f"{work_rng.choice(FIRST_NAMES)} {work_rng.choice(LAST_NAMES)}" for _ in range(2)]
            work_id = f"{author}-work{w:02d}"
            for d in range(docs_per_work):
                doc_rng = make_random(seed, "toy", "doc", a, w, d)
                writer = _Writer(style, nouns, protagonists, doc_rng)
                paragraphs: list[str] = []
                total = 0
                while total < words_per_doc:
                    paragraph = " ".join(writer.sentence() for _ in range(doc_rng.randint(3, 6)))
                    paragraphs.append(paragraph)
                    total += count_words(paragraph)
                documents.append(
                    Document(
                        id=f"{work_id}-{d:03d}",
                        text="\n\n".join(paragraphs),
                        author=author,
                        work=work_id,
                        language="en",
                        licence="synthetic (CC0)",
                        meta={"generator": "toy", "topic": topic},
                    )
                )
    return Corpus(documents, name=name, meta={"generator": "toy", "seed": seed, "style_strength": style_strength})
