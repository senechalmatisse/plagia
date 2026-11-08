"""Schéma d'étiquettes commun et correspondances entre jeux d'étiquettes.

Schéma minimal : `PER` (personne), `ORG` (organisation), `LOC` (lieu), `MISC` (divers), `DATE` (date ou heure), `NUM` (nombre ou montant).
"""

from __future__ import annotations

MINIMAL_LABELS: tuple[str, ...] = ("PER", "ORG", "LOC", "MISC", "DATE", "NUM")

ONTONOTES: dict[str, str] = {
    "PERSON": "PER",
    "NORP": "MISC",
    "FAC": "LOC",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "DATE",
    "TIME": "DATE",
    "PERCENT": "NUM",
    "MONEY": "NUM",
    "QUANTITY": "NUM",
    "ORDINAL": "NUM",
    "CARDINAL": "NUM",
}
CONLL: dict[str, str] = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}
STANFORD_3: dict[str, str] = {"PERSON": "PER", "ORGANIZATION": "ORG", "LOCATION": "LOC"}
STANFORD_4: dict[str, str] = {**STANFORD_3, "MISC": "MISC"}
STANFORD_7: dict[str, str] = {
    **STANFORD_3,
    "MONEY": "NUM",
    "PERCENT": "NUM",
    "DATE": "DATE",
    "TIME": "DATE",
}
# spaCy français (WikiNER) : PER, LOC, ORG, MISC
WIKINER: dict[str, str] = {"PER": "PER", "LOC": "LOC", "ORG": "ORG", "MISC": "MISC"}
DBPEDIA_ROOTS: dict[str, str] = {
    "dbo:Person": "PER",
    "schema:Person": "PER",
    "foaf:Person": "PER",
    "dbo:Organisation": "ORG",
    "schema:Organization": "ORG",
    "dbo:Place": "LOC",
    "schema:Place": "LOC",
    "dbo:Location": "LOC",
}

SCHEMES: dict[str, dict[str, str]] = {
    "ontonotes": ONTONOTES,
    "conll": CONLL,
    "stanford3": STANFORD_3,
    "stanford4": STANFORD_4,
    "stanford7": STANFORD_7,
    "wikiner": WIKINER,
    "minimal": {label: label for label in MINIMAL_LABELS},
}

# Étiquettes Penn Treebank → étiquettes universelles (Stanford POS, spaCy tag_).
PTB_TO_UPOS: dict[str, str] = {
    "CC": "CCONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "AUX",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "PDT": "DET",
    "POS": "PART",
    "PRP": "PRON",
    "PRP$": "PRON",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "ADP",
    "SYM": "SYM",
    "TO": "PART",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "WRB": "ADV",
    ".": "PUNCT",
    ",": "PUNCT",
    ":": "PUNCT",
    "``": "PUNCT",
    "''": "PUNCT",
    "-LRB-": "PUNCT",
    "-RRB-": "PUNCT",
    "HYPH": "PUNCT",
    "NFP": "PUNCT",
    "$": "SYM",
    "#": "SYM",
    "ADD": "X",
    "AFX": "ADJ",
    "GW": "X",
    "XX": "X",
    "_SP": "SPACE",
}


def normalize_label(label: str, scheme: str) -> str:
    """Étiquette du schéma minimal correspondant à `label` dans le jeu `scheme`."""
    mapping = SCHEMES.get(scheme)
    if mapping is None:
        raise KeyError(f"Jeu d'étiquettes inconnu : {scheme}")
    clean = label.upper().removeprefix("B-").removeprefix("I-")
    return mapping.get(clean, "MISC")


def kind_from_types(types: list[str] | tuple[str, ...]) -> str:
    """Étiquette minimale déduite des classes d'une base de connaissances."""
    for cls in types:
        name = cls.partition(":")[2] if cls[:1].isdigit() else cls
        if name in DBPEDIA_ROOTS:
            return DBPEDIA_ROOTS[name]
    return "MISC"


def ptb_to_upos(tag: str) -> str:
    """Étiquette universelle correspondant à une étiquette Penn Treebank."""
    return PTB_TO_UPOS.get(tag, "X")
