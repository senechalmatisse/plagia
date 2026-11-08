"""Ressources lexicales intégrées (mots-outils), sans téléchargement.

Ces listes servent aux attributs stylométriques, au masquage, aux obfuscateurs et à l'annotateur heuristique. Les listes de mots vides de NLTK restent utilisables pour le prétraitement.
"""

from __future__ import annotations

ENGLISH_FUNCTION_WORDS: tuple[str, ...] = tuple(
    """
    a about above according across after afterwards again against ago ahead albeit all almost
    alone along already also although always am among amongst an and another any anybody
    anyhow anyone anything anyway anywhere are around as aside at away back be became because
    become becomes been before beforehand behind being below beneath beside besides between
    beyond both but by can cannot could dare despite did do does doing done down during each
    either else elsewhere enough even ever every everybody everyone everything everywhere
    except few fewer first for former formerly from further furthermore had hardly has have
    having he hence her here hereafter hereby herein hers herself him himself his how however
    i if in indeed inside instead into is it its itself just last latter least less lest
    like little many may me meanwhile might mine more moreover most mostly much must my myself
    namely near neither never nevertheless next no nobody none noone nor not nothing now
    nowhere of off often on once one ones only onto or other others otherwise ought our ours
    ourselves out outside over own past per perhaps quite rather round same several shall she
    should since so some somebody somehow someone something sometime sometimes somewhat
    somewhere soon still such than that the their theirs them themselves then thence there
    thereafter thereby therefore therein thereupon these they this those though through
    throughout thru thus till to together too toward towards under underneath unless unlike
    until up upon us very via was we well were what whatever when whence whenever where
    whereafter whereas whereby wherein whereupon wherever whether which whichever while
    whilst whither who whoever whole whom whomever whose why will with within without would
    yet you your yours yourself yourselves
    """.split()
)

FRENCH_FUNCTION_WORDS: tuple[str, ...] = tuple(
    """
    à afin ai ainsi alors après assez au aucun aucune aujourd auprès aussi autant autre autres
    aux avant avec avoir beaucoup bien car ce ceci cela celle celles celui cependant ces cet
    cette ceux chacun chaque chez ci comme comment contre d dans de depuis des dès donc dont
    du durant elle elles en encore entre et été être eux fait fois hormis hors il ils j jamais
    je jusque l la là laquelle le lequel les lesquels leur leurs lors lorsque lui m ma mais
    malgré me même mes moi moins mon n ne ni non nos notre nous on ont or ou où par parce
    parmi pas pendant peu peut plus plusieurs pour pourquoi pourtant près puis puisque qu
    quand que quel quelle quelles quels qui quoi s sa sans se selon ses si sinon soi son
    sont sous souvent sur t ta tandis tant te tel telle tels tes toi ton toujours tous tout
    toute toutes très tu un une vers voici voilà vos votre vous y
    """.split()
)

PUNCTUATION_MARKS: tuple[str, ...] = (
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "'",
    '"',
    "-",
    "—",
    "(",
    ")",
    "…",
    "’",
    "“",
    "”",
    "«",
    "»",
)

FUNCTION_WORDS = {"en": ENGLISH_FUNCTION_WORDS, "fr": FRENCH_FUNCTION_WORDS}


def function_words(language: str = "en") -> frozenset[str]:
    """Ensemble des mots-outils de la langue (anglais par défaut)."""
    return frozenset(FUNCTION_WORDS.get(language, ENGLISH_FUNCTION_WORDS))
