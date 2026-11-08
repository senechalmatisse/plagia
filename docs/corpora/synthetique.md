# Fiche du corpus - plagiat synthétique

| | |
|---|---|
| Exigence | F1.3 [M] ; étude B (détection de plagiat) |
| Origine | généré par le projet (`plagia.corpus.synthetic`) à partir d'un corpus de base (C50 par défaut) et d'une graine |
| Licence | celle du corpus de base (C50 : CC BY 4.0, à citer) ; le corpus généré n'est pas versionné, seule sa configuration l'est |
| Langue | celle du corpus de base (anglais) |
| Configuration | `configs/corpus/synthetic.yaml` |

## Composition

| Paramètre | Valeur | Exigence F1.3 |
|---|---:|---|
| Documents sources | 1 500 | au moins 1 000 |
| Documents suspects | 3 000, dont la moitié sans plagiat | au moins 1 000, dont la moitié sans plagiat |
| Niveaux d'obfuscation | O0, O1, O2, O3 (O4 en option) | O0 à O3 [M], O4 [S] |
| Tailles de passage | 50, 100 et 300 mots | 50, 100 et 300 mots |
| Passages par niveau et par taille | 250 | au moins 250 |

Niveaux d'obfuscation :

- **O0** copie exacte ;
- **O1** suppressions, insertions et remplacements aléatoires de mots (`edit_rate`, 15 %) ;
- **O2** remplacement par des synonymes WordNet (`synonym_rate`, 50 % des mots remplaçables) ;
- **O3** O2 suivi d'une permutation des phrases ;
- **O4** [S] rétro-traduction ou reformulation par un modèle de langage : adaptateurs `BackTranslation` et `Paraphrase` à brancher sur un traducteur ou un paraphraseur (aucun modèle n'est fourni).

Chaque cas (`cases.jsonl`) enregistre le document suspect, le début et la longueur du passage plagié, la source, le début et la longueur du passage d'origine (décalages de caractères), le niveau, la taille et le jeu ; `pairs.jsonl` liste les paires (suspect, source).

## Construction

```bash
plagia corpus synthetic --base data/corpora/c50 --config configs/corpus/synthetic.yaml --out data/corpora/synthetic
```

1. Les documents de base sont séparés en sources (au moins 320 mots, pour loger un passage de 300 mots) et en hôtes suspects, disjoints.
2. Sources et hôtes sont répartis entre entraînement, validation et test (60/20/20) ; un hôte ne reçoit que des passages de sources du même jeu : aucune source n'est partagée entre jeux.
3. La moitié des hôtes reste sans plagiat ; chaque hôte plagié reçoit une seule cellule (niveau, taille), ce qui permet d'évaluer chaque cellule séparément.
4. Les passages sont copiés depuis un début de phrase de la source, obfusqués, puis insérés entre deux phrases de l'hôte ; une portion de source n'est reprise qu'une fois.
5. Les sources sont renommées `src-…`, les suspects `susp-…`.

Dimensionnement : avec exactement 1 000 suspects (500 plagiés), chaque hôte plagié recevrait 6 ou 7 passages et les sources s'épuiseraient pour les passages de 300 mots (certaines cellules tomberaient sous 250 sur un corpus de la taille de C50). Les valeurs retenues donnent 2 passages par hôte plagié. La commande affiche le nombre de passages par cellule, et toute cellule sous l'objectif est signalée (`meta.shortfall` de `corpus.json`).

Synonymes : WordNet (NLTK, ressources `wordnet` et `omw-1.4`) ; `--synonyms toy` utilise le petit dictionnaire intégré (démonstrations hors ligne).

## Biais connus et limites

- **Circularité** : les synonymes WordNet de O2 et O3 sont proches des ressources que des détecteurs fondés sur WordNet exploiteraient ; d'où la validation externe sur PAN et les résultats détaillés par niveau.
- **Obfuscations artificielles** : O1 produit des phrases agrammaticales, O3 des permutations sans souci de cohérence ; un plagiaire réel reformule autrement (O4, PAN « summary »).
- **Frontières nettes** : les passages sont insérés entre deux phrases, ce qui facilite la localisation des bords par rapport à un plagiat réel.
- **Même domaine** : sources et hôtes viennent du même corpus de dépêches ; les sujets et entités partagés créent des faux positifs plausibles, utiles comme négatifs difficiles.
- **Prévalence** : la moitié des suspects contient du plagiat, bien plus qu'en situation réelle ; les résultats sont aussi donnés à prévalence réaliste (1 paire positive pour 10 négatives).
