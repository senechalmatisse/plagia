# Fiche du corpus - corpus jouets (hors ligne)

| | |
|---|---|
| Rôle | démonstrations (`configs/demo/`), tests automatisés, intégration continue ; **pas** de conclusion scientifique |
| Origine | générés par `plagia.corpus.toy` à partir d'une graine, sans téléchargement |
| Licence | CC0 (indiqué dans le manifeste) : textes générés par le code du projet, sans donnée externe |
| Langue | pseudo-anglais (grammaire simple, vocabulaire réduit) |

## Génération

```bash
plagia corpus toy --kind literary   --out data/corpora/toy_literary     # 8 auteurs × 6 œuvres × 8 textes
plagia corpus toy --kind news       --out data/corpora/toy_news         # 8 auteurs × 48 textes
plagia corpus toy --kind plagiarism --out data/corpora/toy_plagiarism   # 200 sources, 240 suspects
```

Chaque auteur reçoit un style tiré de la graine : connecteurs favoris, longueur et structure des phrases, ponctuation, adjectifs et verbes préférés ; il possède aussi ses propres entités (personnes, lieux, organisations), comme un journaliste qui couvre ses sociétés. Chaque œuvre est centrée sur un sujet. `--style-strength` (0,3 par défaut, entre 0 et 1) rapproche les styles du style moyen : plus il est faible, plus l'attribution est difficile.

| Corpus | Documents | Découpage | Longueur (mots) |
|---|---:|---|---|
| `toy_literary` | 384 (8 auteurs, 48 œuvres) | par œuvre : 5 œuvres en entraînement, 1 en test par auteur (320 / 64) | 150 à 233, médiane 177 |
| `toy_news` | 384 (8 auteurs, un texte = un groupe) | moitié entraînement, moitié test (192 / 192) | 150 à 233, médiane 177 |
| `toy_plagiarism` | 200 sources + 240 suspects | 60/20/20, groupé par source | 350 à 1 036, médiane 386 |

Le corpus de plagiat applique le générateur synthétique (F1.3) à une base jouet, avec les niveaux O0 à O3, des passages de 50, 100 et 300 mots (20 par cellule) et le petit dictionnaire de synonymes intégré à la place de WordNet.

## Limites

- Vocabulaire fermé et grammaire répétitive : les scores sont bien plus élevés que sur des textes réels, et les correspondances fortuites plus fréquentes (d'où `min_density: 3.0` dans `configs/demo/plagiarism.yaml`).
- L'annotateur heuristique (sans spaCy) reconnaît des entités par la casse et des listes fermées : les blocs d'entités et d'étiquettes y sont approximatifs.
- Les effets (H1 à H5) peuvent apparaître en miniature mais ne valident aucune hypothèse : seules les études sur C50, le corpus littéraire, le corpus synthétique et PAN le peuvent.
