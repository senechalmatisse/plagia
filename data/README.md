# Données

Ce répertoire n'est pas versionné (sauf ce fichier) : les textes sous licence ne sont pas redistribués. Tout ce qu'il contient se reconstruit avec la commande `plagia` à partir des configurations du dépôt ; deux reconstructions à graine égale donnent des fichiers identiques (manifestes et empreintes compris).

```
data/
  raw/           données brutes téléchargées ou déposées à la main (archives C50, PAN, AIDA ; livres Gutenberg)
  corpora/       corpus reconstruits, un répertoire par corpus
  cache/         annotations (SQLite : DocBin spaCy, réponses Spotlight et SPARQL), autres caches
```

## Reconstruire les corpus

| Corpus | Commande | Données brutes |
|---|---|---|
| C50 | `plagia corpus c50 --source data/raw/c50.zip --download --validation 0.2 --out data/corpora/c50` | téléchargées depuis l'UCI (`--download`) |
| Littéraire | `plagia corpus gutenberg --config configs/corpus/gutenberg.yaml --out data/corpora/literary` | téléchargées depuis un miroir Gutenberg, gardées dans `data/raw/gutenberg/` |
| Synthétique | `plagia corpus synthetic --base data/corpora/c50 --config configs/corpus/synthetic.yaml --out data/corpora/synthetic` | aucune (dérivé de C50) |
| PAN 2013 | `plagia corpus pan --source data/raw/pan13-train --split train --test data/raw/pan13-test --validation 0.2 --name pan13 --out data/corpora/pan13` | archive Zenodo 3715980, à extraire à la main |
| AIDA | `plagia corpus aida --tsv data/raw/AIDA-YAGO2-dataset.tsv --conll data/raw/conll2003.txt --out data/corpora/aida` | AIDA (MPI) et CoNLL 2003, à obtenir séparément |
| Jouets | `plagia corpus toy --kind literary\|news\|plagiarism --out data/corpora/toy_…` | aucune |

Les fiches de [docs/corpora/](../docs/corpora/) décrivent chaque corpus (origine, licence, effectifs, découpages, biais connus).

## Format d'un corpus

| Fichier | Contenu |
|---|---|
| `documents.jsonl` | un document par ligne : `id`, `text`, `author`, `work`, `language`, `licence`, `split`, `meta` |
| `cases.jsonl` | cas de plagiat : passage suspect (`document`, `start`, `length` en caractères), passage source, score, obfuscation, métadonnées (taille, jeu) |
| `pairs.jsonl` | paires (suspect, source) à comparer |
| `corpus.json` | nom, graine, configuration de construction, effectifs |
| `manifest.jsonl` | une ligne par document : identifiant, auteur, œuvre, langue, longueur (caractères et mots), licence, jeu, empreinte SHA-256, rôle |
| `manifest.sha256` | empreinte du manifeste, vérifiée au chargement |

Un corpus n'est jamais modifié en place : `plagia corpus clean --source <entrée> --out <sortie>` écrit une nouvelle version. `plagia corpus info <répertoire>` affiche les effectifs et vérifie le manifeste et l'absence de fuite.

## Cache d'annotation

`data/cache/annotations.sqlite` associe à chaque texte (empreinte), outil, version et paramètres son annotation sérialisée, et garde les réponses des services externes (Spotlight, SPARQL). `plagia annotate <corpus> --offline` vérifie qu'un corpus est entièrement en cache ; les expériences lisent ce cache sans appeler d'outil.
NB: Supprimer le fichier force un nouveau calcul.
