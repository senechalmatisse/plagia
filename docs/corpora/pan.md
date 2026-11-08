# Fiche du corpus - PAN, alignement de textes

| | |
|---|---|
| Exigence | F1.4 [M] (PAN 2013 ou 2014), [S] PAN-PC-11 ; validation externe de l'étude B |
| Origine | PAN (Webis), tâche *text alignment* : corpus PAN 2013 (Zenodo 3715980), PAN 2014 (Zenodo 3716022), PAN-PC-11 (Zenodo 3250095, 1,7 Go) |
| Licence | usage de recherche (conditions de l'archive Zenodo) ; non redistribué dans le dépôt |
| Langue | anglais (PAN-PC-11 contient aussi des sources en allemand et en espagnol pour le plagiat traduit) |

## Composition

Documents suspects (`suspicious-documentXXXXX.txt`), documents sources (`source-documentXXXXX.txt`), fichier `pairs` (paires suspect-source à aligner) et, pour chaque paire, une annotation XML : un élément `<feature name="plagiarism" …>` par passage plagié, avec ses décalages dans le suspect (`this_offset`, `this_length`) et dans la source (`source_offset`, `source_length`) et le type d'obfuscation. Dans PAN 2013 et 2014, les paires sont regroupées en sous-corpus : sans plagiat, copie sans obfuscation, obfuscation aléatoire, obfuscation par traduction en chaîne, résumé. Le chargeur en déduit les niveaux `none`, `random`, `translation` et `summary` (et `no-plagiarism` pour les paires négatives).

## Construction

```bash
plagia corpus pan --source data/raw/pan13-train --split train --test data/raw/pan13-test --validation 0.2 --name pan13 --out data/corpora/pan13
```

- Les textes sont lus **sans** normalisation (ni Unicode, ni fins de ligne) : les décalages de caractères des annotations restent valides.
- Le corpus d'entraînement de PAN sert à l'apprentissage (classificateur de paires, seuils) ; `--validation` en réserve une partie, groupée par composante connexe de paires, pour qu'aucune source ne soit partagée entre jeux ; le corpus de test de PAN forme le jeu de test.
- Les deux corpus PAN numérotent leurs documents de la même façon : `combine_pan` retire de l'entraînement tout document dont le texte figure aussi dans le test (avec ses cas et ses paires) et, si des identifiants restent communs, préfixe ceux du test par `test/` ; le test reste intact et l'export au format PAN retire le préfixe. Le nombre de documents retirés est enregistré (`meta.dropped_from_train` de `corpus.json`).
- `plagia.corpus.loaders.pan.write_pan_detections` écrit les détections au format XML de PAN, pour l'évaluation officielle.

L'étude `configs/studies/study_b_pan.yaml` utilise les paires fournies par PAN à la place de la recherche des sources (`retrieval.method: pairs`) et rapporte plagdet par type d'obfuscation.

## Biais connus et limites

- Les obfuscations de PAN, construites par les organisateurs, diffèrent des nôtres : c'est précisément l'intérêt de la validation externe (risque de circularité du corpus synthétique).
- Les obfuscations « traduction » et « résumé » sont plus difficiles que O1 à O3 : rappel attendu nettement plus bas.
- Le plagiat généré par des modèles de langage (PAN 2025, [C]) n'est pas couvert ; son chargeur reste à écrire.

## Citation

Potthast, M., Stein, B., Barrón-Cedeño, A., Rosso, P. (2010). *An Evaluation Framework for Plagiarism Detection*. COLING 2010. - Potthast, M. et al. (2013). *Overview of the 5th International Competition on Plagiarism Detection*. CLEF 2013.
