# Résultats

Ce répertoire n'est pas versionné (sauf ce fichier). Chaque expérience lancée par `plagia run <configuration.yaml>` écrit dans `results/<nom>/` (le nom vient de la configuration ; `--output-dir` change la racine).

| Fichier | Contenu |
|---|---|
| `config.yaml`, `config.json` | configuration archivée, telle qu'exécutée |
| `plan.json` | conditions (identifiant, paramètres, graines, ligne de base ou non) et résumé du plan (nombre de conditions et d'exécutions, graines) |
| `environment.json` | Python, système, versions des bibliothèques, modèles spaCy, outils d'annotation (Stanford, Spotlight, base de connaissances, empreinte de la taxonomie), hash Git |
| `runs.jsonl` | journal : une exécution (condition × graine) par ligne, avec métriques, durées, version du code ou message d'erreur ; une relance saute les exécutions déjà faites |
| `results.csv` (et `results.parquet` si pyarrow est installé) | une ligne par exécution : paramètres, graine, métriques, intervalles de confiance par bootstrap, durées |
| `summary.csv` | une ligne par condition : moyenne, écart-type, intervalle de confiance entre graines et intervalle par bootstrap de chaque mesure |
| `comparisons.csv` | comparaisons appariées avec la référence (lignes de base, représentation de base des ablations) : différence et son intervalle, p de McNemar et du bootstrap apparié, corrections de Holm |
| `predictions/*.jsonl` | prédictions de chaque exécution : identifiant, vérité, prédiction et classement des auteurs (attribution) ; paires de fenêtres ou passages avec décalages, étiquette et score (détection) ; groupe de chaque document (regroupement) |
| `artifacts/*.json` | matrices de confusion, attributs discriminants, contributions des blocs, échantillon d'erreurs à classer, courbes précision-rappel, projections et profils de groupes |
| `report.html` ou `report.md` | rapport (`plagia report results/<nom>`, ou `--report` au lancement) |

Les noms des fichiers de prédictions et d'artefacts reprennent les paramètres de la condition et la graine (`rep=R1_clf=nb_authors=4_train=all_len=all_42-….jsonl`).

Autres sorties : `plagia attribute --save` écrit un modèle (`results/models/…`), `plagia detect` un rapport de similarité par document, `plagia cluster` les affectations et la projection, `plagia demo --text` une page de démonstration.
