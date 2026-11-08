# Fiche du corpus - AIDA CoNLL-YAGO (référence pour les entités)

| | |
|---|---|
| Exigence | F1.8 [S] ; qualité des annotateurs F3.7 [S] |
| Origine | Max-Planck-Institut für Informatik (Hoffart et al., 2011) : annotations de liage des mentions du jeu CoNLL 2003 (anglais) vers YAGO2 et Wikipedia |
| Licence | annotations AIDA : CC BY-SA 3.0 ; texte CoNLL 2003 : dépêches Reuters (RCV1), soumises à l'accord de licence de Reuters, à obtenir séparément ; rien n'est redistribué dans le dépôt |
| Langue | anglais |

## Composition

1 393 dépêches Reuters de 1996 et 1997, réparties comme CoNLL 2003 (entraînement, `testa`, `testb`) ; chaque mention d'entité nommée porte, quand elle existe, l'entité YAGO
et l'URL Wikipedia correspondantes (les mentions sans entité dans la base sont marquées `--NME--`). Les types d'entités (PER, ORG, LOC, MISC) viennent des fichiers CoNLL 2003.

## Construction

AIDA ne fournit que les annotations : l'archive contient un programme qui les fusionne avec le texte CoNLL 2003 pour produire `AIDA-YAGO2-dataset.tsv` (voir son README). Pour reporter les types d'entités, concaténez les fichiers CoNLL 2003 dans l'ordre d'AIDA :

```bash
cat eng.train eng.testa eng.testb > data/raw/conll2003.txt
plagia corpus aida --tsv data/raw/AIDA-YAGO2-dataset.tsv --conll data/raw/conll2003.txt --out data/corpora/aida
plagia annotate data/corpora/aida --engine spacy --evaluate
plagia annotate data/corpora/aida --engine spacy --spotlight-url http://localhost:2222/rest --evaluate
```

Le chargeur reconstitue le texte (jetons séparés par des espaces, une phrase par ligne) et convertit les mentions de référence en décalages de caractères (`meta.gold_entities` de chaque document). Les types CoNLL sont reportés sur les mentions de mêmes positions ; si les deux fichiers ne s'alignent pas (nombre de documents ou jetons différents), la fusion est ignorée avec un avertissement.

## Mesures

`plagia annotate … --evaluate` calcule :

- la reconnaissance des entités : précision, rappel et F1 globaux, par type (PER, ORG, LOC, MISC) et pour la seule détection des mentions (empans identiques) ;
- le liage : précision, rappel et F1 sur les mentions liables (empan et ressource identiques ; ressources comparées par leur identifiant, dernier segment de l'URI).

Ces mesures quantifient la propagation des erreurs d'annotation vers les tâches finales (comparaison de spaCy et de Stanford NER, effet du liage sur les blocs `entities` et `types`).

## Biais connus et limites

- Dépêches économiques et politiques de 1996-1997 : les entités, très fréquentes, sont surtout des pays, des sociétés et des personnalités de l'époque.
- Le texte reconstitué (jetons séparés par des espaces) diffère de la typographie d'origine : les annotateurs y sont un peu moins bons que sur du texte naturel.
- Les URI YAGO2 et Wikipedia de 2010 ne correspondent pas toujours aux ressources DBpedia actuelles (pages renommées ou fusionnées) : le rappel du liage est sous-estimé.
- Le schéma commun comprend aussi DATE et NUM, absents de la référence : seuls PER, ORG, LOC et MISC sont évalués.

## Citation

Hoffart, J., Yosef, M. A., Bordino, I., Fürstenau, H., Pinkal, M., Spaniol, M., Taneva, B., Thater, S., Weikum, G. (2011). *Robust Disambiguation of Named Entities in Text*. EMNLP 2011.
