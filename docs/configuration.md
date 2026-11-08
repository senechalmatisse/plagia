# Référence des configurations d'expérience

Une expérience est décrite par un fichier YAML, validé au chargement par `plagia.core.config.load_config` : un type erroné, une valeur hors bornes ou une clé inconnue arrêtent le chargement avec un message qui désigne la clé fautive.
La configuration est immuable et archivée avec les résultats (`config.yaml`, `config.json`) ; son empreinte (sans `n_jobs` ni `output_dir`) identifie l'expérience.

Exemples complets : `configs/demo/` (hors ligne, corpus jouets) et `configs/studies/` (études A, B et C). `plagia run <fichier> --dry-run` affiche le plan sans rien exécuter.

## Premier niveau

| Clé | Type | Défaut | Rôle |
|---|---|---|---|
| `name` | texte (`[A-Za-z0-9_.-]+`) | obligatoire | nom de l'expérience et de son répertoire de résultats |
| `task` | `attribution` \| `verification` \| `plagiarism` \| `clustering` | obligatoire | tâche ; la section du même nom est obligatoire, les autres interdites |
| `description` | texte | `""` | reprise en tête du rapport |
| `seed` | entier | `42` | graine maîtresse ; les graines des exécutions en dérivent |
| `output_dir` | chemin | `results` | les résultats vont dans `<output_dir>/<name>/` |
| `n_jobs` | entier | `1` | processus parallèles (surchargé par `plagia run --n-jobs`) |
| `annotation` | section | voir ci-dessous | annotateurs et cache |
| `protocol` | section | voir ci-dessous | validation croisée, graines, bootstrap |
| `include`, `exclude` | listes de dictionnaires | aucun | filtres de conditions : `{representation: R1, classifier: nb}` retient (ou retire) les conditions dont les paramètres correspondent |

## `annotation`

| Clé | Défaut | Rôle |
|---|---|---|
| `engine` | `spacy` | `spacy`, `stanford`, `heuristic` (hors ligne) ou `none` (jetons seuls) |
| `model` | `en_core_web_sm` | modèle spaCy |
| `batch_size`, `n_process` | `64`, `1` | traitement par lots de spaCy |
| `java` | `java` | exécutable Java (Stanford) |
| `stanford_ner_jar`, `stanford_ner_model` | aucun | fichiers de Stanford NER (modèles à 3, 4 ou 7 classes) |
| `stanford_pos_jar`, `stanford_pos_model` | aucun | fichiers du Stanford POS Tagger |
| `linker` | `none` | `spotlight` pour le liage à DBpedia |
| `spotlight_url` | `http://localhost:2222/rest` | point d'accès Spotlight (local, par Docker) |
| `spotlight_confidence`, `spotlight_support` | `0.5`, `20` | filtrage des liens |
| `knowledge_base` | `none` | `dbpedia` ou `yago` : typage par la hiérarchie de classes |
| `taxonomy_path` | aucun | extrait local de la hiérarchie (TSV, N-Triples ou RDF) |
| `sparql_endpoint` | aucun | point d'accès SPARQL (YAGO), réponses mises en cache |
| `cache_path` | `data/cache/annotations.sqlite` | cache des annotations et des réponses externes |

## `protocol`

| Clé | Défaut | Rôle |
|---|---|---|
| `cv_folds` | `5` | plis de la validation croisée de réglage (stratifiée) |
| `seeds` | `5` | nombre de graines (`seed`, `seed + 1`…) ou liste explicite |
| `bootstrap` | `1000` | rééchantillonnages des intervalles de confiance (0 : aucun) |
| `confidence` | `0.95` | niveau des intervalles |
| `scoring` | `f1_macro` | mesure optimisée pendant le réglage |
| `group_by` | `none` | `work` : plis groupés par œuvre (corpus littéraire) |
| `validation_size` | `0.2` | part de l'entraînement réservée à la validation quand le corpus n'en a pas |
| `tune` | `true` | `false` : pas de grille, paramètres fixes |

## Représentations : blocs de caractéristiques

Une représentation est une liste de blocs concaténés (`FeatureUnion` pondérée). Chaque bloc a un `type` enregistré, un `name` facultatif (unique dans la représentation), un `weight` (> 0), un `mask` facultatif (`entities`, `propn`, `content` : masquage du texte avant ce bloc) et des `params`.

```yaml
representations:
  Style+entities:
    description: n-grammes de caractères, stylométrie, étiquettes et entités
    blocks:
      - {type: char_ngrams, params: {ngram_range: [3, 5], min_df: 2}}
      - {type: stylometry, weight: 0.5}
      - {type: pos_ngrams, params: {ngram_range: [1, 3]}}
      - {type: entities, weight: 0.5}
  R2-masqué:
    blocks:
      - {type: tfidf, mask: entities, params: {ngram_range: [1, 2]}}
```

| Type | Famille | Paramètres principaux (défaut) |
|---|---|---|
| `bow` | mots | `binary` (false), `ngram_range` ([1, 1]), `min_df`, `max_df`, `max_features`, `lowercase`, `punctuation` (false : retirée), `stopwords` (true : conservés), `normalization` (`none`, `lemma`, `stem`), `numbers` (`keep`, `mask`, `drop`), `stopwords_source` (`nltk`, `sklearn`, `builtin`) |
| `tfidf` | mots | mêmes paramètres, `ngram_range` ([1, 2]), `sublinear_tf` (true), `norm` (`l2`), `use_idf` |
| `char_ngrams` | caractères | `ngram_range` ([2, 5]), `analyzer` (`char` ou `char_wb`), `min_df` (2), `sublinear_tf`, `norm` |
| `stylometry` | stylométrie | `function_words`, `punctuation`, `lengths`, `richness` (rapport types/occurrences, hapax, K de Yule, MATTR), `scale` (`minmax`) |
| `pos_ngrams` | étiquettes | `ngram_range` ([1, 3]), `tagset` (`upos`) |
| `entities` | entités | `counts`, `proportions`, `forms` (formes de surface), `type_ngrams`, `ngram_range` ([1, 3]), `level` (`surface` ou `uri`), `labels` |
| `types` | types | `depth` (2) dans la hiérarchie de la base de connaissances, `fallback` (`kind` : type du schéma commun) |
| `lsa` | sémantique | `n_components` (100) |
| `doc2vec`, `word_vectors` | sémantique | Gensim (extra `semantic`) : `vector_size`, `window`, `min_count`, `epochs`, `algorithm` (`word2vec`, `fasttext`) |
| `sentence_embeddings` | sémantique | sentence-transformers (extra `transformers`) : `model` |

Les familles servent aux ablations : chaque famille est isolée puis retirée de la représentation de base.

## Algorithmes

```yaml
classifiers:
  nb:  {type: naive_bayes, grid: {alpha: [0.01, 0.1, 1.0]}}
  svm: {type: linear_svm, grid: {C: [0.1, 1.0, 10.0]}}
  rf:  {type: random_forest, params: {n_estimators: 300}}
  svm_ngrams:
    type: linear_svm
    grid: {C: [0.1, 1.0], features__tfidf__ngram_range: [[1, 1], [1, 2]]}
```

Classificateurs enregistrés : `naive_bayes` (multinomial, maison), `bernoulli_nb`, `linear_svm`, `logreg`, `knn` (cosinus), `complement_nb`, `random_forest`, `gradient_boosting`, `mlp`, `majority`. `params` fixe des paramètres, `grid` donne les valeurs explorées par la validation croisée ; les noms sont ceux de l'estimateur (`C`, `alpha`, `n_estimators`…), le préfixe interne (`clf__estimator__`) est ajouté automatiquement. Une grille peut aussi porter sur les blocs de caractéristiques, avec le chemin complet `features__<nom du bloc>__<paramètre>` (bornes de fréquence, plage de n-grammes).

Algorithmes de regroupement : `kmeans`, `agglomerative` (liaison moyenne, cosinus), `dbscan`, `hdbscan` ; tous opèrent après SVD tronquée et normalisation L2.

## `attribution` (étude A)

| Clé | Défaut | Rôle |
|---|---|---|
| `corpus` | obligatoire | répertoire du corpus |
| `representations`, `classifiers` | obligatoires | dictionnaires nom → définition |
| `n_authors` | `[null]` | sous-ensembles d'auteurs tirés par graine (`null` : tous) |
| `n_train` | `[null]` | textes d'entraînement par auteur |
| `text_length` | `[null]` | longueur des textes de test en mots (troncature) |
| `top_k` | `3` | taille du classement évalué (exactitude@k) |
| `explain_top_k` | `20` | attributs discriminants par auteur |
| `error_sample` | `30` | erreurs exportées pour l'analyse manuelle |
| `majority_baseline` | `true` | ligne de base « classe majoritaire » |
| `open_set` | aucun | `{unknown_authors: 2, threshold: auto}` : auteurs retirés de l'entraînement, seuil τ |
| `ablation` | aucun | `{base: Style+entities, families: [...]}` |

## `verification` [S]

`corpus`, `representations`, `classifiers` (classificateur de paires), `imposters` (méthode des imposteurs), `pairs_per_split` (400), `abstain_margin` (marge d'abstention pour c@1).

## `plagiarism` (étude B)

| Clé | Défaut | Rôle |
|---|---|---|
| `corpus` | obligatoire | corpus synthétique ou PAN |
| `levels` | `[O0, O1, O2, O3]` | niveaux d'obfuscation évalués (`none`, `random`, `translation`, `summary` pour PAN) |
| `sizes` | `[50, 100, 300]` | tailles de passage (0 : non définie, PAN) |
| `feature_sets` | obligatoire | nom → similarités du classificateur de paires |
| `classifiers` | obligatoire | classificateurs de paires |
| `window_sentences` | `3` | fenêtres de phrases des paires |
| `negatives_per_positive` | `3` | négatives (dont difficiles) par positive à l'entraînement |
| `test_prevalence` | `[1, 10]` | négatives par positive au test (équilibré et réaliste) |
| `retrieval` | voir ci-dessous | recherche des sources candidates |
| `alignment` | voir ci-dessous | graines, extension, filtrage |
| `threshold_criterion` | `f1` | seuil choisi sur la validation : maximum de F1 ou `precision` cible |
| `target_precision` | `0.9` | précision visée si `threshold_criterion: precision` |
| `baselines` | `true` | lignes de base : cosinus TF-IDF avec seuil, recouvrement de shingles |
| `passage_level` | `true` | détection complète et mesures de PAN |
| `origin_k` | `5` | exactitude@k de l'origine |
| `max_suspicious` | aucun | limite de documents suspects (essais rapides) |

Similarités disponibles pour `feature_sets` : `cosine_tfidf`, `containment_shingles`, `jaccard_shingles`, `dice_words`, `lcs`, `levenshtein`, `length_ratio` (lexicales, L) ; `jaccard_entities`, `cosine_entities`, `type_divergence`, `kb_type_divergence` (entités et types, E) ; `semantic` (S).

`retrieval` : `method` (`tfidf`, `shingles`, `minhash`, ou `pairs` pour utiliser les paires fournies par PAN), `k` (10), `chunk_sentences` (4), `shingle_size` (5), `num_perm` (128), `bands` (32).

`alignment` : `seeds` (`ngrams`, `sentences`, `both`), `ngram_size` (4), `sentence_threshold` et `dice_threshold` (0,5), `max_gap` (250 caractères), `min_seeds` (2), `min_density` (1,0), `confirmation` (`dice`, `containment`, `edit`), `min_similarity` (0,5), `min_length` (150 caractères), `mode` (`seeds`, `windows`, `both`), `verify` (vérification par le classificateur de paires).

## `clustering` (étude C)

| Clé | Défaut | Rôle |
|---|---|---|
| `corpora` | obligatoire | nom → `{path, split, labels: [author, work]}` |
| `representations` | obligatoire | comme pour l'attribution |
| `dimensions` | `[100, 300]` | dimensions de la SVD tronquée |
| `algorithms` | obligatoire | nom → algorithme de regroupement |
| `k` | `true` | `true` (nombre réel d'auteurs), `silhouette`, `elbow` ou un entier |
| `k_range` | `2…20` | valeurs essayées pour la silhouette et le coude |
| `families_threshold` | aucun | seuil de similarité des familles de quasi-doublons |
