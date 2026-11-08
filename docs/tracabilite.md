# Matrice de traçabilité des exigences

Correspondance entre les exigences du cahier des charges, les modules qui les réalisent et les tests qui les vérifient. Les chemins de modules sont relatifs à `src/plagia/`, ceux des tests à `tests/`.

Statuts :

- **Fait** : implémenté et vérifié par les tests dans l'environnement de développement ;
- **Outillé** : l'outil est prêt, l'exécution (campagne, annotation manuelle) reste à faire ;
- **Partiel**, **Non fait**.

## F1 - Corpus et données

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F1.1 C50, 50 auteurs, sous-ensembles de 5, 10, 20 auteurs par graine | M | Fait, non exécuté ici (téléchargement) | `corpus/loaders/c50.py`, `corpus/splits.py` (`subsample_authors`), `experiments/attribution.py` | `test_loaders.py::TestC50`, `test_corpus.py::TestSplits::test_subsampling_and_truncation` |
| F1.2 corpus littéraire Gutenberg (10 auteurs × 6 œuvres, extraits de 500 à 1 000 mots, découpage par œuvre, miroir) | M | Fait, non exécuté ici (réseau) | `corpus/loaders/gutenberg.py`, `configs/corpus/gutenberg.yaml` | `test_loaders.py::TestGutenberg` |
| F1.3 plagiat synthétique O0 à O3, 50/100/300 mots, ≥ 250 passages par cellule | M | Fait | `corpus/synthetic.py`, `corpus/obfuscation.py`, `configs/corpus/synthetic.yaml` | `test_corpus.py::TestSynthetic`, `::TestObfuscation` |
| F1.3 O4 (rétro-traduction, reformulation) | S | Partiel : adaptateurs `BackTranslation` et `Paraphrase`, sans modèle fourni | `corpus/obfuscation.py` | `test_corpus.py::TestObfuscation::test_o4_with_translator_and_paraphraser` |
| F1.4 PAN 2013 ou 2014 (paires, XML, décalages) | M | Fait, non exécuté ici (données) | `corpus/loaders/pan.py` (`load_pan`, `combine_pan`, `write_pan_detections`) | `test_loaders.py::TestPan`, `test_experiments.py::test_plagiarism_with_given_pairs` |
| F1.4 PAN-PC-11 | S | Fait, non exécuté ici (même format XML) | `corpus/loaders/pan.py` | idem |
| F1.4 PAN 2025 (plagiat généré par modèle de langage) | C | Non fait | - | - |
| F1.5 découpages sans fuite (groupés par œuvre, source, document), intersections vides | M | Fait | `corpus/splits.py` (`assign_group_splits`, `assign_pair_component_splits`, `check_no_leakage`) | `test_corpus.py::TestSplits`, `test_loaders.py::TestPan::test_combine_train_and_test_without_leakage` |
| F1.6 nettoyage, signatures, mentions de l'auteur, doublons exacts | M | Fait | `corpus/cleaning.py` | `test_corpus.py::TestCleaning` |
| F1.6 détection de la langue | S | Fait | `corpus/language.py` | `test_corpus.py::TestCleaning::test_language_detection` |
| F1.7 manifeste versionné, empreintes, graines | M | Fait | `corpus/manifest.py`, `corpus/store.py`, `core/seeds.py` | `test_corpus.py::TestStore` |
| F1.8 AIDA CoNLL-YAGO | S | Fait, non exécuté ici (données) | `corpus/loaders/aida.py` | `test_loaders.py::TestAida`, `test_cli.py::test_c50_pan_and_aida_commands` |

## F2 - Prétraitement et représentations

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F2.1 `Preprocessor` configurable (segmentation, casse, ponctuation, mots vides, lemmatisation ou racinisation) | M | Fait ; lemmatisation (spaCy) et racinisation (NLTK) non exécutées ici | `annotation/preprocess.py` | `test_annotation.py::TestPreprocessor` |
| F2.2 sacs de mots (comptes, binaire), TF-IDF sous-linéaire L2, bornes de fréquence | M | Fait | `features/lexical.py` | `test_features.py::TestLexicalBlocks` |
| F2.3 n-grammes de mots (1 à 3) et de caractères (2 à 5) | M | Fait | `features/lexical.py` | `test_features.py::TestLexicalBlocks::test_bow_tfidf_char` |
| F2.4 stylométrie et n-grammes d'étiquettes morphosyntaxiques | M | Fait | `features/stylometry.py`, `features/pos.py` | `test_features.py::TestStylometryAndPos` |
| F2.5 LSA | S | Fait | `features/embeddings.py` (`Lsa`) | `test_features.py::TestSemanticBlocks::test_lsa` |
| F2.5 Word2Vec ou FastText pondérés, Doc2Vec | S | Fait, non exécuté ici (Gensim) | `features/embeddings.py` | `test_features.py::test_gensim_blocks_when_available` (ignoré sans Gensim) |
| F2.5 plongements de phrases | C | Fait, non exécuté ici (sentence-transformers) | `features/embeddings.py` (`SentenceEmbeddings`) | `test_features.py::TestSemanticBlocks::test_optional_backends_report_missing_dependencies` |
| F2.6 blocs combinables, pondérés, `FeatureUnion` | M | Fait | `features/representation.py` | `test_features.py::TestRepresentation`, `::test_blocks_refit_inside_each_fold` |
| F2.7 masquage (entités, noms propres, mots pleins) | M | Fait | `features/masking.py` | `test_features.py::TestMasking` |

## F3 - Annotation, entités et typage sémantique

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F3.1 étiquetage morphosyntaxique spaCy | M | Fait, non exécuté ici (spaCy) | `annotation/spacy_engine.py` | `test_annotation.py::test_from_spacy_doc_conversion`, `::test_spacy_end_to_end` (marqué `spacy`) |
| F3.1 Stanford POS Tagger | S | Fait, non exécuté ici (Java) | `annotation/stanford.py` | `test_annotation.py::TestStanford` |
| F3.2 reconnaissance d'entités spaCy derrière `EntityRecognizer` | M | Fait, non exécuté ici (spaCy) | `annotation/base.py`, `annotation/spacy_engine.py` | idem F3.1 |
| F3.2 Stanford NER (3, 4 ou 7 classes) | S | Fait, non exécuté ici (Java) | `annotation/stanford.py` | `test_annotation.py::TestStanford` |
| F3.3 schéma commun PER/ORG/LOC/MISC/DATE/NUM | M | Fait | `annotation/schema.py` | `test_annotation.py::TestSchema` |
| F3.4 attributs d'entités (comptes et proportions par type, formes de surface, n-grammes de types), variante masquée | M | Fait | `features/entities.py`, `features/masking.py` | `test_features.py::TestEntityBlocks` |
| F3.5 liage DBpedia Spotlight, typage DBpedia ou YAGO à profondeur réglable | S | Fait, non exécuté ici (service) | `annotation/spotlight.py`, `annotation/knowledge_base.py`, `features/entities.py` (`TypeFeatures`) | `test_annotation.py::TestSpotlightAndKnowledgeBase` |
| F3.6 étiquettes sémantiques AnnoTag | C | Non fait | - | - |
| F3.7 qualité de reconnaissance et de liage sur AIDA | S | Fait | `annotation/quality.py`, `plagia annotate --evaluate` | `test_annotation.py::TestQuality`, `test_cli.py::test_c50_pan_and_aida_commands` |
| F3.7 annotation manuelle, κ de Cohen | C | Outillé (κ disponible), annotation à faire | `evaluation/stats.py` (`cohen_kappa`) | `test_evaluation.py::TestStatistics::test_kappa_and_summary` |
| F3.8 cache (DocBin, SQLite), clé empreinte + version + paramètres, mode hors ligne | M | Fait | `annotation/cache.py`, `experiments/context.py` | `test_annotation.py::TestCache`, `test_experiments.py::test_context_is_shared_and_offline` |

## F4 - Mesures de similarité

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F4.1 interface `Similarity` et registre | M | Fait | `similarity/base.py` | `test_similarity.py::TestSetMeasures::test_classes` |
| F4.2 Jaccard, Dice, recouvrement, cosinus, shingles (n de 3 à 8) | M | Fait | `similarity/sets.py`, `similarity/vector.py`, `features/shingles.py` | `test_similarity.py::TestSetMeasures`, `::TestVectorAndSemantic::test_cosine` |
| F4.3 Levenshtein et LCS normalisées, implémentation efficace | M | Fait (algorithmes bit-parallèles) | `similarity/sequence.py` | `test_similarity.py::TestSequenceMeasures` |
| F4.4 Jaccard et cosinus d'entités, Jensen-Shannon des types | M | Fait | `similarity/entity.py` | `test_similarity.py::TestEntityMeasures` |
| F4.5 cosinus de vecteurs sémantiques | S | Fait | `similarity/semantic.py` | `test_similarity.py::TestVectorAndSemantic::test_semantic` |
| F4.5 distance de Word Mover | C | Fait, non exécuté ici (Gensim) | `similarity/semantic.py` | idem |
| F4.6 MinHash avec LSH, winnowing | S | Fait (implémentation NumPy) | `similarity/minhash.py` | `test_similarity.py::TestMinHash` |
| F4.7 matrices creuses par blocs (k plus proches voisins), seuil et courbe précision-rappel | M | Fait | `similarity/matrix.py`, `similarity/threshold.py` | `test_similarity.py::TestMatricesAndThresholds` |

## F5 - Attribution d'auteur (T1)

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F5.1 naïve Bayes multinomial écrit à la main, identique à scikit-learn | M | Fait | `models/naive_bayes.py` | `test_models.py::TestNaiveBayes::test_identical_to_multinomial_nb` |
| F5.1 variante de Bernoulli | S | Fait | idem | `::test_identical_to_bernoulli_nb` |
| F5.2 SVM linéaire, régression logistique | M | Fait | `models/sklearn_models.py` | `test_models.py::TestSklearnWrappers` |
| F5.2 k-NN cosinus, Complement NB, forêt aléatoire, gradient boosting | S | Fait | idem | idem |
| F5.2 perceptron multicouche | C | Fait | idem | idem |
| F5.3 interface `Classifier` (`fit`, `predict`, `predict_scores`, `explain`) | M | Fait | `models/base.py` | `test_tasks.py::TestAttribution::test_fit_predict_rank_explain` |
| F5.4 réglage par validation croisée stratifiée à 5 plis (groupée par œuvre), test évalué une fois | M | Fait | `tasks/attribution.py`, `corpus/splits.py` | `test_tasks.py::TestAttribution::test_tuning_grouped_by_work`, `test_corpus.py::TestSplits::test_folds` |
| F5.5 classement des k auteurs les plus probables avec score | M | Fait | `models/base.py` (`rank`) | `test_tasks.py::TestAttribution::test_fit_predict_rank_explain` |
| F5.6 ensemble ouvert (seuil τ calibré, auteur inconnu) | S | Fait | `models/open_set.py`, `experiments/attribution.py` | `test_models.py::test_open_set` |
| F5.7 vérification d'auteur, c@1 | S | Fait | `tasks/verification.py`, `experiments/verification.py` | `test_tasks.py::TestVerification` |
| F5.7 méthode des imposteurs | C | Fait | `tasks/verification.py` (`ImpostersVerifier`) | `test_tasks.py::TestVerification::test_imposters` |

## F6 - Détection de plagiat (T2)

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F6.1 recherche des candidates (TF-IDF, shingles, MinHash), alignement (graines, extension, filtrage), décalages | M | Fait | `tasks/plagiarism/retrieval.py`, `alignment.py`, `pipeline.py` | `test_tasks.py::TestPlagiarism::test_retrieval_methods`, `::test_seed_matching_extension_and_filter` |
| F6.2 copies exactes et modifiées (shingles, recouvrement, confirmation par distance d'édition) | M | Fait | `tasks/plagiarism/alignment.py` (`confirmation`), `pairs.py` | `test_tasks.py::TestPlagiarism::test_exact_copies_detected_above_95_percent`, `::test_seed_matching_extension_and_filter` |
| F6.3 classificateur de paires (lexical, entités, sémantique), négatives difficiles | M | Fait | `tasks/plagiarism/pairs.py`, `dataset.py` | `test_tasks.py::TestPlagiarism::test_pair_dataset_classifier_and_origin`, `::test_windows_mode` |
| F6.4 identification de l'origine (exactitude@1, @k) | M | Fait | `tasks/plagiarism/origin.py` | `test_tasks.py::TestPlagiarism::test_pair_dataset_classifier_and_origin` |
| F6.5 plagiat intrinsèque (ruptures de style) | S | Fait | `tasks/plagiarism/intrinsic.py` | `test_tasks.py::TestPlagiarism::test_intrinsic_detection` |
| F6.6 rapport de similarité (score, couverture, passages surlignés, sources) | S | Fait | `tasks/plagiarism/report.py`, `reporting/html.py` | `test_tasks.py::TestPlagiarism::test_passages_and_report`, `test_cli.py::test_run_report_attribute_detect_cluster` |

## F7 - Regroupement (T3)

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F7.1 k-means après SVD tronquée, agglomératif (liaison moyenne, cosinus) | M | Fait | `models/clustering.py` | `test_models.py::TestClusterers::test_algorithms` |
| F7.1 DBSCAN, HDBSCAN | S | Fait | idem | idem |
| F7.2 choix de k (coude, silhouette) et k réel | M | Fait | `tasks/clustering.py` (`choose_k`, `elbow_point`) | `test_tasks.py::TestClustering::test_choose_k_and_elbow` |
| F7.3 pureté, NMI, ARI, V-mesure, silhouette | M | Fait | `evaluation/clustering.py` | `test_evaluation.py::TestClustering::test_invariance_to_label_permutation` |
| F7.4 familles de quasi-doublons (composantes connexes, Louvain) | S | Fait | `tasks/clustering.py` (`near_duplicate_families`) | `test_tasks.py::TestClustering::test_families_and_profiles` |
| F7.5 interprétation des groupes, projection 2D par auteur, œuvre et groupe | S | Fait (t-SNE ; UMAP non exécuté ici) | `tasks/clustering.py`, `reporting/report.py` | `test_tasks.py::TestClustering::test_families_and_profiles`, `test_experiments.py::test_clustering_and_verification_runs` |

## F8 - Évaluation et explicabilité

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F8.1 mesures (exactitude, F1 macro et micro, exactitude@k, précision, rappel, plagdet, granularité) | M | Fait | `evaluation/classification.py`, `detection.py`, `pan.py` | `test_evaluation.py::TestClassification`, `::TestPanMeasures` |
| F8.2 protocole (5 graines, bootstrap ≥ 1 000, IC à 95 %, test évalué une fois) | M | Fait | `experiments/`, `evaluation/bootstrap.py` | `test_evaluation.py::TestStatistics::test_bootstrap`, `test_experiments.py::TestAttributionRuns` |
| F8.3 lignes de base (classe majoritaire, cosinus TF-IDF avec seuil, recouvrement de shingles) | M | Fait | `experiments/plan.py`, `experiments/plagiarism.py` | `test_experiments.py::test_study_plans_match_specification` |
| F8.4 ablations par famille, tests appariés (McNemar, bootstrap), correction de Holm, Wilcoxon (≥ 10 graines) | M | Fait | `features/representation.py`, `evaluation/ablation.py`, `evaluation/stats.py` | `test_features.py::TestRepresentation::test_families_and_ablations`, `test_evaluation.py::TestStatistics`, `::test_paired_comparisons_with_holm` |
| F8.5 attributs discriminants, contributions des blocs, analyse d'erreurs sur 30 exemples classés à la main | M | Outillé : attributs et contributions calculés, 30 erreurs exportées par exécution avec les catégories proposées ; le classement à la main reste à faire sur les études réelles | `models/`, `evaluation/explain.py`, `experiments/attribution.py`, `reporting/report.py` | `test_evaluation.py::test_explanation_helpers`, `test_tasks.py::TestAttribution` |
| F8.6 courbes d'apprentissage et de robustesse, temps de calcul | S | Fait | `experiments/` (facteurs `n_train`, `text_length`, `n_authors`, niveau, taille ; durées), `reporting/figures.py` | `test_experiments.py::TestAttributionRuns::test_run_resume_and_outputs` |

## F9 - Interface et restitution

| Exigence | Priorité | Statut | Modules | Tests |
|---|---|---|---|---|
| F9.1 commande `plagia` (setup, corpus, annotate, run, attribute, detect, cluster, report) | M | Fait | `cli/main.py` | `test_cli.py` |
| F9.2 configuration YAML validée et archivée | M | Fait | `core/config.py`, `experiments/store.py` | `test_core.py::TestConfig`, `test_experiments.py::test_all_shipped_configs_are_valid` |
| F9.3 résultats CSV, prédictions JSON Lines | M | Fait | `experiments/store.py` | `test_experiments.py::TestAttributionRuns::test_run_resume_and_outputs` |
| F9.3 Parquet | C | Fait (si pyarrow est installé) | idem | idem |
| F9.4 rapport HTML ou Markdown (tableaux, figures, IC, comparaisons) | M | Fait | `reporting/report.py`, `figures.py`, `tables.py` | `test_cli.py::test_run_report_attribute_detect_cluster`, `test_experiments.py` |
| F9.5 démonstrateur (texte → auteurs probables, passages et sources surlignés) | C | Fait ; application Streamlit non exécutée ici | `reporting/demo.py`, `cli/demo_app.py` | `test_cli.py::test_run_report_attribute_detect_cluster`, `::test_demo_serve` |

## Exigences non fonctionnelles (§ 6.2)

| Exigence | Statut | Vérification |
|---|---|---|
| Performance : TF-IDF sur 5 000 textes de 500 mots en moins de 2 minutes ; paires candidates par index inversé ou MinHash ; annotations calculées une fois | Fait | `plagia bench` (environ 26 s sur la machine de développement) ; `similarity/matrix.py`, `similarity/minhash.py` ; cache |
| Reproductibilité : graine unique, versions archivées, résultats identiques | Fait | `test_experiments.py::TestAttributionRuns::test_runs_are_reproducible`, `test_corpus.py::TestStore::test_two_builds_with_same_seed_are_identical` ; `environment.json` |
| Absence de fuite (test automatisé) | Fait | `test_corpus.py::TestSplits`, `test_features.py::TestLexicalBlocks::test_vocabulary_and_idf_fitted_on_training_only` |
| Portabilité : Python 3.13, `uv sync`, environnement verrouillé, CI sur trois systèmes | Partiel | `pyproject.toml` ; `uv.lock` à générer (`uv lock`) ; `.github/workflows/ci.yml` écrit, non exécuté |
| Qualité : annotations de type (mypy), ruff, couverture ≥ 80 % sur corpus, features, similarity, models, tasks | Fait | `mypy` sans erreur ; `ruff check` et `ruff format --check` propres ; couverture des lignes de ces cinq paquets d'environ 97 % à la livraison (mesurée par `sys.monitoring`, coverage.py n'étant pas installé dans l'environnement de développement ; la CI la mesure avec pytest-cov) |
| Extensibilité : registres | Fait | `core/registry.py` ; `test_core.py::TestRegistry`, `test_annotation.py::TestHeuristic::test_regex_annotator_and_registry` |
| Licences et éthique | Fait | `README.md`, fiches des corpus |
| Documentation (README, docstrings, schéma d'architecture, fiches des corpus) | Fait | `README.md`, `docs/` |
| Architecture en couches | Fait | `test_architecture.py` |

## Jalons (§ 8)

| Jalon | Critère | État |
|---|---|---|
| J1 corpus | corpus reconstruits à l'identique ; aucune fuite | Fait sur les corpus jouets et synthétiques (tests) ; C50, Gutenberg et PAN à construire |
| J2 représentations et ligne de base | naïve Bayes identique à scikit-learn ; ligne de base d'attribution sur C50 > 60 % | Identité vérifiée ; exactitude sur C50 à mesurer |
| J3 annotation et similarités | entités spaCy et cache opérationnels ; qualité sur AIDA ; performance ; installation 3.13.17 et versions verrouillées | Cache, AIDA et performance outillés et testés ; spaCy à valider avec le modèle installé ; `uv.lock` à générer |
| J4 attribution | étude A complète avec ablations | Outillé (`configs/studies/study_a_*.yaml`), à exécuter |
| J5 détection | précision et rappel de O0 > 95 % ; plagdet validé sur cas construits ; validation sur PAN | O0 et plagdet vérifiés par les tests ; PAN à exécuter |
| J6 regroupement | mesures invariantes par permutation | Fait (test) ; étude C à exécuter |
| J7 campagne et rapport | rapport final et démonstration | Outillé (rapports générés, démonstrateur) ; campagne à exécuter |
