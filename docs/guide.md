# Guide d'utilisation

Ce guide détaille l'installation, la construction des corpus, l'annotation, les études et la commande `plagia`. Pour un premier essai, le [README](../README.md) suffit.

1. [Installation](#installation)
2. [Prise en main hors ligne](#prise-en-main-hors-ligne)
3. [Construire les vrais corpus](#construire-les-vrais-corpus)
4. [Annoter](#annoter)
5. [Lancer les études](#lancer-les-études)
6. [Référence de la commande](#référence-de-la-commande)
7. [Dépannage](#dépannage)

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et Python 3.13 (`uv python install 3.13`, version de référence 3.13.17). Java (Stanford) et Docker (Spotlight local) sont facultatifs.

```bash
uv lock                                 # première fois : fige les versions dans uv.lock (à versionner)
uv sync                                 # socle, outils de développement et modèle spaCy en_core_web_sm 3.8.0
uv run plagia setup                     # ressources NLTK ; vérifie Java et Spotlight
uv run plagia setup --check-only
```

`uv sync` installe les groupes par défaut `dev` et `models` (le modèle spaCy) ; comme la synchronisation est exacte, un paquet installé à la main hors de ces groupes est retiré au `uv sync` suivant : passez par les groupes et les extras.

| Extra | Contenu | Utilisé par |
|---|---|---|
| `semantic` | Gensim 4.4 | Doc2Vec, Word2Vec/FastText, distance de Word Mover |
| `transformers` | sentence-transformers | plongements de phrases |
| `viz` | umap-learn | projection UMAP (t-SNE sinon) |
| `graph` | networkx | communautés de Louvain (familles de quasi-doublons) |
| `parquet` | pyarrow | `results.parquet` |
| `demo` | Streamlit | démonstrateur interactif |

```bash
uv sync --extra semantic --extra demo     # ou : uv sync --all-extras
uv sync --group notebook                  # Jupyter, pour l'exploration
```

Ressources NLTK utiles : `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4` (`uv run plagia setup --nltk punkt_tab stopwords wordnet omw-1.4`).
Sans elles, la liste de mots vides de scikit-learn ou la liste intégrée prend le relais ; le corpus synthétique demande WordNet (ou `--synonyms toy`).

**Stanford** (facultatif) : téléchargez Stanford NER et le POS Tagger, puis indiquez les fichiers `.jar` et les modèles dans la section `annotation` d'une configuration (`stanford_ner_jar`, `stanford_ner_model`, `stanford_pos_jar`, `stanford_pos_model`).

## Prise en main hors ligne

Les corpus jouets sont générés à partir d'une graine : mêmes fichiers, mêmes empreintes, d'une machine à l'autre. Ils valident la chaîne, pas les hypothèses scientifiques (voir [leur fiche](corpora/jouets.md)).

```bash
uv run plagia corpus toy --kind literary   --out data/corpora/toy_literary
uv run plagia corpus toy --kind news       --out data/corpora/toy_news
uv run plagia corpus toy --kind plagiarism --out data/corpora/toy_plagiarism

# Études réduites (quelques minutes chacune ; --n-jobs 4 pour aller plus vite)
uv run plagia run configs/demo/attribution.yaml --report
uv run plagia run configs/demo/plagiarism.yaml  --report
uv run plagia run configs/demo/clustering.yaml  --report

# Quelques textes du corpus de plagiat, en .txt, avec leur vérité dans index.csv
uv run plagia corpus export data/corpora/toy_plagiarism --split test --role suspicious --with-plagiarism --limit 3 --out exemples

# Attribution : modèle entraîné sur le corpus littéraire, appliqué aux exemples
uv run plagia attribute --train data/corpora/toy_literary --save results/models/toy.joblib
uv run plagia attribute --model results/models/toy.joblib --top-k 3 exemples/*.txt

# Détection : un rapport HTML par texte (passages surlignés et sources)
uv run plagia detect exemples/*.txt --sources data/corpora/toy_plagiarism --out results/detect

# Regroupement : k réel (« true »), choisi par la silhouette (« auto ») ou fixé
uv run plagia cluster data/corpora/toy_literary --blocks char_ngrams stylometry --k auto --out results/clusters

# Démonstrateur : page HTML pour un texte, ou application Streamlit (uv sync --extra demo)
uv run plagia demo --model results/models/toy.joblib --sources data/corpora/toy_plagiarism --text exemples/susp-00001.txt
uv run plagia demo --model results/models/toy.joblib --sources data/corpora/toy_plagiarism --serve
```

## Construire les vrais corpus

Les textes sous licence ne sont pas dans le dépôt ; ces commandes les reconstruisent à l'identique (voir [data/README.md](../data/README.md) et les [fiches](corpora/)). Le corpus synthétique dérive de C50 : construisez C50 d'abord.

| Corpus | Commande |
|---|---|
| Reuters C50 | `plagia corpus c50 --source data/raw/c50.zip --download --validation 0.2 --out data/corpora/c50` |
| Littéraire (Gutenberg) | `plagia corpus gutenberg --config configs/corpus/gutenberg.yaml --out data/corpora/literary` |
| Plagiat synthétique | `plagia corpus synthetic --base data/corpora/c50 --config configs/corpus/synthetic.yaml --out data/corpora/synthetic` |
| PAN 2013 | `plagia corpus pan --source data/raw/pan13-train --split train --test data/raw/pan13-test --validation 0.2 --name pan13 --out data/corpora/pan13` |
| AIDA CoNLL-YAGO | `plagia corpus aida --tsv data/raw/AIDA-YAGO2-dataset.tsv --conll data/raw/conll2003.txt --out data/corpora/aida` |

PAN et AIDA se téléchargent à la main (archives de recherche). `plagia corpus info <répertoire>` affiche les effectifs, vérifie le manifeste et l'absence de fuite ; `plagia corpus clean` écrit une nouvelle version nettoyée ; `plagia corpus export` écrit des documents en `.txt`.

## Annoter

```bash
uv run plagia annotate data/corpora/c50 --engine spacy                                       # remplit data/cache/annotations.sqlite
uv run plagia annotate data/corpora/c50 --engine spacy --offline                             # vérifie que tout est en cache
uv run plagia annotate data/corpora/aida --engine spacy --evaluate                           # qualité sur AIDA
uv run plagia annotate data/corpora/c50 --config configs/studies/study_a_c50_extended.yaml   # liage et typage
```

Les expériences annotent elles-mêmes leurs corpus au premier lancement (une fois, hors de la boucle d'apprentissage) puis lisent le cache ; une relance n'appelle aucun outil externe.
Moteurs : `spacy`, `stanford`, `heuristic` (hors ligne) et `none`.

## Lancer les études

```bash
uv run plagia run configs/studies/study_a_c50.yaml --dry-run       # plan : conditions et exécutions
uv run plagia run configs/studies/study_a_c50.yaml --n-jobs 8      # parallèle, reprise automatique
uv run plagia run configs/studies/study_a_c50.yaml --limit 20      # essai sur 20 exécutions
uv run plagia report results/study_a_c50                           # rapport HTML (--format md)
```

| Configuration | Corpus requis | Conditions | Exécutions |
|---|---|---:|---:|
| `study_a_c50.yaml` | c50 | 288 | 1 500 |
| `study_a_c50_extended.yaml` | c50 (+ Spotlight) | 324 | 1 650 |
| `study_a_literary.yaml` | literary | 24 | 125 |
| `study_b_synthetic.yaml` | synthetic | 48 | 360 |
| `study_b_synthetic_extended.yaml` | synthetic | 108 | 660 |
| `study_b_pan.yaml` | pan13 | 16 | 120 |
| `study_c_clustering.yaml` | c50, literary | 40 | 120 |
| `verification_literary.yaml` | literary | 2 | 10 |

Une exécution est une condition (représentation, classificateur, facteurs) pour une graine ; les exécutions comprennent les lignes de base. Une exécution en erreur est consignée sans arrêter la campagne, et une relance ne refait que ce qui manque (`--no-resume` pour tout recalculer).
La syntaxe complète des fichiers est dans [configuration.md](configuration.md), le contenu des répertoires de résultats dans [results/README.md](../results/README.md).

## Référence de la commande

| Commande | Rôle | Options principales |
|---|---|---|
| `plagia setup` | ressources et vérifications | `--nltk`, `--spacy`, `--spotlight-url`, `--check-only` |
| `plagia corpus <action>` | `c50`, `gutenberg`, `pan`, `aida`, `synthetic`, `toy`, `clean`, `export`, `info` | `--out`, `--seed`, selon l'action |
| `plagia annotate <corpus>` | annotation et cache | `--engine`, `--config`, `--offline`, `--evaluate`, `--spotlight-url` |
| `plagia run <config.yaml>` | étude complète | `--n-jobs`, `--limit`, `--no-resume`, `--dry-run`, `--report` |
| `plagia report <résultats>` | rapport d'une étude | `--format html\|md`, `--out` |
| `plagia attribute` | entraîner, enregistrer, classer | `--train`, `--save`, `--model`, `--top-k`, `--config`, `--engine` |
| `plagia detect <fichiers>` | passages repris et sources | `--sources`, `--out`, `--pair-model`, `--retrieval`, `--intrinsic` |
| `plagia cluster <corpus>` | regroupement | `--blocks`, `--algorithm`, `--k`, `--dimensions` |
| `plagia demo` | démonstrateur | `--model`, `--sources`, `--text`, `--serve` |
| `plagia bench` | contrôle de performance TF-IDF | `--documents`, `--words` |

`plagia <commande> --help` donne toutes les options. Code de sortie 2 et message explicite en cas d'entrée manquante (corpus, modèle, fichier).

## Dépannage

| Symptôme | Cause et solution |
|---|---|
| `Aucun corpus dans data/corpora/…` | Le corpus n'est pas construit ; la commande à lancer est affichée juste en dessous. |
| `Modèle introuvable : results/models/….joblib` | Entraînez et enregistrez d'abord un modèle : `plagia attribute --train <corpus> --save <chemin>`. |
| `Fichier(s) introuvable(s) : …` | Chemin du texte à analyser erroné ; `plagia corpus export` fournit des exemples. |
| `Aucune exécution dans results/…` | L'étude n'a pas tourné (corpus absent, `--dry-run`) : lancez `plagia run` d'abord. |
| `spotlight injoignable (Connection reset by peer)` | Le conteneur charge encore son modèle : attendez la fin (`docker logs -f dbpedia-spotlight.en`). |
| Test `spacy` ignoré, « modèle spaCy absent » | `uv sync` (le groupe `models` est installé par défaut) ; vérifiez avec `plagia setup --check-only`. |
| Streamlit demande une adresse électronique au premier lancement | Facultatif : laissez vide. Le démonstrateur désactive la collecte de statistiques d'usage. |
| Études longues | `--n-jobs` (processus parallèles), `--limit` pour un essai ; la reprise évite de recalculer. |
