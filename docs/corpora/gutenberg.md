# Fiche du corpus - corpus littéraire (Project Gutenberg)

| | |
|---|---|
| Exigence | F1.2 [M] ; études A (attribution inter-œuvres) et C (regroupement), vérification d'auteur [S] |
| Origine | textes du domaine public de Project Gutenberg, téléchargés par un miroir et assemblés par le projet |
| Licence | domaine public aux États-Unis ; l'en-tête et la licence Project Gutenberg sont retirés (le texte n'est alors plus un « Project Gutenberg eBook ») ; vérifier le statut dans le pays d'utilisation ; textes non redistribués dans le dépôt |
| Langue | anglais (britannique et américain), XIXᵉ et début du XXᵉ siècle |
| Configuration | `configs/corpus/gutenberg.yaml` |

## Composition

10 auteurs × 6 œuvres ; 20 extraits de 500 à 1 000 mots par œuvre (coupés en fin de phrase) ; 5 œuvres en entraînement et 1 en test par auteur, soit au plus 10 × 6 × 20 = 1 200 extraits. Chaque extrait porte l'auteur, l'œuvre (titre et identifiant Gutenberg), la langue, la longueur, la licence et le jeu.

| Auteur | Œuvres |
|---|---|
| Jane Austen | Pride and Prejudice, Emma, Sense and Sensibility, Persuasion, Northanger Abbey, Mansfield Park |
| Charles Dickens | A Tale of Two Cities, Great Expectations, Oliver Twist, David Copperfield, Hard Times, Bleak House |
| Mark Twain | Adventures of Huckleberry Finn, The Adventures of Tom Sawyer, A Connecticut Yankee in King Arthur's Court, The Prince and the Pauper, Life on the Mississippi, The Innocents Abroad |
| Arthur Conan Doyle | A Study in Scarlet, The Sign of the Four, The Adventures of Sherlock Holmes, The Hound of the Baskervilles, The Memoirs of Sherlock Holmes, The Lost World |
| H. G. Wells | The Time Machine, The War of the Worlds, The Invisible Man, The Island of Doctor Moreau, The First Men in the Moon, The War in the Air |
| Thomas Hardy | Tess of the d'Urbervilles, Far from the Madding Crowd, Jude the Obscure, The Mayor of Casterbridge, The Return of the Native, The Woodlanders |
| Jack London | The Call of the Wild, White Fang, The Sea-Wolf, The Iron Heel, Martin Eden, Burning Daylight |
| Joseph Conrad | Heart of Darkness, Lord Jim, The Secret Agent, Nostromo, Under Western Eyes, Victory |
| George Eliot | Middlemarch, Silas Marner, Adam Bede, The Mill on the Floss, Romola, Daniel Deronda |
| Robert Louis Stevenson | Treasure Island, Kidnapped, The Strange Case of Dr. Jekyll and Mr. Hyde, The Black Arrow, The Master of Ballantrae, Catriona |

## Construction

```bash
plagia corpus gutenberg --config configs/corpus/gutenberg.yaml --raw data/raw/gutenberg --out data/corpora/literary
```

- **Accès** : le site de Project Gutenberg interdit l'accès automatisé ; le chargeur passe par un miroir (`https://aleph.gutenberg.org` par défaut, modifiable par `mirror`), attend `delay_seconds` (2 s) entre deux requêtes et garde les fichiers bruts en cache (`--raw`) : une reconstruction ne retélécharge rien.
- **Identification** : un identifiant fourni est vérifié contre le titre de l'en-tête du livre ; sans identifiant, ou en cas de discordance, le livre est résolu par titre et auteur dans le catalogue `pg_catalog.csv`.
- **Nettoyage** : décodage et normalisation Unicode (NFC), suppression de l'en-tête, de la licence et des mentions de l'auteur, découpage en paragraphes puis en extraits contigus et disjoints ; si une œuvre fournit plus de 20 extraits, 20 sont tirés par graine.
- **Découpage** : l'œuvre de test de chaque auteur est tirée par graine (ou fixée par `test_work`) ; tous les extraits d'une œuvre sont dans le même jeu, et la validation croisée de réglage est groupée par œuvre (`protocol.group_by: work`).

Le constructeur vérifie les minimums de F1.2 (10 auteurs, 6 œuvres par auteur, 20 extraits par œuvre) et signale toute œuvre trop courte.

## Biais connus et limites

- **Biais de sujet et d'œuvre** : noms de personnages, lieux et thèmes identifient l'œuvre ; l'évaluation inter-œuvres (œuvre de test jamais vue) et le masquage des entités mesurent ce qui reste du style (H2, H5).
- **Genres confondus avec les auteurs** : roman policier (Doyle), anticipation (Wells), récit d'aventures (London, Stevenson) ; le genre aide à reconnaître l'auteur.
- **Époque et variété** : anglais des XIXᵉ et XXᵉ siècles, auteurs britanniques et américains (orthographes différentes) ; les résultats ne se transposent pas à la langue actuelle.
- **Éditions** : transcriptions bénévoles de qualité inégale (coquilles, notes, préfaces d'éditeur qui peuvent subsister).
- **Dépendance des extraits** : les extraits d'une même œuvre sont corrélés ; les intervalles de confiance au niveau de l'extrait sont donc optimistes.
