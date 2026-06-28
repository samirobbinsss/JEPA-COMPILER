# JEPA-v2 — Handoff

> Pour la personne qui reprend le projet. Lis-le en entier une fois : il contient
> le **pourquoi** des décisions, l'**état exact** de ce qui marche, et les
> **next steps** détaillées avec les fichiers à toucher.
>
> Dernière mise à jour : 2026-06-27.

---

## 1. L'objectif en une phrase

Apprendre un **encodeur de programmes auto-supervisé** (GNN entraîné de zéro,
esprit JEPA, **sans labels**) sur des graphes **ProgramML** d'IR LLVM, dont
l'embedding de sortie est **factorisé en deux sous-espaces** :

```
z = [  z_sem  |  z_speed  ]
```

| Bloc | À rapprocher | À éloigner | Capture |
|---|---|---|---|
| **`z_sem`** | les 4 niveaux `-O` d'un **même** source | sources différents | *ce que le code fait* (invariant à l'optim) |
| **`z_speed`** | un **même niveau `-O`** entre sources différents | niveaux `-O` différents | *le profil d'optim / « vitesse »* (invariant au programme) |

Le livrable utile = **l'encodeur** et la qualité du disentanglement, pas un
classifieur ni un décodeur.

---

## 2. Décisions actées (ne pas re-débattre)

- **Représentation = ProgramML** (`pip install programl`). C'est le point central
  de la v2 : remplacer le graphe maison de la v1 (`../jepa-ir`) par la
  représentation GNN-sur-IR standard de la littérature.
- **Auto-supervisé pur (JEPA).** Le niveau `-O` sert UNIQUEMENT à grouper les vues
  positives/négatives. Ce n'est **jamais** une cible de classification.
- **Embedding factorisé** `[z_sem | z_speed]` (cf. tableau ci-dessus).
- **Pas de masquage** dans la v2 (contrairement à la v1). Le signal vient de la
  structure d'invariance croisée (across -O, across programmes).
- **Anti-collapse : VICReg** sur chaque bloc séparément + un terme de
  **décorrélation croisée** `z_sem ⟂ z_speed` (force le disentanglement).
- **Dataset = ExeBench** (fonctions C avec deps réelles), compilé O0..O3.
- **Tout tourne sur le pod RunPod** (Linux x86_64). ProgramML n'a PAS de wheel
  arm64 → impossible sur le Mac. En local : on édite le code seulement.

### LE risque #1 à valider AVANT d'entraîner (gate)
La v1 a prouvé (cf. `../jepa-ir/docs/limitation_non_bijective.md`) que sur
**AnghaBench**, clang **sature dès -O2** : l'IR de O1/O2/O3 est *identique*. Si
c'est aussi le cas sur ExeBench, alors `z_speed` ne pourra **jamais** distinguer
O1/O2/O3 (entrée identique → sortie identique). **Donc la première chose à faire
est le probe** (next step #1) qui mesure le % de graphes distincts O2 vs O3. On
n'entraîne pas tant que ce gate n'est pas vert.

---

## 3. Ce qui est DÉJÀ fait ✅

### Environnement pod — programl MARCHE
`from_cpp` + `to_networkx` produisent un graphe valide sur le pod. Ça a demandé
**7 correctifs enchaînés** (tous non-évidents) — voir §6. Tout est encapsulé dans
[`scripts/setup_pod.sh`](scripts/setup_pod.sh) (idempotent) + le stub dgl dans
[`src/jepa_v2/programl_compat.py`](src/jepa_v2/programl_compat.py).

### Schéma du graphe ProgramML (confirmé empiriquement)
- **Nœuds** : `text` (l'opcode, ex. `alloca`, `add`, `br`), `type` (0/1/2 =
  instruction/variable/constante), `function`, `block`, `features.full_text`
  (l'instruction complète). → **feature de nœud pour l'entraînement = `text`**
  (on construit un vocab).
- **Arêtes** : `flow` ∈ {`0`,`1`,`2`} = **control / data / call** (les 3
  relations, exactement ce qu'on veut) + `position` (ordre des opérandes /
  numéro de branche).
- ⚠️ **`programl.to_pyg` N'EXISTE PAS** en 0.3.2 → on convertit
  `to_networkx` → PyG nous-mêmes (trivial, le schéma est connu).

### Code scaffolté (squelette, pas encore branché)
| Fichier | État | Contenu |
|---|---|---|
| `pyproject.toml` | ✅ | deps + pins (protobuf<3.21, networkx<3, torchdata 0.7.1…) |
| `README.md` | ✅ | présentation du projet |
| `scripts/pod.sh` | ✅ | helper pour parler au pod (run/put/shell) — **lis-le, le proxy est spécial** |
| `scripts/setup_pod.sh` | ✅ | reproduit tout l'env programl sur un pod neuf |
| `src/jepa_v2/config.py` | ✅ | dataclasses (ModelConfig avec sem_dim/speed_dim, LossConfig, TrainConfig) |
| `src/jepa_v2/programl_compat.py` | ✅ | le stub dgl — **à importer avant programl** |
| `src/jepa_v2/model.py` | ✅ (à tester) | `FactoredEncoder` : tronc GNN 3-relations + 2 têtes `z_sem`/`z_speed` |

### Réutilisable depuis la v1 (`../jepa-ir/src/jepa_ir/`)
- `data/splits.py` — **anti-fuite par hash** (pools encoder/predictor/heldout). À copier tel quel.
- `ir/compile.py` — `compile_to_ir(path, opt_level="-O2")` (clang → IR textuel). Réutilisable.
- `train/vicreg.py` — implémentation VICReg (invariance/variance/covariance). À adapter pour les 2 blocs.

---

## 4. NEXT STEPS (dans l'ordre)

> **MAJ 2026-06-28** — toute la chaîne Steps 1→6 est **codée et fumée** sur le pod
> (build→train→eval marche end-to-end). État par step ci-dessous. Le **gate Step 1
> est ROUGE** (O2≈O3, cf. `docs/results_gate_exebench.md`) : décision actée de
> garder **4 classes z_speed** (configurable via `config.SPEED_GROUPS`) et de
> laisser O2/O3 fusionner d'eux-mêmes. Fichiers livrés :
> `scripts/{probe_exebench,build_cache,train,eval_disentangle}.py`,
> `src/jepa_v2/{exebench,data,loss,vicreg,splits}.py`.
> Deps env ajoutées (manquaient) : `torch_geometric`, `datasets<3`, `zstandard`,
> `huggingface_hub`, `matplotlib`, `scikit-learn` — toutes dans `setup_pod.sh`.

> Chaque étape liste : objectif, fichier(s) à créer, et critère de succès (gate).

### Step 1 — GATE : probe ExeBench (O2 ≠ O3 ?) ⛔ BLOQUANT
**Objectif** : vérifier que sur ExeBench, le graphe ProgramML diffère entre
niveaux `-O` (surtout O2 vs O3), sinon `z_speed` est impossible.
**À créer** : `scripts/probe_exebench.py` qui :
1. prend N (~300) programmes ExeBench,
2. les compile en O0/O1/O2/O3 (`compile.py` de la v1, ou `clang -emit-llvm -S -O{k}`),
3. les passe dans `programl.from_llvm_ir` → `to_networkx`,
4. calcule une **signature de graphe** par (programme, niveau) — p.ex.
   `(n_nodes, n_edges, multiset des `text`, multiset des flux d'arêtes)` ou un
   hash canonique,
5. reporte, sur les paires d'un même programme :
   - % `signature(O0) != signature(O1)`
   - % `signature(O1) != signature(O2)`
   - % `signature(O2) != signature(O3)`  ← **le chiffre critique**
**Gate** : viser **≥ ~50 % distincts O2 vs O3**. Si c'est ~0 % comme AnghaBench :
- soit basculer sur **cbench / MiBench** (vrais programmes entiers, O3 se
  distingue le plus),
- soit accepter que `z_speed` ne sépare que `{O0}` vs `{O1,O2,O3}` (résultat
  faible — à valider avec Ulysse avant).
**Note** : `from_llvm_ir` prend le **texte IR**. `from_cpp` compile en interne mais
ne te laisse pas choisir le `-O` → préfère `clang -O{k} -emit-llvm -S` puis
`from_llvm_ir(open(...).read())`. (À vérifier : signature exacte de `from_llvm_ir`.)

### Step 2 — Récupérer le dataset ExeBench
**À créer** : `scripts/fetch_exebench.py`. ExeBench est sur HuggingFace
(`jordiae/exebench`). Champs utiles : le source C + ses deps (pour compiler en
isolation). S'inspirer de `../jepa-ir/scripts/fetch_anghabench.py`.
**Gate** : ~10k fonctions compilables en O0..O3 disponibles sur le pod sous
`/workspace/jepa-v2/data/`.

### Step 3 — Pipeline données : graphe → PyG + cache
**À créer** : `src/jepa_v2/data.py` avec :
- `nx_to_pyg(g)` : networkx ProgramML → `torch_geometric.data.Data`. Construit :
  - `x` : tenseur d'IDs de vocab (depuis node `text`, `<unk>`=0),
  - `edge_index_0/1/2` : un par flux (control/data/call) — l'encodeur les attend
    sous ces clés (cf. `model.py::_edge_lists`),
- construction du **vocab** des `text` (sur un échantillon, top-K, `vocab_size`),
- un `Dataset` PyG qui pour chaque **programme** rend ses **4 vues** O0..O3,
- **splits anti-fuite** : copier `splits.py` de la v1.
- **cache** : sérialiser les `Data` (les graphes sont chers à reconstruire).
**Gate** : un batch se charge ; `FactoredEncoder(batch)` renvoie `(z_sem, z_speed)`
aux bonnes shapes.

### Step 4 — Loss factorisée
**À créer** : `src/jepa_v2/loss.py`. Pour un batch de `P` programmes × 4 vues `-O` :
- **`L_sem`** : rapprocher dans `z_sem` les 4 vues d'un même programme (VICReg
  invariance entre paires de vues d'un même prog) + variance/covariance pour
  l'anti-collapse.
- **`L_speed`** : rapprocher dans `z_speed` les vues de **même niveau `-O`** entre
  programmes différents (positifs = même `-O`), + variance/covariance.
- **`L_cross`** : décorrélation croisée — minimiser la covariance entre les
  dimensions de `z_sem` et celles de `z_speed` (sur le batch). C'est le terme qui
  **force le disentanglement**.
- `L = sem_weight*L_sem + speed_weight*L_speed + cross_decorr_weight*L_cross`.
- Base VICReg réutilisable depuis `../jepa-ir/src/jepa_ir/train/vicreg.py`.
**Gate** : overfit volontaire sur 8 programmes → la loss descend, et déjà on
observe `cos(z_sem) ↑` entre vues d'un même prog et `cos(z_speed) ↑` entre vues de
même `-O`.

### Step 5 — Boucle d'entraînement
**À créer** : `scripts/train.py` (+ éventuellement `src/jepa_v2/train.py`).
- batch = `batch_programs` programmes, chacun avec ses 4 vues,
- Adam + warmup, grad clip, log des sous-pertes + diagnostics anti-collapse
  (emb_std, rang effectif) comme la v1.
- **smoke test** : 50 steps sans NaN sur petit échantillon, puis **run B200**.
**Gate** : la loss converge ; `emb_std` reste ~1.0 (pas de collapse).

### Step 6 — Éval du disentanglement (le résultat à montrer)
**À créer** : `scripts/eval_disentangle.py`.
- Matrices de cosinus **intra/inter** pour chaque bloc :
  - `z_sem` : haute intra-programme (across -O), basse inter-programme.
  - `z_speed` : haute intra-`-O` (across programmes), basse inter-`-O`.
- **t-SNE** ×2 : coloré par `-O` (doit clusteriser sur `z_speed`) et par
  programme (doit clusteriser sur `z_sem`).
- Optionnel : linear probe (juste pour MESURER la séparation, pas pour entraîner).
**Gate** : séparation visible dans les deux projections = le disentanglement marche.

---

## 5. Comment parler au pod (IMPORTANT — le proxy est spécial)

La clé SSH est **dans `../RUNPOD_CONNECT.md`** (gitignored). Installe-la à
`~/.ssh/jepa_asml_remote` (`chmod 600`). Le proxy RunPod a 3 pièges :
1. `ssh host 'cmd'` est **ignoré** → piper les commandes via **stdin** + `exit`.
2. **`scp`/SFTP ne marchent pas** → transférer les fichiers en **base64 via stdin**.
3. **`printf '...%...'`** mange les `%` avant le remote (casse le `%`-format Python).

Tout ça est encapsulé dans [`scripts/pod.sh`](scripts/pod.sh) :
```bash
scripts/pod.sh shell                       # shell interactif
scripts/pod.sh put scripts/train.py        # upload (base64)
scripts/pod.sh run 'python3 scripts/train.py --smoke'
```
Le pod : **B200, ~183 GB VRAM**, torch 2.8 + CUDA. C'est **partagé** → on isole
par l'env (venv dédié sous `/workspace/jepa-v2`), mais on **n'limite jamais le
GPU** (les runs prennent 100 % de la carte, c'est voulu).

---

## 6. Pourquoi programl a été pénible (pour ne pas refaire les erreurs)

`pip install programl` ne suffit pas. La chaîne de 7 correctifs (tous dans
`setup_pod.sh`) :
1. `protobuf<3.21` (sinon "Descriptors cannot be created directly").
2. `torchdata==0.7.1` (programl importe `torchdata.datapipes`, retiré depuis).
3. deps transitives : `packaging setuptools<81 pandas pyyaml pydantic psutil tqdm requests scipy`.
4. **stub dgl** : `import programl` force `import dgl`, cassé sur torch 2.8
   (lib native graphbolt). On n'utilise pas `to_dgl` → on stube dgl
   (`programl_compat.py`, à importer AVANT programl).
5. **libtinfo.so.5** : le binaire natif `clang2graph-10` exige le VRAI ncurses-5
   (un symlink vers so.6 est rejeté). Installer le `.deb` libtinfo5 de jammy.
6. `networkx<3.0` (sinon `to_networkx` → `KeyError: 'edges'`, rename links→edges).
7. `to_pyg` n'existe pas → conversion maison.

Détails complets dans la mémoire projet `programl-pod-install-recipe.md`.

---

## 7. Pointeurs

- **v1 (référence, beaucoup de code réutilisable)** : `../jepa-ir/`
  - `docs/limitation_non_bijective.md` — pourquoi le gate Step 1 existe.
  - `docs/results_mask15.md` — résultats v1 (encodeur + predictor latent O0→O3).
  - `src/jepa_ir/{ir/compile.py, data/splits.py, train/vicreg.py}` — à réutiliser.
- **Papiers JEPA** : `../*.pdf` (LeCun "A Path Towards…", V-JEPA2, EB-JEPA…).
- **CLAUDE.md** (racine `../../`) — la philosophie JEPA du projet (encodeur, pas
  décodeur ; cible = vecteur ; anti-collapse obligatoire).
```
