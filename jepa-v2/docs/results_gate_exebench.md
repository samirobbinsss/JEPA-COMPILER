# GATE (Step 1) — ExeBench O-level distinctness probe

> Mesure faite le 2026-06-27 sur le pod (B200), `scripts/probe_exebench.py`.
> Reproduit : `scripts/pod.sh run 'python3 scripts/probe_exebench.py --n 800 --split test_real'`.

## TL;DR

**Le gate O2 vs O3 est ROUGE** : sur ExeBench (`test_real`), le graphe ProgramML
est identique entre `-O2` et `-O3` pour **98.4 %** des fonctions. C'est le **même
résultat que sur AnghaBench** (`../jepa-ir/docs/limitation_non_bijective.md`) —
confirmé empiriquement, pas supposé. **MAIS** il existe un signal exploitable sur
les paliers bas : O0 est *toujours* distinct, et O1≠O2 est réel sur les fonctions
de taille moyenne+.

## Méthode

- Compilateur = **clang-10 bundlé par programl** (`from_cpp(copts=["-Ok"])`,
  `version="10"`). Pas de clang système sur le pod ; ce binaire évite tout
  mismatch de version avec `llvm2graph-10`.
- Source = `real_deps + func_def` d'une fonction ExeBench (compile en isolation).
- Signature de graphe = sha1 de `(n_nodes, n_edges, multiset des node.text,
  multiset des edge.flow)`. Deux niveaux « distincts » ⇔ signatures différentes.
- Filtre : on jette les graphes `-O0` de moins de 5 nœuds (déclarations / corps
  vides → 231/800 ici), non informatifs.

## Résultats (`test_real`, 555 programmes complets sur 800 tentés)

| Paire | % distinct |
|---|---|
| **O0 ≠ O1** | **100.0 %** |
| **O1 ≠ O2** | 24.9 % |
| **O2 ≠ O3** | **1.6 %** ← le chiffre critique |

### Stratifié par taille (nœuds du graphe -O0)

| Taille | n | O1≠O2 | O2≠O3 |
|---|---|---|---|
| 0–30   | 100 | 0.0 %  | 0.0 % |
| 30–100 | 241 | 16.2 % | 0.8 % |
| 100–300| 173 | 46.2 % | 2.9 % |
| 300+   | 41  | 46.3 % | 4.9 % |

→ La taille **débloque O1≠O2** (0 → 46 %) mais **pas O2≠O3** (plafonne ~5 %).
Quand O2≠O3 arrive, c'est spectaculaire (ex. O2=68 nœuds → O3=382 : déroulage /
vectorisation), mais c'est rare.

### Partitions des 4 niveaux (qui est identique à qui)

| Partition | count | part |
|---|---|---|
| `O0 \| O1=O2=O3` | 415 | 75 % |
| `O0 \| O1 \| O2=O3` | 131 | 24 % |
| `O0 \| O1 \| O2 \| O3` (4 distincts) | 6 | 1 % |
| autres | 3 | <1 % |

Taille des graphes : médiane 74 nœuds, moyenne 178, max 17309 (vs AnghaBench
~22 médian — ExeBench est bien plus gros, mais ça ne suffit pas pour O3).

## Conséquence pour `z_speed`

Distinguer **4 classes O0/O1/O2/O3 est impossible** depuis l'IR de fonctions
isolées : O2 et O3 sont littéralement le même graphe. Cause = clang-10 +
fonctions isolées (rien à inliner inter-procéduralement, boucles courtes → rien à
vectoriser/dérouler). C'est un **résultat**, pas un bug.

Schéma `z_speed` réaliste, par ordre de force du signal :

1. **3 classes `{O0} / {O1} / {O2≈O3}`** (recommandé) — chaque frontière a un
   signal mesuré (O0≠O1 100 %, O1≠O2 46 % sur ≥100 nœuds). On biaise le dataset
   vers les fonctions ≥ ~50 nœuds (là où O1/O2 se sépare).
2. **2 classes `{O0} / {O1,O2,O3}`** — le plus sûr (100 % séparable), mais
   `z_speed` ne capture qu'« optimisé ou pas » (faible).
3. Garder 4 classes — déconseillé : O2/O3 ajoutent du bruit (positifs qui sont en
   fait identiques aux négatifs), ça sabote la décorrélation.

## Si on veut vraiment O2 vs O3

Il faut un corpus de **programmes entiers** (cbench / MiBench / SPEC), où -O3 a de
quoi mordre (inlining inter-procédural, longues boucles). C'est l'extension
naturelle ; hors scope de la passe ExeBench actuelle.
