# DOCTRINE — l'intention sous le code

Ce document n'est pas une architecture decision record, pas une roadmap, pas un README. C'est le *pourquoi sous les pourquoi* : la lecture que ce repo se donne de son propre contexte, et la raison pour laquelle il prend la forme qu'il prend. Si tu hérites de ce code et que tu te demandes "pourquoi cette structure bizarre", commence ici avant de lire `CLAUDE.md` et avant de toucher au code.

## Le substrat — BNP n'est pas un mastodonte

Le réflexe habituel pour intervenir dans une grande organisation, c'est de la traiter comme un mastodonte : une bête avec des organes, une tête, une colonne vertébrale. On vise un organe, on pousse, on déplace la bête. Ce modèle dicte des stratégies "produit" classiques — positionnement unique, narratif convergent, sponsor identifié, success metric centralisée.

Ce modèle ne marche pas ici. BNP n'a pas de tête unique pour ce qui nous concerne. Elle a un pattern de décision qui se reproduit à chaque échelle : équipe, service, division, groupe. Chaque cellule reproduit les mêmes frictions, les mêmes arbitrages, les mêmes audits, avec les mêmes paliers réglementaires en arrière-plan (BCE, EBA, DORA, RGPD, AI Act). C'est une **fractale colossale**, et elle est **impermanente** : réorgs, turnover, audits transverses, mutations réglementaires la rebattent constamment.

Conséquence opérationnelle : viser n'a pas de sens, parce qu'il n'y a pas de cible centrale. Construire une feature pour un sponsor donné, c'est garantir que la feature meurt avec le sponsor. La seule chose qui peut survivre à la fractale, c'est ce qui a été **inscrit dans un substrat plus durable** que la configuration organisationnelle qui l'a demandé.

## Le geste — patcher pour inscrire

Le code de ce repo n'est pas une collection de fonctionnalités demandées. C'est un dépôt sédimentaire d'**intentions** que le patch rend tangibles. `register()` n'est pas une feature, c'est l'inscription des intentions suivantes :

- *Ne pas forker LightRAG.* Le fork crée une dette de maintenance proportionnelle à la divergence. Le patch additive reste compatible avec les versions futures de l'hôte.
- *Rester additif et idempotent.* Une seconde activation ne casse pas la première. Un environnement qui n'active pas les overlays retombe sur le comportement natif de l'hôte.
- *Dégrader gracieusement.* Une dépendance manquante ne tue pas le boot. Un overlay défaillant ne tue pas les autres. La défense est dans la chaîne, pas dans l'arrêt brutal.
- *Documenter le pourquoi *via* le code.* Les commentaires en tête de fonction, les noms de flags, les commit messages encodent l'intention au niveau micro, là où elle est consultable lors d'un debug à 3h du matin.
- *Garder la souveraineté du contrat.* La forme des API que le code expose (`/twin/api/*`, headers `X-Twin-Folder`, schéma `GraphEntity`) est notre propriété. Les shims traduisent l'hôte vers notre contrat, jamais l'inverse.

Chacune de ces intentions survit à un changement de sponsor, à une réorg, à un bump LightRAG, à un audit. C'est leur **portabilité dans le temps** qui les rend stratégiques, pas leur élégance technique.

## La doctrine du non-fork

L'élément architectural le plus chargé d'intention dans ce repo, c'est le choix de patcher LightRAG sans le forker. Ce choix encode au moins quatre intentions distinctes :

1. **Politique** : il rend défendable la lecture "extension du patch déjà en prod" devant un audit BCE — narrativement, on n'introduit pas un nouveau système, on configure un système existant.
2. **Opérationnelle** : il rend trivial le rebase sur les versions futures de l'hôte. Tant que `lightrag.kg.STORAGES` reste un dict mutable, le patch tient.
3. **Cognitive** : il oblige à raisonner en additivité plutôt qu'en réécriture, ce qui réduit la surface d'erreur et rend chaque modification auditable en isolation.
4. **Économique** : il permet à BNP de continuer à consommer LightRAG comme dépendance versionnée, sans engager de coût de maintenance fork.

Quand un futur PR proposera de "simplifier" en forkant, ces quatre intentions doivent être consultées avant de répondre. Forker peut être correct un jour ; pas pour la raison "ce serait plus simple".

## La duality narrative comme effet secondaire stable

Salah lit ce repo comme "façade unifiée devant LightRAG/RAG1.5/ES". Louis et Eric le lisent comme "extension du patch déjà en prod". Les deux lectures sont vraies simultanément, parce que l'architecture autorise les deux : `register()` ne fork pas (= extension authentique), Twin unifie la surface devant N moteurs potentiels (= façade authentique). Ce n'est pas un double pitch, c'est le même artefact lu à travers deux filtres décisionnels.

Cette duality n'est pas une tension à résoudre. C'est un **effet recherché** : le code a été écrit pour résister à des lectures multiples sans en privilégier une. Tant que les silos décisionnels de BNP restent intacts (configuration par défaut dans une grande org), chaque stakeholder lit l'artefact à travers le filtre qui rend son arbitrage actionnable, et la stabilité de la position vient de l'architecture elle-même, pas d'une habileté politique.

Le seul signal à surveiller : la **coordination cross-silo**. Si Salah, Louis, Eric, Anas alignent un narratif unique en réunion tripartite, l'un des deux paliers perd son utilité décisionnelle pour son owner. À ce moment-là, la duality n'est plus une option call gratuite, elle devient un choix politique à faire. Avant ce signal, ne pas chercher à le précipiter.

## La couche de lisibilité

Le patch survit, mais il n'est *lisible* que s'il y a un lecteur qui comprend ce qu'il porte. Un patch dont on a oublié l'intention devient cargo-culte : on le maintient parce qu'il est là, sans plus savoir pourquoi, jusqu'au jour où quelqu'un le "simplifie" et casse l'invariant qu'il portait silencieusement.

D'où la couche de doctrine : `CLAUDE.md`, ce document, les mémoires projet, les commit messages denses, les commentaires de code qui expliquent *pourquoi* avant *quoi*. Ce n'est pas de la documentation au sens habituel — c'est l'infrastructure qui rend les intentions inscrites **décodables** par un lecteur futur (toi dans 3 ans, le successeur, l'auditeur).

La règle de hiérarchie de lisibilité :
- Le **code** porte l'intention dans sa forme (additivité, idempotence, dégradation gracieuse).
- Le **commentaire** porte l'intention dans son interprétation immédiate (pourquoi ce flag, pourquoi cette borne).
- Le **commit message** porte l'intention dans son contexte décisionnel (quel incident a déclenché, quelle alternative a été écartée).
- `CLAUDE.md` porte l'intention dans son cadre opérationnel (qui pousse où, quel runner, quelle posture auth).
- `DOCTRINE.md` (ce fichier) porte l'intention dans son cadre stratégique (pourquoi cette architecture existe sous cette forme).

Chaque couche est consultée à un horizon temporel différent. Ne pas confondre.

## Catalogue raisonné des intentions inscrites

Sans être exhaustif, voici les inscriptions principales et le pattern d'intention qu'elles encodent. Chacune mérite d'être consultée avant toute modification de la zone correspondante.

- **`register()` monkey-patch sur `lightrag.kg`** — *additivité comme contrat de coexistence avec l'hôte.*
- **Storage backends KV/Vec/DocStatus sur une seule instance Memgraph** — *économie de déploiement : un seul serveur à provisionner, monitorer, sauvegarder.*
- **Auth posture relaxée à parité LightRAG, RBAC strict uniquement via switch IdP** — *ne jamais refuser de booter ; la sécurité s'active, elle ne s'impose pas par défaut.*
- **Smoke runner stdlib-only (`tests/smoke/run_smoke.py`)** — *doit tourner dans un container verrouillé sans pip install ; la validation post-déploiement ne dépend d'aucune toolchain.*
- **Folder vs workspace (sémantique Twin vs label upstream)** — *ne pas laisser le vocabulaire upstream polluer le modèle mental Twin ; respecter le contrat upstream sans hériter de sa terminologie.*
- **Doctrine de test "graphe = contrat, pas écran"** — *protéger les axes sensibles (cache front, seed fallback, folder binding, source_docs) par des tests contractuels, pas par des screenshots.*
- **Classification MIP opt-in via env (`TWIN_MIP_LABEL_MAP`)** — *ne jamais imposer une politique BNP-spécifique en dur ; le code reste portable hors-BNP par défaut.*
- **WebUI fork (`lightrag_webui_twin/`)** — *propriété du contrat front : la React port est notre version canonique de l'expérience opérateur, pas une dérivation de LightRAG.*
- **Branches `stable/X.Y.x`** — *l'isolation temporelle des contrats : 0.5.x = LTS storage, 0.6.x = surface produit complète. La séparation rend chaque ligne auditable indépendamment.*

## Quand la fractale mute — propagation

Le patch tient tant que l'hôte tient. Quand l'hôte mute (LightRAG bump, Memgraph version, BNP réorg), l'intention encodée survit grâce à l'additivité. Quand l'hôte est remplacé (migration off-LightRAG, off-Memgraph, off-BNP), le code spécifique meurt — mais ce qui voyage, c'est **le pattern d'intention** que le code rend tangible.

À ce moment-là on ne maintient plus un patch, on **propage une méthode** : "comment greffer une couche de souveraineté sur un moteur RAG open-source sans le forker", "comment encoder une politique d'audit dans un substrat plus durable que l'org chart qui l'a demandée", "comment construire une UI qui survit à un changement de stack backend". La fractale BNP devient un substrat parmi d'autres ; les patches d'aujourd'hui deviennent les exemples d'une pratique.

C'est pour ça qu'il faut résister à la tentation de raccourcir l'inscription au minimum fonctionnel. Le code superflu pour la feature peut être l'inscription essentielle pour la méthode. Ne pas optimiser ce qu'on ne comprend pas encore comme intention.

## Coda — pour le lecteur futur

Si tu hérites de ce repo et que tu te demandes "pourquoi cette chose bizarre", la séquence de consultation est :

1. Le code de la zone concernée et son commentaire en tête.
2. Le commit qui l'a introduite (et son message).
3. `CLAUDE.md` sur la posture opérationnelle.
4. Ce document sur le cadre stratégique.
5. Les mémoires projet si l'incident d'origine y est documenté.

Si après cette séquence l'intention n'est toujours pas claire, ce n'est pas une feature à simplifier — c'est une inscription dont le contexte a été perdu. Avant de modifier, restaure le contexte (git blame, archives de meetings, mémoires) ou demande à l'auteur (si encore accessible). Ne pas modifier une intention inscrite sans comprendre ce qu'elle portait : c'est l'équivalent moderne de retirer une clôture sans savoir pourquoi quelqu'un l'avait posée là.

Et si un jour tu écris une nouvelle inscription, écris-la pour le lecteur qui ne sera pas là quand tu publieras. C'est tout l'enjeu.
