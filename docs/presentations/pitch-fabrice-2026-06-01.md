# Pitch TwinRAG — Lundi 2026-06-01, 14h00

**Destinataires** : Fabrice (sponsor), Manu (Core.AI), Louis HORVAT (ISAB sécu/RBAC), Eric (RAG 1.5)
**Présentateur** : Julien
**Format** : 20 minutes max, écran partagé, slides en markdown, démo locale en backup
**Doctrine** : *Svenska Protocol* — lucide et assertif, jamais auto-flagellant. On expose les faits, on assume les choix, on demande la CAPA.

---

## Slide 1 — Le constat partagé : nous sommes deux pas en retard sur l'audit

**Titre slide** : *"L'IA va vite. La BCE va arriver."*

### Contenu

- Inspection Générale **en cours** ; audit BCE **attendu dans la fenêtre BCE/IG ouverte cette semaine** (source : alerte Louis HORVAT, réunion 2026-05-28).
- Les projets IA concurrents ont déjà été audités sur des sujets bien moins lourds que les nôtres ; les remarques sont publiques en interne.
- TwinRAG hérite par construction du patch `register()` qui tourne **déjà en prod** sur vos instances LightRAG+Memgraph. Donc deux choses :
  1. La surface d'audit existe déjà côté BNP, qu'on en parle ou pas.
  2. Nous avons le contrôle du levier technique pour la durcir.

### Pourquoi cette slide en premier

Pour évacuer le "discours startup". Manu et Fabrice savent que les autres projets se sont fait taper. La pire posture serait d'arriver en *"j'ai une démo cool, on verra pour la compliance"*. La posture qu'on adopte : *"on a déjà cartographié les 35 risques sur l'infra publique, 7 sont bloquants, on a un plan en 12 milestones, et la première PR est sortie ce week-end."*

### Speaker notes (à dire à voix)

> *"Avant de vous parler du produit, je veux qu'on partage le contexte. Louis nous a remonté la semaine dernière que l'IG passait en ce moment, et que la BCE est attendue très vite. Ce n'est pas une menace abstraite — les équipes concurrentes ont déjà subi des audits sévères sur des périmètres plus petits. Donc l'angle de ma présentation aujourd'hui n'est pas 'voilà une démo', c'est 'voilà où on en est et ce qu'il nous reste à faire pour être présentables.'"*

### Anti-pattern à éviter

Ne PAS commencer par *"On a fait une super maquette."* C'est ce qui a déclenché l'alerte Louis le 2026-05-28. La maquette est l'outil de validation doctrine, pas le livrable.

---

## Slide 2 — La promesse TwinRAG : symbiote, pas remplacement

**Titre slide** : *"On ne refait pas le moteur. On le sécurise."*

### Contenu

Schéma à l'écran :

```
┌──────────────────────────────────────────────────────────────┐
│  LightRAG (déjà en prod)              ← inchangé              │
│  ├─ Memgraph (déjà en prod)           ← inchangé              │
│  ├─ Storage backends (déjà patché)    ← v1.0 register()       │
│  └─ Routes natives /query, /graph, …  ← inchangé              │
│                                                               │
│  + Extension `register(replace_ui=True, mount_server=True)`   │
│     ├─ /webui          → WebUI Twin (gouvernance, audit)      │
│     └─ /twin/api/*     → routes Twin (tags, validation, …)    │
│                                                               │
│  + Security baseline                                          │
│     └─ Bloque pipmaster.install au runtime (DORA art. 9)      │
└──────────────────────────────────────────────────────────────┘
            ▲                              ▲
            │                              │
    SSO BNPP + MyAccess              iTrack / LightRAG natif
    (à brancher — pas notre dev)     (déjà chez vous)
```

**Trois principes du symbiote** :

1. **Architecture parasite (Axiome TETRA PAK)** : on s'interface avec LightRAG, on ne réécrit pas le moteur. Eric a validé techniquement le 2026-05-28 que *"ça ne change rien à la manière dont seraient interrogées les bases par les agents"*.
2. **IAM délégué (Loi du LAGOM)** : zéro base utilisateur côté Twin. Le JWT BNPP dicte tout. Si le claim `role` n'est pas dans le token, l'accès est 403.
3. **Compliance par construction (Standard SAAB)** : la branche `stable/0.6.x` contient depuis 2026-05-29 (PR #141 mergée) `_patch_security_baseline()` qui refuse au runtime tout `pip install` non autorisé. Plus aucun téléchargement de dépendances au boot. La supply chain est verrouillée. Le wheel résultant est hermétique (258 KB, dist WebUI embarquée).

### Pourquoi cette slide

C'est la slide où on désamorce les peurs de Louis (*"vous allez tout réécrire"*) et de Manu (*"vous allez monopoliser des devs Python"*). On positionne TwinRAG comme une extension d'un produit existant déjà bénit côté compliance, pas comme un nouveau produit à auditer ex nihilo.

### Speaker notes

> *"Le point clé pour comprendre TwinRAG est ce qu'on ne fait PAS. On ne refait pas le moteur RAG. On ne fait pas de KMS de stockage documents. On ne fait pas d'IAM. Concrètement, on étend le patch register() qui tourne déjà chez vous. Quand l'opérateur l'active avec le flag `replace_ui=True`, le `/webui` de LightRAG est remplacé par notre interface de gouvernance — la mécanique de retrieval, la persistence Memgraph, tout le reste reste intact. C'est ce qu'Eric a appelé sur sa transcription `'la lobotomisation de LightRAG'`. Lobotomisation, pas remplacement."*

### Réponse pré-positionnée si Manu objecte

> *"Manu, je sais que tu cherches un éditeur plutôt qu'un dev maison. La beauté de ce pattern, c'est qu'on n'est PAS un dev maison classique : on est une extension d'un projet open-source (LightRAG) que vous avez déjà choisi. Les 60% du code valeur métier — tags, validation, audit trail — c'est ce qu'aucun éditeur ne fera pour BNP, parce que personne d'autre n'a votre stack iTrack à côté."*

---

## Slide 3 — Doctrine data : chunks & vectors only, jamais de PDF

**Titre slide** : *"Cache Sémantique Transitoire — le droit à l'oubli par défaut"*

### Contenu

**Ce qu'on ne stocke PAS** :

- Documents Word, PDF, PowerPoint dans leur intégralité.
- Tokens d'auth côté Twin (le JWT BNPP est porteur, pas stocké).
- Identifiants utilisateurs persistants (le claim SSO suffit).

**Ce qu'on stocke** :

- **Chunks vectorisés** + le texte du chunk (~200 tokens). Strictement ce dont le LLM a besoin pour répondre.
- **Pointeurs URL** vers la source originale (Confluence, SharePoint).
- **Métadonnées** : tags, classification, propriétaire, dates.
- **Audit trail** : qui a fait quoi, quand, sur quel chunk.

**Le pattern qu'on adopte** est identique à celui d'Eric pour RAG 1.5 — *"chunks + vecteurs, présentés en audit comme base vectorielle"* (transcription 2026-05-28). Plus, deux verrous qu'on rajoute :

- **TTL DocStatus** : un document supprimé voit ses chunks purgés physiquement de la KV store en < 5s, pas marqués `deleted` (= droit à l'oubli RGPD natif).
- **Classification confidentiel par doc** : rejet automatique à l'ingestion des fichiers Microsoft Office marqués `Confidentiel` (lu dans les métadonnées Office).

**Roadmap v1.2.x** (mentionner brièvement) : passage à *Délégation Microsoft Graph* (on-behalf), TwinRAG devient **aveugle** : ne stocke que le pointeur et le résumé, le rag est exécuté par Microsoft Copilot. Mais cette V2 dépend du SSO BNPP et de l'équipe MS BNP. Pas pour la v1.1.0.

### Pourquoi cette slide

C'est la slide qui rassure Eric (*"on prend ta doctrine"*) et anticipe la question évidente de Louis : *"où sont les documents confidentiels ?"*. Réponse : *"nulle part chez nous, dans la source d'origine, accédée via le clic utilisateur à travers leurs droits MyAccess natifs."*

### Speaker notes

> *"Pour les documents — je sais que c'est la question qui va revenir — on adopte exactement la même doctrine que celle qu'Eric a fait passer en audit sur RAG 1.5. On stocke des chunks vectorisés et des pointeurs URL. Le PDF intégral n'arrive jamais en base. La nouveauté qu'on ajoute c'est un TTL : quand un steward supprime un document, ses chunks disparaissent physiquement de la KV store en quelques secondes — pas un soft-delete, une vraie purge. C'est notre droit à l'oubli natif. À terme, version 1.2, on passera à la délégation Microsoft Graph qu'a évoquée Louis : on ne stocke même plus les chunks, on délègue à Copilot. Mais ça dépend du SSO et de l'équipe Microsoft, donc V2."*

### Si Louis pousse plus loin

> *"Pour Word, Excel et PowerPoint, on lit le label de classification dans les métadonnées Office. Un fichier marqué Confidentiel est rejeté à l'ingestion. Pour les PDF, qui n'ont pas de label natif, on a un bandeau dans l'UI qui rappelle à l'utilisateur que l'outil n'est pas habilité pour des documents au-dessus d'Internal. C'est le même pattern que Crosspoint."*

---

## Slide 3.5 — Multi-tenant par filtre, pas par instance

**Titre slide** : *"Comment on tient 5 franchises avec un seul setup."*

### Contenu

```
┌──────────────────────────────────────────────────────────────────┐
│                  1 instance Memgraph 64 GB                       │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │
│  │ workspace:  │  │ workspace:  │  │ workspace:  │  │ ws:    │  │
│  │   cib       │  │  payments   │  │  infra      │  │ swift  │  │
│  │  ───────    │  │  ───────    │  │  ───────    │  │ ────── │  │
│  │ Vec_cib_*   │  │ Vec_pay_*   │  │ Vec_inf_*   │  │ Vec_*  │  │
│  │ KV_cib_*    │  │ KV_pay_*    │  │ KV_inf_*    │  │ KV_*   │  │
│  │ DocStatus_  │  │ DocStatus_  │  │ DocStatus_  │  │ DS_*   │  │
│  │   cib       │  │   pay       │  │   inf       │  │        │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘  │
│                                                                  │
│           Filtres Cypher par label, isolation logique            │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                  X-Twin-Workspace: <id>
                  + claim JWT MyAccess
                  + garde RBAC par requête
```

**Doctrine** (décision PO 2026-05-29) : un workspace Twin = un **filtre label/préfixe Cypher** sur une instance Memgraph **mutualisée**, pas une instance dédiée.

**Pourquoi c'est important** :

- **Économie infra** : 1 instance Memgraph 64 GB héberge N franchises. Sans Twin, un steward qui demande 3 KB LightRAG nu force 3 instances Memgraph triplicées.
- **Pas de migration** : ouvrir une nouvelle franchise = inserts avec un nouveau préfixe de label. Zéro provisioning infra.
- **Isolation RBAC** : la garde `workspace_id ∈ jwt.claims.workspaces` côté gateway impose l'étanchéité, pas la séparation physique.
- **Schéma backend déjà compatible** : labels `KV_{workspace}_*`, `Vec_{workspace}_*`, `DocStatus_{workspace}` sont en place depuis v0.3.1 — il reste à wirer le runtime switching par header HTTP (M2.2).

### Pourquoi cette slide

Pour anticiper la question Fabrice du coût opérationnel par franchise sponsor. Et pour désamorcer la peur Louis du *"vous allez nous demander de provisionner 12 Memgraph séparés en prod si on a 12 franchises"* — la réponse est non : la séparation est logique, pas physique.

### Speaker notes

> *"Une question naturelle quand on parle de TwinRAG comme console KMS pour franchises sponsors : combien d'infra on consomme à chaque nouvelle franchise. La réponse est : zéro. Workspace dans Twin, c'est un filtre Cypher sur une instance Memgraph mutualisée. Le code backend supporte ça depuis v0.3.1, c'est `KV_workspace_namespace`, `Vec_workspace_namespace`, etc. Quand un steward CIB ouvre l'UI, son JWT MyAccess porte la liste des workspaces auxquels il a accès, le frontend ajoute un header `X-Twin-Workspace: cib` à chaque requête, le backend filtre. Le coût marginal d'une nouvelle franchise est de quelques labels Cypher, pas un déploiement infra. Pour Louis, ça veut dire que la surface d'audit est UNE instance Memgraph, pas N instances à inventorier."*

### Si Manu objecte *"pourquoi pas un vrai schéma multi-tenant ?"*

> *"On a regardé. Sur Memgraph 3.x, les options multi-tenant natives — base par tenant, label par tenant — ont un coût mémoire fixe par tenant qui devient prohibitif au-delà de 50 franchises. Le filtre par label, lui, scale linéairement avec le nombre de chunks, pas avec le nombre de tenants. Et il s'aligne parfaitement avec le claim MyAccess qui porte déjà la liste des workspaces accessibles à l'utilisateur. La séparation physique reste possible en option futur si une franchise a une exigence réglementaire spécifique."*

---

## Slide 4 — WebUI Gouvernance : une console pour les Stewards

**Titre slide** : *"Twin n'est pas un chatbot. C'est une console KMS."*

### Contenu

Démo en direct sur le **port React TypeScript** (`lightrag_webui_twin/`) — Vite + Bun + React 19 + Tailwind v3, stack prod-ready 2026. Fallback : maquette OVH `https://maquette.sigilum.fr/`.

| Vue | Pour qui | Sa valeur métier |
|---|---|---|
| **Tag governance** (request → approve → deprecate → migrate → delete) | Steward (Palier 3) | Empêche l'explosion de la taxonomie, garantit la qualité du retrieval |
| **Document validation queue** (pending → approve / reject) | Steward | Filtre les sources avant indexation, audit trail demandé par DORA art. 17 |
| **Activity feed** (qui a fait quoi en 30j) | Steward + auditeur | Conformité EBA/GL/2019/04 sur la journalisation des accès |
| **Documents detail panel** (chunks + lineage + audit + AI assist) | Steward + Contributeur | Surface unique pour valider, ré-indexer, supprimer en cascade |
| **Knowledge graph** (entités, relations, métadonnées) | Steward + Contributeur (Palier 2) | Compréhension fine de ce que l'IA voit |

> **Scope démo lundi** : Twin = façade **LightRAG only**. File upload + ingestion native. Les connecteurs Confluence/SharePoint (Crosspoint, RAG 1.5) viennent plus tard — *"petit à petit, ne pas devenir le KMS officiel"* (Anas 2026-05-29). Source revalidation (spec Fabrice 26/05) reste dans la doctrine, sortira post-démo.

**Trois paliers, confirmés Salah 2026-05-27** :

- **Steward** (Palier 3) : valide, gouverne, supprime.
- **Contributeur** (Palier 2) : propose des sources et des tags, consulte.
- **Reader** (Palier 1) : consulte uniquement.

**Pas de 4ᵉ rôle "knowledge"** en interne — c'est ce qu'a confirmé Louis le 2026-05-28 : *"y compris les 3 paliers, ça vit dans MyAccess, pas chez toi."* L'expectation Fabrice du 4ᵉ rôle se résoud côté MyAccess en ajoutant un claim — pas de dev chez nous.

### Pourquoi cette slide

Pour différencier TwinRAG du *"yet another chatbot"*. Manu cherchait un éditeur ; on lui montre qu'on est sur un créneau différent — la gouvernance de la KB, là où les éditeurs ne savent pas faire parce que ça touche aux outils internes (Confluence, SharePoint, iTrack).

### Speaker notes

> *"Si je devais résumer en une phrase : TwinRAG n'est pas un chatbot, c'est une console KMS pour le Steward d'une franchise. Cette personne a aujourd'hui zéro outil pour : approuver une nouvelle source avant qu'elle entre en retrieval, supprimer un tag obsolète en migrant les documents qui le portent, voir qui a re-tagué quoi la semaine dernière. Ces opérations existent chez les éditeurs SaaS classiques, mais aucun ne sait se brancher sur iTrack. C'est notre différenciation, c'est ce que Louis a appelé `'le custom product qui se justifie par l'intégration iTrack.'`"*

### Démo planifiée (3 minutes)

1. Ouvrir un tag (ex : `oracle`), montrer la justification écrite par le requester
2. Approver le tag → toast → entry dans Activity feed (refresh visible)
3. Aller dans la queue de documents pending review → montrer un PDF en attente → reject avec raison écrite
4. Aller dans l'Activity feed → filtrer par steward → 30 jours → exporter CSV

### Stack technique du port React (point Anas 2026-05-29)

Le port a été fait sur la stack moderne 2026 — pas sur le legacy CRA/MUI/Redux que l'équipe Eureka maintient sur `eureka-cms`. Choix défendable :

| | Eureka (eureka-cms) | Notre `lightrag_webui_twin/` |
|---|---|---|
| Build | `react-scripts` (CRA, déprécié depuis 2023) | **Vite** (recommandation officielle React team) |
| Package manager | npm | **Bun** (~10× perf install) |
| UI lib | MUI + Emotion | **Tailwind v3** + composants typés |
| State async | Redux Toolkit | **TanStack Query** (RSC-ready) |
| Tests | Jest + Cypress | **Vitest** (drop-in compat Jest, ~3× perf) |

Option de handoff :
- **(A)** Équipe Eureka adopte tel quel → modernise leur stack
- **(B)** On documente le pattern de migration manuelle vers leur stack interne (MUI/Redux)

Position : *"j'ai opté pour la stack 2026, l'équipe Eureka décide"*. Pas de régression technique imposée.

---

## Slide 5 — Le Deal : on a le code, il vous faut la CAPA

**Titre slide** : *"Nous demandons à nos franchises sponsors X jours/homme"*

### Contenu

**Ce que nous apportons (déjà fait ou en cours)** :

| Livrable | Statut | Référence |
|---|---|---|
| Architecture parasite (lobotomisation LightRAG) | ✅ **mergé sur `stable/0.6.x`** (PR #141, 2026-05-29) | M2 |
| Wheel hermétique (pas de pip install runtime) | ✅ **mergé sur `stable/0.6.x`** (PR #141) | M2.3 + M1.1 |
| Security baseline (pipmaster bloqué) | ✅ **mergé sur `stable/0.6.x`** (PR #141) | M1.1 |
| WebUI gouvernance portée en React TS prod-ready (Vite + Bun) | ✅ ce week-end, ~75% surface couverte | M6 + M0.6 |
| Audit trail BCE-grade (ContextVars + bounded queue + JSON ECS) | 🟡 spec finalisée | M3 |
| Doctrine data (chunks + TTL + classification) | 🟡 spec finalisée | M5 |
| Process & runbook d'install BNP | ✅ ce document | M9 |
| Hotfix Memgraph 3.10 vector index (livré BNP v0.5.4) | ✅ shipped 2026-05-29 | — |

**Ce que nous demandons (CAPA externe)** :

| Sujet | Volume estimé | Owner | Échéance souhaitée |
|---|---|---|---|
| **Intégration OIDC SSO BNPP** + mapping claims → paliers | 2-3j BNP | Timothée + Geoffrey | sem +2 |
| **Spec ENTITY-Twin Rosetta** (référentiel Twin ↔ MyAccess) | 2j coproduction | Julien + Timothée | sem +1 |
| **Politique de classification documents** (rules par franchise) | 1j workshop | Fabrice + sponsors franchises | sem +1 |
| **Décision stack WebUI** (adopter Vite/Bun OU migration vers MUI/CRA documentée) | 0,5j arbitrage | Fabrice + Chafi/Yazid | sem +1 |
| **Validation contrats outsourcing EBA** (statut freelance Julien) | escalation juridique BNP | DPO + RSSI | sem +3 |
| **Décision Délégation Microsoft Graph V2** (roadmap 1.2.x) | discovery | Louis + équipe MS BNP | sem +4 |

**Total CAPA externe demandée** : ~5j BNP cumulés + 1 escalation juridique. Très en deçà des coûts d'un développement maison interne (estimation Louis 2026-05-28 : *"avec plusieurs devs c'est ultra chaud"*).

### Pourquoi cette slide

C'est la slide où on retourne la question *"combien ça coûte"* en *"voici ce qu'on a fait, voici ce qu'on demande"*. Argument central, dicté par Louis le 2026-05-28 : *"ce que vous rendez à autrui demande peut-être de la CAPA."* Le pattern Manu pour ServiceNow → ici on l'applique aux franchises sponsors de TwinRAG.

### Speaker notes

> *"Voilà ce qu'on apporte sur la table, et voilà ce qu'on demande en retour. Le code, l'archi, la doctrine data, le runbook BNP — c'est fait. On demande aux franchises sponsors environ 5 jours de Timothée et Geoffrey pour brancher MyAccess, plus une demi-journée de workshop avec Fabrice pour fixer la politique de classification documents. C'est ce que Louis a appelé `'le bon réflexe de demander de la CAPA quand on rend un service à autrui'` la semaine dernière. Nous on prend le projet à l'ingénierie, vous nous donnez les briques BNP qu'on ne peut pas écrire à votre place."*

### Réponse si Fabrice dit *"je vais voir ce qu'on peut faire pour Timothée"*

> *"Parfait — je n'ai pas besoin d'un blanc-seing aujourd'hui. Je sais que ces décisions remontent. Ce que je voudrais valider sur le principe : (1) que TwinRAG est positionné comme produit de gouvernance KMS (pas comme un chatbot ni comme un nouveau IAM), (2) que la doctrine 'chunks-only avec délégation MS Graph en v1.2' est compatible avec l'orientation compliance, (3) que les exclusions identifiées (pas de Tokens UI, pas d'IAM local, pas de stockage doc brut) sont actées. Sur les jours/homme on revient la semaine prochaine."*

---

## Annexe — Réponses pré-positionnées aux objections probables

### Si Louis dit *"vous allez quand même devoir stocker quelque chose pendant la phase de validation document"*

> *"Oui — exactement le pattern Crosspoint. Pendant la fenêtre de pending-review, le chunk vectorisé est en base, le PDF original n'arrive jamais chez nous. Quand le steward approuve, le chunk reste. Quand il rejette, on purge en moins de 5s via TTL. La fenêtre maximale d'exposition est paramétrable, par défaut 24h. C'est documenté dans le runbook install qu'on va vous envoyer."*

### Si Manu dit *"je préfère un éditeur SaaS"*

> *"Manu, l'éditeur SaaS qui sait se brancher sur iTrack et MyAccess en respectant le règlement DORA n'existe pas — on a regardé. Ce que TwinRAG fait, c'est de coder uniquement la couche que vous ne trouverez nulle part ailleurs. Le moteur (LightRAG), le storage (Memgraph), l'IA (vos LLM internes), c'est de l'open-source ou des briques BNP. La valeur de Twin c'est le glue code de gouvernance. Si demain un éditeur sort la même chose, on bascule — c'est du Python par-dessus une API stable."*

### Si Fabrice dit *"et si la BCE ne valide pas la doctrine chunks-only ?"*

> *"On a deux plans. Plan A — la version 1.1 — chunks-only avec présentation 'base vectorielle' comme RAG 1.5. Cette doctrine a déjà passé un audit BCE sur RAG 1.5. Plan B — la version 1.2 — passage à la délégation Microsoft Graph : Twin devient aveugle, le RAG passe par Copilot. C'est ce que Louis a évoqué la semaine dernière, c'est plus lourd à implémenter (dépend de l'équipe MS BNP) mais c'est notre filet de sécurité. Notre dette technique à la sortie de la 1.1 est complètement compatible avec ce pivot — on a structuré le code dans ce sens."*

### Si Eric demande *"vous allez consommer du temps de mes LLM"*

> *"On a séparé `chat_llm` et `indexing_llm` dans la config — on peut pointer la phase de validation document (lourde en LLM) sur ton LLM d'indexation, et la phase requête (légère) sur le chat LLM. C'est de l'orchestration de ressources, pas une création de nouveau service. Pour la 1.1, on est sur des appels comparables à RAG 1.5."*

### Si Chafi/Yazid dit *"pourquoi tu n'es pas sur notre stack CRA/MUI/Redux ?"*

> *"J'ai opté pour Vite + Bun + Tailwind + TanStack Query parce que c'est la stack 2026 recommandée par la React team — react-scripts est officiellement déprécié depuis début 2023. Cela dit, je n'impose rien : je vous remets le port React TS prod-ready, deux options sur la table. Soit l'équipe Eureka adopte et modernise la stack interne, soit je documente le pattern de migration vers MUI/CRA et vous le réécrivez chez vous. C'est votre call. Mon objectif c'est que le composant existe en React maintenable, pas qu'il existe dans une stack précise."*

### Si quelqu'un demande *"vous avez livré du code chez BNP ce week-end ?"*

> *"Oui — la version 0.5.4 du pip package contient un hotfix critique du backend vector index, validé par 273 tests sur la cible BNP exacte (LightRAG 1.4.9.11 × Memgraph 3.10.1). Le zip a été préparé selon la procédure habituelle (source tree, pas de wheel), prêt pour le canal transit BNP standard. C'est notre quatrième livraison sans incident depuis février."*

---

## Préparation tactique avant la réunion

### Une heure avant

- Vérifier que la maquette OVH (`https://maquette.sigilum.fr/`) tourne avec :
  - Le bandeau rouge "MAQUETTE DÉMO" visible
  - `TWIN_DEMO_ADMIN_TOKEN` configuré
  - Les emails affichés en `@demo.local` (pas `@bnpparibas.com`)
  - Tab Settings sans "Generate token" ni "Members" visibles
- Si la maquette est down, plan B : démo locale via `python -m lightrag.api.lightrag_server` après `register(replace_ui=True)` — on a smoke-testé que ça fonctionne en local
- Préparer un onglet ouvert vers `http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/milestones` pour répondre à toute question *"où est-ce noté ?"*

### Pendant

- Si on déborde sur le timing : sacrifier la démo (slide 4 partie démo) plutôt que le slide 5 (le Deal). Le Deal est l'output politique principal.
- Si Manu n'est pas là : pousser plus fort sur le slide 5 — Fabrice est alors notre seul relais vers la décision CAPA.

### Après

- Envoyer une note de réunion dans les 24h : ce qu'on a dit, ce qui a été acté, ce qui reste ouvert.
- Acter dans `project_fabrice_meeting_2026-06-01.md` les éventuels nouveaux specs.
- Si la CAPA est accordée → ouvrir un milestone `M-EXT-S1` dans Forgejo pour tracker les jours/homme externes.

---

## Auteurs

- Julien (architecte produit)
- Préparation assistée par Claude Opus 4.7 (1M context) — `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

## Références

- `project_louis_eric_meeting_2026-05-28.md` — transcription complète Louis × Eric
- `project_fabrice_meeting_2026-05-26.md` — spec source revalidation + 4ᵉ rôle
- `project_salah_meeting_2026-05-27.md` — confirmation 3 paliers + lobotomisation
- `project_manu_meeting_2026-05-22.md` — verdict "Uber de la connaissance"
- `project_vihn_meeting_2026-05-20.md` — héritage MyAccess obligatoire
- `docs/audits/` — corpus 8 prismes + 5 gaps (~9000 lignes)
- Forgejo milestones M0..M11 — http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/milestones
