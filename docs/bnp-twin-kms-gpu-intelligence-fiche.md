# Fiche synthèse — GPU Intelligence Twin KMS (2× NVIDIA L40S)

| Champ | Valeur |
|---|---|
| Date | 2026-07-04 |
| Statut | Proposition de cadrage (suite attribution des 2 GPU) |
| Audience | Salah (SRE/DevOps), sponsors Twin KMS |
| Références internes | `docs/audits/retrieval-tuning/audit-2026-07-03.md` (stabilité retrieval), `docs/audits/intelligence-layer/audit-2026-07-03.md` (passe produit L3) |

## 1. Principe directeur

**Les 2× L40S ne servent aucun trafic de production.** Le serving souverain de prod reste **LLM as a Service** (gpt-oss-120b aujourd'hui, migration Gemma 4 31B planifiée). Les GPU sont une **plateforme R&D/innovation pour les bases de connaissance** :

- aucune dépendance runtime de Twin KMS vers ces cartes — indisponibilité GPU = zéro impact prod ;
- règle de transfert : tout ce qui y est validé se déploie ensuite **en config** sur la prod LLMaaS, jamais en re-développement ;
- pour garantir ce transfert : **iso-modèle avec la cible LLMaaS** (Gemma 4 31B-it), donc le banc qualifie aussi la migration gpt-oss → Gemma.

## 2. Stack technique

| Élément | Choix | Justification |
|---|---|---|
| Modèle | **Gemma 4 31B-it, FP8** | Validé groupe, Apache 2.0, cible LLMaaS, multimodal texte+image, 256K ctx ; FP8 natif L40S (Transformer Engine) |
| Serving | vLLM (OpenAI-compatible, structured output activé) | Contrat exact du code L3 : `chat.completions` + JSON mode |
| Topologie | **1 réplica par GPU, 2 lanes** : GPU 1 = interactive (dev/proto), GPU 2 = batch (jobs nocturnes) | Modèle FP8 ≈ 32 GB → tient sur 1 carte avec marge KV ; le batch ne dégrade jamais l'interactif ; pas de tensor parallel ni de NVLink requis |
| Cologé lane batch | Embeddings + reranker (modèles légers) | Bancs de comparaison retrieval (UC1) |
| Fallback modèle | gpt-oss-120b (aussi validé groupe) | Nécessite TP=2 (~63 GB) → consomme les 2 cartes ; réservé si Gemma échoue un bench |

## 3. Les quatre use cases

| # | Use case | Contenu | Livrable | Métrique de succès |
|---|---|---|---|---|
| 1 | **Retrieval Quality Lab** | Protocole de stabilité (même question N fois — sujet remonté par les utilisateurs), comparaison embeddings/rerankers, tuning top-k/seuils, sélection automatique du mode de retrieval | Réglages de config validés, déployables sur la prod LLMaaS | Variance des sources/réponses avant/après ; gain de pertinence mesuré sur jeu de questions réelles |
| 2 | **Ingestion Intelligence** (proto) | Tags/résumés proposés à l'upload, dédoublonnage sémantique, qualité de chunking, parsing des documents visuels (captures, schémas, tableaux — Gemma multimodal). **Tout passe par les workflows d'approbation existants, rien ne s'auto-applique** | Prototype branché sur un folder pilote | Taux d'acceptation des propositions par les stewards |
| 3 | **Groundedness Validator** | LLM-as-judge en batch sur les traces Q&R : chaque réponse est-elle réellement supportée par ses sources citées ? | Métrique de qualité continue + rapport périodique | % réponses « fully grounded » ; métrique défendable en audit |
| 4 | **R&D Twin KMS** (l'imposé) | Industrialisation du pipeline ontologie existant (runs corpus complet, profil batch nocturne) + **nettoyage du graphe** : fusion d'entités synonymes, détection d'entités trop génériques, relations manquantes — en dry-run/approve | File de propositions de curation validables par un humain | Taux d'acceptation des fusions/nettoyages proposés |

Extension produit visée (post-pilotes, cf. audit L3) : le **Curateur nocturne** — consolidation des sorties UC2/UC3/UC4 en une file quotidienne unique de propositions steward.

## 4. Plan de bench d'entrée (avant tout use case)

1. **Compatibilité contrat L3** : le code envoie `reasoning_effort` en `extra_body` — vérifier que vLLM+Gemma l'ignore proprement (sinon micro-patch côté Twin, trivial, identifié). JSON mode via structured output sur les 6 appels concernés.
2. **Qualité français** : le pipeline query L3 est 100 % FR — jeu d'éval FR (intent, reformulation, synthèse) Gemma 4 31B FP8 vs gpt-oss-120b.
3. **Débit par lane** : tokens/s en interactif (latence de synthèse 2K tokens) et en batch (throughput extraction ontologie), FP8 sur L40S.
4. **FP8 vs BF16** : bascule TP=2 BF16 uniquement si une régression qualité FP8 est **mesurée** — on n'ajoute pas de complexité pour un problème non constaté.

## 5. Ce que ces GPU ne font pas

- Pas de serving de trafic utilisateur (prod = LLMaaS, point).
- Pas de fine-tuning de modèles au lancement (réévaluable après 6 mois de bancs).
- Pas d'auto-application de modifications à la KB : toute proposition générée passe en validation humaine avec audit trail (doctrine produit validée).
- Pas de nouveau référentiel de droits : l'accès suit MyAccess comme le reste de Twin.

## 6. Besoins infra (checklist SRE)

- Host avec les 2 cartes, image vLLM via la filière d'images habituelle.
- Volume pour les poids Gemma 4 31B-it (modèle validé groupe ; transit par le canal artefacts standard, pas de pull direct externe).
- Monitoring GPU standard (DCGM/exporter existant), alerting basique (mémoire, température, disponibilité endpoint).
- Réseau : exposition LAN projet uniquement ; pas d'ouverture au-delà.
- Scheduling : cron/night-batch pour la lane 2 (jobs UC3/UC4).

## 7. Décisions ouvertes (avant d'engager UC2/UC3)

1. **Persistance de dérivés LLM** (résumés, tags proposés, descriptions d'images) vs doctrine de stockage actuelle « chunks+vecteurs » — cadrage compliance unique qui débloque UC2 et l'extension Curateur.
2. **Rétention des traces de requêtes** utilisées par UC1/UC3 (anonymisation par empreinte déjà disponible côté code).
3. Folder pilote pour les prototypes UC2/UC4.
