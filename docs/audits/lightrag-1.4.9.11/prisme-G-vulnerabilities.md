# Prisme G - Vulnerabilites & supply chain LightRAG 1.4.9.11

Date: 2026-05-28

Scope audite:

- Code LightRAG exact: `lightrag-hku==1.4.9.11`, installe pour audit dans `/private/tmp/lrag14911/lightrag`.
- Environnement local audite: `.venv/bin/pip list --format=json` puis `.venv/bin/pip-audit --format=json --output /private/tmp/prisme-g-pip-audit.json`.
- Package tiers: `twindb-lightrag-memgraph` uniquement pour le plan de durcissement via `register()`.

Limite importante: le venv local contient `lightrag-hku==1.4.10`, pas `1.4.9.11`. Les constats de code ci-dessous viennent bien du wheel exact `1.4.9.11`. Les CVE de dependances viennent du venv effectivement installe. Les deux CVE LightRAG citees couvrent aussi `1.4.9.11`, car leurs plages affectees incluent respectivement `<= 1.4.12` et `<= 1.4.13`.

## Synthese executive

Posture globale: **high risk** pour un deploiement BNP expose tel quel.

Raisons principales:

- Auth native LightRAG vulnerable par defaut: secret JWT public en clair, confirme par CVE-2026-30762, et confusion d'algorithme JWT jusqu'a `1.4.13`.
- Defaults serveur permissifs: bind `0.0.0.0`, CORS `*`, methodes et headers `*`, whitelist `/api/*`, SSL desactive par defaut.
- Surface de dependances exposee: `pip-audit` remonte 29 vulnerabilities connues dans 14 packages de l'environnement, dont LightRAG, aiohttp, starlette, cryptography, ecdsa, requests, urllib3.
- Code source avec patterns dangereux: Cypher/AGE construit par f-string, appels HTTP sans timeout, installation dynamique de packages au runtime via `pipmaster`.
- Supply chain active mais immature pour un contexte bancaire: projet public tres actif, Trusted Publishing PyPI sur releases recentes, mais politique de support incoherente/ancienne et dependances non pinnees strictement dans le wheel.

## Sources externes utilisees

- GitHub Advisory CVE-2026-30762 / GHSA-mcww-4hxq-hfr3, hardcoded JWT secret LightRAG: https://github.com/advisories/GHSA-mcww-4hxq-hfr3
- GitHub Security Advisory GHSA-8ffj-4hx4-9pgf, JWT algorithm confusion LightRAG: https://github.com/HKUDS/LightRAG/security/advisories/GHSA-8ffj-4hx4-9pgf
- PyPI `lightrag-hku`, versions, Trusted Publishing, Sigstore provenance: https://pypi.org/project/lightrag-hku/
- GitHub `HKUDS/LightRAG`, activite repo et releases: https://github.com/HKUDS/LightRAG and https://github.com/HKUDS/LightRAG/releases
- GitHub Security Policy `HKUDS/LightRAG`: https://github.com/HKUDS/LightRAG/security/policy
- `pip-audit` documentation: https://github.com/pypa/pip-audit
- aiohttp advisory GHSA-w2fm-2cpv-w7v5 / CVE-2026-22815: https://github.com/advisories/GHSA-w2fm-2cpv-w7v5
- cryptography advisory GHSA-p423-j2cm-9vmq / CVE-2026-39892: https://github.com/advisories/GHSA-p423-j2cm-9vmq
- Starlette/X41 advisory GHSA-86qp-5c8j-p5mr: https://x41-dsec.de/lab/advisories/x41-2026-002-starlette/

## 1. Defaults dangereux

| Finding | Severite | Preuve | Risque | Remediation |
| --- | --- | --- | --- | --- |
| Bind reseau public par defaut | high | `/private/tmp/lrag14911/lightrag/api/config.py:91-94` definit `--host` a `0.0.0.0`. | Exposition non intentionnelle sur toutes les interfaces, surtout si auth/CORS restent faibles. | Forcer `HOST=127.0.0.1` par defaut en local, ou bind interne Kubernetes/service mesh. Exiger reverse proxy TLS/auth en production. |
| CORS ouvert | high | `config.py:390` met `CORS_ORIGINS="*"`. `lightrag_server.py:447-456` ajoute `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`. | Toute origine web peut appeler l'API; combinaison dangereuse avec tokens et `X-New-Token`. | Refuser `*` en production. Lire une allowlist explicite BNP, desactiver credentials cross-origin sauf besoin justifie. |
| JWT secret public par defaut | high | `config.py:397` definit `TOKEN_SECRET` a `lightrag-jwt-default-secret`; `auth.py:25` charge ce secret; `auth.py:71` signe avec. CVE-2026-30762 confirme que `<= 1.4.12` est affecte, corrige en `1.4.13`. | Forge de JWT et bypass auth si `AUTH_ACCOUNTS` est configure sans secret custom. Bloquant BCE. | Refuser de booter si auth activee et secret absent/faible. Rotation obligatoire. Preferer validation OAuth2/JWKS externe plutot que secret local. |
| JWT algorithm configurable par env | high | `config.py:400` lit `JWT_ALGORITHM`; `auth.py:87` decode avec `algorithms=[self.algorithm]`. GHSA-8ffj-4hx4-9pgf indique une confusion `alg=none` jusqu'a `1.4.13`, corrigee en `1.4.14`. | Bypass d'auth par token forge selon advisory; risque d'erreur de config vers un algo interdit. | Interdire `none`, allowlist stricte (`RS256`/`ES256` ou HS256 legacy seulement), tests de regression JWT. |
| Whitelist API large | high | `config.py:393` met `WHITELIST_PATHS="/health,/api/*"` par defaut. | Les routes `/api/*` peuvent etre traitees comme publiques selon la dependency auth native; incompatible avec un baseline zero trust. | Remplacer par `/health` seulement. Toute route metier doit passer par OAuth2/mTLS. |
| SSL desactive par defaut | medium | `config.py:173-188` configure `--ssl` via env `SSL`, False par defaut, cert/key seulement valides si SSL active. | Tokens et API keys en clair si exposition directe. | TLS obligatoire au reverse proxy; refuser bind public sans TLS/mTLS en mode production. |
| Timeout serveur peut etre infini | medium | `config.py:115-120` documente `Use None for infinite timeout`. | Connexions et jobs bloquants, DoS par epuisement de ressources. | Plafond de timeout par defaut, max configurable, cancellation cooperative. |
| API key native acceptee | high/compliance | `lightrag_server.py:343-344` lit `LIGHTRAG_API_KEY` ou `--key`. | Les API keys sont interdites dans le contexte BNP; mecanisme non conforme au modele MyAccess/OAuth2/mTLS. | Desactiver l'API key native dans notre boot, ou la garder seulement derriere flag legacy explicitement non-prod. |
| Installation dynamique au demarrage | high/supply-chain | `lightrag_server.py:1454-1467` installe `uvicorn`, `tiktoken`, `fastapi` via `pipmaster` si manquants. | Telechargement runtime non controle, non reproductible, incompatible avec SBOM et change control DORA. | Monkey-patcher en no-op en production; construire une image immuable avec dependances pinnees et auditees. |

## 2. CVE / vulns connues dans l'arbre de dependances

Outil: `pip-audit` scanne les environnements Python contre la Python Packaging Advisory Database / PyPI JSON API et OSV. Le rapport local a trouve **29 vulnerabilities dans 14 packages**. Les packages editables locaux `twindb-kbs` et `twindb-lightrag-memgraph` sont ignores par l'outil car absents de PyPI.

| Package installe | Version | Severite audit | Advisory / CVE | Affected / fix | Impact | Upgrade recommande |
| --- | ---: | --- | --- | --- | --- | --- |
| `lightrag-hku` | 1.4.10 dans venv, 1.4.9.11 cible | high | CVE-2026-30762 / GHSA-mcww-4hxq-hfr3 | `<=1.4.12`, fix `1.4.13` | Secret JWT hardcode, auth bypass. | Minimum `1.4.14`, idealement `1.4.16+` apres tests; notre patch doit aussi bloquer secret par defaut. |
| `lightrag-hku` | 1.4.10 dans venv, 1.4.9.11 cible | high | CVE-2026-39413 / GHSA-8ffj-4hx4-9pgf | `<=1.4.13`, fix `1.4.14` | JWT algorithm confusion / `alg=none`, auth bypass selon advisory. | Minimum `1.4.14`; ajouter tests d'acceptation rejetant `alg=none`. |
| `aiohttp` | 3.13.3 | medium | 10 advisories dont CVE-2026-22815 / GHSA-w2fm-2cpv-w7v5 | `<=3.13.3`, fix `3.13.4` | Header/trailer handling pouvant mener a epuisement memoire; autres GHSA corriges dans le meme patch. | Pin `aiohttp>=3.13.4`. |
| `starlette` | 0.52.1 | high | PYSEC-2026-161 / GHSA-86qp-5c8j-p5mr | `<1.0.1`, fix `1.0.1` | Host header non valide pouvant contourner de l'auth basee sur `request.url.path`; Starlette est la base de FastAPI. | Attendre/valider compat FastAPI vers Starlette `1.0.1+`; en attendant, reverse proxy avec Host allowlist et middleware Host validation. |
| `cryptography` | 46.0.4 | medium | GHSA-p423-j2cm-9vmq, GHSA-m959-cc7f-wv43, GHSA-r6ph-v2qm-q3c2 | fixes `46.0.5`, `46.0.6`, `46.0.7` | Buffer overflow et bugs crypto selon advisories. | Pin `cryptography>=46.0.7`. |
| `ecdsa` | 0.19.1 | medium/high | CVE-2024-23342 / GHSA-wj6h-64fc-37mp; CVE-2026-33936 / GHSA-9f5j-8jwj-x28g | one no-fix; fix `0.19.2` | Risques crypto; dependance transitive via `python-jose[cryptography]`. | Eviter `python-jose` si possible; remplacer par lib maintenue compatible JWKS (`pyjwt[crypto]`, `authlib`) et pin `ecdsa>=0.19.2` tant que present. |
| `requests` | 2.32.5 | medium | CVE-2026-25645 / GHSA-gc5v-m9x4-r6x2 | fix `2.33.0` | Vuln HTTP client selon advisory. | Pin `requests>=2.33.0`. |
| `urllib3` | 2.6.3 | medium | PYSEC-2026-141/142, CVE-2026-44431/44432 | fix `2.7.0` | Vulns HTTP transport selon advisories. | Pin `urllib3>=2.7.0` et verifier compat `requests`. |
| `python-dotenv` | 1.2.1 | low/medium | CVE-2026-28684 / GHSA-mf9w-mj56-hr94 | fix `1.2.2` | Parsing/env handling selon advisory. | Pin `python-dotenv>=1.2.2`. |
| `idna` | 3.11 | medium | CVE-2026-45409 / GHSA-65pc-fj4g-8rjx | fix `3.15` | IDNA parsing selon advisory. | Pin `idna>=3.15`. |
| `pyasn1` | 0.6.2 | medium | CVE-2026-30922 / GHSA-jr27-m4p2-rc6r | fix `0.6.3` | ASN.1 parsing selon advisory. | Pin `pyasn1>=0.6.3`. |
| `pip` | 26.0 | dev/tooling | CVE-2026-3219, CVE-2026-6357 | fix `26.1` | Build/install tooling vulnerable; risque supply chain en CI/build. | Pin bootstrap `pip>=26.1` dans images. |
| `black`, `pygments`, `pytest` | dev tools | dev/tooling | multiple | fixes disponibles | Non runtime si exclus de l'image prod. | Exclure de l'image prod; mettre a jour dans image dev/CI. |

Packages critiques demandes sans vuln remontee par `pip-audit` dans cet environnement: `fastapi 0.135.1` directement, `pydantic 2.12.5`, `neo4j 6.1.0`, `openai 2.17.0`, `json_repair 0.57.1`, `tiktoken 0.12.0`, `PyJWT 2.12.1`, `httpx 0.28.1`. Cela ne vaut pas attestation de securite; c'est seulement "aucune advisory connue par la source pip-audit a la date du scan".

## 3. Patterns a risque dans le code LightRAG

| Pattern | Severite | Preuve | Risque | Remediation |
| --- | --- | --- | --- | --- |
| Cypher Memgraph f-string avec label dynamique | high | `/private/tmp/lrag14911/lightrag/kg/memgraph_impl.py:91-94` construit `CREATE INDEX ON :{workspace_label}(entity_id)`. | Injection Cypher via label/workspace si la normalisation est insuffisante ou contournee. | Valider workspace avec regex stricte (`^[A-Za-z_][A-Za-z0-9_]*$`) et echapper/parametrer partout ou l'API le supporte. Notre incident Cassandre justifie un guard central. |
| AGE/PostgreSQL Cypher f-string avec node id | high | `postgres_impl.py:4267-4271` interpole `label` dans `MATCH (n:base {entity_id: "{label}"})`; `postgres_impl.py:4307-4311` fait pareil dans `MERGE`. | Injection AGE/Cypher par `node_id`/`entity_id`, ecriture ou lecture non autorisee. | Parametrer au niveau SQL si possible; sinon encoder en JSON/literal Cypher sur une fonction unique testee par fuzzing. |
| AGE/PostgreSQL edge f-string avec source/target/properties | high | `postgres_impl.py:4345-4353` interpole `src_label`, `tgt_label`, `edge_properties`; `postgres_impl.py:4374-4377` interpole `label` dans DELETE. | Injection destructrice potentielle (`DETACH DELETE`) et corruption KG. | Meme remediation: builder Cypher safe, validation d'identifiants, tests payloads adversariaux. |
| Appels HTTP sans timeout | medium | `llm/jina.py:22-24` ouvre `aiohttp.ClientSession()` puis `session.post()` sans timeout; `rerank.py:290-291` idem. | Requetes pendantes, epuisement workers, DoS par fournisseur lent. | Monkey-patcher wrappers HTTP ou contribuer upstream: `aiohttp.ClientTimeout(total=...)`, connect/read timeouts, retry budget. |
| Installation de dependances a l'import | high/supply-chain | `llm/jina.py:2-8` installe `aiohttp` et `tenacity` via `pipmaster` lors de l'import si manquants. | Import non deterministe, execution reseau inattendue, bypass du processus d'approbation dependances. | Patch no-op de `pipmaster.install` ou suppression des branches runtime install en prod. Dependances resolvees en build image seulement. |
| Installation de dependances au boot API | high/supply-chain | `api/lightrag_server.py:1454-1467` installe `uvicorn`, `tiktoken`, `fastapi`. | Meme risque, mais au demarrage du service. | Forcer fail-fast si dependance absente. Interdire egress package manager en production. |
| Secrets placeholders dans storage deprecie | medium | `kg/deprecated/chroma_impl.py:69-72` utilise `auth_token` par defaut `secret-token`. | Si chemin deprecie reactive, token partage connu. | Remplacer par absence de token par defaut et fail-fast si auth requise. |

Recherche negative: pas de `eval(`, `exec(`, `pickle`, `subprocess`, `shell=True`, `os.system`, `yaml.load` detecte dans `/private/tmp/lrag14911/lightrag/**/*.py` avec les patterns audites.

## 4. Secrets hardcodes

| Secret/default | Severite | Preuve | Evaluation |
| --- | --- | --- | --- |
| `lightrag-jwt-default-secret` | high | `api/config.py:397`; confirme par CVE-2026-30762. | Secret public exploitable. Bloquant BCE si non neutralise. |
| `secret-token` Chroma deprecated | medium | `kg/deprecated/chroma_impl.py:72`. | Chemin deprecie mais dangereux si utilise. |
| API keys cloud | low | Recherche `rg` sur tokens/API keys/OpenAI/Bearer. | Pas de vraie cle cloud hardcodee detectee dans le wheel LightRAG audite; uniquement noms d'env vars et exemples/placeholders. |
| Credentials graph stores | medium | Memgraph/Neo4j/Milvus lisent env/config; Memgraph defaults username/password vides. | Pas un secret hardcode, mais defaults faibles si service expose. |

## 5. Posture supply chain

Risque opinion: **medium-high** pour production bancaire sans encapsulation.

Elements positifs:

- Projet public tres actif: GitHub affiche environ 35k stars, 5k forks, environ 8k commits et des releases frequentes jusqu'a `v1.5.0rc3` le 2026-05-26.
- PyPI montre des releases recentes (`1.4.13` le 2026-04-02, `1.4.14` le 2026-04-12, `1.4.16` le 2026-05-07) et Trusted Publishing/Sigstore provenance sur les artefacts recents.
- Les deux advisories LightRAG ont ete publiees et corrigees upstream relativement vite (`1.4.13` puis `1.4.14`).

Elements negatifs:

- La security policy visible sur GitHub declare `1.2.x` non supporte et `1.3.x` supporte, sans mention claire des branches `1.4.x`/`1.5.x`; cela semble obsolescent par rapport aux releases actuelles.
- Le wheel `1.4.9.11` depend largement de ranges non pinnees (`aiohttp`, `fastapi`, `pydantic`, etc.), donc l'arbre effectif varie selon date de resolution et index.
- Presence de `pipmaster` et installation dynamique au runtime, incompatible avec une supply chain bancaire reproductible.
- Upstream a deja accumule deux CVE d'auth sur la meme zone fonctionnelle; cela pese plus qu'une simple vuln de dependance.

Decision BNP/DORA: ne pas consommer LightRAG directement depuis PyPI en production. Construire un artefact interne, SBOM, verrou de dependances, audit `pip-audit`/SCA, signature interne, tests de securite auth, et egress package manager bloque.

## 6. Plan de durcissement dans notre `register()`

Priorite: notre patch doit **durcir sans affaiblir** et doit fail-fast en production.

Ordre recommande dans `src/twindb_lightrag_memgraph/__init__.py`:

1. Conserver les patches registry/storage existants en premier si necessaires a l'import LightRAG.
2. Ajouter `_patch_security_baseline()` avant tout import de `lightrag.api.lightrag_server`.
3. Ajouter `_patch_lightrag_server_ui()` et `_mount_twin_server()` apres `_patch_version_string()` et apres `_patch_security_baseline()`, mais avant `_registered = True`.
4. Ne jamais importer `lightrag.api.lightrag_server` avant d'avoir applique les patches d'arguments/config/version, car les objets module-level (`auth_handler`, `global_args`, `app`, routes, middleware) sont captures au premier import.

Contenu minimal de `_patch_security_baseline()`:

- Refuser `TOKEN_SECRET` absent ou egal a `lightrag-jwt-default-secret` si auth native reste active.
- Refuser `JWT_ALGORITHM=none`; allowlist stricte.
- Forcer `WHITELIST_PATHS=/health` sauf override explicite non-prod.
- Refuser `CORS_ORIGINS=*` en mode production; imposer allowlist d'origines BNP.
- Refuser `LIGHTRAG_API_KEY` en mode BNP, ou n'autoriser que `TWINDB_LEGACY_API_KEY_AUTH=1` pour migration limitee.
- Refuser bind `HOST=0.0.0.0` sans reverse proxy/TLS/mTLS declare.
- Patch no-op/fail-fast pour `check_and_install_dependencies()` afin d'interdire `pipmaster.install` au boot.
- Patch wrappers HTTP connus (`jina`, `rerank`, `lollms`) pour imposer `aiohttp.ClientTimeout(total=...)` si upstream non corrige.
- Ajouter un middleware Host allowlist tant que Starlette/FastAPI n'est pas sur une version non vulnerable et validee.
- Logguer explicitement la baseline appliquee sans jamais logguer de secret.

Strategie CVE:

- Court terme: patch runtime + image verrouillee, car upgrade `1.4.14+` peut modifier le comportement LightRAG.
- Moyen terme: tester upgrade `lightrag-hku>=1.4.14` minimum; preferer `1.4.16` si compatible avec nos monkey-patches.
- Long terme: remplacer auth native par OAuth2/JWKS/mTLS, retirer API key et comptes locaux, integrer le role MyAccess `knowledge` depuis SSO.

## 7. Risques compliance BCE / DORA art. 9

| Pratique | Severite compliance | Justification |
| --- | --- | --- |
| Secret JWT public par defaut | bloquant | Vulnerabilite connue, authentification falsifiable, CVE publiee. |
| Confusion JWT `alg=none` jusqu'a `1.4.13` | bloquant | Auth bypass selon advisory upstream; inacceptable pour donnees connaissance. |
| API key native | bloquant BNP | Interdit par le contexte; non liee a MyAccess/SSO/mTLS. |
| CORS `*` + credentials + token renewal header | serieux | Exposition cross-origin non maitrisee. |
| Whitelist `/api/*` | serieux | Risque d'endpoints metier publics selon config. |
| Installation dynamique de packages | bloquant DORA | Pas de reproductibilite, pas de controle de changement, pas de SBOM fiable. |
| Dependances vulnerables detectees par SCA | serieux | 29 findings; necessite plan d'upgrade/exception documente. |
| Requetes HTTP sans timeout | serieux | Resilience insuffisante, risque DoS/lenteur fournisseur. |
| F-string Cypher/AGE | serieux | Risque injection/corruption KG; deja comparable a incident Cassandre. |
| Security policy upstream ambigue | cosmetic/serieux | Pas bloquant seul, mais renforce la necessite d'une gouvernance interne. |

