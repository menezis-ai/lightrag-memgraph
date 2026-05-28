# Runbook d'installation — `twindb-lightrag-memgraph` v1.1.0

| Champ | Valeur |
|---|---|
| **Document** | `docs/operations/install-runbook.md` |
| **Version cible du package** | `1.1.0` |
| **Version cible de LightRAG** | `1.4.9.11` (= version BNP prod) |
| **Version cible de Memgraph** | `3.7.2` ou `3.10.1` (MAGE) |
| **Audience** | Opérateur BNP qui installe le wheel ; auditeur IG/BCE qui vérifie le processus |
| **Statut** | Draft pour review (sera figé en cas de release) |
| **Owner produit** | Julien |
| **Owner technique BNP** | À définir (escalation Fabrice) |
| **Période de validité** | À chaque release `1.1.x`, ce document est révisé |

> **Lecture rapide pour un auditeur IG** : sections §1 (Prérequis), §3 (Procédure), §4 (Vérification), §6 (Rollback) — environ 5 minutes. Le reste est opérationnel pour les ops BNP.

---

## §1 — Prérequis

### 1.1 — Système hôte

| Composant | Exigence | Notes |
|---|---|---|
| OS | Linux x86_64 ou aarch64 (RHEL 9 / Debian 12 validés ; macOS uniquement en dev) | Le wheel est `py3-none-any` donc indépendant de l'OS au sens strict, mais les deps natives le sont |
| Python | 3.10 / 3.11 / 3.12 / 3.13 — **pas** 3.14 sur BNP (numpy footgun documenté) | Matrice CI couvre 3.10–3.13 |
| Mémoire libre | ≥ 4 Gio | LightRAG + Memgraph driver + cache |
| Disque | ≥ 1 Gio pour le venv + 1 Gio par doc-corpus de référence | Wheel lui-même = 258 KB |
| Sortie internet | **Bloquée recommandée** | Le wheel est hermétique (cf. §1.4) |

### 1.2 — Services internes BNP

| Service | Adresse | Rôle |
|---|---|---|
| **Memgraph 3.7.2 ou 3.10.1** | `bolt://<host>:7687` (TLS recommandé) | Graph + KV + Vector + DocStatus |
| **LLM provider interne** | URL configurable par env | Chat (`chat_llm`) et indexation (`indexing_llm`) — peuvent être identiques |
| **SSO BNPP** | URL JWKS BNPP | Validation des JWT (à la sortie v1.2 ; en v1.1 mode legacy-fallback documenté) |
| **MyAccess** | API d'introspection (futur) | Source des claims `role` / `palier` |
| **Log-as-a-Service** (Splunk / ELK / Datadog) | URL d'ingestion | Sink optionnel pour l'audit trail |

### 1.3 — Variables d'environnement requises

| Variable | Requis | Exemple | Description |
|---|---|---|---|
| `MEMGRAPH_URI` | **Oui** | `bolt://memgraph-internal.bnp:7687` | Endpoint Memgraph |
| `MEMGRAPH_USERNAME` | Selon config | `lightrag` | Si auth Memgraph activée |
| `MEMGRAPH_PASSWORD` | Selon config | (depuis Bitwarden / coffre BNP) | **Ne jamais committer** |
| `MEMGRAPH_USE_TLS` | Recommandé | `true` | Active TLS sur la connexion Bolt |
| `OPENAI_API_KEY` (ou équivalent provider interne) | **Oui** | (depuis coffre) | Clé d'accès au LLM |
| `LLM_BINDING_HOST` | **Oui** | `https://llm-interne.bnp/v1` | URL du provider LLM |
| `EMBEDDING_BINDING_HOST` | **Oui** | (idem ou différent) | URL du provider embeddings |
| `EMBEDDING_DIM` | **Oui** | `1024` | Doit matcher le modèle embedding choisi |
| `LIGHTRAG_KV_STORAGE` | **Oui** | `MemgraphKVStorage` | Active notre backend KV |
| `LIGHTRAG_VECTOR_STORAGE` | **Oui** | `MemgraphVectorDBStorage` | Active notre backend Vector |
| `LIGHTRAG_DOC_STATUS_STORAGE` | **Oui** | `MemgraphDocStatusStorage` | Active notre backend DocStatus |
| `LIGHTRAG_GRAPH_STORAGE` | **Oui** | `MemgraphStorage` | Active le graph backend natif LightRAG |
| `TWIN_AUTH_JWKS_URL` | **Oui (prod)** | URL JWKS BNPP | Validation OAuth2 (v1.1 stub, v1.2 plein) |
| `TWIN_AUTH_LEGACY_FALLBACK` | Non | `false` (par défaut) | **Ne jamais** mettre à `true` en prod BNP |
| `TWIN_LOG_SINK` | Recommandé | `stdout` ou `splunk` ou `file` | Destination de l'audit trail |
| `TWIN_LOG_LEVEL` | Optionnel | `INFO` | Niveau de log applicatif |

### 1.4 — Caractéristiques du wheel

- **Hermétique** : aucune dépendance n'est téléchargée au runtime. Le wheel embarque le WebUI Twin (fichiers statiques) et toutes les deps Python sont déclarées et résolues à `pip install`.
- **Signé** (à partir de la v1.1.0) : tag Git `v1.1.0` signé via GPG ; signature Sigstore sur l'artefact PyPI/Forgejo.
- **SBOM** : un fichier `docs/sbom-1.1.0.json` (CycloneDX) accompagne chaque release. **L'auditeur peut le récupérer depuis le tag de release Forgejo.**
- **Pas d'install dynamique** : la fonction `_patch_security_baseline()` neutralise `pipmaster.install` au démarrage. Toute tentative d'installation runtime de la part de LightRAG natif est bloquée et journalisée. C'est documenté dans `docs/audits/lightrag-1.4.9.11/prisme-G-vulnerabilities.md`.

---

## §2 — Choix de l'environnement cible

### 2.1 — Décision préalable : mono-process vs gunicorn

| Cas d'usage | Recommandation |
|---|---|
| Dev BNP / staging unique-instance | `lightrag-server` (mono-process uvicorn) |
| Pré-prod multi-instance | `lightrag-gunicorn` |
| Prod BNP | `lightrag-gunicorn` derrière reverse-proxy TLS (Nginx / Apache / iTrack frontal) |

### 2.2 — Décision préalable : flags `register()` à activer

L'opérateur doit décider explicitement :

| Flag | Effet | Recommandation prod BNP |
|---|---|---|
| `replace_ui=True` | Remplace `/webui` natif par notre console Twin | **Oui** si on déploie la version gouvernance |
| `mount_server=True` | Monte les routes Twin sur `/twin/api` | **Oui** si `replace_ui=True` |
| `security_baseline=True` | Bloque pipmaster + autoinstall LightRAG | **Toujours** (= défaut) |
| `webui_dist` | Chemin custom vers dist WebUI | Laisser `None` (utilise le dist embarqué) |
| `twin_api_prefix` | URL de la sous-app | `/twin/api` (défaut) |

---

## §3 — Procédure d'installation

### 3.1 — Sur un host vierge

```bash
# 1. Créer un utilisateur dédié sans shell interactif (sécurité)
sudo useradd --system --shell /usr/sbin/nologin --create-home twinrag

# 2. Préparer un venv Python (ici 3.12)
sudo -u twinrag python3.12 -m venv /opt/twinrag/venv

# 3. Activer le venv (ou utiliser le chemin absolu pip)
sudo -u twinrag /opt/twinrag/venv/bin/pip install --no-index \
    --find-links /chemin/vers/le/wheel \
    "twindb-lightrag-memgraph[server,intelligence,tracing]==1.1.0"
```

**Notes** :
- `--no-index` désactive la connexion à PyPI. Toutes les deps doivent être présentes en local (offline install).
- Les extras `[server]` + `[intelligence]` + `[tracing]` activent les optional-dependencies correspondants (auth, ReAct, LangSmith).
- Si l'opérateur n'a pas téléchargé toutes les deps : substituer `--no-index --find-links DIR` par `--index-url <pypi-interne-bnp>` selon la politique BNP.

### 3.2 — Création du fichier `.env`

```bash
sudo -u twinrag tee /opt/twinrag/.env > /dev/null <<'ENV'
# Memgraph
MEMGRAPH_URI=bolt://memgraph-internal.bnp:7687
MEMGRAPH_USE_TLS=true
LIGHTRAG_KV_STORAGE=MemgraphKVStorage
LIGHTRAG_VECTOR_STORAGE=MemgraphVectorDBStorage
LIGHTRAG_DOC_STATUS_STORAGE=MemgraphDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=MemgraphStorage

# LLM (à adapter selon le provider interne BNP)
LLM_BINDING_HOST=https://llm-interne.bnp/v1
EMBEDDING_BINDING_HOST=https://llm-interne.bnp/v1
EMBEDDING_DIM=1024

# Auth Twin
TWIN_AUTH_JWKS_URL=https://sso-bnpp/.well-known/jwks.json
TWIN_AUTH_LEGACY_FALLBACK=false

# Observabilité
TWIN_LOG_SINK=splunk
TWIN_LOG_LEVEL=INFO
ENV
sudo chmod 600 /opt/twinrag/.env
sudo chown twinrag:twinrag /opt/twinrag/.env
```

### 3.3 — Création du wrapper de démarrage qui appelle `register()`

Le script `register_and_serve.py` est le point d'entrée minimal qui active les flags TwinRAG **avant** de lancer LightRAG :

```bash
sudo -u twinrag tee /opt/twinrag/register_and_serve.py > /dev/null <<'PY'
#!/usr/bin/env python3
"""Active TwinRAG extensions, puis délègue à lightrag.api.lightrag_server."""
import twindb_lightrag_memgraph

twindb_lightrag_memgraph.register(
    replace_ui=True,        # remplace /webui par notre console Twin
    mount_server=True,      # monte /twin/api
    security_baseline=True, # bloque pipmaster et autoinstall (Prisme G)
)

from lightrag.api.lightrag_server import main
main()
PY
sudo chmod 755 /opt/twinrag/register_and_serve.py
```

### 3.4 — Service systemd

```bash
sudo tee /etc/systemd/system/twinrag.service > /dev/null <<'UNIT'
[Unit]
Description=TwinRAG (LightRAG + register patch)
After=network.target

[Service]
Type=exec
User=twinrag
Group=twinrag
WorkingDirectory=/opt/twinrag
EnvironmentFile=/opt/twinrag/.env
ExecStart=/opt/twinrag/venv/bin/python /opt/twinrag/register_and_serve.py
Restart=on-failure
RestartSec=10
# Durcissement
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/twinrag /var/log/twinrag
# Pas d'accès internet (le wheel est hermétique — vérifier qu'aucune dep ne fuit)
PrivateNetwork=false   # à mettre true si tout le trafic LLM/Memgraph passe par un proxy local
StandardOutput=journal
StandardError=journal
SyslogIdentifier=twinrag

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable twinrag
sudo systemctl start twinrag
```

---

## §4 — Vérification post-install

### 4.1 — Le service démarre proprement

```bash
sudo systemctl status twinrag
# Expected: Active: active (running)
```

```bash
sudo journalctl -u twinrag -n 50 --no-pager
```

Lignes attendues dans le journal (extrait) :

```
twindb_lightrag_memgraph: twindb: pipmaster runtime install blocked (security baseline)
twindb_lightrag_memgraph: twindb: lightrag check_and_install_dependencies neutralized
twindb-lightrag-memgraph v1.1.0 — PATCH APPLIED SUCCESSFULLY
  Graph DB ........ Memgraph (MemgraphStorage, patched for TLS + multi-db)
  Vector DB ....... Memgraph native vector_search (MemgraphVectorDBStorage)
  KV Storage ...... Memgraph (MemgraphKVStorage)
  DocStatus ....... Memgraph (MemgraphDocStatusStorage)
twindb_lightrag_memgraph: twindb: WebUI mount at /webui swapped → .../twindb_lightrag_memgraph/webui_dist
twindb_lightrag_memgraph: twindb: Twin sub-app mounted at /twin/api (lifespan chained)
twindb_lightrag_memgraph.server.app: L2 patch applied (WebUI phase-1 router)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9621
```

**Signe pathologique critique** : *toute* mention de `pipmaster` *qui ne soit pas* `blocked` est un signal d'alerte SOC. La security baseline a échoué — arrêter et investiguer.

### 4.2 — Les endpoints HTTP répondent

```bash
# Healthcheck LightRAG natif (route inchangée)
curl -sf http://localhost:9621/health | jq .

# Healthcheck sous-app Twin (route ajoutée par mount_server=True)
curl -sf http://localhost:9621/twin/api/health | jq .

# WebUI Twin servi à /webui (le HTML reçu doit contenir la signature de notre fork)
curl -sf http://localhost:9621/webui/index.html | grep -q 'Twin' && echo "OK: WebUI Twin servi"
```

### 4.3 — Memgraph est joignable et nos schémas sont créés

```bash
sudo -u twinrag /opt/twinrag/venv/bin/python <<'PY'
import os, asyncio
from twindb_lightrag_memgraph._pool import get_driver

async def check():
    driver, db = get_driver()
    async with driver.session(database=db) as session:
        result = await session.run("SHOW INDEXES")
        labels = [r["label_name"] async for r in result if "label_name" in r.keys()]
    print(f"Memgraph reachable; indexes on labels: {labels[:10]}")

asyncio.run(check())
PY
```

### 4.4 — La security baseline est active

```bash
sudo -u twinrag /opt/twinrag/venv/bin/python <<'PY'
import twindb_lightrag_memgraph as t
t.register(security_baseline=True)
import pipmaster
try:
    pipmaster.install("fake-pkg-for-test")
    print("FAIL: pipmaster.install n'a PAS été bloqué")
    raise SystemExit(1)
except RuntimeError as e:
    if "TwinRAG security baseline" in str(e):
        print("OK: pipmaster.install bloqué par la security baseline")
    else:
        print(f"FAIL: erreur inattendue: {e}")
        raise SystemExit(1)
PY
```

### 4.5 — Le JWT BNPP est validé correctement (smoke test à faire en pre-prod)

```bash
# Token de test valide
curl -sf -H "Authorization: Bearer ${VALID_TEST_JWT}" \
  http://localhost:9621/twin/api/documents | jq .

# Token absent → doit renvoyer 401
test "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9621/twin/api/documents)" = "401" \
  && echo "OK: 401 sans token"
```

---

## §5 — Opération courante

### 5.1 — Logs

```bash
sudo journalctl -u twinrag -f
```

### 5.2 — Restart contrôlé (zero-downtime via 2 instances derrière LB recommandé)

```bash
sudo systemctl restart twinrag
sudo journalctl -u twinrag -n 30 --no-pager | grep -i "PATCH APPLIED SUCCESSFULLY" || echo "WARN: patch register() non confirmé après restart"
```

### 5.3 — Surveillance Memgraph

Voir le runbook Memgraph dédié — TwinRAG n'a pas de spécificité au-delà de la connectivité Bolt.

### 5.4 — Métriques

À partir de la milestone M8 (livraison post-1.1.0), un endpoint Prometheus sera exposé à `/twin/api/metrics`. En 1.1.0, observer via les logs JSON ECS.

---

## §6 — Rollback

**Objectif** : revenir à l'état antérieur (LightRAG natif, sans extensions Twin) en deux commandes maximum.

### 6.1 — Cas 1 : l'extension Twin doit être désactivée mais on garde le service en marche

Modifier le wrapper `register_and_serve.py` :

```python
# Désactiver les extensions sans réinstaller
twindb_lightrag_memgraph.register(
    replace_ui=False,         # WebUI revient au natif LightRAG
    mount_server=False,       # routes /twin/api retirées
    security_baseline=True,   # à garder ON sauf justification compliance
)
```

```bash
sudo systemctl restart twinrag
```

**Effet** : LightRAG sert son `/webui` natif et expose ses routes natives ; aucune route `/twin/api` ; aucun mount swap ; storages Memgraph **restent en place** (= les données restent accessibles via les endpoints natifs LightRAG).

### 6.2 — Cas 2 : downgrade complet vers v1.0.x (storage uniquement, pas d'UI/server étendu)

```bash
sudo -u twinrag /opt/twinrag/venv/bin/pip install --no-index --find-links /chemin/wheels/v1.0 \
  "twindb-lightrag-memgraph==1.0.0"
sudo systemctl restart twinrag
```

**Effet** : retour à la version qui tourne aujourd'hui en prod. `register()` sans flags. Aucune perte de données — schémas Memgraph sont compatibles ascendants.

### 6.3 — Cas 3 : désinstallation totale du package twindb (retour à LightRAG vanilla)

```bash
sudo -u twinrag /opt/twinrag/venv/bin/pip uninstall -y twindb-lightrag-memgraph
sudo systemctl stop twinrag
```

Puis modifier `register_and_serve.py` pour supprimer l'import `twindb_lightrag_memgraph`. Le service ne devrait plus être lancé par cette unité ; relancer LightRAG directement via `/opt/twinrag/venv/bin/lightrag-server`.

**Effet** : LightRAG vanilla, plus aucun patch. **Attention** : les storages Memgraph précédemment configurés via les backends `MemgraphKVStorage` / `MemgraphVectorDBStorage` / `MemgraphDocStatusStorage` ne seront plus servis. Migration à prévoir.

### 6.4 — Critères pour décider d'un rollback

| Symptôme | Rollback recommandé |
|---|---|
| Le `/twin/api/health` répond `500` mais `/health` natif répond | §6.1 (désactiver Twin, garder le service) |
| Pipmaster bloque un boot légitime (cas exceptionnel — vérifier la cause) | `register(security_baseline=False)` temporairement, **escalation immédiate** |
| Régression fonctionnelle sur le retrieval natif LightRAG | §6.2 (downgrade vers 1.0.x) |
| Découverte de vulnérabilité sécu post-déploiement | §6.1 ou §6.3 selon gravité |

---

## §7 — Troubleshooting

### 7.1 — `RuntimeError: Runtime pip install blocked by TwinRAG security baseline`

**Symptôme** : un boot du service échoue en relayant ce message.
**Cause** : LightRAG natif (ou une lib appelée) tente un install runtime via `pipmaster`. La baseline a fait son travail.
**Diagnostic** : journaliser le stack trace ; identifier la lib coupable (souvent `lightrag.llm.ollama`, `lightrag.llm.jina` ou un nouveau binding).
**Résolution** : ajouter cette lib en dépendance pinned dans `pyproject.toml` côté package twindb (suivre la procédure de release 1.1.x).
**Contournement temporaire** : `register(security_baseline=False)` + alerte sécu interne BNP. Ne pas laisser en production plus de 24h.

### 7.2 — `FileNotFoundError: register(replace_ui=True): no WebUI dist found`

**Symptôme** : le boot échoue à résoudre le `webui_dist`.
**Cause** : le wheel installé ne contient pas le répertoire `webui_dist/` (artefact pré-1.1.0 ou build script non exécuté avant `python -m build`).
**Résolution** : vérifier le wheel avec `python -c "import zipfile; z=zipfile.ZipFile('twindb_lightrag_memgraph-1.1.0-py3-none-any.whl'); print([n for n in z.namelist() if 'webui_dist' in n])"`. Si vide, reconstruire avec `scripts/build_webui.sh` puis `python -m build`.

### 7.3 — `Memgraph.ClientError ... Trying to get a property from a deleted object`

**Symptôme** : erreur sporadique sur `aquery` ou `adelete_by_doc_id`.
**Cause connue** : race condition dans le pipeline buffered writes / read cache sous charge concurrente. Identifié sur le test `test_multiple_documents_partial_delete` en suite e2e.
**Résolution v1.1.0** : retry au niveau applicatif. v1.2.x prévoit un audit du `_BufferedGraphProxy`.

### 7.4 — Twin sub-app monte mais retourne 500 sur toutes les routes

**Cause** : le lifespan de la sub-app n'a pas été chainé correctement (régression sur `_patch_lightrag_server_create_app`).
**Diagnostic** : `journalctl -u twinrag | grep "Twin sub-app mounted"` → si présent mais routes 500, vérifier les logs pour `_get_rag()` ou `RuntimeError: LightRAG not initialized`.
**Résolution** : `register(mount_server=False)` temporairement, escalation chez Julien pour patch.

---

## §8 — Audit trail et conformité

### 8.1 — Ce qui est journalisé par défaut (v1.1.0)

Le middleware audit (M3, livraison v1.1.0 en cours) loggue en JSON ECS-compatible vers le sink configuré par `TWIN_LOG_SINK` :

- `auth.login.success` / `auth.login.failure`
- `retrieval.query.submitted` / `retrieval.query.completed` / `retrieval.query.failed`
- `document.upload.accepted` / `document.index.completed` / `document.index.failed`
- `document.delete.requested` / `document.delete.completed`
- `tag.create` / `tag.approve` / `tag.reject` / `tag.deprecate` / `tag.delete`
- `storage.write` / `storage.delete` / `storage.drop` / `cache.clear`
- `graph.entity.*` / `graph.relation.*`

Chaque event contient au minimum : `ts` (UTC ISO-8601), `event_type`, `event_id`, `trace_id`, `actor.id`, `actor.role`, `source.ip`, `http.method`, `http.path`, `status`, `outcome`, `workspace`, `resource.type/id`.

### 8.2 — Rétention

Selon DORA art. 9 et EBA/GL/2019/04 : minimum 5 ans pour les events relatifs aux accès aux données sensibles. La rotation locale (`journalctl`) n'est PAS suffisante — un sink durable externe (Splunk / ELK / Datadog) doit être configuré pour la production.

### 8.3 — Conformité par configuration

| Exigence | Configuration |
|---|---|
| DORA art. 9 (chiffrement at-rest et in-transit) | `MEMGRAPH_USE_TLS=true` + chiffrement disque côté Memgraph (responsabilité BNP) |
| DORA art. 9 (supply chain integrity) | `security_baseline=True` (défaut) ; SBOM v1.1.0 conservé |
| EBA/GL/2019/04 (audit log) | `TWIN_LOG_SINK` configuré vers sink durable |
| RGPD art. 17 (droit à l'effacement) | TTL DocStatus + purge physique KV au delete (livraison M5) |
| AI Act art. 4 (AI literacy) | Disclaimer "système IA" dans le footer WebUI (livraison M9) |
| AI Act art. 50 (transparence) | WebUI affiche les citations sources (déjà en place) |

---

## §9 — Procédures de release (interne au projet)

### 9.1 — Avant chaque release `1.1.x`

1. Mettre à jour la version dans `pyproject.toml`.
2. Mettre à jour `changelog.md`.
3. Exécuter `scripts/build_webui.sh` pour rebuilder le dist embarqué.
4. Exécuter `python -m build --wheel --sdist`.
5. Vérifier que le wheel contient bien `webui_dist/index.html` et **PAS** `webui_dist/mockServiceWorker.js`.
6. Générer le SBOM : `cyclonedx-py environment --of json --output-file docs/sbom-<version>.json`.
7. Tagger Git : `git tag -s -a v1.1.x` (signature GPG obligatoire).
8. Pousser le tag et créer une release Forgejo avec les artefacts attachés.

### 9.2 — Communication aux opérateurs BNP

Une note de release est envoyée par mail à la liste opérateurs BNP avec :

- Numéro de version
- Lien vers la release Forgejo
- Highlights (sécurité, breaking changes, nouveautés)
- Migration guide depuis la version précédente si breaking change
- Lien vers ce runbook révisé

---

## §10 — Hors-scope de ce document

- **Runbook Memgraph** : voir documentation Memgraph officielle + spécifique BNP.
- **Runbook iTrack / LightRAG natif** : voir documentation BNP propre.
- **Runbook SSO BNPP** : voir documentation MyAccess / SSO BNP.
- **Procédure de gestion d'incident DORA** : `docs/operations/incident-reporting.md` (à créer en M9.4).

---

## §11 — Auteurs et changelog

| Date | Version | Auteur | Modification |
|---|---|---|---|
| 2026-05-29 | Draft initial | Julien + Claude Opus 4.7 (1M ctx) | Création |

`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
