# Prisme D - Auth, sessions & security baseline LightRAG 1.4.9.11

Audit realise sur le wheel exact `lightrag-hku==1.4.9.11` extrait dans `/private/tmp/lrag14911`. Les references `$LRAG/...` pointent vers cette version cible.

Contexte BNP : les API keys sont interdites, la cible est OAuth2 ou mTLS. Le role MyAccess `knowledge` doit venir du SSO, pas d'une gestion locale LightRAG.

## 1. Mecanisme d'auth natif

LightRAG natif combine trois comportements :

- JWT signe localement, emis par `/login`.
- API key statique via header `X-API-Key`.
- Mode ouvert si ni comptes internes ni API key ne sont configures.

JWT :

- `$LRAG/api/auth.py:23` definit `AuthHandler`.
- `$LRAG/api/auth.py:25` lit le secret depuis `global_args.token_secret`.
- `$LRAG/api/auth.py:26` lit l'algorithme depuis `global_args.jwt_algorithm`.
- `$LRAG/api/auth.py:64` calcule `exp = datetime.utcnow() + timedelta(hours=expire_hours)`.
- `$LRAG/api/auth.py:71` signe avec `jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)`.
- `$LRAG/api/auth.py:87` valide avec `jwt.decode(token, self.secret, algorithms=[self.algorithm])`.

Configuration crypto :

- `$LRAG/api/config.py:397` definit `TOKEN_SECRET`, defaut `"lightrag-jwt-default-secret"`.
- `$LRAG/api/config.py:400` definit `JWT_ALGORITHM`, defaut `"HS256"`.
- Il n'y a pas de support JWKS, OIDC discovery, audience, issuer, scopes, ni validation de certificat client.

API key :

- `$LRAG/api/config.py:167` ajoute `--key`.
- `$LRAG/api/config.py:169` lit `LIGHTRAG_API_KEY`.
- `$LRAG/api/lightrag_server.py:344` recalcule `api_key = os.getenv("LIGHTRAG_API_KEY") or args.key`.
- `$LRAG/api/utils_api.py:103` cree une security `APIKeyHeader`.
- `$LRAG/api/utils_api.py:105` fixe le nom de header a `X-API-Key`.
- `$LRAG/api/utils_api.py:226` a `:231` accepte la requete si `api_key_header_value == api_key`.

OAuth2 natif :

- LightRAG utilise `OAuth2PasswordBearer` uniquement comme mecanisme Swagger/FastAPI pour lire un bearer token : `$LRAG/api/utils_api.py:98`.
- Le login est un Password Grant local via `OAuth2PasswordRequestForm`, pas un vrai Authorization Code + PKCE : `$LRAG/api/lightrag_server.py:1162`.
- Il n'y a pas de Basic auth native detectee.

Roles natifs :

- Le payload contient `role`, defaut `"user"` : `$LRAG/api/auth.py:19`.
- Les tokens guest utilisent `role="guest"` : `$LRAG/api/lightrag_server.py:1138` et `:1166`.
- Les tokens utilisateur internes utilisent `role="user"` : `$LRAG/api/lightrag_server.py:1184`.
- Aucun role `knowledge`, aucun mapping MyAccess, aucun scope OAuth2.

## 2. Stockage des credentials

Secrets et comptes :

- `.env` est charge dans `auth.py` avec `load_dotenv(dotenv_path=".env", override=False)` : `$LRAG/api/auth.py:13`.
- `.env` est aussi charge dans `config.py` : `$LRAG/api/config.py:49`.
- `AUTH_ACCOUNTS` est lu depuis l'environnement, defaut chaine vide : `$LRAG/api/config.py:396`.
- `TOKEN_SECRET` est lu depuis l'environnement, defaut faible et public : `$LRAG/api/config.py:397`.
- `TOKEN_EXPIRE_HOURS` defaut 48h : `$LRAG/api/config.py:398`.
- `GUEST_TOKEN_EXPIRE_HOURS` defaut 24h : `$LRAG/api/config.py:399`.
- `LIGHTRAG_API_KEY` est lu depuis l'environnement ou `--key` : `$LRAG/api/config.py:167`, `$LRAG/api/config.py:169`, `$LRAG/api/lightrag_server.py:344`.

Stockage en memoire :

- `AuthHandler.__init__` initialise `self.accounts = {}` : `$LRAG/api/auth.py:29`.
- Si `AUTH_ACCOUNTS` existe, il parse chaque entree `username:password` et stocke le mot de passe en clair : `$LRAG/api/auth.py:31` a `:34`.
- Le login compare directement `auth_handler.accounts.get(username) != form_data.password` : `$LRAG/api/lightrag_server.py:1179` a `:1181`.
- Aucun hash, salt, KDF, verrouillage de compte, rotation de secret, ou stockage externe.

Generation :

- Les JWT sont generes a la demande par `auth_handler.create_token(...)`.
- `/auth-status` genere un token guest si aucun compte interne n'est configure : `$LRAG/api/lightrag_server.py:1132` a `:1144`.
- `/login` genere aussi un token guest quand aucun compte interne n'est configure : `$LRAG/api/lightrag_server.py:1162` a `:1178`.
- Il n'y a pas de generation automatique de secret fort au boot. Si `TOKEN_SECRET` manque, le secret par defaut est utilise.

## 3. Dependencies FastAPI

Dependency principale :

- `$LRAG/api/utils_api.py:80` expose `get_combined_auth_dependency(api_key=None)`.
- Cette fonction cree une closure `combined_dependency(...)` : `$LRAG/api/utils_api.py:109`.
- Le serveur principal instancie une closure par app : `$LRAG/api/lightrag_server.py:459`.

Routes protegees dans `lightrag_server.py` :

- `/health` utilise `dependencies=[Depends(combined_auth)]` : `$LRAG/api/lightrag_server.py:1197` a `:1199`.
- Les routers documents, query, graph, ollama creent chacun leur propre `combined_auth` :
  - documents : `$LRAG/api/routers/document_routes.py:2042` a `:2046`.
  - query : `$LRAG/api/routers/query_routes.py:193` a `:199`.
  - graph : `$LRAG/api/routers/graph_routes.py:89` a `:92`.
  - ollama : `$LRAG/api/routers/ollama_api.py:229` a `:233`.

Routes non protegees dans le serveur natif :

- `/docs` : `$LRAG/api/lightrag_server.py:1106`.
- `/docs/oauth2-redirect` : `$LRAG/api/lightrag_server.py:1119`.
- `/` : `$LRAG/api/lightrag_server.py:1124`.
- `/auth-status` : `$LRAG/api/lightrag_server.py:1132`.
- `/login` : `$LRAG/api/lightrag_server.py:1162`.
- `/webui` static mount : `$LRAG/api/lightrag_server.py:1345`.
- Fallback `/webui` et `/webui/` si assets absents : `$LRAG/api/lightrag_server.py:1357` a `:1361`.

Surcharge possible :

- FastAPI `app.dependency_overrides` peut surcharger une dependency par identite de fonction, mais ici la dependency est une closure creee dans `get_combined_auth_dependency()`. Elle n'est pas un symbole stable importable.
- Hook robuste : patcher `lightrag.api.utils_api.get_combined_auth_dependency` avant que `lightrag_server.create_app()` et les router factories soient executes.
- Hook post-create possible : parcourir `app.routes`, extraire les `Depends` deja attaches, et poser `app.dependency_overrides[closure] = oauth_dependency`. C'est faisable mais fragile.
- Hook serveur recommande depuis Prisme B : wrapper `lightrag.api.lightrag_server.create_app`, appeler l'original, puis normaliser les dependencies/routes avant `uvicorn.run()`.

## 4. CORS

Config native :

- `CORS_ORIGINS` est lu depuis l'environnement, defaut `"*"` : `$LRAG/api/config.py:390`.
- `get_cors_origins()` retourne `["*"]` si la valeur vaut `"*"` ; sinon split par virgule : `$LRAG/api/lightrag_server.py:438` a `:445`.
- Middleware CORS : `$LRAG/api/lightrag_server.py:447` a `:456`.

Parametres appliques :

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-New-Token"],
)
```

Baseline : ouvert a tous par defaut, credentials autorises, toutes methodes et tous headers autorises. C'est configurable par `CORS_ORIGINS`, mais pas verrouille par defaut.

## 5. Failles et pratiques discutables

1. Secret JWT faible par defaut : `TOKEN_SECRET` vaut `"lightrag-jwt-default-secret"` si absent (`$LRAG/api/config.py:397`). Tout deploiement sans secret explicite a des JWT forgeables.

2. API key autorisee comme mecanisme de protection : `LIGHTRAG_API_KEY` / `X-API-Key` (`$LRAG/api/lightrag_server.py:344`, `$LRAG/api/utils_api.py:105`). Non conforme au contexte BNP.

3. Whitelist par defaut trop large : `WHITELIST_PATHS` vaut `"/health,/api/*"` (`$LRAG/api/config.py:393`). La dependency accepte tout path whitelist avant token/API key (`$LRAG/api/utils_api.py:117` a `:123`). Les endpoints Ollama sous `/api/*`, y compris `/api/generate` et `/api/chat`, sont donc ouverts par defaut malgre leurs `Depends`.

4. Bypass API key par token guest en mode API-key seule : si `AUTH_ACCOUNTS` est vide mais `LIGHTRAG_API_KEY` est configure, `/auth-status` ou `/login` emettent un token guest (`$LRAG/api/lightrag_server.py:1136` a `:1144`, `:1164` a `:1178`). Ensuite `combined_dependency` accepte `role=="guest"` quand `auth_configured` est false (`$LRAG/api/utils_api.py:203` a `:205`) avant la verification API key. Donc l'API key peut etre contournee.

5. Passwords internes en clair : `AUTH_ACCOUNTS` est parse en `username:password` et stocke tel quel (`$LRAG/api/auth.py:31` a `:34`), puis compare en clair (`$LRAG/api/lightrag_server.py:1179` a `:1181`).

6. Pas de claims enterprise : pas de `iss`, `aud`, `scope`, `azp`, `client_id`, ni role MyAccess `knowledge`. Le role est local (`guest`/`user`) et injecte par LightRAG.

7. Sliding session par header custom : auto-renew active par defaut (`$LRAG/api/config.py:403`), renouvelle quand le token approche du seuil (`$LRAG/api/utils_api.py:134` a `:181`) et expose `X-New-Token` en CORS (`$LRAG/api/lightrag_server.py:454`). Cela prolonge les sessions sans integration SSO ni revocation centrale.

8. Logout/revocation absents : JWT stateless HS local, aucune blacklist, introspection, session server-side, ou revocation SSO.

9. Login sans controles anti-bruteforce : `/login` compare directement les credentials et retourne 401 (`$LRAG/api/lightrag_server.py:1179` a `:1181`), sans rate limit, lockout, audit event, captcha, ou detection.

10. CORS ouvert par defaut avec credentials : `CORS_ORIGINS="*"` (`$LRAG/api/config.py:390`) et `allow_credentials=True` (`$LRAG/api/lightrag_server.py:451`).

11. Health divulgue de la configuration : `/health` retourne bindings, modeles, backends, directories et flags (`$LRAG/api/lightrag_server.py:1249` a `:1297`). Il est aussi dans la whitelist par defaut.

12. Auth state calcule au module import : `auth_configured = bool(auth_handler.accounts)` dans `utils_api.py` (`$LRAG/api/utils_api.py:76` a `:77`) et dans `lightrag_server.py` (`$LRAG/api/lightrag_server.py:80` a `:81`). Changer les comptes apres import ne recalcule pas ces copies.

## 6. Plan de swap OAuth2 / mTLS

Objectif cible :

- Browser/UI : OAuth2 Authorization Code + PKCE via SSO BNP.
- Service-to-service : OAuth2 Client Credentials ou mTLS selon le flux BNP.
- Role `knowledge` : lu depuis les claims SSO/MyAccess, jamais gere dans LightRAG.
- API keys : desactivees et retirees du chemin nominal.

Point d'injection recommande :

1. Avant `create_app()` : patcher `lightrag.api.utils_api.get_combined_auth_dependency`.
2. Retourner une dependency stable qui valide :
   - bearer access token via JWKS (`iss`, `aud`, `exp`, `nbf`, signature, `kid`) ou introspection endpoint ;
   - presence du role/scope MyAccess `knowledge` ;
   - eventuellement certificat client via headers de reverse proxy mTLS (`X-Forwarded-Client-Cert`) seulement si le proxy est de confiance.
3. Garder `lightrag_server.create_app` comme factory native, mais enveloppee depuis notre `register()` pour installer le patch avant les router factories.

Pseudo-flux :

```text
register()
  _patch_version_string()
  _patch_lightrag_server_auth()
    import lightrag.api.utils_api as utils_api
    utils_api.get_combined_auth_dependency = make_oauth_dependency_factory(...)
  _patch_lightrag_server_ui()

lightrag_server.create_app()
  combined_auth = patched get_combined_auth_dependency(...)
  routers create their combined_auth from patched factory
```

Validator OAuth2 :

- JWKS mode : cache JWKS par `kid`, verifier signature RS256/ES256, `iss`, `aud`, `exp`, `nbf`, puis extraire `roles`, `groups`, `scope` ou claim MyAccess equivalent.
- Introspection mode : appeler endpoint introspection pour tokens opaques, verifier `active`, `client_id`, `scope`, expiration, et role `knowledge`.
- Refuser tout token local LightRAG en mode BNP strict.

Cohabitation transition :

- Mode `strict`: OAuth2/mTLS uniquement, pas de JWT local, pas d'API key.
- Mode `legacy-fallback`: essayer OAuth2/mTLS d'abord ; si absent, accepter JWT LightRAG seulement si un flag explicite `TWIN_AUTH_LEGACY_FALLBACK=true` est active et jamais en prod BNP.
- Desactiver `WHITELIST_PATHS=/api/*`; au minimum `WHITELIST_PATHS=/docs,/docs/oauth2-redirect,/webui/*` selon besoin, mais pas les endpoints data.
- Desactiver token auto-renew natif (`TOKEN_AUTO_RENEW=false`) des que le SSO gere la session.
- Ne plus exposer `/login` local ; le remplacer par redirect/login SSO ou le retirer de l'OpenAPI.

mTLS :

- Terminer mTLS au reverse proxy ou ingress BNP.
- Ne faire confiance aux headers client-cert que si l'app est derriere un proxy de confiance qui strippe les headers entrants.
- Mapper l'identite cert (`subject`, SAN, SPIFFE, ou header normalise BNP) vers un principal et verifier le role `knowledge` via SSO/annuaire, pas via config locale.

Notre `server/auth.py` :

- Il ne peut pas se substituer tel quel au natif par simple import : il attend `Authorization: Bearer <key-or-jwt>` et a ses propres globals (`_static_api_key`, `_jwt_secret`) dans `src/twindb_lightrag_memgraph/server/auth.py:32` a `:39`.
- Il est surchargeable dans notre propre app parce que les routes utilisent `Depends(require_auth)` stable (`src/twindb_lightrag_memgraph/server/app.py:247` a `:293`).
- Pour LightRAG natif, `require_auth` ne remplace pas automatiquement les closures `combined_auth`. Il faut soit patcher `get_combined_auth_dependency`, soit appliquer `app.dependency_overrides` sur les closures apres creation.
- Notre module est utile comme squelette d'interface stable (`require_auth`) mais doit etre remplace par un validator OAuth2/mTLS. Il conserve aujourd'hui les memes problemes de fond : API key statique, JWT HS256 local, defaults `admin`/`changeme`, CORS ouvert dans notre app (`src/twindb_lightrag_memgraph/server/app.py:207` a `:214`).

## Risques compliance pour audit BCE

| Risque | Severite | Evidence | Remediation |
| --- | --- | --- | --- |
| Secret JWT par defaut public | bloquant | `$LRAG/api/config.py:397` | Exiger secret externe fort ou supprimer JWT local en prod |
| API key comme auth applicative | bloquant | `$LRAG/api/utils_api.py:103` a `:107`, `:226` a `:231` | Remplacer par OAuth2/mTLS |
| `/api/*` ouvert par defaut | bloquant | `$LRAG/api/config.py:393`, `$LRAG/api/utils_api.py:117` a `:123` | Retirer `/api/*` de whitelist |
| Contournement API key via guest token | bloquant | `$LRAG/api/lightrag_server.py:1136` a `:1144`, `$LRAG/api/utils_api.py:203` a `:205` | Ne jamais emettre/accepter guest token si API key ou auth externe active |
| Mots de passe internes en clair | bloquant | `$LRAG/api/auth.py:31` a `:34` | Supprimer comptes internes, deleguer au SSO |
| Role `knowledge` absent du SSO | bloquant | `$LRAG/api/auth.py:19`, `$LRAG/api/lightrag_server.py:1184` | Mapper claims MyAccess et refuser sans role |
| Pas de validation issuer/audience/scope | bloquant | JWT decode local `$LRAG/api/auth.py:87` | JWKS/introspection avec `iss`, `aud`, scopes |
| Pas de revocation centralisee | serieux | JWT stateless local `$LRAG/api/auth.py:71`, `:87` | Tokens SSO courts + introspection/revocation |
| Token auto-renew local par defaut | serieux | `$LRAG/api/config.py:403`, `$LRAG/api/utils_api.py:134` a `:181` | Desactiver, laisser le SSO gerer sessions |
| Login sans anti-bruteforce | serieux | `$LRAG/api/lightrag_server.py:1162` a `:1181` | Supprimer login local ou ajouter controles compensatoires |
| CORS wildcard avec credentials | serieux | `$LRAG/api/config.py:390`, `$LRAG/api/lightrag_server.py:448` a `:456` | Liste blanche stricte par environnement |
| Health divulgue config et est whitelist | serieux | `$LRAG/api/config.py:393`, `$LRAG/api/lightrag_server.py:1249` a `:1297` | Auth obligatoire ou health minimal |
| Auth state copie au module import | cosmetic | `$LRAG/api/utils_api.py:76` a `:77`, `$LRAG/api/lightrag_server.py:80` a `:81` | Recalcul dynamique ou config immutable explicite |

## Hookpoints conseilles

1. `register()` avant import/boot serveur : installer `_patch_lightrag_server_auth()` avant `_patch_lightrag_server_ui()`.
2. Patch preferentiel : `lightrag.api.utils_api.get_combined_auth_dependency`.
3. Patch complementaire : wrapper `lightrag.api.lightrag_server.create_app` pour :
   - retirer ou remplacer `/login` et `/auth-status` ;
   - verifier que `/api/*` n'est pas whitelist ;
   - remplacer dependencies residuelles si des closures ont deja ete creees ;
   - durcir CORS ;
   - monter les endpoints OAuth2 callback/metadata si necessaire.
4. Pour notre serveur Twin : remplacer `src/twindb_lightrag_memgraph.server.auth.require_auth` par une dependency stable OAuth2/mTLS et utiliser `app.dependency_overrides[require_auth]` en tests seulement.

## Bottom line

Le modele natif LightRAG `1.4.9.11` est acceptable pour un outil local, pas pour BNP : secret JWT par defaut, API key, guest tokens, whitelist `/api/*`, absence d'issuer/audience/scope et roles locaux. Le swap doit se faire avant `create_app()` en remplacant la factory `get_combined_auth_dependency`, puis en enveloppant `create_app` pour nettoyer les routes publiques et CORS. Le role MyAccess `knowledge` doit etre une decision du validator SSO/mTLS, pas un champ JWT genere par LightRAG.
