# Blackboard

## agent:worker

### Vérification non destructive — Option B (2026-06-14)

Commandes lancées depuis la racine du dépôt, sans correction ni modification de fichiers source/config :

1. `cargo fmt -- --check`
   - Statut exit : 0
   - Durée : real ~1.861s
   - Diagnostics significatifs : aucun warning/erreur affiché.

2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - Statut exit : 0
   - Durée : real ~43.308s (`Finished ... in 43.13s`).
   - Diagnostics significatifs : aucun warning/erreur final.

3. `cargo test --workspace --all-features`
   - Statut exit : 0
   - Durée : real ~2m12.012s.
   - Diagnostics significatifs : warnings Cargo de collisions de noms de sortie pour plusieurs examples (`custom_tool`, `eval`, `guardrails`, `hello_agent`, `mcp_agent`, `memory`, `multi_agent`, `simple_agent`) entre les packages `heartbit-core` et `heartbit`; note Cargo indique que cela pourrait devenir une erreur future (cargo issue #6313). Tests et doctests visibles terminés avec succès, notamment bloc final `heartbit-tui`: 299 passed, 0 failed; doctests `heartbit`, `heartbit_core`, `heartbit_ghost`, `heartbit_macro`, `heartbit_sensors`, `heartbit_telegram` OK avec quelques doctests ignorés.

Aucune commande demandée n'a échoué ni été impossible à exécuter. Aucun correctif appliqué.

## agent:researcher

### TERMINÉ — audit statique Option B non destructif

Résumé final:
- Aucun secret de production évident détecté par recherche statique; constats prudents sur identifiants de dev/exemple faibles dans `daemon-dev.toml`, `telegram-test.toml`, `docker-compose*.yml`.
- Risque principal: daemon HTTP peut exposer routes sensibles si aucun `HEARTBIT_API_KEY`/`[daemon.auth].bearer_tokens`/JWKS n'est configuré; le code log seulement un avertissement et merge les routes protégées. Références: `crates/heartbit-cli/src/daemon/mod.rs:1091-1105`, `1244-1261`, `1282-1290`.
- Metrics daemon publiques/non authentifiées: `/v1/metrics` dans routes publiques (`crates/heartbit-cli/src/daemon/mod.rs:1244-1248`), handler Prometheus (`crates/heartbit-cli/src/daemon/handlers.rs:711-734`), metrics par défaut activées (`crates/heartbit-core/src/config/daemon.rs:849-854`). `daemon-lite.toml:17,21-22` bind `0.0.0.0:8080` + metrics enabled.
- Configs exposées/faibles: `docker-compose.daemon.yml:12,23` et `docker-compose.yml:31` utilisent mot de passe Postgres `heartbit`; `docker-compose.yml:27-28` expose 5432; `docker-compose.yml:45-50` expose Kafka 9092/0.0.0.0. `daemon-dev.toml:32,113` DSN `hbv:hbv_dev`; `telegram-test.toml:16` DSN `heartbit:heartbit`.
- Occurrences panic-family Rust: heuristique sur `git grep` => runtime env. 104 `unwrap()`, 211 `expect(`, 5 `panic!`, 0 `todo!`/`unimplemented!`; inline tests env. 5015 `unwrap()`, 1003 `expect(`, 299 `panic!`, 3 `unimplemented!`; fichiers tests/examples 189 occurrences. Fichier runtime le plus chargé: `crates/heartbit-core/src/tool/builtins/bash.rs` (43 occurrences, principalement `Mutex::lock().expect(...)`, p.ex. lignes 206,334,356).
- CI présent: `.github/workflows/ci.yml` exécute fmt, clippy `-D warnings`, `cargo test --no-run`, `cargo test --lib` lignes 27-37; release utilise secret GitHub `CARGO_REGISTRY_TOKEN` lignes 102-109. `.dockerignore` présent mais minimal (`target/`, `.git/`, `.github/`, `*.md`, `!CLAUDE.md`).
- Vérification croisée: cohérent avec `agent:worker` (CI/tests/clippy OK). Pas de contradiction identifiée.
