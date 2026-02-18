# Heartbit - Plan d'implémentation initial

> Archive du plan d'étude et des décisions prises avant le premier commit.
> Ce document est une référence historique. Les décisions actives sont dans CLAUDE.md.

## Context

Heartbit est un runtime multi-agent enterprise en Rust. L'objectif est un orchestrateur qui spawn des sub-agents, les fait travailler en parallèle, et agrège les résultats. On construit lean et on fait grossir organiquement.

**Jour 1** : Un orchestrateur multi-agents fonctionnel.
**Pas de** : NATS, event sourcing, DAG scheduler, durable execution. On ajoute quand le besoin arrive.

---

## 1. Structure minimale

```
heartbit/
  Cargo.toml                  # Workspace
  crates/
    heartbit/                 # Tout dans un seul crate lib
      src/
        lib.rs
        llm/
          mod.rs              # LlmProvider trait
          anthropic.rs        # Premier provider (streaming + tool calls)
          types.rs            # Message, ToolCall, CompletionRequest/Response
        agent/
          mod.rs              # Agent trait + AgentRunner
          orchestrator.rs     # Orchestrateur qui spawn des sub-agents
          context.rs          # Contexte de conversation (messages, tools)
        tool/
          mod.rs              # Tool trait
          mcp.rs              # Client MCP (agentgateway) - Phase 2
    heartbit-cli/             # Binaire
      src/
        main.rs               # CLI minimal
```

2 crates. C'est tout. On split quand un module devient trop gros.

---

## 2. Ce qu'on construit le jour 1

### 2.1 LLM Provider (Anthropic uniquement)

- Streaming SSE natif (pas de crate tiers, on parse les events nous-mêmes)
- Support natif des parallel tool calls (Claude les émet déjà)
- Gestion du `tool_use` / `tool_result` content blocks

### 2.2 Agent trait + Runner

- Agent trait: name, system_prompt, tools, run
- AgentContext: messages, max_turns
- AgentRunner: loop LLM → tool calls → tool results → repeat
- Parallel tool execution via JoinSet

### 2.3 Orchestrator

- L'orchestrateur est lui-même un Agent
- Son outil principal: `delegate_task` pour spawner des sub-agents en parallèle
- Flat hierarchy: sub-agents ne spawnent pas

### 2.4 CLI minimal

- Lit ANTHROPIC_API_KEY depuis l'env
- Prend une tâche en argument
- Spawn l'orchestrateur avec des sub-agents préconfigurés

---

## 3. Roadmap

| Quand | Quoi | Comment |
|-------|------|---------|
| **Semaine 2** | Tools MCP via agentgateway | `tool/mcp.rs` : client MCP (Streamable HTTP) |
| **Semaine 2** | Streaming output | L'orchestrateur stream les résultats |
| **Semaine 3** | Providers additionnels | `llm/openai.rs`, `llm/google.rs`, `llm/ollama.rs` |
| **Semaine 3** | Config TOML | Agent definitions depuis des fichiers config |
| **Quand nécessaire** | NATS JetStream | Distribution multi-machine |
| **Quand nécessaire** | Event sourcing | Crash recovery |
| **Quand nécessaire** | API REST/gRPC | Clients externes |
| **Dernier** | Xavyo integration | Production enterprise |

---

## 4. Décisions techniques

| Choix | Décision | Pourquoi |
|-------|----------|----------|
| Parallel tool execution | `tokio::JoinSet` | Simple, zero overhead |
| Inter-agent communication | `tokio::mpsc` channels | In-process, zero latence |
| LLM streaming | Parser SSE maison sur `reqwest` | Pas de dépendance inutile |
| Error handling | `thiserror` + `anyhow` | Standard Rust |
| Serialization | `serde` + `serde_json` | Standard |
| HTTP client | `reqwest` (rustls-tls, stream) | Standard, async natif |
| Agent hierarchy | Flat (1 niveau) | Simple, suffisant pour le jour 1 |

---

## 5. Dépendances Cargo (minimales)

```toml
tokio, reqwest, serde, serde_json, thiserror, anyhow, uuid, tracing, tracing-subscriber, bytes, futures
```

10 dépendances. On ajoute au besoin.

---

## 6. Études préliminaires

### Pourquoi pas un DAG scheduler ?
JoinSet suffit pour le jour 1. Les sub-agents sont indépendants, pas de dépendances entre eux.
On ajoutera un scheduler si on a besoin de chaîner des agents avec des dépendances.

### Pourquoi pas NATS dès le départ ?
tokio channels pour la comm in-process. NATS quand on distribue sur plusieurs machines.
Pas d'over-engineering avant le besoin.

### Pourquoi un parser SSE maison ?
Les crates SSE existantes ajoutent des dépendances et de la complexité pour un format simple.
Le format SSE est `data: ...\n\n`. On parse ça avec quelques lignes de code.

### Pourquoi flat hierarchy ?
Les agents récursifs (sub-agents qui spawnent des sub-sub-agents) créent de la complexité
et des risques de boucle infinie. Un seul niveau suffit pour commencer.
