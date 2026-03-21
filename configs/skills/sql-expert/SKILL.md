---
name = "sql-expert"
description = "Query optimization, indexing strategies, migrations, anti-patterns, and transaction isolation"
tags = ["sql", "database", "postgres", "optimization", "migrations"]
max_inject_tokens = 2000
---

# SQL Expert

## Query Optimization

Use `EXPLAIN ANALYZE` (not just `EXPLAIN`) to see actual vs estimated rows and execution time.

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.name, COUNT(o.id)
FROM users u
JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name;
```

Key things to look for: sequential scans on large tables (missing index), nested loops with high row counts (consider hash join), large sort operations (add index for ORDER BY), row estimate mismatches (run `ANALYZE` to update statistics).

Use `WHERE EXISTS (SELECT 1 ...)` instead of `WHERE id IN (SELECT ...)` for correlated subqueries — the optimizer can short-circuit.

## Indexing Strategies

```sql
-- Composite index: column order matters (leftmost prefix rule)
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at DESC);
-- Supports: WHERE user_id = ? AND created_at > ?
-- Supports: WHERE user_id = ?
-- Does NOT support: WHERE created_at > ? (alone)

-- Partial index: smaller, faster for filtered queries
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';

-- Covering index: avoids table lookup
CREATE INDEX idx_users_email_name ON users (email) INCLUDE (name);
```

Don't index columns with low cardinality (boolean, status with 3 values) unless combined with selective columns. Index foreign keys — un-indexed FKs cause sequential scans on DELETE/UPDATE of parent rows.

`pg_stat_user_indexes`: check `idx_scan` = 0 for unused indexes. Drop them — they slow writes.

## Migrations

- One migration per change, never modify existing migrations.
- Backward-compatible changes: add column (nullable/default), add index concurrently, add table.
- Breaking changes in phases: add new column -> backfill -> deploy code using new column -> drop old column.
- `CREATE INDEX CONCURRENTLY` — doesn't lock the table (but can't run inside a transaction).
- Always test migrations against a production-sized dataset. A 3-second migration on dev can take 3 hours on prod.
- Include `DOWN` migration for rollback, even if it's destructive.

## Transaction Isolation

PostgreSQL default is `READ COMMITTED`. Understand the tradeoffs:

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Use Case |
|-------|-----------|-------------------|-------------|----------|
| READ COMMITTED | No | Yes | Yes | Default, most OLTP |
| REPEATABLE READ | No | No | No (in PG) | Consistent reports |
| SERIALIZABLE | No | No | No | Financial, inventory |

`SERIALIZABLE` in PostgreSQL uses SSI (not locking) — retry on serialization failures:

```sql
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- ... operations ...
COMMIT;
-- On ERROR 40001: retry the entire transaction
```

Use `SELECT ... FOR UPDATE` in `READ COMMITTED` for row-level locking when you read-then-write.

## Anti-Patterns

- `SELECT *` in application queries: wastes bandwidth, breaks on schema changes. List columns explicitly.
- N+1 queries: fetch parent, then loop fetching children. Use `JOIN` or batch `WHERE id = ANY($1)`.
- `OFFSET` for pagination on large tables: scans and discards rows. Use keyset pagination: `WHERE id > $last_id ORDER BY id LIMIT 20`.
- `NOT IN (NULL)` returns no rows — use `NOT EXISTS` instead.
- Storing comma-separated values: use array columns or junction tables. Violates 1NF, prevents indexing.
- Missing `LIMIT` on unbounded queries: one bad query returns 10M rows and OOMs your app.
- `UPDATE ... SET updated_at = NOW()` without `WHERE` clause: updates every row in the table.
- UUIDs as primary key: random UUIDs cause index fragmentation. Use UUIDv7 (time-ordered) or `BIGSERIAL`.
