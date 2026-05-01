---
name = "api-design"
description = "REST conventions, error formats, versioning, pagination, rate limiting, and OpenAPI"
tags = ["api", "rest", "openapi", "http", "design"]
max_inject_tokens = 2000
---

# API Design Expert

## REST Conventions

Resources are nouns, not verbs. Use plural: `/users`, `/orders/{id}/items`.

```
GET    /users          → 200 + list
POST   /users          → 201 + created resource + Location header
GET    /users/{id}     → 200 + resource | 404
PUT    /users/{id}     → 200 + updated resource (full replace)
PATCH  /users/{id}     → 200 + updated resource (partial)
DELETE /users/{id}     → 204 (no body)
```

Use HTTP methods correctly: `GET` is safe and idempotent. `PUT` is idempotent. `POST` is neither. `PATCH` with JSON Merge Patch (`application/merge-patch+json`) for partial updates.

Nested resources for clear ownership: `/users/{id}/orders`. Max 2 levels of nesting — beyond that, promote to top-level with filter: `/orders?user_id=123`.

## Error Format

Use RFC 7807 Problem Details consistently:

```json
{
  "type": "https://api.example.com/errors/insufficient-funds",
  "title": "Insufficient Funds",
  "status": 422,
  "detail": "Account balance is $10.00, but transfer requires $25.00",
  "instance": "/transfers/abc-123",
  "balance": 1000,
  "required": 2500
}
```

Map errors to correct status codes: 400 (malformed request), 401 (no/invalid auth), 403 (authenticated but not authorized), 404 (not found), 409 (conflict/duplicate), 422 (valid syntax but semantic error), 429 (rate limited), 500 (server bug), 503 (temporarily unavailable).

Never return 200 with an error body. Never expose stack traces or internal details in production.

## Versioning

URL path versioning (`/v1/users`) is the simplest and most widely adopted. Header versioning (`Accept: application/vnd.api+json; version=2`) is more "correct" but harder to test and debug.

Rules: never break existing clients on a version. Additive changes (new fields, new endpoints) don't need a version bump. Removing fields, changing types, or renaming endpoints require a new version. Support N-1 version minimum. Deprecation headers: `Deprecation: true`, `Sunset: Sat, 01 Jan 2025 00:00:00 GMT`.

## Pagination

Keyset (cursor-based) pagination for performance and consistency:

```
GET /users?after=eyJpZCI6MTAwfQ&limit=20

Response:
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6MTIwfQ",
    "has_more": true
  }
}
```

Cursor is an opaque base64-encoded token (e.g., `{"id": 120}`). Clients must not parse it. Offset pagination (`?page=5&per_page=20`) is acceptable for small datasets but suffers from drift on inserts and O(n) performance.

Always cap `limit`/`per_page` with a server maximum (e.g., 100). Default to a reasonable value (20-50).

## Rate Limiting

Return rate limit headers on every response:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1704067200
Retry-After: 30
```

429 response must include `Retry-After`. Implement with token bucket or sliding window. Rate limit by API key, not IP (IPs are shared behind NATs/proxies). Different tiers for different endpoints (reads vs writes).

## OpenAPI

- Write spec first (design-first), generate server stubs and client SDKs.
- `operationId` on every endpoint — drives SDK method names.
- Use `$ref` for shared schemas in `components/schemas/`.
- `required` array on objects — don't rely on consumers reading `nullable`.
- `examples` on request/response bodies — better than descriptions for understanding.
- Validate spec in CI: `spectral lint openapi.yaml`.
- Generate types, not runtime clients: `openapi-typescript` for TS, `openapiv3` crate for Rust.

## Anti-Patterns

- Verbs in URLs: `/getUser`, `/createOrder` — use HTTP methods instead.
- Returning 200 for everything with `{ "success": false }` — defeats HTTP semantics.
- Unbounded list endpoints without pagination — one request returns 1M records.
- Accepting and silently ignoring unknown fields — use strict parsing, return 400.
- Auth tokens in query parameters — logged in server access logs, browser history, proxy caches.
