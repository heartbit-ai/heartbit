---
name = "typescript-expert"
description = "TypeScript strict mode, generics, discriminated unions, build tools, and module systems"
tags = ["typescript", "javascript", "generics", "build-tools", "types"]
max_inject_tokens = 2000
---

# TypeScript Expert

## Strict Mode

Enable all strict checks in `tsconfig.json`. The key flags beyond `"strict": true`:

```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noPropertyAccessFromIndexSignature": true
  }
}
```

`noUncheckedIndexedAccess` makes array/object index access return `T | undefined` — catches real bugs. `exactOptionalPropertyTypes` distinguishes `{ x?: string }` (missing) from `{ x: string | undefined }` (present but undefined).

## Generics

Constrain generics with `extends`. Use `infer` in conditional types for extraction.

```typescript
type ExtractPromise<T> = T extends Promise<infer U> ? U : never;

function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}
```

Avoid over-generic code. If a generic parameter is used only once, you probably don't need it — use a concrete type or union instead.

Use `satisfies` operator to validate a value matches a type without widening:

```typescript
const config = {
  port: 3000,
  host: "localhost",
} satisfies ServerConfig;
// typeof config retains literal types, not widened to ServerConfig
```

## Discriminated Unions

Model state machines with tagged unions. Exhaustive checking via `never`:

```typescript
type Result<T> =
  | { status: "ok"; data: T }
  | { status: "error"; error: Error }
  | { status: "loading" };

function handle<T>(result: Result<T>) {
  switch (result.status) {
    case "ok": return result.data;
    case "error": throw result.error;
    case "loading": return null;
    default: {
      const _exhaustive: never = result;
      throw new Error(`Unhandled: ${_exhaustive}`);
    }
  }
}
```

Prefer discriminated unions over optional fields: `{ type: "admin"; permissions: string[] } | { type: "guest" }` instead of `{ isAdmin?: boolean; permissions?: string[] }`.

## Build Tools

- **Vite**: dev server + production bundler. Use for web apps. `vite.config.ts` with `defineConfig()`.
- **tsup**: `esbuild`-powered bundler for libraries. Outputs ESM + CJS with `.d.ts` generation.
- **tsx**: drop-in `node` replacement for running `.ts` files directly (uses esbuild under the hood).
- **tsc**: only use for type checking (`tsc --noEmit`), not for building. Too slow for builds.
- **Biome**: replaces ESLint + Prettier. Single tool, faster, fewer config files.

## Module Systems

Use ESM (`"type": "module"` in `package.json`). Set `"module": "ESNext"` and `"moduleResolution": "bundler"` (or `"nodenext"` for pure Node.js).

Dual-package publishing: `exports` field in `package.json`:

```json
{
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    }
  }
}
```

Barrel files (`index.ts` re-exports) hurt tree-shaking — use direct imports in large packages.

## Common Pitfalls

- `as` casts bypass type checking — use type guards or `satisfies` instead.
- `Record<string, T>` allows any key access without errors. Use `Map<string, T>` or `Partial<Record<K, T>>`.
- `enum` has runtime cost and quirks. Prefer `as const` objects or union types.
- `Promise<void>` vs `Promise<undefined>` — the former allows no return statement.
- `any` propagates silently. Use `unknown` and narrow with type guards.
- `Object.keys()` returns `string[]`, not `(keyof T)[]` — this is intentional (structural typing).
- `!` non-null assertion hides bugs. Prefer optional chaining `?.` with explicit handling.
