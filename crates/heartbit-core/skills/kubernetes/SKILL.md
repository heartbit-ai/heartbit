---
name = "kubernetes"
description = "K8s resource limits, RBAC, health checks, debugging, common misconfigs, and helm patterns"
tags = ["kubernetes", "k8s", "devops", "helm", "containers"]
max_inject_tokens = 2000
---

# Kubernetes Expert

## Resource Limits

Always set both requests and limits. Requests determine scheduling, limits prevent noisy neighbors.

```yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi
```

CPU limits cause throttling (CFS quota), not OOMKill. Memory limits trigger OOMKill. Set memory request = limit to avoid OOM surprises (Guaranteed QoS class). CPU requests should be realistic — over-requesting wastes cluster capacity.

Use `LimitRange` on namespaces for defaults. `ResourceQuota` to cap total namespace consumption.

## Health Checks

Three probes, distinct purposes:

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8080 }
  initialDelaySeconds: 10
  periodSeconds: 15
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /ready, port: 8080 }
  periodSeconds: 5
  failureThreshold: 2
startupProbe:
  httpGet: { path: /healthz, port: 8080 }
  periodSeconds: 5
  failureThreshold: 30  # 150s total startup budget
```

Liveness: is the process stuck? Restart on failure. Don't check dependencies here — a database outage shouldn't restart your app. Readiness: can this pod serve traffic? Remove from service endpoints on failure. Startup: for slow-starting apps, prevents liveness kills during init.

## RBAC

Principle of least privilege. Never use `cluster-admin` for application workloads.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: myapp
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list"]
    resourceNames: ["myapp-config"]  # scope to specific resources
```

Use `ServiceAccount` per deployment. Disable automounting when not needed: `automountServiceAccountToken: false`.

## Debugging

```bash
# Pod won't start
kubectl describe pod <name> -n <ns>    # Events section shows scheduling/pull/mount failures
kubectl logs <pod> -c <container> --previous   # Logs from crashed container

# Networking
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash
nslookup <service>.<namespace>.svc.cluster.local
curl -v http://<service>:8080/healthz

# Resource pressure
kubectl top pods -n <ns> --sort-by=memory
kubectl get events -n <ns> --sort-by='.lastTimestamp'

# Exec into running pod
kubectl exec -it <pod> -c <container> -- /bin/sh
```

## Common Misconfigs

- Missing `podDisruptionBudget`: voluntary evictions (node drain) kill all replicas simultaneously.
- `imagePullPolicy: Always` without image digest: pulls on every restart, slow + registry dependency.
- Secrets in ConfigMaps: use `Secret` resources + external secrets operator. Never commit to git.
- No `topologySpreadConstraints`: all replicas land on same node, single point of failure.
- `hostNetwork: true` or `hostPID: true` in production: breaks isolation, security risk.
- Missing `securityContext`: `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`.

## Helm Patterns

- Use `values.yaml` for defaults, override per environment: `helm install -f values-prod.yaml`.
- `_helpers.tpl` for shared labels and selectors — DRY across templates.
- `{{- include "mychart.labels" . | nindent 4 }}` for consistent labeling.
- Pin chart versions in `Chart.lock`. Run `helm dependency update` in CI.
- `helm template` + `kubeval`/`kubeconform` for pre-deploy validation.
- `helm diff upgrade` before `helm upgrade` to preview changes.
- Use `{{- toYaml .Values.resources | nindent 12 }}` to inject resource blocks from values.
