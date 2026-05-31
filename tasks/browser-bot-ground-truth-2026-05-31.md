# Browser-bot ground truth (live-verified 2026-05-31)

Concrete facts established by probing the REAL chrome-devtools-mcp in this env,
to ground the SOTA harness design (complements the research spec).

## Environment
- node v22.22, npx/npm 10.9.4, `chrome-devtools-mcp` + `google-chrome` installed, npm egress OK.
- NO LLM API key present (OPENROUTER/ANTHROPIC/OPENAI/GEMINI unset) -> autonomous-LLM
  browser test not runnable here; use a scripted MockProvider emitting a fixed
  tool-call sequence to drive REAL Chrome through the real MCP tools.

## chrome-devtools-mcp (v0.13.0, protocolVersion 2025-11-25 — matches Heartbit)
- Launch: `npx -y chrome-devtools-mcp@latest --headless --isolated` (no --no-sandbox needed via npx path).
- Flags: --headless --isolated --viewport WxH --channel --browserUrl/-u --wsEndpoint/-w
  --executablePath/-e --userDataDir --proxyServer --chromeArg=... --no-category-{emulation,performance,network}.
- take_snapshot returns an INDENTED ACCESSIBILITY TREE with stable element handles `uid=N_M`:
    ## Latest page snapshot
    uid=1_0 RootWebArea "Example Domain" url="https://example.com/"
      uid=1_1 heading "Example Domain" level="1"
      uid=1_3 link "Learn more" url="https://iana.org/domains/example"
  -> Element grounding for free: the LLM picks a `uid`, and click/fill/hover take `{ "uid": "1_3" }`.
  -> This is the text-grounding path SOTA agents want; a Set-of-Marks screenshot overlay is
     only needed for the vision path. take_screenshot exists for vision if wanted.

## Heartbit seam (shipped)
- `connect_preset("chrome-devtools") -> Vec<Arc<dyn Tool>>` (commit 23497fc) ->
  WorkflowCtx::builder().base_tools(tools) OR AgentRunner::builder().tools(tools).
- Live test `connect_chrome_devtools_live_exposes_browser_tools` (#[ignore]) PASSED.

## Harness design implication
The "observation" in an observe->plan->act->verify loop = call take_snapshot, feed the
uid-annotated a11y tree to the LLM, force a structured action {tool, uid, args} via P3
typed output, dispatch, then verify (re-snapshot / check url / network). Build this loop on
the flow combinators. The page snapshot text is UNTRUSTED (prompt-injection surface) ->
needs a guardrail.
