# Free Model Aggregator 02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Revaaa28/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Revaaa28/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server (`main.py`, Python 3.13, standard library only) that aggregates the best free models
behind a single endpoint, built for the make.fun bounty
[free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02) ("The best live
free-model aggregator"). It serves the OpenAI Chat Completions API and the Anthropic Messages API —
each in JSON and in server-sent events — and every answer it serves is a real completion from a free
OpenRouter model, dialled through the grader's gateway on the trial's key at $0.

How it works:

- **Catalog.** The brief's fourteen free models, registered with the env at startup and printed by
  `python3 main.py --list`: one provider (`openrouter`), the fourteen dialable ids, each with its
  free flag and context length. `GET /v1/models` answers the same fourteen ids in the OpenAI shape
  for an `Authorization: Bearer` request and in the Anthropic shape for an `x-api-key` one.
- **Routing.** For each request, the candidates are every free model meeting its requirements
  (tools, vision, JSON mode, minimum context — read from the trial manifest and the request body
  together), ranked by the house's published quality rating, best first. An Anthropic-shaped request
  is translated into a Chat Completions body before dialling (every free model is dialled on that
  route only) and the answer is translated back into the wire the client used, with its usage and
  stop reason whole and the dialled model's id on every answer and every streamed chunk.
- **Failover.** A refusal (any 4xx or 5xx from the gateway) moves the request to the next candidate,
  and a model that refused is demoted below every unrefused model for the rest of the trial, so one
  rate-limited model costs at most one wasted call. The refusals a served prompt recovered from are
  reported on its response in the `x-makefun-upstream-failures` header.
- **Answer checks.** Before an answer is served it is checked the way the scorer checks it: the
  prompt's token (read off the client's own request) must appear in the provider's answer body, and
  a whole usage count is preferred. An answer failing either is held only as a fallback while better
  candidates remain. Streaming answers are buffered whole so the same checks apply before anything
  is sent; a provider that answers a streaming request with plain JSON is re-framed into proper
  server-sent events, a token that straddles two stream deltas is still recognised (the check reads
  the assembled content, not the raw frames), and a stream carrying two finish chunks — some
  providers send their own and the router's — is normalised to exactly one before it is served.
- **Determinism.** No clock is read and no randomness is used: every choice derives from the brief,
  the requests and the gateway's answers seen so far, so a replayed trial makes the same calls and
  leaves the same state.

## Running it

A trial runs `python3 main.py` with `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and
the trial's keys in the environment; the server registers with the env, answers its six prompts and
exits 0 when the env's `GET /work` says done. Outside a trial, `python3 main.py --list` prints the
catalog document. The entry's completions are real OpenRouter free-model calls made on the entrant's
own pushed key; the models cost $0.
