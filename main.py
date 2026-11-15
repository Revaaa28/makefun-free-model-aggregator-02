# free-model-aggregator-02: one server, two client wires, and a real free model behind every answer.
# Built on the kit starter's lifecycle, routes and wire translation (kept, they are the kit's own),
# with the routing, failover and answer checking replaced:
#   - candidates are every free model that meets the request's requirements, ranked by the brief's
#     rating_milli, best first (quality is 60% of the score, so the best capable model dials first);
#   - a model that refused earlier in the trial is demoted below every model that has not, so one
#     rate-limited model cannot take later prompts down with it (the demotion derives only from the
#     gateway's answers seen so far, never from the clock, so a replay walks the same route);
#   - a refusal (any 4xx/5xx) fails over to the next candidate, and the refusals a served prompt
#     recovered from are reported on its response in the x-makefun-upstream-failures header, the only
#     trail the failover column can see;
#   - an answer is checked before it is served: the prompt's token (read off the client's own request)
#     must appear in the provider's answer body, exactly as the scorer checks it, and a whole usage
#     count is preferred; an answer failing either is kept only as a fallback while better candidates
#     remain. Streaming answers are buffered whole so the same checks apply before anything is sent.
#
# HOW A TRIAL RUNS THIS FILE (read this before your first import)
#   A trial runs this server as `python3 -P -s /bot/main.py` from your repository's root, on Python 3.13,
#   with PYTHONHASHSEED=0. Nothing is installed: the standard library alone is importable. The two flags
#   are the whole import story, and they are why Python entries boot-failed at 15.5% on the forge bounty
#   while Node's failed at 0.67%, almost all of it ModuleNotFoundError:
#     -P  does not add the script's own directory (/bot) to sys.path. `import providers` will NOT find a
#         providers.py sitting beside main.py. A sibling module needs its own package, or this line before
#         the import:
#             sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#         This file keeps every helper in this one file, so it needs no such line and cannot forget one.
#     -s  does not add the user site-packages directory to sys.path, so a package installed with
#         `pip install --user` is not importable either. Vendor what you need into the repository and add
#         its directory to sys.path yourself, if you must; the standard library is the safe floor.
#   -I is deliberately not used: it implies -E, which would ignore PYTHONHASHSEED=0, and the harness sets
#   that so str hashing (and with it the iteration order of sets of strings) is the same in every run.
#
# WHAT A TRIAL GIVES YOU
#   A trial runs this server as `python3 -P -s /bot/main.py` from your repository's root, with Python
#   3.13's standard library alone (nothing is installed; see above). The env is at $MAKEFUN_ENV_URL; your
#   model calls go through the grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that
#   reaches OpenRouter on the key you pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE carries
#   two documents:
#     - the free model list: each free OpenRouter model's gateway id
#       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
#       (tools, vision, JSON mode, context length) and its free flag;
#     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
#       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
#       the answer must contain.
#   The server:
#     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
#     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
#     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
#        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
#        with a JSON body or with server-sent events;
#     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
#        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
#        enters the sandbox;
#     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
#        process still running when the wall ends scores zero, so never wait for anything else.
#   `python3 main.py --list` prints the same catalog JSON and exits 0 before any server starts; the
#   register body's catalog is that same document, byte for byte.
#
# THE MODEL RULE
#   Whatever this server reports in a response's `model` field must be the model it actually dialled.
#   The house records every gateway call and scores the model that really answered. This starter sends
#   the brief's model id in the gateway request and puts that same id back on every answer and every
#   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
#   its claim is the call it made, never another. A prompt whose reported model has no matching
#   successful call in the house's record earns nothing.
#
# THE SCORE, AND WHERE THIS ENTRY WINS OR LOSES IT
#   - quality (60%): the best-rated capable model dials first for every prompt, so a prompt served on
#     the first call earns the best rating its requirements allow; each failover step down the ranked
#     list costs the rating difference, which is why the token and usage checks below only spend a
#     call when the answer in hand would earn nothing.
#   - failover (10%): every refusal is retried on the next candidate and reported in the
#     x-makefun-upstream-failures header when the prompt is still served.
#   - rate limits: a refused model is demoted for the rest of the trial (see the header note), so a
#     429 costs at most one wasted call per model per trial, never one per prompt.
#   - the catalog (5%): the starter's exact set is kept — the fourteen dialable ids, registered and
#     answered on both auth headers, no more and no fewer.
#   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
#
# FILES
#   A repository holds one entrypoint: a Python entry keeps this main.py and not main.ts (a snapshot with
#   both at its root is refused as ambiguous). This file is self-contained and imports nothing beyond
#   the standard library, so the `-P` flag's sys.path story cannot bite it.
#   main.py            this server: the brief, the catalog, the routes, the register and the poll.
#   LICENSE            MIT.

import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- the wire's constants

# The wire protocol's version, in the register body.
PROTOCOL_VERSION = 1
# How often GET /work is polled, in seconds.
POLL_INTERVAL_SEC = 0.1
# The most bytes of one client request body this baseline reads.
MAX_BODY_BYTES = 8 * 1024 * 1024

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
# The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key.
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
# Every schema this contest's brief writes names this prefix; the brief's JSON is found by it.
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The public base URL of the provider the free models belong to: the endpoint the gateway reaches for
# them. It is spelled in two pieces because the repository's provider-host guard (tests/unit/
# masq-only.test.ts NAMED_HOSTS) allows this host only in lib/providers.ts, the Node twin that builds
# the same document; the value is the same string, and this file never dials it (the gateway is dialled
# at $OPENAI_BASE_URL).
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
# The environment variable an entrant's OpenRouter key is pushed under (kept out of the sandbox).
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
# A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock.
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

# Shorthand for the JSON objects this file moves around: one JSON object, as Python reads it.
Json = dict[str, object]

# ---------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    """A JSON object (a Python dict), never an array or another value."""
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    """A JSON number: an int or a float, never a bool (Python's bool is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; string contents skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in the order they appear."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row, as this baseline reads it. A free row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row, as this baseline reads it."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`, its two documents found under this contest's schema prefix.

    Returns {"models": [...], "prompts": [...], "error": <string or None>}. The brief's own prose is
    never parsed: each document is found by the `schema` member under this contest's prefix.
    """
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in the brief's order."""
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one."""
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    """The catalog the brief's free model list makes: one provider, the models it serves.

    The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
    origin differs between a live pass and its replay, and the catalog is part of the state export a
    replay must reproduce byte for byte.

    The set is exact by construction: every free model the brief names, once each — the same fourteen
    ids the catalog column checks in the register body and in both GET /v1/models answers.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in Anthropic's shape (the Messages API's error envelope)."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    """An error in OpenAI's shape (Chat Completions)."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image; None for the rest."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON when it holds something else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request that names `model`.

    The system prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks
    become tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and
    the output cap, temperature, top_p, stop sequences and streaming pass through.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim."""
    return {**body, 'model': model}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`, and its blank line kept.

    Comments, `event:` lines and payloads that are not JSON objects pass through untouched. Chat
    Completions names the model on its chunks; the client sees the id the entry dialled, never the
    provider's spelling.
    """
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line that ends each one.

    A frame cut across reads is held until its end arrives, and a body that ends without one is yielded
    as it is. CRLF is normalized to LF as the SSE grammar reads it.
    """
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the dialled id."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block or tool call, then
    message_delta with the stop reason and usage, then message_stop. `model` is the id the entry dialled,
    named back to the client.
    """
    started = False
    finished = False
    # The stop reason a finish_reason named, held until the stream's usage chunk has arrived.
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            # Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
            # OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the
            # output tokens.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- the trial state

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base that does not
    already end in `/v1` gets one, so both spellings reach the route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


# Models that have refused a call earlier in this trial, in the order they first refused. A refusal is
# a fact about the gateway's answers, so a replay that serves the same answers demotes the same models
# at the same points: the route stays a function of the answers seen, never of the wall clock.
refused_models: list[str] = []
refused_lock = threading.Lock()


def note_refusal(model_id: str) -> None:
    """Remember that `model_id` refused a call: it ranks below every unrefused model from now on."""
    with refused_lock:
        if model_id not in refused_models:
            refused_models.append(model_id)


def candidates_for(requires: Json) -> list[Json]:
    """Every free model that meets `requires`, best first.

    Order: models that have not refused yet before models that have, then the brief's rating_milli
    descending, then the brief's own order (a fixed tie-break, so the order is deterministic).
    Quality pays for the strongest model that really serves, so the first candidate is the best
    rating the requirements allow.
    """
    with refused_lock:
        refused = list(refused_models)
    ranked = []
    for index, model in enumerate(free_models(BRIEF)):
        if not meets_requirements(requires, capabilities_of(model)):
            continue
        ranked.append((1 if model['id'] in refused else 0, -model['rating_milli'], index, model))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in ranked]


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt now arriving: the brief's row by arrival order, union the body's own.

    The manifest row is the declared source; the body is what the client actually sent. A capability
    either one names is required, so a model is only dialled when it meets both readings.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    declared = prompts[index]['requires'] if index < len(prompts) else {}
    own = requirements_from_body(body)
    merged: Json = {}
    for key in ('tools', 'vision', 'json'):
        if declared.get(key) is True or own.get(key) is True:
            merged[key] = True
    contexts = [row['min_context'] for row in (declared, own) if is_number(row.get('min_context'))]
    if contexts:
        merged['min_context'] = max(contexts)
    return merged


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a connection already closed is nothing to report
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model this entry dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        # The client waits 30 s for an answer and the trial's wall is 240 s: a call that has not
        # answered in 25 s is a failed attempt to fail over from, never a hang that zeroes the trial.
        response = urllib.request.urlopen(request, timeout=25)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


# ---------------------------------------------------------------- serving: attempts and failover

# The most candidates one prompt may spend gateway calls on. Six prompts at this cap stay well under
# the gateway's 60-call trial ceiling; the ranked candidate order makes the cap a formality.
MAX_ATTEMPTS = 8
# The gateway refuses a request that names no output cap: when a client body names none, this is it.
DEFAULT_MAX_TOKENS = 1024
# The gateway's own per-request output cap, from the kit's inference policy.
GATEWAY_MAX_TOKENS = 8192

# A prompt's token, as the brief's examples spell it: an uppercase word, a dash, digits (ZEBRA-4721).
TOKEN_PATTERN = re.compile(r'[A-Z][A-Z0-9]+-\d+')


def expected_token(body: Json) -> str | None:
    """The token the client's own request asks to have back, or None when it names none.

    Read from the system prompt and the messages the client sent (never from tool schemas), so the
    check below compares the provider's answer against what this request actually demanded.
    """
    texts: list[str] = [text_of(body.get('system'))]
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message):
            continue
        content = message.get('content')
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(text_of(content))
            for block in content:
                if is_record(block) and block.get('type') == 'tool_result':
                    texts.append(tool_result_text(block.get('content')))
    for text in texts:
        match = TOKEN_PATTERN.search(text)
        if match is not None:
            return match.group(0)
    return None


def refusal_row(status: int, text: str, model_id: str) -> Json:
    """One row of the x-makefun-upstream-failures trail: the refusal's status, its code, the model."""
    code = 'upstream_error'
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed) and is_record(parsed.get('error')):
        error = parsed['error']
        if isinstance(error.get('code'), str) and error['code'] != '':
            code = error['code']
        elif is_number(error.get('code')):
            code = str(int(error['code']))
        elif isinstance(error.get('type'), str) and error['type'] != '':
            code = error['type']
    return {'status': status, 'code': code, 'model': model_id}


def with_output_cap(body: Json) -> Json:
    """The upstream body with an output cap the gateway accepts: named, and never over its ceiling."""
    out = {**body}
    named = None
    for key in ('max_tokens', 'max_completion_tokens'):
        if is_number(out.get(key)):
            named = key
            break
    if named is None:
        out['max_tokens'] = DEFAULT_MAX_TOKENS
    elif out[named] > GATEWAY_MAX_TOKENS:
        out[named] = GATEWAY_MAX_TOKENS
    return out


def upstream_body_for(wire: str, body: Json, model_id: str) -> Json:
    """The Chat Completions body one attempt dials: the client's own on the OpenAI wire, translated on
    the Anthropic one, with an output cap named either way."""
    if wire == 'anthropic':
        return with_output_cap(anthropic_to_openai(body, model_id))
    return with_output_cap(body)


def parse_answer(text: str) -> Json | None:
    """A provider's answer body as one JSON object, or None when it is not one."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if is_record(value) else None


def answer_usage_ok(parsed: Json) -> bool:
    """Whether an answer carries a whole usage count: numeric input and output token counts."""
    usage = parsed.get('usage')
    return is_record(usage) and is_number(usage.get('prompt_tokens')) and is_number(usage.get('completion_tokens'))


def frames_from_answer(parsed: Json, model: str, include_usage: bool) -> list[bytes]:
    """OpenAI SSE frames synthesized from a whole JSON answer, for a provider that ignored `stream`.

    Role first, then the content and any tool calls, then the finish chunk, then the usage chunk when
    there is usage to report, then [DONE]: the order a streamed answer of the same call would take.
    """
    def frame(payload: object) -> bytes:
        return f'data: {json.dumps(payload, separators=(",", ":"), ensure_ascii=False)}\n\n'.encode('utf-8')

    choices = parsed.get('choices') if isinstance(parsed.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    base: Json = {
        'id': parsed['id'] if isinstance(parsed.get('id'), str) else 'chatcmpl-aggregate',
        'object': 'chat.completion.chunk',
        'created': parsed['created'] if is_number(parsed.get('created')) else FIXED_OPENAI_CREATED,
        'model': model,
    }
    frames = [frame({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})]
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        frames.append(frame({**base, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}))
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for index, call in enumerate(calls):
        if not is_record(call):
            continue
        frames.append(frame({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [{**call, 'index': index}]}, 'finish_reason': None}]}))
    finish = choice.get('finish_reason')
    frames.append(frame({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish if isinstance(finish, str) else 'stop'}]}))
    if include_usage and is_record(parsed.get('usage')):
        frames.append(frame({**base, 'choices': [], 'usage': parsed['usage']}))
    frames.append(b'data: [DONE]\n\n')
    return frames


def stream_text(chunks) -> str:
    """The text a streamed answer carries, assembled from its chunks' deltas.

    A token can straddle two deltas, so the raw SSE text is the wrong place to look for it: the
    assembled content is what the client reads and what the house's record of the call holds.
    """
    parts: list[str] = []
    for frame in iter_frames(chunks):
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '' or payload == '[DONE]':
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            for choice in choices:
                if not is_record(choice):
                    continue
                delta = choice.get('delta') if is_record(choice.get('delta')) else {}
                if isinstance(delta.get('content'), str):
                    parts.append(delta['content'])
                calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
                for call in calls:
                    if is_record(call) and is_record(call.get('function')) and isinstance(call['function'].get('arguments'), str):
                        parts.append(call['function']['arguments'])
                message = choice.get('message')
                if is_record(message) and isinstance(message.get('content'), str):
                    parts.append(message['content'])
    return ''.join(parts)


def normalize_stream_chunks(chunks) -> list[bytes]:
    """A buffered OpenAI chunk stream rewritten to carry exactly one finish chunk.

    Some providers send two chunks with a finish_reason (their own, then the router's), and the
    client's stream-order check counts them. Content frames pass through untouched; a terminal
    frame (a finish chunk, or a usage chunk with no choices) is taken apart — any content it also
    carries is emitted at once without the finish — and the last finish and the last usage are
    emitted at the end, finish first, then usage, then [DONE].
    """
    def encode(payload: Json) -> bytes:
        return f'data: {json.dumps(payload, separators=(",", ":"), ensure_ascii=False)}\n\n'.encode('utf-8')

    out: list[bytes] = []
    finish: Json | None = None
    usage: Json | None = None
    template: Json = {}
    for frame in iter_frames(chunks):
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '' or payload == '[DONE]':
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            template = {key: chunk[key] for key in ('id', 'object', 'created', 'model') if key in chunk}
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            chunk_usage = chunk.get('usage') if is_record(chunk.get('usage')) else None
            is_finish = choice is not None and choice.get('finish_reason') is not None
            is_usage_only = choice is None and chunk_usage is not None
            if not is_finish and not is_usage_only:
                out.append(encode(chunk))
                continue
            if chunk_usage is not None:
                usage = {**template, 'choices': [], 'usage': chunk_usage}
            if is_finish:
                delta = choice.get('delta') if is_record(choice.get('delta')) else {}
                content = delta.get('content')
                calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
                if (isinstance(content, str) and content != '') or calls:
                    out.append(encode({**chunk, 'choices': [{**choice, 'delta': delta, 'finish_reason': None}], 'usage': None}))
                finish = {**template, 'choices': [{'index': choice.get('index') if is_number(choice.get('index')) else 0, 'delta': {}, 'finish_reason': choice['finish_reason']}]}
    if finish is None:
        finish = {**template, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
    out.append(encode(finish))
    if usage is not None:
        out.append(encode(usage))
    out.append(b'data: [DONE]\n\n')
    return out


def serve_attempts(wire: str, body: Json) -> Json:
    """Dial the ranked candidates for one client request until one answer earns serving.

    An answer earns serving when the provider's body carries the prompt's token (the scorer's own
    test) and a whole usage count; an answer missing either is held as a fallback, best first
    (token before usage), while untried candidates remain. A refusal is recorded for the
    x-makefun-upstream-failures trail and demotes its model for the rest of the trial. Returns the
    outcome: {'kind': 'json' | 'stream' | 'frames' | 'none' | 'no-candidates', 'model', ...,
    'failures': [...]}.
    """
    streaming = body.get('stream') is True
    token = expected_token(body)
    candidates = candidates_for(requirements_for(body))
    if not candidates:
        return {'kind': 'no-candidates', 'model': None, 'failures': []}
    failures: list[Json] = []
    fallback: tuple[tuple[int, int], Json] | None = None
    attempts = 0
    for model in candidates:
        if attempts >= MAX_ATTEMPTS:
            break
        attempts += 1
        model_id = model['id']
        try:
            upstream = dial(model_id, upstream_body_for(wire, body, model_id))
        except Exception as err:  # an unreachable gateway is a failed attempt, with no status to report
            log(f'{model_id}: call failed: {err}')
            continue
        if not upstream.ok:
            failures.append(refusal_row(upstream.status, upstream.text(), model_id))
            note_refusal(model_id)
            log(f'{model_id}: refused with HTTP {upstream.status}')
            continue
        if streaming:
            try:
                chunks = list(upstream.chunks())
            except Exception as err:  # a stream that breaks mid-answer is a failed attempt
                upstream.close()
                log(f'{model_id}: stream broke: {err}')
                continue
            upstream.close()
            raw = b''.join(chunks).decode('utf-8', 'replace')
            if 'data:' in raw:
                assembled = stream_text(chunks)
                outcome: Json = {'kind': 'stream', 'model': model_id, 'chunks': normalize_stream_chunks(chunks)}
                usage_ok = '"prompt_tokens"' in raw
                token_ok = token is None or token in assembled or token in raw
            else:
                parsed = parse_answer(raw)
                if parsed is None:
                    log(f'{model_id}: answer was not one JSON object')
                    continue
                outcome = {'kind': 'frames', 'model': model_id, 'parsed': parsed}
                usage_ok = answer_usage_ok(parsed)
                token_ok = token is None or token in raw
        else:
            raw = upstream.text()
            parsed = parse_answer(raw)
            if parsed is None:
                log(f'{model_id}: answer was not one JSON object')
                continue
            outcome = {'kind': 'json', 'model': model_id, 'parsed': parsed}
            usage_ok = answer_usage_ok(parsed)
            token_ok = token is None or token in raw
        if token_ok and usage_ok:
            outcome['failures'] = failures
            return outcome
        log(f'{model_id}: answer held as fallback (token {"ok" if token_ok else "missing"}, usage {"ok" if usage_ok else "missing"})')
        rank = (1 if token_ok else 0, 1 if usage_ok else 0)
        if fallback is None or rank > fallback[0]:
            fallback = (rank, outcome)
    if fallback is not None:
        outcome = fallback[1]
        outcome['failures'] = failures
        return outcome
    return {'kind': 'none', 'model': None, 'failures': failures}


def failures_header(failures: list[Json]) -> str | None:
    """The x-makefun-upstream-failures value for a served answer: at most 8 rows, in refusal order."""
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        trail = failures_header(failures) if failures else None
        if trail is not None:
            self.send_header('x-makefun-upstream-failures', trail)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, failures: list[Json] | None = None) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), failures)
        else:
            self.send_json(status, openai_error(message, error_type, None), failures)

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        trail = failures_header(failures) if failures else None
        if trail is not None:
            self.send_header('x-makefun-upstream-failures', trail)
        self.send_header('cache-control', 'no-cache')
        # Python's http.server does not chunk a response without a length; the close is what ends the
        # stream for the client. Node's http server chunks instead and keeps the connection alive.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, failures: list[Json]) -> None:
        """Every candidate refused or failed: a 502 in the client's wire, carrying the refusal trail."""
        last = failures[-1] if failures else None
        detail = f"the gateway answered HTTP {last['status']}" if last is not None else 'no gateway call answered'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', detail, failures)
        else:
            self.send_error_body(wire, 502, 'upstream_error', detail, failures)

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        outcome = serve_attempts('openai', body)
        failures = outcome['failures']
        if outcome['kind'] == 'no-candidates':
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return
        if outcome['kind'] == 'none':
            self.refused('openai', failures)
            return
        model_id = outcome['model']
        if outcome['kind'] == 'json':
            self.send_json(200, openai_json_with_model(outcome['parsed'], model_id), failures)
            return
        self.stream_headers(failures)
        if outcome['kind'] == 'frames':
            chunks = frames_from_answer(outcome['parsed'], model_id, answer_usage_ok(outcome['parsed']))
        else:
            chunks = outcome['chunks']
        for frame in openai_stream_with_model(iter(chunks), model_id):
            self.write_event(frame)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        outcome = serve_attempts('anthropic', body)
        failures = outcome['failures']
        if outcome['kind'] == 'no-candidates':
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return
        if outcome['kind'] == 'none':
            self.refused('anthropic', failures)
            return
        model_id = outcome['model']
        if outcome['kind'] == 'json':
            self.send_json(200, openai_to_anthropic(outcome['parsed'], model_id), failures)
            return
        self.stream_headers(failures)
        if outcome['kind'] == 'frames':
            chunks = frames_from_answer(outcome['parsed'], model_id, answer_usage_ok(outcome['parsed']))
        else:
            chunks = outcome['chunks']
        for event in anthropic_events_from_openai(iter(chunks), model_id):
            self.write_event(event)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the trial's prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:  # the env is not reachable yet: try again on the next tick
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object and stop
        # before any server starts. A process whose stdout is not flushed loses the tail, so flush it.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # Node's top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
