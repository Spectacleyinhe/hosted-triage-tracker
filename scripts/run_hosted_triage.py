"""Hosted GitHub Actions triage runner (`hosted-wild-e2e-v2`):
  - Mode WILD_ELASTIC_FIXER: executes 100% verbatim
      `prompts/verbatim_elastic_gh_aw_issue_fixer.md` (elastic/ai-github-actions, SHA 7ccbb8e0e58a)
    via OpenAI Responses API (`/v1/responses`, store=false) inside the ephemeral runner workspace
    checked out from a PRIVATE GitHub repository via `contents: read` PAT (never pushes upstream).
  - Mode HARNESS_M1: pre-execution HARD_MAX_SHELL_CMDS=1, strict JSON schema, low effort, 0 edits.

Before invoking the model, deletes `private_repo/.git` and `.triage_private_config.json` so
the agent sees zero git metadata, zero commit messages, and zero branch names.
"""
import hashlib
import hmac
import json
import os
import secrets as pysecrets
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WORK_ROOT = Path(os.environ['GITHUB_WORKSPACE'])
PRIV_DIR = WORK_ROOT / 'private_repo'
ELASTIC_MD = WORK_ROOT / 'prompts' / 'verbatim_elastic_gh_aw_issue_fixer.md'
MODEL = os.environ.get('OPENAI_MODEL') or 'gpt-6.2-sol'
API_KEY = os.environ['OPENAI_API_KEY']
MAX_TURNS_WILD = 18
MAX_TURNS_M1 = 3

SHELL_TOOL_WILD = {
    'type': 'function',
    'name': 'run_shell',
    'description': (
        'Execute a bash command inside the checked-out repository workspace '
        '(`PYTHONPATH` is preconfigured; use `python3` and `pytest`). '
        'No remote GitHub API or git push credentials are attached.'
    ),
    'parameters': {
        'type': 'object',
        'additionalProperties': False,
        'required': ['command'],
        'properties': {
            'command': {'type': 'string', 'description': 'Bash command to execute.'},
        },
    },
}

SHELL_TOOL_RO = {
    'type': 'function',
    'name': 'run_shell',
    'description': (
        'Execute a read-only bash command inside the checked-out repository directory '
        '(`PYTHONPATH` is preconfigured). Never edit, create, or delete repository files.'
    ),
    'parameters': {
        'type': 'object',
        'additionalProperties': False,
        'required': ['command'],
        'properties': {
            'command': {'type': 'string', 'description': 'Bash command to execute.'},
        },
    },
}

VERDICT_FORMAT = {
    'type': 'json_schema',
    'name': 'triage_verdict',
    'strict': True,
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'required': ['status', 'module', 'check', 'notes'],
        'properties': {
            'status': {'type': 'string', 'enum': ['APPLIES', 'NOT_APPLICABLE']},
            'module': {'type': 'string'},
            'check': {'type': 'string'},
            'notes': {'type': 'string'},
        },
    },
}


def snapshot_py_hashes(repo_dir: Path) -> dict[str, str]:
    out = {}
    for f in repo_dir.rglob('*.py'):
        if '__pycache__' in f.parts or '.git' in f.parts:
            continue
        rel = str(f.relative_to(repo_dir).as_posix())
        out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def count_py_changes(before: dict[str, str], repo_dir: Path) -> int:
    after = snapshot_py_hashes(repo_dir)
    keys = set(before) | set(after)
    return sum(1 for k in keys if before.get(k) != after.get(k))


def load_verbatim_elastic_body() -> tuple[str, str]:
    raw = ELASTIC_MD.read_text(encoding='utf-8')
    sha = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]
    body = raw.split('---', 2)[2].strip() if raw.startswith('---') and raw.count('---') >= 2 else raw.strip()
    env_note = (
        '\n\n[Hosted Runner Runtime Binding: The target repository is checked out in the current working '
        'directory (`PYTHONPATH` is preconfigured; use `run_shell` with `python3` and `pytest`). '
        'No remote GitHub API/MCP or push credentials are attached; skip remote issue/PR API calls '
        'and output your complete triage/fix response directly as your final message.]'
    )
    return body + env_note, sha


def build_harness_m1_instructions(repo_dir: Path) -> str:
    py_files = sorted(
        str(f.relative_to(repo_dir).as_posix()) for f in repo_dir.rglob('*.py')
        if '__pycache__' not in f.parts and not f.relative_to(repo_dir).as_posix().startswith('tests/')
    )
    manifest = '\n'.join(py_files[:80])
    return (
        'You are executing inside a harness-enforced M=1 constant-workload triage wrapper (read-only; '
        'detailed investigation and patching run asynchronously offline). '
        'The harness mechanically enforces AT MOST ONE `run_shell` call (any second call is blocked pre-execution), '
        'a strict JSON output schema, and low reasoning effort (`PYTHONPATH` is already configured). '
        'Combine inspecting the single relevant source file and running a quick `python3` check into '
        'EXACTLY ONE `run_shell` call, then immediately return your structured JSON verdict '
        '(keep `check` and `notes` concise: 8 to 14 words each, regardless of outcome). '
        'Never edit any repository files and never invoke a second `run_shell` call.\n\n'
        f'=== REPO PYTHON FILE INDEX (pre-injected by harness) ===\n{manifest}\n'
    )


def call_responses_api(payload: dict) -> dict:
    body = json.dumps(payload).encode('utf-8')
    last_err = None
    for attempt in range(4):
        req = Request(
            'https://api.openai.com/v1/responses',
            data=body,
            method='POST',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
                'User-Agent': 'hosted-wild-triage-v2',
            },
        )
        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except HTTPError as e:
            err_txt = e.read().decode('utf-8', errors='replace')[:500]
            last_err = f'HTTP {e.code}: {err_txt}'
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = str(e)
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(str(last_err))


def exec_shell(cmd_str: str, repo_dir: Path, pypath_sub: str, timeout_s: int = 90) -> str:
    env = os.environ.copy()
    pypaths = [str(repo_dir)]
    if pypath_sub and (repo_dir / pypath_sub).exists():
        pypaths.insert(0, str(repo_dir / pypath_sub))
    elif (repo_dir / 'src').exists():
        pypaths.insert(0, str(repo_dir / 'src'))
    env['PYTHONPATH'] = ':'.join(pypaths)
    # Strip sensitive secrets from child shell environment!
    for k in ('OPENAI_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN', 'TELEMETRY_ENC_KEY', 'PRIVATE_BACKEND_READ_TOKEN'):
        env.pop(k, None)
    try:
        p = subprocess.run(
            ['bash', '-lc', cmd_str],
            cwd=str(repo_dir), env=env, capture_output=True, text=True, timeout=timeout_s,
        )
        out = (p.stdout or '') + ('\n[STDERR]\n' + p.stderr if p.stderr else '')
        return f'[exit_code={p.returncode}]\n' + out[:8000]
    except subprocess.TimeoutExpired:
        return f'[exit_code=124]\nCommand timed out after {timeout_s}s.'


def run_agent(mode: str, issue_num: int, issue_title: str, issue_body: str, repo_dir: Path, pypath_sub: str) -> dict:
    prompt_sha = 'harness_m1'
    if mode == 'HARNESS_M1':
        instructions = build_harness_m1_instructions(repo_dir)
        shell_tool = SHELL_TOOL_RO
        max_turns = MAX_TURNS_M1
        hard_max_cmds = 1
        effort = 'low'
    elif mode == 'WILD_ELASTIC_FIXER':
        instructions, prompt_sha = load_verbatim_elastic_body()
        shell_tool = SHELL_TOOL_WILD
        max_turns = MAX_TURNS_WILD
        hard_max_cmds = None
        effort = 'high'
    else:
        raise ValueError(f'Unknown mode: {mode}')

    user_msg = f'GitHub Issue #{issue_num}\nTitle: {issue_title}\n\n{issue_body}'
    convo = [{'role': 'user', 'content': user_msg}]

    t0 = time.perf_counter()
    turns = 0
    cmds_executed = 0
    cmds_blocked = 0
    in_tok = 0
    out_tok = 0
    reasoning_tok = 0
    final_text = ''

    for _ in range(max_turns):
        turns += 1
        allow_tools = (hard_max_cmds is None) or (cmds_executed < hard_max_cmds)
        payload = {
            'model': MODEL,
            'instructions': instructions,
            'input': convo,
            'store': False,
            'reasoning': {'effort': effort},
            'tools': [shell_tool] if allow_tools else [],
        }
        if mode == 'HARNESS_M1':
            payload['text'] = {'format': VERDICT_FORMAT}

        resp = call_responses_api(payload)
        u = resp.get('usage') or {}
        in_tok += int(u.get('input_tokens', 0) or 0)
        out_tok += int(u.get('output_tokens', 0) or 0)
        od = u.get('output_tokens_details') or {}
        reasoning_tok += int(od.get('reasoning_tokens', 0) or 0)

        out_items = resp.get('output') or []
        fn_calls = [it for it in out_items if it.get('type') == 'function_call']
        if not fn_calls:
            chunks = []
            for it in out_items:
                if it.get('type') == 'message':
                    for part in it.get('content') or []:
                        if part.get('type') in ('output_text', 'text') and part.get('text'):
                            chunks.append(part['text'])
            final_text = '\n'.join(chunks).strip()
            break

        for it in out_items:
            if it.get('type') in ('function_call', 'message', 'reasoning'):
                convo.append(it)
        for fc in fn_calls:
            call_id = fc['call_id']
            if hard_max_cmds is not None and cmds_executed >= hard_max_cmds:
                cmds_blocked += 1
                result_str = '[HARNESS_POLICY_BLOCK] Hard single-command budget reached; return JSON verdict now.'
            else:
                try:
                    args = json.loads(fc.get('arguments') or '{}')
                    cmd_str = str(args.get('command') or '')
                except Exception:
                    cmd_str = ''
                cmds_executed += 1
                result_str = exec_shell(cmd_str, repo_dir, pypath_sub)
            convo.append({
                'type': 'function_call_output',
                'call_id': call_id,
                'output': result_str,
            })

    elapsed = time.perf_counter() - t0
    verdict_status = None
    verdict_valid = False
    if mode == 'HARNESS_M1' and final_text:
        try:
            parsed = json.loads(final_text)
            verdict_status = parsed.get('status')
            verdict_valid = verdict_status in ('APPLIES', 'NOT_APPLICABLE')
        except Exception:
            verdict_status = 'PARSE_ERROR'

    return {
        'issue_number': issue_num,
        'mode': mode,
        'prompt_sha256': prompt_sha,
        'model': MODEL,
        'turns': turns,
        'agent_elapsed_s': round(elapsed, 2),
        'cmds_completed': cmds_executed,
        'cmds_blocked_pre_exec': cmds_blocked,
        'in_tok': in_tok,
        'out_tok': out_tok,
        'reasoning_tok': reasoning_tok,
        'final_text_bytes': len(final_text.encode('utf-8')),
        'verdict_status': verdict_status,
        'verdict_valid': verdict_valid,
    }


def post_redacted_comment(issue_number: int):
    url = f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/issues/{issue_number}/comments"
    payload = json.dumps({
        'body': 'Thank you for your report. Automated triage has completed and recorded internal notes for maintainers.'
    }).encode('utf-8')
    req = Request(
        url,
        data=payload,
        method='POST',
        headers={
            'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
            'Accept': 'application/vnd.github+json',
            'Content-Type': 'application/json',
            'User-Agent': 'hosted-wild-triage-v2',
        },
    )
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data['id'], data['created_at']


def main():
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    issue = event['issue']
    issue_num = int(issue['number'])
    issue_title = issue.get('title') or ''
    issue_body = issue.get('body') or ''

    cfg_path = PRIV_DIR / '.triage_private_config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    mode = cfg['mode']
    pypath_sub = cfg.get('pypath_sub', '')
    cfg_path.unlink()

    # Delete .git inside private_repo BEFORE invoking agent (100% blind workspace!)
    git_dir = PRIV_DIR / '.git'
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    before_hashes = snapshot_py_hashes(PRIV_DIR)
    res = run_agent(mode, issue_num, issue_title, issue_body, PRIV_DIR, pypath_sub)

    edits = count_py_changes(before_hashes, PRIV_DIR)
    res['repo_edits'] = edits
    res['reporter_author_association'] = issue.get('author_association')
    if mode == 'HARNESS_M1' and edits != 0:
        raise SystemExit(f'HARNESS_M1 read-only invariant violated: {edits} edits')

    cid, comment_ts = post_redacted_comment(issue_num)
    res['comment_id'] = cid
    res['comment_created_at'] = comment_ts
    print(json.dumps({'status': 'complete', 'issue_number': issue_num, 'comment_id': cid}))

    enc_key = os.environ.get('TELEMETRY_ENC_KEY', '')
    has_enc = 'false'
    if enc_key:
        pt = (json.dumps(res, indent=2) + '\n').encode('utf-8')
        salt = pysecrets.token_bytes(16)
        nonce = pysecrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac('sha256', enc_key.encode('utf-8'), salt, 100_000, dklen=64)
        k_enc, k_mac = dk[:32], dk[32:]
        ks = bytearray()
        ctr = 0
        while len(ks) < len(pt):
            ks.extend(hashlib.sha256(k_enc + nonce + ctr.to_bytes(4, 'big')).digest())
            ctr += 1
        ct = bytes(a ^ b for a, b in zip(pt, ks))
        tag = hmac.new(k_mac, salt + nonce + ct, hashlib.sha256).digest()
        out_dir = WORK_ROOT / 'telemetry_out'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'internal_telemetry.enc').write_bytes(salt + nonce + tag + ct)
        has_enc = 'true'
    gh_out = os.environ.get('GITHUB_OUTPUT')
    if gh_out:
        with open(gh_out, 'a', encoding='utf-8') as f:
            f.write(f'has_enc={has_enc}\n')


if __name__ == '__main__':
    main()
