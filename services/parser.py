"""
Parser for Claude Code JSONL session files.

Reads ~/.claude/projects/{project-dir}/{session-id}.jsonl files and extracts
structured data: messages, tool calls, subagent activity, and token usage.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


CLAUDE_DIR = Path.home() / '.claude'
PROJECTS_DIR = CLAUDE_DIR / 'projects'


@dataclass
class ToolCall:
    tool_name: str
    tool_use_id: str
    input_params: dict = field(default_factory=dict)
    output_result: str = ''
    timestamp: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class Message:
    role: str  # 'user' | 'assistant'
    content_text: str = ''
    tool_calls: list = field(default_factory=list)  # list of ToolCall
    tool_results: list = field(default_factory=list)  # list of ToolCall (matched)
    timestamp: Optional[str] = None
    model: Optional[str] = None
    uuid: Optional[str] = None
    parent_uuid: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    has_thinking: bool = False
    thinking_text: str = ''


@dataclass
class SubAgent:
    agent_id: str
    prompt: str = ''
    messages: list = field(default_factory=list)  # list of Message
    model: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class SessionData:
    session_id: str
    slug: str = ''
    project_dir: str = ''
    messages: list = field(default_factory=list)  # list of Message
    subagents: list = field(default_factory=list)  # list of SubAgent
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model: Optional[str] = None
    cwd: str = ''
    git_branch: str = ''
    version: str = ''
    team_name: str = ''
    agent_name: str = ''
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0


_MAX_JSONL_SIZE = 500 * 1024 * 1024  # 500 MB
_MAX_TOOL_RESULT_SIZE = 50 * 1024 * 1024  # 50 MB


def parse_jsonl_file(filepath: str) -> list[dict]:
    """Parse a JSONL file into a list of raw JSON objects."""
    entries = []
    try:
        if os.path.getsize(filepath) > _MAX_JSONL_SIZE:
            return entries
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (OSError, IOError):
        pass
    return entries


def _extract_text_from_content(content) -> str:
    """Extract plain text from message content (string or content blocks array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    text = block.get('text', '')
                    if text.strip():
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(parts)
    return ''


def _extract_thinking_from_content(content) -> str:
    """Extract thinking text from content blocks."""
    if not isinstance(content, list):
        return ''
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'thinking':
            thinking = block.get('thinking', '')
            if thinking:
                parts.append(thinking)
    return '\n'.join(parts)


def _extract_tool_uses(content) -> list[ToolCall]:
    """Extract tool_use blocks from content."""
    if not isinstance(content, list):
        return []
    tools = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_use':
            tc = ToolCall(
                tool_name=block.get('name', 'unknown'),
                tool_use_id=block.get('id', ''),
                input_params=block.get('input', {}),
            )
            tools.append(tc)
    return tools


def _extract_tool_results(content) -> list[tuple]:
    """Extract tool_result blocks from content. Returns list of (tool_use_id, result_text)."""
    if not isinstance(content, list):
        return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_result':
            tool_use_id = block.get('tool_use_id', '')
            result_content = block.get('content', '')
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                parts = []
                for r in result_content:
                    if isinstance(r, dict):
                        parts.append(r.get('text', ''))
                    elif isinstance(r, str):
                        parts.append(r)
                result_text = '\n'.join(parts)
            else:
                result_text = str(result_content)
            results.append((tool_use_id, result_text))
    return results


def _read_tool_result_file(session_dir: str, tool_use_id: str) -> Optional[str]:
    """Try to read large tool result from external file."""
    if not tool_use_id or '/' in tool_use_id or '\\' in tool_use_id or '..' in tool_use_id:
        return None
    result_file = os.path.join(session_dir, 'tool-results', f'{tool_use_id}.txt')
    try:
        if os.path.getsize(result_file) > _MAX_TOOL_RESULT_SIZE:
            return None
        with open(result_file, 'r', encoding='utf-8') as f:
            return f.read()
    except (OSError, IOError):
        return None


def parse_session(project_dir: str, session_id: str) -> SessionData:
    """Parse a session JSONL file into structured SessionData."""
    jsonl_path = os.path.join(project_dir, f'{session_id}.jsonl')
    session_dir = os.path.join(project_dir, session_id)
    entries = parse_jsonl_file(jsonl_path)

    session = SessionData(session_id=session_id, project_dir=project_dir)

    # Maps tool_use_id -> ToolCall for linking results
    pending_tool_calls: dict[str, ToolCall] = {}
    # Collect messages (skip progress and file-history-snapshot for main timeline)
    messages: list[Message] = []
    # Track seen UUIDs to deduplicate streamed assistant messages
    seen_uuids: dict[str, int] = {}  # uuid -> index in messages list

    for entry in entries:
        entry_type = entry.get('type')
        timestamp = entry.get('timestamp')

        # Extract session metadata from any entry
        if not session.cwd and entry.get('cwd'):
            session.cwd = entry['cwd']
        if not session.git_branch and entry.get('gitBranch'):
            session.git_branch = entry['gitBranch']
        if not session.version and entry.get('version'):
            session.version = entry['version']
        if not session.slug and entry.get('slug'):
            session.slug = entry['slug']
        if not session.team_name and entry.get('teamName'):
            session.team_name = entry['teamName']
        if not session.agent_name and entry.get('agentName'):
            session.agent_name = entry['agentName']

        # Track time range
        if timestamp:
            if not session.start_time or timestamp < session.start_time:
                session.start_time = timestamp
            if not session.end_time or timestamp > session.end_time:
                session.end_time = timestamp

        if entry_type == 'file-history-snapshot':
            continue

        if entry_type == 'progress':
            # Progress entries are subagent streaming - skip for main timeline
            # but extract slug if present
            continue

        msg_data = entry.get('message', {})
        if not msg_data:
            continue

        role = msg_data.get('role', '')
        content = msg_data.get('content', [])
        uuid = entry.get('uuid')
        parent_uuid = entry.get('parentUuid')
        is_sidechain = entry.get('isSidechain', False)

        # Skip sidechain messages (subagent messages in main file)
        if is_sidechain:
            continue

        if entry_type == 'user' and role == 'user':
            # Check for tool_result content blocks
            tool_results_data = _extract_tool_results(content)

            if tool_results_data:
                # This is a tool result message - match results to pending tool calls
                for tool_use_id, result_text in tool_results_data:
                    # Try external file for large results
                    external = _read_tool_result_file(session_dir, tool_use_id)
                    if external is not None:
                        result_text = external

                    if tool_use_id in pending_tool_calls:
                        pending_tool_calls[tool_use_id].output_result = result_text
                        if timestamp:
                            pending_tool_calls[tool_use_id].duration_ms = _calc_duration(
                                pending_tool_calls[tool_use_id].timestamp, timestamp
                            )
            else:
                # Regular user message
                text = _extract_text_from_content(content)
                if text.strip():
                    msg = Message(
                        role='user',
                        content_text=text,
                        timestamp=timestamp,
                        uuid=uuid,
                        parent_uuid=parent_uuid,
                    )
                    messages.append(msg)

        elif entry_type == 'assistant' and role == 'assistant':
            model = msg_data.get('model')
            if model and not session.model:
                session.model = model

            usage = msg_data.get('usage', {})

            # Deduplicate streamed assistant messages (same uuid = same message, take latest)
            if uuid and uuid in seen_uuids:
                idx = seen_uuids[uuid]
                existing = messages[idx]
                # Merge: append new text, add new tool calls
                new_text = _extract_text_from_content(content)
                if new_text.strip() and new_text.strip() not in existing.content_text:
                    if existing.content_text:
                        existing.content_text += new_text
                    else:
                        existing.content_text = new_text

                thinking = _extract_thinking_from_content(content)
                if thinking:
                    existing.has_thinking = True
                    if thinking not in existing.thinking_text:
                        existing.thinking_text = thinking

                new_tools = _extract_tool_uses(content)
                existing_ids = {tc.tool_use_id for tc in existing.tool_calls}
                for tc in new_tools:
                    if tc.tool_use_id not in existing_ids:
                        tc.timestamp = timestamp
                        existing.tool_calls.append(tc)
                        pending_tool_calls[tc.tool_use_id] = tc

                # Update tokens to latest (cumulative from streaming)
                if usage:
                    existing.input_tokens = usage.get('input_tokens', existing.input_tokens)
                    existing.output_tokens = usage.get('output_tokens', existing.output_tokens)
                    existing.cache_read_tokens = usage.get('cache_read_input_tokens', existing.cache_read_tokens)
                    existing.cache_creation_tokens = usage.get('cache_creation_input_tokens', existing.cache_creation_tokens)

                existing.timestamp = timestamp  # update to latest
                continue

            text = _extract_text_from_content(content)
            thinking = _extract_thinking_from_content(content)
            tool_uses = _extract_tool_uses(content)

            msg = Message(
                role='assistant',
                content_text=text.strip(),
                timestamp=timestamp,
                model=model,
                uuid=uuid,
                parent_uuid=parent_uuid,
                has_thinking=bool(thinking),
                thinking_text=thinking,
            )

            if usage:
                msg.input_tokens = usage.get('input_tokens', 0)
                msg.output_tokens = usage.get('output_tokens', 0)
                msg.cache_read_tokens = usage.get('cache_read_input_tokens', 0)
                msg.cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)

            for tc in tool_uses:
                tc.timestamp = timestamp
                msg.tool_calls.append(tc)
                pending_tool_calls[tc.tool_use_id] = tc

            if uuid:
                seen_uuids[uuid] = len(messages)
            messages.append(msg)

    session.messages = messages

    # Compute token totals
    for msg in messages:
        session.total_input_tokens += msg.input_tokens
        session.total_output_tokens += msg.output_tokens
        session.total_cache_read_tokens += msg.cache_read_tokens
        session.total_cache_creation_tokens += msg.cache_creation_tokens

    # Parse subagents
    subagents_dir = os.path.join(session_dir, 'subagents')
    if os.path.isdir(subagents_dir):
        for sa_file in sorted(os.listdir(subagents_dir)):
            if sa_file.endswith('.jsonl'):
                agent_id = sa_file.replace('.jsonl', '').replace('agent-', '')
                sa = _parse_subagent(subagents_dir, sa_file, agent_id)
                if sa:
                    session.subagents.append(sa)

    return session


def _parse_subagent(subagents_dir: str, filename: str, agent_id: str) -> Optional[SubAgent]:
    """Parse a subagent JSONL file."""
    filepath = os.path.join(subagents_dir, filename)
    entries = parse_jsonl_file(filepath)
    if not entries:
        return None

    sa = SubAgent(agent_id=agent_id)
    seen_uuids: dict[str, int] = {}
    pending_tool_calls: dict[str, ToolCall] = {}

    for entry in entries:
        entry_type = entry.get('type')
        timestamp = entry.get('timestamp')
        msg_data = entry.get('message', {})

        if not msg_data:
            continue

        role = msg_data.get('role', '')
        content = msg_data.get('content', [])
        uuid = entry.get('uuid')

        if timestamp:
            if not sa.start_time or timestamp < sa.start_time:
                sa.start_time = timestamp
            if not sa.end_time or timestamp > sa.end_time:
                sa.end_time = timestamp

        if entry_type == 'user' and role == 'user':
            tool_results_data = _extract_tool_results(content)
            if tool_results_data:
                for tool_use_id, result_text in tool_results_data:
                    if tool_use_id in pending_tool_calls:
                        pending_tool_calls[tool_use_id].output_result = result_text
            else:
                text = _extract_text_from_content(content)
                if text.strip():
                    if not sa.prompt:
                        sa.prompt = text[:500]
                    msg = Message(role='user', content_text=text, timestamp=timestamp, uuid=uuid)
                    sa.messages.append(msg)

        elif entry_type == 'assistant' and role == 'assistant':
            model = msg_data.get('model')
            if model and not sa.model:
                sa.model = model

            usage = msg_data.get('usage', {})

            if uuid and uuid in seen_uuids:
                idx = seen_uuids[uuid]
                existing = sa.messages[idx]
                new_text = _extract_text_from_content(content)
                if new_text.strip() and new_text.strip() not in existing.content_text:
                    if existing.content_text:
                        existing.content_text += new_text
                    else:
                        existing.content_text = new_text

                new_tools = _extract_tool_uses(content)
                existing_ids = {tc.tool_use_id for tc in existing.tool_calls}
                for tc in new_tools:
                    if tc.tool_use_id not in existing_ids:
                        tc.timestamp = timestamp
                        existing.tool_calls.append(tc)
                        pending_tool_calls[tc.tool_use_id] = tc

                if usage:
                    existing.input_tokens = usage.get('input_tokens', existing.input_tokens)
                    existing.output_tokens = usage.get('output_tokens', existing.output_tokens)
                    existing.cache_read_tokens = usage.get('cache_read_input_tokens', existing.cache_read_tokens)
                    existing.cache_creation_tokens = usage.get('cache_creation_input_tokens', existing.cache_creation_tokens)
                existing.timestamp = timestamp
                continue

            text = _extract_text_from_content(content)
            tool_uses = _extract_tool_uses(content)

            msg = Message(
                role='assistant',
                content_text=text.strip(),
                timestamp=timestamp,
                model=model,
                uuid=uuid,
            )
            if usage:
                msg.input_tokens = usage.get('input_tokens', 0)
                msg.output_tokens = usage.get('output_tokens', 0)
                msg.cache_read_tokens = usage.get('cache_read_input_tokens', 0)
                msg.cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)

            for tc in tool_uses:
                tc.timestamp = timestamp
                msg.tool_calls.append(tc)
                pending_tool_calls[tc.tool_use_id] = tc

            if uuid:
                seen_uuids[uuid] = len(sa.messages)
            sa.messages.append(msg)

    return sa


def _calc_duration(start_ts: Optional[str], end_ts: Optional[str]) -> Optional[float]:
    """Calculate duration in ms between two ISO-8601 timestamps."""
    if not start_ts or not end_ts:
        return None
    try:
        start = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_ts.replace('Z', '+00:00'))
        return (end - start).total_seconds() * 1000
    except (ValueError, TypeError):
        return None


def get_session_ids(project_dir: str) -> list[str]:
    """List all session IDs in a project directory."""
    sessions = []
    try:
        for f in os.listdir(project_dir):
            if f.endswith('.jsonl'):
                sessions.append(f.replace('.jsonl', ''))
    except OSError:
        pass
    return sessions


def get_session_summary(project_dir: str, session_id: str) -> dict:
    """Get a lightweight summary of a session without parsing all messages."""
    jsonl_path = os.path.join(project_dir, f'{session_id}.jsonl')
    summary = {
        'session_id': session_id,
        'slug': '',
        'start_time': None,
        'end_time': None,
        'model': None,
        'message_count': 0,
        'tool_call_count': 0,
        'has_subagents': False,
        'first_prompt': '',
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_cache_read_tokens': 0,
        'total_cache_creation_tokens': 0,
        'model_breakdowns': {},  # model_name -> {input, output, cache_read, cache_create}
        'team_name': '',
        'agent_name': '',
    }

    subagents_dir = os.path.join(project_dir, session_id, 'subagents')
    if os.path.isdir(subagents_dir):
        sa_files = [f for f in os.listdir(subagents_dir) if f.endswith('.jsonl')]
        summary['has_subagents'] = len(sa_files) > 0
        summary['subagent_count'] = len(sa_files)
    else:
        summary['has_subagents'] = False
        summary['subagent_count'] = 0

    entries = parse_jsonl_file(jsonl_path)
    seen_hashes = set()  # messageId:requestId for dedup (matches ccusage)
    seen_uuids = set()   # fallback for entries without messageId/requestId

    for entry in entries:
        entry_type = entry.get('type')
        timestamp = entry.get('timestamp')

        if not summary['slug'] and entry.get('slug'):
            summary['slug'] = entry['slug']
        if not summary['team_name'] and entry.get('teamName'):
            summary['team_name'] = entry['teamName']
        if not summary['agent_name'] and entry.get('agentName'):
            summary['agent_name'] = entry['agentName']

        if timestamp:
            if not summary['start_time'] or timestamp < summary['start_time']:
                summary['start_time'] = timestamp
            if not summary['end_time'] or timestamp > summary['end_time']:
                summary['end_time'] = timestamp

        if entry_type in ('user', 'assistant'):
            msg = entry.get('message', {})
            uuid = entry.get('uuid')
            is_sidechain = entry.get('isSidechain', False)
            if is_sidechain:
                continue

            # Deduplicate streaming entries by messageId:requestId (ccusage algorithm)
            msg_id = msg.get('id')
            req_id = entry.get('requestId')
            if msg_id and req_id:
                h = f'{msg_id}:{req_id}'
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            elif uuid:
                if uuid in seen_uuids:
                    continue
                seen_uuids.add(uuid)

            model = msg.get('model')
            if model:
                summary['model'] = model  # Track latest model

            role = msg.get('role', '')
            content = msg.get('content', [])

            if entry_type == 'user' and role == 'user':
                # Count only real user messages (not tool results)
                if isinstance(content, str) or (isinstance(content, list) and not any(
                    isinstance(b, dict) and b.get('type') == 'tool_result' for b in content
                )):
                    summary['message_count'] += 1
                    # Capture first user prompt
                    if not summary['first_prompt']:
                        summary['first_prompt'] = _extract_text_from_content(content)[:200]

            elif entry_type == 'assistant' and role == 'assistant':
                summary['message_count'] += 1
                usage = msg.get('usage', {})
                u_input = usage.get('input_tokens', 0)
                u_output = usage.get('output_tokens', 0)
                u_cache_read = usage.get('cache_read_input_tokens', 0)
                u_cache_create = usage.get('cache_creation_input_tokens', 0)
                summary['total_input_tokens'] += u_input
                summary['total_output_tokens'] += u_output
                summary['total_cache_read_tokens'] += u_cache_read
                summary['total_cache_creation_tokens'] += u_cache_create

                # Track per-model breakdown for accurate cost calculation
                m_name = model or 'unknown'
                if m_name not in summary['model_breakdowns']:
                    summary['model_breakdowns'][m_name] = {
                        'input': 0, 'output': 0,
                        'cache_read': 0, 'cache_create': 0,
                    }
                mb = summary['model_breakdowns'][m_name]
                mb['input'] += u_input
                mb['output'] += u_output
                mb['cache_read'] += u_cache_read
                mb['cache_create'] += u_cache_create

                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'tool_use':
                            summary['tool_call_count'] += 1

    # Include subagent tokens in model_breakdowns for cost calculation
    _aggregate_subagent_tokens(
        os.path.join(project_dir, session_id, 'subagents'),
        summary['model_breakdowns'],
    )

    return summary


def _aggregate_subagent_tokens(subagents_dir: str, model_breakdowns: dict):
    """Aggregate token usage from subagent JSONL files into model_breakdowns.

    Uses messageId:requestId deduplication matching ccusage behavior.
    """
    if not os.path.isdir(subagents_dir):
        return

    for sa_file in os.listdir(subagents_dir):
        if not sa_file.endswith('.jsonl'):
            continue
        sa_path = os.path.join(subagents_dir, sa_file)
        seen_hashes = set()
        for entry in parse_jsonl_file(sa_path):
            msg = entry.get('message', {})
            usage = msg.get('usage')
            if not usage or not isinstance(usage, dict):
                continue
            inp = usage.get('input_tokens')
            out = usage.get('output_tokens')
            if inp is None or out is None:
                continue

            # Deduplicate by messageId:requestId
            msg_id = msg.get('id')
            req_id = entry.get('requestId')
            if msg_id and req_id:
                h = f'{msg_id}:{req_id}'
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

            model = msg.get('model') or 'unknown'
            if model not in model_breakdowns:
                model_breakdowns[model] = {
                    'input': 0, 'output': 0,
                    'cache_read': 0, 'cache_create': 0,
                }
            mb = model_breakdowns[model]
            mb['input'] += inp
            mb['output'] += out
            mb['cache_read'] += usage.get('cache_read_input_tokens', 0)
            mb['cache_create'] += usage.get('cache_creation_input_tokens', 0)


def parse_stats_cache() -> dict:
    """Parse ~/.claude/stats-cache.json for global statistics."""
    stats_file = CLAUDE_DIR / 'stats-cache.json'
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, IOError, json.JSONDecodeError):
        return {}
