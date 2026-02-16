"""
High-level data access layer for Claude Code session viewer.

Provides functions to list projects, sessions, get session details,
and aggregate statistics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .parser import (
    CLAUDE_DIR,
    PROJECTS_DIR,
    get_session_ids,
    get_session_summary,
    parse_jsonl_file,
    parse_session,
    parse_stats_cache,
)


def list_projects() -> list[dict]:
    """
    List all projects with their metadata.
    Returns list of dicts with: name, dir_name, path, session_count, last_active
    """
    projects = []
    if not PROJECTS_DIR.is_dir():
        return projects

    for dir_name in sorted(os.listdir(PROJECTS_DIR)):
        project_path = PROJECTS_DIR / dir_name
        if not project_path.is_dir():
            continue

        # Convert dir name to readable project name
        # e.g. '-home-user-develop-projectname' -> 'projectname'
        name = _dir_name_to_project_name(dir_name)

        # Count sessions and find last active
        session_ids = get_session_ids(str(project_path))
        if not session_ids:
            continue

        last_active = None
        original_path = ''
        for sid in session_ids:
            jsonl_file = project_path / f'{sid}.jsonl'
            try:
                mtime = os.path.getmtime(jsonl_file)
                ts = datetime.fromtimestamp(mtime).isoformat()
                if not last_active or ts > last_active:
                    last_active = ts
            except OSError:
                pass
            # Extract cwd from first entry that has it (for original path)
            if not original_path:
                original_path = _get_cwd_from_session(str(jsonl_file))

        projects.append({
            'name': name,
            'dir_name': dir_name,
            'path': str(project_path),
            'session_count': len(session_ids),
            'last_active': last_active,
            'original_path': original_path,
        })

    # Sort by last_active descending
    projects.sort(key=lambda p: p.get('last_active') or '', reverse=True)
    return projects


def list_sessions(project_dir_name: str) -> list[dict]:
    """
    List all sessions for a project with summaries.
    Returns list of session summary dicts sorted by start_time descending.
    """
    project_path = PROJECTS_DIR / project_dir_name
    if not project_path.is_dir():
        return []

    session_ids = get_session_ids(str(project_path))
    sessions = []

    for sid in session_ids:
        summary = get_session_summary(str(project_path), sid)
        summary['project_dir_name'] = project_dir_name
        sessions.append(summary)

    # Sort by start_time descending
    sessions.sort(key=lambda s: s.get('start_time') or '', reverse=True)
    return sessions


def get_session_detail(project_dir_name: str, session_id: str) -> Optional[dict]:
    """
    Get full session detail including all messages, tool calls, and subagents.
    Returns a dict representation of SessionData.
    """
    project_path = PROJECTS_DIR / project_dir_name
    if not project_path.is_dir():
        return None

    session = parse_session(str(project_path), session_id)

    # Build timeline: interleaved messages and tool calls
    timeline = []
    for msg in session.messages:
        entry = {
            'type': 'message',
            'role': msg.role,
            'content': msg.content_text,
            'timestamp': msg.timestamp,
            'model': msg.model,
            'uuid': msg.uuid,
            'has_thinking': msg.has_thinking,
            'thinking_text': msg.thinking_text,
            'input_tokens': msg.input_tokens,
            'output_tokens': msg.output_tokens,
            'cache_read_tokens': msg.cache_read_tokens,
            'cache_creation_tokens': msg.cache_creation_tokens,
            'tool_calls': [],
        }

        for tc in msg.tool_calls:
            tool_entry = {
                'tool_name': tc.tool_name,
                'tool_use_id': tc.tool_use_id,
                'input_params': _summarize_tool_input(tc.tool_name, tc.input_params),
                'input_params_raw': tc.input_params,
                'output_result': _truncate(tc.output_result, 5000),
                'output_result_full_length': len(tc.output_result),
                'timestamp': tc.timestamp,
                'duration_ms': tc.duration_ms,
            }
            entry['tool_calls'].append(tool_entry)

        timeline.append(entry)

    # Build subagent data
    subagents = []
    for sa in session.subagents:
        sa_data = {
            'agent_id': sa.agent_id,
            'prompt': _truncate(sa.prompt, 500),
            'model': sa.model,
            'start_time': sa.start_time,
            'end_time': sa.end_time,
            'message_count': len(sa.messages),
            'messages': [],
        }
        for msg in sa.messages:
            sa_msg = {
                'role': msg.role,
                'content': msg.content_text,
                'timestamp': msg.timestamp,
                'model': msg.model,
                'tool_calls': [],
            }
            for tc in msg.tool_calls:
                sa_msg['tool_calls'].append({
                    'tool_name': tc.tool_name,
                    'input_params': _summarize_tool_input(tc.tool_name, tc.input_params),
                    'input_params_raw': tc.input_params,
                    'output_result': _truncate(tc.output_result, 3000),
                    'timestamp': tc.timestamp,
                })
            sa_data['messages'].append(sa_msg)
        subagents.append(sa_data)

    # Compute tool usage summary
    tool_summary = {}
    for msg in session.messages:
        for tc in msg.tool_calls:
            name = tc.tool_name
            if name not in tool_summary:
                tool_summary[name] = {'count': 0, 'total_duration_ms': 0}
            tool_summary[name]['count'] += 1
            if tc.duration_ms:
                tool_summary[name]['total_duration_ms'] += tc.duration_ms

    # Calculate duration
    duration_ms = None
    if session.start_time and session.end_time:
        try:
            start = datetime.fromisoformat(session.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(session.end_time.replace('Z', '+00:00'))
            duration_ms = (end - start).total_seconds() * 1000
        except (ValueError, TypeError):
            pass

    return {
        'session_id': session.session_id,
        'slug': session.slug,
        'project_dir_name': project_dir_name,
        'cwd': session.cwd,
        'git_branch': session.git_branch,
        'version': session.version,
        'model': session.model,
        'team_name': session.team_name,
        'agent_name': session.agent_name,
        'start_time': session.start_time,
        'end_time': session.end_time,
        'duration_ms': duration_ms,
        'total_input_tokens': session.total_input_tokens,
        'total_output_tokens': session.total_output_tokens,
        'total_cache_read_tokens': session.total_cache_read_tokens,
        'total_cache_creation_tokens': session.total_cache_creation_tokens,
        'message_count': len(session.messages),
        'timeline': timeline,
        'subagents': subagents,
        'tool_summary': tool_summary,
    }


def get_global_stats() -> dict:
    """Get global statistics from stats-cache.json."""
    stats = parse_stats_cache()
    if not stats:
        return {
            'total_sessions': 0,
            'total_messages': 0,
            'model_usage': {},
            'daily_activity': [],
            'daily_model_tokens': [],
        }

    # Compute aggregate token totals across all models
    model_usage = stats.get('modelUsage', {})
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0
    total_cost = 0.0
    for model_name, usage in model_usage.items():
        if isinstance(usage, dict):
            total_input += usage.get('inputTokens', 0)
            total_output += usage.get('outputTokens', 0)
            total_cache_read += usage.get('cacheReadInputTokens', 0)
            total_cache_create += usage.get('cacheCreationInputTokens', 0)
            total_cost += usage.get('costUSD', 0)

    return {
        'total_sessions': stats.get('totalSessions', 0),
        'total_messages': stats.get('totalMessages', 0),
        'first_session_date': stats.get('firstSessionDate'),
        'model_usage': model_usage,
        'daily_activity': stats.get('dailyActivity', []),
        'daily_model_tokens': stats.get('dailyModelTokens', []),
        'longest_session': stats.get('longestSession'),
        'hour_counts': stats.get('hourCounts', {}),
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_cache_read_tokens': total_cache_read,
        'total_cache_creation_tokens': total_cache_create,
        'total_cost_usd': total_cost,
    }


def get_dashboard_data() -> dict:
    """Get aggregated data for the dashboard view."""
    projects = list_projects()
    stats = get_global_stats()

    # Recent sessions across all projects (last 10)
    recent_sessions = []
    for proj in projects[:10]:
        sessions = list_sessions(proj['dir_name'])
        for s in sessions[:3]:
            s['project_name'] = proj['name']
            recent_sessions.append(s)
    recent_sessions.sort(key=lambda s: s.get('start_time') or '', reverse=True)
    recent_sessions = recent_sessions[:10]
    enrich_sessions_with_agent_info(recent_sessions)

    # Check active status for recent sessions
    for s in recent_sessions:
        s['is_active'] = is_session_active(s.get('project_dir_name', ''), s.get('session_id', ''))

    return {
        'projects': projects,
        'stats': stats,
        'recent_sessions': recent_sessions,
        'project_count': len(projects),
    }


def get_project_token_stats() -> list[dict]:
    """Aggregate token usage per project across all sessions.

    Returns list of dicts sorted by total tokens descending, each containing:
    name, dir_name, session_count, total_input, total_output,
    total_cache_read, total_cache_write, grand_total.
    """
    projects = list_projects()
    result = []

    for proj in projects:
        sessions = list_sessions(proj['dir_name'])
        total_input = sum(s.get('total_input_tokens', 0) for s in sessions)
        total_output = sum(s.get('total_output_tokens', 0) for s in sessions)
        total_cache_read = sum(s.get('total_cache_read_tokens', 0) for s in sessions)
        total_cache_write = sum(s.get('total_cache_creation_tokens', 0) for s in sessions)
        grand_total = total_input + total_output + total_cache_read + total_cache_write
        total_messages = sum(s.get('message_count', 0) for s in sessions)
        total_tools = sum(s.get('tool_call_count', 0) for s in sessions)

        result.append({
            'name': proj['name'],
            'dir_name': proj['dir_name'],
            'original_path': proj.get('original_path', ''),
            'session_count': proj['session_count'],
            'last_active': proj.get('last_active'),
            'total_input': total_input,
            'total_output': total_output,
            'total_cache_read': total_cache_read,
            'total_cache_write': total_cache_write,
            'grand_total': grand_total,
            'total_messages': total_messages,
            'total_tools': total_tools,
        })

    result.sort(key=lambda p: p['grand_total'], reverse=True)
    return result


def get_project_usage_detail(project_dir_name: str) -> Optional[dict]:
    """Get detailed token usage breakdown for a single project.

    Returns project info, per-session token table, per-model aggregation,
    and daily token usage.
    """
    projects = list_projects()
    project = next((p for p in projects if p['dir_name'] == project_dir_name), None)
    if not project:
        return None

    sessions = list_sessions(project_dir_name)
    enrich_sessions_with_agent_info(sessions)

    # Aggregate totals
    total_input = sum(s.get('total_input_tokens', 0) for s in sessions)
    total_output = sum(s.get('total_output_tokens', 0) for s in sessions)
    total_cache_read = sum(s.get('total_cache_read_tokens', 0) for s in sessions)
    total_cache_write = sum(s.get('total_cache_creation_tokens', 0) for s in sessions)
    grand_total = total_input + total_output + total_cache_read + total_cache_write
    total_messages = sum(s.get('message_count', 0) for s in sessions)
    total_tools = sum(s.get('tool_call_count', 0) for s in sessions)

    # Per-session token breakdown (sorted by total tokens desc)
    session_rows = []
    for s in sessions:
        s_input = s.get('total_input_tokens', 0)
        s_output = s.get('total_output_tokens', 0)
        s_cache_read = s.get('total_cache_read_tokens', 0)
        s_cache_write = s.get('total_cache_creation_tokens', 0)
        s_total = s_input + s_output + s_cache_read + s_cache_write
        session_rows.append({
            'session_id': s['session_id'],
            'slug': s.get('slug', ''),
            'model': s.get('model', ''),
            'start_time': s.get('start_time'),
            'message_count': s.get('message_count', 0),
            'tool_call_count': s.get('tool_call_count', 0),
            'agent_name': s.get('agent_name', ''),
            'team_name': s.get('team_name', ''),
            'input_tokens': s_input,
            'output_tokens': s_output,
            'cache_read_tokens': s_cache_read,
            'cache_write_tokens': s_cache_write,
            'total_tokens': s_total,
        })
    session_rows.sort(key=lambda r: r['total_tokens'], reverse=True)

    # Per-model aggregation
    model_stats = {}
    for s in sessions:
        model = s.get('model') or 'unknown'
        if model not in model_stats:
            model_stats[model] = {
                'input': 0, 'output': 0, 'cache_read': 0,
                'cache_write': 0, 'total': 0,
                'session_count': 0, 'message_count': 0,
            }
        ms = model_stats[model]
        s_input = s.get('total_input_tokens', 0)
        s_output = s.get('total_output_tokens', 0)
        s_cache_read = s.get('total_cache_read_tokens', 0)
        s_cache_write = s.get('total_cache_creation_tokens', 0)
        ms['input'] += s_input
        ms['output'] += s_output
        ms['cache_read'] += s_cache_read
        ms['cache_write'] += s_cache_write
        ms['total'] += s_input + s_output + s_cache_read + s_cache_write
        ms['session_count'] += 1
        ms['message_count'] += s.get('message_count', 0)

    # Sort models by total tokens desc
    model_list = [
        {'name': name, **stats}
        for name, stats in sorted(model_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    ]

    # Daily token aggregation
    daily_stats = {}
    for s in sessions:
        start = s.get('start_time')
        if not start:
            continue
        try:
            date_str = start[:10]  # YYYY-MM-DD
        except (TypeError, IndexError):
            continue
        if date_str not in daily_stats:
            daily_stats[date_str] = {
                'input': 0, 'output': 0, 'cache_read': 0,
                'cache_write': 0, 'total': 0,
                'session_count': 0, 'message_count': 0,
            }
        ds = daily_stats[date_str]
        s_input = s.get('total_input_tokens', 0)
        s_output = s.get('total_output_tokens', 0)
        s_cache_read = s.get('total_cache_read_tokens', 0)
        s_cache_write = s.get('total_cache_creation_tokens', 0)
        ds['input'] += s_input
        ds['output'] += s_output
        ds['cache_read'] += s_cache_read
        ds['cache_write'] += s_cache_write
        ds['total'] += s_input + s_output + s_cache_read + s_cache_write
        ds['session_count'] += 1
        ds['message_count'] += s.get('message_count', 0)

    daily_list = [
        {'date': date, **stats}
        for date, stats in sorted(daily_stats.items())
    ]

    return {
        'project': project,
        'total_input': total_input,
        'total_output': total_output,
        'total_cache_read': total_cache_read,
        'total_cache_write': total_cache_write,
        'grand_total': grand_total,
        'total_messages': total_messages,
        'total_tools': total_tools,
        'session_count': len(sessions),
        'sessions': session_rows,
        'models': model_list,
        'daily': daily_list,
    }


def _dir_name_to_project_name(dir_name: str) -> str:
    """Convert directory name to a readable project name.
    e.g. '-home-user-develop-projectname' -> 'projectname'
    """
    parts = dir_name.strip('-').split('-')
    # Find the last meaningful part(s) after common prefixes
    # Skip home, username, and common dir names
    skip = {'home', 'user', 'develop', 'projects', 'Downloads'}
    meaningful = [p for p in parts if p and p not in skip]
    if meaningful:
        return '-'.join(meaningful)
    return dir_name


def _get_cwd_from_session(jsonl_path: str) -> str:
    """Extract the cwd from the first entry in a session JSONL file."""
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cwd = entry.get('cwd')
                    if cwd:
                        return cwd
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        pass
    return ''


def _summarize_tool_input(tool_name: str, params: dict) -> str:
    """Create a human-readable summary of tool input parameters."""
    if not params:
        return ''

    if tool_name == 'Bash':
        cmd = params.get('command', '')
        desc = params.get('description', '')
        return desc if desc else (cmd[:100] + '...' if len(cmd) > 100 else cmd)

    if tool_name == 'Read':
        return params.get('file_path', '')

    if tool_name == 'Write':
        path = params.get('file_path', '')
        content = params.get('content', '')
        return f'{path} ({len(content)} chars)'

    if tool_name == 'Edit':
        path = params.get('file_path', '')
        old = params.get('old_string', '')
        return f'{path} (replacing {len(old)} chars)'

    if tool_name == 'Glob':
        pattern = params.get('pattern', '')
        path = params.get('path', '')
        return f'{pattern}' + (f' in {path}' if path else '')

    if tool_name == 'Grep':
        pattern = params.get('pattern', '')
        path = params.get('path', '')
        return f'/{pattern}/' + (f' in {path}' if path else '')

    if tool_name == 'Task':
        desc = params.get('description', '')
        sa_type = params.get('subagent_type', '')
        return f'{sa_type}: {desc}' if sa_type else desc

    if tool_name in ('TaskCreate', 'TaskUpdate', 'TaskGet', 'TaskList'):
        subject = params.get('subject', '')
        task_id = params.get('taskId', '')
        return subject or f'task #{task_id}' if task_id else ''

    if tool_name == 'WebFetch':
        return params.get('url', '')

    if tool_name == 'WebSearch':
        return params.get('query', '')

    if tool_name == 'SendMessage':
        recipient = params.get('recipient', '')
        msg_type = params.get('type', '')
        return f'{msg_type} to {recipient}' if recipient else msg_type

    # Generic fallback
    keys = list(params.keys())
    if len(keys) == 1:
        val = str(params[keys[0]])
        return val[:100] + '...' if len(val) > 100 else val
    return ', '.join(f'{k}={str(v)[:50]}' for k, v in list(params.items())[:3])


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + '...'


# ---------------------------------------------------------------------------
# Teams & Tasks
# ---------------------------------------------------------------------------

TEAMS_DIR = CLAUDE_DIR / 'teams'
TASKS_DIR = CLAUDE_DIR / 'tasks'


def list_teams() -> list[dict]:
    """List all teams with their config and task summary."""
    teams = []
    if not TEAMS_DIR.is_dir():
        return teams

    for team_name in sorted(os.listdir(TEAMS_DIR)):
        team_path = TEAMS_DIR / team_name
        if not team_path.is_dir():
            continue

        config = _read_team_config(team_name)
        if not config:
            continue

        tasks = _read_team_tasks(team_name)
        task_counts = {'total': len(tasks), 'pending': 0, 'in_progress': 0, 'completed': 0}
        for t in tasks:
            s = t.get('status', 'pending')
            if s in task_counts:
                task_counts[s] += 1

        teams.append({
            'name': config.get('name', team_name),
            'description': config.get('description', ''),
            'created_at': config.get('createdAt'),
            'lead_agent_id': config.get('leadAgentId', ''),
            'lead_session_id': config.get('leadSessionId', ''),
            'member_count': len(config.get('members', [])),
            'members': config.get('members', []),
            'task_counts': task_counts,
            'is_active': _is_team_active(team_name),
        })

    # Active teams first, then by creation time
    teams.sort(key=lambda t: (not t['is_active'], -(t.get('created_at') or 0)))
    return teams


def get_team_detail(team_name: str) -> Optional[dict]:
    """Get full team detail including config, members, and all tasks with deps."""
    config = _read_team_config(team_name)
    if not config:
        return None

    tasks = _read_team_tasks(team_name)
    members = config.get('members', [])

    # Enrich members with their task assignments
    member_tasks = {m.get('name', ''): [] for m in members}
    for t in tasks:
        owner = t.get('owner', '')
        if owner and owner in member_tasks:
            member_tasks[owner].append(t.get('id'))

    enriched_members = []
    for m in members:
        name = m.get('name', '')
        enriched_members.append({
            'agent_id': m.get('agentId', ''),
            'name': name,
            'agent_type': m.get('agentType', ''),
            'prompt': m.get('prompt', ''),
            'model': m.get('model', ''),
            'joined_at': m.get('joinedAt'),
            'cwd': m.get('cwd', ''),
            'assigned_tasks': member_tasks.get(name, []),
        })

    # Build task dependency graph info
    # Compute "depth" (column) for each task based on dependency chain
    task_map = {t['id']: t for t in tasks}
    for t in tasks:
        t['depth'] = _compute_task_depth(t['id'], task_map, {})

    max_depth = max((t['depth'] for t in tasks), default=0)

    # Group tasks by status for summary
    task_counts = {'total': len(tasks), 'pending': 0, 'in_progress': 0, 'completed': 0}
    for t in tasks:
        s = t.get('status', 'pending')
        if s in task_counts:
            task_counts[s] += 1

    progress_pct = 0
    if task_counts['total'] > 0:
        progress_pct = int(task_counts['completed'] / task_counts['total'] * 100)

    return {
        'name': config.get('name', team_name),
        'description': config.get('description', ''),
        'created_at': config.get('createdAt'),
        'lead_agent_id': config.get('leadAgentId', ''),
        'lead_session_id': config.get('leadSessionId', ''),
        'members': enriched_members,
        'tasks': tasks,
        'task_counts': task_counts,
        'max_depth': max_depth,
        'progress_pct': progress_pct,
        'is_active': _is_team_active(team_name),
    }


def _read_team_config(team_name: str) -> Optional[dict]:
    """Read a team's config.json."""
    config_path = TEAMS_DIR / team_name / 'config.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_team_tasks(team_name: str) -> list[dict]:
    """Read all task files for a team, sorted by id."""
    tasks_dir = TASKS_DIR / team_name
    tasks = []
    if not tasks_dir.is_dir():
        return tasks

    for fname in os.listdir(tasks_dir):
        if not fname.endswith('.json'):
            continue
        try:
            with open(tasks_dir / fname, 'r', encoding='utf-8') as f:
                task = json.load(f)
            if isinstance(task, dict) and task.get('id'):
                # Ensure arrays exist
                task.setdefault('blocks', [])
                task.setdefault('blockedBy', [])
                task.setdefault('owner', '')
                task.setdefault('status', 'pending')
                task.setdefault('depth', 0)
                tasks.append(task)
        except (OSError, json.JSONDecodeError):
            continue

    tasks.sort(key=lambda t: int(t.get('id', '0') or '0'))
    return tasks


def _compute_task_depth(task_id: str, task_map: dict, cache: dict) -> int:
    """Compute the topological depth of a task (for graph layout)."""
    if task_id in cache:
        return cache[task_id]
    task = task_map.get(task_id)
    if not task:
        return 0
    blocked_by = task.get('blockedBy', [])
    if not blocked_by:
        cache[task_id] = 0
        return 0
    depth = max(_compute_task_depth(bid, task_map, cache) for bid in blocked_by) + 1
    cache[task_id] = depth
    return depth


def get_member_detail(team_name: str, member_name: str) -> Optional[dict]:
    """Get full detail for a team member including tasks and activity timeline."""
    config = _read_team_config(team_name)
    if not config:
        return None

    # Find the member
    members = config.get('members', [])
    member = None
    for m in members:
        if m.get('name') == member_name:
            member = m
            break
    if not member:
        return None

    # Get all tasks for this team
    all_tasks = _read_team_tasks(team_name)

    # Build enriched task info for tasks assigned to this member
    assigned_tasks = []
    for t in all_tasks:
        if t.get('owner', '') == member_name:
            assigned_tasks.append(t)

    task_counts = {'total': len(assigned_tasks), 'pending': 0, 'in_progress': 0, 'completed': 0}
    for t in assigned_tasks:
        s = t.get('status', 'pending')
        if s in task_counts:
            task_counts[s] += 1

    # Try to find the member's subagent session
    # The lead session's subagents dir may contain this member's JSONL
    lead_session_id = config.get('leadSessionId', '')
    agent_id = member.get('agentId', '')
    subagent_session = None
    subagent_link = None  # (project_dir_name, session_id) for linking

    if lead_session_id and agent_id:
        subagent_session, subagent_link = _find_member_subagent(
            lead_session_id, agent_id, member_name
        )

    # Compute member stats from subagent session
    stats = {
        'message_count': 0,
        'tool_call_count': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
    }
    if subagent_session:
        for msg in subagent_session.get('messages', []):
            stats['message_count'] += 1
            stats['total_input_tokens'] += msg.get('input_tokens', 0)
            stats['total_output_tokens'] += msg.get('output_tokens', 0)
            stats['tool_call_count'] += len(msg.get('tool_calls', []))

    # Get prompt: prefer config prompt, fallback to subagent's first message
    prompt = member.get('prompt', '')
    if not prompt and subagent_session:
        prompt = subagent_session.get('prompt', '')

    return {
        'name': member.get('name', ''),
        'agent_id': agent_id,
        'agent_type': member.get('agentType', ''),
        'prompt': prompt,
        'model': member.get('model', ''),
        'joined_at': member.get('joinedAt'),
        'cwd': member.get('cwd', ''),
        'team_name': team_name,
        'team_description': config.get('description', ''),
        'assigned_tasks': assigned_tasks,
        'task_counts': task_counts,
        'all_team_tasks': all_tasks,
        'stats': stats,
        'subagent_session': subagent_session,
        'subagent_link': subagent_link,
        'is_active': any(t.get('status') == 'in_progress' for t in assigned_tasks),
    }


def _find_member_subagent(lead_session_id: str, agent_id: str,
                          member_name: str) -> tuple[Optional[dict], Optional[tuple]]:
    """Try to find a team member's subagent session data.

    Searches across all projects for the lead session, then looks in its
    subagents directory for a matching agent.

    Returns (subagent_data_dict, (project_dir_name, session_id)) or (None, None).
    """
    from .parser import _parse_subagent, _extract_text_from_content

    if not PROJECTS_DIR.is_dir():
        return None, None

    for dir_name in os.listdir(PROJECTS_DIR):
        project_path = PROJECTS_DIR / dir_name
        if not project_path.is_dir():
            continue

        # Check if the lead session exists in this project
        lead_jsonl = project_path / f'{lead_session_id}.jsonl'
        if not lead_jsonl.is_file():
            continue

        # Found the lead session's project
        session_dir = project_path / lead_session_id
        subagents_dir = session_dir / 'subagents'
        if not subagents_dir.is_dir():
            continue

        # Try exact match by agent_id
        for sa_file in sorted(os.listdir(subagents_dir)):
            if not sa_file.endswith('.jsonl'):
                continue

            sa_agent_id = sa_file.replace('.jsonl', '').replace('agent-', '')

            # Match by agent_id or by name appearing in the prompt
            if sa_agent_id == agent_id:
                sa = _parse_subagent(str(subagents_dir), sa_file, sa_agent_id)
                if sa:
                    return _subagent_to_dict(sa), (dir_name, lead_session_id)

        # If no exact match, try to find by scanning prompts for member name
        for sa_file in sorted(os.listdir(subagents_dir)):
            if not sa_file.endswith('.jsonl'):
                continue
            sa_agent_id = sa_file.replace('.jsonl', '').replace('agent-', '')
            sa = _parse_subagent(str(subagents_dir), sa_file, sa_agent_id)
            if sa and member_name.lower() in (sa.prompt or '').lower():
                return _subagent_to_dict(sa), (dir_name, lead_session_id)

    return None, None


def _subagent_to_dict(sa) -> dict:
    """Convert a SubAgent dataclass to a dict with timeline data."""
    messages = []
    for msg in sa.messages:
        m = {
            'role': msg.role,
            'content': msg.content_text,
            'timestamp': msg.timestamp,
            'model': msg.model,
            'input_tokens': msg.input_tokens,
            'output_tokens': msg.output_tokens,
            'tool_calls': [],
        }
        for tc in msg.tool_calls:
            m['tool_calls'].append({
                'tool_name': tc.tool_name,
                'tool_use_id': tc.tool_use_id,
                'input_params': _summarize_tool_input(tc.tool_name, tc.input_params),
                'input_params_raw': tc.input_params,
                'output_result': _truncate(tc.output_result, 5000),
                'timestamp': tc.timestamp,
                'duration_ms': tc.duration_ms,
            })
        messages.append(m)

    return {
        'agent_id': sa.agent_id,
        'prompt': sa.prompt,
        'model': sa.model,
        'start_time': sa.start_time,
        'end_time': sa.end_time,
        'message_count': len(sa.messages),
        'messages': messages,
    }


def get_session_agent_map() -> dict:
    """Build a mapping from session_id -> agent info by scanning team configs.

    Returns dict: { session_id: { team_name, agent_name, agent_type } }
    """
    mapping = {}
    if not TEAMS_DIR.is_dir():
        return mapping

    for team_name in os.listdir(TEAMS_DIR):
        config = _read_team_config(team_name)
        if not config:
            continue

        t_name = config.get('name', team_name)
        lead_agent_id = config.get('leadAgentId', '')
        lead_session_id = config.get('leadSessionId', '')
        members = config.get('members', [])

        # Map lead session to the lead member
        if lead_session_id:
            lead_name = ''
            for m in members:
                if m.get('agentId') == lead_agent_id:
                    lead_name = m.get('name', '')
                    break
            mapping[lead_session_id] = {
                'team_name': t_name,
                'agent_name': lead_name or 'team-lead',
                'agent_type': 'orchestrator',
            }

        # Map member sessions by scanning subagent dirs under the lead session
        if lead_session_id and PROJECTS_DIR.is_dir():
            for dir_name in os.listdir(PROJECTS_DIR):
                sa_dir = PROJECTS_DIR / dir_name / lead_session_id / 'subagents'
                if not sa_dir.is_dir():
                    continue
                for sa_file in os.listdir(sa_dir):
                    if not sa_file.endswith('.jsonl'):
                        continue
                    sa_id = sa_file.replace('.jsonl', '').replace('agent-', '')
                    # Match to team member by agentId
                    for m in members:
                        agent_id = m.get('agentId', '')
                        if agent_id == sa_id or agent_id.split('@')[0] == sa_id:
                            # The subagent ID is also a session key in some views
                            mapping[sa_id] = {
                                'team_name': t_name,
                                'agent_name': m.get('name', ''),
                                'agent_type': m.get('agentType', ''),
                            }
                            break
                break  # Only need to find one project dir with the lead session

    return mapping


# Module-level cache for session-agent map (refreshed per request)
_session_agent_cache: dict = {}
_session_agent_cache_time: float = 0


def get_session_agent_info(session_id: str) -> Optional[dict]:
    """Get agent info for a session ID, using cached map."""
    global _session_agent_cache, _session_agent_cache_time
    import time
    now = time.time()
    # Refresh cache every 30 seconds
    if now - _session_agent_cache_time > 30:
        _session_agent_cache = get_session_agent_map()
        _session_agent_cache_time = now
    return _session_agent_cache.get(session_id)


def enrich_sessions_with_agent_info(sessions: list[dict]) -> list[dict]:
    """Add agent_name and team_name to session dicts if they belong to a team."""
    agent_map = get_session_agent_map()
    for s in sessions:
        sid = s.get('session_id', '')
        info = agent_map.get(sid)
        if info:
            s['agent_name'] = info['agent_name']
            s['team_name'] = info['team_name']
            s['agent_type'] = info['agent_type']
    return sessions


def _is_team_active(team_name: str) -> bool:
    """Check if a team has any in_progress tasks or recently modified files."""
    import time
    tasks = _read_team_tasks(team_name)
    if any(t.get('status') == 'in_progress' for t in tasks):
        return True
    # Also check if task files were recently modified
    tasks_dir = TASKS_DIR / team_name
    if tasks_dir.is_dir():
        try:
            for fname in os.listdir(tasks_dir):
                fpath = tasks_dir / fname
                mtime = os.path.getmtime(fpath)
                if (time.time() - mtime) < 120:
                    return True
        except OSError:
            pass
    return False


# ---------------------------------------------------------------------------
# Live / Active session support
# ---------------------------------------------------------------------------

ACTIVE_THRESHOLD_SECONDS = 120  # sessions modified within 2 minutes are "active"


def is_session_active(project_dir_name: str, session_id: str) -> bool:
    """Check if a session JSONL file was recently modified."""
    import time
    jsonl_path = PROJECTS_DIR / project_dir_name / f'{session_id}.jsonl'
    try:
        mtime = os.path.getmtime(jsonl_path)
        return (time.time() - mtime) < ACTIVE_THRESHOLD_SECONDS
    except OSError:
        return False


def get_active_sessions() -> list[dict]:
    """Find all currently active sessions across all projects."""
    import time
    active = []
    if not PROJECTS_DIR.is_dir():
        return active

    now = time.time()
    for dir_name in os.listdir(PROJECTS_DIR):
        project_path = PROJECTS_DIR / dir_name
        if not project_path.is_dir():
            continue

        for f in os.listdir(project_path):
            if not f.endswith('.jsonl'):
                continue
            fpath = project_path / f
            try:
                mtime = os.path.getmtime(fpath)
                if (now - mtime) < ACTIVE_THRESHOLD_SECONDS:
                    sid = f.replace('.jsonl', '')
                    summary = get_session_summary(str(project_path), sid)
                    summary['project_dir_name'] = dir_name
                    summary['project_name'] = _dir_name_to_project_name(dir_name)
                    summary['is_active'] = True
                    summary['last_modified'] = datetime.fromtimestamp(mtime).isoformat()
                    active.append(summary)
            except OSError:
                continue

    active.sort(key=lambda s: s.get('last_modified') or '', reverse=True)
    return active


def get_jsonl_path(project_dir_name: str, session_id: str) -> Optional[str]:
    """Get the full filesystem path to a session JSONL file."""
    p = PROJECTS_DIR / project_dir_name / f'{session_id}.jsonl'
    return str(p) if p.is_file() else None


def get_session_dir(project_dir_name: str, session_id: str) -> str:
    """Get the session directory path (for tool-results etc)."""
    return str(PROJECTS_DIR / project_dir_name / session_id)


def parse_incremental_entries(jsonl_path: str, session_dir: str,
                              from_line: int) -> tuple[list[dict], int]:
    """Parse new JSONL entries starting from a given line number.

    Returns (list of timeline-format entries, new line count).
    Each entry is pre-formatted for direct use in the frontend.
    """
    entries = []
    line_num = 0
    pending_tool_calls: dict[str, dict] = {}

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line_num <= from_line:
                    # For lines before from_line, still track tool calls
                    # so we can match results to calls
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = raw.get('message', {})
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'tool_use':
                                pending_tool_calls[block.get('id', '')] = {
                                    'tool_name': block.get('name', ''),
                                    'timestamp': raw.get('timestamp'),
                                }
                    continue

                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry = _raw_entry_to_timeline(raw, session_dir, pending_tool_calls)
                if entry:
                    entries.append(entry)

    except (OSError, IOError):
        pass

    return entries, line_num


def _raw_entry_to_timeline(raw: dict, session_dir: str,
                           pending_tool_calls: dict) -> Optional[dict]:
    """Convert a raw JSONL entry to a frontend-ready timeline dict."""
    from .parser import (
        _extract_text_from_content,
        _extract_thinking_from_content,
        _extract_tool_uses,
        _extract_tool_results,
        _read_tool_result_file,
    )

    entry_type = raw.get('type')
    timestamp = raw.get('timestamp')
    is_sidechain = raw.get('isSidechain', False)

    if is_sidechain or entry_type in ('file-history-snapshot', 'progress'):
        return None

    msg_data = raw.get('message', {})
    if not msg_data:
        return None

    role = msg_data.get('role', '')
    content = msg_data.get('content', [])
    uuid = raw.get('uuid')

    if entry_type == 'user' and role == 'user':
        # Check if this is a tool result
        from .parser import _extract_tool_results
        tool_results_data = _extract_tool_results(content)
        if tool_results_data:
            # Tool result - match to pending calls and emit as a tool_result event
            results = []
            for tool_use_id, result_text in tool_results_data:
                external = _read_tool_result_file(session_dir, tool_use_id)
                if external is not None:
                    result_text = external
                results.append({
                    'tool_use_id': tool_use_id,
                    'output_result': _truncate(result_text, 5000),
                    'output_result_full_length': len(result_text),
                })
            return {
                'type': 'tool_result',
                'timestamp': timestamp,
                'results': results,
            }
        else:
            text = _extract_text_from_content(content)
            if text.strip():
                return {
                    'type': 'message',
                    'role': 'user',
                    'content': text,
                    'timestamp': timestamp,
                    'uuid': uuid,
                    'tool_calls': [],
                    'has_thinking': False,
                    'thinking_text': '',
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cache_read_tokens': 0,
                    'cache_creation_tokens': 0,
                    'model': None,
                }

    elif entry_type == 'assistant' and role == 'assistant':
        text = _extract_text_from_content(content)
        thinking = _extract_thinking_from_content(content)
        tool_uses = _extract_tool_uses(content)
        usage = msg_data.get('usage', {})
        model = msg_data.get('model')

        tool_calls = []
        for tc in tool_uses:
            pending_tool_calls[tc.tool_use_id] = {
                'tool_name': tc.tool_name,
                'timestamp': timestamp,
            }
            tool_calls.append({
                'tool_name': tc.tool_name,
                'tool_use_id': tc.tool_use_id,
                'input_params': _summarize_tool_input(tc.tool_name, tc.input_params),
                'input_params_raw': tc.input_params,
                'output_result': '',
                'output_result_full_length': 0,
                'timestamp': timestamp,
                'duration_ms': None,
            })

        return {
            'type': 'message',
            'role': 'assistant',
            'content': text.strip(),
            'timestamp': timestamp,
            'uuid': uuid,
            'model': model,
            'has_thinking': bool(thinking),
            'thinking_text': thinking,
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'cache_read_tokens': usage.get('cache_read_input_tokens', 0),
            'cache_creation_tokens': usage.get('cache_creation_input_tokens', 0),
            'tool_calls': tool_calls,
        }

    return None
