"""
Claude Code Session Viewer - Flask Application

A web interface for browsing and analyzing Claude Code session logs.
"""

import json
import os
import time

from flask import Flask, render_template, jsonify, abort, Response, request

from services.data import (
    list_projects,
    list_sessions,
    get_session_detail,
    get_global_stats,
    get_dashboard_data,
    get_active_sessions,
    is_session_active,
    get_jsonl_path,
    get_session_dir,
    parse_incremental_entries,
    list_teams,
    get_team_detail,
    get_member_detail,
    enrich_sessions_with_agent_info,
    get_project_token_stats,
    get_project_usage_detail,
    get_subagent_detail,
    group_sessions_by_date,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Dashboard - show all projects, stats, recent sessions."""
    data = get_dashboard_data()
    return render_template('index.html', **data)


@app.route('/usage')
def usage():
    """Token usage breakdown page."""
    stats = get_global_stats()
    projects = list_projects()
    return render_template('usage.html', stats=stats, projects=projects)


@app.route('/usage/projects')
def usage_projects():
    """Compare token usage across all projects."""
    project_stats = get_project_token_stats()
    stats = get_global_stats()
    return render_template('usage_projects.html', project_stats=project_stats, stats=stats)


@app.route('/usage/project/<path:project_dir_name>')
def usage_project(project_dir_name):
    """Detailed token usage breakdown for a single project."""
    detail = get_project_usage_detail(project_dir_name)
    if not detail:
        abort(404)
    return render_template('usage_project.html', **detail)


@app.route('/project/<path:project_dir_name>')
def project_detail(project_dir_name):
    """List sessions for a project."""
    sessions = list_sessions(project_dir_name)
    enrich_sessions_with_agent_info(sessions)
    projects = list_projects()
    project = next((p for p in projects if p['dir_name'] == project_dir_name), None)
    if not project:
        abort(404)
    date_groups = group_sessions_by_date(sessions)
    return render_template('project.html', project=project, sessions=sessions, date_groups=date_groups)


@app.route('/api/session/<path:project_dir_name>/<session_id>')
def api_session(project_dir_name, session_id):
    """JSON API for session data."""
    detail = get_session_detail(project_dir_name, session_id)
    if not detail:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(detail)


# ---------------------------------------------------------------------------
# Teams & Tasks routes
# ---------------------------------------------------------------------------

@app.route('/teams')
def teams_list():
    """List all teams."""
    teams = list_teams()
    return render_template('teams.html', teams=teams)


@app.route('/team/<team_name>')
def team_detail_view(team_name):
    """Team detail with members and task dependency graph."""
    detail = get_team_detail(team_name)
    if not detail:
        abort(404)
    return render_template('team_detail.html', team=detail)


@app.route('/api/teams')
def api_teams():
    """JSON API - list all teams."""
    return jsonify(list_teams())


@app.route('/api/team/<team_name>')
def api_team(team_name):
    """JSON API - team detail."""
    detail = get_team_detail(team_name)
    if not detail:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(detail)


@app.route('/team/<team_name>/member/<member_name>')
def member_detail_view(team_name, member_name):
    """Member detail page with tasks and activity timeline."""
    detail = get_member_detail(team_name, member_name)
    if not detail:
        abort(404)
    return render_template('member_detail.html', member=detail)


@app.route('/api/team/<team_name>/member/<member_name>')
def api_member(team_name, member_name):
    """JSON API - member detail."""
    detail = get_member_detail(team_name, member_name)
    if not detail:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(detail)


@app.template_filter('ms_to_datetime')
def ms_to_datetime(ms):
    """Convert millisecond timestamp to readable datetime."""
    if not ms:
        return ''
    from datetime import datetime
    try:
        dt = datetime.fromtimestamp(ms / 1000)
        return dt.strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError, OSError):
        return ''


# ---------------------------------------------------------------------------
# Live monitoring routes
# ---------------------------------------------------------------------------

@app.route('/live')
def live_dashboard():
    """Show currently active sessions."""
    active = get_active_sessions()
    enrich_sessions_with_agent_info(active)
    return render_template('live.html', active_sessions=active)


@app.route('/live/<path:project_dir_name>/<session_id>')
def live_session(project_dir_name, session_id):
    """Live monitoring view for an active session."""
    detail = get_session_detail(project_dir_name, session_id)
    if not detail:
        abort(404)
    detail['is_active'] = is_session_active(project_dir_name, session_id)
    # Embed mode: minimal chrome for use inside monitor panes
    if request.args.get('embed') == '1':
        return render_template('live_embed.html', session=detail)
    return render_template('live_session.html', session=detail)


@app.route('/monitor')
def monitor():
    """Multi-pane live monitoring dashboard."""
    active = get_active_sessions()
    enrich_sessions_with_agent_info(active)
    return render_template('monitor.html', active_sessions=active)


@app.route('/api/active')
def api_active():
    """JSON API - list active sessions."""
    active = get_active_sessions()
    enrich_sessions_with_agent_info(active)
    return jsonify(active)


@app.route('/api/session/<path:project_dir_name>/<session_id>/stream')
def stream_session(project_dir_name, session_id):
    """SSE endpoint - stream new entries as they appear in the JSONL file."""
    jsonl_path = get_jsonl_path(project_dir_name, session_id)
    if not jsonl_path:
        return jsonify({'error': 'Not found'}), 404

    session_dir = get_session_dir(project_dir_name, session_id)

    def generate():
        # Start from current end of file
        last_line = 0
        try:
            with open(jsonl_path, 'r') as f:
                for _ in f:
                    last_line += 1
        except OSError:
            pass

        # Send initial line count so client knows where we start
        yield f"data: {json.dumps({'type': 'init', 'line_count': last_line})}\n\n"

        idle_count = 0
        while True:
            try:
                new_entries, new_line_count = parse_incremental_entries(
                    jsonl_path, session_dir, last_line
                )

                if new_entries:
                    idle_count = 0
                    for entry in new_entries:
                        yield f"data: {json.dumps(entry, default=str)}\n\n"
                    last_line = new_line_count
                else:
                    idle_count += 1
                    # Send heartbeat every 10 seconds to keep connection alive
                    if idle_count % 5 == 0:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'line_count': last_line})}\n\n"

                    # Stop streaming after 5 minutes of no activity
                    if idle_count > 150:
                        yield f"data: {json.dumps({'type': 'timeout', 'message': 'No activity for 5 minutes'})}\n\n"
                        break

            except GeneratorExit:
                break
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Parse error'})}\n\n"

            time.sleep(2)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/api/session/<path:project_dir_name>/<session_id>/updates')
def api_session_updates(project_dir_name, session_id):
    """Polling endpoint - get new entries since a given line number."""
    jsonl_path = get_jsonl_path(project_dir_name, session_id)
    if not jsonl_path:
        return jsonify({'error': 'Not found'}), 404

    session_dir = get_session_dir(project_dir_name, session_id)
    from_line = request.args.get('from_line', 0, type=int)

    entries, new_line_count = parse_incremental_entries(
        jsonl_path, session_dir, from_line
    )

    return jsonify({
        'entries': entries,
        'line_count': new_line_count,
        'is_active': is_session_active(project_dir_name, session_id),
    })


# ---------------------------------------------------------------------------
# Enhanced session detail (auto-detect active)
# ---------------------------------------------------------------------------

@app.route('/session/<path:project_dir_name>/<session_id>')
def session_detail(project_dir_name, session_id):
    """Full session detail view. Redirects to live view if active."""
    detail = get_session_detail(project_dir_name, session_id)
    if not detail:
        abort(404)
    detail['is_active'] = is_session_active(project_dir_name, session_id)
    return render_template('session.html', session=detail)


@app.route('/session/<path:project_dir_name>/<session_id>/subagent/<agent_id>')
def subagent_detail_view(project_dir_name, session_id, agent_id):
    """Dedicated subagent detail page."""
    detail = get_subagent_detail(project_dir_name, session_id, agent_id)
    if not detail:
        abort(404)
    return render_template('subagent_detail.html', subagent=detail)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


# ---------------------------------------------------------------------------
# Template filters
# ---------------------------------------------------------------------------

@app.template_filter('format_tokens')
def format_tokens(value):
    """Format token count with commas."""
    if not value:
        return '0'
    return f'{value:,}'


@app.template_filter('compact_tokens')
def compact_tokens(value):
    """Format token count in compact form: 1.2B, 65.8M, 12.3K."""
    if not value:
        return '0'
    value = int(value)
    if value >= 1_000_000_000:
        return f'{value / 1_000_000_000:.1f}B'
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if value >= 10_000:
        return f'{value / 1_000:.1f}K'
    return f'{value:,}'


@app.template_filter('format_duration')
def format_duration(ms):
    """Format duration from ms to human readable."""
    if not ms:
        return '-'
    seconds = ms / 1000
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes = seconds / 60
    if minutes < 60:
        return f'{minutes:.1f}m'
    hours = minutes / 60
    return f'{hours:.1f}h'


@app.template_filter('time_ago')
def time_ago(timestamp):
    """Convert ISO timestamp to relative time."""
    if not timestamp:
        return ''
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()
        if seconds < 60:
            return 'just now'
        if seconds < 3600:
            m = int(seconds / 60)
            return f'{m}m ago'
        if seconds < 86400:
            h = int(seconds / 3600)
            return f'{h}h ago'
        days = int(seconds / 86400)
        if days < 30:
            return f'{days}d ago'
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return timestamp or ''


@app.template_filter('format_cost')
def format_cost(value):
    """Format USD cost value. Shows $X.XX for >= $0.01, <$0.01 otherwise."""
    if not value or value <= 0:
        return '-'
    if value >= 100:
        return f'${value:,.0f}'
    if value >= 0.01:
        return f'${value:.2f}'
    return '<$0.01'


@app.template_filter('short_time')
def short_time(timestamp):
    """Format timestamp to short time."""
    if not timestamp:
        return ''
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%H:%M:%S')
    except (ValueError, TypeError):
        return ''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
