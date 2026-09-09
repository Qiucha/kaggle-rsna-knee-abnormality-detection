"""Translation orchestration checks; never call a translation provider."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('translation_runner', Path(__file__).resolve().parents[1] / 'scripts/translate_reports_claude.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_notebook_request_and_original_source():
    contract = {'MODEL': 'opus', 'SYSTEM_PROMPT': 'exact prompt\n', 'user_prefix': 'Translate this report into standard English:\n\n'}
    cmd = runner.command('Original\nreport', contract)
    assert cmd == ['claude', '-p', 'Translate this report into standard English:\n\nOriginal\nreport', '--model', 'opus', '--output-format', 'json', '--max-turns', '1', '--system-prompt', 'exact prompt\n', '--tools', '', '--mcp-config', '{"mcpServers":{}}', '--strict-mcp-config']


def test_resume_rejects_wrong_identity_and_conflicting_success(tmp_path):
    rows = [{'StudyInstanceUID': 'a', 'Report': 'original'}, {'StudyInstanceUID': 'b', 'Report': 'other'}]
    item = dict(runner.identity(rows[0], 0, 'contract'), status='ok', translation='English')
    a = tmp_path/'a.jsonl'; b = tmp_path/'b.jsonl'
    a.write_text(json.dumps(item)+'\n')
    assert list(runner.load_results([a], rows, 'contract')) == ['a']
    with pytest.raises(ValueError, match='mismatch'):
        runner.load_results([a], rows[::-1], 'contract')
    b.write_text(json.dumps(dict(item, translation='different'))+'\n')
    with pytest.raises(ValueError, match='Conflicting'):
        runner.load_results([a,b], rows, 'contract')


def test_failed_and_empty_results_not_treated_as_complete(tmp_path):
    rows = [{'StudyInstanceUID': 'a', 'Report': 'original'}]
    item = dict(runner.identity(rows[0], 0, 'c'), status='failed')
    p = tmp_path/'results.jsonl'; p.write_text(json.dumps(item)+'\n')
    assert runner.load_results([p], rows, 'c') == {}
    p.write_text(json.dumps(dict(item, status='ok', translation='  '))+'\n')
    with pytest.raises(ValueError, match='Empty'):
        runner.load_results([p], rows, 'c')


def test_provider_error_retry_and_capture_metadata():
    row = {'StudyInstanceUID': 'a', 'Report': 'original'}
    contract = {'MODEL': 'opus', 'SYSTEM_PROMPT': 'prompt', 'user_prefix': 'prefix', 'MAX_ATTEMPTS': 2, 'TIMEOUT_S': 180}
    calls = [SimpleNamespace(returncode=0, stdout=json.dumps({'is_error': True, 'result': 'error'})),
             SimpleNamespace(returncode=0, stdout=json.dumps({'result': 'English', 'modelUsage': {'actual-model': {}}, 'usage': {'input_tokens': 1}}))]
    with patch.object(runner.subprocess, 'run', side_effect=calls) as call, patch.object(runner.time, 'sleep'):
        result = runner.translate(row, 0, contract, 'c')
    assert call.call_count == 2
    assert result['status'] == 'ok'
    assert 'actual-model' in result['model_usage']


def test_merge_preserves_english_and_refuses_missing_rows(tmp_path):
    bundle = tmp_path/'bundle'; bundle.mkdir()
    source = 'StudyInstanceUID,Report,Original Language\na,Original English,en\nb,Original Spanish,es\n'
    (bundle/'source.csv').write_text(source)
    (bundle/'contract.json').write_text(json.dumps({'source_sha256': runner.digest(source.encode())}))
    _, h, rows = runner.load_bundle(bundle)
    result_path = tmp_path/'results.jsonl'; result_path.write_text('')
    args = SimpleNamespace(bundle=bundle, results=[result_path], output=tmp_path/'merged.csv')
    with pytest.raises(ValueError, match='missing'):
        runner.merge(args)
    assert not args.output.exists()
    result_path.write_text(json.dumps(dict(runner.identity(rows[1], 1, h), status='ok', translation='Translated English'))+'\n')
    runner.merge(args)
    output = runner.read_csv(args.output)
    assert [r['Translated Report'] for r in output] == ['Original English', 'Translated English']
    with pytest.raises(FileExistsError):
        runner.merge(args)


def test_cli_event_list_extracts_final_result():
    row = {'StudyInstanceUID': 'a', 'Report': 'original'}
    contract = {'MODEL': 'opus', 'SYSTEM_PROMPT': 'prompt', 'user_prefix': 'prefix', 'MAX_ATTEMPTS': 1, 'TIMEOUT_S': 180}
    events = [{'type': 'system'}, {'type': 'assistant'}, {'type': 'result', 'is_error': False, 'result': 'English', 'modelUsage': {'claude-opus-5': {}}}]
    with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(events))):
        result = runner.translate(row, 0, contract, 'c')
    assert result['translation'] == 'English'
    assert result['model_usage'] == {'claude-opus-5': {}}


def test_reverse_run_checkpoints_and_stops_after_failure(tmp_path):
    rows = [{'StudyInstanceUID': str(i), 'Report': f'report {i}'} for i in range(5)]
    args = SimpleNamespace(bundle=tmp_path, start=0, stop=5, workers=1, reverse=True,
                           output=tmp_path/'tail.jsonl', completed=[], execute=True, max_failed_rows=1)
    seen = []
    def fake_translate(row, index, contract, contract_hash):
        seen.append(index)
        return dict(runner.identity(row, index, contract_hash), status='failed', error='quota')
    with patch.object(runner, 'load_bundle', return_value=({}, 'c', rows)), patch.object(runner, 'translate', side_effect=fake_translate), patch.object(runner.subprocess, 'check_output', return_value='test CLI'):
        with pytest.raises(RuntimeError, match='1 rows failed'):
            runner.run(args)
    assert seen == [4]
    assert json.loads(args.output.read_text())['row_index'] == 4


def test_explicit_opus_effort():
    contract = {'MODEL': 'claude-opus-5', 'EFFORT': 'low', 'SYSTEM_PROMPT': 'prompt', 'user_prefix': 'prefix'}
    cmd = runner.command('original', contract)
    assert cmd[cmd.index('--model')+1] == 'claude-opus-5'
    assert cmd[-2:] == ['--effort','low']
