'''
Tests for the Job class and the portioning it relies on.

These cover the parts that need no API access: portioning, info-file migration, status derivation,
and orphan-batch adoption.
'''
import json
import os

import pytest

from decarb import util
from decarb.llm import prompt
from decarb.llm.prompt import Job, JOB_SCHEMA_VERSION, jsonl_portions


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _record(custom_id: str, content: str = 'some prompt text') -> dict:
    return {
        'custom_id': custom_id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {'model': 'gpt-5.1', 'messages': [{'role': 'user', 'content': content}]},
    }


def _custom_id(i: int) -> str:
    # matches the {company}_{report_type}_{year}_{term} convention of prompt_id()
    return f'exxon_annual-report_{2000 + i}_ccs'


@pytest.fixture
def records() -> list[dict]:
    return [_record(_custom_id(i), f'prompt number {i} ' * (i + 1)) for i in range(50)]


@pytest.fixture
def term_collection(tmp_path, monkeypatch):
    '''A throwaway term collection on disk, with the module-level dirs pointed at tmp_path.'''
    terms_dir = tmp_path / 'data' / 'terms'
    out_dir = tmp_path / 'out'
    (terms_dir / 'test-terms' / 'prompts').mkdir(parents=True)
    (terms_dir / 'test-terms' / 'batches').mkdir(parents=True)
    (out_dir / 'test-terms' / 'batch-results').mkdir(parents=True)

    monkeypatch.setattr(prompt, 'TERMS_DIR', str(terms_dir))
    monkeypatch.setattr(prompt, 'OUT_DIR', str(out_dir))
    monkeypatch.setattr(prompt, 'BASE_DIR', str(tmp_path))
    # resolve_path closes over util.BASE_DIR, so patching the prompt module alone is not enough
    monkeypatch.setattr(util, 'BASE_DIR', str(tmp_path))
    return terms_dir / 'test-terms'


@pytest.fixture
def job(term_collection, records, monkeypatch):
    '''A Job backed by a 50-record jsonl, with terms/companies resolution stubbed out.'''
    jsonl_path = term_collection / 'prompts' / ('0' * 32 + '.jsonl')
    jsonl_path.write_text(''.join(json.dumps(r) + '\n' for r in records))

    class _FakeTerms:
        uuid = 'terms-test'
        folder_name = 'test-terms'
        batches_folder = str(term_collection / 'batches')
        prompts_folder = str(term_collection / 'prompts')

    class _FakeCompanies:
        uuid = 'cos-test'
        companies = []

    monkeypatch.setattr(Job, '_resolve_terms', lambda self, terms_id, path: _FakeTerms())
    monkeypatch.setattr(Job, '_find_companies', lambda self: _FakeCompanies())
    monkeypatch.setattr(prompt, 'CompanyCollection', lambda names: _FakeCompanies())

    return Job(str(jsonl_path))


# --------------------------------------------------------------------------- #
# portioning
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kwargs', [
    {'strategy': 'max_tokens', 'max_tokens': 200},
    {'strategy': 'max_prompts', 'max_prompts': 7},
    {'strategy': 'split_into_n', 'split_into_n': 4},
])
def test_portions_cover_every_record_exactly_once(records, kwargs):
    '''Every prompt lands in exactly one portion - no gaps, no duplicates.'''
    portions = list(jsonl_portions(records, **kwargs))
    assert portions, 'expected at least one portion'

    seen = [cid for p in portions for cid in p.custom_ids]
    assert seen == [r['custom_id'] for r in records]  # same set, same order, no repeats


@pytest.mark.parametrize('kwargs', [
    {'strategy': 'max_tokens', 'max_tokens': 200},
    {'strategy': 'max_prompts', 'max_prompts': 7},
    {'strategy': 'split_into_n', 'split_into_n': 4},
])
def test_portion_indices_are_exclusive_and_contiguous(records, kwargs):
    '''start_index/end_index slice the input exactly, and portions abut without gaps.'''
    portions = list(jsonl_portions(records, **kwargs))

    assert portions[0].start_index == 0
    assert portions[-1].end_index == len(records)
    for p in portions:
        assert records[p.start_index:p.end_index] == p.portion
    for earlier, later in zip(portions, portions[1:]):
        assert earlier.end_index == later.start_index


def test_portions_accept_a_file_path(records, tmp_path):
    '''A path and an equivalent list of dicts produce identical portions.'''
    path = tmp_path / 'prompts.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))

    from_path = list(jsonl_portions(str(path), strategy='max_prompts', max_prompts=7))
    from_list = list(jsonl_portions(records, strategy='max_prompts', max_prompts=7))

    assert [p.custom_ids for p in from_path] == [p.custom_ids for p in from_list]


def test_split_into_n_yields_requested_number_of_portions(records):
    '''split_into_n must not consume its input before iterating it.'''
    assert len(list(jsonl_portions(records, strategy='split_into_n', split_into_n=5))) == 5


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def test_new_job_is_created_not_completed(job):
    '''A job with no batches must not read as completed via a vacuous all().'''
    assert job.batches == []
    assert job.status == 'created'
    assert len(job.missing_prompts) == job.n_prompts


@pytest.mark.parametrize('statuses, expected', [
    (['completed'], 'completed'),
    (['completed', 'completed'], 'completed'),
    (['failed'], 'failed'),
    (['in_progress', 'completed'], 'working'),
    (['failed', 'completed'], 'part-failed-completed'),
    (['failed', 'in_progress'], 'part-failed-working'),
])
def test_job_status_is_derived_from_batches(job, statuses, expected):
    for i, status in enumerate(statuses):
        job._update_batch({'id': f'batch_{i}', 'status': status, 'custom_ids': []})
    job._update_job_status('auto')
    assert job.status == expected


def test_superseded_batches_are_ignored_by_status(job):
    job._update_batch({'id': 'batch_old', 'status': 'failed', 'custom_ids': [], 'superseded': True})
    job._update_batch({'id': 'batch_new', 'status': 'completed', 'custom_ids': []})
    job._update_job_status('auto')
    assert job.status == 'completed'


# --------------------------------------------------------------------------- #
# migration
# --------------------------------------------------------------------------- #

def test_migration_from_legacy_portions_info_recovers_every_prompt(job, records):
    '''
    Legacy portions-info files record an inclusive end_index, which the v1 conversion expanded with
    range(start, end) - dropping the last prompt of every batch. Migration must use prompt_count
    and leave no gaps.
    '''
    legacy = {
        'metadata': {'parent': job.prompt_jsonl_path, 'strategy': 'max_tokens', 'strategy_param': 750_000},
        'batches': {
            # inclusive end_index, exactly as the old pipeline wrote it
            'batch_a': {'start_index': 0, 'end_index': 19, 'token_count': 100, 'prompt_count': 20, 'completed': True, 'status': 'completed'},
            'batch_b': {'start_index': 20, 'end_index': 49, 'token_count': 100, 'prompt_count': 30, 'completed': True, 'status': 'completed'},
        },
    }
    migrated = job._migrate_info(legacy)

    assert migrated['schema_version'] == JOB_SCHEMA_VERSION
    covered = [cid for b in migrated['batches'] for cid in b['custom_ids']]
    assert covered == job.custom_ids  # all 50, in order, no gaps
    assert covered[19] == _custom_id(19)  # the index the old expansion dropped


def test_migration_from_v1_indices_repairs_off_by_one(job):
    '''
    A v1 info file whose indices are one short of prompt_count is repaired from prompt_count,
    which is what the batch actually contained.
    '''
    v1 = {
        'parent': job.prompt_jsonl_path,
        'batches': [
            {'id': 'batch_a', 'indices': list(range(0, 19)), 'prompt_count': 20, 'status': 'completed'},
        ],
    }
    migrated = job._migrate_info(v1)

    custom_ids = migrated['batches'][0]['custom_ids']
    assert len(custom_ids) == 20
    assert custom_ids[-1] == _custom_id(19)


def test_migration_of_v2_is_a_no_op(job):
    v2 = {'schema_version': JOB_SCHEMA_VERSION, 'parent': job.prompt_jsonl_path, 'batches': []}
    assert job._migrate_info(v2) is v2


# --------------------------------------------------------------------------- #
# coverage and orphan adoption
# --------------------------------------------------------------------------- #

def _write_result_file(term_collection, batch_id, custom_ids, created_at):
    '''Write the batch detail json and result jsonl an orphaned batch would leave behind.'''
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'{batch_id}.jsonl'), 'w') as f:
        for cid in custom_ids:
            f.write(json.dumps({
                'custom_id': cid,
                'response': {'status_code': 200, 'body': {'choices': [{'message': {'content': '{}'}}]}},
                'error': None,
            }) + '\n')
    with open(os.path.join(str(term_collection), 'batches', f'{batch_id}.json'), 'w') as f:
        json.dump({'id': batch_id, 'status': 'completed', 'created_at': created_at}, f)


def test_completed_prompts_ignores_errored_records(job, term_collection):
    _write_result_file(term_collection, 'batch_ok', [_custom_id(0), _custom_id(1)], created_at=100)
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    with open(os.path.join(results_dir, 'batch_ok.jsonl'), 'a') as f:
        f.write(json.dumps({'custom_id': _custom_id(2), 'response': None, 'error': {'code': 'boom'}}) + '\n')

    job._update_batch({'id': 'batch_ok', 'status': 'completed',
                       'custom_ids': [_custom_id(0), _custom_id(1), _custom_id(2)],
                       'result_path': os.path.relpath(os.path.join(results_dir, 'batch_ok.jsonl'), prompt.BASE_DIR)})

    assert job.completed_prompts == {_custom_id(0), _custom_id(1)}
    assert _custom_id(2) in job.missing_prompts


def test_adopts_orphan_that_fills_a_gap(job, term_collection):
    '''An unclaimed batch covering missing prompts, created after the job started, is adopted.'''
    job._update_batch({'id': 'batch_own', 'status': 'failed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 100, 'result_path': None})

    _write_result_file(term_collection, 'batch_orphan', [_custom_id(0)], created_at=200)

    assert job.adopt_orphan_batches() == ['batch_orphan']
    assert job.batch_by_id('batch_orphan')['adopted'] is True
    assert job.completed_prompts == {_custom_id(0)}


def test_does_not_adopt_orphan_predating_the_job(job, term_collection):
    '''
    custom_ids identify a report/term combination, not a job, so an earlier unrelated run over the
    same reports produces batches whose custom_ids are also a subset of this job's.
    '''
    job._update_batch({'id': 'batch_own', 'status': 'failed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 100, 'result_path': None})

    _write_result_file(term_collection, 'batch_earlier', [_custom_id(0)], created_at=50)

    assert job.adopt_orphan_batches() == []


def test_does_not_adopt_orphan_with_foreign_custom_ids(job, term_collection):
    job._update_batch({'id': 'batch_own', 'status': 'failed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 100, 'result_path': None})

    _write_result_file(term_collection, 'batch_foreign', ['shell_annual-report_1999_ccs'], created_at=200)

    assert job.adopt_orphan_batches() == []


def test_does_not_adopt_orphan_that_fills_no_gap(job, term_collection):
    '''A duplicate of work already accounted for adds nothing and must be left alone.'''
    _write_result_file(term_collection, 'batch_own', [_custom_id(0)], created_at=100)
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    job._update_batch({'id': 'batch_own', 'status': 'completed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 100,
                       'result_path': os.path.relpath(os.path.join(results_dir, 'batch_own.jsonl'), prompt.BASE_DIR)})

    _write_result_file(term_collection, 'batch_duplicate', [_custom_id(0)], created_at=200)

    assert job.adopt_orphan_batches() == []


def test_failed_batch_is_retired_once_its_prompts_have_results(job, term_collection):
    '''A batch that failed but whose prompts were answered by a later batch must not leave the job part-failed.'''
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    job._update_batch({'id': 'batch_failed', 'status': 'failed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 100, 'result_path': None})

    _write_result_file(term_collection, 'batch_retry', [_custom_id(0)], created_at=200)
    job._update_batch({'id': 'batch_retry', 'status': 'completed', 'custom_ids': [_custom_id(0)],
                       'submitted_at': 200,
                       'result_path': os.path.relpath(os.path.join(results_dir, 'batch_retry.jsonl'), prompt.BASE_DIR)})

    job._resolve_duplicate_coverage()
    job._update_job_status('auto')

    assert job.batch_by_id('batch_failed')['superseded'] is True
    assert job.status == 'completed'
    assert job.output_paths == [job.batch_by_id('batch_retry')['result_path']]


# --------------------------------------------------------------------------- #
# comprehensiveness
# --------------------------------------------------------------------------- #

@pytest.fixture
def expected(job, monkeypatch):
    '''Stub the on-disk prompt scan: the first 60 custom_ids are expected, the job holds 50.'''
    drafted = {_custom_id(i) for i in range(60)}
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: drafted))
    return drafted


def test_not_comprehensive_when_prompts_are_absent(job, expected):
    assert job.prompts_comprehensive is False
    assert job.results_comprehensive is False

    cov = job.coverage()
    assert cov['n_expected'] == 60
    assert cov['n_prompts_present'] == 50
    assert cov['missing_from_input'] == [_custom_id(i) for i in range(50, 60)]
    # every prompt absent from the input is also missing a result
    assert set(cov['missing_from_input']) <= set(cov['missing_results'])


def test_prompts_comprehensive_without_results(job, term_collection, monkeypatch):
    '''A job holding every expected prompt but no results is prompt-comprehensive only.'''
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: {_custom_id(i) for i in range(50)}))

    assert job.prompts_comprehensive is True
    assert job.results_comprehensive is False
    assert job.coverage()['n_completed'] == 0


def test_results_comprehensive_when_every_expected_prompt_has_a_result(job, term_collection, monkeypatch):
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: {_custom_id(i) for i in range(50)}))

    all_ids = [_custom_id(i) for i in range(50)]
    _write_result_file(term_collection, 'batch_all', all_ids, created_at=100)
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    job._update_batch({'id': 'batch_all', 'status': 'completed', 'custom_ids': all_ids,
                       'submitted_at': 100,
                       'result_path': os.path.relpath(os.path.join(results_dir, 'batch_all.jsonl'), prompt.BASE_DIR)})

    assert job.prompts_comprehensive is True
    assert job.results_comprehensive is True

    cov = job.coverage()
    assert cov == {**cov, 'missing_results': [], 'missing_from_input': [], 'unexpected': []}
    assert cov['missing_by_year'] == {}


def test_coverage_reports_unexpected_prompts(job, monkeypatch):
    '''Prompts in the job with no prompt file on disk are listed under `unexpected`.'''
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: {_custom_id(i) for i in range(40)}))

    assert job.coverage()['unexpected'] == sorted(_custom_id(i) for i in range(40, 50))


def test_coverage_tallies_missing_by_field(job, monkeypatch):
    '''custom_id is {company}_{report_type}_{year}_{term}, so each tally keys off its own field.'''
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: {_custom_id(0), _custom_id(1)}))

    cov = job.coverage()
    assert cov['missing_by_company'] == {'exxon': 2}
    assert cov['missing_by_report_type'] == {'annual-report': 2}
    assert cov['missing_by_year'] == {'2000': 1, '2001': 1}
    assert cov['missing_by_term'] == {'ccs': 2}


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #

def _sibling_job(term_collection, monkeypatch, name, custom_ids):
    '''A second job in the same term collection, covering the given custom_ids.'''
    path = term_collection / 'prompts' / f'{name}.jsonl'
    path.write_text(''.join(json.dumps(_record(cid)) + '\n' for cid in custom_ids))
    return Job(str(path))


def test_merge_unions_prompts_and_batches(job, term_collection, monkeypatch):
    other_ids = [f'bp_annual-report_{2000 + i}_ccs' for i in range(10)]
    other = _sibling_job(term_collection, monkeypatch, 'b' * 32, other_ids)

    _write_result_file(term_collection, 'batch_a', [_custom_id(0)], created_at=100)
    _write_result_file(term_collection, 'batch_b', [other_ids[0]], created_at=100)
    results_dir = os.path.join(prompt.OUT_DIR, 'test-terms', 'batch-results')
    job._update_batch({'id': 'batch_a', 'status': 'completed', 'custom_ids': [_custom_id(0)], 'submitted_at': 100,
                       'result_path': os.path.relpath(os.path.join(results_dir, 'batch_a.jsonl'), prompt.BASE_DIR)})
    other._update_batch({'id': 'batch_b', 'status': 'completed', 'custom_ids': [other_ids[0]], 'submitted_at': 100,
                         'result_path': os.path.relpath(os.path.join(results_dir, 'batch_b.jsonl'), prompt.BASE_DIR)})

    merged = job.merge(other)

    assert merged.n_prompts == 60
    assert merged.custom_ids[:50] == job.custom_ids  # this job's prompts first, order preserved
    assert set(merged.custom_ids) == set(job.custom_ids) | set(other_ids)
    assert {b['id'] for b in merged.batches} == {'batch_a', 'batch_b'}
    assert merged.completed_prompts == {_custom_id(0), other_ids[0]}
    assert merged.info['merged_from'] == sorted([job.id, other.id])


def test_merge_deduplicates_shared_prompts(job, term_collection, monkeypatch):
    overlapping = [_custom_id(i) for i in range(45, 55)]
    other = _sibling_job(term_collection, monkeypatch, 'c' * 32, overlapping)

    merged = job.merge(other)

    assert len(merged.custom_ids) == len(set(merged.custom_ids))
    assert merged.n_prompts == 55


def test_merge_is_idempotent(job, term_collection, monkeypatch):
    other = _sibling_job(term_collection, monkeypatch, 'd' * 32, ['bp_annual-report_1999_ccs'])
    assert job.merge(other).id == job.merge(other).id


def test_add_operator_merges(job, term_collection, monkeypatch):
    other = _sibling_job(term_collection, monkeypatch, 'e' * 32, ['bp_annual-report_1999_ccs'])
    assert (job + other).id == job.merge(other).id


def test_merge_leaves_sources_untouched(job, term_collection, monkeypatch):
    other = _sibling_job(term_collection, monkeypatch, 'f' * 32, ['bp_annual-report_1999_ccs'])
    job.merge(other)

    assert job.n_prompts == 50
    assert other.n_prompts == 1


def test_merge_rejects_different_term_collections(job, term_collection, monkeypatch):
    other = _sibling_job(term_collection, monkeypatch, '9' * 32, ['bp_annual-report_1999_ccs'])

    class _OtherTerms:
        uuid = 'terms-other'
        folder_name = 'other-terms'
        batches_folder = str(term_collection / 'batches')
        prompts_folder = str(term_collection / 'prompts')

    monkeypatch.setattr(other, 'terms', _OtherTerms())

    with pytest.raises(ValueError, match='different term collections'):
        job.merge(other)


def test_merged_job_can_be_comprehensive_when_sources_are_not(job, term_collection, monkeypatch):
    '''Two jobs that each cover part of the expected set are comprehensive once merged.'''
    other_ids = [f'bp_annual-report_{2000 + i}_ccs' for i in range(10)]
    other = _sibling_job(term_collection, monkeypatch, '8' * 32, other_ids)

    drafted = {_custom_id(i) for i in range(50)} | set(other_ids)
    monkeypatch.setattr(Job, 'expected_prompts', property(lambda self: drafted))

    assert job.prompts_comprehensive is False
    assert other.prompts_comprehensive is False

    merged = job.merge(other)
    assert merged.prompts_comprehensive is True
