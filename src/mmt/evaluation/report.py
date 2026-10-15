'''E08 - exports, static HTML report, frozen manifest and acceptance rules. The renderer reads exported
files only: it never retrains, recalibrates or reselects a threshold.'''
import base64, io, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Template
from .io_utils import canonical_hash, sha256_file, dump_json

def export_risk_coverage(curve, path, operating_tau=None):
    df = pd.DataFrame(curve); df['frozen_operating_point'] = (df.tau == operating_tau) if operating_tau is not None else False
    df.to_csv(path, index=False)

def export_comparison(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)          # per-run records are kept; never replaced by seed means

# ---------------------------------------------------------------- frozen manifest
def freeze_manifest(files):
    '''files: {name: path}. A later change to ANY file means a new version and a disclosed re-evaluation.'''
    entries = {k: sha256_file(v) for k, v in sorted(files.items())}
    return dict(schema_version=1, entries=entries, manifest_hash=canonical_hash(entries))

def verify_frozen(manifest, files):
    bad = [k for k, h in manifest['entries'].items() if k not in files or sha256_file(files[k]) != h]
    if bad:
        raise RuntimeError(f'frozen manifest violated by: {bad}')

# ---------------------------------------------------------------- acceptance
def _value(m, metric):
    for sec in ('quality_S', 'quality_A', 'coverage', 'probability'):
        if m.get(sec, {}).get(metric) is not None: return m[sec][metric]
    return None

def evaluate_acceptance(rules, results):
    '''results: {scenario_id: metrics.json dict}. Missing mandatory bound => not_qualified (never pass).'''
    out = []
    for r in rules:
        rec = dict(rule=r['name'], metric=r['metric'], scenario=r['scenario'], bound=r.get('bound'),
                   direction=r['direction'], measured=None, denominator=None, result=None, reason=None)
        m = results.get(r['scenario'])
        if r.get('bound') is None:
            rec.update(result='not_qualified', reason='bound not defined and frozen')
        elif m is None:
            rec.update(result='not_applicable', reason='scenario not evaluated')
        else:
            v = _value(m, r['metric']); rec['measured'] = v; rec['denominator'] = m['counts']['eligible']
            cov = m['coverage'].get('availability_coverage')
            if v is None:
                rec.update(result='fail', reason='metric undefined (nothing computable)')
            elif r.get('min_coverage') is not None and (cov is None or cov < r['min_coverage']):
                rec.update(result='fail', reason=f'availability coverage {cov} below minimum {r["min_coverage"]}')
            else:
                good = v >= r['bound'] if r['direction'] == 'min' else v <= r['bound']
                rec['result'] = 'pass' if good else 'fail'
        out.append(rec)
    results_set = {x['result'] for x in out}
    pq = ('not_qualified' if 'not_qualified' in results_set else 'fail' if 'fail' in results_set
          else 'pass' if 'pass' in results_set else 'not_qualified')
    return out, pq

def final_status(software_tests_passed, acceptance_status):
    '''Two separate statuses: passing software checks never implies predictive suitability.'''
    return dict(implementation_verified=bool(software_tests_passed), predictive_qualification=acceptance_status)

# ---------------------------------------------------------------- report
def _png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

TEMPLATE = Template('''<!doctype html><html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>body{font-family:sans-serif;max-width:980px;margin:2em auto;padding:0 1em}table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}th{background:#eee}.warn{background:#fff3cd;padding:8px}</style></head><body>
<h1>{{ title }}</h1>
<p class="warn"><b>Status:</b> implementation_verified = {{ status.implementation_verified }};
predictive_qualification = {{ status.predictive_qualification }}. Passing software checks does not establish predictive suitability.</p>
<h2>Scenario results</h2>
<table><tr><th>scenario</th><th>eligible</th><th>computable</th><th>availability cov.</th><th>accepted cov.</th>
<th>macro-F1 (S)</th><th>bal. acc.</th><th>NLL</th><th>Brier</th><th>ECE</th><th>sel. risk</th></tr>
{% for r in rows %}<tr><td style="text-align:left">{{ r.scenario }}</td><td>{{ r.eligible }}</td><td>{{ r.computable }}</td>
<td>{{ r.av }}</td><td>{{ r.ac }}</td><td>{{ r.f1 }}</td><td>{{ r.ba }}</td><td>{{ r.nll }}</td><td>{{ r.brier }}</td><td>{{ r.ece }}</td><td>{{ r.risk }}</td></tr>{% endfor %}</table>
{% for name, img in images %}<h2>{{ name }}</h2><img src="data:image/png;base64,{{ img }}">{% endfor %}
<h2>Acceptance rules</h2>
<table><tr><th>rule</th><th>scenario</th><th>bound</th><th>measured</th><th>denominator</th><th>result</th><th>reason</th></tr>
{% for a in acceptance %}<tr><td style="text-align:left">{{ a.rule }}</td><td style="text-align:left">{{ a.scenario }}</td><td>{{ a.bound }}</td>
<td>{{ a.measured }}</td><td>{{ a.denominator }}</td><td>{{ a.result }}</td><td style="text-align:left">{{ a.reason }}</td></tr>{% endfor %}</table>
<h2>Limitations</h2><ul>{% for l in limitations %}<li>{{ l }}</li>{% endfor %}</ul>
<h2>Provenance</h2><pre>{{ provenance }}</pre></body></html>''')

def _f(x, n=4): return 'null' if x is None else f'{x:.{n}f}'

def render_report(run_dirs, out_html, title, status, acceptance=(), limitations=(), provenance=None, curve_csv=None):
    '''run_dirs: {scenario_id: path to an exported run directory}.'''
    rows, rel = [], None
    for sid, d in run_dirs.items():
        m = json.loads((Path(d) / 'metrics.json').read_text()); c, q, p = m['counts'], m['quality_S'], m['probability']
        rows.append(dict(scenario=sid, eligible=c['eligible'], computable=c['computable'], av=_f(m['coverage']['availability_coverage']),
                         ac=_f(m['coverage']['accepted_coverage']), f1=_f(q['macro_f1']), ba=_f(q['balanced_accuracy']),
                         nll=_f(p.get('nll')), brier=_f(p.get('brier')), ece=_f(p.get('ece')), risk=_f(m['coverage']['selective_risk'])))
        if rel is None and (Path(d) / 'reliability.csv').exists(): rel = pd.read_csv(Path(d) / 'reliability.csv')
    images = []
    ok = [r for r in rows if r['f1'] != 'null']
    if ok:
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(ok) + 1.5))
        ax.barh([r['scenario'] for r in ok], [float(r['f1']) for r in ok]); ax.set_xlabel('macro-F1 on computable decisions (see coverage column)')
        ax.invert_yaxis(); images.append(('Degradation by scenario', _png(fig)))
    if rel is not None:
        r2 = rel.dropna(); fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.plot(r2.mean_confidence, r2.accuracy, 'o-'); ax.set_xlabel('mean confidence'); ax.set_ylabel('empirical accuracy')
        images.append(('Reliability diagram (first run listed)', _png(fig)))
    if curve_csv and Path(curve_csv).exists():
        rc = pd.read_csv(curve_csv).dropna(subset=['risk']); fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rc.coverage, rc.risk, '.-'); ax.set_xlabel('accepted coverage (of eligible)'); ax.set_ylabel('selective risk')
        images.append(('Risk-coverage curve', _png(fig)))
    html = TEMPLATE.render(title=title, status=status, rows=rows, images=images, acceptance=list(acceptance),
                           limitations=list(limitations), provenance=json.dumps(provenance or {}, indent=1, default=str))
    Path(out_html).write_text(html, encoding='utf8')
    return out_html
