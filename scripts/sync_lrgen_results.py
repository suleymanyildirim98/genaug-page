#!/usr/bin/env python3
"""Build the HR->LR generator (lr_generation) run log for lrgen.html.

One tab per training run (Run 1, Run 2, ...). Each run has a description, its variables, a quantitative-metrics table
and a realism table (both seeds), degradation matching on the hand-cropped references where measured, and two visual
sets: the textured DRealSR reference sweep and the hand-crop sweep.

Reads from WORK_DIR/evaluations (written by hr2lr.realism_stats, hr2lr.reference_sweep and the hand-crop sweep) and
writes assets/lrgen/data.json plus JPEG copies of the figures under assets/lrgen/img/<run>/.

    python scripts/sync_lrgen_results.py [--work-dir /scratch/suleymanyildirim22/lr_generation_work]

Every number on the page comes from these files; the descriptions below are the experiment notes.
"""
import argparse
import datetime as dt
import json
import shutil
import statistics as st
from pathlib import Path

from PIL import Image

PAGE = Path(__file__).resolve().parents[1]
OUT = PAGE / 'assets' / 'lrgen'
P = 'DRealSR_x4_S_'
REALISM = ['DRealSR_x4_S_native_metrics_realism', 'DRealSR_x4_S_native_metrics_deresnet64_realism',
           'DRealSR_x4_S_native_metrics_nce_realism', 'DRealSR_x4_S_native_metrics_nce_weights_realism',
           'DRealSR_x4_S_native_metrics_nce01_gan_realism']
HAND = {64: 'hand_crops_nref64_last'}
N = 'nref64_deresnet_nlpips'

SHARED = [['Data', 'DRealSR ×4 train split, camera-balanced sampling, 256 px HR / 64 px LR crops'],
          ['Training', '50k steps, generator body frozen until 10k, bf16'],
          ['Base losses', 'native low-pass MSE 1.0 and native power spectrum (PSD, YCbCr) 0.1, both on the 64 px output'],
          ['Reference', 'a native LR crop from another capture, encoded by a frozen degradation encoder'],
          ['Evaluation', '744 DRealSR ×4 test crops, last checkpoint, seeds 0 and 1, native 64 px output (256 px output area-downscaled by 4)']]

RUNS = [
    dict(key='nref_x5', size=128, emb=None, base=None,
         title='X5 encoder, 128 px reference, grid LPIPS',
         vars=[['Encoder', 'X5 (frozen, 128-d)'], ['Reference size', '128 px'], ['Perceptual loss', 'LPIPS 2.0 on the 256 px grid'],
               ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='The starting point of this series: the reference-conditioned generator with LPIPS computed on the 256 px HR grid.',
         purpose='Baseline for the perceptual-loss and encoder comparisons.',
         findings=['Reference point for Runs 2–6.'],
         verdict='Baseline.'),
    dict(key='nref_x5_nlpips', size=128, emb=None, base='nref_x5',
         title='Native LPIPS instead of grid LPIPS',
         vars=[['Encoder', 'X5'], ['Reference size', '128 px'], ['Perceptual loss', 'LPIPS 2.0 on the native 64 px output'],
               ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='LPIPS moved from the 256 px grid to the native 64 px LR; everything else as Run 1.',
         purpose='Only the native LR is used. Grid LPIPS also constrains the content inside each 4×4 block, which never reaches the LR.',
         findings=['Best LPIPS and DISTS of the X5 128 px runs on both seeds.',
                   'FID ties with the discriminator run (Run 4), 37.5 vs 37.3.'],
         verdict='Native LPIPS is better than grid LPIPS for the native output.'),
    dict(key='nref_x5_nolpips', size=128, emb=None, base='nref_x5',
         title='No LPIPS',
         vars=[['Encoder', 'X5'], ['Reference size', '128 px'], ['Perceptual loss', 'none'], ['Discriminator', 'none'],
               ['Degradation loss', 'none']],
         summary='Run 1 without any perceptual loss: only native low-pass MSE and PSD.',
         purpose='Check whether LPIPS is needed at all.',
         findings=['Clearly worse quality: FID about 47 vs 38.', 'Outputs change most with the noise input.'],
         verdict='LPIPS is needed.'),
    dict(key='nref_x5_gan002', size=128, emb=None, base='nref_x5',
         title='Grid LPIPS + native discriminator',
         vars=[['Encoder', 'X5'], ['Reference size', '128 px'], ['Perceptual loss', 'LPIPS 2.0 on the 256 px grid'],
               ['Discriminator', 'native, weight 0.02, lr 1e-4, from 10k steps'], ['Degradation loss', 'none']],
         summary='Run 1 plus a discriminator on the native 64 px output.',
         purpose='Test whether adversarial training adds camera-specific realism.',
         findings=['Best camera-detail slope of the X5 runs (about 0.96) and more use of the noise input than Runs 1–2.', 'Small PSNR cost; FID level with Run 2.'],
         verdict='The discriminator helps realism without hurting FID.'),
    dict(key='nref64_x5', size=64, emb=None, base='nref_x5',
         title='X5 encoder, 64 px reference',
         vars=[['Encoder', 'X5'], ['Reference size', '64 px (centre of the 128 px native reference)'],
               ['Perceptual loss', 'LPIPS 2.0 on the 256 px grid'], ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='As Run 1, but the encoder sees a 64 px reference (256 px of HR area).',
         purpose='Hand-picked references are small; measure the cost of a 64 px reference.',
         findings=['Quality within seed noise of Run 1; training loss about 5% higher.',
                   'Slightly less of the per-camera detail spread is reproduced.'],
         verdict='64 px references are usable at a small cost.'),
    dict(key='nref_deresnet', size=128, emb=None, base='nref_x5',
         title='DEResNet encoder, 128 px reference',
         vars=[['Encoder', 'DEResNet (S3Diff, frozen, 256-d)'], ['Reference size', '128 px'],
               ['Perceptual loss', 'LPIPS 2.0 on the 256 px grid'], ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='Run 1 with DEResNet as the degradation encoder.',
         purpose='X5 sees the hand-cropped references as almost identical; DEResNet separates them more.',
         findings=['Ties with X5 (Run 1) within seed noise.'],
         verdict='No gain from switching encoders at 128 px.'),
    dict(key='nref64_deresnet', size=64, emb='grid LPIPS', base='nref64_x5',
         title='DEResNet encoder, 64 px reference',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'LPIPS 2.0 on the 256 px grid'],
               ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='Run 5 with DEResNet as the degradation encoder.',
         purpose='The configuration used for the hand-cropped 64 px references.',
         findings=['Ties with X5 64 px (Run 5) within seed noise.',
                   'On the hand crops, outputs follow how blurry the reference is, not its noise or JPEG artefacts.'],
         verdict='Base for the all-native runs.'),
    dict(key=N, size=64, emb='native LPIPS', base='nref64_deresnet',
         title='All-native DEResNet 64 px',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'LPIPS 2.0 on the native 64 px output'],
               ['Discriminator', 'none'], ['Degradation loss', 'none']],
         summary='Every loss at native resolution: Run 7 with LPIPS moved to the native LR.',
         purpose='Repeat the native-LPIPS gain of Run 2 with the DEResNet 64 px setup.',
         findings=['Better than Run 7 on both seeds: LPIPS about 8% lower, FID about 2.8 points better.',
                   'Faint horizontal stripes on flat regions in the textured sweep, which the grid-LPIPS run does not show '
                   '(likely the unconstrained 4×4 block content; not yet confirmed).'],
         verdict='Best quality of all runs.'),
    dict(key=f'{N}_gan002', size=64, emb='native LPIPS + disc', base=N,
         title='All-native + discriminator',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02, lr 1e-4, from 10k steps'], ['Degradation loss', 'none']],
         summary='Run 8 plus the native discriminator.',
         purpose='Add camera-specific realism on top of the best-quality run.',
         findings=['Per-camera detail spread almost fully reproduced (slope about 0.97); more use of the noise input.',
                   'FID unchanged within seed noise.'],
         verdict='Best realism at no quality cost.'),
    dict(key=f'{N}_nce01', size=64, emb='native LPIPS + deg NCE 0.1', base=N,
         title='All-native + degradation loss 0.1',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'], ['Discriminator', 'none'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, τ 0.1, τ_t 0.05, K 4, from 10k steps (1k ramp)']],
         summary='Run 8 plus a contrastive degradation loss. Each HR crop is generated with four references from the batch (same '
                 'noise); for each reference, DEResNet must pick the output it produced. Near-identical references share the target.',
         purpose='In Runs 7–9 the outputs did not follow their reference. With identical content in all four outputs, only the '
                 'degradation can decide which output belongs to which reference.',
         findings=['Outputs follow their reference: own reference ranked closest 54% of the time (chance 25%).',
                   'Costs about 4 FID points against Run 8.',
                   'In the textured sweep the Canon reference makes grass much more contrasty, and a faint woven texture appears '
                   'with the striped-fabric reference (likely reference content leaking in).'],
         verdict='The loss works, at a quality cost.'),
    dict(key=f'{N}_nce005', size=64, emb='native LPIPS + deg NCE 0.05', base=N,
         title='All-native + degradation loss 0.05',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'], ['Discriminator', 'none'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.05']],
         summary='Run 10 with half the degradation-loss weight.',
         purpose='Find a weight with less quality cost.',
         findings=['Own reference ranked closest 38%.', 'FID cost about 3 points against Run 8.'],
         verdict='In between Runs 8 and 10.'),
    dict(key=f'{N}_nce002', size=64, emb='native LPIPS + deg NCE 0.02', base=N,
         title='All-native + degradation loss 0.02',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'], ['Discriminator', 'none'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.02']],
         summary='Run 10 with a fifth of the degradation-loss weight.',
         purpose='Lowest weight of the sweep.',
         findings=['Already costs about 3 FID points, with no matching gain (25%, chance).'],
         verdict='Too weak to help, still costs quality.'),
    dict(key=f'{N}_nce005_gan002', size=64, emb='native LPIPS + deg NCE 0.05 + disc', base=f'{N}_gan002',
         title='All-native + degradation loss 0.05 + discriminator',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'], ['Degradation loss', 'soft-target InfoNCE, weight 0.05']],
         summary='Run 11 plus the native discriminator.',
         purpose='Check whether the discriminator recovers the quality lost to the degradation loss.',
         findings=['Recovers part of the FID cost (about 39.5 vs 40.7 for Run 11).',
                   'Keeps 38% own-reference ranking; per-camera detail slope about 0.97.'],
         verdict='The discriminator offsets part of the degradation-loss cost.'),
    dict(key=f'{N}_nce01_gan002', size=64, emb='native LPIPS + deg NCE 0.1 + disc', base=f'{N}_gan002',
         title='All-native + degradation loss 0.1 + discriminator',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'], ['Degradation loss', 'soft-target InfoNCE, weight 0.1']],
         summary='Run 10 plus the native discriminator.',
         purpose='Keep the strongest reference-following while recovering quality.',
         findings=['Keeps 54% own-reference ranking with FID 40.5 vs 41.8 for Run 10.',
                   'Outputs spread much more like the references do: output-vs-output similarity 0.82 vs 0.95 for Run 9 (references: 0.76).'],
         verdict='Best trade-off for reference-following.'),
]


def load(ev):
    models, rt = {}, None
    for name in REALISM:
        j = json.loads((ev / name / 'realism_stats.json').read_text())
        for run, m in j['models'].items():
            models.setdefault(run, m)
        rt = rt or j
    floor = {k: st.mean(v[k] for v in rt['round_trip_ratio'].values()) for k in ('noise', 'detail', 'hf_power', 'saturation')}
    floor['fid'] = rt['round_trip_native_fid']
    emb = json.loads((ev / HAND[64] / 'embedding_similarity.json').read_text())['DEResNet']
    return models, floor, emb


def seeds(models, key):
    return [models[f'{P}{key}_s{s}'] for s in (0, 1)]


def stat(m, kind, k):
    if kind == 'overall':
        return m['overall'][k]
    if kind == 'ratio':
        return st.mean(v[k] for v in m['ratio'].values())
    return m['camera_spread']['x4'][k]['slope']


def metric_table(title, note, spec, models, run, base, labels, extra=None):
    """spec: (label, kind, key, digits, better) with better in 'min', 'max', 'one' (closest to 1) or None."""
    cur, ref = seeds(models, run['key']), (seeds(models, base['key']) if base else None)
    headers = ['Metric', 'Seed 0', 'Seed 1', 'Mean']
    if ref:
        headers += [f'{labels[base["key"]]} (mean)', 'Change']
    if extra:
        headers.append(extra[0])
    rows = []
    for label, kind, key, d, better in spec:
        a, b = (stat(m, kind, key) for m in cur)
        mean = (a + b) / 2
        row = [label, f'{a:.{d}f}', f'{b:.{d}f}', f'<strong>{mean:.{d}f}</strong>']
        if ref:
            rm = st.mean(stat(m, kind, key) for m in ref)
            delta = mean - rm
            good = None if better is None else {'min': delta < 0, 'max': delta > 0, 'one': abs(mean - 1) < abs(rm - 1)}[better]
            cls = '' if good is None or abs(delta) < 10 ** -d else (' class="up"' if good else ' class="down"')
            row += [f'{rm:.{d}f}', f'<span{cls}>{delta:+.{d}f}</span>']
        if extra:
            row.append(extra[1].get(key, '—'))
        rows.append(row)
    return dict(title=title, headers=headers, rows=rows, note=note)


def quality(models, run, base, labels):
    spec = [('LPIPS ↓', 'overall', 'native_lpips', 4, 'min'), ('DISTS ↓', 'overall', 'native_dists', 4, 'min'),
            ('FID ↓', 'overall', 'native_fid', 2, 'min'), ('PSNR (dB)', 'overall', 'native_psnr', 2, None)]
    return metric_table('Quantitative metrics', 'Native 64 px output against the real LR of the same crop. Lower LPIPS, DISTS and '
                        'FID is better; FID varies by up to ~4 points between seeds. PSNR rewards smooth outputs and is not used to '
                        'select runs. Change: green = better than the comparison run, red = worse.', spec, models, run, base, labels)


def realism(models, run, base, labels, floor):
    spec = [('Noise ratio', 'ratio', 'noise', 3, 'one'), ('Detail ratio', 'ratio', 'detail', 3, 'one'),
            ('HF power ratio', 'ratio', 'hf_power', 3, 'one'), ('Saturation ratio', 'ratio', 'saturation', 3, 'one'),
            ('Noise camera slope', 'slope', 'noise', 3, 'one'), ('Detail camera slope', 'slope', 'detail', 3, 'one'),
            ('HF camera slope', 'slope', 'hf_power', 3, 'one'), ('Noise use (z_std ÷ real noise) ↑', 'overall', 'z_std_over_real_noise', 3, 'max'),
            ('Diversity (LPIPS between noise draws) ↑', 'overall', 'diversity_lpips', 4, 'max')]
    rt = {k: f'{floor[k]:.2f}' for k in ('noise', 'detail', 'hf_power', 'saturation')}
    return metric_table('Realism statistics', 'Ratios are generated ÷ real, averaged over the six cameras (1 = real). The last column '
                        'is the score of the real LR itself sent through the same measurement, so it is the best reachable. Camera '
                        'slopes: how much of the real difference between cameras is reproduced (1 = all). Noise use: how much the '
                        'output changes with the noise input, relative to the real noise level. Change is green when closer to 1 '
                        '(ratios and slopes) or higher (noise use, diversity).',
                        spec, models, run, base, labels, extra=('Real LR round trip', rt))


def matching(emb, runs, labels):
    def off(m):
        return st.mean(m[i][j] for i in range(4) for j in range(4) if i != j)
    rows = [['Hand-cropped references (target)', f'{off(emb["ref_ref"]):.3f}', '—', '—', '—', '—']]
    for r in runs:
        d = emb['models'][r['emb']]
        own = st.mean(d['gen_ref'][i][i] for i in range(4))
        oth = off(d['gen_ref'])
        rows.append([labels[r['key']], f'{off(d["gen_gen"]):.3f}', f'{own:.3f}', f'{oth:.3f}', f'{own - oth:+.3f}',
                     f'{d["top1_matched_ref"] * 100:.0f}%'])
    rows[1][0] = f'<strong>{rows[1][0]}</strong>'
    return dict(title='Degradation matching (hand-cropped references, DEResNet cosine similarity)',
                headers=['', 'Output vs output', 'Output vs own reference', 'Output vs other references', 'Gap ↑', 'Own ranked closest ↑'],
                rows=rows, note='3 HR crops × 4 references × 2 seeds, native 64 px, DEResNet embeddings. Output vs output should be '
                'near the reference-vs-reference value (the target row). Own ranked closest: chance is 25%; lowdeg2 and highdeg2 '
                'are near twins (0.98), so 100% is not reachable.')


def save(src, dst, width):
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert('RGB')
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, quality=85, optimize=True)
    return str(dst.relative_to(PAGE))


def visuals(ev, run, rid):
    key = run['key']
    out = {'sweep': [], 'hand': []}
    sw = ev / f'{P}{key}_s0_reference_sweep_textured'
    if (sw / 'manifest.json').exists():
        man = json.loads((sw / 'manifest.json').read_text())
        refs = ', '.join(r['capture_id'] for r in man['references'])
        out['sweep_note'] = (f'Seed 0. Six DRealSR ×4 test photos (the most textured of 16 random crops each). Each figure keeps the HR '
                             f'crop and the noise fixed and changes only the reference: its own real LR, then cross-capture references '
                             f'({refs}). The same photos and references are used for every run, so tabs can be compared directly.')
        for i, s in enumerate(man['sources']):
            f = sw / f'HR_{i:02d}_x4' / 'comparison.png'
            if f.exists():
                out['sweep'].append(dict(src=save(f, OUT / 'img' / rid / f'sweep_{i}.jpg', 900), caption=s['image_id'].replace('_x4', '')))
    else:
        out['sweep_note'] = 'The textured reference sweep has not been run for this checkpoint yet.'
    out['hand_note'] = ('Rows: the three hand-cropped HR inputs (fixed noise per row). Columns: the four hand-cropped 64 px references '
                        '(lowdeg, lowdeg2, highdeg, highdeg2). Cells: the generated native 64 px LR, nearest-neighbour zoom.')
    if run['size'] == 128:
        out['hand_note'] = ('Not run: this model was trained with 128 px references, and the hand-cropped references are 64 px. '
                            'See the 64 px runs (Run 5 and Runs 7–14).')
        return out
    for s in (0, 1):
        f = ev / HAND[64] / f'grid_{P}{key}_s{s}.png'
        if f.exists():
            out['hand'].append(dict(src=save(f, OUT / 'img' / rid / f'hand_s{s}.jpg', 1200), caption=f'Seed {s}'))
    if not out['hand']:
        out['hand_note'] = 'The hand-crop sweep is not available for this run.'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', default='/scratch/suleymanyildirim22/lr_generation_work')
    ev = Path(ap.parse_args().work_dir) / 'evaluations'
    models, floor, emb = load(ev)
    ids = {r['key']: f'run{i + 1}' for i, r in enumerate(RUNS)}
    labels = {k: f'Run {v[3:]}' for k, v in ids.items()}
    byk = {r['key']: r for r in RUNS}
    shutil.rmtree(OUT / 'img', ignore_errors=True)
    runs = []
    for r in RUNS:
        base = byk.get(r['base'])
        tables = [quality(models, r, base, labels), realism(models, r, base, labels, floor)]
        if r['emb']:
            tables.append(matching(emb, [r] + ([base] if base and base['emb'] else []), labels))
        runs.append(dict(id=ids[r['key']], name=labels[r['key']], title=r['title'], summary=r['summary'], purpose=r['purpose'],
                         checkpoint=f'{P}{r["key"]}_s0, _s1', variables=r['vars'],
                         compared_to=(f'{labels[base["key"]]} · {base["title"]}' if base else 'none (baseline)'),
                         findings=r['findings'], verdict=r['verdict'], tables=tables, visuals=visuals(ev, r, ids[r['key']])))

    def mean(r, k):
        return st.mean(m['overall'][k] for m in seeds(models, r['key']))

    def short(v):
        return {'LPIPS 2.0 on the 256 px grid': 'grid LPIPS', 'LPIPS 2.0 on the native 64 px output': 'native LPIPS',
                'native LPIPS 2.0': 'native LPIPS'}.get(v, v)
    overview = dict(title='All runs at a glance (mean of both seeds)',
                    headers=['Run', 'Encoder', 'Reference', 'Perceptual loss', 'Discriminator', 'Degradation loss', 'LPIPS ↓', 'DISTS ↓',
                             'FID ↓', 'Own ranked closest ↑'],
                    rows=[[f'<a href="#{ids[r["key"]]}">{labels[r["key"]]}</a>', r['vars'][0][1].split(' (')[0], r['vars'][1][1].split(' (')[0],
                           short(r['vars'][2][1]), 'yes' if r['vars'][3][1] != 'none' else '—',
                           '—' if r['vars'][4][1] == 'none' else r['vars'][4][1].split('weight ')[1].split(',')[0],
                           f'{mean(r, "native_lpips"):.4f}', f'{mean(r, "native_dists"):.4f}', f'{mean(r, "native_fid"):.1f}',
                           f'{emb["models"][r["emb"]]["top1_matched_ref"] * 100:.0f}%' if r['emb'] else '—'] for r in RUNS],
                    note='Own ranked closest is measured on the hand-cropped references, for the 64 px DEResNet runs only (chance 25%).')
    glossary = [['LPIPS / DISTS', 'perceptual difference between the generated and the real native LR of the same crop (lower is better)'],
                ['FID', f'distribution distance between generated and real native LRs (lower is better); the real LR round trip scores {floor["fid"]:.2f}'],
                ['PSNR', 'pixel agreement with the real LR; rewards smooth outputs, not used for selection'],
                ['Noise / detail / HF power / saturation ratio', 'the statistic of generated LRs divided by that of real LRs, per camera, then averaged (1 = real)'],
                ['Camera slope', 'regression slope of generated per-camera values on real per-camera values (1 = cameras differ as much as in real data)'],
                ['Noise use', 'standard deviation of the output across noise draws on flat regions, divided by the real noise level'],
                ['Own ranked closest', 'share of outputs whose DEResNet embedding is closest to the reference they were generated with'],
                ['Native', 'the 256 px generator output area-downscaled by 4 to the 64 px LR; all numbers on this page are native']]
    data = dict(generated=dt.date.today().isoformat(), shared=SHARED, overview=overview, glossary=glossary, runs=runs)
    (OUT / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    print(f'wrote {OUT / "data.json"}: {len(runs)} runs')
    for r in runs:
        print(r['id'], 'sweep', len(r['visuals']['sweep']), 'hand', len(r['visuals']['hand']))


if __name__ == '__main__':
    main()
