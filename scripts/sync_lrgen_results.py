#!/usr/bin/env python3
"""Build the HR->LR generator (lr_generation) experiment log for lrgen.html.

Reads the native-resolution realism reports, the hand-crop embedding report and the qualitative grids written by
lr_generation (hr2lr.realism_stats, hr2lr.reference_sweep and the hand-crop sweep) from WORK_DIR, and writes
assets/lrgen/data.json plus JPEG copies of the figures under assets/lrgen/img/.

    python scripts/sync_lrgen_results.py [--work-dir /scratch/suleymanyildirim22/lr_generation_work]

Every number on the page comes from these files; the descriptions below are the experiment notes.
"""
import argparse
import datetime as dt
import json
import statistics as st
from pathlib import Path

from PIL import Image

PAGE = Path(__file__).resolve().parents[1]
OUT = PAGE / 'assets' / 'lrgen'
P = 'DRealSR_x4_S_'
REALISM = ['DRealSR_x4_S_native_metrics_realism', 'DRealSR_x4_S_native_metrics_deresnet64_realism',
           'DRealSR_x4_S_native_metrics_nce_realism', 'DRealSR_x4_S_native_metrics_nce_weights_realism',
           'DRealSR_x4_S_native_metrics_nce01_gan_realism']


def load(ev):
    models, rt = {}, None
    for name in REALISM:
        j = json.loads((ev / name / 'realism_stats.json').read_text())
        for run, m in j['models'].items():
            models.setdefault(run, m)
        rt = rt or j
    floor = {k: st.mean(v[k] for v in rt['round_trip_ratio'].values()) for k in ('noise', 'detail', 'hf_power', 'saturation')}
    floor['fid'] = rt['round_trip_native_fid']
    emb = json.loads((ev / 'hand_crops_nref64_last' / 'embedding_similarity.json').read_text())
    return models, floor, emb


def pair(models, run, key):
    return [models[f'{P}{run}_s{s}']['overall'][key] for s in (0, 1)]


def best_marks(values, mode):
    """Index of the best row: mode 'min', 'max' or 'one' (closest to 1)."""
    score = {'min': lambda v: v, 'max': lambda v: -v, 'one': lambda v: abs(v - 1)}[mode]
    return min(range(len(values)), key=lambda i: score(values[i]))


def quality_table(models, arms, note=''):
    keys = [('native_lpips', 'LPIPS ↓', 4, 'min'), ('native_dists', 'DISTS ↓', 4, 'min'),
            ('native_fid', 'FID ↓', 1, 'min'), ('native_psnr', 'PSNR', 2, None)]
    vals = {k: [pair(models, run, k) for _, run in arms] for k, *_ in keys}
    best = {k: best_marks([st.mean(v) for v in vals[k]], mode) for k, _, _, mode in keys if mode}
    rows = []
    for i, (label, _) in enumerate(arms):
        row = [label]
        for k, _, d, _ in keys:
            cell = '/'.join(f'{x:.{d}f}' for x in vals[k][i])
            row.append(f'<strong>{cell}</strong>' if best.get(k) == i else cell)
        rows.append(row)
    return dict(title='Quality on the native 64 px LR (seed 0 / seed 1)', headers=['Experiment'] + [h for _, h, _, _ in keys],
                rows=rows, note=note or 'Lower is better for LPIPS, DISTS and FID; the best mean of the two seeds is bold. '
                'FID differs by up to ~4 points between seeds. PSNR is not used for selection.')


def realism_table(models, arms, floor):
    def r(run, k):
        return st.mean(st.mean(v[k] for v in models[f'{P}{run}_s{s}']['ratio'].values()) for s in (0, 1))

    def sl(run, k):
        return st.mean(models[f'{P}{run}_s{s}']['camera_spread']['x4'][k]['slope'] for s in (0, 1))

    def o(run, k):
        return st.mean(pair(models, run, k))
    cols = [('Noise', lambda a: r(a, 'noise'), 2, None), ('Detail', lambda a: r(a, 'detail'), 2, None),
            ('HF power', lambda a: r(a, 'hf_power'), 2, None), ('Saturation', lambda a: r(a, 'saturation'), 3, None),
            ('Noise slope', lambda a: sl(a, 'noise'), 2, 'one'), ('Detail slope', lambda a: sl(a, 'detail'), 2, 'one'),
            ('HF slope', lambda a: sl(a, 'hf_power'), 2, 'one'), ('z_std', lambda a: o(a, 'z_std_over_real_noise'), 2, 'max'),
            ('Diversity', lambda a: o(a, 'diversity_lpips'), 4, 'max')]
    vals = [[f(run) for _, f, _, _ in cols] for _, run in arms]
    best = {j: best_marks([v[j] for v in vals], mode) for j, (_, _, _, mode) in enumerate(cols) if mode}
    rows = [[label] + [(f'<strong>{v:.{cols[j][2]}f}</strong>' if best.get(j) == i else f'{v:.{cols[j][2]}f}')
                       for j, v in enumerate(vals[i])] for i, (label, _) in enumerate(arms)]
    note = (f'Mean of both seeds. Noise / detail / HF power / saturation are generated ÷ real (1 = real); a real LR round-tripped '
            f'through the same measurement reaches only {floor["noise"]:.2f} / {floor["detail"]:.2f} / {floor["hf_power"]:.2f} / '
            f'{floor["saturation"]:.2f}. Slopes: how much of the real per-camera spread is reproduced (1 = all). z_std: output change '
            'per unit of input noise, over the real noise level. Diversity: LPIPS between outputs for different noise.')
    return dict(title='Realism analysis', headers=['Experiment'] + [c for c, *_ in cols], rows=rows, note=note)


def matching_table(emb, arms, encoder='DEResNet'):
    e = emb[encoder]

    def off(m):
        return st.mean(m[i][j] for i in range(4) for j in range(4) if i != j)
    vals = []
    for label, key in arms:
        d = e['models'][key]
        own = st.mean(d['gen_ref'][i][i] for i in range(4))
        other = off(d['gen_ref'])
        vals.append((label, off(d['gen_gen']), own, other, own - other, d['top1_matched_ref']))
    b = best_marks([v[5] for v in vals], 'max')
    rows = [[lab, f'{gg:.3f}', f'{own:.3f}', f'{oth:.3f}', f'{gap:+.3f}', (f'<strong>{t * 100:.0f}%</strong>' if i == b else f'{t * 100:.0f}%')]
            for i, (lab, gg, own, oth, gap, t) in enumerate(vals)]
    return dict(title=f'Degradation matching ({encoder} on the hand-cropped references)',
                headers=['Experiment', 'Output vs output', 'Own reference', 'Other references', 'Gap ↑', 'Own ranked closest ↑'],
                rows=rows, note=f'3 HR crops × 4 references × 2 seeds, native 64 px. The four references are {off(e["ref_ref"]):.3f} similar '
                'to each other, so outputs should spread about as much. "Own ranked closest" is chance at 25%; lowdeg2 and highdeg2 '
                'are near twins (0.98), so 100% is not reachable.')


def matrix_table(title, m, names, note=''):
    return dict(title=title, headers=[''] + names, rows=[[names[i]] + [f'{v:.3f}' for v in row] for i, row in enumerate(m)], note=note)


def figure(ev, src, name, caption):
    OUT.joinpath('img').mkdir(parents=True, exist_ok=True)
    im = Image.open(ev / src).convert('RGB')
    if im.width > 1500:
        im = im.resize((1500, round(im.height * 1500 / im.width)), Image.LANCZOS)
    im.save(OUT / 'img' / f'{name}.jpg', quality=88, optimize=True)
    return dict(src=f'assets/lrgen/img/{name}.jpg', caption=caption)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', default='/scratch/suleymanyildirim22/lr_generation_work')
    ev = Path(ap.parse_args().work_dir) / 'evaluations'
    models, floor, emb = load(ev)
    hc = 'hand_crops_nref64_last'
    refs = ['lowdeg', 'lowdeg2', 'highdeg', 'highdeg2']
    n = 'nref64_deresnet_nlpips'
    tex = lambda run, hr: f'{P}{run}_reference_sweep_textured/HR_{hr:02d}_x4/comparison.png'

    experiments = [
        dict(id='lpips', date='2026-11-07', title='Which perceptual loss: grid, native or none?',
             summary='The same X5-conditioned generator trained with LPIPS on the 256 px HR grid, with LPIPS on the native 64 px LR, '
                     'without LPIPS, and with grid LPIPS plus a discriminator.',
             purpose='Grid metrics had rejected native LPIPS, but only the native LR output is used. Losses that act only at native size '
                     'leave the content inside each 4×4 block free, which grid metrics punish even though it never reaches the LR.',
             setup=[['Data', 'DRealSR ×4 only, camera-balanced'], ['Encoder', 'X5 (frozen, 128-d), 128 px native reference'],
                    ['Base losses', 'native low-pass MSE 1.0, native PSD (YCbCr) 0.1'], ['Steps', '50k, body unfrozen at 10k'],
                    ['Evaluation', '744 test crops, last checkpoint, native 64 px output']],
             tables=[quality_table(models, [('Grid LPIPS 2.0', 'nref_x5'), ('Native LPIPS 2.0', 'nref_x5_nlpips'),
                                            ('No LPIPS', 'nref_x5_nolpips'), ('Grid LPIPS + discriminator 0.02', 'nref_x5_gan002')]),
                     realism_table(models, [('Grid LPIPS 2.0', 'nref_x5'), ('Native LPIPS 2.0', 'nref_x5_nlpips'),
                                            ('No LPIPS', 'nref_x5_nolpips'), ('Grid LPIPS + discriminator 0.02', 'nref_x5_gan002')], floor)],
             findings=['Native LPIPS is best on LPIPS and DISTS on both seeds; on FID it ties with the discriminator run (37.5 vs 37.4).',
                       'Removing LPIPS is clearly worse (FID about 47 vs 38), although its outputs vary most with the noise input.',
                       'The discriminator gives the best camera-detail slope and uses the noise input most, at a small PSNR cost.'],
             verdict='Use native LPIPS. Grid metrics were misleading for native-only losses.'),
        dict(id='refsize', date='2026-11-07', title='Reference size: 64 px vs 128 px (X5)',
             summary='The X5 encoder sees the centre 64 px of the 128 px native reference instead of all of it.',
             purpose='Hand-picked references are small; check how much is lost when the reference is 64 px (256 px of HR area).',
             setup=[['Encoder', 'X5, native reference, ref_native_size 64 vs 128'], ['Losses', 'grid LPIPS 2.0, native low-pass MSE, native PSD'],
                    ['Data', 'DRealSR ×4']],
             tables=[quality_table(models, [('128 px reference', 'nref_x5'), ('64 px reference', 'nref64_x5')]),
                     realism_table(models, [('128 px reference', 'nref_x5'), ('64 px reference', 'nref64_x5')], floor)],
             findings=['Quality is within seed noise; training loss was about 5% higher with 64 px.',
                       'The 64 px reference reproduces slightly less of the per-camera detail spread.'],
             verdict='64 px references are usable at a small cost.'),
        dict(id='encoder', date='2026-11-07', title='Degradation encoder: DEResNet vs X5',
             summary='The frozen DEResNet (S3Diff, 256-d) replaces X5 as the degradation encoder, at 128 px and 64 px references.',
             purpose='X5 sees the hand-cropped references as almost identical; DEResNet separates them more. Does the generator benefit?',
             setup=[['Losses', 'grid LPIPS 2.0, native low-pass MSE, native PSD'], ['Data', 'DRealSR ×4']],
             tables=[quality_table(models, [('X5, 128 px', 'nref_x5'), ('X5, 64 px', 'nref64_x5'), ('DEResNet, 128 px', 'nref_deresnet'),
                                            ('DEResNet, 64 px', 'nref64_deresnet')]),
                     realism_table(models, [('X5, 128 px', 'nref_x5'), ('X5, 64 px', 'nref64_x5'), ('DEResNet, 128 px', 'nref_deresnet'),
                                            ('DEResNet, 64 px', 'nref64_deresnet')], floor)],
             findings=['DEResNet and X5 tie within seed noise at both reference sizes.',
                       'Neither encoder makes the generator transfer noise or JPEG; outputs follow mainly blur.'],
             verdict='No reason to switch encoders on these metrics alone.'),
        dict(id='handcrops', date='2026-11-07', title='Hand-cropped references: what the encoders see',
             summary='Four hand-cropped 64 px LR references (two labelled low degradation, two high) applied to three hand-cropped HR '
                     'crops, with fixed noise per HR.',
             purpose='Check whether the generator follows the user-labelled degradation, and whether the encoders can tell the references apart.',
             setup=[['References', ', '.join(refs) + ' (64×64)'], ['HR inputs', 'hr, hr2, hr3 (256×256)'], ['Models', 'X5 64 px, DEResNet 64 px']],
             tables=[matrix_table('X5 cosine similarity between the references', emb['X5']['ref_ref'], refs, 'X5 sees all four as almost the same.'),
                     matrix_table('DEResNet cosine similarity between the references', emb['DEResNet']['ref_ref'], refs,
                                  'DEResNet separates them more, but part of the spread is image content, not degradation.')],
             figures=[figure(ev, f'{hc}/grid_{P}nref64_x5_s0.png', 'handcrop_x5_64', 'X5 64 px model on the hand crops (seed 0)'),
                      figure(ev, f'{hc}/grid_{P}nref64_deresnet_s0.png', 'handcrop_deresnet_64', 'DEResNet 64 px model on the hand crops (seed 0)')],
             findings=['Outputs follow how blurry the reference is (lowdeg2 softest, highdeg sharpest), not the user labels.',
                       'Noise and JPEG artefacts of the references are not transferred.',
                       'A downscaled HR with no degradation scores up to 0.93 against a reference of similar content, so DEResNet similarity mixes content and degradation.'],
             verdict='The encoders separate references mainly by blur; the generator responds to that one axis.'),
        dict(id='allnative', date='2026-11-08', title='All-native DEResNet 64 px, with and without a discriminator',
             summary='Every loss at native resolution (LPIPS moved to the native LR) with the DEResNet 64 px reference, then a native '
                     'discriminator added on top.',
             purpose='Repeat the native-LPIPS gain with DEResNet, and test whether a discriminator adds realism.',
             setup=[['Encoder', 'DEResNet, 64 px native reference'], ['Losses', 'native low-pass MSE 1.0, native LPIPS 2.0, native PSD 0.1'],
                    ['Discriminator', 'native, weight 0.02, lr 1e-4, from 10k steps']],
             tables=[quality_table(models, [('Grid LPIPS (reference)', 'nref64_deresnet'), ('All-native', n), ('All-native + discriminator', f'{n}_gan002')]),
                     realism_table(models, [('Grid LPIPS (reference)', 'nref64_deresnet'), ('All-native', n), ('All-native + discriminator', f'{n}_gan002')], floor),
                     matching_table(emb, [('Grid LPIPS (reference)', 'grid LPIPS'), ('All-native', 'native LPIPS'), ('All-native + discriminator', 'native LPIPS + disc')])],
             figures=[figure(ev, f'{hc}/grid_{P}{n}_s0.png', 'handcrop_allnative', 'All-native on the hand crops (seed 0)'),
                      figure(ev, f'{hc}/grid_{P}{n}_gan002_s0.png', 'handcrop_allnative_gan', 'All-native + discriminator on the hand crops (seed 0)')],
             findings=['All-native beats grid LPIPS on both seeds: LPIPS −7%, FID about 2.5 points better.',
                       'The discriminator matches per-camera detail much better (slope about 0.97) and uses the noise input more, with FID unchanged.',
                       'Neither makes outputs follow the reference: own reference ranked closest only 25% of the time (chance).'],
             verdict='All-native is the best on quality; adding the discriminator is the best on realism.'),
        dict(id='degnce', date='2026-11-08', title='Contrastive degradation loss (soft-target InfoNCE)',
             summary='Each HR crop is generated with four references from the batch (same noise). For each reference, DEResNet must pick the '
                     'output it produced among those four; near-identical references share the target. Tried at weights 0.02, 0.05 and 0.1, '
                     'and at 0.05 and 0.1 with the discriminator.',
             purpose='Outputs did not follow their reference. Comparing outputs that share the same HR content means content cannot decide '
                     'the answer, only degradation can.',
             setup=[['Base', 'all-native DEResNet 64 px'], ['Loss', 'w_deg_nce 0.02 / 0.05 / 0.1, τ = 0.1, soft-target τ_t = 0.05, K = 4'],
                    ['Schedule', 'from 10k steps, ramped over 1k'], ['Discriminator', 'native, weight 0.02 (where marked)'],
                    ['Code', 'hr2lr/losses.py deg_nce_soft, hr2lr/train.py _deg_nce']],
             tables=[quality_table(models, [('All-native', n), ('+ discriminator', f'{n}_gan002'), ('+ deg 0.02', f'{n}_nce002'),
                                            ('+ deg 0.05', f'{n}_nce005'), ('+ deg 0.1', f'{n}_nce01'), ('+ deg 0.05 + disc', f'{n}_nce005_gan002'),
                                            ('+ deg 0.1 + disc', f'{n}_nce01_gan002')]),
                     realism_table(models, [('All-native', n), ('+ discriminator', f'{n}_gan002'), ('+ deg 0.02', f'{n}_nce002'),
                                            ('+ deg 0.05', f'{n}_nce005'), ('+ deg 0.1', f'{n}_nce01'), ('+ deg 0.05 + disc', f'{n}_nce005_gan002'),
                                            ('+ deg 0.1 + disc', f'{n}_nce01_gan002')], floor),
                     matching_table(emb, [('All-native', 'native LPIPS'), ('+ discriminator', 'native LPIPS + disc'),
                                          ('+ deg 0.02', 'native LPIPS + deg NCE 0.02'), ('+ deg 0.05', 'native LPIPS + deg NCE 0.05'),
                                          ('+ deg 0.1', 'native LPIPS + deg NCE 0.1'), ('+ deg 0.05 + disc', 'native LPIPS + deg NCE 0.05 + disc'),
                                          ('+ deg 0.1 + disc', 'native LPIPS + deg NCE 0.1 + disc')])],
             figures=[figure(ev, f'{hc}/grid_{P}{n}_nce01_s0.png', 'handcrop_nce01', 'Degradation loss 0.1 on the hand crops (seed 0)'),
                      figure(ev, f'{hc}/grid_{P}{n}_nce01_gan002_s0.png', 'handcrop_nce01_gan', 'Degradation loss 0.1 + discriminator on the hand crops (seed 0)')],
             findings=['The loss is the only change that makes outputs follow their reference: up to 54% own-reference ranking vs 25%, with output spread close to the references.',
                       'It costs quality at every weight; 0.02 already costs about 3 FID points with no matching gain.',
                       'The discriminator recovers part of the cost and restores noise-input use: 0.1 + disc keeps 54% matching with FID 40.5 vs 41.8 alone.'],
             verdict='A trade-off: all-native for quality, degradation loss 0.1 + discriminator for reference-following.'),
        dict(id='sweep', date='2026-11-08', title='Textured reference sweep and a stripe artefact',
             summary='The DRealSR reference sweep now keeps the least flat of 16 random crops per photo (--texture-candidates 16, protocol '
                     'v2_textured), over 6 test photos; the old sweep often landed on blank walls.',
             purpose='Look at how outputs change with the reference on textured content, and find where fine stripes come from.',
             setup=[['Sweep', 'hr2lr.reference_sweep, 6 sources, 2 cross-capture references + the matched one'],
                    ['Models', 'all-native, + discriminator, + deg 0.1, + deg 0.1 + discriminator (seed 0)']],
             figures=[figure(ev, tex(f'{n}_nce01_gan002_s0', 5), 'sweep_grass_nce01_gan', 'Grass, degradation loss 0.1 + discriminator'),
                      figure(ev, tex(f'{n}_s0', 5), 'sweep_grass_allnative', 'Grass, all-native'),
                      figure(ev, tex(f'{n}_nce01_s0', 0), 'sweep_wall_nce01', 'Flat wall, degradation loss 0.1'),
                      figure(ev, tex(f'{n}_s0', 0), 'sweep_wall_allnative', 'Flat wall, all-native')],
             findings=['Faint horizontal stripes on flat regions appear in all-native too, not only with the degradation loss or the discriminator; the grid-LPIPS model was smooth.',
                       'Likely cause (not yet confirmed): with every loss at native size, nothing constrains the pattern inside each 4×4 block.',
                       'With the degradation loss, the Canon reference makes the grass much more contrasty; with the striped-fabric reference a faint woven texture appears, which looks like reference content leaking in.'],
             verdict='Next: test the stripes with a small grid-level loss or a blurred downscale, and check content leakage with same-scene ×2/×3/×4 references.'),
    ]
    glossary = [['LPIPS / DISTS', 'perceptual difference between generated and real native LR of the same crop (lower is better)'],
                ['FID', f'distribution distance of generated vs real native LRs (lower is better; round-trip floor {floor["fid"]:.2f})'],
                ['PSNR', 'pixel agreement with the real LR; rewards smooth outputs, not used for selection'],
                ['Native', 'the 256 px generator output area-downscaled by 4 to the 64 px LR; all numbers here are native']]
    data = dict(generated=dt.date.today().isoformat(), glossary=glossary, experiments=experiments)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    print(f'wrote {OUT / "data.json"} with {len(experiments)} experiments')


if __name__ == '__main__':
    main()
