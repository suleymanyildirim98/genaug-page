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
           'DRealSR_x4_S_native_metrics_nce01_gan_realism', 'DRealSR_x4_S_native_metrics_x5nce_s0_realism',
           'DRealSR_x4_S_native_metrics_x5nce_s1_realism'] + [f'DRealSR_x4_S_native_metrics_{t}_s{s}_realism'
                                                            for t in ('ownonly', 'capmean', 'hrneg', 'hrneg005', 'delta', 'delta02', 'delta05', 'delta02b8', 'dsynrec', 'synrec', 'dsynnce') for s in (0, 1)]
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
                   'The native sweep is clean on flat regions. (Stripes seen in an earlier sweep were in the 256 px output inside 4×4 blocks, which average out in the native LR.)'],
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
                   'In the native sweep the Canon reference makes grass and edges clearly more contrasty, and the striped-facade reference (panasonic_187) puts a faint repeating pattern on a flat wall: reference texture leaks into the native LR.'],
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
                   'Outputs spread much more like the references do: output-vs-output similarity 0.82 vs 0.95 for Run 9 (references: 0.76).',
                   'The texture leak of Run 10 remains in the native sweep but is weaker and blotchier with the discriminator.'],
         verdict='Best trade-off for reference-following.'),
    dict(key='nref64_x5_nlpips_nce01_gan002', size=64, emb='X5 + native LPIPS + deg NCE 0.1 (X5 judge) + disc', base=f'{N}_nce01_gan002',
         title='Run 14 with the X5 encoder (conditioning and degradation loss)',
         vars=[['Encoder', 'X5 (frozen, 128-d), also judges the degradation loss'], ['Reference size', '64 px'],
               ['Perceptual loss', 'native LPIPS 2.0'], ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, on L2-normalised X5 embeddings']],
         summary='Run 14 with DEResNet replaced by X5 everywhere: X5 encodes the reference and X5 embeddings of the output and '
                 'the references define the degradation loss.',
         purpose='Runs 10 and 14 copy reference texture into the output (a lattice on flat walls and doubled edges with the '
                 'panasonic_187 reference). Test whether this comes from DEResNet mixing content into its degradation embedding.',
         findings=['The texture leak is gone. Fine texture on flat regions with the panasonic_187 reference (high-pass std, '
                   'seed 0 sweep, real LR 2.41 for comparison): 3.13, the same as Run 8 without a degradation loss (3.08); Run 14: 3.62.',
                   'Better quality than Run 14 on both seeds: LPIPS 0.0329 vs 0.0385, FID 38.5 vs 40.5.',
                   'Reference following is weak: outputs change little between references, much like Run 8. Judged by '
                   'DEResNet the own reference is ranked closest 25% of the time (chance), vs 54% for Run 14; judged by '
                   'X5 it is 50%, vs 33–42% for the other runs.',
                   'X5 sees the four hand-cropped references as nearly identical (cosine 0.98), so the loss has little to separate.'],
         verdict='Changing the encoder removes the artifacts, but mostly because the X5 loss asks for very little reference '
                 'dependence. It does not give artifact-free reference following.'),
    dict(key=f'{N}_nce01_gan002_ownonly', size=64, emb='deg NCE 0.1 + disc, own-only target', base=f'{N}_nce01_gan002',
         title='Run 14 with an own-reference-only degradation loss',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, own-reference question only, scored against the target LR']],
         summary='As Run 14, but the degradation loss asks only "which of the 4 outputs was made with the own reference?", and '
                 'scores it against the DEResNet embedding of the target LR instead of the reference crop. The 3 cross-reference '
                 'outputs are only distractors.',
         purpose='The target LR shares the reference\'s capture but not its content, so copying reference texture should no '
                 'longer raise the score.',
         findings=['The texture leak is about halved: flat-region fine texture with panasonic_187 3.37 (Run 8: 3.08, Run 14: 3.62), '
                   'but a faint lattice is still visible on the flat wall.',
                   'Reference following drops from 54% to 42% own ranked closest (chance 25%); output-vs-output similarity 0.87 '
                   '(Run 14: 0.82, references: 0.76).',
                   'Quality between Run 8 and Run 14: LPIPS 0.0360, FID 40.0.'],
         verdict='Partial fix: less leak, less reference following. The leak is not fully removed.'),
    dict(key=f'{N}_nce01_gan002_capmean', size=64, emb='deg NCE 0.1 + disc, capture-mean target', base=f'{N}_nce01_gan002',
         title='Run 14 with capture-mean degradation targets',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, each reference scored against the mean embedding of its capture']],
         summary='As Run 14, but each reference\'s target is the mean DEResNet embedding of about 260 crops from the other, '
                 'non-overlapping patches of its capture, instead of the reference crop.',
         purpose='Averaging over many crops of the same photo keeps the degradation and removes the specific texture that leaked.',
         findings=['The averaged targets barely separate cameras (cosine 0.93 same camera vs 0.92 different camera), so the loss '
                   'asks for little.',
                   'Reference following is lost: 21% own ranked closest (chance 25%); output-vs-output similarity 0.96, like Run 8.',
                   'Flat-region fine texture with panasonic_187 3.36: still above Run 8 (3.08), with a faint diagonal pattern on the wall.',
                   'LPIPS 0.0350, FID 39.1.'],
         verdict='Does not work: averaging removes the degradation differences along with the content.'),
    dict(key=f'{N}_nce01_gan002_hrneg', size=64, emb='deg NCE 0.1 + disc, HR negative', base=f'{N}_nce01_gan002',
         title='Run 14 with the reference HR as a negative in the degradation loss',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, plus the reference\'s own HR (downscaled) as an extra negative, '
                'weighted by how far the reference LR is from its HR']],
         summary='As Run 14, but each output must also be farther from the DEResNet embedding of the reference\'s clean HR '
                 '(area-downscaled to native) than from its reference LR. The HR negative is weighted by 1 − cos(ref LR, ref HR) '
                 '/ 0.1, so it is off for near-clean references.',
         purpose='The HR and LR of the reference share content and differ only in degradation, so pushing away from the HR '
                 'should reward the degradation and not the content.',
         findings=['The leak gets worse: flat-region fine texture with panasonic_187 4.29 (Run 14: 3.62, Run 8: 3.08, real LR: 2.41). '
                   'The montage shows bright fragments of the reference copied into the output corners.',
                   'Strongest reference following of all runs: 62% own ranked closest (Run 14: 54%, chance 25%); '
                   'output-vs-output similarity 0.74.',
                   'Worst quality of the DEResNet runs on both seeds: LPIPS 0.0452, FID 43.7 (Run 14: 0.0385, 40.5).'],
         verdict='Does not work: the HR negative pushes the outputs further from clean content, so the generator copies more of '
                 'the reference, not less.'),
    dict(key=f'{N}_nce005_gan002_hrneg', size=64, emb='deg NCE 0.05 + HR negative 0.05 + disc', base=f'{N}_nce01_gan002',
         title='Run 18 with the total degradation weight matched to Run 14',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE + HR negative, weight 0.05 + 0.05, total 0.1 as in Run 14']],
         summary='As Run 18, but both degradation terms at 0.05, so the total degradation weight equals Run 14\'s 0.1 '
                 '(Run 18 had 0.2).',
         purpose='Separate the effect of the HR negative from the doubled loss strength of Run 18. Run 12 (soft-target InfoNCE 0.05 '
                 '+ disc, no HR negative) is the control for the HR term itself.',
         findings=['Matching the weight explains part of Run 18\'s leak, not all: flat-region fine texture with panasonic_187 3.89 '
                   '(Run 18: 4.29, Run 14: 3.62, Run 12: 3.44, real LR: 2.41). The montage still shows the lattice and a bright '
                   'border artefact.',
                   'Reference following as Run 14: 54% own ranked closest (chance 25%); output-vs-output similarity 0.80.',
                   'LPIPS 0.0396, FID 41.3: worse than Run 14 (0.0385, 40.5) and Run 12 (0.0351, 39.5).'],
         verdict='The HR negative itself increases the leak, also at a fair weight. Dropped.'),
    dict(key=f'{N}_nce01_gan002_delta', size=64, emb='deg NCE 0.1 + disc, delta target', base=f'{N}_nce01_gan002',
         title='Run 14 with the degradation direction as the target (delta target)',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.1, target normalize(J(r) − J(h)) with h the reference\'s clean HR']],
         summary='As Run 14, but each reference\'s target is its degradation direction normalize(J(r) − J(h)): the DEResNet '
                 'embedding of the reference LR minus that of its own HR (area-downscaled). The condition stays E(r), so no HR is '
                 'needed at inference.',
         purpose='Copying the reference\'s content raises the similarity to J(r) and J(h) alike, so it no longer raises the score; '
                 'only moving along the reference\'s LR-versus-HR difference does.',
         findings=['The leak is gone: flat-region fine texture with panasonic_187 3.22 (Run 9 without degradation loss: 3.23, '
                   'Run 14: 3.62, real LR: 2.41); no lattice on the flat wall in the montage.',
                   'Reference following is kept at a moderate level: matched minus unmatched similarity 0.11 (Run 14: 0.18, '
                   'Run 12: 0.12, Run 8: 0.04); output-vs-output similarity 0.90 (Run 14: 0.82, Run 8: 0.95); own ranked closest 33% '
                   '(chance 25%; 24 cases, noisy).',
                   'Better quality than Run 14: LPIPS 0.0364, FID 39.7 (Run 14: 0.0385, 40.5); highest output diversity of the '
                   'DEResNet runs (0.016).'],
         verdict='Best trade-off so far: Run 12-level reference following without its leak. Candidate base for the next runs.'),
    dict(key=f'{N}_nce02_gan002_delta', size=64, emb='deg NCE 0.2 + disc, delta target', base=f'{N}_nce01_gan002_delta',
         title='Run 20 with the delta-target weight doubled (0.2)',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.2, delta target normalize(J(r) − J(h))']],
         summary='As Run 20, with the degradation loss weight raised from 0.1 to 0.2.',
         purpose='With the delta target, copying reference content no longer pays, so a stronger weight should buy reference '
                 'following without bringing the leak back.',
         findings=['No extra leak: flat-region fine texture with panasonic_187 3.23 on seed 0 (Run 20: 3.22, Run 14: 3.62, real LR: '
                   '2.41); the montage shows only faint horizontal streaks on the flat wall, no lattice.',
                   'Stronger reference following: matched minus unmatched similarity 0.16 (Run 20: 0.11, Run 14: 0.18); own in '
                   'closest 2 88% (Run 20: 67%, Run 14: 100%); output-vs-output similarity 0.84.',
                   'Quality drops: LPIPS 0.0414, FID 43.0 (Run 20: 0.0364, 39.7; Run 14: 0.0385, 40.5).'],
         verdict='Following close to Run 14 without its leak, but quality falls below Run 14.'),
    dict(key=f'{N}_nce05_gan002_delta', size=64, emb='deg NCE 0.5 + disc, delta target', base=f'{N}_nce01_gan002_delta',
         title='Run 20 with the delta-target weight at 0.5',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.5, delta target normalize(J(r) − J(h))']],
         summary='As Run 20, with the degradation loss weight raised from 0.1 to 0.5.',
         purpose='Find where the delta target stops paying off as its weight grows.',
         findings=['Strongest following of the delta runs: matched minus unmatched similarity 0.18 (Run 14: 0.18); own ranked '
                   'closest 63%, own in closest 2 88%; output-vs-output similarity 0.79.',
                   'The leak returns in another form: flat-region fine texture with panasonic_187 4.03 on seed 0 (Run 14: 3.62); the '
                   'montage shows bright border artefacts on the image edges instead of a lattice.',
                   'Much worse quality: LPIPS 0.0585, FID 55.9 (Run 20: 0.0364, 39.7).'],
         verdict='Too strong: the encoder is gamed through border artefacts and quality collapses. Dropped.'),
    dict(key=f'{N}_nce02_gan002_delta_b8', size=64, emb='deg NCE 0.2 + disc, delta target, border 8', base=f'{N}_nce02_gan002_delta',
         title='Run 21 with the degradation loss on the output interior (border crop)',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE, weight 0.2, delta target, on the central 48 px (8 px border dropped)']],
         summary='As Run 21 (delta 0.2), but outputs and targets are embedded on their central 48 px only, so edge pixels cannot '
                 'raise the degradation score.',
         purpose='Run 22 (delta 0.5) gamed the encoder with bright borders. Test whether removing the edges from the loss recovers '
                 'quality at weight 0.2.',
         findings=['Quality is not recovered: LPIPS 0.0414, FID 42.2 (Run 21: 0.0414, 43.0; Run 20: 0.0364, 39.7). A check on the '
                   'existing runs showed the same: Run 21 is 14% worse than Run 20 in LPIPS on the full 64 px, the central 48 px and '
                   'the central 32 px alike, so the drop is not at the border.',
                   'Reference following falls back to Run 20\'s level: matched minus unmatched similarity 0.10 (Run 21: 0.16, Run 20: '
                   '0.11), own in closest 2 63%. Part of Run 21\'s extra following was read from the border.',
                   'Leak unchanged: flat-region fine texture with panasonic_187 3.21 on seed 0 (Run 20: 3.22).'],
         verdict='No gain: the border crop removes the extra following but not the quality cost. Dropped.'),
    dict(key=f'{N}_nce01_gan002_delta_synrec', size=64, emb='delta 0.1 + disc + synthetic recon', base=f'{N}_nce01_gan002_delta',
         title='Run 20 + synthetic same-content pairs in the reconstruction losses',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE + synthetic reconstruction, weight 0.1 + 1.0, delta target on real batches']],
         summary='As Run 20, plus synthetic pairs: each step 4 HRs x 4 known degradations (anisotropic blur, x4 area, noise, JPEG; '
                 'blur range matched to DRealSR). The reference is the degradation applied to another HR, the target the same '
                 'degradation applied to the input HR; the native losses (low-pass MSE, LPIPS, PSD) are applied against it. The '
                 'discriminator sees real batches only.',
         purpose='Teach reference following with an exact target: only applying the reference\'s degradation to the input reaches '
                 'it, and copying the reference\'s content raises the loss. No encoder in this term.',
         findings=['Synthetic benchmark (known degradations on test HRs, no encoder): strong following, own target closest 51% '
                   '(Run 20: 29%, chance 25%), low-pass PSNR to the exact target 39.8 dB (Run 20: 33.8); leak over signal 0.40 '
                   '(Run 20: 0.73).',
                   'On real references the following does not improve: matched minus unmatched similarity 0.11 (Run 20: 0.11).',
                   'Real-data quality: LPIPS 0.0366 (Run 20: 0.0364), FID 41.2 (39.7). Diversity halves (0.008, Run 20: 0.016) and '
                   'the outputs carry sharper speckled fine detail than the real LR: flat-region fine texture with panasonic_187 '
                   '3.68 on seed 0 (Run 20: 3.22).'],
         verdict='Follows synthetic degradations well, but this does not transfer to real references and the outputs get an '
                 'oversharp synthetic look.'),
    dict(key=f'{N}_gan002_synrec', size=64, emb='disc + synthetic recon', base=f'{N}_gan002',
         title='Run 9 + synthetic same-content pairs in the reconstruction losses (no encoder in the loss)',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'synthetic reconstruction, weight 1.0, no InfoNCE']],
         summary='As Run 9 (no degradation loss), plus the synthetic reconstruction term of Run 24. No encoder anywhere in the loss.',
         purpose='Test whether synthetic pairs alone teach reference following, without the encoder-based loss.',
         findings=['Synthetic benchmark as Run 24: own target closest 51%, low-pass PSNR 39.6 dB, leak over signal 0.43 (Run 9: 28%, '
                   '34.2 dB, 0.68).',
                   'Weak following on real references: matched minus unmatched similarity 0.05 (Run 9: 0.04, Run 20: 0.11); '
                   'output-vs-output similarity 0.95.',
                   'Best LPIPS of the new runs, 0.0346, FID 39.9 (Run 9: 0.0336, 37.9); diversity halves (0.009) and flat-region '
                   'fine texture rises to 3.87 on seed 0 (Run 9: 3.23).'],
         verdict='Synthetic following does not transfer to real references; the synthetic targets add oversharp texture.'),
    dict(key=f'{N}_nce01_gan002_delta_synnce', size=64, emb='delta 0.1 + disc + synthetic InfoNCE', base=f'{N}_nce01_gan002_delta',
         title='Run 20 + synthetic same-content pairs inside the degradation loss only',
         vars=[['Encoder', 'DEResNet'], ['Reference size', '64 px'], ['Perceptual loss', 'native LPIPS 2.0'],
               ['Discriminator', 'native, weight 0.02'],
               ['Degradation loss', 'soft-target InfoNCE + synthetic InfoNCE, weight 0.1 + 0.1, delta target on real batches']],
         summary='As Run 20, plus a second soft-target InfoNCE on the same synthetic pairs as Run 24: each output G(x, z, E(r_k)) is '
                 'scored against J(y_k), the DEResNet embedding of the exact synthetic target. The target has the output\'s own '
                 'content, so only the degradation can pick the answer. No pixel loss on the synthetic targets.',
         purpose='Use synthetic pairs only in the degradation loss, keeping the image losses on real data.',
         findings=['Lowest leak of all runs: flat-region fine texture with panasonic_187 3.13 on seed 0 and 3.03 on seed 1 (Run 20: '
                   '3.22 / 3.35, Run 9 without degradation loss: 3.23); clean flat wall in the montage. Leak over signal on the '
                   'synthetic benchmark 0.65 (Run 20: 0.73).',
                   'Reference following as Run 20: matched minus unmatched similarity 0.11, own in closest 2 67%.',
                   'Quality as Run 20 or slightly better: LPIPS 0.0357, FID 40.1 (Run 20: 0.0364, 39.7); diversity 0.015.'],
         verdict='Small but consistent gain over Run 20: less leak at equal following and quality. New best base.'),
]


def load(ev):
    models, rt = {}, None
    for name in REALISM:
        if not (ev / name / 'realism_stats.json').exists():   # a seed still being evaluated
            continue
        j = json.loads((ev / name / 'realism_stats.json').read_text())
        for run, m in j['models'].items():
            models.setdefault(run, m)
        rt = rt or j
    floor = {k: st.mean(v[k] for v in rt['round_trip_ratio'].values()) for k in ('noise', 'detail', 'hf_power', 'saturation')}
    floor['fid'] = rt['round_trip_native_fid']
    emb = json.loads((ev / HAND[64] / 'embedding_similarity.json').read_text())['DEResNet']
    return models, floor, emb


def seeds(models, key, keep_missing=False):
    """Seed 0 and 1 results; a seed not evaluated yet is None (kept only with keep_missing)."""
    out = [models.get(f'{P}{key}_s{s}') for s in (0, 1)]
    return out if keep_missing else [m for m in out if m]


def stat(m, kind, k):
    if kind == 'overall':
        return m['overall'][k]
    if kind == 'ratio':
        return st.mean(v[k] for v in m['ratio'].values())
    return m['camera_spread']['x4'][k]['slope']


def metric_table(title, note, spec, models, run, base, labels, extra=None):
    """spec: (label, kind, key, digits, better) with better in 'min', 'max', 'one' (closest to 1) or None."""
    cur, ref = seeds(models, run['key'], keep_missing=True), (seeds(models, base['key']) if base else None)
    headers = ['Metric', 'Seed 0', 'Seed 1', 'Mean']
    if ref:
        headers += [f'{labels[base["key"]]} (mean)', 'Change']
    if extra:
        headers.append(extra[0])
    rows = []
    for label, kind, key, d, better in spec:
        vals = [stat(m, kind, key) if m else None for m in cur]
        mean = st.mean(v for v in vals if v is not None)
        row = [label] + [f'{v:.{d}f}' if v is not None else 'pending' for v in vals] + [f'<strong>{mean:.{d}f}</strong>']
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
    rows = [['Hand-cropped references (target)', f'{off(emb["ref_ref"]):.3f}', '—', '—', '—', '—', '—']]
    for r in runs:
        d = emb['models'][r['emb']]
        own = st.mean(d['gen_ref'][i][i] for i in range(4))
        oth = off(d['gen_ref'])
        rows.append([labels[r['key']], f'{off(d["gen_gen"]):.3f}', f'{own:.3f}', f'{oth:.3f}', f'{own - oth:+.3f}',
                     f'{d["top1_matched_ref"] * 100:.0f}%', f'{d["top2_matched_ref"] * 100:.0f}%'])
    rows[1][0] = f'<strong>{rows[1][0]}</strong>'
    return dict(title='Degradation matching (hand-cropped references, DEResNet cosine similarity)',
                headers=['', 'Output vs output', 'Output vs own reference', 'Output vs other references', 'Gap ↑', 'Own ranked closest ↑', 'Own in closest 2 ↑'],
                rows=rows, note='3 HR crops × 4 references × 2 seeds, native 64 px, DEResNet embeddings. Output vs output should be '
                'near the reference-vs-reference value (the target row). Own ranked closest: chance is 25%; own in closest 2: chance is 50%; lowdeg2 and highdeg2 '
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
    sw = ev / f'{P}{key}_s0_reference_sweep_native'
    if (sw / 'manifest.json').exists():
        man = json.loads((sw / 'manifest.json').read_text())
        refs = ', '.join(r['capture_id'] for r in man['references'])
        out['sweep_note'] = (f'Seed 0. Six DRealSR ×4 test photos (the most textured of 16 random crops each). Each figure keeps the HR '
                             f'crop and the noise fixed and changes only the reference: its own real LR, then cross-capture references '
                             f'({refs}). Rows: reference, generated native 64 px LR (area-downscaled output, nearest-neighbour zoom; the first '
                             f'column is the real native LR), its centre, and the difference to the matched output ×4. The same photos '
                             f'and references are used for every run, so tabs can be compared directly.')
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
                            'See the 64 px runs (Run 5 and Runs 7–26).')
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
                             'FID ↓', 'Own ranked closest ↑', 'Own in closest 2 ↑'],
                    rows=[[f'<a href="#{ids[r["key"]]}">{labels[r["key"]]}</a>', r['vars'][0][1].split(' (')[0], r['vars'][1][1].split(' (')[0],
                           short(r['vars'][2][1]), 'yes' if r['vars'][3][1] != 'none' else '—',
                           '—' if r['vars'][4][1] == 'none' else r['vars'][4][1].split('weight ')[1].split(',')[0],
                           f'{mean(r, "native_lpips"):.4f}', f'{mean(r, "native_dists"):.4f}', f'{mean(r, "native_fid"):.1f}',
                           f'{emb["models"][r["emb"]]["top1_matched_ref"] * 100:.0f}%' if r['emb'] else '—',
                           f'{emb["models"][r["emb"]]["top2_matched_ref"] * 100:.0f}%' if r['emb'] else '—'] for r in RUNS],
                    note='Own ranked closest / in closest 2 are measured on the hand-cropped references, for the 64 px DEResNet runs only (chance 25% / 50%).')
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
