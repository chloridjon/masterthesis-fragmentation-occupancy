"""Source-checked display repair for Chapter-5 TODO 5-07.

Original digitisation tables and all fitting inputs stay immutable. Recover
only source-visible G. spirurus upper bounds withheld by an overbroad artwork
rectangle, and clip the reviewed ceiling stroke to the probability axis.
Keep mask-sensitive columns missing; do not interpolate across missing limits.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
BASE = FINAL / 'data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction'
OUT = FINAL / 'results/00_thesis_figures/04_selected_bird_sum_surfaces/review/20261102_todo5_07'
DATA = FINAL / 'data/processed/00_thesis_figures/04_bird_band_repair_20261102'
IMAGE = FINAL / 'data/external/morante_filho_2021/figure2_publisher.png'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    source = BASE / '11_bird_figure2_native_pixel_traces.csv'
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    calibration_path = BASE / '11_figure2_panel_calibration.json'
    config_path = FINAL / 'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction/11_figure2_digitization.json'
    config = json.loads(config_path.read_text())
    assert sha(IMAGE) == config['image_sha256']
    original_image = Image.open(IMAGE).convert('RGB')
    a = np.asarray(original_image)
    # Source review: the neutral-grey upper boundary lies clear of the bird.
    # Two colour-mask-sensitive candidates remain withheld even in this ROI.
    visible_columns = set(range(109, 144)) - {115, 139}
    recovered = []
    ceiling = []
    reviewed = []
    for row in rows:
        if row['species_id'] != 'mf2021_07':
            continue
        result = dict(row)
        result['repair'] = 'unchanged_source_reading'
        x = int(row['x_source_pixel'])
        if x in visible_columns:
            assert row['upper_status'] == 'potential_illustration_occlusion'
            assert row['upper_mask_spread_pixels'] and float(row['upper_mask_spread_pixels']) <= 4
            assert int(row['band_neutral_pixel_count']) >= 4
            y = int(float(row['upper_candidate_y_pixel']))
            pixel = a[y, x].astype(float)
            assert np.ptp(pixel) <= config['grey_chroma_max']
            assert config['grey_intensity_min'] <= pixel.mean() <= config['grey_intensity_max']
            result.update(psi_ci_upper_digitized=row['upper_candidate_probability'], upper_available='True',
                          upper_status='source_visible_boundary_reviewed', repair='overbroad_artwork_rectangle')
            recovered.append(x)
        elif row['upper_status'] == 'outside_probability_axis':
            # All 51 reviewed candidates are the same ceiling line stroke.
            assert float(row['upper_candidate_y_pixel']) == 418
            assert float(row['upper_mask_spread_pixels']) == 0
            assert 1 - 2 * float(row['psi_per_pixel']) <= float(row['psi_figure2_curve']) <= 1
            assert 1 < float(row['upper_candidate_probability']) < 1 + 2 * float(row['psi_per_pixel'])
            result.update(psi_ci_upper_digitized='1.0', upper_available='True',
                          upper_status='source_ceiling_stroke_axis_clip', upper_axis_resolution_snap='True',
                          repair='reviewed_probability_ceiling')
            ceiling.append(x)
        for side in ['lower', 'upper']:
            if result[f'{side}_available'] == 'True':
                value = float(result[f'psi_ci_{side}_digitized'])
                assert 0 <= value <= 1
        if result['lower_available'] == result['upper_available'] == 'True':
            assert float(result['psi_ci_lower_digitized']) <= float(result['psi_ci_upper_digitized'])
        reviewed.append(result)
    assert set(recovered) == visible_columns and len(ceiling) == 51
    output = DATA / '04_G_spirurus_reviewed_native_band.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(reviewed[0]))
        writer.writeheader(); writer.writerows(reviewed)
    # Review every recovered upper point against the source, without altering
    # the image used as data. Cyan: recovered visible boundary; magenta: axis.
    overlay = original_image.copy()
    draw = ImageDraw.Draw(overlay)
    for row in reviewed:
        x = int(row['x_source_pixel'])
        if row['repair'] == 'overbroad_artwork_rectangle':
            y = int(float(row['upper_candidate_y_pixel']))
            draw.ellipse((x-1, y-1, x+1, y+1), fill='#0072B2')
        elif row['repair'] == 'reviewed_probability_ceiling':
            draw.point((x, 420), fill='#CC0077')
    overlay.crop((85, 405, 320, 625)).resize((1175, 1100)).save(OUT / '04_reviewed_source_overlay.png')
    support = [r for r in reviewed if r['within_observed_domain'] == 'True' and .07 <= float(r['H_source_pixel']) <= .925]
    missing = [dict(x=int(r['x_source_pixel']), H=float(r['H_source_pixel']), lower_status=r['lower_status'], upper_status=r['upper_status'])
               for r in support if r['lower_available'] != 'True' or r['upper_available'] != 'True']
    report = dict(status='source_values_checked_visual_review_pending', source_image_sha256=sha(IMAGE),
                  source_trace_sha256=sha(source), original_joined_band_sha256=sha(BASE / '11_bird_curves_main_H_grid_with_figure2_bands.csv'),
                  calibration_sha256=sha(calibration_path), digitization_config_sha256=sha(config_path),
                  review_source_sha256=sha(Path(__file__)), corrected_table_sha256=sha(output),
                  recovered_visible_upper_columns=recovered, clipped_ceiling_columns=ceiling,
                  remaining_unresolved_columns_in_display=missing,
                  scientific_scope='Source-image display correction only; coverage remains assumed 95%; original digitisation and every central curve, fit and RMSE unchanged.',
                  no_gap_interpolation=True, display_grid='Native source columns for panel a band only; all central/model curves remain on the original 77 H levels.')
    (OUT / '04_band_source_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status','remaining_unresolved_columns_in_display','scientific_scope']}, indent=2))


if __name__ == '__main__':
    main()
