"""Build the selected video-based real mini without copying UE assets or raw RGB."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build(source_root, selection, output):
    if output.exists():
        raise FileExistsError(output)
    selected = json.loads(selection.read_text())
    output.mkdir(parents=True)
    index = []
    for entry in selected['cases']:
        source = source_root / entry['source_relative_path']
        original = source / 'manifest.json'
        poses_file = source / 'pose/poses.json'
        if digest(original) != entry['source_manifest_sha256'] or digest(poses_file) != entry['source_poses_sha256']:
            raise ValueError(f'Source changed: {source.name}')
        m = json.loads(original.read_text())
        rows = json.loads(poses_file.read_text())
        start, end = entry['export_source_range']
        if start != 0 or m['split']['eval'] != entry['query_source_range']:
            raise ValueError('Selection requires zero-based, unchanged Context/Query ranges')
        case = output / entry['mini_id']
        case.mkdir()
        manifest = {k: m[k] for k in ['dataset', 'protocol', 'scene', 'fps', 'resolution', 'camera', 'split', 'prompt', 'camera_control'] if k in m}
        manifest.update(case_id=entry['mini_id'], num_frames=end, trajectory=entry['trajectory'], task=dict(m.get('task', {})), frame_indexing='zero_based', range_convention='half_open')
        manifest['phase_ranges'] = {}
        for key, value in m.get('phase_ranges', {}).items():
            ranges = [value] if isinstance(value[0], int) else value
            clipped = [[max(0, a), min(end, b)] for a, b in ranges if max(0, a) < min(end, b)]
            if clipped:
                manifest['phase_ranges'][key] = clipped[0] if isinstance(value[0], int) else clipped
        if m['protocol'] == 'STRAIGHT_LOOP':
            manifest['task'].update(description='Observe at A, move to B, then turn at B; Query is the final 90 degrees of the turn.', return_segment_in_full_video=False, return_segment_in_eval=False)
            manifest['task'].pop('return_indices', None)
            if manifest['phase_ranges']['eval_turnaround'] != manifest['split']['eval']:
                raise ValueError('Straight Query boundary mismatch')
        manifest['videos'] = {'context': 'context.mp4', 'eval': 'eval.mp4'}
        manifest['rgb_storage'] = {'mode': 'decode_context_video', 'prepare_command': 'scenemembench prepare --case <case>', 'query_reference': 'eval.mp4', 'note': 'Context PNGs are decoded from the supplied encoded video, not original capture PNGs.'}
        manifest['release'] = {'version': 'greek-island-mini-v1', 'source_case_id': m['case_id'], 'source_relative_path': entry['source_relative_path'], 'source_frame_range': [0, end], 'task': entry['task'], 'video_bytes_preserved': True}
        clean_rows = []
        for i, row in enumerate(rows[:end]):
            if row['frame_id'] != i:
                raise ValueError('Noncontiguous source frame IDs')
            clean = {k: row[k] for k in ['frame_id', 'timestamp', 'phase', 'waypoint', 'action', 'primary_pose', 'pose_unreal', 'camera_to_world', 'relative_yaw_unwrapped_deg'] if k in row}
            clean['rgb'] = f'rgb/{i:06d}.png'
            clean_rows.append(clean)
        save(case / 'manifest.json', manifest)
        save(case / 'pose/poses.json', clean_rows)
        (case / 'prompt.txt').write_text(manifest.get('prompt', '') + '\n')
        lineage = {'source_case_id': m['case_id'], 'source_relative_path': entry['source_relative_path'], 'retained_source_range': [0, end], 'source_manifest_sha256': digest(original), 'source_poses_sha256': digest(poses_file), 'source_videos': {}}
        for name in ['context.mp4', 'eval.mp4']:
            shutil.copyfile(source / name, case / name)
            lineage['source_videos'][name] = {'sha256': digest(source / name), 'bytes': (source / name).stat().st_size}
        save(case / 'source.json', lineage)
        index.append({'case_id': entry['mini_id'], 'path': entry['mini_id'], 'trajectory': entry['trajectory'], 'task': entry['task'], 'context_frames': m['split']['context'][1], 'query_frames': m['split']['eval'][1] - m['split']['eval'][0]})
    save(output / 'index.json', {'version': 'greek-island-mini-v1', 'cases': index, 'distribution_status': 'local review; scene redistribution terms not yet verified'})
    (output / 'USAGE.txt').write_text('Greek Island mini v1\n\nPrepare each case before inference:\n  scenemembench prepare --case <case>\n  scenemembench validate --case <case>\n\nContext RGB is decoded from context.mp4. Query images are evaluation references only.\nVideos retain their source encoded bytes. Full videos and UE source assets are not included.\nThis local review artifact does not grant redistribution rights to third-party scene content.\n')
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output / 'checksums.json', hashes)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        tar.add(output, arcname=output.name)
    save(archive.with_suffix(archive.suffix + '.json'), {'sha256': digest(archive), 'bytes': archive.stat().st_size, 'files': len(hashes), 'cases': len(index)})
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.selection, args.output)
