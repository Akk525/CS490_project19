import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.youcook2_loader import _discover_annotation_file, _extract_step_times
from src.utils.io import read_json


def _youtube_url(video_id: str) -> str:
    if video_id.startswith("http://") or video_id.startswith("https://"):
        return video_id
    return f"https://www.youtube.com/watch?v={video_id}"


def _frame_time(bounds: Dict[str, float | None], fallback_seconds: float) -> float:
    start = bounds.get("start")
    end = bounds.get("end")
    if start is not None and end is not None and end > start:
        return start + ((end - start) / 2.0)
    if start is not None:
        return start
    return fallback_seconds


def _iter_targets(database: Dict, limit: int | None) -> Iterable[Tuple[str, List[Dict[str, float | None]]]]:
    count = 0
    for video_id, item in database.items():
        step_times = _extract_step_times(item)
        if not step_times:
            continue
        yield video_id, step_times
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one YouTube frame per YouCook2 annotated step for VLM disruptions."
    )
    parser.add_argument("--raw-dir", default="data/raw/youcook2")
    parser.add_argument("--output-dir", default="data/raw/youcook2/frames")
    parser.add_argument("--limit", type=int, help="Optional max number of videos to process.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--failure-log", default="outputs/frame_download_failures.jsonl")
    parser.add_argument("--format", default="bestvideo[height<=720]+bestaudio/best[height<=720]/best")
    parser.add_argument("--method", choices=["video-cache", "temp-video", "direct-url", "clip", "pipe"], default="video-cache")
    parser.add_argument("--pipe", action="store_true", help="Pipe yt-dlp output directly to ffmpeg instead of using a temporary clip file.")
    parser.add_argument("--video-cache-dir", default="data/raw/youcook2/videos")
    parser.add_argument("--cookies-from-browser", help="Browser name to pass to yt-dlp, e.g. chrome, safari, or firefox.")
    parser.add_argument("--cookies", help="Path to a Netscape cookies.txt file to pass to yt-dlp.")
    parser.add_argument("--js-runtimes", help="JavaScript runtime spec to pass to yt-dlp, e.g. node or deno.")
    parser.add_argument("--remote-components", help="Remote component policy for yt-dlp, e.g. ejs:github.")
    args = parser.parse_args()

    raw_dir = (ROOT / args.raw_dir).resolve() if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    ann_path = _discover_annotation_file(raw_dir)
    payload = read_json(ann_path)
    database = payload.get("database") if isinstance(payload, dict) else None
    if not isinstance(database, dict):
        raise ValueError('Unexpected YouCook2 format: expected key "database" in annotations JSON.')

    output_dir.mkdir(parents=True, exist_ok=True)
    video_cache_dir = (ROOT / args.video_cache_dir).resolve() if not Path(args.video_cache_dir).is_absolute() else Path(args.video_cache_dir).resolve()
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    failure_log = (ROOT / args.failure_log).resolve() if not Path(args.failure_log).is_absolute() else Path(args.failure_log).resolve()
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    successes = 0

    for video_id, step_times in _iter_targets(database, args.limit):
        video_dir = output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        url = _youtube_url(video_id)
        pending_steps = [
            (step_index, bounds, video_dir / f"step_{step_index}.jpg")
            for step_index, bounds in enumerate(step_times)
            if args.overwrite or not (video_dir / f"step_{step_index}.jpg").exists()
        ]
        if not pending_steps:
            continue

        cached_video: Path | None = None
        if args.method == "video-cache" and not args.dry_run:
            cached_video = _ensure_cached_video(video_id, url, args.format, video_cache_dir, args)
            if cached_video is None:
                failures += len(pending_steps)
                continue

        temp_video_dir: tempfile.TemporaryDirectory | None = None
        if args.method == "temp-video" and not args.dry_run:
            temp_video_dir = tempfile.TemporaryDirectory(prefix="youcook2_video_")
            cached_video = _ensure_cached_video(video_id, url, args.format, Path(temp_video_dir.name), args)
            if cached_video is None:
                temp_video_dir.cleanup()
                failures += len(pending_steps)
                continue

        try:
            for step_index, bounds, out_path in pending_steps:
                timestamp = _frame_time(bounds, fallback_seconds=max(0.0, float(step_index)))
                print(f"{video_id} step {step_index}: {timestamp:.3f}s -> {out_path}")
                if not args.dry_run:
                    method = "pipe" if args.pipe else args.method
                    if method in {"video-cache", "temp-video"}:
                        if cached_video is None:
                            proc = subprocess.CompletedProcess(["cache-video", url], returncode=1)
                        else:
                            proc = _extract_frame_from_local_video(cached_video, timestamp, out_path)
                    elif method == "pipe":
                        cmd = _pipe_command(args.format, timestamp, url, out_path, args)
                        print(cmd)
                        proc = subprocess.run(cmd, shell=True)
                    elif method == "clip":
                        proc = _download_via_temp_clip(args.format, timestamp, url, out_path, args)
                    else:
                        proc = _download_from_direct_url(args.format, timestamp, url, out_path, args)
                    if proc.returncode != 0:
                        failures += 1
                        record = {
                            "video_id": video_id,
                            "step_index": step_index,
                            "timestamp": timestamp,
                            "output_path": str(out_path),
                            "returncode": proc.returncode,
                            "cmd": proc.args if isinstance(proc.args, str) else " ".join(str(x) for x in proc.args),
                        }
                        with failure_log.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")
                        print(f"WARNING: failed frame download for {video_id} step {step_index}; logged to {failure_log}")
                        if args.fail_fast:
                            raise subprocess.CalledProcessError(proc.returncode, cmd)
                        continue
                    successes += 1
        finally:
            if temp_video_dir is not None:
                temp_video_dir.cleanup()

    print(f"Frame download complete: successes={successes}, failures={failures}, failure_log={failure_log}")


def _yt_dlp_auth_args(args: argparse.Namespace) -> List[str]:
    cmd_args: List[str] = []
    if args.cookies_from_browser:
        cmd_args.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies:
        cmd_args.extend(["--cookies", args.cookies])
    if args.js_runtimes:
        cmd_args.extend(["--js-runtimes", args.js_runtimes])
    if args.remote_components:
        cmd_args.extend(["--remote-components", args.remote_components])
    return cmd_args


def _pipe_command(format_spec: str, timestamp: float, url: str, out_path: Path, args: argparse.Namespace) -> str:
    left_cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        *_yt_dlp_auth_args(args),
        "--format",
        format_spec,
        "--download-sections",
        f"*{timestamp:.3f}-{timestamp + 0.100:.3f}",
        "--force-keyframes-at-cuts",
        "--output",
        "-",
        url,
    ]
    right_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        str(out_path),
    ]
    return " | ".join(
        [
            " ".join(shlex.quote(x) for x in left_cmd),
            " ".join(shlex.quote(x) for x in right_cmd),
        ]
    )


def _download_via_temp_clip(format_spec: str, timestamp: float, url: str, out_path: Path, args: argparse.Namespace) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory(prefix="youcook2_frame_") as tmpdir:
        clip_template = str(Path(tmpdir) / "clip.%(ext)s")
        ytdlp_cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            *_yt_dlp_auth_args(args),
            "--format",
            format_spec,
            "--download-sections",
            f"*{timestamp:.3f}-{timestamp + 0.100:.3f}",
            "--force-keyframes-at-cuts",
            "--output",
            clip_template,
            url,
        ]
        print(" ".join(shlex.quote(x) for x in ytdlp_cmd))
        ytdlp_proc = subprocess.run(ytdlp_cmd)
        if ytdlp_proc.returncode != 0:
            return ytdlp_proc

        clips = [p for p in Path(tmpdir).iterdir() if p.is_file()]
        if not clips:
            return subprocess.CompletedProcess(ytdlp_cmd, returncode=1)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clips[0]),
            "-frames:v",
            "1",
            str(out_path),
        ]
        print(" ".join(shlex.quote(x) for x in ffmpeg_cmd))
        return subprocess.run(ffmpeg_cmd)


def _download_from_direct_url(format_spec: str, timestamp: float, url: str, out_path: Path, args: argparse.Namespace) -> subprocess.CompletedProcess:
    get_url_cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        *_yt_dlp_auth_args(args),
        "--format",
        format_spec,
        "--get-url",
        url,
    ]
    print(" ".join(shlex.quote(x) for x in get_url_cmd))
    resolved = subprocess.run(get_url_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if resolved.returncode != 0:
        if resolved.stderr:
            print(resolved.stderr.strip())
        return resolved

    direct_urls = [line.strip() for line in resolved.stdout.splitlines() if line.strip()]
    if not direct_urls:
        return subprocess.CompletedProcess(get_url_cmd, returncode=1)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        direct_urls[0],
        "-frames:v",
        "1",
        str(out_path),
    ]
    print(" ".join(shlex.quote(x) for x in [*ffmpeg_cmd[:7], "<direct-video-url>", *ffmpeg_cmd[8:]]))
    return subprocess.run(ffmpeg_cmd)


def _ensure_cached_video(video_id: str, url: str, format_spec: str, video_cache_dir: Path, args: argparse.Namespace) -> Path | None:
    existing = sorted(video_cache_dir.glob(f"{video_id}.*"))
    if existing:
        return existing[0]

    output_template = str(video_cache_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        *_yt_dlp_auth_args(args),
        "--format",
        format_spec,
        "--output",
        output_template,
        url,
    ]
    print(" ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"WARNING: failed video download for {video_id}")
        return None

    downloaded = sorted(video_cache_dir.glob(f"{video_id}.*"))
    return downloaded[0] if downloaded else None


def _extract_frame_from_local_video(video_path: Path, timestamp: float, out_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    print(" ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(cmd)


if __name__ == "__main__":
    main()
