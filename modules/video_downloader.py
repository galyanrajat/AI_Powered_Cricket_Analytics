from pathlib import Path
import shutil
import yt_dlp
from modules.logger import get_logger
from modules.cache_manager import needs_update, update_cache

logger = get_logger("video_downloader")

def ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None

def run(url: str, output_path: Path) -> Path:
    """
    Downloads a video from YouTube using yt-dlp.
    If ffmpeg is available, merges best video + audio.
    If not, downloads video-only format.
    Uses caching to skip re-download unless URL changes.
    """
    step_config = {"url": url}

    # Skip if already exists & unchanged
    if output_path.exists() and not needs_update("video_downloader", [output_path], step_config):
        logger.info("Video already downloaded — skipping.")
        return output_path

    logger.info(f"Downloading video from: {url}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select download settings based on ffmpeg availability
    if ffmpeg_available():
        logger.info("ffmpeg detected — downloading video + audio with merge")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": str(output_path),
            "quiet": True,
            "merge_output_format": "mp4"
        }
    else:
        logger.warning("⚠ ffmpeg not found — downloading video-only (no audio)")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]/mp4",
            "outtmpl": str(output_path),
            "quiet": True
        }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download video from {url}")

    logger.info(f"Video saved to {output_path}")

    update_cache("video_downloader", [output_path], step_config)
    return output_path
