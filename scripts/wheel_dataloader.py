"""
Wheel Loader Construction Site Data Collection Pipeline
Downloads YouTube videos, extracts frames, and filters for quality.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import subprocess
import argparse
from tqdm import tqdm
import hashlib
import sys


class LoaderDataCollector:
    def __init__(self, output_dir="loader_dataset", fps=1, min_laplacian=100,
                 ffmpeg_path=None, ytdlp_path=None):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract (1 = 1 frame per second)
            min_laplacian: Minimum Laplacian variance for blur detection (higher = sharper)
            ffmpeg_path: Full path to ffmpeg executable (None = auto-detect)
            ytdlp_path: Full path to yt-dlp executable (None = auto-detect)
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.min_laplacian = min_laplacian

        # Auto-detect ffmpeg path
        self.ffmpeg_path = self._find_ffmpeg(ffmpeg_path)
        # Auto-detect yt-dlp path
        self.ytdlp_path = self._find_ytdlp(ytdlp_path)

        # Create directory structure
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.videos_dir, self.frames_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.metadata = []

    def _find_ffmpeg(self, custom_path=None):
        """Auto-detect ffmpeg location."""
        if custom_path:
            return custom_path

        # Try imageio-ffmpeg first (works in venv)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"✓ Found ffmpeg via imageio-ffmpeg: {ffmpeg_exe}")
            return ffmpeg_exe
        except ImportError:
            pass

        # Try system PATH
        try:
            result = subprocess.run(["ffmpeg", "-version"],
                                    capture_output=True, check=True)
            print("✓ Found ffmpeg in system PATH")
            return "ffmpeg"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try common Windows locations
        if sys.platform == "win32":
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe"
            ]
            for path in common_paths:
                if Path(path).exists():
                    print(f"✓ Found ffmpeg at: {path}")
                    return str(path)

        return "ffmpeg"  # fallback, will fail in check_dependencies

    def _find_ytdlp(self, custom_path=None):
        """Auto-detect yt-dlp location."""
        if custom_path:
            return custom_path

        # Try to find in venv Scripts/bin directory
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # We're in a virtual environment
            if sys.platform == "win32":
                venv_ytdlp = Path(sys.prefix) / "Scripts" / "yt-dlp.exe"
            else:
                venv_ytdlp = Path(sys.prefix) / "bin" / "yt-dlp"

            if venv_ytdlp.exists():
                print(f"✓ Found yt-dlp in virtual environment: {venv_ytdlp}")
                return str(venv_ytdlp)

        # Try system PATH
        try:
            result = subprocess.run(["yt-dlp", "--version"],
                                    capture_output=True, check=True)
            print("✓ Found yt-dlp in system PATH")
            return "yt-dlp"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return "yt-dlp"  # fallback, will fail in check_dependencies

        # Create directory structure
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.videos_dir, self.frames_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.metadata = []

    def check_dependencies(self):
        """Check if required tools are installed."""
        all_ok = True

        # Check yt-dlp
        try:
            subprocess.run([self.ytdlp_path, "--version"],
                           capture_output=True, check=True)
            print(f"✓ yt-dlp is accessible")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ yt-dlp not found")
            print("\nTo install yt-dlp:")
            print("  pip install yt-dlp")
            all_ok = False

        # Check ffmpeg
        try:
            subprocess.run([self.ffmpeg_path, "-version"],
                           capture_output=True, check=True)
            print(f"✓ ffmpeg is accessible")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ ffmpeg not found")
            print("\nTo install ffmpeg (RECOMMENDED for virtual environments):")
            print("  pip install imageio-ffmpeg")
            print("\nAlternatively, install system-wide:")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            print("  Or use: choco install ffmpeg")
            all_ok = False

        if not all_ok:
            print("\n" + "=" * 60)
            print("QUICK FIX for PyCharm/Virtual Environment users:")
            print("=" * 60)
            print("Run these commands in your PyCharm terminal:")
            print("  pip install imageio-ffmpeg yt-dlp")
            print("Then run this script again.")
            print("=" * 60)

        return all_ok

    def download_video(self, url, video_id=None):
        """
        Download video from YouTube.

        Args:
            url: YouTube video URL
            video_id: Optional custom ID, otherwise extracted from URL

        Returns:
            Path to downloaded video file
        """
        if video_id is None:
            # Extract video ID from URL
            video_id = url.split("v=")[-1].split("&")[0]

        output_path = self.videos_dir / f"{video_id}.mp4"

        if output_path.exists():
            print(f"Video {video_id} already downloaded, skipping...")
            return output_path

        print(f"Downloading video {video_id}...")

        cmd = [
            self.ytdlp_path,
            "--cookies-from-browser", "firefox",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_path),
            url
        ]
        


        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Downloaded: {video_id}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {video_id}: {e}")
            return None

    def calculate_blur(self, image):
        """
        Calculate Laplacian variance to detect blur.
        Higher values = sharper image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def has_text_overlay(self, image, threshold=0.15):
        """
        Detect if image has significant text/UI overlay.
        Uses edge detection to find potential text regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check top and bottom 15% for UI elements
        h = edges.shape[0]
        top_region = edges[:int(h * 0.15), :]
        bottom_region = edges[int(h * 0.85):, :]

        top_density = np.sum(top_region > 0) / top_region.size
        bottom_density = np.sum(bottom_region > 0) / bottom_region.size

        return (top_density > threshold) or (bottom_density > threshold)

    def extract_frames(self, video_path, video_id):
        """
        Extract frames from video with quality filtering.

        Args:
            video_path: Path to video file
            video_id: Video identifier

        Returns:
            Number of frames extracted
        """
        print(f"Extracting frames from {video_id}...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Error opening video: {video_path}")
            return 0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps

        # Calculate frame sampling interval
        frame_interval = int(video_fps / self.fps)

        frame_count = 0
        extracted_count = 0

        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)

        with tqdm(total=int(total_frames / frame_interval),
                  desc=f"Processing {video_id}") as pbar:

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample frames at specified FPS
                if frame_count % frame_interval == 0:
                    # Quality checks
                    blur_score = self.calculate_blur(frame)
                    has_overlay = self.has_text_overlay(frame)

                    # Filter criteria
                    if blur_score >= self.min_laplacian and not has_overlay:
                        timestamp = frame_count / video_fps
                        frame_filename = f"{video_id}_frame_{extracted_count:06d}_t{timestamp:.2f}.jpg"
                        frame_path = video_frames_dir / frame_filename

                        # Save frame
                        cv2.imwrite(str(frame_path), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])

                        # Store metadata
                        self.metadata.append({
                            "video_id": video_id,
                            "frame_id": extracted_count,
                            "frame_path": str(frame_path.relative_to(self.output_dir)),
                            "timestamp": timestamp,
                            "blur_score": float(blur_score),
                            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                            "extracted_at": datetime.now().isoformat()
                        })

                        extracted_count += 1

                    pbar.update(1)

                frame_count += 1

        cap.release()
        print(f"✓ Extracted {extracted_count} quality frames from {video_id}")

        return extracted_count

    def process_video(self, url, video_id=None):
        """
        Complete pipeline: download and extract frames from a single video.

        Args:
            url: YouTube video URL
            video_id: Optional custom identifier
        """
        video_path = self.download_video(url, video_id)

        if video_path and video_path.exists():
            self.extract_frames(video_path, video_id or video_path.stem)

    def process_video_list(self, urls):
        """
        Process multiple videos from a list of URLs.

        Args:
            urls: List of YouTube video URLs
        """
        for i, url in enumerate(urls, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing video {i}/{len(urls)}")
            print(f"{'=' * 60}")
            self.process_video(url)

    def save_metadata(self):
        """Save collected metadata to JSON file."""
        metadata_path = self.metadata_dir / "frames_metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump({
                "collection_date": datetime.now().isoformat(),
                "total_frames": len(self.metadata),
                "fps": self.fps,
                "min_blur_threshold": self.min_laplacian,
                "frames": self.metadata
            }, f, indent=2)

        print(f"\n✓ Metadata saved to {metadata_path}")
        print(f"Total frames collected: {len(self.metadata)}")

    def generate_summary(self):
        """Generate a summary report of the collection."""
        if not self.metadata:
            print("No frames collected yet.")
            return

        video_counts = {}
        for item in self.metadata:
            vid = item['video_id']
            video_counts[vid] = video_counts.get(vid, 0) + 1

        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames collected: {len(self.metadata)}")
        print(f"Number of videos: {len(video_counts)}")
        print("\nFrames per video:")
        for vid, count in sorted(video_counts.items()):
            print(f"  {vid}: {count} frames")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect wheel loader construction site video data"
    )
    parser.add_argument(
        "--output-dir",
        default="loader_dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=int,
        default=100,
        help="Minimum Laplacian variance for sharpness (default: 100)"
    )
    parser.add_argument(
        "--urls-file",
        help="Path to text file with YouTube URLs (one per line)"
    )
    parser.add_argument(
        "--ffmpeg-path",
        help="Full path to ffmpeg executable (e.g., C:/ffmpeg/bin/ffmpeg.exe)"
    )
    parser.add_argument(
        "--ytdlp-path",
        help="Full path to yt-dlp executable"
    )

    args = parser.parse_args()

    # Default video URLs from the task
    default_urls = [
        "https://www.youtube.com/watch?v=o5LxOWSQSIk",
            ]

    # Initialize collector
    collector = LoaderDataCollector(
        output_dir=args.output_dir,
        fps=args.fps,
        min_laplacian=args.blur_threshold,
        ffmpeg_path=args.ffmpeg_path,
        ytdlp_path=args.ytdlp_path
    )

    # Check dependencies
    if not collector.check_dependencies():
        print("\nPlease install missing dependencies:")
        print("  pip install yt-dlp opencv-python numpy tqdm")
        return

    # Get URLs
    if args.urls_file:
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = default_urls

    print(f"\nStarting data collection:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Frame extraction rate: {args.fps} FPS")
    print(f"  Blur threshold: {args.blur_threshold}")
    print(f"  Number of videos: {len(urls)}\n")

    # Process videos
    collector.process_video_list(urls)

    # Save metadata and generate summary
    collector.save_metadata()
    collector.generate_summary()

    print("\n✓ Data collection complete!")
    print(f"Find your dataset in: {args.output_dir}/")


if __name__ == "__main__":
    main()
