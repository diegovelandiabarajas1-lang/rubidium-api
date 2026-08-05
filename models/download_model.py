#!/usr/bin/env python3
"""
Model downloader for Rubidium API.
Downloads GGUF models from HuggingFace with progress bar and verification.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


CHUNK_SIZE = 8192


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_progress(url: str, dest: Path, expected_size: int = None, sha256: str = None) -> bool:
    """Download file with progress bar."""
    try:
        req = Request(url, headers={"User-Agent": "Rubidium-API/1.0"})
        with urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0)) or expected_size
            downloaded = 0
            
            print(f"Downloading to: {dest}")
            print(f"Expected size: {format_bytes(total_size) if total_size else 'Unknown'}")
            
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        print(f"\r[{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)})", end="", flush=True)
                    else:
                        print(f"\rDownloaded: {format_bytes(downloaded)}", end="", flush=True)
            
            print()  # New line after progress
            
    except (URLError, HTTPError, TimeoutError) as e:
        print(f"\nDownload failed: {e}")
        if dest.exists():
            dest.unlink()
        return False
    
    # Verify SHA256 if provided
    if sha256:
        print("Verifying SHA256...")
        actual_sha256 = calculate_sha256(dest)
        if actual_sha256.lower() != sha256.lower():
            print(f"SHA256 mismatch!")
            print(f"  Expected: {sha256}")
            print(f"  Actual:   {actual_sha256}")
            dest.unlink()
            return False
        print("SHA256 verified successfully.")
    
    print(f"Download complete: {dest} ({format_bytes(dest.stat().st_size)})")
    return True


def load_model_config(config_path: Path) -> dict:
    """Load model configuration from JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for Rubidium API")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct-q4", 
                        help="Model config name (without .json)")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--output-dir", default="models", help="Output directory for GGUF files")
    parser.add_argument("--force", action="store_true", help="Force re-download if exists")
    parser.add_argument("--list", action="store_true", help="List available model configs")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("Available model configurations:")
        for config_file in models_dir.glob("*.json"):
            with open(config_file, "r") as f:
                config = json.load(f)
            print(f"  {config['model_id']}: {config['name']} ({config['size_gb']} GB)")
        return
    
    config_file = models_dir / f"{args.model}.json"
    if not config_file.exists():
        print(f"Config not found: {config_file}")
        print(f"Available configs: {[f.stem for f in models_dir.glob('*.json')]}")
        return 1
    
    config = load_model_config(config_file)
    
    gguf_filename = config["filename"]
    dest_path = output_dir / gguf_filename
    
    if dest_path.exists() and not args.force:
        print(f"File already exists: {dest_path}")
        if config.get("sha256"):
            print("Verifying existing file...")
            actual_sha256 = calculate_sha256(dest_path)
            if actual_sha256.lower() == config["sha256"].lower():
                print("File verified, skipping download.")
                return 0
            else:
                print("SHA256 mismatch, re-downloading...")
        else:
            print("Use --force to re-download.")
            return 0
    
    print(f"Model: {config['name']}")
    print(f"Repo: {config['huggingface_repo']}")
    print(f"File: {gguf_filename}")
    print(f"Size: {config['size_gb']} GB")
    print(f"License: {config['license']}")
    print()
    
    url = config["download_url"]
    sha256 = config.get("sha256") or None
    expected_size = int(config["size_gb"] * 1024 * 1024 * 1024)
    
    success = download_with_progress(url, dest_path, expected_size, sha256)
    
    if success:
        print(f"\nModel ready at: {dest_path.absolute()}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())