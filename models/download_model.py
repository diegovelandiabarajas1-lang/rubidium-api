#!/usr/bin/env python3
"""
Download Llama-3.2-1B-Instruct Q4_K_M from HuggingFace
~1.0 GB, Apache 2.0, 128K context
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

MODEL_CONFIG = {
    "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
    "size_gb": 1.0,
    "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf"
}

def download_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    output_path = models_dir / MODEL_CONFIG["filename"]
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"✅ Modelo ya existe: {output_path} ({size_mb:.1f} MB)")
        return
    
    print(f"⬇️  Descargando {MODEL_CONFIG['filename']} ({MODEL_CONFIG['size_gb']} GB)...")
    print(f"   Desde: {MODEL_CONFIG['url']}")
    
    try:
        response = requests.get(MODEL_CONFIG["url"], stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Descargando"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        final_size = output_path.stat().st_size / 1e9
        print(f"\n✅ Descarga completada: {output_path} ({final_size:.2f} GB)")
        
    except KeyboardInterrupt:
        print("\n❌ Descarga cancelada")
        if output_path.exists():
            output_path.unlink()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    download_model()