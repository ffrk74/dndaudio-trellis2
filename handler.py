"""Atelier TRELLIS.2 sur RunPod Serverless : une image (fond transparent de préférence) → un GLB texturé.

Entrée (job["input"]) :
    image_base64   : image en base64 (ou image_url : adresse publique de l'image)
    seed           : graine (défaut 42) — changer de graine donne une autre version
    resolution     : "512" | "1024" | "1024_cascade" | "1536_cascade" (défaut "1024_cascade")
    decimation     : nombre de sommets du GLB final (défaut 60000)
    texture_size   : taille des textures (défaut 1024)
Sortie :
    {"ok": true, "glb_gzip_base64": "...", "secondes": {...}}  ou  {"ok": false, "erreur": "...", "trace": "..."}
"""
import base64
import gzip
import io
import os
import tempfile
import time
import traceback
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import runpod
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Chargé une fois par démarrage de machine (DINOv3 et RMBG-2.0 se téléchargent ici, avec HF_TOKEN).
t0 = time.time()
PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
PIPELINE.cuda()
print(f"[atelier] prêt en {time.time() - t0:.0f} s")

MAX_SORTIE = 10 * 1024 * 1024 - 64 * 1024  # réponse RunPod : on reste sous 10 Mo


def lire_image(inp):
    if inp.get("image_url"):
        with urllib.request.urlopen(inp["image_url"], timeout=60) as r:
            data = r.read()
    else:
        b64 = inp["image_base64"]
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def handler(job):
    try:
        inp = job.get("input") or {}
        if not (inp.get("image_base64") or inp.get("image_url")):
            return {"ok": False, "erreur": "image_base64 ou image_url requis"}
        t = {}
        a = time.time()
        image = lire_image(inp)
        with torch.inference_mode():
            mesh = PIPELINE.run(image, seed=int(inp.get("seed", 42)), pipeline_type=inp.get("resolution", "1024_cascade"))[0]
        t["generation"] = round(time.time() - a, 1)
        a = time.time()
        mesh.simplify(16_777_216)  # limite de nvdiffrast
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(inp.get("decimation", 60_000)), texture_size=int(inp.get("texture_size", 1024)),
            remesh=True, remesh_band=1, remesh_project=0, verbose=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            chemin = f.name
        try:
            glb.export(chemin, extension_webp=True)
            data = open(chemin, "rb").read()
        finally:
            os.unlink(chemin)
        t["export"] = round(time.time() - a, 1)
        out = base64.b64encode(gzip.compress(data))
        if len(out) > MAX_SORTIE:
            return {"ok": False, "erreur": f"GLB trop gros ({len(out)} octets) : baisser decimation ou texture_size"}
        return {"ok": True, "glb_gzip_base64": out.decode("ascii"), "octets": len(data), "secondes": t}
    except Exception as e:  # le service doit toujours répondre
        return {"ok": False, "erreur": str(e), "trace": traceback.format_exc()}
    finally:
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
