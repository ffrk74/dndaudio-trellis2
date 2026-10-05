"""Atelier TRELLIS.2 : une image (fond transparent de préférence) → un GLB texturé.

Deux façons de tourner :
  - Pod (MODE=pod) : petit serveur HTTP sur le port 8000, protégé par ATELIER_TOKEN. Les fabrications passent
    dans une file (une à la fois sur la carte) ; on les suit par leur numéro (le proxy RunPod coupe les requêtes
    longues, d'où cette file plutôt qu'une réponse directe).
        GET  /sante            → {"pret": bool, "carte": "...", "file": n}
        POST /travaux          → {"id": "..."}       (corps : les réglages ci-dessous)
        GET  /travaux/<id>     → {"etat": "en file|en cours|fini|échec", ...résultat}
  - Serverless RunPod (par défaut).

Réglages : image_base64 (ou image_url), seed (42), resolution ("1024_cascade" ; "512", "1024", "1536_cascade"),
decimation (60000), texture_size (1024).
Résultat : {"ok": true, "glb_gzip_base64": "...", "octets": n, "secondes": {...}}  ou  {"ok": false, "erreur": "..."}
"""
import base64
import gzip
import io
import json
import os
import queue
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

PIPELINE = None
PRET = threading.Event()
MAX_SORTIE = 10 * 1024 * 1024 - 64 * 1024  # réponse Serverless : on reste sous 10 Mo


def charger():
    """Chargé une fois par démarrage (DINOv3 et RMBG-2.0 se téléchargent ici, avec HF_TOKEN)."""
    global PIPELINE
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    PIPELINE.cuda()
    PRET.set()
    print(f"[atelier] prêt en {time.time() - t0:.0f} s sur {torch.cuda.get_device_name(0)}", flush=True)


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


def fabriquer(inp, limite=None):
    import o_voxel
    try:
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
        if limite and len(out) > limite:
            return {"ok": False, "erreur": f"GLB trop gros ({len(out)} octets) : baisser decimation ou texture_size"}
        return {"ok": True, "glb_gzip_base64": out.decode("ascii"), "octets": len(data), "secondes": t}
    except Exception as e:  # l'atelier doit toujours répondre
        return {"ok": False, "erreur": str(e), "trace": traceback.format_exc()}
    finally:
        torch.cuda.empty_cache()


def mode_pod():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    jeton = os.environ.get("ATELIER_TOKEN", "")
    travaux, file = {}, queue.Queue()

    def ouvrier():
        PRET.wait()
        while True:
            tid = file.get()
            t = travaux[tid]
            t["etat"], t["debut"] = "en cours", time.time()
            r = fabriquer(t.pop("entree"))
            t.update(r, etat="fini" if r.get("ok") else "échec", duree=round(time.time() - t["debut"], 1))

    class Accueil(BaseHTTPRequestHandler):
        def repondre(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def autorise(self):
            return not jeton or self.headers.get("Authorization") == f"Bearer {jeton}"

        def do_GET(self):
            if self.path == "/sante":
                return self.repondre(200, {"pret": PRET.is_set(), "carte": torch.cuda.get_device_name(0), "file": file.qsize()})
            if not self.autorise():
                return self.repondre(401, {"erreur": "jeton"})
            if self.path.startswith("/travaux/"):
                t = travaux.get(self.path.split("/")[-1])
                return self.repondre(200, t) if t else self.repondre(404, {"erreur": "inconnu"})
            self.repondre(404, {"erreur": "chemin"})

        def do_POST(self):
            if not self.autorise():
                return self.repondre(401, {"erreur": "jeton"})
            if self.path != "/travaux":
                return self.repondre(404, {"erreur": "chemin"})
            inp = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            tid = uuid.uuid4().hex
            travaux[tid] = {"id": tid, "etat": "en file", "entree": inp, "cree": time.time()}
            file.put(tid)
            self.repondre(200, {"id": tid, "file": file.qsize()})

        def log_message(self, *args):
            pass

    threading.Thread(target=charger, daemon=True).start()
    threading.Thread(target=ouvrier, daemon=True).start()
    print("[atelier] serveur du Pod sur le port 8000", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), Accueil).serve_forever()


if __name__ == "__main__":
    if os.environ.get("MODE") == "pod":
        mode_pod()
    else:
        import runpod
        charger()
        runpod.serverless.start({"handler": lambda job: fabriquer(job.get("input") or {}, MAX_SORTIE)})
