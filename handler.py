"""Atelier TRELLIS.2 : une image (fond transparent de préférence) → un GLB texturé.

Deux façons de tourner :
  - Pod (MODE=pod) : petit serveur HTTP sur le port 8000, protégé par ATELIER_TOKEN. Les fabrications passent
    dans une file ; on les suit par leur numéro (le proxy RunPod coupe les requêtes longues, d'où cette file plutôt
    qu'une réponse directe). Plusieurs fabrications tournent à la fois si la mémoire de la carte le permet.
        GET  /sante            → {"pret": bool, "carte": "...", "file": n, "places": n}
        POST /travaux          → {"id": "..."}       (corps : les réglages ci-dessous)
        GET  /travaux/<id>     → {"etat": "en file|en cours|fini|échec", ...résultat}
  - Serverless RunPod (par défaut).

Réglages : image_base64 (ou image_url), ou plusieurs vues du même sujet : images_base64 [face, profil, dos…] avec
multi = "alterne" (une vue par pas de calcul) | "moyenne" (toutes les vues à chaque pas, plus lent) ; seed (42),
resolution ("1024_cascade" ; "512", "1024", "1536_cascade"), decimation (60000), texture_size (1024) ;
steps {structure, shape, texture} (12 par défaut) et guidance {structure, shape, texture} (7,5 · 7,5 · 1) : étapes
et guidage de chaque phase ; max_tokens (taille maximale de la forme en 1536 : selon la mémoire de la carte).
Résultat : {"ok": true, "glb_gzip_base64": "...", "octets": n, "secondes": {...}, "resolution_effective": n}
ou {"ok": false, "erreur": "..."}
"""
import base64
import collections
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
GO = 1024 ** 3
# Ce qui est propre à chaque fabrication (chacune tourne dans son fil) : son avancement, ses vues, ses tirages au sort.
ICI = threading.local()

# Avancement d'une fabrication (lu par GET /travaux/<id>) : étape, pas de calcul, pourcentage.
ETAPES = [("prep", "Préparation de l'image", 4), ("structure", "Structure", 12), ("forme", "Forme", 34),
          ("textures", "Textures", 26), ("decodage", "Décodage", 9), ("export", "Export du modèle", 15)]


def avancer(cle, frac=0.0, detail=""):
    p = getattr(ICI, "progres", None)
    if p is None:
        return
    i = [k for k, _, _ in ETAPES].index(cle)
    debut, poids = sum(w for _, _, w in ETAPES[:i]), ETAPES[i][2]
    p.update(cle=cle, etape=ETAPES[i][1], numero=i + 1, total=len(ETAPES), detail=detail,
             pct=min(99, round(debut + poids * max(0.0, min(1.0, frac)))))


# Plusieurs vues (astuce de TRELLIS v1, sans réentraînement) : le modèle ne connaît qu'une image à la fois ; on lui
# présente les vues tour à tour (« alterne ») ou on fait la moyenne de ses prédictions sur toutes (« moyenne »).
def brancher_multi_vues():
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
    base = FlowEulerSampler._inference_model

    def inference(self, model, x_t, t, cond=None, **kw):
        n = cond.shape[0] if torch.is_tensor(cond) else 1
        multi = getattr(ICI, "multi", None)
        if multi is None or n <= 1:
            return base(self, model, x_t, t, cond, **kw)
        if multi["mode"] == "alterne":
            if t != multi["t"]:  # nouveau pas (le guidage appelle deux fois par pas : on garde la même vue)
                multi["t"], multi["pas"] = t, multi["pas"] + 1
            i = multi["pas"] % n
            return base(self, model, x_t, t, cond[i:i + 1], **kw)
        return sum(base(self, model, x_t, t, cond[i:i + 1], **kw) for i in range(n)) / n

    FlowEulerSampler._inference_model = inference
    get_cond = PIPELINE.get_cond

    def get_cond_vues(image, resolution, include_neg_cond=True):
        if isinstance(image, list) and len(image) == 1 and isinstance(image[0], list):
            image = image[0]  # run() emballe l'image dans une liste : ici, c'est déjà la liste des vues
        return get_cond(image, resolution, include_neg_cond)

    PIPELINE.get_cond = get_cond_vues


def suivre_pas():
    """Les échantillonneurs de TRELLIS comptent leurs pas avec tqdm : on écoute ce compte."""
    import trellis2.pipelines.samplers.flow_euler as fe

    def compteur(it, desc="", disable=False, **_):
        items = list(it)
        p = getattr(ICI, "progres", None)
        p = {} if p is None else p
        cle = "structure" if "sparse" in desc else "forme" if "shape" in desc else "textures" if "texture" in desc else None
        passe = 0
        if cle == "forme":  # en cascade, la forme se fait en deux passes (basse puis haute résolution)
            passe = p.get("_passes_faites", 0)
            p["_passes_faites"] = passe + 1
        nb = p.get("_passes", 1) if cle == "forme" else 1
        for i, x in enumerate(items):
            if cle:
                avancer(cle, (passe + i / max(1, len(items))) / nb, f"pas {i + 1}/{len(items)}" + (f" · passe {passe + 1}/{nb}" if nb > 1 else ""))
            yield x

    fe.tqdm = compteur


class TorchDuFil:
    """Le module torch tel que le voit le pipeline de TRELLIS.2, avec un générateur de hasard propre à chaque fil :
    deux fabrications simultanées ne mélangent pas leurs tirages. Même graine → même forme, en parallèle ou non (et
    les mêmes tirages qu'avant : un générateur neuf donne la même suite que torch.manual_seed)."""

    def __getattr__(self, nom):
        return getattr(torch, nom)

    @staticmethod
    def manual_seed(graine):
        g = torch.Generator()
        g.manual_seed(int(graine))
        ICI.hasard = g
        return g

    @staticmethod
    def randn(*taille, **kw):
        g = getattr(ICI, "hasard", None)
        if g is not None and "generator" not in kw and kw.get("device") in (None, "cpu"):
            kw["generator"] = g
        return torch.randn(*taille, **kw)


def proteger_etats_partages():
    """Deux réglages du pipeline sont partagés par toutes les fabrications : la taille d'image de l'analyseur
    (512 ou 1024) et la résolution du décodeur. Chaque passage qui les règle puis s'en sert se fait seul."""
    import trellis2.pipelines.trellis2_image_to_3d as tp
    tp.torch = TorchDuFil()
    verrou_cond, verrou_decodeur = threading.Lock(), threading.Lock()
    get_cond, decode_forme = PIPELINE.get_cond, PIPELINE.decode_shape_slat

    def get_cond_seul(*a, **k):
        with verrou_cond:
            return get_cond(*a, **k)

    def decode_forme_seul(*a, **k):
        with verrou_decodeur:
            return decode_forme(*a, **k)

    PIPELINE.get_cond = get_cond_seul
    PIPELINE.decode_shape_slat = decode_forme_seul


MAX_SORTIE = 10 * 1024 * 1024 - 64 * 1024  # réponse Serverless : on reste sous 10 Mo
MEMOIRE = 0  # mémoire de la carte (Go)


def charger():
    """Chargé une fois par démarrage (DINOv3 et RMBG-2.0 se téléchargent ici, avec HF_TOKEN)."""
    global PIPELINE, MEMOIRE
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    MEMOIRE = torch.cuda.get_device_properties(0).total_memory / GO
    PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    # Grande carte : tous les modèles restent dessus (par défaut, chacun y monte puis en redescend à chaque étape).
    PIPELINE.low_vram = MEMOIRE < 32
    PIPELINE.cuda()
    suivre_pas()
    brancher_multi_vues()
    proteger_etats_partages()
    decode = PIPELINE.decode_latent

    def decode_suivi(*a, **k):
        avancer("decodage", 0.1)
        return decode(*a, **k)

    PIPELINE.decode_latent = decode_suivi
    PRET.set()
    print(f"[atelier] prêt en {time.time() - t0:.0f} s sur {torch.cuda.get_device_name(0)} ({MEMOIRE:.0f} Go,"
          f" modèles {'déplacés à chaque étape' if PIPELINE.low_vram else 'gardés sur la carte'})", flush=True)


# Taille maximale de la forme en 1536 (jetons) : au-delà, TRELLIS.2 baisse la résolution. Plus haut sur une grande carte.
def jetons_max():
    return 65536 if MEMOIRE >= 70 else 49152


def reglages_phases(inp):
    """Étapes et guidage de chaque phase (structure, forme, textures) ; ce qui manque garde la valeur de TRELLIS.2."""
    etapes, guidage = inp.get("steps") or {}, inp.get("guidance") or {}
    out = {}
    for cle, nom in (("structure", "sparse_structure_sampler_params"), ("shape", "shape_slat_sampler_params"), ("texture", "tex_slat_sampler_params")):
        p = {}
        if etapes.get(cle) is not None:
            p["steps"] = max(4, min(100, int(etapes[cle])))
        if guidage.get(cle) is not None:
            p["guidance_strength"] = max(0.0, min(20.0, float(guidage[cle])))
        out[nom] = p
    return out


def decoder(b64):
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


def lire_image(inp):
    if inp.get("image_url"):
        with urllib.request.urlopen(inp["image_url"], timeout=60) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    return decoder(inp["image_base64"])


def manque_de_memoire(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def fabriquer(inp, limite=None, progres=None, jetons=None):
    import o_voxel
    ICI.progres = progres if progres is not None else {}
    try:
        vues = inp.get("images_base64") or []
        if not (vues or inp.get("image_base64") or inp.get("image_url")):
            return {"ok": False, "erreur": "image_base64, images_base64 ou image_url requis"}
        t = {}
        a = time.time()
        resolution = inp.get("resolution", "1024_cascade")
        ICI.progres.update(_passes=2 if "cascade" in resolution else 1, _passes_faites=0)
        avancer("prep")
        if len(vues) > 1:
            image = [PIPELINE.preprocess_image(decoder(v)) for v in vues]  # chaque vue détourée et cadrée
            ICI.multi = {"mode": "moyenne" if inp.get("multi") == "moyenne" else "alterne", "pas": -1, "t": None}
        else:
            image = decoder(vues[0]) if vues else lire_image(inp)
            ICI.multi = None
        jetons = jetons or int(inp.get("max_tokens") or jetons_max())
        try:
            with torch.inference_mode():
                maillages, (forme, texture, res) = PIPELINE.run(
                    image, seed=int(inp.get("seed", 42)), pipeline_type=resolution, preprocess_image=len(vues) <= 1,
                    max_num_tokens=jetons, return_latent=True, **reglages_phases(inp))
            mesh = maillages[0]
            del forme, texture
        finally:
            ICI.multi = None
        t["generation"] = round(time.time() - a, 1)
        a = time.time()
        avancer("export", 0.05, "simplification")
        mesh.simplify(16_777_216)  # limite de nvdiffrast
        avancer("export", 0.3, "dépliage et cuisson des textures")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(inp.get("decimation", 60_000)), texture_size=int(inp.get("texture_size", 1024)),
            remesh=True, remesh_band=1, remesh_project=0, verbose=False,
        )
        del mesh
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
        return {"ok": True, "glb_gzip_base64": out.decode("ascii"), "octets": len(data), "secondes": t,
                "resolution_effective": int(res), "jetons_max": jetons}
    except Exception as e:  # l'atelier doit toujours répondre
        if manque_de_memoire(e):
            return {"ok": False, "memoire": True, "erreur": f"mémoire de la carte insuffisante : {e}"}
        return {"ok": False, "erreur": str(e), "trace": traceback.format_exc()}
    finally:
        ICI.progres = None
        torch.cuda.empty_cache()


class Carte:
    """La carte graphique, partagée en places selon sa mémoire : une fabrication en 512 en prend une, une plus grande
    deux, une reprise après un manque de mémoire la prend toute. Chacune passe à son tour (pas de resquille)."""

    def __init__(self, places):
        self.places, self.prises, self.attente, self.c = places, 0, collections.deque(), threading.Condition()

    def prendre(self, n):
        n = max(1, min(n, self.places))
        with self.c:
            moi = object()
            self.attente.append(moi)
            self.c.wait_for(lambda: self.attente[0] is moi and self.prises + n <= self.places)
            self.attente.popleft()
            self.prises += n
            self.c.notify_all()
        return n

    def rendre(self, n):
        with self.c:
            self.prises -= n
            self.c.notify_all()


def places_carte():
    """Combien de places sur la carte : PLACES si réglé, sinon selon sa mémoire (80 Go : 4 ; 48 Go : 2 ; 24 Go : 1)."""
    if os.environ.get("PLACES"):
        return max(1, int(os.environ["PLACES"]))
    return 4 if MEMOIRE >= 70 else 2 if MEMOIRE >= 40 else 1


def mode_pod():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    jeton = os.environ.get("ATELIER_TOKEN", "")
    travaux, file = {}, queue.Queue()
    activite = [time.time()]
    carte = [None]

    def veilleur():
        """Sécurité : sans aucune demande pendant ARRET_INACTIF minutes, le Pod se supprime lui-même (plus aucun frais).
        RunPod fournit au Pod son identifiant et une clé limitée à lui-même (RUNPOD_POD_ID, RUNPOD_API_KEY)."""
        minutes = float(os.environ.get("ARRET_INACTIF", "30"))
        pod, cle = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
        while minutes > 0 and pod and cle:
            time.sleep(60)
            occupe = any(t["etat"] in ("en file", "en cours") for t in travaux.values())
            if not occupe and time.time() - activite[0] > minutes * 60:
                print(f"[atelier] inactif depuis {minutes:.0f} min : suppression du Pod", flush=True)
                req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE", headers={"Authorization": f"Bearer {cle}"})
                try:
                    urllib.request.urlopen(req, timeout=30)
                except Exception as e:
                    print(f"[atelier] suppression impossible : {e}", flush=True)
                return

    def ouvrier():
        PRET.wait()
        while True:
            tid = file.get()
            t = travaux[tid]
            entree = t.pop("entree")
            # Une fabrication en 512 (essai rapide) prend une place, les autres deux.
            n = carte[0].prendre(1 if entree.get("resolution") == "512" else 2)
            try:
                t["etat"], t["debut"], t["progres"] = "en cours", time.time(), {}
                r = fabriquer(entree, progres=t["progres"])
            finally:
                carte[0].rendre(n)
            if r.get("memoire"):
                # Manque de mémoire : on la refait seule sur la carte, avec la taille de forme d'origine de TRELLIS.2.
                print("[atelier] mémoire insuffisante : reprise seule sur la carte", flush=True)
                n = carte[0].prendre(carte[0].places)
                try:
                    t["progres"] = {}
                    r = fabriquer(entree, progres=t["progres"], jetons=49152)
                finally:
                    carte[0].rendre(n)
            t.pop("progres", None)
            t.update(r, etat="fini" if r.get("ok") else "échec", duree=round(time.time() - t["debut"], 1))

    def demarrer():
        charger()
        carte[0] = Carte(places_carte())
        print(f"[atelier] {carte[0].places} place(s) sur la carte : jusqu'à {carte[0].places} essais en 512, ou {max(1, carte[0].places // 2)} fabrication(s) haute qualité à la fois", flush=True)
        for _ in range(carte[0].places):
            threading.Thread(target=ouvrier, daemon=True).start()

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
            if self.path == "/sante":  # consulter l'état ne compte pas comme une activité
                return self.repondre(200, {"pret": PRET.is_set(), "carte": torch.cuda.get_device_name(0), "file": file.qsize(),
                                           "places": carte[0].places if carte[0] else None, "memoire_go": round(MEMOIRE)})
            activite[0] = time.time()
            if not self.autorise():
                return self.repondre(401, {"erreur": "jeton"})
            if self.path.startswith("/travaux/"):
                t = travaux.get(self.path.split("/")[-1])
                if not t:
                    return self.repondre(404, {"erreur": "inconnu"})
                vue = {k: v for k, v in t.items() if k not in ("entree", "progres")}
                if t["etat"] == "en cours":
                    vue["progres"] = {k: v for k, v in dict(t.get("progres") or {}).items() if not k.startswith("_")}
                    vue["ecoule"] = round(time.time() - t["debut"])
                elif t["etat"] == "en file":
                    vue["position"] = sum(1 for x in travaux.values() if x["etat"] == "en file" and x["cree"] <= t["cree"])
                    vue["pret"] = PRET.is_set()
                return self.repondre(200, vue)
            self.repondre(404, {"erreur": "chemin"})

        def do_POST(self):
            activite[0] = time.time()
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

    threading.Thread(target=demarrer, daemon=True).start()
    threading.Thread(target=veilleur, daemon=True).start()
    print("[atelier] serveur du Pod sur le port 8000", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), Accueil).serve_forever()


if __name__ == "__main__":
    if os.environ.get("MODE") == "pod":
        mode_pod()
    else:
        import runpod
        charger()
        runpod.serverless.start({"handler": lambda job: fabriquer(job.get("input") or {}, MAX_SORTIE)})
