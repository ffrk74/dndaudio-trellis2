# Atelier TRELLIS.2 (DndAudio)

Service RunPod Serverless : on lui envoie une image (fond transparent), il renvoie un objet 3D GLB texturé.
Modèle : [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT). Utilise DINOv3 (Meta) et RMBG-2.0 (Bria),
à accès contrôlé : ils ne sont pas inclus dans l'image et se téléchargent au démarrage avec `HF_TOKEN`
(variable secrète du service RunPod, d'un compte qui a accepté leurs conditions).

- Construction : GitHub Actions publie `ghcr.io/<compte>/<dépôt>:<commit>` à chaque envoi sur `main`.
- Service RunPod : GPU A40 (48 Go), Workers Min 0, Workers Max 1, Idle Timeout court.
- Entrée : `image_base64` ou `image_url`, `seed`, `resolution`, `decimation`, `texture_size`, `steps` et `guidance`
  (étapes et guidage des phases structure / forme / textures), `max_tokens` (voir `handler.py`).
- Pod : les modèles restent sur la carte (≥ 32 Go) ; plusieurs fabrications à la fois selon la mémoire (80 Go : deux
  en haute qualité ou quatre essais en 512 ; 48 Go : une, ou deux essais) ; reprise seule en cas de manque de mémoire.
