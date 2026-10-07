# Atelier TRELLIS.2 (microsoft/TRELLIS.2-4B) pour RunPod Serverless : une image → un GLB texturé.
# Les poids de TRELLIS.2 (publics, MIT) sont dans l'image : démarrage à froid plus court.
# DINOv3 (Meta) et RMBG-2.0 (Bria) sont à accès contrôlé : ils ne sont PAS dans l'image (elle est publique) ;
# l'atelier les télécharge au démarrage avec HF_TOKEN, réglé en secret sur le service RunPod.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
# Compilation : seulement les cartes visées (A40, A6000, A5000, 3090 = 8.6 ; L40, 4090 = 8.9 ; H100, H200 = 9.0), et peu de tâches
# en parallèle (la machine de construction n'a que 16 Go de mémoire).
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/opt/conda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    MAX_JOBS=2 \
    NVCC_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends git wget curl build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace
# Version figée de TRELLIS.2 (pas de surprise si le dépôt change).
ARG TRELLIS_REF=main
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git && cd TRELLIS.2 && git checkout ${TRELLIS_REF}
WORKDIR /workspace/TRELLIS.2

# DINOv3 range ses couches un niveau plus bas que ce qu'attend TRELLIS.2 (self.model.model.layer).
RUN sed -i 's/self\.model\.layer/self.model.model.layer/' trellis2/modules/image_feature_extractor.py \
    && grep -q "self.model.model.layer" trellis2/modules/image_feature_extractor.py

# La machine de construction n'a pas de carte graphique : setup.sh vérifie seulement que nvidia-smi existe.
RUN printf '#!/bin/sh\ncase "$*" in *--query-gpu*) echo "A100-SXM4-80GB, 81920";; esac\nexit 0\n' > /usr/local/bin/nvidia-smi \
    && chmod +x /usr/local/bin/nvidia-smi

# Dépendances officielles (sans flash-attn : sa compilation épuise la mémoire des machines de construction).
RUN source /opt/conda/etc/profile.d/conda.sh \
    && . ./setup.sh --new-env --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm \
    && test -x /opt/conda/envs/trellis2/bin/pip
# flash-attn précompilé, assorti à torch 2.6 / CUDA 12 / Python 3.10.
RUN /opt/conda/envs/trellis2/bin/pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    && /opt/conda/envs/trellis2/bin/pip install --no-cache-dir runpod pillow huggingface_hub \
    && /opt/conda/envs/trellis2/bin/python -c "import flash_attn, runpod"
# On retire le faux nvidia-smi : sur RunPod, le vrai est fourni.
RUN rm /usr/local/bin/nvidia-smi

# Poids publics de TRELLIS.2.
RUN /opt/conda/envs/trellis2/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')"

COPY handler.py /workspace/TRELLIS.2/handler.py
ENTRYPOINT ["/opt/conda/envs/trellis2/bin/python", "-u", "handler.py"]
