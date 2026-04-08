FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=appuser

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    curl \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN mkdir -p /workspace /home/${USERNAME}/.cache /home/${USERNAME}/.ssh && \
    chown -R ${USERNAME}:${USERNAME} /workspace /home/${USERNAME}

USER ${USERNAME}

CMD ["sleep", "infinity"]