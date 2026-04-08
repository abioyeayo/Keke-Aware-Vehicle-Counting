# Keke-Aware-Vehicle-Counting
The dataset, training, inference, and evaluation codes for the "Keke-Aware Vehicle Counting for Traffic Measurement Using YOLO: Dataset and Field Evaluation" research paper.


## Docker setup

This project can be run inside Docker Compose with GPU support for training and evaluation.

### Requirements

- Ubuntu 24.04
- Docker
- Docker Compose
- NVIDIA driver installed and working
- NVIDIA Container Toolkit installed
- A CUDA-capable GPU

### Verify GPU access on the host

Run:

```bash
nvidia-smi
```

If the GPU is detected correctly, Docker should be able to use it.

### Build and start the container

The compose configuration uses the `HOST_UID` and `HOST_GID` values defined in the project `.env` file so files created inside the container are owned correctly on the host system.


Build and start the container:
```bash
sudo docker compose build
sudo docker compose up -d
```

Open a shell inside the container:
```bash
sudo docker compose exec keke-dev bash
```

### Verify GPU access inside the container

Inside the container, run:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

### Train the model

Inside the container, run:
```bash
python train.py --model yolo26l.pt --epochs 100 --imgsz 640 --name train_keke_rev1
```

### Run the traffic counting analysis

Inside the container, run:
```bash
python vehicle_counter_analysis.py
```

### Stop the container

From the host terminal:
```bash
sudo docker compose down
```
