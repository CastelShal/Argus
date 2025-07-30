![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Open Source](https://img.shields.io/badge/status-alpha-indigo)

# Argus

**Argus** is a modular facial recognition pipeline designed for real-time face detection, identification, and tracking across multiple video streams. Built with scalability and privacy in mind, it aims to support home and corporate security.

## Project Purpose

Argus addresses the need for a private, on-premise security system capable of processing live camera feeds in real-time. It is built with modularity and performance in mind.

## Project Goals

- Replace expensive cloud-based surveillance with a **local, secure alternative**
- Support **real-time monitoring** using standard IP/RTSP video streams
- Provide **logging and alerting** on unknown individuals
- Remain **fully open-source and auditable**
- Be **scalable** from home use to enterprise deployment

## Key Features
-  Per-frame facial detection using OpenCV + DLib
-  Web-based dashboard for visualization and alerts
-  Works with Any IP/RTSP Camera  
-  Facial recognition using FaceNet
-  Local, Self Hosted Processing  
-  Dockerized deployment for easy containerization

## System Requirements

### Recommended Hardware
- **CPU**: Intel i7 / AMD Ryzen 7
- **RAM**: 16 GB
- **GPU**: GTX 1660 / AMD 6600 XT+
- **Camera**: IP/RTSP Stream capable (720p or higher)

### Software
- **OS**: Debian or Fedora / Windows 10+
- **Dependencies**: Python 3.10, OpenCV, DLib, TensorFlow/PyTorch, Flask, FaceNet



## Deployment Instructions
### 1. Docker
- Pull the docker image from [container repository](ghcr.io/castelshal/argus:latest)
- Setup the training folder as outlined in [training guide](./training/instructions.md)
- Replace the path in docker-compose.yml with your training folder
- Ensure the IP streams are accessible on your network
- Run `docker-compose up .`
- Visit `http://localhost:5000/` for the Web UI
### 2. Direct Deploy
- Clone the repository
- Install the following dependencies:
``` 
build-essential cmake python3-dev python3-pip libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libboost-all-dev 
```
- Optionally you can install:
```
libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
```
- Create a virtual environment (Python 3.10.18)
- `pip install -r requirements.txt`.
- Setup the training folder as outlined in the [training guide](./training/instructions.md).
- Run `python src/train.py` to complete training.
- Run `python src/main.py` to run.
- Visit `http://localhost:5000` for the Web UI