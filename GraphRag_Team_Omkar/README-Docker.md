# Praxia5Chronic - Server Deployment via Docker Compose

This document provides instructions for deploying the entire Praxia5Chronic application on a server using Docker Compose.

The included `docker-compose.yml` spins up:
1. **The FastAPI Application Container**: Serves the GraphRAG dashboard at port `8000`.
2. **The Ollama Container**: A dedicated LLM server exposed on port `11434` for the API container to hit safely. 

## 1. Prerequisites 
Ensure the server you are migrating this code to has:
- Docker installed (`sudo apt-get install docker.io`)
- Docker Compose installed (`sudo apt-get install docker-compose-plugin` or `docker-compose`)
- (Optional) NVIDIA drivers and the NVIDIA Container Toolkit for GPU acceleration.

## 2. Booting the Application
Navigate to the extracted repository folder containing the `docker-compose.yml` file and run:

```bash
docker-compose up -d --build
```
> **Note**: The first run will build the `Dockerfile` for the Python API, properly installing system OS-level dependencies like `tesseract-ocr` and `poppler-utils` needed for document parsing.

## 3. Mandatory Model Configuration (Do this once)
By default, the Ollama container comes empty. The LLMs used by the framework (`deepseek-r1:8b` and the `Med42` model) must be downloaded manually into the running container before the GraphRAG dashboard is usable. 

```bash
# Pull DeepSeek-R1 (8B)
docker exec -it updated_internship_gemini-ollama-1 ollama run deepseek-r1:8b

# Pull Med42 Healthcare Model
docker exec -it updated_internship_gemini-ollama-1 ollama run hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M
```
*(Note: If you rename the project folder, Docker Compose changes the container prefix. You can always use `docker ps` to find the exact container name).*

## 4. Enabling GPU Acceleration (Optional but recommended)
If the server has an Nvidia GPU, it is highly recommended you utilize the GPU pass-through capabilities of Docker. By default, the `docker-compose.yml` handles inference strictly on the CPU for maximum platform compatibility.

To enable the GPU:
1. Ensure the server has installed the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Open the `docker-compose.yml` file.
3. Uncomment the `deploy` block situated dynamically at the bottom under the `ollama` service configuration.

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
4. Restart the containers with `docker-compose down` followed by `docker-compose up -d`.
