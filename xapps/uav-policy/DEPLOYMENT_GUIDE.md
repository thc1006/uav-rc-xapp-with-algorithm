# Deployment Guide - UAV Policy xApp

This guide provides step-by-step instructions for deploying the UAV Policy xApp in different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Deployment](#local-development-deployment)
3. [Docker Container Deployment](#docker-container-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Integration with E2 Simulator](#integration-with-e2-simulator)
6. [Verification and Testing](#verification-and-testing)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 1+ cores (2+ recommended)
- **Memory**: 512 MB+ (1 GB recommended)
- **Disk**: 1 GB+ free space
- **Network**: HTTP/REST API access (port 5000 default)

### Software Requirements

#### For Local Deployment
- Python 3.10+ (3.11 recommended)
- pip package manager
- git (for source control)

#### For Docker Deployment
- Docker CE 20.10+ or Docker Desktop
- docker-compose (optional)

#### For Kubernetes Deployment
- Kubernetes 1.21+ cluster
- kubectl CLI 1.21+
- kustomize 4.0+ (for K8s customization)
- Helm 3.0+ (optional, for advanced deployments)

## Local Development Deployment

### Step 1: Clone Repository

```bash
cd /path/to/uav-rc-xapp-with-algorithms/xapps/uav-policy
```

### Step 2: Create Virtual Environment

```bash
# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies

```bash
# Install runtime dependencies
pip install flask werkzeug

# Install development dependencies (optional)
pip install pytest pytest-cov
```

### Step 4: Set Environment Variables

```bash
# Configure server
export LOG_LEVEL=INFO
export SERVER_HOST=0.0.0.0
export SERVER_PORT=5000
export DEBUG=false
```

### Step 5: Run Tests (Recommended)

```bash
# Set Python path for imports
export PYTHONPATH="$(pwd)/src"

# Run all tests
python3 -m pytest tests/ -v --cov=src/uav_policy --cov-report=term-missing

# Expected output: 13 tests pass with ~68% coverage
```

### Step 6: Start Server

```bash
# Start the server
python3 -m uav_policy.main

# Expected output:
# 2025-11-21 02:30:00 - uav_policy.main - INFO - Starting UAV policy xApp server...
# 2025-11-21 02:30:00 - uav_policy.main - INFO - Server configuration: host=0.0.0.0, port=5000, debug=false
# WARNING in app.run_simple (werkzeug.py:1036): This is a development server. Do not use it in production.
# Running on http://0.0.0.0:5000
```

### Step 7: Verify Server is Running

In another terminal:

```bash
# Health check
curl http://localhost:5000/health

# Response:
# {"status":"healthy","timestamp":"2025-11-21T02:30:00.000000","service":"uav-policy-xapp"}
```

### Step 8: Test E2 Indication Endpoint

```bash
# Submit E2 indication
curl -X POST http://localhost:5000/e2/indication \
  -H "Content-Type: application/json" \
  -d '{
    "uav_id": "uav-001",
    "position": {"x": 100.0, "y": 50.0, "z": 120.0},
    "path_position": 0.5,
    "radio_snapshot": {
      "serving_cell_id": "cell-A",
      "neighbor_cell_ids": ["cell-B"],
      "rsrp_serving": -88.0,
      "rsrp_best_neighbor": -82.0,
      "prb_utilization_serving": 0.75,
      "prb_utilization_slice": 0.6
    }
  }'

# Expected response:
# {
#   "uav_id": "uav-001",
#   "target_cell_id": "cell-A",
#   "slice_id": null,
#   "prb_quota": 5,
#   "reason": "...",
#   "timestamp": "2025-11-21T02:30:00.000000"
# }
```

## Docker Container Deployment

### Step 1: Build Docker Image

```bash
# Navigate to project directory
cd /path/to/uav-rc-xapp-with-algorithms/xapps/uav-policy

# Build image
docker build -t uav-policy-xapp:latest .

# Verify build
docker images | grep uav-policy-xapp
```

### Step 2: Run Container

```bash
# Run in foreground (for testing)
docker run -it \
  --name uav-policy-xapp \
  -p 5000:5000 \
  -e LOG_LEVEL=INFO \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=5000 \
  -e DEBUG=false \
  uav-policy-xapp:latest

# Or run in background
docker run -d \
  --name uav-policy-xapp \
  -p 5000:5000 \
  -e LOG_LEVEL=INFO \
  uav-policy-xapp:latest

# View logs
docker logs -f uav-policy-xapp
```

### Step 3: Test Container

```bash
# Health check
curl http://localhost:5000/health

# Check logs
docker logs uav-policy-xapp | tail -20

# Get container stats
docker stats uav-policy-xapp
```

### Step 4: Stop and Clean Up

```bash
# Stop container
docker stop uav-policy-xapp

# Remove container
docker rm uav-policy-xapp

# Remove image (if needed)
docker rmi uav-policy-xapp:latest
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  uav-policy-xapp:
    build:
      context: .
    ports:
      - "5000:5000"
    environment:
      LOG_LEVEL: INFO
      SERVER_HOST: 0.0.0.0
      SERVER_PORT: 5000
      DEBUG: "false"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    # Optional: resource limits
    # deploy:
    #   resources:
    #     limits:
    #       cpus: '0.5'
    #       memory: 512M
```

Run with Docker Compose:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f uav-policy-xapp

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Step 1: Create Namespace

```bash
# Create namespace for O-RAN RIC
kubectl create namespace oran-ric

# Verify namespace
kubectl get namespaces | grep oran-ric
```

### Step 2: Load Docker Image to Kubernetes

If using a local Kubernetes cluster (minikube, kind):

```bash
# Minikube
minikube image load uav-policy-xapp:latest

# Or kind
kind load docker-image uav-policy-xapp:latest --name <cluster-name>

# Or push to registry
docker tag uav-policy-xapp:latest <registry>/uav-policy-xapp:latest
docker push <registry>/uav-policy-xapp:latest
```

### Step 3: Deploy Using Kustomize

```bash
# Navigate to k8s directory
cd k8s

# Preview resources that will be created
kubectl kustomize . | head -50

# Apply all resources
kubectl apply -k .

# Or apply individual resources
kubectl apply -f serviceaccount.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Step 4: Verify Deployment

```bash
# Check deployment status
kubectl get deployment -n oran-ric uav-policy-xapp

# Expected output:
# NAME               READY   UP-TO-DATE   AVAILABLE   AGE
# uav-policy-xapp    1/1     1            1           10s

# Check pods
kubectl get pods -n oran-ric -l app=uav-policy-xapp

# Check services
kubectl get svc -n oran-ric uav-policy-xapp

# Check logs
kubectl logs -n oran-ric -l app=uav-policy-xapp

# Stream logs
kubectl logs -n oran-ric -l app=uav-policy-xapp -f
```

### Step 5: Port Forward for Local Testing

```bash
# Forward port 5000 locally
kubectl port-forward -n oran-ric svc/uav-policy-xapp 5000:5000

# In another terminal, test the service
curl http://localhost:5000/health
```

### Step 6: Configure Service Access

For external access, modify the service type:

```bash
# Patch service to use LoadBalancer
kubectl patch svc -n oran-ric uav-policy-xapp -p '{"spec": {"type": "LoadBalancer"}}'

# Or use Ingress
kubectl create ingress -n oran-ric uav-policy-ingress \
  --rule="uav-policy.example.com/*=uav-policy-xapp:5000"
```

### Step 7: Update ConfigMap

To change configuration without redeploying:

```bash
# Edit configmap
kubectl edit configmap -n oran-ric uav-policy-config

# Or apply new config
kubectl patch configmap -n oran-ric uav-policy-config --type merge \
  -p '{"data":{"log_level":"DEBUG"}}'
```

### Step 8: Scale Deployment

```bash
# Scale to multiple replicas
kubectl scale deployment -n oran-ric uav-policy-xapp --replicas=3

# Watch scaling
kubectl rollout status deployment/uav-policy-xapp -n oran-ric
```

### Step 9: Clean Up

```bash
# Delete all resources
kubectl delete -k k8s/ --namespace oran-ric

# Or delete individual resources
kubectl delete deployment,service,configmap,sa -n oran-ric -l app=uav-policy-xapp

# Delete namespace (optional)
kubectl delete namespace oran-ric
```

## Integration with E2 Simulator

### Architecture

```
E2 Simulator
    |
    | HTTP POST
    v
UAV Policy xApp (/e2/indication)
    |
    | ResourceDecision
    v
RC xApp / E2 Node
```

### Configuration for E2 Simulator

Configure your E2 simulator to send indications to the xApp:

```yaml
# E2 Simulator Config
e2_indication_handler:
  type: http
  endpoint: http://uav-policy-xapp:5000/e2/indication
  method: POST
  headers:
    Content-Type: application/json
  retry_policy:
    max_retries: 3
    backoff_multiplier: 1.5
    timeout: 5s
```

### Sample Indication Payload

The E2 simulator should send data in this format:

```json
{
  "uav_id": "uav-001",
  "position": {
    "x": 1000.0,
    "y": 2000.0,
    "z": 500.0
  },
  "path_position": 0.45,
  "slice_id": "uav-hd-video",
  "radio_snapshot": {
    "serving_cell_id": "gNB-1",
    "neighbor_cell_ids": ["gNB-2", "gNB-3"],
    "rsrp_serving": -92.0,
    "rsrp_best_neighbor": -86.0,
    "prb_utilization_serving": 0.82,
    "prb_utilization_slice": 0.65
  },
  "flight_plan": {
    "segments": [
      {
        "start_pos": 0.0,
        "end_pos": 0.5,
        "planned_cell_id": "gNB-1",
        "slice_id": "uav-hd-video",
        "base_prb_quota": 25
      },
      {
        "start_pos": 0.5,
        "end_pos": 1.0,
        "planned_cell_id": "gNB-2",
        "slice_id": "uav-hd-video",
        "base_prb_quota": 30
      }
    ]
  },
  "service_profile": {
    "name": "uav-hd-video",
    "target_bitrate_mbps": 15.0,
    "min_sinr_db": -5.0
  }
}
```

### Integration with RC xApp

Send ResourceDecision to RC xApp (implement separately):

```python
# Example: Forward decision to RC xApp
decision_response = requests.post(
    "http://rc-xapp:8080/control",
    json={
        "uav_id": decision.uav_id,
        "target_cell": decision.target_cell_id,
        "slice": decision.slice_id,
        "prb_quota": decision.prb_quota,
    }
)
```

## Verification and Testing

### Health Checks

```bash
# HTTP endpoint health check
curl http://localhost:5000/health

# Response includes timestamp for freshness check
# Status code 200 = healthy
```

### Load Testing

```bash
# Install ab (Apache Bench) or hey
# apt-get install apache2-utils  # Ubuntu
# brew install hey  # macOS

# Run load test
ab -n 100 -c 10 -p request.json -T application/json \
  http://localhost:5000/e2/indication

# Or with hey
hey -n 1000 -c 50 -m POST -H "Content-Type: application/json" \
  -d @request.json http://localhost:5000/e2/indication
```

### Performance Monitoring

```bash
# Monitor resource usage (Docker)
docker stats uav-policy-xapp

# Monitor resource usage (Kubernetes)
kubectl top pods -n oran-ric -l app=uav-policy-xapp

# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/health

# curl-format.txt:
# time_namelookup:  %{time_namelookup}\n
# time_connect:     %{time_connect}\n
# time_appconnect:  %{time_appconnect}\n
# time_pretransfer: %{time_pretransfer}\n
# time_redirect:    %{time_redirect}\n
# time_starttransfer: %{time_starttransfer}\n
# time_total:       %{time_total}\n
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

#### 2. Module Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="$(pwd)/src"

# Verify package structure
ls -la src/uav_policy/
ls -la src/uav_policy/__init__.py
```

#### 3. Container Won't Start

```bash
# Check logs
docker logs uav-policy-xapp

# Try running interactively
docker run -it uav-policy-xapp:latest /bin/bash

# Check image layers
docker history uav-policy-xapp:latest
```

#### 4. Kubernetes Pod Failing

```bash
# Describe pod for events
kubectl describe pod -n oran-ric <pod-name>

# Check logs
kubectl logs -n oran-ric <pod-name> --previous

# Check resource usage
kubectl top pod -n oran-ric <pod-name>

# Exec into pod for debugging
kubectl exec -it -n oran-ric <pod-name> -- /bin/bash
```

#### 5. Connection Refused

```bash
# Check if service is listening
netstat -tuln | grep 5000  # Linux
lsof -i :5000  # macOS

# Check firewall
ufw allow 5000  # Ubuntu
```

### Debug Mode

Enable debug logging:

```bash
# Local
export LOG_LEVEL=DEBUG
python3 -m uav_policy.main

# Docker
docker run -e LOG_LEVEL=DEBUG -p 5000:5000 uav-policy-xapp:latest

# Kubernetes
kubectl patch configmap -n oran-ric uav-policy-config \
  --type merge -p '{"data":{"log_level":"DEBUG"}}'
```

## Performance Tuning

### For High-Volume E2 Indications

```bash
# Increase history size (environment variable would be needed)
# Currently hardcoded to 1000 decisions

# Increase Flask workers
# Use Gunicorn in production:
pip install gunicorn
gunicorn --workers 4 --threads 2 -b 0.0.0.0:5000 uav_policy.server:create_app()
```

### Kubernetes Resource Optimization

Edit `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    cpu: 200m        # Increase for higher throughput
    memory: 256Mi    # Increase if handling large histories
  limits:
    cpu: 1000m
    memory: 1Gi
```

## Monitoring and Metrics

### Structured Logging

All logs are formatted with timestamp, module, level, and message.

### Decision Auditing

Access decision history via REST API:

```bash
# Get last 100 decisions
curl http://localhost:5000/decisions?limit=100

# Get statistics
curl http://localhost:5000/stats
```

### Integration with ELK Stack

```bash
# Forward logs to Elasticsearch (requires additional configuration)
# Logs can be parsed and indexed for analysis
```

## Production Checklist

- [ ] Tests passing (13/13)
- [ ] Code coverage >60%
- [ ] Docker image builds successfully
- [ ] Container health checks passing
- [ ] Kubernetes manifests valid
- [ ] Resource limits configured
- [ ] RBAC policies defined
- [ ] Logging configured
- [ ] Monitoring configured
- [ ] E2 simulator integration tested
- [ ] Load testing completed
- [ ] Documentation up-to-date

## References

- [README_HTTP_API.md](README_HTTP_API.md) - API documentation
- [../CLAUDE.md](../CLAUDE.md) - Project overview
- [../../docs/algorithms.md](../../docs/algorithms.md) - Algorithm details
