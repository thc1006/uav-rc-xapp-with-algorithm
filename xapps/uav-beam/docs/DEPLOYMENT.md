# UAV Beam Tracking xApp - Deployment Guide

This document provides comprehensive instructions for deploying the UAV Beam Tracking xApp in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Environment Variables](#environment-variables)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| Memory | 512 MB | 2 GB |
| Disk | 100 MB | 500 MB |
| Python | 3.8+ | 3.10+ |

### Software Dependencies

```bash
# Core dependencies
Python >= 3.8
Flask >= 2.0.0
NumPy >= 1.20.0
SciPy >= 1.7.0
Requests >= 2.25.0

# Development dependencies (optional)
pytest >= 6.0.0
pytest-cov >= 2.0.0
black >= 21.0
flake8 >= 3.9.0

# ML dependencies (optional)
torch >= 1.9.0
scikit-learn >= 0.24.0
```

### Network Requirements

| Port | Protocol | Purpose |
|------|----------|---------|
| 5001 | TCP/HTTP | REST API |
| 4560 | TCP | RMR (if using O-RAN RIC) |

---

## Local Development

### Installation

```bash
# Navigate to xApp directory
cd xapps/uav-beam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install package in development mode
pip install -e ".[dev]"

# Verify installation
uav-beam-xapp --help
```

### Running Locally

```bash
# Default configuration
uav-beam-xapp

# Custom configuration
uav-beam-xapp \
    --host 0.0.0.0 \
    --port 5001 \
    --num-beams-h 16 \
    --num-beams-v 8 \
    --debug

# Using Python module directly
python -m uav_beam.main --port 5001
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=uav_beam --cov-report=html

# Run specific test
pytest tests/test_beam_tracker.py -v

# Run with verbose output
pytest tests/ -v -s
```

### Development Server

For development with auto-reload:

```python
# dev_server.py
from uav_beam.server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
        use_reloader=True
    )
```

```bash
python dev_server.py
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile` in the xApp root directory:

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY setup.py .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -s /bin/bash xapp
USER xapp

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
ENTRYPOINT ["uav-beam-xapp"]
CMD ["--host", "0.0.0.0", "--port", "5001"]
```

### Building Docker Image

```bash
# Build image
docker build -t uav-beam-xapp:latest .

# Build with specific tag
docker build -t uav-beam-xapp:0.1.0 .

# Build with build args
docker build \
    --build-arg PYTHON_VERSION=3.10 \
    -t uav-beam-xapp:latest .
```

### Running Docker Container

```bash
# Basic run
docker run -d \
    --name uav-beam-xapp \
    -p 5001:5001 \
    uav-beam-xapp:latest

# With custom configuration
docker run -d \
    --name uav-beam-xapp \
    -p 5001:5001 \
    -e UAV_BEAM_NUM_BEAMS_H=32 \
    -e UAV_BEAM_NUM_BEAMS_V=16 \
    uav-beam-xapp:latest

# With volume for logs
docker run -d \
    --name uav-beam-xapp \
    -p 5001:5001 \
    -v /var/log/uav-beam:/app/logs \
    uav-beam-xapp:latest

# Interactive mode for debugging
docker run -it --rm \
    -p 5001:5001 \
    uav-beam-xapp:latest \
    --debug
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  uav-beam-xapp:
    build:
      context: .
      dockerfile: Dockerfile
    image: uav-beam-xapp:latest
    container_name: uav-beam-xapp
    ports:
      - "5001:5001"
    environment:
      - UAV_BEAM_HOST=0.0.0.0
      - UAV_BEAM_PORT=5001
      - UAV_BEAM_NUM_BEAMS_H=16
      - UAV_BEAM_NUM_BEAMS_V=8
      - UAV_BEAM_DEBUG=false
      - LOG_LEVEL=INFO
    volumes:
      - xapp-logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M

  # Optional: Prometheus metrics exporter
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - uav-beam-xapp

  # Optional: Grafana dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  xapp-logs:
  grafana-data:
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f uav-beam-xapp

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## Kubernetes Deployment

### Kubernetes Manifests

#### Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: uav-beam-xapp
  labels:
    app: uav-beam-xapp
```

#### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: uav-beam-config
  namespace: uav-beam-xapp
data:
  # Beam configuration
  NUM_BEAMS_H: "16"
  NUM_BEAMS_V: "8"
  BEAM_FAILURE_THRESHOLD_DB: "-10.0"
  PREDICTION_HORIZON_MS: "20.0"

  # Predictor configuration
  MAX_PREDICTION_HORIZON_MS: "500.0"
  MAX_VELOCITY: "30.0"

  # Server configuration
  HOST: "0.0.0.0"
  PORT: "5001"
  LOG_LEVEL: "INFO"
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uav-beam-xapp
  namespace: uav-beam-xapp
  labels:
    app: uav-beam-xapp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: uav-beam-xapp
  template:
    metadata:
      labels:
        app: uav-beam-xapp
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "5001"
        prometheus.io/path: "/statistics"
    spec:
      containers:
        - name: uav-beam-xapp
          image: uav-beam-xapp:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 5001
              name: http
              protocol: TCP
          envFrom:
            - configMapRef:
                name: uav-beam-config
          resources:
            requests:
              cpu: "500m"
              memory: "256Mi"
            limits:
              cpu: "2000m"
              memory: "1Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 5001
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 5001
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - uav-beam-xapp
                topologyKey: kubernetes.io/hostname
```

#### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: uav-beam-xapp
  namespace: uav-beam-xapp
  labels:
    app: uav-beam-xapp
spec:
  type: ClusterIP
  ports:
    - port: 5001
      targetPort: 5001
      protocol: TCP
      name: http
  selector:
    app: uav-beam-xapp
```

#### Ingress (Optional)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: uav-beam-xapp
  namespace: uav-beam-xapp
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - host: uav-beam.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: uav-beam-xapp
                port:
                  number: 5001
```

#### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: uav-beam-xapp
  namespace: uav-beam-xapp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: uav-beam-xapp
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Deploying to Kubernetes

```bash
# Create namespace and apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Or apply all at once
kubectl apply -f k8s/

# Check deployment status
kubectl -n uav-beam-xapp get pods
kubectl -n uav-beam-xapp get services
kubectl -n uav-beam-xapp get hpa

# View logs
kubectl -n uav-beam-xapp logs -f deployment/uav-beam-xapp

# Port forward for local access
kubectl -n uav-beam-xapp port-forward svc/uav-beam-xapp 5001:5001

# Scale deployment
kubectl -n uav-beam-xapp scale deployment uav-beam-xapp --replicas=3
```

### Helm Chart (Optional)

Directory structure:

```
uav-beam-xapp-chart/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    configmap.yaml
    hpa.yaml
    ingress.yaml
    _helpers.tpl
```

`Chart.yaml`:

```yaml
apiVersion: v2
name: uav-beam-xapp
description: UAV Beam Tracking xApp for O-RAN Near-RT RIC
type: application
version: 0.1.0
appVersion: "0.1.0"
```

`values.yaml`:

```yaml
replicaCount: 2

image:
  repository: uav-beam-xapp
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 5001

ingress:
  enabled: false
  className: nginx
  hosts:
    - host: uav-beam.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 2000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

config:
  numBeamsH: 16
  numBeamsV: 8
  beamFailureThresholdDb: -10.0
  predictionHorizonMs: 20.0
  logLevel: INFO
```

```bash
# Install chart
helm install uav-beam-xapp ./uav-beam-xapp-chart -n uav-beam-xapp

# Upgrade
helm upgrade uav-beam-xapp ./uav-beam-xapp-chart -n uav-beam-xapp

# Uninstall
helm uninstall uav-beam-xapp -n uav-beam-xapp
```

---

## Environment Variables

### Complete Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `UAV_BEAM_HOST` | `0.0.0.0` | Server bind host |
| `UAV_BEAM_PORT` | `5001` | Server bind port |
| `UAV_BEAM_DEBUG` | `false` | Enable debug mode |
| `UAV_BEAM_NUM_BEAMS_H` | `16` | Horizontal beam count |
| `UAV_BEAM_NUM_BEAMS_V` | `8` | Vertical beam count |
| `UAV_BEAM_NUM_ANTENNA_H` | `8` | Horizontal antenna elements |
| `UAV_BEAM_NUM_ANTENNA_V` | `8` | Vertical antenna elements |
| `UAV_BEAM_FAILURE_THRESHOLD` | `-10.0` | Beam failure threshold (dB) |
| `UAV_BEAM_PREDICTION_HORIZON` | `20.0` | Prediction horizon (ms) |
| `UAV_BEAM_RECOVERY_TIMER` | `50.0` | Recovery timer (ms) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |

### Loading Environment Variables

Add to main.py for environment variable support:

```python
import os

def get_config_from_env():
    """Load configuration from environment variables"""
    return {
        "beam": {
            "num_beams_h": int(os.getenv("UAV_BEAM_NUM_BEAMS_H", 16)),
            "num_beams_v": int(os.getenv("UAV_BEAM_NUM_BEAMS_V", 8)),
            "num_antenna_elements_h": int(os.getenv("UAV_BEAM_NUM_ANTENNA_H", 8)),
            "num_antenna_elements_v": int(os.getenv("UAV_BEAM_NUM_ANTENNA_V", 8)),
            "beam_failure_threshold_db": float(os.getenv("UAV_BEAM_FAILURE_THRESHOLD", -10.0)),
            "prediction_horizon_ms": float(os.getenv("UAV_BEAM_PREDICTION_HORIZON", 20.0)),
        },
        "predictor": {
            "max_prediction_horizon_ms": float(os.getenv("UAV_BEAM_MAX_PREDICTION_HORIZON", 500.0)),
        }
    }
```

---

## Monitoring and Logging

### Logging Configuration

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)

def setup_logging(level="INFO", format_type="json"):
    """Configure logging"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler()

    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    logger.addHandler(handler)
```

### Prometheus Metrics

Add metrics endpoint:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
INDICATIONS_TOTAL = Counter(
    'uav_beam_indications_total',
    'Total E2 indications received',
    ['ue_id']
)

BEAM_SWITCHES_TOTAL = Counter(
    'uav_beam_switches_total',
    'Total beam switches',
    ['ue_id', 'action']
)

DECISION_LATENCY = Histogram(
    'uav_beam_decision_latency_seconds',
    'Beam decision latency',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

ACTIVE_UES = Gauge(
    'uav_beam_active_ues',
    'Number of active UEs being tracked'
)

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain'}
```

### Prometheus Configuration

`prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'uav-beam-xapp'
    static_configs:
      - targets: ['uav-beam-xapp:5001']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import this dashboard JSON for Grafana:

```json
{
  "title": "UAV Beam Tracking xApp",
  "panels": [
    {
      "title": "Indications per Second",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(uav_beam_indications_total[1m])",
          "legendFormat": "{{ue_id}}"
        }
      ]
    },
    {
      "title": "Beam Switches",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(uav_beam_switches_total)"
        }
      ]
    },
    {
      "title": "Decision Latency (p99)",
      "type": "gauge",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(uav_beam_decision_latency_seconds_bucket[5m]))"
        }
      ]
    },
    {
      "title": "Active UEs",
      "type": "stat",
      "targets": [
        {
          "expr": "uav_beam_active_ues"
        }
      ]
    }
  ]
}
```

### Log Aggregation

For centralized logging with ELK/EFK stack:

```yaml
# Filebeat sidecar configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*.log
      processors:
        - add_kubernetes_metadata:
            host: ${NODE_NAME}
            matchers:
            - logs_path:
                logs_path: "/var/log/containers/"

    output.elasticsearch:
      hosts: ['${ELASTICSEARCH_HOST:elasticsearch}:${ELASTICSEARCH_PORT:9200}']
      index: "uav-beam-xapp-%{+yyyy.MM.dd}"
```

---

## Troubleshooting

### Common Issues

#### xApp Not Starting

```bash
# Check logs
docker logs uav-beam-xapp
kubectl -n uav-beam-xapp logs deployment/uav-beam-xapp

# Common causes:
# 1. Port already in use
#    Solution: Change port or stop conflicting service

# 2. Missing dependencies
#    Solution: Rebuild image or reinstall packages

# 3. Permission issues
#    Solution: Check file permissions and user context
```

#### High Latency

```bash
# Check resource usage
docker stats uav-beam-xapp
kubectl -n uav-beam-xapp top pods

# Possible causes:
# 1. CPU throttling
#    Solution: Increase CPU limits

# 2. Memory pressure
#    Solution: Increase memory limits

# 3. Too many UEs
#    Solution: Scale horizontally
```

#### Beam Tracking Issues

```bash
# Check statistics
curl http://localhost:5001/statistics

# Check specific UE state
curl http://localhost:5001/ue/uav-001

# Common issues:
# 1. High beam failure rate
#    - Check beam_failure_threshold_db setting
#    - Verify signal quality from gNB

# 2. Low prediction accuracy
#    - Check prediction_horizon_ms setting
#    - Verify position/velocity data quality
```

### Health Check Commands

```bash
# Basic health
curl http://localhost:5001/health

# Detailed statistics
curl http://localhost:5001/statistics | jq

# List tracked UEs
curl http://localhost:5001/statistics | jq '.tracked_uavs'

# Check configuration
curl http://localhost:5001/config | jq
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Docker
docker run -e UAV_BEAM_DEBUG=true uav-beam-xapp:latest --debug

# Kubernetes
kubectl -n uav-beam-xapp set env deployment/uav-beam-xapp UAV_BEAM_DEBUG=true

# Local
uav-beam-xapp --debug
```

### Performance Tuning

```bash
# Increase worker processes (production)
gunicorn -w 4 -b 0.0.0.0:5001 "uav_beam.server:create_app()"

# Use async server
pip install gunicorn[gevent]
gunicorn -k gevent -w 4 -b 0.0.0.0:5001 "uav_beam.server:create_app()"
```

### Recovery Procedures

```bash
# Reset xApp state
curl -X POST http://localhost:5001/reset

# Restart deployment
kubectl -n uav-beam-xapp rollout restart deployment/uav-beam-xapp

# Force recreate pods
kubectl -n uav-beam-xapp delete pods -l app=uav-beam-xapp
```
