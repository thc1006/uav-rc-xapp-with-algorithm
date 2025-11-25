# Quick Start Guide - UAV Policy xApp HTTP API

## 30-Second Overview

The UAV Policy xApp is an HTTP server that:
1. Receives E2 indications (POST to `/e2/indication`)
2. Runs policy algorithms
3. Returns resource allocation decisions

## Installation and Run

### Option 1: Local (Python)

```bash
cd /home/thc1006/dev/uav-rc-xapp-with-algorithms/xapps/uav-policy

# Install dependencies
pip install flask werkzeug

# Run server
export PYTHONPATH="src"
python3 -m uav_policy.main

# Server runs on http://localhost:5000
```

### Option 2: Docker

```bash
cd /home/thc1006/dev/uav-rc-xapp-with-algorithms/xapps/uav-policy

# Build image
docker build -t uav-policy-xapp:latest .

# Run container
docker run -d -p 5000:5000 --name uav-policy uav-policy-xapp:latest

# View logs
docker logs -f uav-policy
```

### Option 3: Kubernetes

```bash
cd /home/thc1006/dev/uav-rc-xapp-with-algorithms/xapps/uav-policy

# Create namespace
kubectl create namespace oran-ric

# Deploy
kubectl apply -k k8s/

# Check status
kubectl get pods -n oran-ric -l app=uav-policy-xapp

# Port forward for testing
kubectl port-forward -n oran-ric svc/uav-policy-xapp 5000:5000
```

## Test the Server

### Health Check
```bash
curl http://localhost:5000/health
```

### Submit E2 Indication
```bash
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
```

### Get Decision History
```bash
curl http://localhost:5000/decisions?limit=10
```

### Get Statistics
```bash
curl http://localhost:5000/stats
```

## Run Tests

```bash
export PYTHONPATH="src"
python3 -m pytest tests/ -v --cov=src/uav_policy
```

Expected: 13 tests pass, 68% coverage

## Project Structure

```
src/uav_policy/
├── policy_engine.py    - Core algorithms (unchanged)
├── server.py           - HTTP API server (NEW)
└── main.py             - Entry point (UPDATED)

tests/
├── test_http_server.py  - API tests (NEW)
└── test_policy_engine.py - Algorithm tests

k8s/
├── deployment.yaml
├── service.yaml
├── configmap.yaml
├── serviceaccount.yaml
└── kustomization.yaml
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/uav_policy/server.py` | HTTP API | NEW |
| `src/uav_policy/main.py` | Entry point | UPDATED |
| `tests/test_http_server.py` | API tests | NEW |
| `Dockerfile` | Container image | NEW |
| `k8s/*.yaml` | Kubernetes manifests | NEW |
| `README_HTTP_API.md` | API docs | NEW |
| `DEPLOYMENT_GUIDE.md` | Deployment help | NEW |

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/e2/indication` | Submit E2 indication |
| GET | `/decisions` | Get decision history |
| GET | `/stats` | Get statistics |

## Environment Variables

- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc) - default: INFO
- `SERVER_HOST` - Bind address - default: 0.0.0.0
- `SERVER_PORT` - Server port - default: 5000
- `DEBUG` - Flask debug mode - default: false

## Troubleshooting

### "Module not found" error
```bash
export PYTHONPATH="$(pwd)/src"
```

### Port 5000 already in use
```bash
# Change port
export SERVER_PORT=5001

# Or kill process on 5000
lsof -i :5000  # Find PID
kill -9 <PID>
```

### Docker build fails
```bash
# Clean up and retry
docker system prune -a
docker build -t uav-policy-xapp:latest .
```

### Tests fail
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
python3 -m pytest tests/ -vv -s
```

## Next Steps

1. Read `README_HTTP_API.md` for full API documentation
2. See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions
3. Check `IMPLEMENTATION_SUMMARY.md` for technical details
4. Review `tests/test_http_server.py` for usage examples

## Common Tasks

### Check if server is running
```bash
curl -s http://localhost:5000/health | grep healthy
# Output: "healthy" in response means server is running
```

### View server logs
```bash
# Local
tail -f /tmp/uav_policy.log  # If redirected

# Docker
docker logs -f uav-policy

# Kubernetes
kubectl logs -f -n oran-ric -l app=uav-policy-xapp
```

### Scale to multiple replicas (Kubernetes)
```bash
kubectl scale deployment -n oran-ric uav-policy-xapp --replicas=3
```

### Update configuration (Kubernetes)
```bash
kubectl patch configmap -n oran-ric uav-policy-config \
  --type merge -p '{"data":{"log_level":"DEBUG"}}'
```

### Access from outside cluster
```bash
# Port forward
kubectl port-forward -n oran-ric svc/uav-policy-xapp 5000:5000

# Or use LoadBalancer
kubectl patch svc -n oran-ric uav-policy-xapp \
  -p '{"spec":{"type":"LoadBalancer"}}'
```

## Support

For detailed information:
- API usage: See `README_HTTP_API.md`
- Deployment: See `DEPLOYMENT_GUIDE.md`
- Implementation: See `IMPLEMENTATION_SUMMARY.md`
- Project: See `CLAUDE.md` and `spec/uav_oran_usecase.md`
