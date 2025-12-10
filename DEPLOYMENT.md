# CQE Unified Runtime - Deployment Guide

This guide covers deploying the CQE Unified Runtime across various environments and platforms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Python Package](#python-package)
3. [Docker](#docker)
4. [Kubernetes](#kubernetes)
5. [Cloud Platforms](#cloud-platforms)
6. [Development](#development)
7. [Production](#production)

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- (Optional) Docker
- (Optional) Kubernetes cluster
- (Optional) Cloud account (AWS/GCP/Azure)

### Install from Source

```bash
# Clone repository
git clone https://github.com/cqe-research/cqe-unified-runtime.git
cd cqe-unified-runtime

# Install dependencies
pip install -r requirements.txt

# Test installation
python3 cqe_cli.py info
```

---

## Python Package

### Install as Package

```bash
# From source
pip install -e .

# With all extras
pip install -e ".[all]"

# With specific extras
pip install -e ".[dev]"      # Development tools
pip install -e ".[viz]"      # Visualization
pip install -e ".[ml]"       # Machine learning
```

### Use as Library

```python
# Import CQE components
from layer2_geometric.e8.lattice import E8Lattice
from layer4_governance.gravitational import GravitationalLayer
from layer3_operational.morsr import MORSRExplorer

# Use E8 lattice
e8 = E8Lattice()
vector = [1, 2, 3, 4, 5, 6, 7, 8]
projected = e8.project(vector)

# Calculate digital root
grav = GravitationalLayer()
dr = grav.calculate_digital_root(432)  # → 9

# Run MORSR optimization
morsr = MORSRExplorer()
result = morsr.explore(initial_state, max_iterations=100)
```

### Command Line Interface

```bash
# Show system info
cqe info

# E8 operations
cqe e8 project 1,2,3,4,5,6,7,8
cqe e8 roots --count 20

# Leech operations
cqe leech project 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24

# Digital root
cqe dr 432

# MORSR optimization
cqe morsr optimize --iterations 1000

# Aletheia analysis
cqe aletheia analyze "text to analyze"
```

---

## Docker

### Build Image

```bash
# Build Docker image
docker build -t cqe-unified-runtime:4.0.0-beta .

# Build with custom tag
docker build -t myregistry/cqe-runtime:latest .
```

### Run Container

```bash
# Run CLI
docker run --rm cqe-unified-runtime:4.0.0-beta python3 cqe_cli.py info

# Run API server
docker run -p 8000:8000 cqe-unified-runtime:4.0.0-beta python3 cqe_server.py

# Run with volume mounts
docker run -v $(pwd)/data:/app/data -p 8000:8000 cqe-unified-runtime:4.0.0-beta
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

**Services:**
- `cqe-api`: REST API server (port 8000)
- `cqe-explorer`: Interactive explorer (port 8080)
- `cqe-worker`: Background worker

---

## Kubernetes

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=cqe-runtime
kubectl get svc cqe-runtime-service

# View logs
kubectl logs -l app=cqe-runtime -f

# Scale deployment
kubectl scale deployment cqe-runtime --replicas=5
```

### Access Service

```bash
# Get service URL
kubectl get svc cqe-runtime-service

# Port forward for local access
kubectl port-forward svc/cqe-runtime-service 8000:80

# Access API
curl http://localhost:8000/health
```

### Helm Chart (Coming Soon)

```bash
# Install with Helm
helm install cqe-runtime ./helm/cqe-runtime

# Upgrade
helm upgrade cqe-runtime ./helm/cqe-runtime

# Uninstall
helm uninstall cqe-runtime
```

---

## Cloud Platforms

### AWS

#### EC2 Deployment

```bash
# Deploy with CloudFormation
aws cloudformation create-stack \
  --stack-name cqe-runtime \
  --template-body file://deployment/aws/cloudformation.yaml \
  --parameters \
    ParameterKey=KeyName,ParameterValue=your-key-pair \
    ParameterKey=InstanceType,ParameterValue=t3.medium

# Get outputs
aws cloudformation describe-stacks --stack-name cqe-runtime
```

#### ECS Deployment

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name cqe-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://deployment/aws/task-definition.json

# Create service
aws ecs create-service \
  --cluster cqe-cluster \
  --service-name cqe-service \
  --task-definition cqe-runtime \
  --desired-count 3
```

#### Lambda Deployment

```bash
# Package for Lambda
zip -r cqe-lambda.zip . -x "*.git*" -x "*__pycache__*"

# Deploy to Lambda
aws lambda create-function \
  --function-name cqe-runtime \
  --runtime python3.11 \
  --handler lambda_handler.handler \
  --zip-file fileb://cqe-lambda.zip \
  --role arn:aws:iam::ACCOUNT:role/lambda-role
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/cqe-runtime

# Deploy to Cloud Run
gcloud run deploy cqe-runtime \
  --image gcr.io/PROJECT_ID/cqe-runtime \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create cqe-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2

# Deploy to GKE
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### Microsoft Azure

#### Azure Container Instances

```bash
# Create resource group
az group create --name cqe-rg --location eastus

# Deploy container
az container create \
  --resource-group cqe-rg \
  --name cqe-runtime \
  --image cqe-unified-runtime:4.0.0-beta \
  --dns-name-label cqe-runtime \
  --ports 8000
```

#### Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group cqe-rg \
  --name cqe-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group cqe-rg --name cqe-cluster

# Deploy
kubectl apply -f deployment/kubernetes/deployment.yaml
```

---

## Development

### Local Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Development Server

```bash
# Run API server with auto-reload
python3 cqe_server.py --reload

# Run on custom port
python3 cqe_server.py --port 8080

# Run with multiple workers
python3 cqe_server.py --workers 4
```

### Environment Variables

```bash
# Set environment
export CQE_ENV=development
export CQE_LOG_LEVEL=debug

# Use .env file
cat > .env << EOF
CQE_ENV=development
CQE_LOG_LEVEL=debug
CQE_DATA_DIR=/app/data
CQE_LOG_DIR=/app/logs
EOF
```

---

## Production

### Production Checklist

- [ ] Use production-grade WSGI server (uvicorn with workers)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure auto-scaling
- [ ] Set up backups
- [ ] Enable health checks
- [ ] Use secrets management
- [ ] Configure rate limiting
- [ ] Set up CI/CD pipeline
- [ ] Enable security scanning

### Production Configuration

```bash
# Run with production settings
export CQE_ENV=production
export CQE_LOG_LEVEL=info

# Run with multiple workers
python3 cqe_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Metrics endpoint (if enabled)
curl http://localhost:8000/metrics

# Logs
tail -f logs/cqe-runtime.log
```

### Security

```bash
# Use secrets for sensitive data
export CQE_SECRET_KEY=your-secret-key

# Enable HTTPS
python3 cqe_server.py \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

---

## Performance Tuning

### Optimization Tips

1. **Caching**: Enable caching for frequently accessed data
2. **Workers**: Use multiple workers for CPU-bound tasks
3. **Async**: Use async operations where possible
4. **Batch Processing**: Process multiple requests in batches
5. **GPU**: Use GPU for Scene8 video generation

### Resource Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 2 GB
- Disk: 5 GB

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 20+ GB
- GPU: Optional (for Scene8)

**Production:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 50+ GB
- GPU: Recommended (for Scene8)

---

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH=/path/to/cqe_unified_runtime:$PYTHONPATH
```

**Port Already in Use:**
```bash
# Use different port
python3 cqe_server.py --port 8080

# Kill existing process
lsof -ti:8000 | xargs kill -9
```

**Docker Issues:**
```bash
# Rebuild image
docker-compose build --no-cache

# Clean up
docker system prune -a
```

---

## Support

- **Documentation**: https://cqe-unified-runtime.readthedocs.io/
- **Issues**: https://github.com/cqe-research/cqe-unified-runtime/issues
- **Discussions**: https://github.com/cqe-research/cqe-unified-runtime/discussions

---

## License

MIT License - See LICENSE file for details

---

**CQE Unified Runtime v4.0.0-beta**  
**90% Complete | 297 Files | 133,517 Lines**  
**Production Ready | Universally Deployable**
