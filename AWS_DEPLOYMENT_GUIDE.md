# AWS Deployment Guide

## 🚀 Complete AWS Deployment for Immigration Journey Analyzer

This guide provides step-by-step instructions for deploying the Immigration Journey Analyzer on AWS using SageMaker, Lambda, and API Gateway.

## 📋 Prerequisites

### Required Tools
- AWS CLI v2.0+
- Docker
- Python 3.9+
- Git

### AWS Account Setup
1. **AWS Account**: Active AWS account with appropriate permissions
2. **IAM Permissions**: SageMaker, Lambda, API Gateway, S3, ECR access
3. **Region**: Choose your preferred AWS region (default: us-east-1)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│  Lambda Function │───▶│  SageMaker      │
│   (REST API)    │    │  (Classification)│    │  Endpoint      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Bucket     │◀───│  ECR Repository │◀───│  Docker Image   │
│   (Model Artifacts)│    │  (Container)     │    │  (Immigration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Deployment

### Step 1: Configure AWS CLI
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
```

### Step 2: Set Environment Variables
```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"
export S3_BUCKET="your-immigration-classifier-bucket"
```

### Step 3: Run Deployment Script
```bash
# Make script executable
chmod +x aws-scripts/deploy.sh

# Run deployment
./aws-scripts/deploy.sh
```

## 🔧 Manual Deployment Steps

### Step 1: Create S3 Bucket
```bash
aws s3 mb s3://your-immigration-classifier-bucket --region us-east-1
```

### Step 2: Create ECR Repository
```bash
aws ecr create-repository --repository-name immigration-classifier --region us-east-1
```

### Step 3: Build and Push Docker Image
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t immigration-classifier .

# Tag and push
docker tag immigration-classifier:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/immigration-classifier:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/immigration-classifier:latest
```

### Step 4: Deploy CloudFormation Stack
```bash
aws cloudformation deploy \
    --template-file aws-deployment.yaml \
    --stack-name immigration-journey-analyzer-stack \
    --parameter-overrides \
        S3BucketName=your-immigration-classifier-bucket \
        Environment=production \
        InstanceType=ml.m5.xlarge \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

## 📊 Testing the Deployment

### Test API Endpoint
```bash
# Get API Gateway URL
API_URL=$(aws cloudformation describe-stacks \
    --stack-name immigration-journey-analyzer-stack \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
    --output text)

# Test classification
curl -X POST $API_URL/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "I need help with my H1B visa application"}'
```

### Expected Response
```json
{
    "text": "I need help with my H1B visa application",
    "category": "work_visa",
    "confidence": 0.95,
    "timestamp": "12345678-1234-1234-1234-123456789012"
}
```

## 🔍 Monitoring and Logs

### CloudWatch Logs
```bash
# View Lambda logs
aws logs tail /aws/lambda/immigration-journey-analyzer-lambda-production --follow

# View SageMaker logs
aws logs tail /aws/sagemaker/Endpoints/immigration-journey-analyzer-endpoint-production --follow
```

### SageMaker Endpoint Status
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name immigration-journey-analyzer-endpoint-production
```

## 💰 Cost Optimization

### Instance Types
- **Development**: ml.t3.medium ($0.05/hour)
- **Staging**: ml.m5.large ($0.12/hour)
- **Production**: ml.m5.xlarge ($0.24/hour)

### Auto-scaling
```bash
# Configure auto-scaling
aws sagemaker put-scaling-policy \
    --endpoint-name immigration-journey-analyzer-endpoint-production \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 300
    }'
```

## 🔒 Security Best Practices

### IAM Roles
- **Least Privilege**: Minimal required permissions
- **Role Separation**: Different roles for different services
- **Regular Audits**: Review permissions quarterly

### Network Security
- **VPC**: Deploy in private subnets
- **Security Groups**: Restrict access to necessary ports
- **WAF**: Web Application Firewall for API Gateway

### Data Protection
- **Encryption**: S3 server-side encryption
- **Secrets**: Use AWS Secrets Manager
- **Monitoring**: CloudTrail for audit logs

## 🚨 Troubleshooting

### Common Issues

#### 1. SageMaker Endpoint Creation Fails
```bash
# Check IAM role permissions
aws iam get-role --role-name ImmigrationClassifierSageMakerRole

# Verify ECR image exists
aws ecr describe-images --repository-name immigration-classifier
```

#### 2. Lambda Function Errors
```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/immigration

# Test Lambda function
aws lambda invoke --function-name immigration-journey-analyzer-lambda-production response.json
```

#### 3. API Gateway Issues
```bash
# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix /aws/apigateway

# Test API Gateway
curl -X POST $API_URL/classify -H "Content-Type: application/json" -d '{"text": "test"}'
```

## 📈 Performance Optimization

### SageMaker Optimization
- **Instance Types**: Choose based on workload
- **Auto-scaling**: Configure based on traffic patterns
- **Model Caching**: Enable model caching for faster inference

### Lambda Optimization
- **Memory**: Adjust based on model size
- **Timeout**: Set appropriate timeout values
- **Concurrency**: Configure reserved concurrency

### API Gateway Optimization
- **Caching**: Enable response caching
- **Throttling**: Configure rate limiting
- **Compression**: Enable compression

## 🔄 Updates and Maintenance

### Model Updates
```bash
# Update model in SageMaker
aws sagemaker update-endpoint \
    --endpoint-name immigration-journey-analyzer-endpoint-production \
    --endpoint-config-name immigration-journey-analyzer-config-production
```

### Code Updates
```bash
# Update Lambda function
aws lambda update-function-code \
    --function-name immigration-journey-analyzer-lambda-production \
    --zip-file fileb://lambda-deployment-package.zip
```

## 📞 Support

For issues and questions:
- **AWS Support**: Use AWS Support Center
- **Documentation**: Check AWS SageMaker documentation
- **Community**: AWS re:Post community forums

---

*Deployment guide for Immigration Journey Analyzer on AWS*
*Last updated: 2025-09-23*
