#!/bin/bash
# AWS Deployment Script for Immigration Journey Analyzer

set -e

# Configuration
PROJECT_NAME="immigration-journey-analyzer"
AWS_REGION="us-east-1"
S3_BUCKET="your-immigration-classifier-bucket"
ECR_REPOSITORY="immigration-classifier"
SAGEMAKER_ROLE="ImmigrationClassifierSageMakerRole"

echo "🚀 Starting AWS deployment for Immigration Journey Analyzer..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI first."
    exit 1
fi

# Check if logged in to AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ Not logged in to AWS. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI configured"

# Create S3 bucket if it doesn't exist
echo "📦 Creating S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION || echo "Bucket already exists"

# Create ECR repository
echo "🐳 Creating ECR repository..."
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION || echo "Repository already exists"

# Get ECR login token
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t $ECR_REPOSITORY .

# Tag image for ECR
docker tag $ECR_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Push image to ECR
echo "📤 Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Create SageMaker role if it doesn't exist
echo "👤 Creating SageMaker IAM role..."
aws iam create-role \
    --role-name $SAGEMAKER_ROLE \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || echo "Role already exists"

# Attach policies to role
aws iam attach-role-policy \
    --role-name $SAGEMAKER_ROLE \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name $SAGEMAKER_ROLE \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Deploy CloudFormation stack
echo "☁️ Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file aws-deployment.yaml \
    --stack-name $PROJECT_NAME-stack \
    --parameter-overrides \
        S3BucketName=$S3_BUCKET \
        Environment=production \
        InstanceType=ml.m5.xlarge \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

echo "✅ Deployment complete!"
echo "📊 SageMaker Endpoint: $(aws cloudformation describe-stacks --stack-name $PROJECT_NAME-stack --query 'Stacks[0].Outputs[?OutputKey==`SageMakerEndpoint`].OutputValue' --output text --region $AWS_REGION)"
echo "🌐 API Gateway URL: $(aws cloudformation describe-stacks --stack-name $PROJECT_NAME-stack --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' --output text --region $AWS_REGION)"
