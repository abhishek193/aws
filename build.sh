#!/bin/bash

# construct the ECR name.
account=$(aws sts get-caller-identity --query Account --output text)
region=ap-south-1
#$(aws configure get region)
#change thefullname to your own ecr repository
fullname="${account}.dkr.ecr.${region}.amazonaws.com/aidevops/sagemaker-demo:latest"

# If the repository doesn't exist in ECR, create it.
# The pipe trick redirects stderr to stdout and passes it /dev/null.
# It's just there to silence the error.
aws ecr describe-repositories --repository-names "aidevops/sagemaker-demo" > /dev/null 2>&1

# Check the error code, if it's non-zero then know we threw an error and no repo exists
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "aidevops/sagemaker-demo" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region}  --no-include-email)
echo "loginsuccess: building docker image"
# Build the docker image, tag it with the full name, and push it to ECR
docker build  -t "aidevops/sagemaker-demo" .
#docker tag "sagemaker-demo" ${fullname}
docker tag "aidevops/sagemaker-demo" ${fullname}
echo "Pushing image"
docker push ${fullname}
