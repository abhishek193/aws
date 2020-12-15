from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker import get_execution_role
import sagemaker as sage
import boto3

# this line of code require additional iam:GetRole permissions.
role = get_execution_role()

sess = sage.Session()

client = boto3.client('sagemaker')
name = client.list_training_jobs(SortBy='CreationTime')['TrainingJobSummaries'][0]['TrainingJobName']

s3_model_path = 's3://sagemaker-aidevops/model/' + name + '/output/model.tar.gz'
print('S3 path for model', s3_model_path)

sagemaker_model = TensorFlowModel(model_data = s3_model_path,
                                  role = role,
                                  framework_version = '1.12.0',
                                  py_version='py2',
                                  entry_point = 'train1.py')

print('Starting deployment...')
predictor = sagemaker_model.deploy(initial_instance_count=1,
                                   instance_type='ml.m4.large')

endpoint_name = predictor.endpoint
print('Finished deployment of Endpoint: ', endpoint_name)

endpoint_file = open("endpoint_name.txt","w")
endpoint_file.write(endpoint_name)
endpoint_file.close()

print('Endpoint name written to file: endpoint_name.txt')