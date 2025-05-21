import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

role_arn = "arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-xxxxxxx"
bucket = "tu-bucket-modelos"
model_artifact_path = f"s3://{bucket}/model.tar.gz"
image_uri = sagemaker.image_uris.retrieve("xgboost", region="us-east-1", version="1.5-1")

# Inicializa sesión de SageMaker
sagemaker_session = sagemaker.Session()
model = Model(
    image_uri=image_uri,
    model_data=model_artifact_path,
    role=role_arn,
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="clothes-price-endpoint"
)

predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

print("[INFO] Modelo desplegado exitosamente")
