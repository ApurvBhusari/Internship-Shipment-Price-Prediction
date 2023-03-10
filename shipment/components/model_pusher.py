from shipment.predictor import ModelResolver
from shipment.entity.config_entity import ModelPusherConfig
from shipment.exception import InsuranceException
import os,sys
from shipment.utils import load_object,save_object
from shipment.logger import logging
from shipment.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact
from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_transformation import DataTransformation
from shipment.components.data_validation import DataValidation
from shipment.components.model_evaluation import ModelEvaluation
from shipment.components.model_trainer import ModelTrainer
from shipment.entity import config_entity

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
    data_transformation_artifact:DataTransformationArtifact,
    model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            #load object
            logging.info(f"Loading transformer model and target encoder")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            #model pusher dir
            logging.info(f"Saving model into model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)


            #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()
            target_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
             saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise InsuranceException(e, sys)


training_pipeline_config = config_entity.TrainingPipelineConfig()
#data ingestion
data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
print(data_ingestion_config.to_dict())
data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
#data validation
data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
data_validation = DataValidation(data_validation_config=data_validation_config,
                         data_ingestion_artifact=data_ingestion_artifact)
        
data_validation_artifact = data_validation.initiate_data_validation()

# data transformation
data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
          data_ingestion_artifact=data_ingestion_artifact)
data_transformation_artifact = data_transformation.initiate_data_transformation()
          
#model trainer
model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
model_trainer_artifact = model_trainer.initiate_model_trainer()

#model pusher
model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
model_pusher = ModelPusher(model_pusher_config=model_pusher_config, 
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact)
model_pusher_artifact = model_pusher.initiate_model_pusher()