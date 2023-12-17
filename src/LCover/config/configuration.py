
import os
from LCover.constants import *
from LCover.utils.common import read_yaml, create_directories
from LCover.entity.config_entity import (DataIngestionConfig,
                                        PrepareBaseModelConfig,
                                        TrainingConfig,
                                        EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            unzip_dir=Path(config.unzip_dir),
            image_size=int(self.params.IMAGE_SIZE)
        )

        return data_ingestion_config
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        root_data_dir = self.config.data_ingestion.root_dir
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            root_data_dir=Path(root_data_dir),
            params_ratio=params.RATIO,
            params_classes=params.CLASSES,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_learning_rate=params.LEARNING_RATE,
            parmas_early_stopping=params.EARLY_STOPPING
        )

        return training_config
    


    def get_evaluation_config(self) -> EvaluationConfig:

        training = self.config.training
        params = self.params
        root_data_dir = self.config.data_ingestion.root_dir

        eval_config = EvaluationConfig(
            root_data_dir=Path(root_data_dir),
            path_of_model=Path(training.trained_model_path),
            stage_model=Path(training.stage_dir),
            model_name=params.MODEL_NAME,
            model_version=params.VERSION,
            params_batch_size=params.BATCH_SIZE,
            params_classes=params.CLASSES,
            all_params=self.params
        )
        return eval_config

