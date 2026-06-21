import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utills import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        '''
        this function is reponsible for data transformation
        '''
        try:
            numerical_columns=["writing score","reading score"]
            categorical_columns=["gender","race/ethnicity","parental level of education","lunch","test preparation course"]

            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                         ("scaler",StandardScaler())  
                       ]
            )

            cat_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy= "most_frequent")),
                       ("One_hot_encoder",OneHotEncoder()),
                       ("scaller",StandardScaler(with_mean=False))]
            )

            logging.info("categorical columns encoding is done")
            logging.info("numerical columns scalling is done")
            preprocessor=ColumnTransformer(
                [
                    ("Numerical_pipleine_pe_kamm_hoga",num_pipeline,numerical_columns),
                    ("Categorical_pipleine_pe_kamm_hoga",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_tansformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("we have train and test data")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math score"
            numerical_columns=["writing score","reading score"]
            input_train_data=train_df.drop(columns=[target_column_name],axis=1)
            target_train_data= train_df[target_column_name]
            input_test_data=test_df.drop(columns=[target_column_name],axis=1)
            target_test_data= test_df[target_column_name]
            logging.info("Applyting column  transformation on train and test data")
            input_train_data_after_preprocessing=preprocessing_obj.fit_transform(input_train_data) #it will return an array
            input_test_data_after_preprocessing=preprocessing_obj.transform(input_test_data)# it will return an array
            final_train_data=np.c_[input_train_data_after_preprocessing,np.array(target_train_data)]
            final_test_data=np.c_[input_test_data_after_preprocessing,np.array(target_test_data)]
            logging.info("yaha tak train and test data preprocess kar liya hai")
            logging.info("ab hum object ko save karenge")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                final_train_data,
                final_test_data,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

            


            
