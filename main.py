from fastapi import FastAPI
from rasa.core.agent import Agent
from rasa.model_training import train_nlu
from pydantic import BaseModel
from typing import  Dict, Optional
import os


app = FastAPI()
MODEL_SAVING_DIR = './Models'



# Request Schema for creating NLU model
class NLURequest(BaseModel):
    nluData: Dict
    modelName: str


class RasaNLUModel:
    _instance_ = {}   # Contains active RASA NLU models


    def init():
        """
            Loads default rasa model while server being initialised
        """
        defaultModelPath = MODEL_SAVING_DIR + '/default.tar.gz'
        RasaNLUModel._instance_['default'] = Agent.load(defaultModelPath)


    def loadModel(modelpath, modelName):
        """
            loads custom Rasa NLU models into server without stopping the active server
        """
        RasaNLUModel._instance_[modelName] = Agent.load(modelpath)



@app.post("/trainNLUModel")
async def train_nlu_model(data: NLURequest):


    # saving custom nlu data with clientId as for training we need file to be present
    file_name = f'{data.modelName}_nlu.yml'
    file = open(file_name, "w")
    file.write('version: "3.0" \nnlu:\n')

    # Saving training examples along with intent
    for intent in data.nluData:
        file.write("- intent: {intent_name}\n".format(intent_name=intent))
        file.write("  examples: |\n")
        intent_examples = data.nluData[intent]

        for example in intent_examples:
            file.write("    - {}\n".format(example))

    file.close()

    # training NLU model based on the data recieved from client
    # saving it in trained folder for models
    nlu_model = train_nlu('config.yml',file_name,MODEL_SAVING_DIR,fixed_model_name= data.modelName)

    # removing the created yml file as it is not required after training
    if os.path.exists(file_name):
        os.remove(file_name)


    RasaNLUModel.loadModel(MODEL_SAVING_DIR+'/'+data.modelName, modelName=data.modelName)


    return {"message":"Model Loaded into memory successfully"}




@app.get("/predictText")
async def read_item(modelName: str, query: str):

    agent_nlu = RasaNLUModel._instance_[modelName]
    message = await agent_nlu.parse_message(query)
    # print(message)

    return {"prediction_info": message}





class Server:
    @staticmethod
    def loadModels():
        RasaNLUModel.init()
        print("Model loading done...")

Server.loadModels()
