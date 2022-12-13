from flask import Flask
from flask import jsonify, request
import logging
from flask import request
from services.model import ModelService


logging.basicConfig(filename='/results/output.log', level=logging.DEBUG)
app = Flask(__name__)
model_service = ModelService()


@app.route('/healthcheck')    
def healthcheck():
    return "ok"


@app.route('/predict', methods=['GET'])    
def predict():
    part_id = request.args.get('part_id')
    model_id = request.args.get('model_id')
    use_historical = bool(request.args.get('use_historical'))
    data_df = model_service.load_data_to_predict(part_id, use_historical)
    result = model_service.predict(model_id, data_df)
    return  jsonify(result)   


@app.route('/last_processed_part')
def last_processed_part():  
    result = model_service.get_last_processed_part()
    return jsonify(result)



@app.route('/rate_champion')
def rate_champion():  
    champion_id = model_service.get_champion_model_id()
    result = model_service.rate_model(champion_id)
    return jsonify(result)



@app.route('/rate_model')
def rate_model():  
    model_id = request.args.get('model_id')
    result = model_service.rate_model(model_id)
    return jsonify(result)


    
@app.route('/evaluate_models')
def evaluate_models():  
    result = model_service.evaluate_models()
    return jsonify(result)


@app.route('/initial_train')
def initial_train():  
    result = model_service.initial_train()
    return jsonify(result)


@app.route('/train')
def train():
    part_id = request.args.get('part_id')
    if part_id is None:
        part_id = model_service.get_last_processed_part()["last_part"]

    result = model_service.train(part_id)
    return jsonify(result)



@app.route('/valid_data')    
def valid_data():
    #Not mandatory
    part_id = request.args.get('part_id')
    if part_id is None:
        part_id = model_service.get_last_processed_part()["last_part"]

    result = model_service.valid_data(part_id)
    return jsonify(result)
     


if __name__ == "__main__":
    app.run(debug=True)
