python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
python models/train_classifier.py data/DisasterResponse.db models/model_rfc_gridsearchcv.pkl
