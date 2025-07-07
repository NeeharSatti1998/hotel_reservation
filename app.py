import joblib
from config.paths_config import MODEL_OUTPUT_PATH,CONFIG_PATH
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from utils.common_functions import read_yaml

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)
config = read_yaml(CONFIG_PATH)
expected_col = config["data_processing"] ["matching_columns"]

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':

        form_data={

        'no_of_adults' : int(request.form['no_of_adults']),
        'no_of_children' : int(request.form['no_of_children']),
        'no_of_weekend_nights'  : int(request.form['no_of_weekend_nights']),
        'no_of_week_nights' : int(request.form['no_of_week_nights']),
        'room_type_reserved' : request.form['room_type_reserved'],
        'lead_time' : int(request.form['lead_time']),
        'arrival_year' : int(request.form['arrival_year']),
        'arrival_month' : int(request.form['arrival_month']),
        'arrival_date' : int(request.form['arrival_date']),
        'avg_price_per_room' : float(request.form['avg_price_per_room']),
        'no_of_special_requests' : int(request.form['no_of_special_requests']),
        'type_of_meal_plan' : request.form['type_of_meal_plan'],
        'required_car_parking_space' : str(request.form['required_car_parking_space']),
        'market_segment_type' : request.form['market_segment_type'],
        'repeated_guest' : str(request.form['repeated_guest']),
        'no_of_previous_cancellations' : int(request.form['no_of_previous_cancellations']),
        'no_of_previous_bookings_not_canceled' : int(request.form['no_of_previous_bookings_not_canceled'])

        }


        df = pd.DataFrame([form_data])

        df['required_car_parking_space'] = df['required_car_parking_space'].astype('object')
        df['repeated_guest'] = df['repeated_guest'].astype('object')
        df['arrival_year'] = df['arrival_year'].astype('object')
        df['arrival_month'] = df['arrival_month'].astype('object')


        arrivate_date_cat = []
        for i in df['arrival_date']:
            if i <=12:
                arrivate_date_cat.append('start_month')
            elif i>12 and i<21:
                arrivate_date_cat.append('mid_month')
            else:
                arrivate_date_cat.append('end_month')

        df['arrival_date'] = arrivate_date_cat

        skew_cols = [
        'no_of_children',
        'no_of_week_nights',
        'lead_time',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'no_of_special_requests'
        ]

        for col in skew_cols:
            df[col] = np.log1p(df[col])

        room_map = {'Room_Type 1':1,'Room_Type 2':2,'Room_Type 3':3,'Room_Type 4':4,'Room_Type 5':5,'Room_Type 6':6,'Room_Type 7':7}
        df['room_type_reserved'] = df['room_type_reserved'].map(room_map)

        df['market_segment_type'] = df['market_segment_type'].replace(['Corporate','Aviation','Complimentary'],'other')

        df=pd.get_dummies(df,columns=['type_of_meal_plan','required_car_parking_space','market_segment_type','repeated_guest','arrival_year','arrival_month','arrival_date'],drop_first=True)

        for col in df.columns:
                if df[col].dtypes == 'bool':
                    df[col]=df[col].astype('int')

        for col in expected_col:
            if col not in df.columns:
                df[col] = 0
        
        df=df[expected_col]

        predictions = loaded_model.predict(df)

        return render_template('index.html',prediction = predictions[0])
    return render_template('index.html',prediction = None)

if __name__ == '__main__':
    print("App running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0',port=5000)



