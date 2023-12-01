import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import gradio as gr

def predict(studyHours, prevScore, ExtraActivity, SleepHours, QP_Practise):

    # Reading data(csv_file)
    df = pd.read_csv('data\Student_Performance.csv')

    # Handling missing values or NA values
    if df.shape[0] < 1000:
        df.fillna(0) 
    else:
        df.dropna()

    # Converting the categorical values to numerical values
    encoder = LabelEncoder()
    label = encoder.fit_transform(df['Extracurricular Activities'])
    df['Extracurricular Activities'] = label

    # Splitting the data
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values.reshape(-1,1)
    print(y)
    print(y.shape)

    # Scaling the data
    scaler = StandardScaler()
    x_sc = scaler.fit_transform(x)

    # Training the model
    model = LinearRegression()
    model.fit(x_sc, y)

    # Encoding categorical value
    extAct = int(encoder.transform([ExtraActivity]))

    # Predicting the data
    pred = model.predict(scaler.transform([[studyHours,prevScore,extAct,SleepHours,QP_Practise]]))
    conv_pred = int(pred)
    round_pred = round(conv_pred,2)

    return round_pred

iface = gr.Interface(
    title="Predicting Student Performance",
    fn = predict,
    inputs = [
        gr.Number(value=0,label='Studied hours',info='Total studied hours per day'),
        gr.Slider(0,100,value=0,label='Previous score',info='Scores should be in between 0 to 100'),
        gr.Radio(['Yes','No'],label='Extracuricular Activities',info='Student participatres in Extarcurricular activities'),
        gr.Number(value=0,label='Sleep Hours',info='Total sleep hours per day'),
        gr.Number(value=0,label='Practised QP',info='Total practised Question papers')
    ],
    outputs = [
        gr.Number(label='Student Performance')
    ]
)

if __name__ == "__main__":
    iface.launch()