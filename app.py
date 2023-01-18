from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def run():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   BMI = int(request.form['a'])
   Smoking = int(request.form['b'])
   AlcoholDrinking = int(request.form['c'])
   DiffWalking = int(request.form['d'])
   Sex = int(request.form['e'])
   AgeCategory = int(request.form['f'])
   race = int(request.form['g'])
   Diabetic = int(request.form['h'])
   PhysicalActivity = int(request.form['i'])
   GenHealth = int(request.form['j'])
   SleepTime = int(request.form['k'])
   arr = np.array([[BMI, Smoking, AlcoholDrinking, DiffWalking, Sex, AgeCategory, race, Diabetic, PhysicalActivity, GenHealth, SleepTime]])
   hdp = model.predict(arr)
   print(hdp)
    # output = '{0:.{1}f}'.format(hdp[0])
   if hdp==0:
    ans= "You are not at risk"
   else:
    ans="You are at risk"
   return render_template('index.html', pred = ans)

if __name__ == '__main__':
    app.run(debug=True)

