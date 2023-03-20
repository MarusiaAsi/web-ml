import os
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join('static')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('моя первая сеть.h5')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html', title='')
    if request.method == 'POST':
        file = request.files['f']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

        # Преобразуем картинку в массив
        x = image.img_to_array(img)
        # Меняем форму массива в плоский вектор
        x = x.reshape(1, 784)
        # Инвертируем изображение
        x = 255 - x
        # Нормализуем изображение
        x /= 255

        prediction = model.predict(x)

        classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка',
                   'ботинки']
        prediction = np.argmax(prediction)

        return render_template('index.html', path=filename,
                               title=f' Нейросеть думает, что - это  {classes[prediction]}')


if __name__ == '__main__':
    app.run(port=5050, host='127.0.0.1')
