from flask import Flask, request, jsonify
import numpy as np
import requests
import tensorflow as tf

model = tf.keras.models.load_model("model/brain_tumor_model.h5")
abc = {'glioma_tumor': 0, 'no_tumor': 1, 'meningioma_tumor': 2, 'pituitary_tumor': 3}

app = Flask(__name__)

@app.route('/')
def prediction():
    try:
        link = request.args['link']
        print(link)
        file_path = link
        content = requests.get(file_path).content
        content = tf.image.decode_jpeg(content, channels=3)
        content = tf.cast(content, tf.float32)
        content /= 255.0
        content = tf.image.resize(content, [150, 150])
        content = model.predict(content)
        com = {'predict': content}
        return jsonify(com)
        # for key in abc:
        #    return jsonify(com)
        #   if content == abc[key]:
        #        hj = key
        #        com = {'predict': hj}
        #        return jsonify(com)

    return 'hello'


if __name__ == '__main__':
    app.run()
