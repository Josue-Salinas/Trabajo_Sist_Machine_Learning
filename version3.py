from flask import Flask, request, jsonify, render_template_string
import numpy as np
import base64
import io
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=20)
model.fit(X_train, y_train)

def preprocess(img_b64):
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, 784)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Predicción de números mediante imágenes</title>
<style>
body{
margin:0;
font-family:Arial;
background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
color:white;
text-align:center;
}

h1{
margin-top:20px;
color:#38bdf8;
text-shadow:0px 0px 10px #38bdf8;
}

.container{
margin-top:40px;
}

input{
padding:10px;
background:#1f2937;
color:white;
border-radius:8px;
border:1px solid #38bdf8;
}

button{
margin-top:20px;
padding:12px 22px;
border:none;
border-radius:10px;
background:#38bdf8;
color:black;
font-weight:bold;
cursor:pointer;
}

button:hover{
transform:scale(1.05);
}

#out{
font-size:50px;
margin-top:20px;
color:#facc15;
text-shadow:0px 0px 10px #facc15;
}
</style>
</head>
<body>

<h1>Predicción de números mediante imágenes</h1>

<div class="container">
<input type="file" id="file" accept="image/*"><br>
<button onclick="predict()">Predecir</button>

<div id="out">-</div>
</div>

<script>
async function predict(){
let file=document.getElementById("file").files[0];
let reader=new FileReader();

reader.onload=async function(){
let r=await fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({image:reader.result})
});
let d=await r.json();
document.getElementById("out").innerText=d.predicted_digit;
}

reader.readAsDataURL(file);
}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    x = preprocess(request.get_json()["image"])
    p = model.predict(x)[0]
    return jsonify({"predicted_digit": int(p)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)