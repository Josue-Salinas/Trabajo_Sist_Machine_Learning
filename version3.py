from flask import Flask, request, jsonify, render_template_string
import numpy as np
import base64
import io
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

mnist = fetch_openml("mnist_784", version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)

model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, verbose=False)
model.fit(X, y)

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
<title>Reconocimiento de Números mediante Escritura</title>
<style>
body{
margin:0;
font-family:Arial;
background:linear-gradient(135deg,#0f0f0f,#1a1a2e,#16213e);
color:white;
text-align:center;
}

h1{
margin-top:20px;
color:#00ffe1;
text-shadow:0px 0px 10px #00ffe1;
}

p{
color:#bbb;
}

canvas{
background:black;
border:3px solid #00ffe1;
border-radius:10px;
box-shadow:0px 0px 15px #00ffe1;
}

button{
margin:10px;
padding:10px 20px;
border:none;
border-radius:8px;
cursor:pointer;
font-size:16px;
}

.btn1{background:#00ffe1;color:black;font-weight:bold;}
.btn2{background:#ff4d6d;color:white;}

#out{
font-size:28px;
color:#ffd369;
margin-top:20px;
}
</style>
</head>
<body>

<h1>Reconocimiento de Números mediante Escritura</h1>
<p>Dibuja un número del 0 al 9</p>

<canvas id="c" width="280" height="280"></canvas>

<br>

<button class="btn1" onclick="predict()">Predecir</button>
<button class="btn2" onclick="clearC()">Limpiar</button>

<div id="out">Resultado: -</div>

<script>
let c=document.getElementById("c");
let ctx=c.getContext("2d");

ctx.strokeStyle="#00ffe1";
ctx.lineWidth=18;
ctx.lineCap="round";

let drawing=false;

c.onmousedown=()=>{drawing=true;ctx.beginPath();}
c.onmouseup=()=>{drawing=false;}
c.onmousemove=(e)=>{
if(!drawing)return;
ctx.lineTo(e.offsetX,e.offsetY);
ctx.stroke();
}

function clearC(){
ctx.fillStyle="black";
ctx.fillRect(0,0,280,280);
document.getElementById("out").innerText="Resultado: -";
}

async function predict(){
let img=c.toDataURL();
let r=await fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({image:img})
});
let d=await r.json();
document.getElementById("out").innerText="Resultado: " + d.predicted_digit;
}

clearC();
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
    prob = np.max(model.predict_proba(x))
    return jsonify({
        "predicted_digit": int(p),
        "confidence": float(prob)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)