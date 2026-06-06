from flask import Flask, request, jsonify, render_template_string
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

status = {"message": "Listo"}

W = np.random.randn(784, 10) * 0.01
b = np.zeros(10)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def predict(x):
    z = np.dot(x, W) + b
    p = softmax(z[0])
    return int(np.argmax(p)), float(np.max(p))

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
<title>MNIST AI</title>
</head>
<body style="background:#111;color:white;text-align:center;font-family:Arial">

<h1>MNIST AI</h1>

<canvas id="c" width="280" height="280" style="background:black;border:2px solid white"></canvas>

<br><br>

<button onclick="predict()">Predecir</button>
<button onclick="clearC()">Limpiar</button>

<h2 id="out">-</h2>

<script>
let c=document.getElementById("c");
let ctx=c.getContext("2d");

ctx.strokeStyle="white";
ctx.lineWidth=20;

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
}

async function predict(){
let img=c.toDataURL();
let r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image:img})});
let d=await r.json();
document.getElementById("out").innerText=d.predicted_digit;
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
def pred():
    x = preprocess(request.get_json()["image"])
    p, conf = predict(x)
    return jsonify({
        "predicted_digit": p,
        "confidence": conf
    })

@app.route("/status")
def st():
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)