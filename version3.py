from flask import Flask, request, jsonify, render_template_string
import numpy as np
import base64
import io
import struct
import gzip
import urllib.request
import threading
from PIL import Image

app = Flask(__name__)

URLS = {
    "images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
}

X = None
y = None

status = {"loaded": False, "loading": False, "message": "No cargado"}

def download(url):
    r = urllib.request.urlopen(url, timeout=60)
    return gzip.decompress(r.read())

def images(data):
    m, n, r, c = struct.unpack(">IIII", data[:16])
    x = np.frombuffer(data[16:], dtype=np.uint8)
    return x.reshape(n, 784).astype(np.float32) / 255.0

def labels(data):
    m, n = struct.unpack(">II", data[:8])
    return np.frombuffer(data[8:], dtype=np.uint8)

def load():
    global X, y, status
    status["loading"] = True
    status["message"] = "Descargando..."
    try:
        img = download(URLS["images"])
        lab = download(URLS["labels"])
        X = images(img)
        y = labels(lab)
        status = {"loaded": True, "loading": False, "message": "OK", "samples": len(y)}
    except Exception as e:
        status = {"loaded": False, "loading": False, "message": str(e)}

class SimpleNN:
    def __init__(self):
        self.W = np.random.randn(784, 10) * 0.01
        self.b = np.zeros((1, 10))

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict(self, x):
        p = np.dot(x, self.W) + self.b
        p = self.softmax(p[0])
        return np.argmax(p), p

model = SimpleNN()

def preprocess(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, 784)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>MNIST V3</title>
</head>
<body style="background:#0f0f0f;color:white;text-align:center;font-family:Arial;">

<h2>MNIST V3 Cloud</h2>
<div id="status">Cargando...</div>

<canvas id="c" width="280" height="280" style="background:black;border:1px solid white;"></canvas><br>

<button onclick="predict()">Predecir</button>
<button onclick="clearC()">Limpiar</button>
<button onclick="sample()">Sample cloud</button>

<h1 id="out">-</h1>

<script>
let c=document.getElementById("c");
let ctx=c.getContext("2d");
ctx.strokeStyle="white";
ctx.lineWidth=15;

let drawing=false;

c.onmousedown=e=>{drawing=true;ctx.beginPath();}
c.onmouseup=e=>{drawing=false;}
c.onmousemove=e=>{
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

async function sample(){
let r=await fetch("/sample");
let d=await r.json();
let img=new Image();
img.onload=()=>ctx.drawImage(img,0,0,280,280);
img.src=d.image;
}

async function statusCheck(){
let r=await fetch("/status");
let d=await r.json();
document.getElementById("status")=d.message;
}  
statusCheck();
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/status")
def st():
    return jsonify(status)

@app.route("/predict", methods=["POST"])
def predict():
    x = preprocess(request.get_json()["image"])
    p, probs = model.predict(x)
    return jsonify({"predicted_digit": int(p), "confidence": float(probs[p])})

@app.route("/sample")
def sample():
    i = np.random.randint(0, len(X))
    img = (X[i].reshape(28, 28) * 255).astype(np.uint8)
    im = Image.fromarray(img)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"image": "data:image/png;base64," + b64})

if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    app.run(debug=True, port=5003)