
from flask import Flask, send_from_directory
import pathlib
app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/stream/<path:path>')
def stream_files(path):
    return send_from_directory('stream', path)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
