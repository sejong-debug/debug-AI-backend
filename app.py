from flask import Flask, request


app = Flask(__name__)

@app.route('/detect-issue', methods=['POST'])
def detectIssues():
    image = request.files['image']
    
    
    return image.filename