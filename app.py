from flask import Flask, request
from AiModel import detectIssue

app = Flask(__name__)

@app.route('/detect-issue', methods=['POST'])
def detectIssues():
    image = request.files['image']
    result = detectIssue(image)

    return {'issues': result}
