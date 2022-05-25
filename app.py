from flask import Flask, request
from AiModel import detectIssue

app = Flask(__name__)

@app.route('/detect-issue', methods=['POST'])
def detectIssues():
    board_image_id = request.form.get('boardImageId')
    image = request.files['image']
    result = detectIssue(image)

    return {'boardImageId': board_image_id, 'issues': result}
