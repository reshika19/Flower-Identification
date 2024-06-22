from flask import Flask, request, render_template
from flower import recognize_flower, get_wikipedia_link

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # Save file to disk
        filepath = 'uploads/' + f.filename
        f.save(filepath)
        # Run flower recognition code
        flower = recognize_flower(filepath)
        wikipedia_link = get_wikipedia_link(flower)
        return render_template('index.html', flower=flower, wikipedia_link=wikipedia_link)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
