from flask import Flask, request, flash, render_template,redirect, url_for
from static.image_processor import GenderTransformer
import os.path
import os, shutil
import time

from static import config

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./static/images"

engine = GenderTransformer()

def clear_image_directory():
    dir = config.input_dir
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(url_for('index'))
    uploaded_image = request.files['file']

    if uploaded_image.filename == '':
        # flash('No selected file')
        return redirect(url_for('index'))

    clear_image_directory()
    global path, out_name
    path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_image.filename)
    print(path)
    out_name = 'out'
    uploaded_image.save(path)
    try:
        for gender in ['male', 'female']:
            engine.transform_gender(
                gender=gender,
                input_path=config.input_dir,
                input_name=uploaded_image.filename,
                output_name_specif=out_name
            )
    except IndexError:
        return "Sorry, we hadn't found any face on your image. Please, select another one"

    return redirect(url_for('results_page'))


@app.route('/results')
def results_page():
    # the next line is for making Flask not cache the image
    dynamic_picture_ensurer = str(time.time())

    return render_template("results.html",
                           results="Results",
                           male_img='.'+app.config["UPLOAD_FOLDER"]+'/'+out_name+'_male.jpg?random='+dynamic_picture_ensurer,
                           female_img='.'+app.config["UPLOAD_FOLDER"]+'/'+out_name+'_female.jpg?random='+dynamic_picture_ensurer
                           )


if __name__ == '__main__':
    app.run(ssl_context='adhoc')
