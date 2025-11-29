from flask import Blueprint, render_template, request, jsonify, send_file
import uuid, os
from .tasks import process_file
from threading import Thread

main = Blueprint("main", __name__)


@main.route("/")
def index():
    return render_template("upload.html")


@main.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    task_id = str(uuid.uuid4())

    input_path = f"app/uploads/{task_id}.dat"
    result_path = f"app/static/results/{task_id}.txt"
    plot_path = f"app/static/results/{task_id}.png"

    file.save(input_path)

    # запуск фоновой обработки
    Thread(target=process_file, args=(task_id, input_path, result_path, plot_path)).start()

    return jsonify({"task_id": task_id})


@main.route("/download/<task_id>")
def download(task_id):
    return send_file(f"app/static/results/{task_id}.txt", as_attachment=True)
