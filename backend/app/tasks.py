import time
import matplotlib.pyplot as plt
from . import socketio
from app.model import MlModel


def process_file(task_id, input_path, result_path, plot_path):
    model = MlModel()
    model.process_file(input_path, result_path)

    # 10 шагов обработки
    for i in range(11):
        percent = i * 10
        socketio.emit("progress", {
            "task_id": task_id,
            "progress": percent
        })
        model.process_file(input_path, result_path)

    # создаём файл результата
    with open(result_path, "w") as f:
        f.write("File processed successfully.\n")

    # генерируем график
    plt.plot([1,2,3,4,5], [1,4,2,8,3])
    plt.title("График обработки")
    plt.savefig(plot_path)

    # отправляем сообщение о завершении
    socketio.emit("finished", {
        "task_id": task_id,
        "result_url": f"/download/{task_id}",
        "plot_url": f"/static/results/{task_id}.png"
    })
