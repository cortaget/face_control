import os
import shutil

# Путь к папке img
folder_path = "img"

# Проверяем, существует ли папка
if os.path.exists(folder_path):
    # Удаляем все файлы и подпапки в img
    shutil.rmtree(folder_path)
    print(f"Все файлы из папки '{folder_path}' были удалены.")

    # Создаем пустую папку img
    os.makedirs(folder_path)
    print(f"Папка '{folder_path}' была создана заново.")
else:
    print(f"Папка '{folder_path}' не найдена.")
