# U-Net 128x128 Moon Segmentation

Проект для бинарной сегментации лунных изображений с помощью U-Net.
- Вход: 128x128 RGB
- Выход: 128x128 бинарная маска
- Модель: U-Net, base_channels=32, глубина=4
- Датасет: data/images и data/masks

Запуск:
1. Клонируем репозиторий
2. Устанавливаем зависимости: pip install -r requirements.txt
3. Кладём картинки в data/images и маски в data/masks
4. Запускаем обучение: python train.py
