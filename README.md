# Hybrid Visual SLAM for UAV: ORB-SLAM3 + ZoeDepth

Гибридная система монокулярного визуального SLAM для беспилотного летательного аппарата. Классический алгоритм ORB-SLAM3 дополнен нейросетевой оценкой метрической глубины (ZoeDepth): карта глубины используется для перераспределения ORB-признаков в пользу геометрически достоверных областей кадра до начала трекинга. Цель — повысить устойчивость инициализации и сопровождения траектории при низкой освещённости, осадках и малотекстурных сценах.

Проект выполнен в рамках выпускной квалификационной работы (СПбГЭТУ «ЛЭТИ», направление 01.03.02 «Прикладная математика и информатика»).

## Возможности

- Два режима работы: базовый монокулярный ORB-SLAM3 (`orbslam`) и глубинно-управляемый (`ade_mono_depth`).
- Оценка метрической глубины ZoeDepth (варианты `N`, `K`, `NK`), поддержка CUDA и FP16, прореживание инференса (`inference_stride`) и понижение входного разрешения.
- Отбор признаков по глубине, реализованный в C++ внутри ORB-SLAM3: рекурсивное квадродеревянное разбиение кадра, жадный выбор точек по пространственному и глубинному критериям, стохастическое удаление дальних точек, откат к исходному набору признаков при недостаточном удержании.
- Адаптивная активация управления: по числу подряд успешных кадров, по состоянию трекинга (инициализация / восстановление), по оценке информативности карты глубины, с адаптивным порогом глубины (перцентиль карты).
- Предобработка для низкой освещённости: CLAHE, гамма-коррекция, шумоподавление, повышение резкости.
- Источники данных: манифест с временными метками, видеофайл, последовательность изображений, UDP-поток.
- Оценка точности (ATE, RPE) с выравниванием Умеямы (Sim(3)) через пакет `evo`.
- Визуализация: интерактивные 3D-графики Plotly (траектория, облако точек, карты Atlas), вид сверху с наложением эталонной траектории.
- Режим сравнения: последовательный запуск базового и глубинного вариантов с итоговым отчётом `depth_comparison.json`.

## Архитектура

```
Источник кадров ──► [Улучшение изображения] ──► ZoeDepth ──► Оценка карты глубины
 (media_sources)        (LowLightEnhancer)     (depth_estimation)        │
                                                                         ▼
                                    Решение об активации управления (DepthSlam)
                                                                         │
                                  ┌──────────────────────────────────────┴───────┐
                                  ▼                                              ▼
              process_mono_depth_guided (C++)                         process_mono (C++)
              отбор признаков по глубине + ORB-SLAM3                 штатный ORB-SLAM3
                                  └──────────────────────┬───────────────────────┘
                                                         ▼
                              Траектория, карта, метрики, графики (runs/…)
```

| Каталог / файл | Назначение |
|---|---|
| `main.py` | Точка входа: разбор конфигурации, выбор конвейера, режим сравнения, оценка траектории |
| `media_sources/` | Конфигурация (dataclass-модели YAML/JSON), источники кадров, контракт `FramePacket` |
| `image_processing/depth_slam.py` | Конвейер `ade_mono_depth` |
| `image_processing/orbslam_runner.py` | Базовый конвейер `orbslam`, сохранение результатов |
| `image_processing/depth_estimation.py` | Обёртка ZoeDepth |
| `image_processing/depth_guidance.py` | Эталонная Python-реализация отбора признаков (для тестов; в рантайме используется C++) |
| `image_processing/image_enhancement.py` | Предобработка для низкой освещённости |
| `image_processing/trajectory_evaluation.py` | ATE/RPE через `evo` |
| `visualization/` | Живой просмотр и построение графиков |
| `orbslam/wrapper.cpp`, `orbslam/CMakeLists.txt` | pybind11-обёртка `pyorbslam3_custom` |
| `configs/` | Примеры конфигураций для наборов данных |

## Требования

- Linux (x86_64), Python 3.12 (имя собранного расширения содержит `cpython-312`).
- Модифицированная сборка ORB-SLAM3 с методом `TrackMonocularDepthGuided`, структурой `DepthGuidanceConfig` и методами статистики отбора признаков в `Tracking`. Обёртка не соберётся с оригинальным ORB-SLAM3.
- CMake ≥ 3.10, pybind11, OpenCV, Eigen3, OpenGL, GLEW.
- Python-пакеты: `numpy`, `opencv-python`, `pyyaml`, `pillow`, `torch` (с CUDA для режима `device: cuda`), `plotly`, `matplotlib`, `evo`.

## Установка

```bash
git clone <repo-url>
cd <repo>

python3.12 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python pyyaml pillow torch plotly matplotlib evo
```

Сборка расширения `pyorbslam3_custom`:

```bash
cd orbslam
mkdir -p build && cd build
cmake .. -DORB_SLAM3_DIR=/path/to/ORB_SLAM3
cmake --build . -j
```

Собранный модуль помещается в корень проекта (`LIBRARY_OUTPUT_DIRECTORY` задан как `..` относительно исходного каталога `orbslam`). После изменений в ORB-SLAM3 достаточно повторить `cmake --build .`.

Веса ZoeDepth загружаются через `torch.hub` при `weights_path: auto`; также можно указать путь к локальному файлу весов.

## Запуск

Базовый монокулярный ORB-SLAM3:

```bash
python3 main.py --config configs/office_loop_1/office_loop_1_train.example.yaml
```

Глубинно-управляемый конвейер:

```bash
python3 main.py --config configs/old_town_4/old_town_4_train_depth.yaml
```

Сравнительный запуск (кандидат против базового варианта):

```bash
python3 main.py \
  --config configs/old_town_4/old_town_4_train_depth.yaml \
  --compare-against configs/old_town_4/old_town_4_train.example.yaml
```

Дополнительные параметры: `--log-level DEBUG`, `--runner-log-mode {full,fps_adaptive}`.
