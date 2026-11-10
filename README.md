# DDSM on HSE cHARISMa

Текущая цель — четыре метода на DNA promoter benchmark. Подробный обзор,
источники, границы воспроизведения и аудит: [DNA_REVIEW.md](DNA_REVIEW.md).
Для проверки очереди используйте `bash tools/preflight_test.sh`, затем
`sbatch --partition=test jobs/gpu_probe_test.sbatch` из корня checkout.
Загрузка DNA непосредственно на кластер: `bash tools/download_promoter_data.sh`.
Подготовка promoter-файлов: `bash tools/prepare_promoter_data.sh`.
Потоковое построение genome memmap в `test`:
`sbatch --partition=test jobs/promoter_memmap_test.sbatch`.
Четыре common-backbone DNA adapters готовы. Короткая проверка всех процессов:

```bash
bash tools/setup_bio_dependencies.sh
bash tools/run_bio_selftest.sh
sbatch jobs/promoter_four_smoke.sbatch
```

После успешного smoke-теста запустите 30-минутные promoter pilots:

```bash
git pull --ff-only origin ddsm-charisma
sbatch --test-only jobs/promoter_four_pilot.sbatch
sbatch jobs/promoter_four_pilot.sbatch
```

Pilot array запускает четыре метода в `test`, не более двух задач одновременно:
1000 optimizer steps, batch 32, 8192/1024 promoter windows, четыре validation
batch и 128 генераций. Checkpoint обновляется каждые 100 шагов. Внутренний
лимит обучения 1050 секунд оставляет время на validation и sampling до
30-минутного лимита Slurm.

Проверка после завершения, где `JOB_ID` — номер array job из `sbatch`:

```bash
sacct -j JOB_ID --format=JobID,JobName,Partition,State,Elapsed,ExitCode,NodeList
source .dna_benchmark.env
python tools/summarize_smoke.py "${DNA_BENCHMARK_ROOT}/runs/pilot-JOB_ID"
```

Smoke и pilot являются common-backbone сравнением, а не paper-scale
reproduction. Бюджет pilot можно уменьшить без изменения файла:

```bash
sbatch --export=ALL,PILOT_MAX_STEPS=500,PILOT_SAMPLE_COUNT=64 \
  jobs/promoter_four_pilot.sbatch
```

Ниже сохранён прежний MNIST-тест официальной реализации DDSM.

Воспроизводимый запуск официальной реализации статьи:

> Avdeyev et al., *Dirichlet Diffusion Score Model for Biological
> Sequence Generation*, ICML 2023.

Официальный upstream: <https://github.com/jzhoulab/ddsm>

Зафиксированный commit: `ff20be49138589ae0f477b4039feafa7343de2a2`.

## Что именно мы запускаем

Первый job — 30-минутная интеграционная проверка DDSM на static binarized
MNIST. Он проверяет весь вычислительный путь:

1. бинарный пиксель переводится в категориальный one-hot вектор;
2. Jacobi/Wright–Fisher forward diffusion создаёт точку на симплексе;
3. сеть получает noisy sample и непрерывное время;
4. weighted denoising score-matching loss считается в stick-breaking
   координатах;
5. выполняются backward и optimizer step;
6. Euler–Maruyama sampler решает reverse SDE и сохраняет изображения.

Это проверка установки, а не значение из Table 1: тест использует маленький
noise cache, максимум 20 optimizer steps и уменьшенную сеть. Лимит очереди
30 минут; фактическая длительность пока не измерена, завершение не гарантируется.

Официальное значение статьи на binarized MNIST — `78.04 ± 0.37 nats`.
Его evaluation использует опубликованный checkpoint, cache из 100000
траекторий на 4000 временных точках и многократный stochastic Hutchinson
trace estimator. Такой расчёт не помещается в 30-минутный test job.

## 1. Подключение к cHARISMa

```bash
ssh -p 2222 YOUR_LOGIN@cluster.hpc.hse.ru
```

Расчёты на login-сервере запускать нельзя. На нём разрешены клонирование,
установка окружения и подготовка данных.

## 2. Скачать wrapper непосредственно на кластер

```bash
cd ~
git clone --branch ddsm-charisma --single-branch \
  https://github.com/amedeoff/hello-pkg.git ddsm-charisma
cd ddsm-charisma
```

Личный компьютер не используется.

## 3. Создать окружение и подготовить данные

```bash
bash setup_charisma.sh
```

Скрипт:

- загружает официальный модуль `python`/Miniconda;
- создаёт Conda environment `ddsm-repro` на Python 3.11;
- устанавливает закреплённые версии PyTorch и научных библиотек;
- определяет доступный scratch workspace командой `ws_find ft-models`;
- клонирует официальный DDSM в `<workspace>/ddsm-reproduction/ddsm`;
- переключает upstream на зафиксированный commit;
- скачивает fixed binarized-MNIST split на login-сервере;
- копирует только cluster wrapper в официальный checkout.

## 4. Проверить проект и очередь

```bash
mp
scontrol show partition test | grep -E 'PartitionName|MaxTime|State'
```

Если `mp` показывает несколько проектов, в последующей команде нужен
`-A PROJECT_ID`.

## 5. Поставить A100 test job

Для одного проекта:

```bash
sbatch jobs/ddsm_mnist_test.sbatch
```

Для нескольких проектов:

```bash
sbatch -A PROJECT_ID jobs/ddsm_mnist_test.sbatch
```

Запрашиваются `partition=test`, один узел `type_e`, одна NVIDIA A100 80 GB,
4 CPU и лимит 30 минут. Явный запрос `--mem=32G` удалён после ошибки Slurm.
Точная причина отказа требует вывода `scontrol`/`sinfo`; успешная отправка
в другую очередь не доказывает невозможность запуска в test.
Отправлять из корня checkout; для отправки из другого каталога экспортировать
`DDSM_WRAPPER_ROOT` с абсолютным путём к checkout.

## 6. Наблюдение

```bash
mj
mj --start
tail -f ddsm-mnist-test-JOB_ID.log
```

Отмена:

```bash
scancel JOB_ID
```

Для текущей учётной записи setup использует workspace `ft-models`, поэтому
рабочий корень будет иметь вид:

```text
/scratch/ws/tschichkanov-ft-models/ddsm-reproduction
```

Путь не захардкожен: он каждый раз определяется через `ws_find` и сохраняется
в `.ddsm_charisma.env` для Slurm job. При другом ID workspace можно выполнить:

```bash
DDSM_WORKSPACE_ID=OTHER_ID bash setup_charisma.sh
```

Результаты:

```text
<workspace>/ddsm-reproduction/runs/test-JOB_ID/
├── best_model.pt
├── last_checkpoint.pt
├── metrics.jsonl
├── samples.png
└── time_weights.pt
```

Успешный тест обязан показать `cuda_available True`, конечные train/validation
loss и строку `saved=...`. После него можно переходить к тяжёлому официальному
paper evaluation; сначала тестируем инфраструктуру, чтобы не расходовать часы
A100 на ошибку окружения.

## Официальная документация cHARISMa

- <https://hpc.hse.ru/instructions/base>
- <https://hpc.hse.ru/instructions/run>
- <https://hpc.hse.ru/instructions/python/anaconda>
- <https://hpc.hse.ru/hardware/hpc-cluster>
