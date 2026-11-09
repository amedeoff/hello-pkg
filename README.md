# DDSM on HSE cHARISMa

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
noise cache, 20 optimizer steps и уменьшенную сеть, чтобы гарантированно
уложиться в очередь `test`.

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
- клонирует официальный DDSM в `/scratch/$USER/ddsm-reproduction/ddsm`;
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
4 CPU, 32 GB RAM и жёсткий лимит 30 минут.

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

Результаты:

```text
/scratch/$USER/ddsm-reproduction/runs/test-JOB_ID/
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
