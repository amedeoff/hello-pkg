# Четыре семейства моделей на DNA: протокол воспроизведения и аудит

Дата проверки: 2026-11-09. Статус: обзор первоисточников и статический аудит кода;
полные DNA-эксперименты в этом репозитории пока НЕ реализованы и НЕ запущены.
Кластер доступен пользователю, прямого SSH-доступа из этой сессии нет.

## 1. Выбранная задача

Human promoter generation conditioned on transcription initiation signal:
генерация 1024 нуклеотидов вокруг TSS по профилю CAGE. Это опубликованный
общий бенчмарк DDSM и Dirichlet FM. Называть его единственным универсальным
стандартом для всей генерации DNA было бы неправильно.

Основной источник данных: https://zenodo.org/records/7943307.
Архив data.tar.gz около 10.2 GB, best_models.tar.gz около 199.8 MB.
Это размер загрузки, а не требуемое место после распаковки и создания genome memmap.
Архив включает дополнительные задачи; для нас нужен promoter_design.
До распаковки проверить содержимое архива, README внутри него, свободное место,
наличие FASTA, индекса, bigWig, blacklist/tabix, весов Sei и списка его признаков.
Не считать наличие data.tar.gz признаком готовности датасета.

Проверено непосредственно в коде обоих репозиториев:

- длина окна 1024; GRCh38; FANTOM_CAT TSS;
- сначала выбираются первые 100000 строк TSS, затем применяется split;
- train: кроме chr8, chr9, chr10; validation: chr10; test: chr8 и chr9;
- у отрицательной цепи есть сдвиг координат на 1 и разворот каналов/позиционной оси сигнала;
- Dirichlet FM при обучении использует rand_offset=10, при оценке 0;
- в promoter_module условием выступает batch[:, :, 4:5]; нельзя самовольно
  заменить его двумя каналами и ожидать совместимости checkpoint;
- размеры фактических split надо вычислить после загрузки, а не объявлять 100000 train samples.

Сохранять upstream-разбиение для повторения опубликованного эксперимента.
Отдельно измерить совпадения и reverse-complement совпадения между split:
разбиение по хромосомам само по себе не гарантирует отсутствие гомологов.
Любое дополнительное удаление дублей меняет benchmark и требует отдельного варианта.

## 2. Какие именно четыре модели

| Семейство | Конкретный исходный метод | Что воспроизводим |
|---|---|---|
| Discrete diffusion | D3PM-uniform / multinomial diffusion | DNA-baseline из сравнения DDSM; не произвольный masked denoiser |
| Discrete flow matching | Gat et al., Discrete Flow Matching | Метод на DNA с явно обозначенным адаптером |
| Dirichlet diffusion | Avdeyev et al., DDSM | Авторский promoter experiment |
| Dirichlet flow matching | Stark et al., Dirichlet FM | Авторский promoter experiment |

### D3PM

Нуклеотиды остаются категориями. Для uniform-переходов
Q_t=(1-beta_t)I+beta_t*11^T/K, K=4; накопленный переход — произведение Q_t.
Сеть оценивает распределение чистого токена по зашумлённой последовательности,
времени и CAGE. Обратный шаг строится через точный categorical posterior.
Нужно воспроизвести scheduler, число шагов, objective (variational/hybrid),
параметризацию и декодирование конкретного DNA-baseline. Одна CE и argmax
на каждом шаге не являются автоматически реализацией D3PM.
Проверки: строки Q нормированы, вероятности неотрицательны; posterior для K=4
сходится с прямым перебором; корректен последний шаг t=1; prior соответствует Q.

### Discrete FM

Путь лежит в пространстве категориальных распределений. Для mixture-path
каждая позиция берётся из source или target согласно scheduler kappa(t).
Сеть предсказывает posterior; sampler использует интенсивности переходов CTMC.
Выбор uniform или mask source — существенная часть спецификации. В первом
DNA-сравнении разумно использовать uniform source с K=4; mask-вариант — отдельно.
Нельзя заменить CTMC непрерывной интерполяцией one-hot: это другой метод.
Проверки: неотрицательные внедиагональные rates, нулевая сумма строки генератора,
корректность шага solver, обработка сингулярности около t=1 и сохранение категорий.
Официальная библиотека Meta содержит MixtureDiscreteProbPath и
MixtureDiscreteEulerSolver. Её текстовый pipeline не является готовым DNA benchmark.

### DDSM

Каждая позиция — точка трёхмерного симплекса в R^4. Forward-процесс
использует Jacobi diffusion и stick-breaking; стационарная мера — Dirichlet.
Обучается score с соответствующими весами; генерация решает reverse SDE.
Это не добавление Gaussian noise к one-hot и не Dirichlet FM с другим названием.
Проверки: преобразование score между координатами, K=4 и K-1 stick variables,
speed_balance, интервал времени, endpoint epsilon, noise-cache metadata,
порядок букв и корректная трактовка time dilation.
Исходная команда подготовки noise в USAGE.md содержит опечатку
`---speed_balance`; фактический argparse принимает `--speed_balance`.
Большой noise cache требует отдельного измерения времени и памяти, его нельзя
незаметно генерировать заново в каждом 30-минутном задании.

### Dirichlet FM

Использует условный путь Dirichlet(1+t*one_hot(x_clean)), denoising CE и
выведенное для этого пути векторное поле. CE сама по себе не определяет sampler.
Генерация — интегрирование ODE, с контролем принадлежности симплексу.
Проверки: временная параметризация/alpha, таблица c_factor, конечность коэффициентов,
сумма координат и минимальная координата. Частые проекции и nan_to_num нельзя
считать безвредными: логировать число вмешательств и сравнивать с upstream.
Режим `riemannian` в этом репозитории обозначает linear FM, а не Discrete FM Gat.
Distillation — отдельный эксперимент с teacher и дополнительной стоимостью.

## 3. Воспроизведение и общее сравнение

Сначала повторить native DDSM и Dirichlet FM: исходная сеть, обработка данных,
checkpoint, conditioning, evaluator, sampler и выбранный протокол статьи.
Для D3PM требуется найти/восстановить конфигурацию DNA-baseline DDSM; базовый
авторский D3PM-репозиторий сам по себе её не предоставляет.
Для Discrete FM нужен DNA-adapter и верификация на маленьком категориальном
распределении с известным ответом. Эта строка не является повторением DNA-таблицы Gat.

Затем отдельное контролируемое сравнение: одинаковые split, CAGE, доступ к данным,
сопоставимый backbone и одинаковые правила выбора checkpoint. Изменения входного
слоя для разных процессов допустимы, но число параметров нужно вывести в отчёт.
Нельзя одновременно требовать побитовую верность разным статьям и менять все
архитектуры на одну без обозначения нового протокола.

Для короткой проверки: один фиксированный subset train, 1–2 validation batches,
несколько optimizer steps, 8–16 generated sequences, проверка сохранения/загрузки.
Размер subset, batch и число шагов — настройки проверки, не утверждения из статьи.
Затем короткий пилот с измерением секунд/step и sampling/evaluation overhead.
По пилоту выбирать реальный бюджет; завершение четырёх обучений за 30 минут
не подтверждено. Очередь test остаётся для самостоятельных ограниченных проверок.
Для длительного обучения следует проверить правила очереди и выбрать разрешённый
режим; не дробить расчёт для обхода установленного лимита.

Полная оценка: минимум 3 seeds; checkpoint выбирается только по validation;
test используется после фиксации конфигурации. Сохранять все seeds, не лучший из них.
Равные epochs не равны равному compute: сообщать GPU-hours, train steps, samples seen,
число параметров, peak VRAM, sampling NFE и wall-clock с CUDA synchronization.

## 4. Метрики

Главная paper-метрика promoter benchmark — SP-MSE по фиксированному Sei evaluator.
В проверенном коде окно дополняется по 1536 позиций с каждой стороны до 4096,
выбираются H3K4me3 признаки, затем получается агрегированная оценка.
Крайне важно сохранить checkpoint Sei, порядок каналов и точную агрегацию.
SP-MSE отражает surrogate-предсказание, не экспериментальную активность DNA.

Дополнительные диагностические метрики (не выдавать за оригинальные):
GC-распределение, 3–6-mer частоты/JS divergence, разнообразие, доля точных совпадений
с train, nearest-neighbor similarity и reverse-complement duplicates.
Считать их на равном числе generated sequences и с доверительными интервалами.
Training loss DDSM, D3PM и FM имеет разные масштабы и не ранжирует качество между моделями.
FID из MNIST и enhancer FBD не являются готовой заменой promoter SP-MSE.

## 5. Найденные ошибки и зоны риска

| Проблема | Действие/проверка |
|---|---|
| Отказ Slurm с --mem=32G | Повторить test без явного mem; причина ещё не доказана, нужны scontrol/sinfo |
| Переключение сразу двух параметров | Исправлено: возвращён test; принятие gpu-ef-quick ничего не доказывает о test |
| BASH_SOURCE у Slurm указывает в spool | Исправлено: SLURM_SUBMIT_DIR либо DDSM_WRAPPER_ROOT |
| JOB_ID как буквальный текст | В командах подставлять полученный номер |
| Workspace истекает | ws_list, контроль срока и сохранение важных checkpoint до очистки |
| Только импорт torch проверен | GPU probe выполняет forward/backward внутри test allocation |
| CUDA runtime/driver несовместимы | Проверка nvidia-smi и реальная CUDA-операция, а не torch.version.cuda |
| Все зависимости в одном env | Раздельные env; pip check; freeze после рабочего запуска |
| README Dirichlet FM смешивает версии torch/CUDA/PyG | Не исполнять весь список слепо; проверить реально импортируемые модули |
| Selene fork и C extensions | Зафиксировать commit custom_target_support; проверить компилятор, Cython/NumPy ABI |
| Интернет отсутствует на compute node | Загрузить данные/веса/пакеты заранее, W&B выключить или offline |
| 10.2 GB архив и genome memmap | Проверить квоту и распакованный объём; не держать весь геном в каждом worker |
| Несогласованные пути DDSM | Код ищет ../data/*.fa, архив описан как data/promoter_design; нужен проверенный path adapter |
| Условие и ориентация DNA | Проверить A/C/G/T mapping, reverse complement, strand offset, CAGE канал |
| Неоднозначные N/blacklists | Посчитать, повторить политику автора; argmax для N может создать ложные A |
| PyTorch 2.6+ weights_only default | Работать с закреплённой версией и доверенными upstream checkpoint |
| Lightning/torch checkpoint несовместимы | Отдельный load test до обучения; strict shapes; не скрывать missing keys |
| os.getlogin() в batch | Может бросить OSError без tty; заменить в wrapper на getpass.getuser при необходимости |
| Потенциальная ошибка Sei cache у Dirichlet FM | promoter_module.py присваивает self.sei_cache = sei_profile вместо словаря по batch_idx; требует regression check и явного patch |
| NaN c_factor и simplex projection | Счётчики, fail on excessive repairs, уменьшение шага и сравнение upstream |
| DataLoader зависает/OOM | Начать num_workers=0, затем 2/4; ограничить OpenMP; измерять RSS |
| Тайм-аут до сохранения | Atomic periodic checkpoint, optimizer/RNG/step restore, проверить resume |
| Параллельные jobs портят cache | Один подготовительный writer, read-only caches после проверки |
| Случайное использование test для early stopping | Раздельные конфигурации validation/test; manifest sample IDs |

Это реестр проверенных и вероятных проблем, не гарантия перечисления всех ошибок.
Предполагаемые дефекты авторского кода нужно подтвердить тестом до изменения формул.

## 6. Порядок запуска сейчас

В корне checkout на login node:

```bash
bash tools/preflight_test.sh
sbatch --partition=test jobs/gpu_probe_test.sbatch
```

preflight не отправляет задачу: печатает конфигурацию и вызывает sbatch --test-only.
Если Slurm отклоняет запрос, сохранить полный вывод и не менять очередь автоматически.
gpu_probe — отдельный трёхминутный GPU-тест, не обучение DNA или MNIST.
Старый job 4499418 при необходимости отменяется отдельно: scancel 4499418.

Загрузка исходных данных непосредственно на кластер:

```bash
bash tools/download_promoter_data.sh
```

Скрипт проверяет опубликованные MD5 и сохраняет архивы в
$(ws_find ft-models)/dna-benchmark/downloads. Распаковку и тяжёлый preprocessing
планировать после изучения README архива и бюджета RAM/диска.

Готовность к четырём тренировкам требует ещё: data manifest, окружения с проверенными
imports, четыре adapter/sampler tests, общий evaluator и checkpoints/resume.
Публикация этого обзора не означает, что эти этапы уже выполнены.

## 7. Первичные источники и зафиксированный код

- D3PM, Austin et al. (NeurIPS 2021): https://arxiv.org/abs/2107.03006
  Авторский код: https://github.com/google-research/google-research/tree/master/d3pm
- Discrete FM, Gat et al. (NeurIPS 2024): https://arxiv.org/abs/2407.15595
  https://github.com/facebookresearch/flow_matching
  Проверен checkout 11568d37f8d5a080e12aa7b5305d9c35ae07d136.
- DDSM, Avdeyev et al. (ICML 2023): https://arxiv.org/abs/2305.10699
  https://github.com/jzhoulab/ddsm/tree/ff20be49138589ae0f477b4039feafa7343de2a2
- Dirichlet FM, Stark et al. (ICML 2024): https://proceedings.mlr.press/v235/stark24b.html
  https://github.com/HannesStark/dirichlet-flow-matching/tree/6f360612da8f69ecf4860c075f74ed52b37ce64b
- Данные и веса: https://zenodo.org/records/7943307
- cHARISMa: https://hpc.hse.ru/instructions/run

Авторские репозитории изучены статически. На GPU cHARISMa новые DNA-компоненты
не проверены. Опубликованные числа не записываются как наши измеренные результаты.
