"""English and Russian UI text. English strings are the keys; a missing Russian entry falls back to English.
Numbers follow the language: 1,234.5 in English, 1 234,5 in Russian."""
LANGS = ('en', 'ru')

RU = {
    # navigation and layout
    'Queue': 'Очередь', 'Live input': 'Живой ввод', 'Evidence': 'Доказательства', 'Audit': 'Аудит',
    'Switch light / dark theme': 'Светлая / тёмная тема',
    'the officer decides, not the model': 'решение принимает сотрудник, а не модель',
    'no external APIs: everything is computed on this server': 'без внешних API: всё считается на этом сервере',
    'DEV mode: the models are temporarily trained on the synthetic demo data.': 'Режим DEV: модели временно обучены на синтетических демо-данных.',
    # queue
    'Alert queue': 'Очередь алертов',
    'Demo bank: {n} transactions (latest {t}), each scored by LightGBM and the GNN. Riskiest first.':
        'Демо-банк: {n} транзакций (последняя {t}), каждая оценена LightGBM и GNN. Самые рискованные сверху.',
    'Added live: {n}.': 'Добавлено вживую: {n}.',
    'High (top 0.5%)': 'Высокий (топ 0,5%)', 'Medium (up to top 2%)': 'Средний (до топ 2%)', 'Low': 'Низкий',
    'High': 'Высокий', 'Medium': 'Средний', 'All': 'Все', 'Filter by level': 'Фильтр по уровню',
    'Time': 'Время', 'Sender → Receiver': 'Отправитель → Получатель', 'Amount': 'Сумма', 'Channel': 'Канал', 'Risk': 'Риск',
    'Level': 'Уровень', 'Decision': 'Решение', 'live': 'вживую', 'previous 50': 'предыдущие 50', 'next 50': 'следующие 50',
    'confirmed': 'подтверждено', 'escalated': 'передано', 'dismissed': 'отклонено',
    # account
    'Account': 'Счёт', 'Direction': 'Направление', 'Counterparty': 'Контрагент', 'out': 'исходящий', 'in': 'входящий',
    '{n} transactions': 'транзакций: {n}', 'highest score {p}': 'максимальная оценка {p}',
    # alert
    'Alert': 'Алерт', 'Transaction': 'Транзакция', 'added live': 'добавлена вживую',
    'probability of money laundering': 'вероятность отмывания денег',
    'rank {r} of {n}': 'место {r} из {n}', 'rank {r}': 'место {r}',
    'Who → whom': 'Кто → кому', 'Sender': 'Отправитель', 'Receiver': 'Получатель',
    'not a bank customer / external account': 'не клиент банка / внешний счёт',
    'Two models': 'Две модели',
    'LightGBM (the payment, account history and earlier risk)': 'LightGBM (платёж, история счетов и прошлый риск)',
    'GNN (the network of accounts)': 'GNN (сеть счетов)',
    'Final score: the two model scores combined with weights learnt on the validation period (stacking).':
        'Итоговая оценка: оценки двух моделей, объединённые с весами, обученными на валидационном периоде (стекинг).',
    'Why (SHAP)': 'Почему (SHAP)', 'raises risk': 'повышает риск', 'lowers risk': 'снижает риск',
    "LightGBM's contribution of each feature (TreeSHAP), grouped by meaning; the numbers are this transaction's actual features.":
        'Вклад каждого признака LightGBM (TreeSHAP), сгруппированный по смыслу; числа — реальные признаки этой транзакции.',
    'Evidence subgraph: transactions around the two accounts (±7 days)': 'Подграф доказательств: транзакции вокруг двух счетов (±7 дней)',
    'Money flows between the accounts': 'Денежные потоки между счетами', 'risk': 'риск',
    'The thick line is this alert. Every line and account can be clicked.': 'Толстая линия — этот алерт. Любую линию и счёт можно открыть.',
    'Officer decision': 'Решение сотрудника', 'Reason': 'Причина', 'Officer': 'Сотрудник', 'Full name': 'Имя и фамилия',
    'For example: money from 14 different accounts collected in one day, then withdrawn in cash':
        'Например: за один день собраны деньги с 14 разных счетов, затем сняты наличными',
    'Save decision (written to the audit log)': 'Сохранить решение (записывается в журнал аудита)',
    'Score at the time: {p} · audit entry #{n} · hash {h}…': 'Оценка на тот момент: {p} · запись аудита №{n} · хеш {h}…',
    'Confirm as suspicious': 'Подтвердить как подозрительную',
    'Escalate to the financial monitoring unit': 'Передать в отдел финансового мониторинга',
    'Dismiss (false alarm)': 'Отклонить (ложная тревога)',
    # live input
    'Add a new transaction or a whole scheme to the demo bank: features are computed on the spot, both models score it and the queue updates. The models are not told what was added; they only see the transactions.':
        'Добавьте в демо-банк новую транзакцию или целую схему: признаки считаются сразу, обе модели дают оценку, очередь обновляется. Моделям не сообщают, что добавлено; они видят только транзакции.',
    'Demo time: {t}.': 'Время демо: {t}.', 'Added so far: {n}.': 'Добавлено: {n}.',
    '1. One transaction': '1. Одна транзакция', '2. A whole scheme': '2. Целая схема',
    'Sender (demo customer)': 'Отправитель (демо-клиент)', "Receiver's bank": 'Банк получателя', 'Account number': 'Номер счёта',
    'Amount (UZS)': 'Сумма (UZS)', 'Minutes after the demo time': 'Минут после времени демо', 'Add and score': 'Добавить и оценить',
    'Central account': 'Центральный счёт', 'Scheme': 'Схема',
    'Fan-out: one account pays many accounts': 'Fan-out: один счёт платит многим',
    'Mule fan-in: collect from many accounts, then cash out': 'Mule fan-in: сбор со многих счетов и обналичивание',
    'Cycle: transfers around 4 accounts and back': 'Цикл: переводы по кругу через 4 счёта',
    'Number of transactions': 'Число транзакций', 'Add the scheme and score it': 'Добавить схему и оценить',
    'Reset the demo bank to its initial state': 'Вернуть демо-банк в исходное состояние',
    'Accounts the model finds riskiest': 'Счета, которые модель считает самыми рискованными',
    'Added and scored: {n} transactions in {ms} ms ({per} ms per transaction)': 'Добавлено и оценено транзакций: {n} за {ms} мс ({per} мс на транзакцию)',
    'Main reason': 'Главная причина', 'Open the queue': 'Открыть очередь',
    # CSV
    'Score your own data (CSV)': 'Оценить свои данные (CSV)',
    'The file is scored as a separate graph: features and both models work exactly as in the queue.':
        'Файл оценивается как отдельный граф: признаки и обе модели работают так же, как в очереди.',
    'Columns': 'Столбцы', 'Rail': 'Канал', 'Download a sample CSV': 'Скачать пример CSV', 'the last 2 days of the demo bank': 'последние 2 дня демо-банка',
    'Score': 'Оценка', 'Row': 'Строка', 'Reasons': 'Причины',
    '{f}: {n} transactions scored in {ms} ms ({per} ms per transaction)': '{f}: транзакций оценено: {n} за {ms} мс ({per} мс на транзакцию)',
    'The 50 riskiest are shown (row is the line number in the file; the header is line 1).':
        'Показаны 50 самых рискованных (строка — номер строки в файле; заголовок — строка 1).',
    'Could not read the CSV: a comma-separated file with a header row is expected.': 'Не удалось прочитать CSV: нужен файл с разделителем-запятой и строкой заголовка.',
    'Missing columns: {cols}. Required columns: {req}.': 'Не хватает столбцов: {cols}. Нужные столбцы: {req}.',
    'The file must have 10 to 50,000 rows (it has {n}).': 'В файле должно быть от 10 до 50 000 строк (сейчас {n}).',
    'invalid date/time (example: 2026-09-30 14:05:00)': 'неверная дата/время (пример: 2026-09-30 14:05:00)',
    'amount must be a positive number': 'сумма должна быть положительным числом',
    'rail must be one of: {r}': 'канал должен быть одним из: {r}',
    'bank and account columns must not be empty': 'столбцы банка и счёта не должны быть пустыми',
    '{msg} (rows: {rows})': '{msg} (строки: {rows})', 'The file is larger than 20 MB.': 'Файл больше 20 МБ.',
    # audit
    'Audit log': 'Журнал аудита', '✓ Chain intact': '✓ Цепочка цела', '✗ Chain broken from entry #{n}': '✗ Цепочка нарушена начиная с записи №{n}',
    '{n} entries.': 'Записей: {n}.',
    'Each entry stores the SHA-256 hash of the one before it, so editing any past entry breaks the chain.':
        'Каждая запись хранит SHA-256 хеш предыдущей, поэтому изменение любой прошлой записи разрывает цепочку.',
    'Time (UTC)': 'Время (UTC)', 'No decisions yet.': 'Решений пока нет.',
    # evidence
    'Evidence: how good are the models': 'Доказательства: насколько хороши модели',
    'No real metrics yet: they appear here once the models are trained on AMLworld.': 'Реальных метрик пока нет: они появятся после обучения моделей на AMLworld.',
    'Data: {src} — {n} transactions, {l} of them money laundering. Split by time: training {tr}, validation {va}, test {te} (the latest period: {a} — {b}). Every number below is measured on the test period, which the models never saw.':
        'Данные: {src} — {n} транзакций, из них {l} — отмывание денег. Разбиение по времени: обучение {tr}, валидация {va}, тест {te} (последний период: {a} — {b}). Все числа ниже измерены на тестовом периоде, который модели не видели.',
    'precision in the riskiest 0.1%: {c} of {a} alerts are real laundering': 'точность в 0,1% самых рискованных: {c} из {a} алертов — реальное отмывание',
    'precision at the threshold chosen on validation (recall {r})': 'точность на пороге, выбранном на валидации (полнота {r})',
    'of all laundering found by reviewing the riskiest 1%': 'всего отмывания найдено при проверке 1% самых рискованных',
    'PR-AUC of the ensemble (random: {r})': 'PR-AUC ансамбля (случайная модель: {r})', 'ROC-AUC of the ensemble': 'ROC-AUC ансамбля',
    'Ablation: LightGBM, GNN and ensemble (test period)': 'Абляция: LightGBM, GNN и ансамбль (тестовый период)',
    'Model': 'Модель', 'Precision': 'Точность', 'Recall': 'Полнота', 'Top {s}%: precision / recall': 'Топ {s}%: точность / полнота',
    'LightGBM (two stages)': 'LightGBM (две стадии)', 'GNN (GraphSAGE)': 'GNN (GraphSAGE)', 'Ensemble (both)': 'Ансамбль (обе)',
    'PR-AUC is the main metric for rare events (a random model scores the base rate, {r}). F1, precision and recall use the threshold chosen on the validation period. "Top X%": if officers review only the riskiest X% of transactions, how many of those are really laundering (precision) and what share of all laundering they find (recall).':
        'PR-AUC — главная метрика для редких событий (случайная модель получает базовую долю, {r}). F1, точность и полнота считаются на пороге, выбранном на валидационном периоде. «Топ X%»: если сотрудники проверяют только X% самых рискованных транзакций, какая их доля — реальное отмывание (точность) и какую долю всего отмывания они находят (полнота).',
    'Precision–recall curves': 'Кривые точность–полнота', 'Calibration (ensemble)': 'Калибровка (ансамбль)',
    'When the model says "5%", about 5% of such transactions should really be laundering: points close to the diagonal are good. Each point is one probability bin; both axes are logarithmic.':
        'Если модель говорит «5%», примерно 5% таких транзакций должны действительно быть отмыванием: точки у диагонали — хорошо. Каждая точка — один интервал вероятности; обе оси логарифмические.',
    'Most important LightGBM features': 'Самые важные признаки LightGBM',
    'External check: Elliptic (a real Bitcoin transaction graph)': 'Внешняя проверка: Elliptic (реальный граф транзакций Bitcoin)',
    'Elliptic (Weber et al., 2019): {n} Bitcoin transactions, {e} edges. Time steps 1–29 train, 30–34 validation, 35–49 test ({t} labelled transactions). A different domain (crypto, a transaction is a node), so this checks the approach itself, not the bank model.':
        'Elliptic (Weber и др., 2019): {n} транзакций Bitcoin, {e} связей. Шаги времени 1–29 — обучение, 30–34 — валидация, 35–49 — тест ({t} размеченных транзакций). Другая область (крипто, транзакция — узел), поэтому это проверка самого подхода, а не банковской модели.',
    'Illicit F1': 'F1 (незаконные)', 'Steps {k}: PR-AUC / F1': 'Шаги {k}: PR-AUC / F1',
    'A large dark market was shut down at time step 43. After that the illicit transactions look different from anything in the training steps, and every model fails on them; the original paper reports the same. Before the shutdown (steps 35–42) the ensemble finds most illicit transactions; after it, nothing trained on the past does. This is why a deployed model needs regular retraining and an officer in the loop.':
        'На шаге 43 был закрыт крупный даркнет-рынок. После этого незаконные транзакции не похожи ни на что из обучающих шагов, и все модели на них ошибаются; исходная статья сообщает то же самое. До закрытия (шаги 35–42) ансамбль находит большинство незаконных транзакций; после — ничто, обученное на прошлом, их не находит. Поэтому рабочей модели нужны регулярное переобучение и сотрудник в контуре.',
    'Check on the synthetic Uzbek demo bank': 'Проверка на синтетическом узбекском демо-банке',
    'Models trained on AMLworld, run on the Uzbek demo bank ({n} transactions, {p} of them in planted schemes, share {s}). This data is synthetic too and we wrote its generator ourselves, so this is not a headline metric: it only shows how the models carry over to other payment rails and another currency.':
        'Модели, обученные на AMLworld, на узбекском демо-банке ({n} транзакций, из них {p} — во внедрённых схемах, доля {s}). Эти данные тоже синтетические, и генератор мы написали сами, поэтому это не главная метрика: она лишь показывает, как модели переносятся на другие платёжные каналы и другую валюту.',
    'Top 2%: recall': 'Топ 2%: полнота',
    'Speed and cost (measured live on this server)': 'Скорость и стоимость (измерено вживую на этом сервере)',
    'per transaction, median (p50)': 'на транзакцию, медиана (p50)', 'CPU time, mean': 'время CPU, среднее',
    'per million transactions (CPU cost)': 'на миллион транзакций (стоимость CPU)',
    'Over {n} live requests: features + LightGBM + GNN. Cost: Oracle A1 list price (2 OCPU + 12 GB = $0.038/hour), assuming a fully loaded server. No external API fees.':
        'По {n} живым запросам: признаки + LightGBM + GNN. Стоимость: прайс Oracle A1 (2 OCPU + 12 ГБ = $0.038/час) при полной загрузке сервера. Платы за внешние API нет.',
    'No live requests yet: add transactions on the "Live input" or "CSV" page.': 'Живых запросов пока нет: добавьте транзакции на странице «Живой ввод» или «CSV».',
    'When the model is wrong (limitations)': 'Когда модель ошибается (ограничения)',
    'New, unseen schemes: the models learnt from the typologies in AMLworld. On the demo bank, fan-out (one account paying many) is usually flagged, often above 50%, though not for every account profile, while the "mule fan-in" scheme of our generator (collect from many accounts, then cash out) is scored weakly (a few percent). A pilot must therefore retrain on the bank\'s own data.':
        'Новые, невиданные схемы: модели обучены на типологиях AMLworld. В демо-банке fan-out (один счёт платит многим) обычно выявляется, часто выше 50%, но не для каждого профиля счёта, а схема «mule fan-in» нашего генератора (сбор со многих счетов и обналичивание) получает низкую оценку (несколько процентов). Поэтому в пилоте модель нужно переобучить на данных самого банка.',
    'The GNN hardly carried over from AMLworld to the Uzbek demo bank (table above): its share in the ensemble is small, and the stacking weight accounts for this automatically.':
        'GNN почти не перенёсся с AMLworld на узбекский демо-банк (таблица выше): его доля в ансамбле мала, и вес стекинга учитывает это автоматически.',
    'New accounts with no history (cold start): the window features are empty.': 'Новые счета без истории (холодный старт): оконные признаки пусты.',
    "The training data is synthetic (IBM AMLworld), and so is the demo bank. Real performance is measured in a pilot on the bank's own data.":
        'Обучающие данные синтетические (IBM AMLworld), как и демо-банк. Реальное качество измеряется в пилоте на данных самого банка.',
    'UZS is converted to USD at a fixed rate; Uzbek payment rails are mapped to the closest AMLworld formats (declared in the README).':
        'UZS переводится в USD по фиксированному курсу; узбекские платёжные каналы сопоставлены ближайшим форматам AMLworld (указано в README).',
    'Observed rate': 'Наблюдаемая доля', 'Ideal': 'Идеал', 'Predicted probability (log)': 'Предсказанная вероятность (лог)',
    'Observed rate (log)': 'Наблюдаемая доля (лог)', 'Ensemble': 'Ансамбль',
    'predicted {p}, observed {o}, {n} transactions': 'предсказано {p}, наблюдается {o}, транзакций: {n}',
    'Ensemble (stacking)': 'Ансамбль (стекинг)', 'LightGBM': 'LightGBM', 'GNN': 'GNN', 'GraphSAGE': 'GraphSAGE',
    # reasons (titles come from src/explain.py)
    'Receiver is collecting money from many new accounts': 'Получатель собирает деньги со многих новых счетов',
    'Sender is spreading money across many new accounts': 'Отправитель рассылает деньги на многие новые счета',
    'Money passes through the account quickly': 'Деньги быстро проходят через счёт',
    'Payment history between sender and receiver': 'История платежей между отправителем и получателем',
    'Age and activity of the accounts': 'Возраст и активность счетов',
    'Payment amount': 'Сумма платежа', 'Payment channel': 'Канал платежа', 'Time of payment': 'Время платежа',
    'Same-bank or own-account transfer': 'Перевод внутри банка или на свой счёт',
    "Risk of the accounts' earlier payments": 'Риск предыдущих платежей счетов',
    'Recent turnover of the accounts': 'Недавний оборот счетов',
    # demo data labels
    'Student': 'Студент', 'Salaried employee': 'Наёмный сотрудник', 'Pensioner': 'Пенсионер', 'Small business': 'Малый бизнес',
    'IT freelancer': 'IT-фрилансер',
    'Tashkent': 'Ташкент', 'Samarkand': 'Самарканд', 'Bukhara': 'Бухара', 'Andijan': 'Андижан', 'Fergana': 'Фергана',
    'Namangan': 'Наманган', 'Karshi': 'Карши', 'Nukus': 'Нукус',
    'Bank transfer': 'Банковский перевод', 'Cash': 'Наличные', 'Crypto': 'Крипто',
}


class Lang:
    def __init__(self, code):
        self.code = code if code in LANGS else 'en'

    def t(self, s):
        return RU.get(s, s) if self.code == 'ru' else s

    def num(self, v, d=0):
        s = f'{v:,.{d}f}'
        return s.replace(',', ' ').replace('.', ',') if self.code == 'ru' else s

    def pct(self, v, d=1):
        """A fraction as a percentage: 0.954 -> 95.4%."""
        return self.num(100 * v, d) + '%'

    def uzs(self, v):
        return self.num(v) + ' UZS'
