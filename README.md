# Інтелектуальне збагачення Excel-даних (агентна система з веб-пошуком, LangGraph)

Система приймає Excel-файл і текстовий опис задачі, аналізує структуру даних, **формує
пошуковий запит, шукає дані в інтернеті, витягує потрібне число з результатів** і дописує
його в колонку — зберігаючи всі оригінальні дані. Рішення універсальне (один код для різних
наборів), стійке до помилок і масштабоване (паралельна обробка рядків).

## Архітектура

Агентний граф станів **LangGraph**:

```
read_file → analyze → enrich → save
```

- **read_file** — читає xlsx у `pandas.DataFrame`;
- **analyze** — LLM визначає цільову колонку, одиниці виміру й **шаблон пошукового запиту**
  з плейсхолдерами-колонками (універсальний інтелектуальний пошук);
- **enrich** — агент-дослідник: для кожного рядка **формує запит → шукає в інтернеті
  (`web_search`) → витягує число з результатів** (LLM-екстрактор бере пряму/air-відстань, не
  дорожню; ігнорує ненадійні джерела). Паралельно (`ThreadPoolExecutor`), з кешем запитів;
- **save** — зберігає `*_enriched.xlsx`, не змінюючи оригінальні колонки.

**Пошук** (`web_search`): **Tavily** (якщо заданий `TAVILY_API_KEY`), інакше — **DuckDuckGo**
без ключа. Ретраї з бекофом, кеш однакових запитів, переформулювання запиту за невдачі.

Універсальний інтерфейс:

```python
process_excel(
    file_path="capitals.xlsx",
    task_description="знайди пряму відстань між столицями в км для колонки distance",
    sign="Oleksandr_Vasyleiko",
)
```

## Дані

Повні вхідні набори (по 10 рядків) завантажуються за посиланнями (Google Sheets) у секції
конфігурації; цільова колонка обнуляється — її заповнює система через пошук.

| Файл | Опис |
|------|------|
| `dz_topic_final_Oleksandr_Vasyleiko.ipynb` | Готовий ноутбук з виводами та графіком |
| `dz_topic_final_Oleksandr_Vasyleiko.py` | Джерело з клітинками (`# %%`) |
| `capitals_enriched_Oleksandr_Vasyleiko.xlsx` | Результат 1 — відстані між столицями (10) |
| `mountains_enriched_Oleksandr_Vasyleiko.xlsx` | Результат 2 — висоти гір (10) |

## Запуск

```bash
conda create -n env_mlf -c conda-forge python=3.11 spyder spyder-notebook pandas seaborn=0.13 ipywidgets -y
conda activate env_mlf
pip install openai langgraph langchain openpyxl ddgs
export OPENROUTER_API_KEY="<ваш-ключ>"
export TAVILY_API_KEY="<ваш-tavily-ключ>"   # необов'язково; без нього — DuckDuckGo
jupyter nbconvert --to notebook --execute --inplace dz_topic_final_Oleksandr_Vasyleiko.ipynb
```

Ключі читаються **лише** зі змінних середовища — у коді їх немає.

## Точність

Значення беруться з реальних результатів пошуку; усі 20 збагачених значень — у межах **±1%**
від довідкових (відстані — пряма/great-circle; висоти гір — з Wikipedia та довідкових джерел).
