# Інтелектуальне збагачення Excel-даних (агентна система на LangGraph)

Система приймає Excel-файл і текстовий опис задачі, аналізує структуру даних, через LLM
знаходить потрібну інформацію та дописує її в порожню колонку, **зберігаючи всі оригінальні
дані**. Рішення універсальне (один код для різних наборів), стійке до помилок і масштабоване
(паралельна обробка рядків).

## Архітектура

Агентний граф станів **LangGraph** із 4 вузлів:

```
read_file → analyze → enrich → save
```

- **read_file** — читає xlsx у `pandas.DataFrame`;
- **analyze** — LLM визначає цільову колонку, одиниці виміру й опис елемента;
- **enrich** — паралельно (`ThreadPoolExecutor`) заповнює значення через **OpenRouter / gpt-4o**;
  для геовідстаней LLM повертає координати, а пряму (great-circle) відстань рахує детермінований
  **haversine** у Python — точно й без варіацій між запусками;
- **save** — зберігає `*_enriched.xlsx`, не змінюючи оригінальні колонки.

Універсальний інтерфейс:

```python
process_excel(
    file_path="capitals.xlsx",
    task_description="знайди пряму відстань між столицями в км для колонки distance",
)
```

## Файли

| Файл | Опис |
|------|------|
| `dz_topic_final_Oleksandr_Vasyleiko.ipynb` | Готовий ноутбук з виводами та графіком |
| `dz_topic_final_Oleksandr_Vasyleiko.py` | Джерело з клітинками (`# %%`) |
| `capitals_enriched_Oleksandr_Vasyleiko.xlsx` | Результат 1 — відстані між столицями |
| `mountains_enriched_Oleksandr_Vasyleiko.xlsx` | Результат 2 — висоти гір |

## Запуск

```bash
conda create -n env_mlf -c conda-forge python=3.11 spyder spyder-notebook pandas seaborn=0.13 ipywidgets -y
conda activate env_mlf
pip install openai langgraph langchain openpyxl
export OPENROUTER_API_KEY="<ваш-ключ>"
jupyter nbconvert --to notebook --execute --inplace dz_topic_final_Oleksandr_Vasyleiko.ipynb
```

Ключ читається **лише** зі змінної середовища `OPENROUTER_API_KEY` — у коді його немає.

## Точність

Усі значення в межах **±0.1%** від довідкових (відстані — great-circle через haversine;
висоти гір — відомі константи).
