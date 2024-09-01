import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from datasoup.keys import get_langchain_key, get_open_ai_key
from datasoup.sql_generator.db.db_utils import execute_query, get_table_metadata

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_API_KEY"] = get_langchain_key()
os.environ["OPENAI_API_KEY"] = get_open_ai_key()

DEFAULT_SENTENCE_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DEFAULT_LLM_SERVICE_MODEL = ChatOpenAI(model="gpt-4")


@dataclass(frozen=True)
class SQLColumn:
    name: str
    table_name: str
    description: str
    embedding: np.ndarray


@dataclass
class SQLColumns:
    columns: list[SQLColumn]

    @classmethod
    def from_static_metadata_json(cls):
        tables = get_table_metadata()

        sql_columns: list[SQLColumn] = []
        for table_name, table in tables.items():
            for column in table["columns"]:
                full_col_description = f"column name: {column['name']}, description: {column['description']}, table name: {table_name}, table description: {table['description']}"
                col_obj = SQLColumn(
                    name=column["name"],
                    table_name=table_name,
                    description=full_col_description,
                    embedding=np.array(DEFAULT_SENTENCE_EMBEDDER.encode(full_col_description)),
                )
                sql_columns.append(col_obj)
        return cls(columns=sql_columns)


def _invoke_llm_model(system_prompt: str, human_prompt: str) -> str:
    parser = StrOutputParser()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = DEFAULT_LLM_SERVICE_MODEL.invoke(messages)
    return parser.invoke(response)


class SQLGeneratorAgent:
    DB_COLUMNS = SQLColumns.from_static_metadata_json()

    def _get_related_columns(
        self,
        human_prompt: str,
        num_columns: int = 10,
    ) -> list[SQLColumn]:
        embeded_prompt = DEFAULT_SENTENCE_EMBEDDER.encode(human_prompt)
        # compute cosine similarity
        related_columns = [
            np.dot(col.embedding, embeded_prompt) / (np.linalg.norm(embeded_prompt) * np.linalg.norm(col.embedding))
            for col in self.DB_COLUMNS.columns
        ]
        # get the most similar columns
        top_columns = [
            col for col, _ in sorted(zip(self.DB_COLUMNS.columns, related_columns), key=lambda x: x[1], reverse=True)[:num_columns]
        ]

        return top_columns

    def invoke_agent(self, human_prompt: str) -> str:
        relevant_columns = self._get_related_columns(human_prompt)
        table_to_column_map: dict[str, list[SQLColumn]] = defaultdict(list)
        for col in relevant_columns:
            table_to_column_map[col.table_name].append(col)

        system_prompt = """
        I am a SQL generator and create SQL for SQLite to answer questions.
        I only return SQL with no descriptions or explanations.

        Consider the following tables and columns for the question:\n
        """
        tables_dict = get_table_metadata()
        for table in table_to_column_map:
            system_prompt += f"Table: {tables_dict[table]}\n"
            column_names = [i.name for i in table_to_column_map[table]]
            system_prompt += f"Columns to consider: {', '.join(column_names)}\n"

        return _invoke_llm_model(system_prompt, human_prompt)


class SQLRevisionAgent:
    def invoke_agent(
        self,
        human_prompt: str,
        sql_query: str,
        error_message: str,
    ) -> str:
        system_prompt = f"""
        I am a SQL revision agent and I will attempt to revise the SQL query based on the human prompt.
        """

        human_prompt = f"""
        revise the following query:
        {sql_query}

        intended to answer the question:
        {human_prompt}
        
        based on the following error message:
        {error_message}
        """

        revised_sql = _invoke_llm_model(system_prompt, human_prompt)
        return revised_sql


class SQLRunnerAgent:
    def __init__(
        self,
        sql_generator_agent: SQLGeneratorAgent,
        sql_revision_agent: SQLRevisionAgent,
    ):
        self.sql_generator_agent = sql_generator_agent
        self.sql_revision_agent = sql_revision_agent

    def invoke_agent(
        self,
        human_prompt: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        initial_query = self.sql_generator_agent.invoke_agent(human_prompt)

        test_query = initial_query
        i = 0
        while i <= max_retries:
            try:
                result = execute_query(test_query)
                return pd.DataFrame(result)
            except Exception as e:
                error_message = str(e)
                print(f"Error: {error_message}")
                test_query = self.sql_revision_agent.invoke_agent(human_prompt, test_query, error_message)
                i += 1
        raise Exception("Failed to execute SQL query")


if __name__ == "__main__":
    _sql_runner_agent = SQLRunnerAgent(
        sql_generator_agent=SQLGeneratorAgent(),
        sql_revision_agent=SQLRevisionAgent(),
    )
    _result = _sql_runner_agent.invoke_agent("get sales by store")
    print(_result)
