import json
import os

KEY_FILE = "keys.json"


def _get_key_dict():
    with open(KEY_FILE) as f:
        return json.load(f)


def get_open_ai_key():
    return _get_key_dict()["OPENAI_API_KEY"]


def get_langchain_key():
    return _get_key_dict()["LANGCHAIN_API_KEY"]


if __name__ == "__main__":
    print(_get_key_dict())
