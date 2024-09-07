# Vanilla Machine Learning API

This is a vanilla implementation using `FastAPI` to make inference using a trained model on the famous `iris dataset`.

### Setup

First of all we need to create the virtual environment and install the requirements

```bash
python -m venv venv # to create the env
source venv/bin/activate # to activate the env
pip install -r requirements.txt # to install the requirements
```

### Init the API

To initialize the API we have two options 

1. Write on the terminal the following code: (This option was implemented)

    ```
    uvicorn fast_api:app --host 0.0.0.0 --port 8000
    ```

2. Add the folllowing code at the end of your API script 

    ```python
    if __name__ == '__main__':
        uvicorn.run('fastapi_app:app', port = 8000)
    ```


### How to use the API?

We have two options for testing.

1. Using the FastAPI docs, just go the following route: http://127.0.0.1:8000/docs and then select the post method predict and try it out.
2. Using the CLI:
    ```bash
    curl -X \
    'POST' \
    'http://localhost:8000/predictsepal_length=1&sepal_width=2&petal_length=3&petal_width=0' \
    -H 'accept: application/json' \
    -d ''
    ```

Just replace the attributes and you will get the inferece.