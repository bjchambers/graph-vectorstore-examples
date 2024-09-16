from dotenv import find_dotenv, load_dotenv
import os

def verify_environment():
    """Verify the necessary environment variables are set.
    """
    assert 'OPENAI_API_KEY' in os.environ
    assert 'ASTRA_DB_DATABASE_ID' in os.environ
    assert 'ASTRA_DB_APPLICATION_TOKEN' in os.environ

def initialize_from_colab_userdata():
    """Try to initialize environment from colab `userdata`.
    """
    from google.colab import userdata
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
    os.environ['ASTRA_DB_DATABASE_ID'] = userdata.get('ASTRA_DB_DATABASE_ID')
    os.environ['ASTRA_DB_APPLICATION_TOKEN'] = userdata.get('ASTRA_DB_APPLICATION_TOKEN')

    try:
        os.environ['ASTRA_DB_KEYSPACE'] = userdata.get('ASTRA_DB_KEYSPACE')
    except userdata.SecretNotFoundError as e:
        # User doesn't have a keyspace set, so use the default.
        os.environ.pop('ASTRA_DB_KEYSPACE', None)

    try:
        os.environ['LANGCHAIN_API_KEY'] = userdata.get('LANGCHAIN_API_KEY')
        os.environ['LANGCHAIN_TRACING_V2'] = 'True'
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError) as e:
        print(f"Colab Secret not set / accessible. Not configuring tracing")
        os.environ.pop('LANGCHAIN_API_KEY')
        os.environ.pop('LANGCHAIN_TRACING_V2')

def initialize_from_prompts():
    import getpass

    os.environ['OPENAI_API_KEY'] = getpass('OPENAI_API_KEY')
    os.environ['ASTRA_DB_DATABASE_ID'] = input('ASTRA_DB_DATABASE_ID')
    os.environ['ASTRA_DB_APPLICATION_TOKEN'] = getpass('ASTRA_DB_APPLICATION_TOKEN')

    if (keyspace := input('ASTRA_DB_KEYSPACE (empty for default)')) is not None:
        os.environ['ASTRA_DB_KEYSPACE'] = keyspace
    else:
        os.environ.pop('ASTRA_DB_KEYSPACE', None)

    if (lc_api_key := getpass('LANGCHAIN_API_KEY (empty for no tracing)')) is not None:
        os.environ['LANGCHAIN_API_KEY'] = lc_api_key
        os.environ['LANGCHAIN_TRACING_V2'] = 'True'
    else:
        os.environ.pop('LANGCHAIN_API_KEY')
        os.environ.pop('LANGCHAIN_TRACING_V2')

def initialize_environment():
    """Initialize the environment variables.

    This uses the following:
    1. If a `.env` file is found, load environment variables from that.
    2. If not, and running in colab, set necessary environment variables from secrets.
    3. If necessary variables aren't set by the above, then prompts the user.
    """

    # 1. If a `.env` file is found, load environment variables from that.
    if (dotenv_path := find_dotenv()) is not None:
        load_dotenv(dotenv_path)
        verify_environment()
        return

    # 2. If not, and running in colab, set necesary environment variables from secrets.
    try:
        initialize_from_colab_userdata()
        verify_environment()
        return
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Initialize from prompts.
    initialize_from_prompts()
    verify_environment()
