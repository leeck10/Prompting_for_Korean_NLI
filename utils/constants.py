from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('Prompt_NLI/utils/config.yml')
OPENAI_API_KEY = ''
#OPENAI_API_KEY = ''

try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')


NLI_ANSWERS = ["함의", "중립", "모순"]

