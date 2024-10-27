import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Solo envía logs a stdout
    ]
)

# Configurar el logger específico para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ... resto del código ...
