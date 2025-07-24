# tests/conftest.py

import sys
import asyncio
import matplotlib

def pytest_configure(config):
    matplotlib.use('Agg')
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
