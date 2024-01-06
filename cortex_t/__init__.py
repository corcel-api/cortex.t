import os

if os.environ.get('RICH_TRACEBACK', '') == '0':
    import rich.traceback
    rich.traceback.install = lambda *a, **kw: None
