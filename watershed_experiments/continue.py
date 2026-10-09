import json
import subprocess
import sys
import os
from pathlib import Path
from common import BASE,save

if __name__=='__main__':
    os.environ['PYTHONWARNINGS']='error::RuntimeWarning'
    os.environ['MPLCONFIGDIR']=str(BASE/'matplotlib_cache')
    try:
        for script in ['test_integrity.py','prepare.py','run_pilot.py','audit.py']:
            save(BASE/'status.json',{'state':'running','script':script})
            subprocess.run([sys.executable,'-u',str(BASE/script)],cwd=BASE.parent,check=True)
        save(BASE/'status.json',{'state':'complete','test_evaluated':False})
    except Exception as error:
        save(BASE/'status.json',{'state':'failed','error':str(error)})
        raise
