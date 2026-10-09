import os
import subprocess
import sys
from util import BASE,save


if __name__=='__main__':
    os.environ['PYTHONWARNINGS']='error::RuntimeWarning'
    os.environ['MPLCONFIGDIR']=str(BASE/'matplotlib_cache')
    try:
        for script in ['test_integrity.py','prepare_data.py','run.py','audit.py','evaluate_test.py']:
            save(BASE/'controller_status.json',{'state':'running','script':script})
            subprocess.run([sys.executable,'-u',str(BASE/script)],cwd=BASE.parent,check=True)
        save(BASE/'controller_status.json',{'state':'complete','test_evaluated':True})
    except Exception as error:
        save(BASE/'controller_status.json',{'state':'failed','error':str(error)})
        raise
