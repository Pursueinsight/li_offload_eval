"""Record reproducibility metadata without collecting credentials or unrelated environment variables."""
import argparse,os,platform,subprocess
from pathlib import Path
from caseio import save_json

def capture():
    info=dict(platform=platform.platform(),python=platform.python_version(),allowed_cpus=sorted(os.sched_getaffinity(0)))
    for key,cmd in [('lscpu',['lscpu','-J']),('numa',['numactl','--hardware']),('compiler',['g++','--version'])]:
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=10);info[key]=r.stdout or r.stderr
        except (OSError,subprocess.TimeoutExpired) as e:info[key]=str(e)
    info['cann_paths']={k:os.environ.get(k) for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','CAMODEL_CONFIG_PATH')}
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();save_json(a.out,capture())
if __name__=='__main__':main()
