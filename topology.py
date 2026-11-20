"""Read actual Linux SMT sibling groups, restricted to allowed online CPUs."""
import os
from pathlib import Path

def expand(s):
    result=[]
    for part in s.strip().split(','):
        if part:
            r=part.split('-');result.extend(range(int(r[0]),int(r[-1])+1))
    return result

def discover(node,root=Path('/sys/devices/system'),allowed=None):
    allowed=set(os.sched_getaffinity(0) if allowed is None else allowed)
    online=set(expand((root/'cpu/online').read_text()))
    cpus=allowed&online&set(expand((root/f'node/node{node}/cpulist').read_text()));groups={}
    for c in sorted(cpus):
        sib=tuple(expand((root/f'cpu/cpu{c}/topology/thread_siblings_list').read_text()))
        groups.setdefault(sib,[]).append(c)
    return sorted(groups.values(),key=lambda v:v[0])
