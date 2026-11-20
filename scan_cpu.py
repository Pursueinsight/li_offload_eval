"""One process per total physical-core count; supports one or multiple NUMA nodes."""
import argparse,csv,json,os,shlex,subprocess,sys
from pathlib import Path
from topology import discover
from caseio import save_json

def parse_ids(value,minimum):
    fields=value.split(',')
    if not fields or any(not s.strip() for s in fields):
        raise ValueError('Use a nonempty comma-separated integer list')
    values=[int(s) for s in fields]
    if min(values)<minimum or len(set(values))!=len(values):
        raise ValueError(f'Values must be unique integers >= {minimum}')
    return values

def core_order(nodes,by_node,policy):
    if policy=='compact':
        return [(node,g) for node in nodes for g in by_node[node]]
    return [(node,by_node[node][i])
            for i in range(max(map(len,by_node.values())))
            for node in nodes if i<len(by_node[node])]

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--case',required=True);p.add_argument('--out',required=True)
    p.add_argument('--node',default='0',help='NUMA nodes, e.g. 0 or 0,1; one process uses these nodes')
    p.add_argument('--cores',default='1,2,4,8,16,24,32,48',help='Total physical-core counts, not counts per node')
    p.add_argument('--smt',action='store_true')
    p.add_argument('--core-policy',choices=['balanced','compact'],default='balanced',
                   help='balanced alternates nodes; compact fills nodes in the supplied order')
    p.add_argument('--memory-policy',choices=['auto','bind','interleave'],default='auto',
                   help='auto binds on one active node, interleaves across multiple active nodes')
    p.add_argument('--wait-policy',choices=['PASSIVE','ACTIVE'],default='PASSIVE')
    p.add_argument('--topk-policy',choices=['auto','independent'],default='auto',
                   help='auto preserves native group selection; independent sets groups=threads at every point')
    p.add_argument('--groups',type=int,default=None,
                   help='Explicit native TopK groups in auto policy; 0 uses native automatic selection')
    p.add_argument('--order',choices=['sorted','unordered'],default='unordered',
                   help='Explicitly forward output ordering, independent of cpu_bench.py defaults')
    p.add_argument('--dry-run',action='store_true')
    a,extra=p.parse_known_args()
    try:
        nodes=parse_ids(a.node,0);counts=parse_ids(a.cores,1)
        by_node={node:discover(node) for node in nodes}
    except (ValueError,OSError) as e:p.error(str(e))
    empty=[node for node in nodes if not by_node[node]]
    if empty:p.error(f'No allowed online physical cores on NUMA nodes {empty}')
    ordered=core_order(nodes,by_node,a.core_policy)
    all_cpus=[cpu for _,g in ordered for cpu in g]
    if len(set(all_cpus))!=len(all_cpus):p.error('Topology contains overlapping NUMA CPU groups')
    if max(counts)>len(ordered):p.error(f'Total physical-core counts must be <= {len(ordered)}')
    if a.smt and any(len(g)!=2 for _,g in ordered[:max(counts)]):
        p.error('SMT2 requires exactly two allowed online siblings on every selected core')
    if any(x.split('=')[0] in ('--threads','--require-pinning') for x in extra):
        p.error('Threads and pinning are managed by scanner')
    if a.topk_policy=='independent' and a.groups is not None:
        p.error('--topk-policy independent manages --groups; do not pass both')
    min_threads=min(counts)*(2 if a.smt else 1)
    if a.groups is not None and not 0<=a.groups<=min_threads:
        p.error(f'--groups must be in 0..{min_threads} for every point in this scan')
    root=Path(a.out).resolve();rows=[];baseline=None;identity=None
    if not a.dry_run and root.exists() and any(root.iterdir()):
        p.error('Choose a new or empty scan output directory')
    for n in counts:
        selected=ordered[:n]
        active=[node for node in nodes if any(owner==node for owner,_ in selected)]
        cpus=[cpu for _,g in selected for cpu in (g if a.smt else g[:1])]
        requested_groups=len(cpus) if a.topk_policy=='independent' else (a.groups or 0)
        per_node={str(node):sum(owner==node for owner,_ in selected) for node in nodes}
        memory_policy=a.memory_policy
        if memory_policy=='auto':memory_policy='bind' if len(active)==1 else 'interleave'
        node_list=','.join(map(str,active))
        memory_arg=('--membind=' if memory_policy=='bind' else '--interleave=')+node_list
        dest=root/f'cores_{n}'
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS=str(len(cpus)),OMP_THREAD_LIMIT=str(len(cpus)),
                   OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',
                   OMP_PLACES=','.join('{'+str(cpu)+'}' for cpu in cpus),
                   OMP_PROC_BIND='TRUE',OMP_WAIT_POLICY=a.wait_policy,
                   OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
        env.pop('GOMP_CPU_AFFINITY',None);env.pop('KMP_AFFINITY',None)
        cmd=['numactl',f'--physcpubind={",".join(map(str,cpus))}',memory_arg,
             sys.executable,str(Path(__file__).with_name('cpu_bench.py')),*extra,
             '--groups',str(requested_groups),'--order',a.order,
             '--case',str(Path(a.case).resolve()),'--threads',str(len(cpus)),
             '--require-pinning','--out',str(dest)]
        print(f'CORES_PER_NODE={per_node} MEMORY_POLICY={memory_policy} MEMORY_NODES={node_list}',flush=True)
        print(f'TOPK_POLICY={a.topk_policy} GROUPS_REQUESTED={requested_groups} ORDER={a.order}',flush=True)
        print('OMP_PLACES='+env['OMP_PLACES']+' '+shlex.join(cmd),flush=True)
        if a.dry_run:continue
        subprocess.run(cmd,env=env,check=True)
        r=json.loads((dest/'result.json').read_text())
        key=(r['case_id'],r['native_library_sha256'])
        if identity is not None and identity!=key:raise RuntimeError('Case/library changed across points')
        identity=key
        if set(r['worker_cpus'])!=set(cpus) or r['actual_threads']!=len(cpus):
            raise RuntimeError('Unexpected thread placement')
        if a.topk_policy=='independent' and r.get('topk_groups')!=len(cpus):
            raise RuntimeError('Native TopK groups do not match independent policy; check CPU ABI/source')
        t=r['native_median_us'];baseline=baseline or (n,t)
        rows.append(dict(physical_cores=n,threads=len(cpus),native_us=t,
            host_us=r['host_median_us'],
            native_p90_us=r.get('native_p90_us'),
            prepare_us=r['stage_median_us'].get('prepare'),
            score_us=r['stage_median_us']['score'],
            topk_us=r['stage_median_us']['topk'],speedup=baseline[1]/t,
            efficiency=(baseline[1]/t)/(n/baseline[0]),
            active_nodes=','.join(map(str,active)),
            physical_cores_per_node=json.dumps(per_node,sort_keys=True),
            memory_policy=memory_policy,memory_nodes=node_list,
            topk_policy=a.topk_policy,topk_groups_requested=requested_groups,
            topk_groups_actual=r.get('topk_groups'),order=a.order,
            worker_cpus=','.join(map(str,r['worker_cpus'])),result=str(dest/'result.json')))
        root.mkdir(parents=True,exist_ok=True)
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        save_json(root/'scan.json',dict(schema_version=2,case_id=r['case_id'],
            numa_node=nodes[0] if len(nodes)==1 else None,numa_nodes=nodes,
            core_policy=a.core_policy,memory_policy_requested=a.memory_policy,
            smt=a.smt,baseline_cores=baseline[0],rows=rows,wait_policy=a.wait_policy,
            topk_policy=a.topk_policy,groups_requested=a.groups,order=a.order,
            memory_note='Policy requested via numactl; actual page placement is not measured. Interleave is shared-memory placement, not per-node Key sharding.'))

if __name__=='__main__':main()
