#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Convert LFR .dat to repo layout.")
    p.add_argument('--in', dest='inp', type=Path, required=True, help='Input dir or prefix for *_network.dat and *_communities.dat')
    p.add_argument('--key', type=str, required=True, help='Dataset key (e.g., LFR_low)')
    p.add_argument('--out', type=Path, required=True, help='Output base directory (e.g., datasets/)')
    return p.parse_args()

def locate(inp: Path, key: str):
    if inp.is_dir():
        net = inp / f"{key}_network.dat"
        com = inp / f"{key}_communities.dat"
    else:
        net = inp.parent / f"{inp.name}_network.dat"
        com = inp.parent / f"{inp.name}_communities.dat"
    if not (net.exists() and com.exists()):
        raise FileNotFoundError("Could not find dat files")
    return net, com

def read_edges(path: Path):
    S=set()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            s=line.strip()
            if not s: continue
            a,b=s.split(); u=int(a)-1; v=int(b)-1
            if u==v: continue
            if u>v: u,v=v,u
            S.add((u,v))
    return S

def read_comms(path: Path):
    node2={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            s=line.strip()
            if not s: continue
            a,b=s.split(); u=int(a)-1; c=int(b)-1
            node2.setdefault(u,[]).append(c)
    for u,cs in node2.items(): node2[u]=sorted(set(cs))
    return node2

def invert(node2):
    maxc = max((c for cs in node2.values() for c in cs), default=-1)
    K = maxc+1
    comms=[[] for _ in range(K)]
    for u,cs in node2.items():
        for c in cs: comms[c].append(u)
    return [sorted(set(cl)) for cl in comms if cl]

def write_layout(out_dir: Path, key: str, edges, comms, node2):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/f"{key}_network.txt",'w',encoding='utf-8') as fo:
        for u,v in sorted(edges): fo.write(f"{u} {v}
")
    with open(out_dir/f"{key}_communities.txt",'w',encoding='utf-8') as fo:
        for cl in comms: fo.write("\t".join(map(str,cl))+"\n")
    with open(out_dir/f"{key}_node_to_communities.csv",'w',encoding='utf-8') as fo:
        # variable-length rows: node_id,comm1,comm2,...
        for u in sorted(node2): fo.write(str(u)+(","+(",".join(map(str,node2[u])) if node2[u] else ""))+"\n")

def main():
    args=parse_args()
    net,com=locate(args.inp,args.key)
    edges=read_edges(net); node2=read_comms(com); comms=invert(node2)
    out_dir=args.out/args.key; write_layout(out_dir,args.key,edges,comms,node2)
    print(f"[✓] Wrote: {out_dir}/{args.key}_network.txt, {out_dir}/{args.key}_communities.txt, {out_dir}/{args.key}_node_to_communities.csv")

if __name__=='__main__': main()
