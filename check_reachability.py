"""Exact BFS of budget-feasible unit-transfer graph for the two-rank controls."""
from collections import deque
from math import prod

def budget(shape,r):return sum(n*x for n,x in zip(shape,r))+prod(r)
def path(shape,init,target):
 B=budget(shape,init);todo=deque([init]);parent={init:None}
 while todo:
  cur=todo.popleft()
  if cur==target:break
  for i in range(len(shape)):
   if cur[i]==1:continue
   for j in range(len(shape)):
    if j==i or cur[j]>=shape[j]:continue
    nxt=list(cur);nxt[i]-=1;nxt[j]+=1;nxt=tuple(nxt)
    if nxt not in parent and budget(shape,nxt)<=B:parent[nxt]=cur;todo.append(nxt)
 if target not in parent:return None
 out=[];x=target
 while x is not None:out.append((x,budget(shape,x)));x=parent[x]
 return out[::-1]
if __name__=='__main__':
 for dims in [(15,9,10,9),(12,11,10,9)]:
  ans=path(dims,(4,2,2,2),(2,4,2,2))
  print(dims,ans)
