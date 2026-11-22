// Portable motion feature extractor and random forest; no dependencies.
export function features(frames, ids, duration, space='relative') {
  if(frames.length<5) return null;
  const end=frames.at(-1).t, start=end-duration;
  if(frames[0].t>start+150 || frames.at(-1).t-frames[0].t<duration*.8) return null;
  const out=[];
  for(const id of ids){
    const p=frames.filter(f=>f.t>=start-150).map(f=>({t:f.t,v:(f.pts?.[id]??f.points?.[id])?.[space]?.raw})).filter(p=>Array.isArray(p.v)&&p.v.length===3&&p.v.every(Number.isFinite));
    if(p.length<5)return null;
    const samples=[];let j=0;
    for(let k=0;k<12;k++){
      const t=start+k*duration/11;while(j+1<p.length&&p[j+1].t<t)j++;
      const a=p[j],b=p[Math.min(j+1,p.length-1)];
      if(t<a.t-150||t>b.t+150||b.t-a.t>250)return null;
      const q=Math.max(0,Math.min(1,(t-a.t)/Math.max(1,b.t-a.t)));
      samples.push(a.v.map((v,i)=>v+(b.v[i]-v)*q));
    }
    for(let i=0;i<3;i++){
      const a=samples.map(v=>v[i]), mean=a.reduce((s,v)=>s+v,0)/a.length;
      out.push(a[0],mean,Math.min(...a),Math.max(...a),a.at(-1)-a[0]);
      for(let k=1;k<a.length;k++)out.push(a[k]-a[0]);
    }
  }return out;
}
function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}}
export function trainForest(rows, labels, seed=42, count=45){
  const rand=rng(seed), d=rows[0].x.length, C=labels.length;
  const histogram=rs=>{const h=Array(C).fill(0);for(const r of rs)h[r.y]++;return h};
  const impurity=h=>{const n=h.reduce((s,v)=>s+v,0);return n?1-h.reduce((s,v)=>s+(v/n)**2,0):0};
  function tree(rs,depth){const h=histogram(rs);const leaf={p:h.map(v=>v/rs.length)};if(depth>=9||rs.length<4||impurity(h)===0)return leaf;
    let best=null;const fs=new Set();while(fs.size<Math.min(d,Math.max(4,Math.ceil(Math.sqrt(d)))))fs.add(Math.floor(rand()*d));
    for(const f of fs){const vals=rs.map(r=>r.x[f]).sort((a,b)=>a-b);for(let k=1;k<=5;k++){const i=Math.min(vals.length-1,Math.floor(k*vals.length/6)),v=(vals[i-1]+vals[i])/2;const a=rs.filter(r=>r.x[f]<=v),b=rs.filter(r=>r.x[f]>v);if(!a.length||!b.length)continue;const loss=(a.length*impurity(histogram(a))+b.length*impurity(histogram(b)))/rs.length;if(!best||loss<best.loss)best={f,v,a,b,loss}}}
    if(!best||best.loss>=impurity(h)-1e-9)return leaf;return {f:best.f,v:best.v,l:tree(best.a,depth+1),r:tree(best.b,depth+1)};
  }
  const groups=labels.map((_,i)=>rows.filter(r=>r.y===i));if(groups.some(g=>!g.length))throw Error('Each class needs training examples');
  const trees=Array.from({length:count},()=>tree(groups.flatMap(g=>Array.from({length:Math.max(...groups.map(a=>a.length))},()=>g[Math.floor(rand()*g.length)])),0));return {labels,trees};
}
export function predict(model,x){if(!x)return null;const p=Array(model.labels.length).fill(0);for(let n of model.trees){while(!n.p)n=x[n.f]<=n.v?n.l:n.r;n.p.forEach((v,i)=>p[i]+=v/model.trees.length)}const i=p.indexOf(Math.max(...p));return {label:model.labels[i],index:i,score:p[i],scores:p}}
