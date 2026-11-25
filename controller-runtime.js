// Portable, deterministic landmark features and random forest inference.
export function frameFeatures(pose,hands,mode){
 const out=[]; function append(points,n){for(let i=0;i<n;i++){const p=points?.[i];out.push(p?.x??0,p?.y??0,p?.z??0,p?1:0)}}
 if(mode!=='hands')append(pose?.landmarks?.[0],33);
 if(mode!=='pose'){for(const side of ['left','right']){const i=(hands?.handednesses||[]).findIndex(x=>x?.[0]?.categoryName?.toLowerCase()===side);append(hands?.landmarks?.[i],21)}}
 return out;
}
export function windowFeatures(buffer,t,cfg){
 const n=cfg.frames,step=1000/cfg.sampleHz, first=t-(n-1)*step;
 if(!buffer.length||buffer[0].t>first+step/2)return null;
 const out=[];let j=0;
 for(let i=0;i<n;i++){const target=first+i*step;while(j+1<buffer.length&&Math.abs(buffer[j+1].t-target)<Math.abs(buffer[j].t-target))j++;if(Math.abs(buffer[j].t-target)>Math.max(100,step*1.5))return null;out.push(...buffer[j].x)}
 return out;
}
export function trainForest(rows,classes,seed=42){
 let state=seed;const rand=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
 const d=rows[0].x.length,k=classes.length;
 const counts=rs=>{const c=Array(k).fill(0);for(const r of rs)c[r.y]++;return c};
 const impurity=(c,n)=>1-c.reduce((s,v)=>s+(v/n)**2,0);
 function tree(rs,depth){const c=counts(rs),n=rs.length;if(depth>=9||n<5||c.includes(n))return {p:c.map(v=>v/n)};
 let best=null,score=impurity(c,n);for(let q=0;q<Math.min(48,Math.ceil(Math.sqrt(d)*2));q++){const f=Math.floor(rand()*d),v=(rs[Math.floor(rand()*n)].x[f]+rs[Math.floor(rand()*n)].x[f])/2;const l=[],r=[];for(const x of rs)(x.x[f]<=v?l:r).push(x);if(!l.length||!r.length)continue;const s=(l.length*impurity(counts(l),l.length)+r.length*impurity(counts(r),r.length))/n;if(s<score-1e-9){score=s;best={f,v,l,r}}}
 return best?{f:best.f,v:best.v,l:tree(best.l,depth+1),r:tree(best.r,depth+1)}:{p:c.map(v=>v/n)};
 }
 return Array.from({length:31},()=>tree(Array.from({length:rows.length},()=>rows[Math.floor(rand()*rows.length)]),0));
}
export function predict(model,x){const p=Array(model.classes.length).fill(0);for(let node of model.trees){while(!node.p)node=x[node.f]<=node.v?node.l:node.r;node.p.forEach((v,i)=>p[i]+=v/model.trees.length)}const i=p.indexOf(Math.max(...p));return {label:model.classes[i],confidence:p[i],probabilities:p}}
export function validateModel(m){if(m?.format!=='modellab-controller-v1'||m.algorithm!=='random_forest'||!['pose','hands','both'].includes(m.input?.mode)||!Number.isInteger(m.input.frames)||m.input.frames<1||m.input.frames>60||m.input.sampleHz!==30||!Array.isArray(m.classes)||m.classes.length<2||m.classes.length>20||new Set(m.classes).size!==m.classes.length||m.classes.some(x=>typeof x!=='string'||!x.trim())||!Array.isArray(m.trees)||!m.trees.length||m.trees.length>100)throw Error('Unsupported model format');const d=({pose:132,hands:168,both:300})[m.input.mode]*m.input.frames;let nodes=0;function check(n,depth){if(!n||depth>12||++nodes>100000)throw Error('Invalid tree');if(n.p){if(!Array.isArray(n.p)||n.p.length!==m.classes.length||n.p.some(v=>!Number.isFinite(v)||v<0||v>1)||Math.abs(n.p.reduce((a,b)=>a+b,0)-1)>.001)throw Error('Invalid leaf');return}if(!Number.isInteger(n.f)||n.f<0||n.f>=d||!Number.isFinite(n.v))throw Error('Invalid split');check(n.l,depth+1);check(n.r,depth+1)}m.trees.forEach(n=>check(n,0));return m}
