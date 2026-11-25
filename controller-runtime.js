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
export function trainClassifier(rows,classes,algorithm,options={}){
 if(algorithm==='svm_linear'||algorithm==='svm_rbf')return trainSVM(rows,classes,algorithm,options);
 if(algorithm==='random_forest')return {trees:trainForest(rows,classes)};
 if(algorithm==='knn')return {neighbors:rows.map(r=>({x:r.x,y:r.y})),k:Math.min(5,rows.length)};
 if(algorithm==='centroid'){const d=rows[0].x.length;return {centroids:classes.map((_,i)=>{const group=rows.filter(r=>r.y===i),v=Array(d).fill(0);for(const r of group)for(let j=0;j<d;j++)v[j]+=r.x[j]/group.length;return v})}}
 throw Error('Unknown algorithm');
}
function distance(a,b){let d=0;for(let i=0;i<a.length;i++)d+=(a[i]-b[i])**2;return d/a.length}
export function predict(model,x){
 if(model.algorithm==='svm_linear'||model.algorithm==='svm_rbf'){const z=x.map((v,i)=>(v-model.scaler.mean[i])*model.scaler.inv[i]);const margins=model.svm.map(m=>m.bias+(model.algorithm==='svm_linear'?dot(m.weights,z):m.support.reduce((s,r)=>s+r.coef*rbf(r.x,z,model.gamma),0)));const max=Math.max(...margins),p=margins.map(v=>Math.exp(v-max)),sum=p.reduce((a,b)=>a+b,0),i=margins.indexOf(max);return {label:model.classes[i],confidence:p[i]/sum,probabilities:p.map(v=>v/sum),margins}}
 if(model.algorithm==='knn'||model.algorithm==='centroid'){
 const p=Array(model.classes.length).fill(0);
 if(model.algorithm==='knn'){const near=model.neighbors.map(r=>({y:r.y,d:distance(x,r.x)})).sort((a,b)=>a.d-b.d).slice(0,model.k);for(const r of near)p[r.y]+=1/near.length}
 else{const ds=model.centroids.map(v=>distance(x,v)),min=Math.min(...ds);const w=ds.map(d=>Math.exp(-(d-min)/Math.max(1e-6,min+.0001)));const sum=w.reduce((a,b)=>a+b,0);w.forEach((v,i)=>p[i]=v/sum)}
 const i=p.indexOf(Math.max(...p));return {label:model.classes[i],confidence:p[i],probabilities:p};
 }
 const p=Array(model.classes.length).fill(0);for(let node of model.trees){while(!node.p)node=x[node.f]<=node.v?node.l:node.r;node.p.forEach((v,i)=>p[i]+=v/model.trees.length)}const i=p.indexOf(Math.max(...p));return {label:model.classes[i],confidence:p[i],probabilities:p}}
export function validateModel(m){if(m?.format!=='modellab-controller-v1'||!['random_forest','knn','centroid'].includes(m.algorithm)||!['pose','hands','both'].includes(m.input?.mode)||!Number.isInteger(m.input.frames)||m.input.frames<1||m.input.frames>60||m.input.sampleHz!==30||!Array.isArray(m.classes)||m.classes.length<2||m.classes.length>20||new Set(m.classes).size!==m.classes.length||m.classes.some(x=>typeof x!=='string'||!x.trim()))throw Error('Unsupported model format');const d=({pose:132,hands:168,both:300})[m.input.mode]*m.input.frames;const vector=x=>Array.isArray(x)&&x.length===d&&x.every(Number.isFinite);
 if(m.algorithm==='centroid'){if(!Array.isArray(m.centroids)||m.centroids.length!==m.classes.length||!m.centroids.every(vector))throw Error('Invalid centroids');return m}
 if(m.algorithm==='knn'){if(!Array.isArray(m.neighbors)||!m.neighbors.length||m.neighbors.length>20000||!Number.isInteger(m.k)||m.k<1||m.k>m.neighbors.length||!m.neighbors.every(r=>Number.isInteger(r.y)&&r.y>=0&&r.y<m.classes.length&&vector(r.x)))throw Error('Invalid neighbors');return m}
 if(!Array.isArray(m.trees)||!m.trees.length||m.trees.length>100)throw Error('Invalid forest');let nodes=0;function check(n,depth){if(!n||depth>12||++nodes>100000)throw Error('Invalid tree');if(n.p){if(!Array.isArray(n.p)||n.p.length!==m.classes.length||n.p.some(v=>!Number.isFinite(v)||v<0||v>1)||Math.abs(n.p.reduce((a,b)=>a+b,0)-1)>.001)throw Error('Invalid leaf');return}if(!Number.isInteger(n.f)||n.f<0||n.f>=d||!Number.isFinite(n.v))throw Error('Invalid split');check(n.l,depth+1);check(n.r,depth+1)}m.trees.forEach(n=>check(n,0));return m}

function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
function rbf(a,b,g){let s=0;for(let i=0;i<a.length;i++)s+=(a[i]-b[i])**2;return Math.exp(-g*s)}
// One-vs-rest soft-margin SVM, deterministic simplified SMO dual optimization.
function trainSVM(rows,classes,algorithm,options){
 const d=rows[0].x.length,n=rows.length,C=Number(options.C??1),gamma=Number(options.gamma??1/d);
 if(!(C>0&&gamma>0&&Number.isFinite(C)&&Number.isFinite(gamma)))throw Error('Invalid SVM settings');
 if(n>1200)throw Error('SVM currently supports up to 1200 training examples; collect fewer or use another algorithm.');
 const mean=Array(d).fill(0),inv=Array(d).fill(0);
 for(const r of rows)for(let i=0;i<d;i++)mean[i]+=r.x[i]/n;
 for(const r of rows)for(let i=0;i<d;i++)inv[i]+=(r.x[i]-mean[i])**2/n;
 for(let i=0;i<d;i++)inv[i]=inv[i]>1e-12?1/Math.sqrt(inv[i]):0;
 const X=rows.map(r=>r.x.map((v,i)=>(v-mean[i])*inv[i])),linear=algorithm==='svm_linear';
 const K=Array.from({length:n},()=>new Float64Array(n));
 for(let i=0;i<n;i++)for(let j=0;j<=i;j++)K[i][j]=K[j][i]=linear?dot(X[i],X[j]):rbf(X[i],X[j],gamma);
 const svm=classes.map((_,cls)=>{
 const y=rows.map(r=>r.y===cls?1:-1),a=new Float64Array(n),f=new Float64Array(n);let bias=0,quiet=0,passes=0;
 while(quiet<6&&passes++<150){let changed=0;
 for(let i=0;i<n;i++){const ei=f[i]+bias-y[i];if(!((y[i]*ei<-.001&&a[i]<C-1e-8)||(y[i]*ei>.001&&a[i]>1e-8)))continue;
 const candidates=Array.from({length:n},(_,j)=>j).filter(j=>j!==i).sort((j,k)=>Math.abs(ei-(f[k]+bias-y[k]))-Math.abs(ei-(f[j]+bias-y[j])));
 for(const j of candidates){const ej=f[j]+bias-y[j],ai=a[i],aj=a[j],same=y[i]===y[j],L=same?Math.max(0,ai+aj-C):Math.max(0,aj-ai),H=same?Math.min(C,ai+aj):Math.min(C,C+aj-ai);if(H-L<1e-10)continue;
 const eta=K[i][i]+K[j][j]-2*K[i][j];let nj;if(eta>1e-12)nj=Math.max(L,Math.min(H,aj+y[j]*(ei-ej)/eta));else{const slope=y[j]*(ei-ej);nj=slope>0?H:L}if(Math.abs(nj-aj)<1e-7)continue;
 const ni=ai+y[i]*y[j]*(aj-nj),di=ni-ai,dj=nj-aj;
 const b1=bias-ei-y[i]*di*K[i][i]-y[j]*dj*K[i][j],b2=bias-ej-y[i]*di*K[i][j]-y[j]*dj*K[j][j];bias=ni>1e-8&&ni<C-1e-8?b1:nj>1e-8&&nj<C-1e-8?b2:(b1+b2)/2;
 a[i]=ni;a[j]=nj;for(let k=0;k<n;k++)f[k]+=y[i]*di*K[i][k]+y[j]*dj*K[j][k];changed++;break;
 }}quiet=changed?0:quiet+1}
 if(linear){const weights=Array(d).fill(0);for(let i=0;i<n;i++)if(a[i]>1e-8)for(let j=0;j<d;j++)weights[j]+=a[i]*y[i]*X[i][j];return {bias,weights}}
 return {bias,support:X.flatMap((x,i)=>a[i]>1e-8?[{x,coef:a[i]*y[i]}]:[])};
 });return {scaler:{mean,inv},svm,gamma,C};
}
// Unified selected-landmark features; fixed global IDs, timestamped nearest resampling.
export function selectedFrame(sample,input){return input.ids.flatMap(i=>{const p=(sample?.pts?.[i]??sample?.points?.[i])?.[input.space]?.raw;return Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)?[...p,1]:[0,0,0,0]})}
export function selectedWindow(samples,t,input){const buffer=samples.map(s=>({t:s.t,x:selectedFrame(s,input)}));const x=windowFeatures(buffer,t,input);return x&&x.some((v,i)=>i%4===3&&v===1)?x:null}
export function validateUnified(m){
 if(m?.format!=='motionlab-classifier-v2'||!['random_forest','knn','centroid','svm_linear','svm_rbf'].includes(m.algorithm)||!Array.isArray(m.classes)||m.classes.length<2||m.classes.length>20||new Set(m.classes).size!==m.classes.length||m.classes.some(x=>typeof x!=='string'||!x.trim()))throw Error('Unsupported classifier');
 const c=m.input;if(!c||!['pose','hands','both'].includes(c.mode)||!['image','relative','world'].includes(c.space)||!Array.isArray(c.ids)||!c.ids.length||new Set(c.ids).size!==c.ids.length||c.ids.some(i=>!Number.isInteger(i)||i<0||i>74||(c.mode==='pose'&&i>=33)||(c.mode==='hands'&&i<33))||(c.space!=='image'&&c.ids.some(i=>i>=33))||!Number.isInteger(c.frames)||c.frames<1||c.frames>90||c.sampleHz!==30)throw Error('Invalid input configuration');
 const d=c.ids.length*4*c.frames,vector=v=>Array.isArray(v)&&v.length===d&&v.every(Number.isFinite),finite=Number.isFinite;
 if(m.algorithm==='svm_linear'||m.algorithm==='svm_rbf'){if(!vector(m.scaler?.mean)||!vector(m.scaler?.inv)||m.scaler.inv.some(v=>v<0)||!Array.isArray(m.svm)||m.svm.length!==m.classes.length||!finite(m.gamma)||m.gamma<=0||m.svm.some(r=>!finite(r.bias)||(m.algorithm==='svm_linear'?!vector(r.weights):!Array.isArray(r.support)||r.support.length>1200||r.support.some(v=>!vector(v.x)||!finite(v.coef)))))throw Error('Invalid SVM');}
 else if(m.algorithm==='centroid'){if(!Array.isArray(m.centroids)||m.centroids.length!==m.classes.length||!m.centroids.every(vector))throw Error('Invalid centroids')}
 else if(m.algorithm==='knn'){if(!Array.isArray(m.neighbors)||!m.neighbors.length||m.neighbors.length>20000||!Number.isInteger(m.k)||m.k<1||m.k>m.neighbors.length||m.neighbors.some(r=>!vector(r.x)||!Number.isInteger(r.y)||r.y<0||r.y>=m.classes.length))throw Error('Invalid KNN')}
 else{let count=0;function check(n,depth){if(!n||depth>12||++count>100000)throw Error('Invalid tree');if(n.p){if(!Array.isArray(n.p)||n.p.length!==m.classes.length||n.p.some(v=>!finite(v)||v<0||v>1)||Math.abs(n.p.reduce((a,b)=>a+b,0)-1)>.001)throw Error('Invalid leaf');return}if(!Number.isInteger(n.f)||n.f<0||n.f>=d||!finite(n.v))throw Error('Invalid split');check(n.l,depth+1);check(n.r,depth+1)}if(!Array.isArray(m.trees)||!m.trees.length||m.trees.length>100)throw Error('Invalid forest');m.trees.forEach(n=>check(n,0))}
 return m;
}
